"""Orchestration: request -> pipeline -> encode -> a file that actually exists.

PLAN §5.3, §7.3.

**This module owns the invariant the whole build is judged on.** A generation
that completes without producing a file raises a specific, actionable, typed
error. Never ``None``. Never ``""``. Never a success-shaped result carrying an
empty path. A success path that can return a falsy artifact is a bug whether or
not it is currently reachable.

The predecessor's outage: a mid-graph load failure returned an empty video path,
the UI rendered that as a silent form reset, and the user saw "nothing
happened". No error, no log, no signal, for a week. Four layers each turned a
failure into a falsy value, and every one of them looked reasonable on its own.

Two rules follow, and both are enforced by tests:

* **Every run writes into its own run-scoped directory.** There is no "newest
  file in the output directory" fallback -- that helper has no run correlation
  and will hand a failed run the *previous* run's video, which is worse than an
  empty path because it looks like it worked. This module never goes looking for
  a file it did not write.
* **Dispatch goes through the registry, never through mode identifiers.** No
  mode name appears as a literal here; adding a mode is a registry entry.

``pipeline_factory`` and ``encoder`` are injected seams so failure at every
depth is testable with no GPU and no weights.
"""

from __future__ import annotations

import contextvars
import logging
import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import constraints, modes
from .constraints import DEFAULT_FRAME_RATE
from .errors import DecodeError, EmptyOutputError, EncodeError, ModelLoadError

logger = logging.getLogger(__name__)

#: Set by ``app.py`` to the ``@spaces.GPU``-decorated entrypoint. When present,
#: a generation started outside the GPU worker is routed through it so the run
#: actually reserves a ZeroGPU slot. Unset locally and in tests, where calling
#: straight through is correct.
_GPU_DISPATCH: Callable | None = None

#: Re-entrancy guard: the decorated entrypoint calls back into ``generate()``,
#: and without this it would dispatch to itself forever.
_IN_GPU_WORKER = contextvars.ContextVar("ltx25_in_gpu_worker", default=False)


def set_gpu_dispatch(func: Callable | None) -> None:
    """Register the GPU entrypoint every generation should be routed through."""
    global _GPU_DISPATCH
    _GPU_DISPATCH = func


def gpu_dispatch() -> Callable | None:
    """The registered entrypoint, or None when running without ZeroGPU."""
    return _GPU_DISPATCH


def configured_output_root() -> Path:
    """The output directory, from config, so the writer and the sweep agree.

    They must resolve to the same place: if the startup sweep cleans one
    directory while generations fill another, retention silently protects
    nothing and the disk fills anyway -- which is the failure the sweep exists
    to prevent.
    """
    try:
        from .config import load_config

        return Path(load_config().output_dir)
    except Exception:  # a bad cap must not stop a generation from running
        return Path(DEFAULT_OUTPUT_ROOT)

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "MAX_SEED",
    "RANDOM_SEED",
    "GenerationRequest",
    "GenerationResult",
    "configured_output_root",
    "generate",
    "gpu_dispatch",
    "set_gpu_dispatch",
    "stage_2_enabled",
]

#: ``seed = -1`` means "draw a fresh one". The drawn value comes back on the
#: result, so a good output is always reproducible.
RANDOM_SEED = -1
MAX_SEED = 2**32 - 1

DEFAULT_OUTPUT_ROOT = "outputs"


@dataclass
class GenerationRequest:
    """Everything one generation needs.

    A real object rather than a dict: ``ui/handlers.py`` marshals user input into
    this, and a dict would let a typo pass silently all the way to the pipeline.
    """

    mode: str
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 704
    height: int = 480
    num_frames: int = 121
    #: Frames per second, for BOTH the generation and the container. One value,
    #: threaded to both, so they cannot drift into disagreeing.
    frame_rate: float = DEFAULT_FRAME_RATE
    seed: int = RANDOM_SEED
    image: object | None = None
    video: object | None = None
    audio: object | None = None
    audio_latents: object | None = None
    noise_scale: float = 0.5
    conditions: tuple = ()
    strength: float = 0.8
    guidance_scale: float = 1.0
    num_inference_steps: int = 8
    loras: tuple = ()
    #: Run the stage-2 refinement chain (PLAN §7.4): latent upsample, three more
    #: denoise steps, then the diffusion decode. Off by default because it is a
    #: real quota cost on a metered Space and the user should own that trade.
    #: Forced off for modes whose video is discarded anyway.
    enable_stage_2: bool = False
    #: Let ``duration_head`` choose the length: ``num_frames`` is omitted and
    #: ``min_seconds`` / ``max_seconds`` are passed instead.
    auto_duration: bool = False
    min_seconds: float = 1.0
    max_seconds: float = 5.0
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """A finished run. Every path on it is guaranteed to exist and be non-empty."""

    output_path: Path
    run_id: str
    run_dir: Path
    mode: str
    seed: int
    video_path: Path | None = None
    audio_path: Path | None = None
    warnings: tuple[str, ...] = ()


def _default_pipeline_factory(**kwargs):
    """Resolved late so this module imports with no diffusers present."""
    from . import pipelines

    return pipelines.get_pipeline(**kwargs)


def _default_encoder(output, **kwargs):
    from . import outputs

    return outputs.write_output(output, **kwargs)


def _export_sampling_rate(pipe) -> int | None:
    """The rate to write audio at, read off the vocoder.

    ``outputs._write_audio`` refuses to guess -- correctly, since an assumed rate
    would resample every export silently -- so the value has to be threaded from
    the loaded model to the encoder. Without this, every audio-exporting mode
    fails at the last step.

    Returns ``None`` when there is no vocoder to ask (a stubbed pipeline in a
    test, say); the caller decides whether that is fatal for the mode at hand.
    """
    from . import audio as audio_module
    from .errors import LTXStudioError

    inner = getattr(pipe, "pipeline", pipe)
    vocoder = getattr(inner, "vocoder", None)
    if vocoder is None:
        return None
    try:
        return audio_module.export_sampling_rate_from(vocoder)
    except LTXStudioError:
        return None


def _resolve_seed(seed: int) -> int:
    if seed is None or int(seed) == RANDOM_SEED:
        return secrets.randbelow(MAX_SEED)
    return int(seed)


def _validated(request: GenerationRequest) -> GenerationRequest:
    """Check everything cheap BEFORE a 72 GB load, not after.

    Snapping silently would be worse than refusing: the user asked for a size and
    would receive a different one with no signal.
    """
    constraints.validate_dimensions(request.width, request.height)
    if request.auto_duration:
        if request.max_seconds <= request.min_seconds:
            raise constraints.ConstraintError(
                f"max_seconds ({request.max_seconds}) must be greater than min_seconds "
                f"({request.min_seconds}) for automatic duration."
            )
    else:
        constraints.validate_num_frames(request.num_frames)
    return request


def stage_2_enabled(spec, request: GenerationRequest) -> bool:
    """Whether to run the refinement chain for this request.

    The registry wins over the request: ``skip_stage_2`` marks modes whose video
    is generated and thrown away, and upscaling a frame nobody will see is pure
    quota. Honouring the user's toggle there would be honouring a mistake.
    """
    return bool(request.enable_stage_2) and not spec.skip_stage_2


def _pipeline_kwargs(spec, request: GenerationRequest) -> dict:
    kwargs = spec.build_kwargs(request)
    if stage_2_enabled(spec, request):
        kwargs["enable_stage_2"] = True
    if request.auto_duration:
        # duration_head picks the length, so num_frames must not be sent at all.
        kwargs.pop("num_frames", None)
        kwargs["min_seconds"] = request.min_seconds
        kwargs["max_seconds"] = request.max_seconds
    prompt = kwargs.get("prompt")
    if isinstance(prompt, (list, tuple)):
        # Multishot is one composed prompt plus conditions. A prompt LIST is what
        # makes upstream refuse automatic duration, so we never build one.
        kwargs["prompt"] = "\n".join(str(part) for part in prompt)
    return kwargs


def _decoded(output, mode: str):
    """Reject a pipeline that ran and produced nothing, before we try to encode it."""
    if output is None:
        raise DecodeError(mode=mode, reason="the pipeline returned no output object at all")
    frames = getattr(output, "frames", None)
    if frames is None:
        raise DecodeError(mode=mode, reason="the pipeline returned an output with no frames")
    if len(frames) == 0:
        raise DecodeError(mode=mode, reason="the pipeline returned zero frames")
    return output


def _sweep(root: Path, retention_bytes: int | None) -> tuple[str, ...]:
    """Evict old runs after a successful generation, and log what went.

    PLAN §2.7 requires a sweep after every run: the predecessor filled its disk
    and the Space went unhealthy. Deliberately non-fatal -- the user's video
    exists and is fine, so a retention failure must not turn a good run into an
    error. It is surfaced as a warning instead of being swallowed.
    """
    from . import outputs

    if retention_bytes is None:
        from .config import load_config

        try:
            retention_bytes = int(load_config().output_retention_gb * 1e9)
        except Exception as exc:  # a misconfigured cap must not fail a good run
            return (f"output retention was skipped: {exc}",)
    try:
        evicted = outputs.enforce_retention(root, max_bytes=retention_bytes)
    except Exception as exc:
        message = f"output retention failed: {exc}"
        logger.warning(message)
        return (message,)
    for path in evicted:
        logger.info("evicted old run %s", path)
    return tuple(f"evicted old run {path}" for path in evicted)


def _artifacts_for(written, *, spec, mode: str, run_dir: Path) -> dict:
    """Map what the encoder wrote onto the tracks the mode promised.

    **A mode's ``exports`` tuple is a contract, not a hint.** Declaring
    ``exports=("audio", "video")`` and returning a path for only one of them is
    the never-empty bug one level up: the artifact exists, so nothing downstream
    can tell a real failure from a mapping accident -- and a UI that guards on a
    missing track will correctly report a perfectly good run as broken.

    So every declared export is resolved and checked here, and a missing one
    raises rather than travelling onward as ``None``.

    The encoder may return either a single path -- the common case, and for a
    joint audio+video mode that single muxed file legitimately carries both
    tracks -- or a mapping of track name to path when it writes them separately.
    """
    if isinstance(written, dict):
        resolved = {
            kind: _require_artifact(written.get(kind), mode=mode, run_dir=run_dir, track=kind)
            for kind in spec.exports
        }
        primary = resolved.get("video") or next(iter(resolved.values()))
    else:
        artifact = _require_artifact(written, mode=mode, run_dir=run_dir)
        resolved = dict.fromkeys(spec.exports, artifact)
        primary = artifact

    missing = [kind for kind in spec.exports if not resolved.get(kind)]
    if missing:
        raise EmptyOutputError(
            mode=mode,
            expected_path=(
                f"{run_dir} (mode declares exports={list(spec.exports)} but produced nothing "
                f"for {', '.join(missing)})"
            ),
        )
    resolved["primary"] = primary
    return resolved


def _require_artifact(candidate, *, mode: str, run_dir: Path, track: str = "") -> Path:
    """The last gate. Everything falsy, absent or empty dies here.

    Note what this does NOT do: it does not fall back to anything. The only path
    it will accept is the one the encoder said it wrote.
    """
    label = f" for the {track} track" if track else ""
    if not candidate:
        raise EmptyOutputError(
            mode=mode,
            expected_path=f"{run_dir} (the encoder returned no path at all{label})",
        )
    path = Path(candidate)
    if not path.exists():
        raise EmptyOutputError(mode=mode, expected_path=f"{path}{label}")
    if path.stat().st_size == 0:
        raise EmptyOutputError(mode=mode, expected_path=f"{path} (the file is 0 bytes)")
    return path


def generate(
    request: GenerationRequest | None = None,
    *,
    pipeline_factory: Callable | None = None,
    encoder: Callable | None = None,
    output_root: str | Path | None = None,
    progress: Callable | None = None,
    retention_bytes: int | None = None,
    **params,
) -> GenerationResult:
    """Run one generation and return a result whose artifact provably exists.

    Accepts a :class:`GenerationRequest`, or the request fields as keywords so
    the UI can call ``generate(mode=..., prompt=..., ...)`` directly.
    """
    if request is None:
        request = GenerationRequest(**params)
    elif params:
        request = replace(request, **params)

    # Route through the ZeroGPU entrypoint when one is registered and we are not
    # already inside the worker. Without this the UI calls straight through and
    # no generation ever reserves a GPU slot.
    dispatch = _GPU_DISPATCH
    if dispatch is not None and not _IN_GPU_WORKER.get():
        return dispatch(
            request=request,
            pipeline_factory=pipeline_factory,
            encoder=encoder,
            output_root=output_root,
            retention_bytes=retention_bytes,
        )

    # --- everything cheap first: unknown mode and illegal geometry must not
    # --- cost a model load.
    spec = modes.get_mode(request.mode)
    request = _validated(request)
    seed = _resolve_seed(request.seed)

    run_id = uuid.uuid4().hex[:12]
    root = Path(output_root) if output_root is not None else configured_output_root()
    run_dir = root / f"run-{run_id}"

    factory = pipeline_factory or _default_pipeline_factory
    write = encoder or _default_encoder

    if progress is not None:
        progress(0.0, "loading model")
    try:
        pipe = factory(pipeline=spec.pipeline)
    except Exception as exc:
        raise ModelLoadError(component=spec.pipeline, reason=str(exc)) from exc

    if progress is not None:
        progress(0.2, "generating")
    kwargs = _pipeline_kwargs(spec, request)
    try:
        output = pipe(**kwargs)
    except Exception as exc:
        raise DecodeError(mode=request.mode, reason=f"the pipeline raised {exc!r}") from exc

    output = _decoded(output, request.mode)

    if progress is not None:
        progress(0.9, "encoding")
    try:
        encoder_kwargs = {
            "run_dir": run_dir,
            "mode": request.mode,
            "exports": spec.exports,
            "seed": seed,
            "frame_rate": request.frame_rate,
        }
        sampling_rate = _export_sampling_rate(pipe)
        if sampling_rate is not None:
            encoder_kwargs["sampling_rate"] = sampling_rate
        elif "audio" in spec.exports and encoder is None:
            # Only fatal on the production path: an injected encoder in a test
            # has no vocoder to read and does not need one.
            raise EncodeError(
                path=str(run_dir),
                reason=(
                    f"mode {request.mode!r} exports audio but the pipeline exposes no vocoder "
                    f"to read output_sampling_rate from. Refusing to assume a rate, which would "
                    f"silently resample every export."
                ),
            )
        written = write(output, **encoder_kwargs)
    except Exception as exc:
        raise EncodeError(path=str(run_dir), reason=str(exc)) from exc

    artifacts = _artifacts_for(written, spec=spec, mode=request.mode, run_dir=run_dir)

    warnings = _sweep(root, retention_bytes)

    return GenerationResult(
        output_path=artifacts["primary"],
        run_id=run_id,
        run_dir=run_dir,
        mode=request.mode,
        seed=seed,
        video_path=artifacts.get("video"),
        audio_path=artifacts.get("audio"),
        warnings=warnings,
    )
