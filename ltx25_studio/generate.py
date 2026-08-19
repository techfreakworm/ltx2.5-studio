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

import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import constraints, modes
from .errors import DecodeError, EmptyOutputError, EncodeError, ModelLoadError

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "MAX_SEED",
    "RANDOM_SEED",
    "GenerationRequest",
    "GenerationResult",
    "generate",
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


def _pipeline_kwargs(spec, request: GenerationRequest) -> dict:
    kwargs = spec.build_kwargs(request)
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


def _require_artifact(candidate, *, mode: str, run_dir: Path) -> Path:
    """The last gate. Everything falsy, absent or empty dies here.

    Note what this does NOT do: it does not fall back to anything. The only path
    it will accept is the one the encoder said it wrote.
    """
    if not candidate:
        raise EmptyOutputError(
            mode=mode,
            expected_path=f"{run_dir} (the encoder returned no path at all)",
        )
    path = Path(candidate)
    if not path.exists():
        raise EmptyOutputError(mode=mode, expected_path=str(path))
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

    # --- everything cheap first: unknown mode and illegal geometry must not
    # --- cost a model load.
    spec = modes.get_mode(request.mode)
    request = _validated(request)
    seed = _resolve_seed(request.seed)

    run_id = uuid.uuid4().hex[:12]
    root = Path(output_root) if output_root is not None else Path(DEFAULT_OUTPUT_ROOT)
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
        written = write(
            output,
            run_dir=run_dir,
            mode=request.mode,
            exports=spec.exports,
            seed=seed,
        )
    except Exception as exc:
        raise EncodeError(path=str(run_dir), reason=str(exc)) from exc

    artifact = _require_artifact(written, mode=request.mode, run_dir=run_dir)

    video_path = artifact if "video" in spec.exports else None
    audio_path = artifact if "audio" in spec.exports and video_path is None else None
    return GenerationResult(
        output_path=artifact,
        run_id=run_id,
        run_dir=run_dir,
        mode=request.mode,
        seed=seed,
        video_path=video_path,
        audio_path=audio_path,
    )
