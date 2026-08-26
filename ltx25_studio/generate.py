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

from . import conditioning, constraints, modes
from .constraints import DEFAULT_FRAME_RATE
from .errors import (
    DecodeError,
    EmptyOutputError,
    EncodeError,
    GenerationError,
    ModelLoadError,
)

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
    #: NOTE: there is deliberately no `guidance_scale` or `num_inference_steps`.
    #: The distilled recipe fixes both, and a lever that fights the schedule is
    #: worse than no lever -- the user gets worse output with no way to know why.
    #: `pipelines.distilled_kwargs()` is the single source for the sampling set.
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
    #: Rewrite ``prompt`` with the Gemma enhancer before generating (PLAN §3.5).
    #: Off by default. Handled entirely in the WEB process, before dispatch, so
    #: it costs no GPU-seconds -- see ``enhance.py`` and the call site below.
    #: No companion seed field: LTX-2.5 enhancement decodes greedily, so a seed
    #: for it would be an inert lever, and this build has found seven of those.
    enhance_prompt: bool = False
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


def _prepared_conditions(kwargs: dict, request: GenerationRequest) -> None:
    """Normalise every condition before it reaches the pipeline.

    This is the step that was missing: the UI's dataframe rows travelled through
    ``modes._conditions`` untouched and arrived at the pipeline as raw lists, so
    **the pixel-to-latent index conversion never ran** and an out-of-range anchor
    was never caught. Upstream skips a bad condition with only a log line and
    generates anyway -- the shot does nothing, nobody is told, and the metered
    run is spent regardless.

    Conversion to ``LTX2VideoCondition`` happens later, in ``StageRunner``, where
    diffusers is present. The normalisation here is pure arithmetic and always
    runs, which is what makes it a guard rather than a formatting step.
    """
    raw = kwargs.get("conditions")
    if not raw:
        return
    kwargs["conditions"] = conditioning.build_conditions(
        raw,
        num_frames=request.num_frames,
        temporal_compression_ratio=constraints.DEFAULT_TEMPORAL_COMPRESSION_RATIO,
    )


def _seeded_generator(pipe, seed: int):
    """A generator on the pipeline's own device, seeded from the request.

    **The seed used to reach only the sidecar.** ``_resolve_seed`` ran, the
    number went into ``encoder_kwargs`` as metadata, and **nothing was ever
    seeded** -- there was no generator anywhere in the package. So every run was
    unconditionally random and every "same seed" comparison was two independent
    runs: H10b, the before/after pixel diffs, H12b, H4b.

    The device is read off the pipeline for the same reason the dtype is: a
    CUDA generator is required for CUDA sampling, and ``"cuda"`` hardcoded is
    wrong under offload or a second device.

    Returns ``None`` when torch is absent, which is only true where there is no
    model to seed -- the offline gate. It is not a fallback on a real path.
    """
    try:
        import torch
    except ImportError:
        return None

    target = getattr(getattr(pipe, "pipeline", pipe), "device", None)
    generator = torch.Generator(device=target) if target is not None else torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _supplied_audio_note(mode: str, exports) -> tuple[str, ...]:
    """Say which of the two audio tracks the user is holding. Never silent.

    **An audio-conditioned mode has two candidate tracks and they are not the
    same sound.** The user supplied one; the model, being jointly audio-visual,
    rendered another. ``modes.py``'s mode-11 comment carries the ruling and its
    six reasons -- the short form is that the generated track is the one that
    cannot silently mislead, because muxing the upload back would make the file
    look synchronised whether or not the render actually followed it.

    A correct choice made silently is still a surprise. This is the sentence
    that stops a user who uploaded a song from assuming the sound in the file is
    that song, and it names the remedy, because muxing their own file over the
    video is a thing they can do losslessly and we cannot do for them.

    Gated on ``exports`` rather than on the mode id: a future audio-input mode
    that exports no audio would make this note a lie, and deriving it from the
    contract means the note cannot outlive the behaviour it describes.
    """
    if "audio" not in tuple(exports or ()):
        return ()
    # Lazily, like every other use of this module here: `audio` pulls torch and
    # torchaudio, and `generate` is imported by layers that must stay cheap.
    from .audio import DEFAULT_LENGTH_TOLERANCE_SECONDS

    return (
        f"the audio in this file is the model's OWN track, generated jointly with the "
        f"frames -- not the audio you supplied. Your audio conditioned the render for mode "
        f"{mode!r}; it was not muxed back in. The generated track is aligned to the frames "
        f"by construction and your upload is not (it is reconciled to the video only within "
        f"a {DEFAULT_LENGTH_TOLERANCE_SECONDS}s tolerance and on upstream's latent grid), so "
        f"the two are not interchangeable. Mux your own file over the video yourself if that "
        f"is what you wanted.",
    )


def _prepared_audio(
    kwargs: dict, request: GenerationRequest, pipe, *, exports=()
) -> tuple[str, ...]:
    """Encode the user's audio file into the latents the pipeline actually takes.

    **Audio in does not mean that audio out**, and ``exports`` is threaded in so
    this layer can say so. See :func:`_supplied_audio_note`: whenever audio
    actually entered a run whose mode exports audio, the result carries one
    sentence naming which track the file contains. Mode 11 shipped without it
    and returned mp4s with no audio stream at all; the fix is not only to mux a
    track but to say which track it is.

    **This is the step that did not exist.** There is no ``audio=`` argument on
    any LTX2 pipeline (PLAN §3.1c) -- audio enters as pre-encoded
    ``audio_latents``. The UI collected a file into ``request.audio``,
    ``modes._with_audio`` read ``request.audio_latents``, and *nothing joined
    them*: both ends were built, the encoder existed and was exported, and its
    importer count was zero. So the upload was discarded and the model generated
    fresh audio. The run succeeded, the artifact was valid, every test passed,
    and only the user's own ears would ever have told them -- while the field's
    own help text said the encode had happened.

    Encoding is skipped when ``audio_latents`` is already set -- the
    programmatic route, which needs no file -- but that route is **checked**,
    not waved through: see :func:`_checked_supplied_audio_latents`.
    """
    # Both routes are "the user gave us audio", so both get the note. The
    # early return below is the only path where no audio entered at all, and it
    # is the only one that must stay silent.
    supplied = kwargs.get("audio_latents")
    if supplied is not None:
        return _supplied_audio_note(request.mode, exports) + _checked_supplied_audio_latents(
            supplied, kwargs, request, pipe
        )
    source = getattr(request, "audio", None)
    if source is None:
        return ()

    from . import audio as audio_module

    target = getattr(pipe, "pipeline", pipe)
    audio_vae = getattr(target, "audio_vae", None)
    # The vocoder declares the analysis window the VAE's config does not, so the
    # forward transform needs both components. See audio.MEL_PARAM_SOURCES.
    vocoder = getattr(target, "vocoder", None)
    if audio_vae is None:
        raise GenerationError(
            f"mode {request.mode!r} was given an audio input but the pipeline exposes no "
            f"audio_vae to encode it with. Refusing to continue, because generating without "
            f"it returns a plausible track the user did not ask for and nothing reports it."
        )

    waveform, rate = audio_module.load_waveform(source)
    # Anything this path had to change about the user's file comes back through
    # `notes`, so it reaches `GenerationResult.warnings` rather than the Space
    # log alone. It leads with which track the file will carry, because that is
    # the fact a user who uploaded audio is most likely to get wrong.
    notes: list[str] = list(_supplied_audio_note(request.mode, exports))

    # TRIM long, REFUSE short, never pad, never silent. A file longer than the
    # num_frames-derived video is cut to upstream's own latent grid and the cut
    # is named in the result; a file shorter beyond tolerance is refused naming
    # the largest num_frames it covers. The asymmetry is deliberate: trimming
    # discards excess the user can see and undo, padding would fabricate input
    # they never gave. Note the contrast with the supplied-latents branch,
    # which refuses on ANY mismatch: supplied latents are a programmatic
    # caller who meant something precise, while a file upload is an end user
    # who should not need to know the frame arithmetic. Skipped under
    # auto_duration for the same reason as there: no independent video length
    # exists yet for the audio to disagree with.
    num_frames = kwargs.get("num_frames")
    if num_frames is not None:
        waveform, trim_note = audio_module.reconcile_waveform_with_video(
            waveform,
            sampling_rate=rate,
            params=audio_module.audio_params_from(audio_vae),
            num_frames=int(num_frames),
            frame_rate=float(kwargs.get("frame_rate") or getattr(request, "frame_rate", 0.0)),
        )
        if trim_note is not None:
            notes.append(trim_note)

    # The same generator the pipeline gets: `audio_vae.encode` returns a
    # distribution and `.sample()` draws from it, so an unseeded draw makes two
    # runs differ regardless of any lever under test. That is the third reason
    # H10b established less than it appeared to.
    kwargs["audio_latents"] = audio_module.encode_waveform_to_latents(
        waveform,
        audio_vae=audio_vae,
        sampling_rate=rate,
        notes=notes,
        vocoder=vocoder,
        generator=kwargs.get("generator"),
    )
    return tuple(notes)


def _checked_supplied_audio_latents(latents, kwargs, request, pipe) -> tuple[str, ...]:
    """Refuse a desynced run at the door, or say plainly that we could not check.

    Supplied ``audio_latents`` set the audio length while ``num_frames`` still
    governs the video (PLAN §5.7 row 11), and nothing upstream compares them:
    a mismatched pair generates to completion and emits a desynced file with no
    error anywhere. ``audio.validate_audio_latents_against_video`` was written,
    exported and documented as exactly this check -- and called by **nothing**
    (SPACE_MODE_MATRIX §8 follow-up #6, the third advertised-guard-wired-to-
    nothing found in this build). This is the wiring, at the one seam every
    supplied-latents run passes through.

    Two deliberate non-checks, for different reasons:

    * ``auto_duration`` (``num_frames`` absent from kwargs): the duration head
      picks the video length, so there is no independent video length for the
      supplied audio to disagree with. Nothing to check -- a silent skip.
    * No ``audio_vae`` to read timing parameters from, or a shapeless latents
      object: the check *cannot run*, and "could not check" must never look
      like "checked and fine" (the ``check_mmap_backed`` rule). Reported as a
      warning on the result rather than refusing a run that may well be fine.
    """
    num_frames = kwargs.get("num_frames")
    if num_frames is None:
        return ()

    from . import audio as audio_module

    target = getattr(pipe, "pipeline", pipe)
    audio_vae = getattr(target, "audio_vae", None)
    shape = getattr(latents, "shape", None)
    if audio_vae is None or shape is None:
        reason = (
            "the pipeline exposes no audio_vae to read timing parameters from"
            if audio_vae is None
            else "the latents object has no shape"
        )
        return (
            f"audio/video sync was NOT checked for the supplied audio_latents: {reason}. "
            f"A desynced output cannot be ruled out for this run.",
        )

    frame_rate = kwargs.get("frame_rate") or getattr(request, "frame_rate", None)
    # The same seconds convention the UI readout uses (ui/render.py): the two
    # must agree, or the refusal quotes lengths the user was never shown.
    video_seconds = float(num_frames) / float(frame_rate)
    audio_module.validate_audio_latents_against_video(
        shape,
        params=audio_module.audio_params_from(audio_vae),
        video_seconds=video_seconds,
    )
    return ()


def _apply_loras(pipe, request: GenerationRequest):
    """Attach every staged adapter, asserting each actually took.

    Stage 7 lives in ``lora/apply.py`` and was reachable from nothing: a user
    could upload an adapter, watch it validate, generate, and receive
    base-model output -- the precise silent garbage the whole validator exists
    to prevent, arriving because the last step was never called.

    Two lifecycle facts govern the shape of this function, both consequences of
    the pipeline being a cached singleton (``pipelines._CACHE``) on a warm
    ZeroGPU worker:

    * **A partial attach must not outlive its failure.** If adapter 2 of 2
      fails validation, adapter 1 is already on the singleton; without cleanup
      the *retry* dies on diffusers' duplicate-name check instead of the real
      error. The ``except`` arm below deletes whatever this call attached --
      including the failing name, which ``LoRANotAttachedError`` leaves
      registered-but-useless -- before re-raising.
    * **Activation is collective, not per-adapter.** ``apply_lora`` activates
      the one adapter it just attached, which is correct for its single-adapter
      contract -- but diffusers' ``set_adapters`` REPLACES the active set, so
      after a loop of per-adapter calls only the *last* staged adapter is
      active while every attach report claims success. The one collective call
      below is what makes two staged adapters both apply; it lives here and not
      in ``apply_lora`` because this is the only place the full name/weight
      lists exist.

    The matching detach after generation is :func:`_unload_loras`.
    """
    staged = getattr(request, "loras", ()) or ()
    if not staged:
        return ()
    from .lora import apply as lora_apply
    from .lora import validate_lora

    target = getattr(pipe, "pipeline", pipe)
    reports = []
    weights = []
    try:
        for position, entry in enumerate(staged):
            if isinstance(entry, dict):
                # "path" is the fetched LOCAL file for Hub-sourced adapters --
                # the UI resolves repo ids in the web process precisely so this
                # loop, which runs inside the billed GPU window, only ever
                # reads a file. "source" (what the user typed) is the display
                # name and the local-upload case, where the two coincide.
                source = entry.get("path") or entry.get("source")
                weight = float(entry.get("weight", 1.0))
                validated = entry.get("report") is not None
            else:
                source = entry
                weight = 1.0
                validated = False
            if not validated:
                max_bytes = None
                try:
                    from .config import load_config

                    max_bytes = load_config().lora_max_bytes
                except Exception:
                    max_bytes = None  # fall back to the validator's own default
                # Stages 1-6.5 run here for anything the UI did not already screen.
                # The UI validates what it stages, but a programmatic caller -- or a
                # future UI that forgets -- would otherwise reach stage 7 with no
                # header check, no version check and no dimension check at all.
                validate_lora(
                    source,
                    max_bytes=max_bytes,
                    transformer=getattr(target, "transformer", None),
                )
            reports.append(
                lora_apply.apply_lora(
                    target, source, adapter_name=f"user_lora_{position}", weight=weight
                )
            )
            weights.append(weight)
    except BaseException:
        # Damage control on an error path: the real failure is already
        # propagating, so per-name deletion is tolerant -- a name that never
        # registered must not replace the actual error with a cleanup error.
        _delete_adapters(
            target,
            [report.adapter_name for report in reports] + [f"user_lora_{len(reports)}"],
            tolerant=True,
        )
        raise
    if len(reports) > 1:
        set_adapters = getattr(target, "set_adapters", None)
        if callable(set_adapters):
            set_adapters([report.adapter_name for report in reports], weights)
    return tuple(reports)


def _delete_adapters(target, names, *, tolerant: bool = False) -> None:
    """Delete adapters by name, per name so one failure cannot strand the rest.

    ``tolerant=True`` is for error paths only: cleanup there is best-effort
    because the original exception must keep propagating. On the success path
    a failed delete raises -- see :func:`_unload_loras` for why that is not
    optional.
    """
    delete = getattr(target, "delete_adapters", None)
    if not callable(delete):
        if tolerant:
            return
        raise GenerationError(
            f"the pipeline attached LoRA adapters but exposes no delete_adapters to remove "
            f"them with ({', '.join(names)}). The pipeline is a cached singleton, so an "
            f"adapter that cannot be detached poisons every later run on this worker: the "
            f"same staged adapter dies on a duplicate name, and a run with no adapters "
            f"silently generates through this one. Refusing to leave it attached."
        )
    for name in names:
        try:
            delete(name)
        except Exception as exc:
            if tolerant:
                continue
            raise GenerationError(
                f"failed to detach LoRA adapter {name!r} after generation: {exc}. The "
                f"pipeline is a cached singleton, so a stuck adapter corrupts every later "
                f"run on this worker rather than only this one."
            ) from exc


def _unload_loras(pipe, reports) -> None:
    """Detach everything :func:`_apply_loras` attached, success or failure.

    The pipeline is a cached singleton, so anything left attached outlives the
    request. Without this, the second warm-worker run with the same staged
    adapter dies on diffusers' duplicate-name ValueError ("Adapter name
    user_lora_0 already in use") -- and, strictly worse, a following run with
    **no** staged adapters silently generates through the previous user's LoRA:
    the exact silent-garbage shape stage 7 exists to prevent, arriving through
    the lifecycle instead of the load.

    Runs in a ``finally`` so a failed generation cannot poison the next one.
    A delete failure raises rather than warns -- an adapter we cannot remove
    means every later run on this worker is wrong, which is louder than any
    single run's failure. (If that happens while a generation error is already
    propagating, this error supersedes it with the original chained as
    context; the poisoned worker *is* the bigger problem.)

    Deletion is sufficient only because :func:`_apply_loras` attaches without
    fusing. If fusing is ever added there, the weights merge into the base and
    must be unfused before deletion -- ``lora/apply.py::unfuse_lora`` exists
    for exactly that.
    """
    if not reports:
        return
    target = getattr(pipe, "pipeline", pipe)
    _delete_adapters(target, [report.adapter_name for report in reports])


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


def _artifacts_for(written, *, spec, mode: str, run_dir: Path, notes=None) -> dict:
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
        # The encoder is the only layer that knows what it actually wrote, so
        # this is the only place the contract can be checked in BOTH directions.
        # Declared-but-missing raises. Written-but-undeclared does NOT: the
        # registry is what decides a mode's exports, so an over-generous encoder
        # is over-generous rather than wrong, and failing a run that produced a
        # perfectly good video would be the harsher answer to the milder problem.
        # It is dropped and noted, never silently widened into the result.
        extra = sorted(set(written) - set(spec.exports))
        if extra and notes is not None:
            notes.append(
                f"the encoder produced {', '.join(extra)} for mode {mode!r}, which declares "
                f"exports={list(spec.exports)}; the extra track was not returned"
            )
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


def _with_notes(result: GenerationResult, notes: tuple[str, ...]) -> GenerationResult:
    """Carry the web process's notes onto a result the GPU worker built.

    Prompt enhancement happens before dispatch, so its note is known on this
    side of the fork while the result that has to carry it is built on the
    other. Rather than teach the worker about a decision it played no part in,
    the note is grafted on when the worker's result comes back.

    A no-op when there is nothing to add -- which is every run with the toggle
    off, i.e. the default -- so the un-enhanced path is untouched by this.
    """
    if not notes:
        return result
    return replace(result, warnings=tuple(result.warnings) + notes)


def generate(
    request: GenerationRequest | None = None,
    *,
    pipeline_factory: Callable | None = None,
    encoder: Callable | None = None,
    enhancer: Callable | None = None,
    output_root: str | Path | None = None,
    progress: Callable | None = None,
    retention_bytes: int | None = None,
    **params,
) -> GenerationResult:
    """Run one generation and return a result whose artifact provably exists.

    Accepts a :class:`GenerationRequest`, or the request fields as keywords so
    the UI can call ``generate(mode=..., prompt=..., ...)`` directly.

    ``enhancer`` is the third injected seam, alongside ``pipeline_factory`` and
    ``encoder``: a callable taking ``(prompt, image=...)`` and returning the
    rewritten prompt. It is consulted only when the request asks for
    enhancement, and only outside the GPU worker.
    """
    if request is None:
        request = GenerationRequest(**params)
    elif params:
        request = replace(request, **params)

    # --- prompt enhancement, HERE and nowhere else -------------------------
    # Above the dispatch branch on purpose, and guarded on the worker flag
    # rather than on whether a dispatch is registered, which gives one placement
    # three properties at once:
    #
    #   * on a Space it runs in the WEB process, so an enhanced run reserves and
    #     bills exactly the same GPU-seconds as an unenhanced one;
    #   * the request that crosses the fork already carries the rewritten
    #     prompt, so the worker needs to know nothing about any of this;
    #   * a direct caller with no dispatch registered (local dev, the tests, the
    #     reachability probe) gets identical behaviour in-process.
    #
    # Called unconditionally -- `enhance_if_requested` returns immediately when
    # the toggle is off -- so `enhance.py` is executed on every generation and
    # cannot become a module that only a non-default checkbox reaches.
    enhancement_notes: tuple[str, ...] = ()
    if not _IN_GPU_WORKER.get():
        if request.enhance_prompt:
            # "Everything cheap first" -- the rule the block below states, which
            # has to be restated HERE because enhancement now runs above it. A
            # 2.5 GB load plus a CPU decode is a model load by any reading, and
            # an unknown mode or an illegal frame count must not pay for one
            # before failing. `_validated` is pure, so running it again below
            # costs nothing and cannot disagree with itself.
            modes.get_mode(request.mode)
            _validated(request)

        from . import enhance as enhance_module

        enhanced, enhancement_notes = enhance_module.enhance_if_requested(
            request, enhancer=enhancer
        )
        if enhanced is not None:
            request = replace(request, prompt=enhanced)

    # Route through the ZeroGPU entrypoint when one is registered and we are not
    # already inside the worker. Without this the UI calls straight through and
    # no generation ever reserves a GPU slot.
    dispatch = _GPU_DISPATCH
    if dispatch is not None and not _IN_GPU_WORKER.get():
        return _with_notes(
            dispatch(
                request=request,
                pipeline_factory=pipeline_factory,
                encoder=encoder,
                output_root=output_root,
                retention_bytes=retention_bytes,
            ),
            enhancement_notes,
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
    _prepared_conditions(kwargs, request)
    lora_reports = _apply_loras(pipe, request)
    try:
        # Before `_prepared_audio`, because the audio encode samples from a
        # distribution and must draw from the same seeded stream as the pipeline.
        generator = _seeded_generator(pipe, seed)
        if generator is not None:
            kwargs["generator"] = generator

        # After the pipeline exists, because encoding needs its `audio_vae`.
        audio_notes = _prepared_audio(kwargs, request, pipe, exports=spec.exports)
        size_notes = ()
        if stage_2_enabled(spec, request):
            from . import pipelines

            size_notes = (pipelines.stage_2_size_note(request.width, request.height),)
        try:
            output = pipe(**kwargs)
        except Exception as exc:
            raise DecodeError(mode=request.mode, reason=f"the pipeline raised {exc!r}") from exc

        output = _decoded(output, request.mode)

        if progress is not None:
            progress(0.9, "encoding")
        try:
            # The sink is additive: an encoder that ignores it behaves identically.
            # Without it the dropped-audio note reaches the Space logs only, and the
            # user gets a silent video with no on-screen explanation -- which is most
            # of what the warning was for.
            encoder_warnings: list[str] = []
            encoder_kwargs = {
                "run_dir": run_dir,
                "mode": request.mode,
                "exports": spec.exports,
                "seed": seed,
                "frame_rate": request.frame_rate,
                "warnings": encoder_warnings,
                "counterpart_label": modes.counterpart_label_for(request.mode),
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
        except EncodeError:
            # Already one of ours and already names its target; re-wrapping would
            # read "failed to write output <run_dir>: failed to write output
            # <target>: <reason>" and bury the reason a layer deeper.
            raise
        except Exception as exc:
            raise EncodeError(path=str(run_dir), reason=str(exc)) from exc

        artifacts = _artifacts_for(
            written, spec=spec, mode=request.mode, run_dir=run_dir, notes=encoder_warnings
        )

        warnings = (
            enhancement_notes
            + _sweep(root, retention_bytes)
            + tuple(encoder_warnings)
            + audio_notes
            + size_notes
        )

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
    finally:
        # Success or failure, the singleton goes back to base state. See
        # _unload_loras for why this is a raise-on-failure cleanup, not a
        # best-effort one.
        _unload_loras(pipe, lora_reports)
