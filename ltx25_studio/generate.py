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
    "ProgressReporter",
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
    #: Off by default. Handled INSIDE the GPU worker, so it costs a little
    #: billed time -- ~1-2 s, which ``zerogpu.estimate_duration`` reserves for.
    #: It ran on the CPU in the web process for free until that was measured at
    #: +538 s; see ``enhance.py`` and the call site below.
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
    #: Everything the run wants to say out loud. Plain strings, because this
    #: tuple is the ONLY channel from inside the ``@spaces.GPU`` worker back to
    #: the user -- the enhancement note and the peak-VRAM figures are both built
    #: on the far side of the fork and ride home here.
    warnings: tuple[str, ...] = ()


def _default_pipeline_factory(**kwargs):
    """Resolved late so this module imports with no diffusers present."""
    from . import pipelines

    return pipelines.get_pipeline(**kwargs)


def _default_encoder(output, **kwargs):
    return outputs_module().write_output(output, **kwargs)


def outputs_module():
    """``ltx25_studio.outputs``, imported late like every other seam here.

    A function rather than a module-level import for the reason the rest of this
    file resolves things late: ``generate`` is imported by layers that must stay
    cheap, and the writer is only ever needed once a run is actually happening.
    """
    from . import outputs

    return outputs


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


#: The discriminator that overturned ``4683a57``'s choice, kept as data so every
#: sentence quoting it quotes the same numbers. One controlled pair on the live
#: Space: same pipeline, same 512x960x121 geometry, same seed, one variable --
#: whether an audio input was supplied. Both files carried a well-formed 48 kHz
#: stereo AAC track; only one of them carried any sound.
SUPPRESSION_EVIDENCE = (
    "measured on the live Space at identical geometry and seed: with audio supplied, "
    "-88.3 dBFS (digital silence); with none supplied, -24.2 dBFS of real content"
)


def _muxes_supplied_audio(exports) -> bool:
    """Whether the user's own waveform is what belongs in this mode's container.

    **Derived from the contract, never from the mode id** -- the same rule
    :func:`_supplied_audio_note` already follows, and for the same reason: a
    registry flag can be forgotten by whoever adds the next audio-input mode,
    while a predicate over ``exports`` cannot.

    The predicate is "this run writes a video container AND that container is
    declared to carry sound". That is exactly the situation the measurement
    covers: the model's audio generation is suppressed by the audio input, so the
    track it would mux is empty, and an empty stream inside a video is strictly
    worse than no stream at all -- it looks like a track until you play it.

    It deliberately does NOT cover the audio-only export modes. Their artifact
    *is* the model's track; handing back the upload there would answer a
    different question than the one the mode asks, and ``audio_to_audio`` is out
    of scope by operator ruling. Whether the same suppression makes their export
    silent too is a real question and an open one -- it has not been measured,
    and this predicate is not the place to guess.
    """
    exports = tuple(exports or ())
    return "audio" in exports and "video" in exports


def _supplied_audio_note(mode: str, exports, *, muxed: bool) -> tuple[str, ...]:
    """Say which of the two audio tracks the user is holding. Never silent.

    **An audio-conditioned mode has two candidate tracks and they are not the
    same sound.** The user supplied one; the model, being jointly audio-visual,
    renders another. Which one the file carries is
    :func:`_muxes_supplied_audio`'s decision; this is the sentence that stops the
    user having to guess.

    ⛔ **The default flipped for the video-exporting case, and the old reasoning
    was not wrong -- it was answered.** ``4683a57`` muxed the model's track for
    consistency with ``audio_to_audio`` and because the generated track is
    aligned to the frames by construction. Both of those still hold. What did not
    exist when they were weighed is the measurement in
    :data:`SUPPRESSION_EVIDENCE`: supplying audio *suppresses* the model's own
    audio generation, so on exactly the runs that reasoning governed, the
    "correct" track is digital silence. Consistency with a sibling mode is not
    worth handing a user a file that looks like it has sound.

    ⚠️ **What the flip costs, stated rather than dropped.** Reason 5 of the
    original ruling was that muxing the upload back makes the file look and sound
    synchronised whether or not the model actually followed the audio -- the
    user's own track papering over a render that ignored it. That cost is real,
    and it is now accepted, because the alternative is a silent file, which
    papers over nothing and tells the user nothing either.

    Gated on ``exports`` rather than on the mode id: a future audio-input mode
    that exports no audio would make this note a lie, and deriving it from the
    contract means the note cannot outlive the behaviour it describes.
    """
    if "audio" not in tuple(exports or ()):
        return ()
    # Lazily, like every other use of this module here: `audio` pulls torch and
    # torchaudio, and `generate` is imported by layers that must stay cheap.
    from .audio import DEFAULT_LENGTH_TOLERANCE_SECONDS

    if muxed:
        return (
            f"the audio in this file is the audio YOU supplied, not a track the model "
            f"generated. That is deliberate: supplying audio SUPPRESSES the model's own audio "
            f"generation ({SUPPRESSION_EVIDENCE}), so the track it would have contributed to "
            f"mode {mode!r} is silence with a stream header on it. Your upload is reconciled "
            f"to the video length before it is muxed, so the sound in this file is the same "
            f"waveform that conditioned the render, which is not always the whole of the file "
            f"you uploaded.",
        )
    return (
        f"the audio in this file is the model's OWN track, generated jointly with the "
        f"frames -- not the audio you supplied. Your audio conditioned the render for mode "
        f"{mode!r}; it was not muxed back in. The generated track is aligned to the frames "
        f"by construction and your upload is not (it is reconciled to the video only within "
        f"a {DEFAULT_LENGTH_TOLERANCE_SECONDS}s tolerance and on upstream's latent grid), so "
        f"the two are not interchangeable. Mux your own file over the video yourself if that "
        f"is what you wanted.",
    )


def _supplied_latents_note(mode: str, exports) -> tuple[str, ...]:
    """The programmatic route, where there is no waveform to hand back.

    ``audio_latents`` supplied directly are already encoded: the waveform they
    came from never entered this process, so the substitution
    :func:`_muxes_supplied_audio` asks for is unavailable however much the mode
    would like it. The run still gets the model's track -- and by
    :data:`SUPPRESSION_EVIDENCE` that track is very likely silent, since the
    pipeline receives the same ``audio_latents`` either way and cannot tell which
    route produced them.

    Saying so is the whole point. "Could not do it" and "chose not to do it" must
    never read the same, and a caller who wanted their own audio back has exactly
    one thing to change.
    """
    if not _muxes_supplied_audio(exports):
        return _supplied_audio_note(mode, exports, muxed=False)
    return (
        f"audio reached mode {mode!r} as pre-encoded audio_latents, so there is no waveform "
        f"to mux and this file carries the model's OWN track. Supplying audio suppresses the "
        f"model's audio generation ({SUPPRESSION_EVIDENCE}) and the pipeline cannot tell the "
        f"two input routes apart, so that track is very likely silent. Supply an audio FILE "
        f"rather than latents to get your own audio muxed back.",
    )


def _prepared_audio(kwargs: dict, request: GenerationRequest, pipe, *, exports=()):
    """Encode the user's audio file into the latents the pipeline actually takes.

    Returns ``(notes, supplied_audio)``. ``supplied_audio`` is an
    ``outputs.SuppliedAudio`` when this run's own waveform is the one the
    container must carry (:func:`_muxes_supplied_audio`), and ``None`` otherwise
    -- which covers the audio-only export modes, the runs where no audio was
    given at all, and the programmatic ``audio_latents`` route, where no waveform
    exists in this process to hand back.

    **Audio in does not mean that audio out**, and ``exports`` is threaded in so
    this layer can say so. See :func:`_supplied_audio_note`: whenever audio
    actually entered a run whose mode exports audio, the result carries one
    sentence naming which track the file contains. Mode 11 shipped without it
    and returned mp4s with no audio stream at all; the fix is not only to mux a
    track but to say which track it is.

    ✅ **And for a video-exporting mode the track it hands back is now the
    user's own** -- ``4683a57`` chose the model's, on six reasons that were sound
    when it was written, and a controlled discriminator has since measured the
    model's track as digital silence whenever audio is supplied. The waveform
    that leaves here is the RECONCILED one, taken after
    ``reconcile_waveform_with_video`` and before the encode, so the sound in the
    file is byte-for-byte the sound that conditioned the render. Taking it before
    the reconcile would ship a 10 s upload over a 5 s video; taking it after the
    encode is not possible at all, because the encode consumes it into latents.

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
        return (
            _supplied_latents_note(request.mode, exports)
            + _checked_supplied_audio_latents(supplied, kwargs, request, pipe),
            None,
        )
    source = getattr(request, "audio", None)
    if source is None:
        return (), None

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
    muxes_upload = _muxes_supplied_audio(exports)
    # Anything this path had to change about the user's file comes back through
    # `notes`, so it reaches `GenerationResult.warnings` rather than the Space
    # log alone. It leads with which track the file will carry, because that is
    # the fact a user who uploaded audio is most likely to get wrong.
    notes: list[str] = list(_supplied_audio_note(request.mode, exports, muxed=muxes_upload))

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
    elif muxes_upload:
        # auto_duration: `num_frames` is absent, so there is no video length to
        # reconcile against and the waveform about to be muxed is the raw upload.
        # It is still the better of the two tracks -- the model's is suppressed --
        # but "we could not check the lengths" must not travel as silence, which
        # is the same rule `_checked_supplied_audio_latents` follows one branch up.
        notes.append(
            "automatic duration is on, so the video length was chosen by the duration head "
            "and your audio could NOT be reconciled against it. The file carries your upload "
            "at its full length, so its sound may run longer or shorter than its picture. "
            "Turn automatic duration off and set a frame count to have the two reconciled."
        )

    # Captured HERE: after the reconcile, before the encode. Both halves matter.
    # Before the reconcile it would be the raw upload, and a 10 s file over a 5 s
    # video would ship with 10 s of sound -- the desync `2eaeb2c` fixed at the
    # input end, reintroduced at the output end. After the encode it does not
    # exist: `encode_waveform_to_latents` consumes it into latents and nothing
    # downstream can recover it, which is reason 3 of the original ruling and was
    # a true statement about the code as it stood. This line is what makes it
    # false, rather than an argument that it was never true.
    #
    # The rate travels WITH the waveform, in one frozen record, because it is the
    # UPLOAD's rate and not `vocoder.config.output_sampling_rate` -- 16 kHz
    # against 48 kHz on this stack, and muxing one at the other pitch-shifts the
    # track silently. See `outputs.SuppliedAudio`.
    muxed = (
        outputs_module().SuppliedAudio(waveform=waveform, sampling_rate=int(rate))
        if muxes_upload
        else None
    )

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
    return tuple(notes), muxed


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


# --------------------------------------------------------------------------
# peak-VRAM instrumentation -- PLAN §7.3
# --------------------------------------------------------------------------

#: Every VRAM figure this build prints is a GiB, because that is the unit
#: ``scripts/host_matrix.py``, ``scripts/host_smoke.py`` and
#: ``docs/HINODE_HOST_SETUP.md`` already use. A report that mixes GB and GiB is
#: a report nobody can line up against the numbers already banked.
_GiB = 1024**3


class _VramProbe:
    """Peak VRAM for one generation, split by phase. Inert wherever CUDA is not.

    **Why this exists, and what it has already settled.** Two decisions on this
    path were taken on VRAM grounds with no VRAM measurement behind them: the
    enhancer was loaded and freed on every call for "VRAM safety" (+30.78 s of
    reload per enhanced generation), and the VAE decode was pushed to
    tiled+framewise after a MIG allocator assert that nobody had a headroom
    number for. Neither could be revisited without knowing what the partition
    actually has. So every generation now reports it.

    ✅ **The first one is settled and this probe is what settled it.** The
    numbers it printed showed the partition was 47.375 GiB rather than the
    94.971 GiB the Space was entitled to -- the app had spent its whole life on
    ZeroGPU's ``large`` size because ``auto`` cannot promote. On the corrected
    partition the enhancer is resident and the reload is gone. The reasoning
    ladder is in ``docs/MIG_ALLOCATOR_SIGHTINGS.md``; the point worth keeping
    here is that a decision defended for months fell to one printed number.

    **Three quantities, never conflated.** ``max_memory_allocated`` is what the
    run needed; ``max_memory_reserved`` is what the caching allocator is holding
    and therefore what actually bounds the next allocation; ``total_memory`` is
    what the partition has. Headroom is ``total - reserved``, and a report that
    quotes only ``allocated`` against ``total`` overstates it by the whole size
    of the allocator's slack.

    **Laps, not a running total.** :meth:`split` reads the peak and then resets,
    so each phase reports *its own* window rather than "everything so far". The
    whole-call figure is recovered as the largest lap, which is exact: the
    maximum over two touching windows is the maximum of the two maxima. That is
    what buys phase attribution without threading a measurement object through
    ``enhance.py`` and ``pipelines.py``.

    **Guarded once, here.** ``self._cuda`` is None off CUDA and every method
    returns immediately, so the call sites in :func:`generate` stay
    unconditional. The alternative -- ``if torch.cuda.is_available()`` at four
    call sites -- is how a probe ends up half-instrumented, reporting a phase
    split on one box and a whole-call number on another with nothing saying
    which you are reading.

    **Never raises, never gates.** There is no baseline yet, so there is no
    threshold that could be right, and a wrong one would break a working Space
    for a diagnostic. Every method swallows its own failure and disables the
    probe; a generation that cannot be measured still runs.

    **No synchronize.** ``reset_peak_memory_stats``, ``max_memory_allocated``,
    ``max_memory_reserved``, ``memory_allocated`` and ``get_device_properties``
    all read allocator bookkeeping on the host. None of them synchronises the
    stream, so none of them can perturb the number they are measuring or the
    wall-clock they are measured against. ``empty_cache`` would -- it is not
    called here.
    """

    def __init__(self) -> None:
        self._cuda = None
        self._total = 0
        self._resident = 0
        self._laps: list[tuple[str, int, int]] = []
        try:
            import torch

            if torch.cuda.is_available():
                self._cuda = torch.cuda
        except Exception:  # no torch at all is the normal case in tests
            self._cuda = None

    def start(self) -> None:
        """Open the first window, discarding whatever the worker did before us.

        Reset FIRST: a forked ZeroGPU worker is reused across generations and
        arrives carrying the previous run's peak, so a probe that only read
        would report the largest run the container had ever served.
        """
        cuda = self._cuda
        if cuda is None:
            return
        try:
            cuda.reset_peak_memory_stats()
            # `current_device()` rather than a hardcoded 0: identical on the
            # single-device MIG partition this is aimed at, and correct anywhere
            # the worker is not on device 0.
            self._total = int(cuda.get_device_properties(cuda.current_device()).total_memory)
            # What the transformer is already holding when the run enters the
            # worker. Peak minus this is roughly what the run itself cost.
            self._resident = int(cuda.memory_allocated())
        except Exception as exc:
            logger.debug("peak-VRAM probe could not start: %r", exc)
            self._cuda = None

    def split(self, phase: str) -> None:
        """Close the current window as ``phase`` and open the next one."""
        cuda = self._cuda
        if cuda is None:
            return
        try:
            allocated = int(cuda.max_memory_allocated())
            reserved = int(cuda.max_memory_reserved())
            cuda.reset_peak_memory_stats()
        except Exception as exc:
            logger.debug("peak-VRAM probe could not sample %r: %r", phase, exc)
            self._cuda = None
            return
        self._laps.append((phase, allocated, reserved))
        # Logged as well as returned: a run that dies before it can build a
        # GenerationResult still leaves its numbers in the Space log, and an OOM
        # is exactly the run whose numbers we most want.
        logger.info(
            "peak VRAM, %s: %.3f GiB allocated / %.3f GiB reserved of %.3f GiB total",
            phase,
            allocated / _GiB,
            reserved / _GiB,
            self._total / _GiB,
        )

    def notes(self) -> tuple[str, ...]:
        """The laps as user-visible notes, for ``GenerationResult.warnings``.

        Plain strings, because that tuple crosses the fork on the res queue and
        the one thing this build has learned twice is that what crosses a fork
        must be boring (``tests/test_request_fork_boundary.py``).

        Empty off CUDA, so a local MPS or CPU run reads exactly as it did before
        the probe existed.
        """
        if not self._laps:
            return ()
        lines = []
        if len(self._laps) > 1:
            lines += [
                f"peak VRAM, {phase}: {allocated / _GiB:.3f} GiB allocated / "
                f"{reserved / _GiB:.3f} GiB reserved."
                for phase, allocated, reserved in self._laps
            ]
        allocated = max(lap[1] for lap in self._laps)
        reserved = max(lap[2] for lap in self._laps)
        lines.append(
            f"peak VRAM, whole call: {allocated / _GiB:.3f} GiB allocated / "
            f"{reserved / _GiB:.3f} GiB reserved of {self._total / _GiB:.3f} GiB total, "
            f"with {self._resident / _GiB:.3f} GiB already resident on entry. Reserved is "
            f"what bounds the next allocation; allocated is what the run itself needed."
        )
        return tuple(lines)


# --------------------------------------------------------------------------
# progress -- the third of the operator's original defects
# --------------------------------------------------------------------------
#
# There was none. A 66 s render looked exactly like a hung one, and the only
# signal the user had was a status line written before the run started.
#
# **The channel is Gradio's, and it survives the fork -- which is not obvious
# and is the reason this works at all.** `callback_on_step_end` fires INSIDE the
# `@spaces.GPU` worker, a forked child process, and a naive reading says a
# handle captured in the parent is useless there. It is not, and the mechanism is
# in `spaces` 0.51.1 rather than in anything we wrote 📄:
#
#   * `gr.Progress` reaches the queue through `Progress._progress_callback()`,
#     which needs `LocalContext.blocks` and `LocalContext.event_id`. The worker
#     is a FORK, taken inside the request thread where gradio's
#     `get_function_with_locals` has already set both, so the child inherits
#     `blocks`; `spaces` then re-applies `event_id` per call via
#     `GradioPartialContext` (`zero/gradio.py`), which is what makes a REUSED
#     warm worker report against the right event rather than the first one;
#   * `worker_init` calls `patch_gradio_queue(res_queue)`, which REPLACES
#     `gradio.queueing.Queue.set_progress` on the CLASS with an RPC that puts a
#     `GradioQueueEvent` on the result queue;
#   * both the regular and the generator wrapper drain that queue and replay the
#     event on the parent's real (unpatched) `Queue` (`wrappers.py:243` and
#     `:374`), so a `gr.Progress` call in the worker reaches the browser.
#
# ZeroGPU's own "ZeroGPU init" bar is the same mechanism, which is the strongest
# evidence available offline that it works: `worker_init` drives a tqdm from
# inside the fork and users see it.
#
# So this needs no generator handler and no change to `/on_generate`'s frozen
# `(video, audio, markdown)` contract. `spaces` does the crossing; we only have
# to hand the handle down, which is why `generate(progress=...)` -- a parameter
# that has existed and been wired to NOTHING since it was written -- is the
# thing that finally gets connected.

#: Where each phase sits on the 0..1 bar, as ``(start, end)``.
#:
#: ⚠️ **A phase ORDERING, not a time estimate.** Nothing here measures anything:
#: the numbers say which phase comes before which and that sampling is the long
#: one, and they would be a fabrication if they claimed more than that. What IS
#: measured is the position *within* the sampling bands -- step k of n, with n
#: read off the pipeline's own ``num_timesteps`` -- and that is the part a user
#: watching a 66 s render is actually reading.
#:
#: The single-stage ``sampling`` band deliberately OVERLAPS the two-pass pair: a
#: run takes one or the other, never both, so they are alternatives rather than
#: a sequence.
_PHASE_BANDS = {
    "enhancing the prompt": (0.00, 0.06),
    "loading the model": (0.06, 0.10),
    "preparing the inputs": (0.10, 0.12),
    "sampling": (0.12, 0.88),
    "sampling — pass 1 of 2": (0.12, 0.55),
    "upscaling the latents": (0.55, 0.60),
    "sampling — pass 2 of 2": (0.60, 0.82),
    "decoding the frames": (0.82, 0.88),
    "writing the file": (0.88, 1.00),
}

#: Fallback step counts, by pass, when a pipeline does not report its own.
#: Read off the distilled schedules rather than written down here, so a change
#: to the recipe cannot leave the bar counting to the wrong number.
_DECLARED_STEPS = {
    1: len(constraints.DISTILLED_SIGMA_VALUES),
    2: len(constraints.STAGE_2_DISTILLED_SIGMA_VALUES),
}


class ProgressReporter:
    """Every progress signal the app emits goes through here, or nowhere at all.

    **One object, so there is one place the rules live**, and they are the rules
    a progress bar has to obey to be worth having:

    * **It never raises and never gates.** A generation that cannot be reported
      on still runs. Every method swallows its own failure and disables the
      reporter, exactly as ``_VramProbe`` does -- a diagnostic that can kill a
      metered run is a worse defect than the one it was added for.
    * **It never invents a step count.** ``callback_on_step_end`` is handed the
      pipeline, so the total comes from ``pipeline.num_timesteps`` (upstream
      sets ``self._num_timesteps = len(timesteps)`` 📄 before the loop). Only if
      that is absent does it fall back to the declared schedule length, and if
      *that* is unavailable it reports the step index against ``None`` -- which
      Gradio renders as an unknown total. A made-up denominator is how a bar
      starts lying.
    * **It labels every phase, and the label is not decoration.** Stage 2 is a
      second sampling pass whose steps genuinely restart at one; a bar that
      silently restarts at 0% reads as a crash. :meth:`sampling` therefore
      derives "pass N of M" itself rather than accepting a caller's string, so
      there is no way to emit an unlabelled -- or a wrongly labelled -- pass.

    ``handle`` is a ``gr.Progress``-shaped callable -- ``handle(fraction, desc)``
    or ``handle((index, total), desc)``. ``None`` everywhere there is no UI: the
    offline gate, ``scripts/``, and any direct caller. The reporter is still
    constructed in that case, so the call sites stay unconditional and this class
    cannot decay into a branch that only one code path takes.
    """

    def __init__(self, handle: Callable | None = None) -> None:
        self._handle = handle

    def __bool__(self) -> bool:
        """True when there is somebody listening."""
        return self._handle is not None

    def _emit(self, position, label: str) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle(position, label)
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            logger.debug("progress reporting disabled after %r", exc)
            self._handle = None

    def phase(self, label: str) -> None:
        """Announce a phase at the start of its band."""
        start, _end = _PHASE_BANDS[label]
        self._emit(start, label)

    def upscaling(self) -> None:
        """The stage-2 latent upsample.

        A named method rather than ``phase("...")`` at the call site because the
        call site is ``pipelines.StageRunner``, and the phase vocabulary belongs
        to exactly one module. A string literal there is a second copy of a label
        that the bands table would silently stop containing.
        """
        self.phase("upscaling the latents")

    def decoding(self) -> None:
        """The stage-2 diffusion decode. See :meth:`upscaling` for the shape."""
        self.phase("decoding the frames")

    def sampling(self, *, index: int = 1, passes: int = 1) -> Callable | None:
        """A ``callback_on_step_end`` for one denoise pass, or None when idle.

        ``index``/``passes`` name the pass rather than counting it internally,
        because ``StageRunner`` is the only layer that knows whether this run has
        one sampling pass or two -- and the label has to say so.

        The returned callable matches upstream's signature exactly:
        ``callback_on_step_end(pipeline, i, t, callback_kwargs)`` returning a
        dict, because the pipeline does ``callback_outputs.pop("latents",
        latents)`` on the result 📄. Returning ``None`` there would be an
        ``AttributeError`` inside diffusers on every step.
        """
        label = "sampling" if passes == 1 else f"sampling — pass {index} of {passes}"
        self.phase(label)
        if self._handle is None:
            return None
        start, end = _PHASE_BANDS[label]
        declared = _DECLARED_STEPS.get(index)

        def callback_on_step_end(pipeline, step: int, timestep, callback_kwargs):
            total = getattr(pipeline, "num_timesteps", None) or declared
            done = int(step) + 1
            if total:
                self._emit(start + (end - start) * min(done / total, 1.0), label)
            else:
                # No denominator anywhere: report the step we are on and say
                # nothing about how many there are. Gradio renders an unknown
                # total; a guessed one would render a lie.
                self._emit((done, None), label)
            return {}

        return callback_on_step_end


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
    enhancement, and only on the side of the fork that does the work -- which is
    the GPU worker wherever there is one.

    ``progress`` is a ``gr.Progress``-shaped callable and it is now LIVE. It was
    accepted, documented and called at three sites here from the day this module
    was written, and **no caller ever passed one** -- ``ui/handlers.py`` did not
    take it, ``blocks.py`` did not wire it, and the GPU dispatch below dropped it
    on the floor. That is the inert-lever shape this build has found nine times.
    It is threaded through the dispatch with the other seams now, so the calls
    reach the browser from inside the worker; see :class:`ProgressReporter` for
    why that crossing works.
    """
    if request is None:
        request = GenerationRequest(**params)
    elif params:
        request = replace(request, **params)

    # Route through the ZeroGPU entrypoint when one is registered and we are not
    # already inside the worker. Without this the UI calls straight through and
    # no generation ever reserves a GPU slot.
    #
    # `enhancer` joins `pipeline_factory` and `encoder` here, and it did not have
    # to before: enhancement used to be consumed on THIS side of the branch, so
    # dropping the seam was correct. Now that it runs in the worker, a dropped
    # `enhancer` means an injected seam silently stops being used the moment a
    # dispatch is registered -- the caller passes a fake, the worker ignores it
    # and reaches for 2.5 GB of real Gemma instead. All three are None on the
    # live path, which is what keeps the payload picklable.
    dispatch = _GPU_DISPATCH
    if dispatch is not None and not _IN_GPU_WORKER.get():
        return dispatch(
            request=request,
            pipeline_factory=pipeline_factory,
            encoder=encoder,
            enhancer=enhancer,
            output_root=output_root,
            retention_bytes=retention_bytes,
            # `progress` joins the seams here for the same reason `enhancer`
            # did: the work it reports on happens on the FAR side of this
            # branch, so a handle dropped here is a handle that only ever
            # reports the two lines above it. It pickles -- a fresh
            # `gr.Progress` is `{"track_tqdm": bool, "iterables": []}` and
            # `spaces` already carries one in `GradioPartialContext` -- and
            # `tests/test_request_fork_boundary.py` holds that.
            progress=progress,
        )

    # --- everything cheap first: unknown mode and illegal geometry must not
    # --- cost a model load.
    spec = modes.get_mode(request.mode)
    request = _validated(request)

    # --- peak-VRAM instrumentation, opened HERE ----------------------------
    # Below the dispatch branch, so on a Space this is the first line of the
    # `@spaces.GPU` worker's own work and the window covers everything the
    # billed call does. Below `_validated` too, so "everything cheap first"
    # still holds: a request with an illegal frame count is refused without
    # even constructing a probe.
    #
    # Deliberately NOT above the dispatch: on the web-process side there is no
    # CUDA to measure, and a probe opened there would report the fork's numbers
    # as the run's.
    probe = _VramProbe()
    probe.start()

    # Constructed unconditionally, exactly like the probe above and for the same
    # reason: a reporter built only when a UI is attached is a code path the
    # offline gate never walks. With no handle every method is a no-op, so the
    # call sites below stay branch-free.
    reporter = ProgressReporter(progress)

    # --- prompt enhancement, HERE and nowhere else -------------------------
    # BELOW the dispatch branch, so on a Space this executes inside the
    # `@spaces.GPU` worker. That placement is a correction, and the reason is
    # measured rather than argued: on the CPU in the web process the 600-token
    # greedy decode took ~538 s, against ~1-2 s on the GPU. See `enhance.py`'s
    # docstring for the numbers and for why moving it to CUDA does NOT put the
    # enhancer back into the ZeroGPU packing window.
    #
    # Three properties come from this one line's position:
    #
    #   * it is past `get_mode` and `_validated`, so "everything cheap first"
    #     covers it for free -- a request with an unknown mode or an illegal
    #     frame count is refused before a 2.5 GB load, with no duplicated
    #     validation to keep in step;
    #   * it is above the pipeline factory, so a failed enhancement never first
    #     spends a model load it is going to throw away;
    #   * the note it produces is built on the same side of the fork as the
    #     GenerationResult that carries it, so it rides home in
    #     `warnings` like every other note instead of being grafted on
    #     afterwards.
    #
    # Called unconditionally -- `enhance_if_requested` returns immediately when
    # the toggle is off -- so `enhance.py` is executed on every generation and
    # cannot become a module that only a non-default checkbox reaches.
    from . import enhance as enhance_module

    # ~21 s of dead air before sampling even starts, and until now the only
    # thing on screen said "Running". Announced BEFORE the call rather than
    # after it, which is the whole point -- a label that appears once the phase
    # is over reports history, not progress. Gated on the toggle so a run that
    # does no enhancement does not claim to be doing some; `enhance_if_requested`
    # itself stays unconditional.
    if request.enhance_prompt:
        reporter.phase("enhancing the prompt")
    enhanced, enhancement_notes = enhance_module.enhance_if_requested(request, enhancer=enhancer)
    if enhanced is not None:
        request = replace(request, prompt=enhanced)
        # The split that answers the open question about this feature: is the
        # enhancer's ~2.49 GB *additive* to the transformer's peak, or absorbed
        # in allocator slack the denoise was going to hold anyway? One lap each
        # side and the two figures say which. Taken only when the enhancer
        # actually ran -- `enhanced is None` covers both "toggle off" and
        # "prompt was empty", and in neither case did a rewrite happen, so a lap
        # there would report the baseline under a label claiming otherwise.
        #
        # It stays interesting now that the enhancer is RESIDENT rather than
        # freed per call: this lap is what would show the 2.49 GB becoming a
        # permanent floor under every subsequent generation in the worker, if it
        # ever did. A load happens only on the first enhanced call in a worker,
        # so on later calls this lap reports the decode alone -- which is the
        # measurement that confirms the cache is hitting.
        probe.split("prompt enhancement")

    seed = _resolve_seed(request.seed)

    run_id = uuid.uuid4().hex[:12]
    root = Path(output_root) if output_root is not None else configured_output_root()
    run_dir = root / f"run-{run_id}"

    factory = pipeline_factory or _default_pipeline_factory
    write = encoder or _default_encoder

    reporter.phase("loading the model")
    try:
        pipe = factory(pipeline=spec.pipeline)
    except Exception as exc:
        raise ModelLoadError(component=spec.pipeline, reason=str(exc)) from exc

    # Everything between the load and the first denoise step: the pixel-to-latent
    # condition conversion, the peft walk over ~3320 transformer modules for a
    # staged adapter, and the audio encode. None of it is instantaneous and
    # leaving "loading the model" on screen through it would be a label that is
    # simply wrong rather than merely coarse.
    reporter.phase("preparing the inputs")
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
        # `supplied_audio` is the user's own reconciled waveform when this mode's
        # container is the one it belongs in, and None everywhere else; see
        # `_muxes_supplied_audio`.
        audio_notes, supplied_audio = _prepared_audio(
            kwargs, request, pipe, exports=spec.exports
        )
        # Consumed by `StageRunner`, never forwarded to diffusers -- the same
        # treatment `enable_stage_2` gets, and for the same reason: it is ours.
        # Only when somebody is listening, so an injected fake in the offline
        # gate is handed exactly the kwargs it was handed before.
        if reporter:
            kwargs["progress"] = reporter

        size_notes = ()
        if stage_2_enabled(spec, request):
            from . import pipelines

            size_notes = (pipelines.stage_2_size_note(request.width, request.height),)
        try:
            output = pipe(**kwargs)
        except Exception as exc:
            raise DecodeError(mode=request.mode, reason=f"the pipeline raised {exc!r}") from exc

        output = _decoded(output, request.mode)

        reporter.phase("writing the file")
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
            if supplied_audio is not None:
                # Conditional rather than always-present, like `sampling_rate`
                # below: an injected encoder in a test sees the kwarg only on the
                # runs where a substitution really happened, so "was the user's
                # audio muxed" is answerable by looking rather than by inferring.
                encoder_kwargs["supplied_audio"] = supplied_audio
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

        # The second lap: model load, denoise, decode and encode. Closed here
        # rather than at the pipeline call so nothing CUDA-touching falls into a
        # window that is never read -- the whole-call figure is only the largest
        # lap, so an unclosed window is silently missing from it.
        probe.split("generation (model load through encode)")

        warnings = (
            enhancement_notes
            + _sweep(root, retention_bytes)
            + tuple(encoder_warnings)
            + audio_notes
            + size_notes
            # Last, because it is diagnostic rather than something the user has
            # to act on -- but present on every run, because the one flagship
            # generation that finally gives this build a headroom number is not
            # going to be the run somebody remembered to turn a flag on for.
            + probe.notes()
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
