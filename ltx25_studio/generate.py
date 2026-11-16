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
import inspect
import logging
import math
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import conditioning, constraints, modes
from .constraints import DEFAULT_FRAME_RATE
from .errors import (
    AudioVideoDesyncError,
    DecodeError,
    EmptyOutputError,
    EncodeError,
    GenerationError,
    GpuMemoryError,
    LoRAError,
    LTXStudioError,
    MissingInputError,
    ModelLoadError,
    is_out_of_memory,
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
    "RunRecord",
    "configured_output_root",
    "decoder_policy",
    "generate",
    "gpu_dispatch",
    "prepare_request",
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
    #: ``audio_to_audio`` and ``video_to_video``: how strongly the user's audio or
    #: clip is varied, snapped to the mode's levels (``audio_conditioning``). None
    #: takes the mode's default (0.42 for audio, 0.91 for a clip).
    noise_scale: float | None = None
    conditions: tuple = ()
    #: Kept for the API; no mode reads it now (``video_to_video`` varies its clip).
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
    #: ``min_seconds`` / ``max_seconds`` are passed instead. In a mode that
    #: requires audio it means "match my audio" instead; see `_length_from_audio`.
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
    #: Which decoder finishes the frames: ``"auto"``, ``"diffusion"`` or
    #: ``"conv"``; None takes ``LTX25_DECODER`` (default ``auto``). Stage 2
    #: always finishes with the diffusion decoder.
    decoder: str | None = None


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
    #: What actually happened, as facts for the UI (never parsed from ``warnings``).
    record: RunRecord | None = None


#: Where a run's length came from.
LENGTH_FROM_REQUEST = "request"
LENGTH_FROM_MODEL = "model"
LENGTH_FROM_AUDIO = "audio"
LENGTH_FROM_VIDEO = "video"


@dataclass(frozen=True)
class RunRecord:
    """What one run actually did. Builtins only: it crosses the ZeroGPU fork."""

    mode: str
    requested_width: int
    requested_height: int
    #: The returned picture's size; None for a mode that returns no video.
    width: int | None
    height: int | None
    #: Frames rendered, and their length in seconds at ``frame_rate``.
    frames: int | None
    frame_rate: float
    seconds: float | None
    seed: int
    stage_2: bool
    #: ``"diffusion"`` or ``"conv"``; None when the runner did not say.
    decoder: str | None
    #: The audio stream's level (``audio_conditioning``) where one applies.
    audio_level: float | None
    #: :data:`LENGTH_FROM_REQUEST`, :data:`LENGTH_FROM_MODEL` or :data:`LENGTH_FROM_AUDIO`.
    length_source: str
    #: The prompt sent to the model, after enhancement.
    prompt: str
    #: ``(label, weight, attached module count)`` per adapter.
    adapters: tuple[tuple[str, float, int], ...]
    #: Inside the worker, monotonic, around the whole run.
    elapsed_seconds: float
    #: Peak VRAM reserved over the run, in GiB, where it was measured (CUDA).
    peak_vram_gib: float | None
    #: Plain lines for the collapsed diagnostics block (VRAM laps, the GPU
    #: partition note, the run folder). Not in ``warnings``.
    diagnostics: tuple[str, ...] = ()
    #: The clip-variation level used (``video_to_video``); None where none applies.
    video_level: float | None = None
    #: The device the run ran on: ``"cuda"``, ``"mps"`` or ``"cpu"`` (None if unknown).
    device: str | None = None


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
    if request.decoder is not None:
        from .config import DECODER_CHOICES

        if request.decoder not in DECODER_CHOICES:
            raise constraints.ConstraintError(
                f"decoder={request.decoder!r} is not one of {', '.join(DECODER_CHOICES)}."
            )
    _checked_seed(request.seed)
    # The limit first: a refusal for divisibility must not offer a value past it.
    constraints.validate_dimension_limit(request.width, name="width")
    constraints.validate_dimension_limit(request.height, name="height")
    constraints.validate_dimensions(request.width, request.height)
    if request.auto_duration:
        if request.min_seconds is None or request.max_seconds is None:
            raise constraints.ConstraintError(
                "Automatic duration needs both a shortest and a longest length in seconds."
            )
        if request.max_seconds <= request.min_seconds:
            raise constraints.ConstraintError(
                f"max_seconds ({request.max_seconds}) must be greater than min_seconds "
                f"({request.min_seconds}) for automatic duration."
            )
        constraints.validate_seconds_limit(request.max_seconds)
    else:
        constraints.validate_num_frames_limit(request.num_frames)
        constraints.validate_num_frames(request.num_frames)
    return request


def _checked_seed(seed) -> None:
    """A seed the generator accepts: -1 (random) or 0..MAX_SEED."""
    if seed is None or int(seed) == RANDOM_SEED:
        return
    if not 0 <= int(seed) <= MAX_SEED:
        raise constraints.ConstraintError(
            f"seed={seed} is out of range: use a whole number from 0 to {MAX_SEED}, or -1 for a "
            f"random seed."
        )


# --------------------------------------------------------------------------
# scoping and pre-dispatch checks -- run before a GPU is reserved
# --------------------------------------------------------------------------

#: How a dropped input is described: (subject, what the mode does not use).
_DROPPED_MEDIA = {
    "image": ("An image", "images"),
    "video": ("A video", "videos"),
    "audio": ("An audio file", "audio"),
}


def _condition_entries(value) -> tuple:
    """The shot-list payload as a tuple of entries, empty tables and blank rows removed."""
    if value is None:
        return ()
    if isinstance(value, dict) and "data" in value:
        value = value.get("data") or ()
    elif isinstance(value, dict):
        value = (value,)
    entries = []
    for entry in value:
        if isinstance(entry, (list, tuple)) and all(
            cell is None or not str(cell).strip() for cell in entry
        ):
            continue
        entries.append(entry)
    return tuple(entries)


def scope_request_to_mode(request: GenerationRequest, spec) -> tuple[GenerationRequest, tuple]:
    """Drop every field the mode does not take. Returns ``(request, notes)``.

    The UI keeps the values of fields it hides, and forwards them all; this is
    where they stop. Derived from ``spec.inputs`` and ``spec.optional_inputs``
    only. Idempotent: the worker re-runs it on an already-scoped request and
    gets no notes back.
    """

    def accepts(kind: str) -> bool:
        return modes.accepts(spec, kind)

    label = spec.label
    defaults = GenerationRequest(mode=request.mode)
    changes: dict = {}
    notes: list[str] = []

    for kind, (subject, plural) in _DROPPED_MEDIA.items():
        if getattr(request, kind) is not None and not accepts(kind):
            changes[kind] = None
            notes.append(
                f"{subject} was supplied, but {label} does not use {plural}, so it was ignored."
            )
    if request.audio_latents is not None and not accepts("audio"):
        changes["audio_latents"] = None
        notes.append(
            f"Pre-encoded audio was supplied, but {label} does not use audio, so it was ignored."
        )

    if accepts("text"):
        if request.enhance_prompt and not (request.prompt or "").strip():
            raise constraints.ConstraintError(
                "Enhance is on but the prompt is empty, so there is nothing to enhance. Type a "
                "prompt, or turn prompt enhancement off."
            )
    else:
        for name, subject in (("prompt", "A prompt"), ("negative_prompt", "A negative prompt")):
            value = getattr(request, name)
            if (value or "").strip():
                notes.append(
                    f"{subject} was supplied, but {label} does not take a prompt, so it was "
                    f"ignored."
                )
            if value != "":
                changes[name] = ""
        if request.enhance_prompt:
            changes["enhance_prompt"] = False
            notes.append(
                f"Prompt enhancement was on, but {label} does not take a prompt, so it was not run."
            )

    entries = _condition_entries(request.conditions)
    if entries and not (accepts("image") or accepts("video")):
        entries = ()
        notes.append(f"A shot list was supplied, but {label} does not use one, so it was ignored.")
    if any(isinstance(entry, (list, tuple)) for entry in entries):
        raise constraints.ConstraintError(
            "The shot list carried prompt text, and the model cannot take text as frames: each "
            "shot row needs an image or a video to anchor. To prompt several shots, write them "
            "into the prompt itself, one after another. Clear the shot list to run."
        )
    kept = tuple(entry for entry in entries if accepts(modes.condition_kind(entry)))
    if len(kept) != len(entries):
        notes.append(
            f"A condition was supplied for an input {label} does not use, so it was ignored."
        )
    if kept != request.conditions:
        changes["conditions"] = kept

    # Levers a mode does not read go back to their defaults.
    if (
        not accepts("video")
        or spec.video_strength is not None
        or spec.video_conditioning is not None
    ) and request.strength != defaults.strength:
        changes["strength"] = defaults.strength
    varies = "variation" in (spec.audio_conditioning, spec.video_conditioning)
    if not varies and request.noise_scale != defaults.noise_scale:
        changes["noise_scale"] = defaults.noise_scale
    if spec.skip_stage_2 and request.enable_stage_2:
        changes["enable_stage_2"] = False
    if spec.min_resolution:
        # The mode renders at its pinned size; a hidden width/height is not its input.
        for name in ("width", "height"):
            if getattr(request, name) != modes.MIN_RESOLUTION:
                changes[name] = modes.MIN_RESOLUTION

    return (replace(request, **changes) if changes else request), tuple(notes)


def _supplied_kinds(request: GenerationRequest) -> frozenset[str]:
    """The input kinds a request carries, whatever the mode (shot-list sources included)."""
    kinds = {kind for kind in ("image", "video", "audio") if getattr(request, kind) is not None}
    for entry in _condition_entries(request.conditions):
        if not isinstance(entry, (list, tuple)) and _condition_source(entry) is not None:
            kinds.add(modes.condition_kind(entry))
    if request.audio_latents is not None:
        kinds.add("audio")
    if (request.prompt or "").strip():
        kinds.add("text")
    return frozenset(kinds)


def _alternative_label(spec, missing, supplied) -> str | None:
    """A mode with the same exports that can run on what the user has already supplied.

    Only a mode every one of whose required inputs is supplied: suggesting one
    that is just as stuck (Text to sound with no prompt suggesting Video to
    sound) is no way out. None when there is no such mode.
    """
    for other in modes.MODES.values():
        if (
            other.id != spec.id
            and set(other.exports) == set(spec.exports)
            and not set(other.inputs) & set(missing)
            and set(other.inputs) <= set(supplied)
        ):
            return other.label
    return None


def require_inputs(request: GenerationRequest, spec, *, supplied=frozenset()) -> None:
    """Refuse a run that lacks an input its mode requires."""
    present = _supplied_kinds(request)
    missing = tuple(kind for kind in spec.inputs if kind not in present)
    if missing:
        raise MissingInputError(
            mode=spec.label,
            missing=missing,
            alternative=_alternative_label(spec, missing, frozenset(supplied) | present),
        )


def _audio_seconds(source) -> float:
    """Open an audio file with ``av`` and measure it. No model, no decode if avoidable."""
    return _audio_facts(source)[0]


#: The most channels the audio encode takes (mono or stereo).
MAX_AUDIO_CHANNELS = 2


def _audio_facts(source) -> tuple[float, int]:
    """``(seconds, channels)`` of an audio file, read with ``av``; no decode if avoidable."""
    import av

    name = Path(str(source)).name
    seconds = None
    channels = 0
    try:
        with av.open(str(source)) as container:
            if not container.streams.audio:
                raise constraints.ConstraintError(
                    f"{name} has no audio stream, so there is nothing to condition on. Upload an "
                    f"audio file."
                )
            stream = container.streams.audio[0]
            channels = int(getattr(stream.codec_context, "channels", 0) or 0)
            if stream.duration is not None and stream.time_base is not None:
                seconds = float(stream.duration * stream.time_base)
            else:
                # The sound's own samples (Matroska and WebM give a stream no
                # duration). Never container.duration: the longest stream, which
                # may be a picture that outlasts the sound.
                samples = sum(frame.samples for frame in container.decode(stream))
                seconds = samples / float(stream.rate) if stream.rate else None
    except constraints.ConstraintError:
        raise
    except Exception as exc:  # noqa: BLE001 -- every open failure has one answer here
        raise constraints.ConstraintError(
            f"{name} could not be opened as audio ({type(exc).__name__}). Upload a WAV, MP3, "
            f"FLAC or AAC file."
        ) from exc
    if not seconds or seconds <= 0:
        raise constraints.ConstraintError(
            f"{name} opened, but its length could not be measured. Re-export it as a WAV file."
        )
    if channels > MAX_AUDIO_CHANNELS:
        raise constraints.ConstraintError(
            f"{name} has {channels} audio channels; this app takes mono or stereo audio. Mix it "
            f"down to stereo and upload it again."
        )
    return seconds, channels


#: Allowance for a container's declared length against its decoded samples
#: (measured at most 0.016 s across WAV/FLAC/AAC/Opus); the worker's
#: sample-exact check stays the authority.
_METADATA_MARGIN_SECONDS = 0.02


def _checked_audio_length(request: GenerationRequest, spec, seconds: float | None) -> None:
    """At a set length, audio shorter than the video is refused before dispatch."""
    if seconds is None or request.auto_duration or not modes.requires(spec, "audio"):
        return
    from .audio import DEFAULT_LENGTH_TOLERANCE_SECONDS

    fps = float(request.frame_rate)
    video_seconds = int(request.num_frames) / fps
    if seconds + DEFAULT_LENGTH_TOLERANCE_SECONDS + _METADATA_MARGIN_SECONDS < video_seconds:
        raise _short_audio_refusal(
            video_seconds=video_seconds, audio_seconds=seconds, frame_rate=fps, label=spec.label
        )


def _shortest_usable_num_frames(frame_rate: float) -> int:
    """The shortest length worth setting: more than one frame, and at least 0.1 s."""
    ratio = constraints.DEFAULT_TEMPORAL_COMPRESSION_RATIO
    frames = ratio + 1
    while math.floor(frames / float(frame_rate) * 10 + 1e-9) == 0:
        frames += ratio
    return frames


def _short_audio_refusal(*, video_seconds, audio_seconds, frame_rate, label):
    """An uploaded file shorter than the set length, in the UI's words (gate and worker).

    Names the longest legal length the file covers; where that is a single
    frame (no usable length at all), names the shortest recording that works.
    """
    fps = float(frame_rate)
    fitting = constraints.largest_num_frames_within(audio_seconds, frame_rate=fps)
    shortest = _shortest_usable_num_frames(fps)
    return AudioVideoDesyncError(
        video_seconds=round(float(video_seconds), 3),
        audio_seconds=round(float(audio_seconds), 3),
        largest_fitting_num_frames=fitting,
        route="file",
        frame_rate=fps,
        mode=label,
        # Rounded up, so a recording of exactly this length does cover it.
        shortest_seconds=(
            math.ceil(shortest / fps * 10 - 1e-9) / 10 if fitting < shortest else None
        ),
    )


def _length_from_audio(
    request: GenerationRequest, spec, seconds: float | None
) -> tuple[GenerationRequest, tuple]:
    """In a mode that requires audio, automatic duration means "match my audio".

    The duration head reads only the text, so it cannot hear the upload; it
    never chooses the length here. The frame count is derived from the file
    instead and the request becomes a fixed-length one. A recording too short
    for any usable length is refused as at a set length.
    """
    if not (request.auto_duration and modes.requires(spec, "audio")):
        return request, ()
    if seconds is None:
        raise constraints.ConstraintError(
            f"Automatic duration in {spec.label} measures the audio file you supply, and this run "
            f"carries pre-encoded audio_latents instead. Turn automatic duration off and set the "
            f"frame count (num_frames)."
        )
    fps = float(request.frame_rate)
    num_frames = constraints.largest_num_frames_within(seconds, frame_rate=fps)
    shortest = _shortest_usable_num_frames(fps)
    if num_frames < shortest:
        # Refused in the words a set length gets, never run as a single frame.
        raise _short_audio_refusal(
            video_seconds=shortest / fps, audio_seconds=seconds, frame_rate=fps, label=spec.label
        )
    # The video's own length (the longest legal one the audio covers), in seconds.
    note = f"length set from your audio: {num_frames / fps:.1f} s."
    if seconds * fps > constraints.MAX_NUM_FRAMES:
        note += (
            f" Your audio is longer than the {constraints.MAX_SECONDS:g} s maximum, so it is "
            f"trimmed to fit."
        )
    extras = {**(request.extras or {}), "length_source": LENGTH_FROM_AUDIO}
    return replace(request, auto_duration=False, num_frames=num_frames, extras=extras), (note,)


def _condition_source(entry):
    if isinstance(entry, dict):
        return entry.get("source", entry.get("frames"))
    return getattr(entry, "source", getattr(entry, "frames", None))


def _video_seconds(source) -> float:
    """Open a video input, check it has a picture and its first frame decodes; its length.

    Before a GPU is reserved: a missing, audio-only or undecodable clip is refused
    here, not in the billed worker. The length is the picture's (see
    :func:`_picture_seconds`).
    """
    import av

    name = Path(str(source)).name
    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise constraints.ConstraintError(
                    f"{name} has no picture in it, so it cannot be used as a video. Upload a "
                    f"video file."
                )
            seconds, readable = _picture_seconds(container, container.streams.video[0])
            if not readable:
                raise constraints.ConstraintError(
                    f"{name} opened but none of its frames could be read. Re-export it as an "
                    f"MP4 (H.264) file."
                )
    except constraints.ConstraintError:
        raise
    except Exception as exc:  # noqa: BLE001 -- every open failure has one answer here
        raise constraints.ConstraintError(
            f"{name} could not be opened as a video ({type(exc).__name__}); it may have "
            f"been removed. Upload it again, as an MP4, MOV or WebM file."
        ) from exc
    return float(seconds) if seconds else 0.0


def _picture_seconds(container, stream) -> tuple[float | None, bool]:
    """``(the picture's length, whether a frame decodes)``, in one pass over its packets.

    The video stream's own duration where it has one (MP4, MOV); else its packets
    counted over its own frame rate. Matroska and WebM give a stream no duration,
    and their millisecond timestamps make a span one frame short. Never
    ``container.duration``: the longest stream, so a clip whose sound outlasts its
    picture was planned past its last frame (F70). Counting stops once the clip
    is longer than any run.
    """
    known = None
    if stream.duration is not None and stream.time_base is not None:
        known = float(stream.duration * stream.time_base)
    rate = float(stream.average_rate or stream.guessed_rate or 0)
    enough = math.ceil((constraints.MAX_SECONDS + 1.0) * rate)
    readable, packets = False, 0
    for packet in container.demux(stream):
        if not readable and packet.decode():
            readable = True
        if packet.size:
            packets += 1
        if readable and (known is not None or not rate or packets > enough):
            break
    if known is not None:
        return known, readable
    return (packets / rate if rate else None), readable


def _length_from_clip(
    request: GenerationRequest, spec, seconds: float | None
) -> tuple[GenerationRequest, tuple]:
    """A clip variation takes its length from the clip: the longest legal length it covers."""
    if spec.video_conditioning != "variation" or seconds is None:
        return request, ()
    if (request.extras or {}).get("length_source") == LENGTH_FROM_VIDEO:
        return request, ()  # the worker's second pass
    fps = float(request.frame_rate)
    if seconds <= 0:
        raise constraints.ConstraintError(
            "Your clip opened, but its length could not be measured. Re-export it as an MP4 "
            "(H.264) file."
        )
    num_frames = constraints.largest_num_frames_within(seconds, frame_rate=fps)
    note = f"length set from your clip: {num_frames / fps:.1f} s."
    if seconds * fps > constraints.MAX_NUM_FRAMES:
        note += (
            f" Your clip is longer than the {constraints.MAX_SECONDS:g} s maximum, so it is "
            f"trimmed to fit."
        )
    extras = {**(request.extras or {}), "length_source": LENGTH_FROM_VIDEO}
    return replace(request, auto_duration=False, num_frames=num_frames, extras=extras), (note,)


def _anchor_bound(request: GenerationRequest) -> int:
    """Frames an anchor must fall inside: the clip, or the shortest the head can pick."""
    if not request.auto_duration:
        return int(request.num_frames)
    shortest, _longest = constraints.duration_head_frame_range(
        min_seconds=request.min_seconds,
        max_seconds=request.max_seconds,
        frame_rate=float(request.frame_rate),
    )
    return shortest


def _decode_cap(request: GenerationRequest) -> int:
    """How many frames of a conditioning video are worth decoding."""
    if not request.auto_duration:
        return int(request.num_frames)
    _shortest, longest = constraints.duration_head_frame_range(
        min_seconds=request.min_seconds,
        max_seconds=request.max_seconds,
        frame_rate=float(request.frame_rate),
    )
    return max(round(request.max_seconds * float(request.frame_rate)), longest)


def _checked_conditions(request: GenerationRequest, spec) -> None:
    """Anchor range and source presence, without decoding anything."""
    for condition in spec.build_kwargs(request).get("conditions", ()):
        conditioning.normalise_condition(condition, num_frames=_anchor_bound(request))


def _checked_lora_files(request: GenerationRequest) -> GenerationRequest:
    """Every staged adapter is still on disk, marked in use, and validated.

    Marked so cache eviction cannot remove it during the queue wait (this runs on
    the web side and again at the worker's start). An adapter staged without a
    validation report (a programmatic caller) is validated HERE, before a GPU is
    reserved -- validation hashes the whole file -- and carries its report on.
    """
    from . import lora as lora_module

    staged = []
    for entry in getattr(request, "loras", ()) or ():
        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("source")
            shown = entry.get("source") or path
        else:
            path = shown = entry
        if path is None or not Path(str(path)).is_file():
            raise LoRAError(
                f"The LoRA {Path(str(shown)).name} is no longer on disk, so it cannot be "
                f"applied. Validate it again, or clear it, before generating."
            )
        lora_module.mark_used(path)
        if not isinstance(entry, dict) or entry.get("report") is None:
            entry = dict(entry) if isinstance(entry, dict) else {"source": str(path)}
            entry.setdefault("path", str(path))
            entry["report"] = lora_module.validate_lora(
                str(path), max_bytes=_lora_max_bytes(), **_manifest_override()
            )
        staged.append(entry)
    if not staged:
        return request
    return replace(request, loras=tuple(staged))


def _lora_max_bytes() -> int | None:
    """The configured adapter size limit, or None for the validator's own default."""
    try:
        from .config import load_config

        return load_config().lora_max_bytes
    except Exception:  # noqa: BLE001 -- a bad cap must not change what is validated
        return None


def prepare_request(request: GenerationRequest):
    """Every cheap check, in order, before a GPU is reserved -- the gate ``generate()`` runs.

    Touches no model and no GPU. It does open an audio input with ``av``
    (read-only, to measure it) and checks that staged LoRA files exist.

    Raises the typed ``LTXStudioError`` a run would be refused with; its
    ``str()`` is the message for the user. Otherwise returns
    ``(scoped_request, spec, notes)``, where ``notes`` are the "... was ignored"
    and "length set from your audio" lines the run will carry.

    Runs on the web side of the dispatch and again in the worker; each step is
    idempotent, so the second pass changes nothing and says nothing.
    """
    spec = modes.get_mode(request.mode)
    supplied = _supplied_kinds(request)
    request, scope_notes = scope_request_to_mode(request, spec)
    require_inputs(request, spec, supplied=supplied)
    seconds = _audio_seconds(request.audio) if request.audio is not None else None
    request, length_notes = _length_from_audio(request, spec, seconds)
    _checked_audio_length(request, spec, seconds)
    clip_seconds = _video_seconds(request.video) if request.video is not None else None
    request, clip_notes = _length_from_clip(request, spec, clip_seconds)
    length_notes += clip_notes
    for entry in request.conditions or ():
        source = _condition_source(entry)
        if modes.condition_kind(entry) == "video" and isinstance(source, (str, Path)):
            _video_seconds(source)
    request = _validated(request)
    _checked_conditions(request, spec)
    request = _checked_lora_files(request)
    _checked_stage_2_size(request, spec)
    _checked_decoder(request, spec)
    return request, spec, scope_notes + length_notes


def _checked_decoder(request: GenerationRequest, spec) -> None:
    """An explicit diffusion decode, or stage 2, the decoder could not take: refused now.

    Below the decoder's minimum, or off CUDA beyond what its CPU path can hold.
    ``auto`` is never refused (it falls back to conv with a note).
    """
    stage_2 = stage_2_enabled(spec, request)
    explicit = decoder_policy(spec, request) == "diffusion"
    if not (stage_2 or explicit):
        return
    from . import device as device_module
    from . import pipelines

    frames = None
    if not request.auto_duration:
        frames = request.num_frames
    else:
        frames, _longest = constraints.duration_head_frame_range(
            min_seconds=request.min_seconds,
            max_seconds=request.max_seconds,
            frame_rate=float(request.frame_rate),
        )
    pipelines.check_decoder_minimum(
        pipelines.pinned_decoder_config(),
        width=request.width,
        height=request.height,
        num_frames=frames,
        scale=pipelines.STAGE_2_SPATIAL_SCALE if stage_2 else 1,
        stage_2=stage_2,
    )
    where = device_module.current().device
    if explicit and not stage_2 and where not in (None, "cuda"):
        pipelines.check_cpu_decoder_size(
            width=request.width, height=request.height, num_frames=frames
        )


def _checked_stage_2_size(request: GenerationRequest, spec) -> None:
    """A stage-2 request this device cannot decode is refused before dispatch (no-op on CUDA)."""
    if not stage_2_enabled(spec, request):
        return
    from . import device as device_module
    from .pipelines import _frames_upper_bound

    frames = _frames_upper_bound(
        {
            "num_frames": None if request.auto_duration else request.num_frames,
            "max_seconds": request.max_seconds,
            "frame_rate": request.frame_rate,
        }
    )
    device_module.check_stage_2_size(
        device_module.current().device, request.width, request.height, frames
    )


#: The name ``generate()`` and older callers use.
_prepare_request = prepare_request


def _unwrapped(exc: BaseException) -> BaseException:
    """The real error inside a ``spaces`` 0.52 ``ZeroGPUTaskException``, or ``exc``."""
    inner = getattr(exc, "exception", None)
    return inner if isinstance(inner, BaseException) and inner is not exc else exc


def _with_notes(result, notes: tuple):
    """Put the web side's notes in front of the worker's."""
    if not notes:
        return result
    if not isinstance(result, GenerationResult):
        logger.warning("dropping %d notes: the dispatch returned %r", len(notes), type(result))
        return result
    return replace(result, warnings=tuple(notes) + tuple(result.warnings))


def stage_2_enabled(spec, request: GenerationRequest) -> bool:
    """Whether to run the refinement chain for this request.

    The registry wins over the request: ``skip_stage_2`` marks modes whose video
    is generated and thrown away, and upscaling a frame nobody will see is pure
    quota. Honouring the user's toggle there would be honouring a mistake.
    """
    return bool(request.enable_stage_2) and not spec.skip_stage_2


def decoder_policy(spec, request: GenerationRequest) -> str:
    """The decode policy for this run: the request's, else ``LTX25_DECODER``.

    A mode that discards its video always decodes with conv.
    """
    if "video" not in spec.exports:
        return "conv"
    if request.decoder is not None:
        return request.decoder
    from .config import load_config

    return load_config().decoder


def _pipeline_kwargs(spec, request: GenerationRequest) -> dict:
    kwargs = spec.build_kwargs(request)
    if stage_2_enabled(spec, request):
        kwargs["enable_stage_2"] = True
    policy = decoder_policy(spec, request)
    if policy != "auto":
        # Consumed by the runner, like enable_stage_2; absent means auto.
        kwargs["video_decoder"] = policy
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

    Runs after :func:`_prepared_variation`, so the anchors are checked against the
    frames that will really run: a clip that decoded shorter than planned has
    shortened ``kwargs["num_frames"]``, and a shot past it is refused here, never
    skipped by diffusers.
    """
    raw = kwargs.get("conditions")
    if not raw:
        return ()
    cap = _decode_cap(request)
    bound = _anchor_bound(request)
    if kwargs.get("num_frames"):
        cap = min(cap, int(kwargs["num_frames"]))
        bound = min(bound, int(kwargs["num_frames"]))
    # The largest size a condition is resized to: the stage-2 size when it runs.
    scale = 2 if kwargs.get("enable_stage_2") else 1
    size = (
        int(kwargs.get("height") or request.height) * scale,
        int(kwargs.get("width") or request.width) * scale,
    )
    notes: list[str] = []
    kwargs["conditions"] = conditioning.build_conditions(
        tuple(
            _decoded_video_condition(entry, cap, request.frame_rate, size=size, notes=notes)
            for entry in raw
        ),
        num_frames=bound,
        temporal_compression_ratio=constraints.DEFAULT_TEMPORAL_COMPRESSION_RATIO,
    )
    return tuple(notes)


def _prepared_variation(kwargs: dict, request: GenerationRequest, spec) -> tuple[str, ...]:
    """Decode the clip a variation starts from, at the generation's size and rate.

    The decoded frame count is the authority: a clip whose metadata promised
    more frames than it holds is run at the longest legal length it covers.
    """
    if spec.video_conditioning != "variation":
        return ()
    from . import audio_conditioning

    clip = kwargs.get("video_variation")
    if clip is None:
        return ()
    level = clip.level
    notes: list[str] = [audio_conditioning.video_level_note(level)]
    snapped = audio_conditioning.snapped_level_note(level, request.noise_scale)
    if snapped:
        notes.append(snapped)
    source = clip.source
    if isinstance(source, (str, Path)):
        num_frames = int(kwargs["num_frames"])
        frames = conditioning.decode_video_frames(
            source,
            max_frames=num_frames,
            **_decode_options(
                request.frame_rate,
                size=(int(kwargs["height"]), int(kwargs["width"])),
                notes=notes,
            ),
        )
        if len(frames) < num_frames:
            fitting = constraints.nearest_valid_num_frames(int(len(frames)))[0]
            notes.append(
                f"your clip is shorter than its file said, so the video is "
                f"{fitting / float(request.frame_rate):.1f} s long."
            )
            kwargs["num_frames"] = fitting
            frames = frames[:fitting]
        kwargs["video_variation"] = replace(clip, source=frames)
    return tuple(notes)


def _decode_options(frame_rate, *, size=None, notes=None) -> dict:
    """``frame_rate=`` (and ``size``/``notes`` when given) for the video decode, where it takes them."""
    try:
        parameters = inspect.signature(conditioning.decode_video_frames).parameters
    except (TypeError, ValueError):
        return {}
    wanted = {"frame_rate": float(frame_rate), "size": size, "notes": notes}
    return {
        name: value
        for name, value in wanted.items()
        if name in parameters and (value is not None or name == "frame_rate")
    }


def _decoded_video_condition(entry, cap: int, frame_rate, *, size=None, notes=None):
    """A video condition with its frames decoded, capped at ``cap``; anything else as-is.

    Decoded here rather than inside ``build_conditions`` so the cap can be the
    longest clip the duration head may pick, while anchors are still checked
    against the shortest.
    """
    if modes.condition_kind(entry) != "video":
        return entry
    if isinstance(entry, dict):
        source = entry.get("source", entry.get("frames"))
        index, strength = entry.get("index", 0), entry.get("strength", 1.0)
    else:
        source = getattr(entry, "source", getattr(entry, "frames", None))
        index, strength = getattr(entry, "index", 0), getattr(entry, "strength", 1.0)
    if not isinstance(source, (str, Path)):
        return entry
    frames = conditioning.decode_video_frames(
        source,
        max_frames=cap if cap and cap > 0 else None,
        **_decode_options(frame_rate, size=size, notes=notes),
    )
    return conditioning.ConditionSpec(
        source=frames, kind="video", index=int(index or 0), strength=float(strength)
    )


def _count_from_shape(shape) -> int | None:
    dims = list(shape)
    while len(dims) > 4 and dims[0] == 1:
        dims.pop(0)
    return int(dims[0]) if len(dims) == 4 else None


def _frame_count(output) -> int | None:
    """Frames in a pipeline output, read off its shape; None when it cannot be read."""
    frames = getattr(output, "frames", None)
    if getattr(frames, "shape", None) is not None:
        return _count_from_shape(frames.shape)
    if isinstance(frames, (list, tuple)) and frames:
        first = frames[0]
        if len(frames) == 1 and isinstance(first, (list, tuple)):
            return len(first)
        if len(frames) == 1 and getattr(first, "shape", None) is not None:
            return _count_from_shape(first.shape)
        return len(frames)
    return None


def _chosen_length_note(output, request: GenerationRequest) -> tuple[str, ...]:
    """Under automatic duration, say what length the model picked."""
    if not request.auto_duration:
        return ()
    count = _frame_count(output)
    if count is None:
        return ("the model chose the length itself.",)
    return (f"the model chose the length: {count / float(request.frame_rate):.1f} s.",)


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


def _parameters(function) -> frozenset[str]:
    """The keyword names ``function`` takes ({} where it cannot be inspected)."""
    try:
        return frozenset(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return frozenset()


def _muxes_supplied_audio(exports) -> bool:
    """Whether the user's own waveform is what belongs in this mode's container.

    Derived from the contract, never from the mode id: a run that writes a video
    container declared to carry sound. That is ``audio_to_video``, whose audio is
    frozen -- held exactly while the picture is generated to it -- so the
    upload, length-matched to the video, is the very track the picture was
    generated to. The audio-only exports are not covered: their artifact is the
    model's track, made from the upload.
    """
    exports = tuple(exports or ())
    return "audio" in exports and "video" in exports


def _supplied_audio_note(mode: str, exports, *, muxed: bool, level=None) -> tuple[str, ...]:
    """Say which sound the file carries, whenever audio entered a run that exports audio.

    In the UI's words (UI review 2, B-1). ``mode`` is kept for callers; the note
    no longer names it. ``level`` is the audio variation level, where one applies.
    """
    if "audio" not in tuple(exports or ()):
        return ()
    if muxed:
        return ("the sound in this file is your audio, unchanged.",)
    if level is not None and float(level) >= 1.0:
        return ("this sound is the model's, made fresh; it is not your file.",)
    return ("this sound is the model's, made from yours; it is not your file.",)


def _supplied_latents_note(mode: str, exports, *, level=None) -> tuple[str, ...]:
    """The programmatic route, where there is no waveform to hand back."""
    if not _muxes_supplied_audio(exports):
        return _supplied_audio_note(mode, exports, muxed=False, level=level)
    return (
        "your audio arrived already encoded, so this file carries it decoded by the model, "
        "held fixed while the picture was made. Supply an audio FILE to get your own audio "
        "back unchanged.",
    )


def _audio_level_notes(kwargs: dict, request: GenerationRequest, spec) -> tuple[str, ...]:
    """The level the user's audio was varied at, for a mode that varies it."""
    level = kwargs.get("audio_noise_level")
    if spec.audio_conditioning != "variation" or level is None:
        return ()
    if kwargs.get("audio_latents") is None:
        return ()
    from . import audio_conditioning

    notes = []
    if float(level) < 1.0:
        # At 1.0, "made fresh" is said by the note on which sound the file carries.
        notes.append(audio_conditioning.level_note(level))
    snapped = audio_conditioning.snapped_level_note(level, request.noise_scale)
    if snapped:
        notes.append(snapped)
    return tuple(notes)


def _prepared_audio(kwargs: dict, request: GenerationRequest, pipe, *, spec):
    """Encode the user's audio file into the latents the pipeline actually takes.

    Returns ``(notes, supplied_audio)``. ``supplied_audio`` is an
    ``outputs.SuppliedAudio`` when this run's own waveform is the one the
    container must carry (:func:`_muxes_supplied_audio`), and ``None`` otherwise
    -- which covers the audio-only export modes, the runs where no audio was
    given at all, and the programmatic ``audio_latents`` route, where no waveform
    exists in this process to hand back.

    **Audio in does not mean that audio out**, and ``spec`` is threaded in so
    this layer can say so (and refuse audio its mode does not take). See :func:`_supplied_audio_note`: whenever audio
    actually entered a run whose mode exports audio, the result carries one
    sentence naming which track the file contains. Mode 11 shipped without it
    and returned mp4s with no audio stream at all; the fix is not only to mux a
    track but to say which track it is.

    For a video-exporting mode the track it hands back is the user's own
    (:func:`_muxes_supplied_audio`). The waveform that leaves here is the
    RECONCILED one, taken after
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
    exports = spec.exports
    if not modes.accepts(spec, "audio") and (
        kwargs.get("audio_latents") is not None or getattr(request, "audio", None) is not None
    ):
        # Defence in depth: scope_request_to_mode has already dropped these.
        raise constraints.ConstraintError(
            f"{spec.label} does not use audio, and audio reached the encode step. The request "
            f"must be scoped to the mode first."
        )
    # Both routes are "the user gave us audio", so both get the note. The
    # early return below is the only path where no audio entered at all, and it
    # is the only one that must stay silent.
    supplied = kwargs.get("audio_latents")
    if supplied is not None:
        return (
            _supplied_latents_note(request.mode, exports, level=kwargs.get("audio_noise_level"))
            + _checked_supplied_audio_latents(supplied, kwargs, request, pipe),
            None,
        )
    source = getattr(request, "audio", None)
    if source is None:
        return (), None

    from . import audio as audio_module

    target = getattr(pipe, "pipeline", pipe)
    audio_vae = getattr(target, "audio_vae", None)
    # The forward mel transform reads only the audio VAE (audio.MEL_PARAM_SOURCES,
    # audio.MEL_CONVENTIONS); the vocoder passed below supplies nothing.
    vocoder = getattr(target, "vocoder", None)
    if audio_vae is None:
        raise GenerationError(
            f"mode {request.mode!r} was given an audio input but the pipeline exposes no "
            f"audio_vae to encode it with. Refusing to continue, because generating without "
            f"it returns a plausible track the user did not ask for and nothing reports it."
        )

    num_frames = kwargs.get("num_frames")
    frame_rate = float(kwargs.get("frame_rate") or getattr(request, "frame_rate", 0.0))
    load_options = {}
    if num_frames is not None and "max_seconds" in _parameters(audio_module.load_waveform):
        # Decode no more of the upload than the run can use (an hour-long file
        # to a five-second run is not decoded whole in the GPU window).
        load_options["max_seconds"] = audio_module.upload_seconds_needed(
            int(num_frames), frame_rate
        )
    waveform, rate = audio_module.load_waveform(source, **load_options)
    muxes_upload = _muxes_supplied_audio(exports)
    # Anything this path had to change about the user's file comes back through
    # `notes`, so it reaches `GenerationResult.warnings` rather than the Space
    # log alone. It leads with which track the file will carry, because that is
    # the fact a user who uploaded audio is most likely to get wrong.
    notes: list[str] = list(
        _supplied_audio_note(
            request.mode, exports, muxed=muxes_upload, level=kwargs.get("audio_noise_level")
        )
    )

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
    if num_frames is not None:
        reconcile_options = {}
        if load_options and "source_seconds" in _parameters(
            audio_module.reconcile_waveform_with_video
        ):
            # The trim note names the upload's real length, not the decoded part.
            reconcile_options["source_seconds"] = _audio_seconds(source)
        try:
            waveform, trim_note = audio_module.reconcile_waveform_with_video(
                waveform,
                sampling_rate=rate,
                params=audio_module.audio_params_from(audio_vae),
                num_frames=int(num_frames),
                frame_rate=frame_rate,
                **reconcile_options,
            )
        except AudioVideoDesyncError as exc:
            if exc.route == "file":
                raise
            # The worker's backstop for an uploaded FILE: said as the gate says it,
            # never in terms of audio_latents and num_frames.
            raise _short_audio_refusal(
                video_seconds=exc.video_seconds,
                audio_seconds=exc.audio_seconds,
                frame_rate=frame_rate,
                label=spec.label,
            ) from exc
        if trim_note is not None:
            notes.append(trim_note)
    elif muxes_upload:
        # auto_duration: `num_frames` is absent, so there is no video length to
        # reconcile against and the waveform about to be muxed is the raw upload.
        # "We could not check the lengths" must not travel as silence, which is
        # the same rule `_checked_supplied_audio_latents` follows one branch up.
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

    # A staged local runner holds no transformer between runs; adapters need one.
    ensure_transformer = getattr(pipe, "ensure_transformer", None)
    if callable(ensure_transformer):
        ensure_transformer()
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
                label = _adapter_label(entry.get("source") or source)
                weight = float(entry.get("weight", 1.0))
                validated = entry.get("report") is not None
            else:
                source = entry
                label = _adapter_label(source)
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
                    **_manifest_override(),
                )
            # Checked like any other load, and outside apply_lora, which would
            # report a memory refusal as an unreadable file (review local-runtime-7).
            preflight_adapter = getattr(pipe, "preflight_adapter", None)
            if callable(preflight_adapter):
                preflight_adapter(source)
            reports.append(
                lora_apply.apply_lora(
                    target,
                    source,
                    adapter_name=f"user_lora_{position}",
                    weight=weight,
                    label=label,
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


def _leftover_adapters(pipe) -> tuple[str, ...]:
    """Adapters on the model before this run attached any: on its components, or kept
    by a staged runner for a released transformer (its ``held_adapters``)."""
    from .lora import apply as lora_apply

    names = set(lora_apply.attached_adapters(getattr(pipe, "pipeline", pipe)))
    held = getattr(pipe, "held_adapters", None)
    if callable(held):
        names.update(held())
    return tuple(sorted(names))


def _clear_leftover_adapters(pipe) -> None:
    """Remove every adapter an earlier run left on the model, or refuse this run.

    Runs at the start of every run, before anything is attached. A failed detach
    leaves its adapter on the resident pipeline, and a run that stages none would
    otherwise be sampled through it. Whether the removal worked is judged by
    looking again, not by whether ``delete_adapters`` raised.
    """
    found = _leftover_adapters(pipe)
    if not found:
        return
    logger.warning("removing LoRA adapter(s) left by an earlier run: %s", ", ".join(found))
    target = getattr(pipe, "pipeline", pipe)
    delete = getattr(target, "delete_adapters", None)
    failure: Exception | None = None
    for name in found if callable(delete) else ():
        try:
            delete(name)
        except Exception as exc:  # noqa: BLE001 -- judged by the second look below
            failure = failure or exc
    remaining = _leftover_adapters(pipe)
    if remaining:
        from .lora.apply import LoRALeftoverError

        reason = str(failure) if failure else "the pipeline has no delete_adapters"
        raise LoRALeftoverError(adapters=remaining, reason=reason) from failure


def _adapter_label(source) -> str:
    """How an adapter is named in the run report: what the user gave, a path by its name."""
    text = str(source or "")
    path = Path(text)
    return path.stem if path.is_absolute() else text


def _manifest_override() -> dict:
    """``manifest=None`` when the model is not the pinned one (a snapshot override).

    The checked-in shape manifest describes the pinned transformer; another
    checkpoint is checked against its own live transformer instead.
    """
    from .preflight import snapshot_override

    try:
        return {"manifest": None} if snapshot_override() is not None else {}
    except LTXStudioError:
        return {}


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

#: Every VRAM figure this build prints is a GiB, because that is the unit the
#: earlier measurements in ``docs/HINODE_HOST_SETUP.md`` use. A report that
#: mixes GB and GiB is a report nobody can line up against the numbers already
#: banked.
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
            return
        # Logged here, not only in the run's notes: a run that runs out of memory
        # builds no result, and this is what points at the size (review space-path-2).
        for note in _partition_note_for(self._total):
            logger.warning("GPU partition: %s", note)

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

    @property
    def total_bytes(self) -> int | None:
        """The partition's total memory, when the probe measured it."""
        return self._total or None if self._laps else None

    def peak_gib(self) -> float | None:
        """Peak reserved VRAM over every lap, in GiB; None where nothing was measured."""
        if not self._laps:
            return None
        return round(max(lap[2] for lap in self._laps) / _GiB, 3)

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
    "sampling": (0.12, 0.82),
    "sampling — pass 1 of 2": (0.12, 0.55),
    "upscaling ×2": (0.55, 0.60),
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
        self._position = 0.0

    def __bool__(self) -> bool:
        """True when there is somebody listening."""
        return self._handle is not None

    def _emit(self, position, label: str) -> None:
        if isinstance(position, (int, float)):
            self._position = float(position)
        handle = self._handle
        if handle is None:
            return
        try:
            handle(position, label)
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            logger.debug("progress reporting disabled after %r", exc)
            self._handle = None

    def phase(self, label: str) -> None:
        """Announce a phase at the start of its band; an unknown label keeps the bar where it is."""
        band = _PHASE_BANDS.get(label)
        if band is None:
            logger.debug("progress: unknown phase label %r", label)
            self._emit(self._position, label)
            return
        self._emit(band[0], label)

    def upscaling(self) -> None:
        """The stage-2 latent upsample.

        A named method rather than ``phase("...")`` at the call site because the
        call site is ``pipelines.StageRunner``, and the phase vocabulary belongs
        to exactly one module. A string literal there is a second copy of a label
        that the bands table would silently stop containing.
        """
        self.phase("upscaling ×2")

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


def _partition_notes(probe) -> tuple[str, ...]:
    """Say so when the GPU slice is smaller than the size that was asked for."""
    return _partition_note_for(probe.total_bytes)


def _partition_note_for(total) -> tuple[str, ...]:
    """The partition note for ``total`` bytes against the requested size, if any."""
    if not total:
        return ()
    from . import zerogpu
    from .config import load_config

    try:
        note = zerogpu.partition_note(total, load_config().zerogpu_size)
    except LTXStudioError:
        return ()
    return (note,) if note else ()


def _frame_size(output) -> tuple[int, int] | None:
    """``(width, height)`` of the returned frames, read off the output; None if unreadable."""
    frames = getattr(output, "frames", None)
    first = frames
    while isinstance(first, (list, tuple)) and first:
        first = first[0]
    size = getattr(first, "size", None)
    if isinstance(size, tuple) and len(size) == 2:  # PIL: (width, height)
        return int(size[0]), int(size[1])
    shape = getattr(frames, "shape", None)
    if shape is not None and len(shape) >= 3:
        dims = list(shape)
        if dims[-1] in (1, 3, 4):  # ..., H, W, C
            return int(dims[-2]), int(dims[-3])
        return int(dims[-1]), int(dims[-2])  # ..., C, H, W
    return None


def _length_source(request: GenerationRequest) -> str:
    source = (request.extras or {}).get("length_source")
    if source in (LENGTH_FROM_AUDIO, LENGTH_FROM_VIDEO):
        return source
    return LENGTH_FROM_MODEL if request.auto_duration else LENGTH_FROM_REQUEST


def _run_device(pipe) -> str | None:
    """The device the run used: the runner's own, else this process's resolution."""
    for holder in (pipe, getattr(pipe, "pipeline", None)):
        device = getattr(holder, "device", None)
        kind = str(getattr(device, "type", device) or "").split(":", 1)[0].strip().lower()
        if kind in ("cuda", "mps", "cpu"):
            return kind
    try:
        from . import device as device_module

        return device_module.current().device
    except Exception:  # noqa: BLE001 -- an unresolvable device leaves it unknown
        return None


def _run_record(
    request,
    spec,
    output,
    *,
    kwargs,
    seed,
    run_facts,
    lora_reports,
    elapsed,
    probe,
    diagnostics=(),
    device=None,
) -> RunRecord:
    """The facts of a finished run, for ``GenerationResult.record``."""
    facts = run_facts if isinstance(run_facts, dict) else {}
    returns_video = "video" in spec.exports
    size = _frame_size(output) if returns_video else None
    frames = _frame_count(output)
    rate = float(request.frame_rate)
    level = kwargs.get("audio_noise_level")
    adapters = tuple(
        (
            str(getattr(report, "label", "") or Path(str(report.source)).stem),
            float(report.weight),
            int(report.attached),
        )
        for report in lora_reports or ()
    )
    return RunRecord(
        mode=request.mode,
        requested_width=int(request.width),
        requested_height=int(request.height),
        width=size[0] if size else None,
        height=size[1] if size else None,
        frames=frames,
        frame_rate=rate,
        seconds=round(frames / rate, 3) if frames else None,
        seed=int(seed),
        stage_2=bool(facts.get("stage_2", stage_2_enabled(spec, request))),
        decoder=facts.get("decoder"),
        audio_level=None if level is None else float(level),
        length_source=_length_source(request),
        prompt=str(kwargs.get("prompt") or ""),
        adapters=adapters,
        elapsed_seconds=round(float(elapsed), 3),
        peak_vram_gib=probe.peak_gib(),
        diagnostics=tuple(diagnostics),
        video_level=(
            None
            if kwargs.get("video_variation") is None
            else float(kwargs["video_variation"].level)
        ),
        device=device,
    )


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

    # --- every cheap check, BEFORE the dispatch -----------------------------
    # Scope the request to its mode, require its inputs, validate geometry and
    # length, open the audio, check anchors and staged LoRA files. Both the
    # reservation estimate and the worker see the scoped request; a refusal
    # here costs no ZeroGPU slot. The worker repeats it, idempotently.
    try:
        request, spec, notes = _prepare_request(request)
    except LTXStudioError as exc:
        if not _IN_GPU_WORKER.get():
            exc.gpu_used = False  # refused before dispatch: nothing was billed
        raise

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
        try:
            result = dispatch(
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
        except Exception as exc:
            # spaces 0.52 raises ZeroGPUTaskException(exception=<the real one>)
            # for every worker failure; the UI must see the real one.
            inner = _unwrapped(exc)
            if inner is exc:
                raise
            raise inner from exc
        return _with_notes(result, notes)

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
    started = time.monotonic()
    probe = _VramProbe()
    probe.start()
    finished = False
    try:
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
        #   * the prompt it sends is built on the same side of the fork as the
        #     GenerationResult that carries it, so it rides home in the run
        #     record's `prompt` instead of being grafted on afterwards.
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
        enhanced, enhancement_notes = enhance_module.enhance_if_requested(
            request, enhancer=enhancer
        )
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
        except LTXStudioError:
            # Already typed (a device refusal, a load error): keep it.
            raise
        except Exception as exc:
            if is_out_of_memory(exc):
                raise GpuMemoryError(
                    mode=spec.label, stage="loading the model", reason=str(exc)[:200]
                ) from exc
            raise ModelLoadError(component=spec.pipeline, reason=str(exc)) from exc
        # Before anything is attached: no adapter from an earlier run (F57).
        _clear_leftover_adapters(pipe)

        # Everything between the load and the first denoise step: the pixel-to-latent
        # condition conversion, the peft walk over ~3320 transformer modules for a
        # staged adapter, and the audio encode. None of it is instantaneous and
        # leaving "loading the model" on screen through it would be a label that is
        # simply wrong rather than merely coarse.
        reporter.phase("preparing the inputs")
        kwargs = _pipeline_kwargs(spec, request)
        # The clip first: its decoded frame count is the run's length, and the
        # conditions are checked against that, not against what the file promised.
        clip_notes = _prepared_variation(kwargs, request, spec)
        condition_notes = _prepared_conditions(kwargs, request)
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
            audio_notes, supplied_audio = _prepared_audio(kwargs, request, pipe, spec=spec)
            audio_notes += _audio_level_notes(kwargs, request, spec) + clip_notes + condition_notes
            # Consumed by `StageRunner`, never forwarded to diffusers -- the same
            # treatment `enable_stage_2` gets, and for the same reason: it is ours.
            # Only when somebody is listening, so an injected fake in the offline
            # gate is handed exactly the kwargs it was handed before.
            if reporter:
                kwargs["progress"] = reporter

            try:
                output = pipe(**kwargs)
            except LTXStudioError:
                # Already typed -- a stage-2 ModelLoadError is not a decode failure.
                raise
            except Exception as exc:
                if is_out_of_memory(exc):
                    raise GpuMemoryError(
                        mode=spec.label, stage="generating", reason=str(exc)[:200]
                    ) from exc
                raise DecodeError(mode=request.mode, reason=f"the pipeline raised {exc!r}") from exc

            output = _decoded(output, request.mode)
            length_notes = _chosen_length_note(output, request)
            run_facts = getattr(pipe, "last_run", None)
            decoder_lines = ()
            if isinstance(run_facts, dict) and "video" in spec.exports:
                from . import pipelines

                # Decoder policy is a diagnostic; the record carries which ran.
                decoder_lines = (pipelines.decoder_note(run_facts),)

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
                            f"mode {request.mode!r} exports audio but the pipeline exposes no "
                            f"vocoder to read output_sampling_rate from. Refusing to assume a "
                            f"rate, which would silently resample every export."
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

            # An undeclared track the encoder offered is a builder's fact: diagnostics.
            artifact_lines: list[str] = []
            artifacts = _artifacts_for(
                written, spec=spec, mode=request.mode, run_dir=run_dir, notes=artifact_lines
            )

            # The second lap: model load, denoise, decode and encode. Closed here
            # rather than at the pipeline call so nothing CUDA-touching falls into a
            # window that is never read -- the whole-call figure is only the largest
            # lap, so an unclosed window is silently missing from it.
            probe.split("generation (model load through encode)")

            # Above Diagnostics: facts that change what the user got, in the UI's
            # words (UI review 2, B-1). The upscaled size, the decoder and the prompt
            # sent are the record's, and not repeated here.
            warnings = (
                notes + enhancement_notes + tuple(encoder_warnings) + audio_notes + length_notes
            )
            # Present on every run, but for the collapsed block, not the notes:
            # the flagship run that finally gives a headroom number is not the one
            # somebody remembered to turn a flag on for. Evictions and the decoder
            # policy are diagnostics too.
            diagnostics = (
                decoder_lines
                + tuple(artifact_lines)
                + _sweep(root, retention_bytes)
                + probe.notes()
                + _partition_notes(probe)
                + (f"run folder: {run_dir}",)
            )
            record = _run_record(
                request,
                spec,
                output,
                kwargs=kwargs,
                seed=seed,
                run_facts=run_facts,
                lora_reports=lora_reports,
                elapsed=time.monotonic() - started,
                probe=probe,
                diagnostics=diagnostics,
                device=_run_device(pipe),
            )

            result = GenerationResult(
                output_path=artifacts["primary"],
                run_id=run_id,
                run_dir=run_dir,
                mode=request.mode,
                seed=seed,
                video_path=artifacts.get("video"),
                audio_path=artifacts.get("audio"),
                warnings=warnings,
                record=record,
            )
            finished = True
            return result
        finally:
            # Success or failure, the singleton goes back to base state. See
            # _unload_loras for why this is a raise-on-failure cleanup, not a
            # best-effort one.
            _unload_loras(pipe, lora_reports)
    finally:
        if not finished:
            # A failed run's peak is the one worth having; log it before the error leaves.
            probe.split("failed run")
