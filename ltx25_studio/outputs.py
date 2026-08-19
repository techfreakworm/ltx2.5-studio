"""Write the artifacts, and keep the disk from filling.

PLAN §2.7, §5.3, §9.1.

**Muxing is not optional, and a silent video is the failure mode this module
exists to prevent.** LTX-2.5's headline capability is *joint* audio-visual
generation: ``text_to_audio_video``, ``image_to_audio_video`` and
``image_text_to_audio_video`` all return ``(frames, audio)``. Writing the frames
and dropping the audio produces a file that opens, plays, and is wrong -- no
error, no warning, a plausible-looking artifact. That is the same silent-failure
class as §5.3's empty output, one layer further out, and it is why every write
that has audio in hand muxes it or raises.

**``exports`` describes reality, in both directions.** A mode that declares
``exports=("video",)`` gets a container with *no* audio stream, even though the
model produced one. The tuple is a checked contract elsewhere -- ``generate.py``
raises when a declared track is missing -- and a contract enforced in only one
direction drifts silently in the other: video-only modes would ship files
indistinguishable from the joint modes, and four of the eleven modes PLAN §3.2
argues for would be cosmetic duplicates of three others.

Dropping it is not free and is never silent. PLAN §3.2 already records that the
audio-only modes "still pay full video compute"; the reverse is equally true and
deserves the same candour -- **the video-only modes pay full audio compute**,
because the model is jointly audio-visual and has no video-off switch. So the
audio is rendered either way, and when we discard it we say so: see
:func:`dropped_audio_warning`.

**Audio is stereo and stays stereo.** The vocoder reports two output channels.
Nothing here may mono-downmix: the channel count is read off the array that
arrives and the container layout is *derived* from it (FFmpeg's ``"{n}c"``
spelling), so a stereo track cannot be silently flattened by a layout constant
that says otherwise.

**Nothing assumes a frame rate or a sample rate.** Both must be supplied by the
caller, which knows them: the frame rate is the one the pipeline generated at,
and the sample rate is ``vocoder.config.output_sampling_rate`` (PLAN §5.4). A
default here would be a second, independent assumption about a number that
already lives somewhere else -- and a container written at the wrong rate is a
video that plays at the wrong speed, which again reports as nothing. Absent,
they raise.

``av`` is the one media stack: it is already a pinned dependency for the §5.7 #7
image-conditioning CRF path, and using it here means the audio-only export needs
no second library. It is imported lazily so this module -- and the retention
policy -- stay importable and testable with no media stack present (§9.1).

Retention is not housekeeping, it is an availability requirement: the 2.3 Space
filled its disk and went unhealthy. ``outputs/`` is capped, bounded-bytes,
oldest-first, swept on startup **and** after each run, and it logs what it
evicted -- because discovering your outputs are gone by not finding them is a
worse experience than being told.

Three retention rules that are easy to get wrong:

* **Evict whole run directories, never individual files.** Half an evicted run
  is worse than none of it.
* **Never evict the newest run**, even when it alone exceeds the cap. Generating
  a video and deleting it in the same breath is not a retention policy.
* **A run's age is its oldest file, not its directory mtime.** Writing a second
  artifact into a run touches the directory, which would make the oldest run
  look like the newest and evict exactly backwards.

This module is the **one** place in the package allowed to use ``mtime`` and
directory scanning. Everywhere else they are banned, because that ban is what
stops the predecessor's "newest video in the output dir" fallback growing back
(PLAN §5.3).
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

from .errors import EncodeError, LTXStudioError

_log = logging.getLogger(__name__)

__all__ = [
    "AUDIO_FILENAME",
    "AUDIO_ONLY_CODEC",
    "MAX_AUDIO_CHANNELS",
    "MUXED_AUDIO_CODEC",
    "VIDEO_CODEC",
    "VIDEO_FILENAME",
    "VIDEO_PIXEL_FORMAT",
    "directory_size",
    "dropped_audio_warning",
    "enforce_retention",
    "run_age",
    "write_output",
]

VIDEO_FILENAME = "output.mp4"
AUDIO_FILENAME = "output.wav"

#: H.264 in MP4: the combination every browser Gradio might render this in can
#: actually play. ``yuv420p`` for the same reason -- 4:4:4 output is legal H.264
#: that Safari and most hardware decoders refuse.
VIDEO_CODEC = "libx264"
VIDEO_PIXEL_FORMAT = "yuv420p"

#: AAC for the muxed track (MP4's normal audio codec) and uncompressed PCM for
#: the audio-only export, where the point is to hand back exactly what the
#: vocoder produced rather than a re-compressed approximation of it.
MUXED_AUDIO_CODEC = "aac"
AUDIO_ONLY_CODEC = "pcm_s16le"

#: An audio array's channel axis is the short one. Above this, neither axis is
#: plausibly channels and we refuse to guess rather than transpose a waveform
#: into noise.
MAX_AUDIO_CHANNELS = 8

#: Frame layouts we hand to ``av``. ``rgb24`` is what the pipeline's uint8 output
#: already is; the planar float format is what audio encoders want.
VIDEO_FRAME_FORMAT = "rgb24"
AUDIO_FRAME_FORMAT = "fltp"

#: Channel counts that mean "greyscale" and "RGB" on a frame's last axis.
_GREY_CHANNELS = 1
_RGB_CHANNELS = 3
_UINT8_MAX = 255


# ---------------------------------------------------------------------------
# retention -- PLAN §2.7
# ---------------------------------------------------------------------------


def directory_size(path: Path) -> int:
    """Total bytes of every file under ``path``."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def run_age(path: Path) -> float:
    """When the run started: the mtime of its oldest file.

    Deliberately not the directory's own mtime -- writing an extra artifact into
    a run updates that, which would sort the oldest run to the newest and evict
    in exactly the wrong order.
    """
    stamps = [entry.stat().st_mtime for entry in path.rglob("*") if entry.is_file()]
    return min(stamps) if stamps else path.stat().st_mtime


def enforce_retention(root, *, max_bytes: int) -> tuple[Path, ...]:
    """Evict oldest-first until under the cap. Returns what it evicted.

    Returning the evicted paths rather than logging them here keeps the policy
    testable and lets the caller decide where the record goes -- but the caller
    must log them; a silent sweep is the failure this function exists to avoid.
    """
    if max_bytes <= 0:
        raise LTXStudioError(
            f"output retention cap must be a positive number of bytes, got {max_bytes}. "
            f"Refusing to sweep: a zero or negative cap would delete every output on disk."
        )

    root = Path(root)
    if not root.is_dir():
        # Startup sweeps run before the first generation has created anything.
        return ()

    runs = sorted((entry for entry in root.iterdir() if entry.is_dir()), key=run_age)
    if not runs:
        return ()

    total = sum(directory_size(run) for run in runs)
    evicted: list[Path] = []
    # ``len(runs) > 1`` is the never-evict-the-newest rule: the last survivor is
    # always kept, however large it is.
    while total > max_bytes and len(runs) > 1:
        oldest = runs.pop(0)
        total -= directory_size(oldest)
        _remove_tree(oldest)
        evicted.append(oldest)
    return tuple(evicted)


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# normalising what the pipeline handed us
# ---------------------------------------------------------------------------


def _as_array(value):
    """Whatever we were given -> a numpy array, without importing torch."""
    import numpy as np

    if hasattr(value, "detach"):  # a torch tensor, possibly on device
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        try:
            value = value.numpy()
        except TypeError:
            # numpy has no bfloat16, so `.numpy()` raises "Got unsupported
            # ScalarType BFloat16". WEIGHT_DTYPE **is** bfloat16 and is
            # documented as not configurable, so this is the DEFAULT path for
            # every audio-exporting mode rather than an edge case: six of eleven
            # modes generated successfully and died here, at the last step.
            #
            # Handled in `_as_array` and not in `_as_audio_planes` because this
            # helper promises "whatever we were given -> a numpy array". A dtype
            # it cannot convert is its problem, and fixing it at one caller
            # leaves the other latent -- which is how this survived: a shared
            # helper with two callers where only one ever passed the bad input.
            #
            # float32, deliberately, NOT float16. A silent narrowing inside the
            # module whose entire job is "never silently wrong" would be its own
            # defect; `.float()` is float32 in torch whatever the source dtype.
            #
            # Selected by what numpy can REPRESENT rather than by dtype name, so
            # any future dtype without a numpy equivalent takes the same route
            # without anyone having to remember to add it here.
            if not hasattr(value, "float"):
                raise
            value = value.float().numpy()
    return np.asarray(value)


def _drop_batch_axes(array, *, target_ndim: int, what: str, target: Path):
    """Peel leading singleton axes until the rank is the one we can write."""
    while array.ndim > target_ndim:
        if array.shape[0] != 1:
            raise EncodeError(
                path=str(target),
                reason=(
                    f"the pipeline returned a batch of {array.shape[0]} {what} "
                    f"(shape {tuple(array.shape)}); this writer emits one run's artifact and "
                    f"will not silently keep only the first of them"
                ),
            )
        array = array[0]
    return array


def _as_video_frames(frames, *, target: Path):
    """-> contiguous uint8 ``[F, H, W, 3]`` RGB, ready for ``av``."""
    import numpy as np

    array = _as_array(frames)
    array = _drop_batch_axes(array, target_ndim=4, what="frame stacks", target=target)
    if array.ndim != 4:
        raise EncodeError(
            path=str(target),
            reason=(
                f"expected frames shaped [frames, height, width, channels], got "
                f"{array.ndim}D {tuple(array.shape)}"
            ),
        )

    # Channels-first ``[F, C, H, W]`` is what a torch pipeline hands back.
    if array.shape[-1] not in (_GREY_CHANNELS, _RGB_CHANNELS) and array.shape[1] in (
        _GREY_CHANNELS,
        _RGB_CHANNELS,
    ):
        array = array.transpose(0, 2, 3, 1)
    if array.shape[-1] == _GREY_CHANNELS:
        array = np.repeat(array, _RGB_CHANNELS, axis=-1)
    if array.shape[-1] != _RGB_CHANNELS:
        raise EncodeError(
            path=str(target),
            reason=(
                f"expected {_RGB_CHANNELS}-channel RGB frames, got {array.shape[-1]} channels "
                f"in {tuple(array.shape)}"
            ),
        )

    if array.dtype != np.uint8:
        # Diffusers' float output is [0, 1]; clipping first means an out-of-range
        # value shows as a clamped pixel rather than wrapping into garbage.
        array = (np.clip(array, 0.0, 1.0) * _UINT8_MAX).round().astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_audio_planes(audio, *, target: Path):
    """-> contiguous float32 ``[channels, samples]``. **Never downmixed.**

    The channel axis is identified as the short one rather than assumed to be
    first, because the vocoder's exact output layout is not something this
    module should have to know. If neither axis is plausibly channels we raise:
    transposing a waveform by mistake turns it into noise, silently.
    """
    import numpy as np

    array = _as_array(audio)
    array = _drop_batch_axes(array, target_ndim=2, what="audio tracks", target=target)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise EncodeError(
            path=str(target),
            reason=f"expected audio shaped [channels, samples], got {tuple(array.shape)}",
        )

    channels, samples = array.shape
    if channels > MAX_AUDIO_CHANNELS and samples > MAX_AUDIO_CHANNELS:
        raise EncodeError(
            path=str(target),
            reason=(
                f"cannot tell which axis of {tuple(array.shape)} is channels: neither is small "
                f"enough to be one (limit {MAX_AUDIO_CHANNELS}). Refusing to guess, because "
                f"transposing audio produces noise rather than an error"
            ),
        )
    if channels > samples:  # [samples, channels] -> [channels, samples]
        array = array.T
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def _required(value, *, what: str, source: str, consequence: str, target: Path):
    """Refuse to invent a number the caller is supposed to know."""
    if value is not None:
        return value
    raise EncodeError(
        path=str(target),
        reason=(
            f"no {what} was supplied. It must come from {source} at runtime rather than being "
            f"assumed here -- {consequence}"
        ),
    )


def _add_audio_stream(container, *, channels: int, sampling_rate: int, codec: str):
    """An audio stream whose layout is *derived* from the channel count.

    FFmpeg's ``"{n}c"`` spelling resolves to the canonical layout for that many
    channels, so nothing here names a layout. A constant ``"stereo"`` would be a
    downmix waiting to happen the first time the vocoder changes.
    """
    import av

    stream = container.add_stream(codec, rate=int(sampling_rate))
    stream.layout = av.AudioLayout(f"{channels}c")
    # Opening now makes ``frame_size`` authoritative instead of a guess: fixed
    # frame-size encoders (AAC) reject chunks of any other length.
    stream.codec_context.open()
    return stream


def _encode_audio(container, stream, planes, *, sampling_rate: int) -> None:
    import av

    source = av.AudioFrame.from_ndarray(
        planes, format=AUDIO_FRAME_FORMAT, layout=stream.layout.name
    )
    source.sample_rate = int(sampling_rate)

    resampler = av.AudioResampler(
        format=stream.format.name, layout=stream.layout.name, rate=stream.rate
    )
    fifo = av.audio.fifo.AudioFifo()
    for resampled in resampler.resample(source):
        fifo.write(resampled)

    # ``frame_size`` is 0 for codecs that accept any length (PCM); ``read(None)``
    # then drains the lot in one frame, which is what we want.
    chunk_size = stream.frame_size or None
    while True:
        chunk = fifo.read(chunk_size) if chunk_size else fifo.read()
        if chunk is None:
            break
        for packet in stream.encode(chunk):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)


def dropped_audio_warning(mode: str) -> str:
    """Why a video-only mode's file has no sound.

    The audio existed: the model is jointly audio-visual and renders it whether
    or not the mode exports it. Discarding it without saying so would leave a
    user who expected sound with silence and no explanation -- which is the
    silent-wrongness shape, just quieter.
    """
    return (
        f"mode {mode!r} exports video only, so the audio track this generation produced was "
        f"dropped rather than muxed. The model is jointly audio-visual and has no video-off "
        f"switch, so that audio was rendered and paid for either way -- use the matching "
        f"audio+video mode to keep it."
    )


def _write_video(
    frames, audio, target: Path, *, frame_rate, sampling_rate, include_audio: bool, mode: str = ""
) -> tuple[str, ...]:
    """Encode the frames, muxing the audio track only if the mode exports it.

    Returns any warnings the caller should surface. Audio present with no sample
    rate raises rather than falling back to a video-only write: silently
    dropping the audio track from a *joint* generation is the exact defect this
    module is written against. Dropping it from a **video-only** mode is the
    opposite case -- declared, intended, and reported rather than silent.
    """
    import av

    frame_rate = _required(
        frame_rate,
        what="frame rate",
        source="the request that drove generation (the rate the pipeline generated at)",
        consequence="a container written at the wrong rate plays at the wrong speed and "
        "reports no error",
        target=target,
    )
    notes: list[str] = []
    planes = None
    if include_audio and audio is None:
        raise EncodeError(
            path=str(target),
            reason=(
                f"mode {mode!r} declares that it exports audio, but this generation produced "
                f"none. exports is a contract: emitting the video alone would hand back a file "
                f"that satisfies the shape of the promise and not its content"
            ),
        )
    if audio is not None and not include_audio:
        # Declared, intended, and reported -- never silent. See the module docstring.
        notes.append(dropped_audio_warning(mode))
        _log.warning("%s", notes[-1])
        audio = None
    if audio is not None:
        sampling_rate = _required(
            sampling_rate,
            what="sampling rate",
            source="vocoder.config.output_sampling_rate",
            consequence="this generation produced an audio track, and writing the file without "
            "it would hand back a silent video that looks like a success",
            target=target,
        )
        planes = _as_audio_planes(audio, target=target)

    array = _as_video_frames(frames, target=target)
    _, height, width, _ = array.shape

    container = av.open(str(target), mode="w")
    try:
        video = container.add_stream(
            VIDEO_CODEC, rate=Fraction(float(frame_rate)).limit_denominator()
        )
        video.width = int(width)
        video.height = int(height)
        video.pix_fmt = VIDEO_PIXEL_FORMAT

        audio_stream = None
        if planes is not None:
            audio_stream = _add_audio_stream(
                container,
                channels=planes.shape[0],
                sampling_rate=sampling_rate,
                codec=MUXED_AUDIO_CODEC,
            )

        for frame in array:
            for packet in video.encode(
                av.VideoFrame.from_ndarray(frame, format=VIDEO_FRAME_FORMAT)
            ):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        if audio_stream is not None:
            _encode_audio(container, audio_stream, planes, sampling_rate=sampling_rate)
    finally:
        container.close()
    return tuple(notes)


def _write_audio(audio, target: Path, *, sampling_rate) -> None:
    """Write the audio track alone, preserving its channel count.

    Uncompressed PCM, so what comes back is what the vocoder produced. The
    sample rate must come from ``vocoder.config.output_sampling_rate`` at
    runtime -- never assumed -- and the channel count survives, because the
    vocoder is stereo and a mono downmix is a silent quality regression.
    """
    import av

    if audio is None:
        raise EncodeError(path=str(target), reason="the pipeline produced no audio to export")
    sampling_rate = _required(
        sampling_rate,
        what="sampling rate",
        source="vocoder.config.output_sampling_rate",
        consequence="a file written with the wrong declared rate is pitch-shifted and reports "
        "no error",
        target=target,
    )
    planes = _as_audio_planes(audio, target=target)

    container = av.open(str(target), mode="w")
    try:
        stream = _add_audio_stream(
            container,
            channels=planes.shape[0],
            sampling_rate=sampling_rate,
            codec=AUDIO_ONLY_CODEC,
        )
        _encode_audio(container, stream, planes, sampling_rate=sampling_rate)
    finally:
        container.close()


def write_output(
    output,
    *,
    run_dir,
    mode: str,
    exports=("video",),
    seed: int = 0,
    frame_rate=None,
    sampling_rate=None,
    warnings=None,
    **extra,
) -> Path:
    """Encode the pipeline output into ``run_dir`` and return the artifact path.

    Every run gets its own directory, so an artifact is only ever looked up
    inside the run that produced it (PLAN §5.3).

    ``frame_rate`` is the rate the pipeline generated at and ``sampling_rate``
    is ``vocoder.config.output_sampling_rate``. Neither is defaulted: see the
    module docstring for why a second independent assumption about either is a
    silent-wrongness bug rather than a convenience.

    ``exports`` decides the container's contents, not just its name: a mode that
    does not export ``audio`` gets a file with no audio stream. ``warnings`` is
    an optional list the caller may pass to collect anything worth telling the
    user -- currently the dropped-audio note. It is also logged unconditionally,
    so the information survives a caller that does not pass a sink.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frames = getattr(output, "frames", None)
    audio = getattr(output, "audio", None)
    wants_video = "video" in exports
    target = run_dir / (VIDEO_FILENAME if wants_video else AUDIO_FILENAME)

    try:
        if wants_video:
            notes = _write_video(
                frames,
                audio,
                target,
                frame_rate=frame_rate,
                sampling_rate=sampling_rate,
                include_audio="audio" in exports,
                mode=mode,
            )
            if warnings is not None:
                warnings.extend(notes)
            written = ("video", "audio") if not notes and audio is not None else ("video",)
        else:
            _write_audio(audio, target, sampling_rate=sampling_rate)
            written = ("audio",)
    except LTXStudioError:
        raise
    except Exception as exc:
        raise EncodeError(path=str(target), reason=str(exc)) from exc
    return dict.fromkeys(written, target)
