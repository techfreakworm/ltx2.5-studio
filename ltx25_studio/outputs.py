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

**One track in this module does not come from the model, and it is a caller's
decision, not this module's.** :class:`SuppliedAudio` lets ``generate.py`` name
the waveform the container must carry instead of the pipeline's own. It exists
because the model's audio generation is *suppressed* when audio is supplied as
input -- measured on the live Space at identical geometry and seed:
``audio_to_video`` with audio in wrote a well-formed 48 kHz stereo AAC track at
**-88.3 dBFS** (digital silence) while ``text_to_audio_video`` with no audio in
wrote **-24.2 dBFS** of real content. (Those two runs also differ in pipeline
class and ``noise_scale``, so the cause is not pinned down; the user's own track
is the better one to hand back either way.) The substitute waveform and its own
sample rate both arrive from the caller. The writer converts it to the
soundtrack's format -- the vocoder's rate, stereo -- so the file plays like any
other run's and its pitch and length are unchanged; the run says so when that
changed anything.

**The model's audio is stereo and stays stereo.** The vocoder reports two output
channels. Nothing here may mono-downmix it: the channel count is read off the
array that arrives and the container layout is *derived* from it (FFmpeg's
``"{n}c"`` spelling), so a stereo track cannot be silently flattened by a layout
constant that says otherwise.

**Nothing assumes a frame rate or a sample rate.** Both must be supplied by the
caller, which knows them: the frame rate is the one the pipeline generated at,
and the sample rate is ``vocoder.config.output_sampling_rate`` (PLAN §5.4). A
default here would be a second, independent assumption about a number that
already lives somewhere else -- and a container written at the wrong rate is a
video that plays at the wrong speed, which again reports as nothing. Absent,
they raise.

**The frames are looked at on the way out, and never refused.** A file can be
well-formed, the right length, the right size, full of distinct frames -- and be
flat grey, a picture under a grey veil, or colour static. :func:`assess_picture`
names those three from a few frames; what it finds rides back on the result's
warnings channel as a plain-language hint and **nothing here raises on it**,
because by the time these frames exist the user has paid for them and a
statistic is not a verdict. Emit the file, attach the note, let a human look.

``av`` is the one media stack: it is already a pinned dependency for decoding
conditioning video and uploaded audio, and using it here means the audio-only
export and the resampling of a supplied track need no second library. It is
imported lazily so this module -- and the retention policy -- stay importable
and testable with no media stack present (§9.1).

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
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .errors import EncodeError, LTXStudioError

_log = logging.getLogger(__name__)

__all__ = [
    "AUDIO_FILENAME",
    "AUDIO_ONLY_CODEC",
    "FLAT_MAX_LUMA_STD",
    "LOW_BLOCK_STD",
    "LOW_LUMA_BAND",
    "MAX_AUDIO_CHANNELS",
    "MAX_HIGH_FREQUENCY_RATIO",
    "MIN_STRUCTURE_RATIO",
    "MUXED_AUDIO_CHANNELS",
    "MUXED_AUDIO_CODEC",
    "PICTURE_CONDITIONS",
    "STATIC_MAX_CORRELATION",
    "STATIC_MIN_COLOURFULNESS",
    "STRUCTURE_ANALYSIS_SIZE",
    "STRUCTURE_SAMPLE_FRAMES",
    "VEILED_MAX_COLOURFULNESS",
    "VEILED_MAX_WHITE_LEVEL",
    "VEILED_MIN_BLACK_LEVEL",
    "VIDEO_CODEC",
    "VIDEO_COLOUR_MATRIX",
    "VIDEO_COLOUR_RANGE",
    "VIDEO_COLOUR_STANDARD",
    "VIDEO_CONTAINER_OPTIONS",
    "VIDEO_CRF",
    "VIDEO_FILENAME",
    "VIDEO_PIXEL_FORMAT",
    "VIDEO_PRESET",
    "FrameStructure",
    "PictureCheck",
    "PictureCondition",
    "PictureMeasurements",
    "SuppliedAudio",
    "assess_picture",
    "directory_size",
    "dropped_audio_warning",
    "enforce_retention",
    "measure_structure",
    "run_age",
    "structure_failure",
    "structure_notes",
    "structure_warnings",
    "write_output",
]

VIDEO_FILENAME = "output.mp4"
AUDIO_FILENAME = "output.wav"

#: H.264 in MP4: the combination every browser Gradio might render this in can
#: actually play. ``yuv420p`` for the same reason -- 4:4:4 output is legal H.264
#: that Safari and most hardware decoders refuse.
VIDEO_CODEC = "libx264"
VIDEO_PIXEL_FORMAT = "yuv420p"

#: Visually lossless for 8-bit H.264; x264's default of 23 smears fine detail.
VIDEO_CRF = 17
#: x264's balanced preset; slower ones cost encode time on every run for little gain at CRF 17.
VIDEO_PRESET = "medium"
#: Rec. 709 primaries and transfer (``ColorPrimaries`` / ``ColorTrc`` member): the HD standard.
VIDEO_COLOUR_STANDARD = "BT709"
#: Rec. 709 matrix (``Colorspace`` member), used for the RGB->YUV conversion AND tagged, so the
#: tag describes the pixels; untagged files are often decoded as Rec. 601.
VIDEO_COLOUR_MATRIX = "ITU709"
#: Limited ("TV") range (``ColorRange`` member), what H.264 players expect for yuv420p.
VIDEO_COLOUR_RANGE = "MPEG"
#: The index (``moov``) goes before the media so a browser can start playing before the
#: whole file has arrived.
VIDEO_CONTAINER_OPTIONS = {"movflags": "+faststart"}

#: AAC for the muxed track (MP4's normal audio codec) and uncompressed PCM for
#: the audio-only export, where the point is to hand back exactly what the
#: vocoder produced rather than a re-compressed approximation of it.
MUXED_AUDIO_CODEC = "aac"
AUDIO_ONLY_CODEC = "pcm_s16le"

#: A supplied track is muxed with this many channels: the vocoder's layout, which is
#: what every other run's soundtrack has and what ``tests/verify_streams.py`` expects.
MUXED_AUDIO_CHANNELS = 2

#: Plain downmix of more than two channels: front left/right at full level, every
#: other channel at -3 dB, the LFE left out, each side then normalised so it cannot clip.
_DOWNMIX_FRONT = ("FL", "FR")
_DOWNMIX_OTHER_WEIGHT = 0.7071


@dataclass(frozen=True)
class SuppliedAudio:
    """The user's OWN waveform, to be muxed in place of the model's track.

    **A pair, not two arguments, because the two must not drift.** The rate here
    is the *upload's* rate, and it is emphatically not
    ``vocoder.config.output_sampling_rate`` -- on this stack those are 16 kHz and
    48 kHz respectively, and muxing a 16 kHz waveform at 48 kHz pitch-shifts it
    silently, which is precisely the defect class :func:`_required` exists to
    refuse. Carrying the waveform and its rate as one frozen record means the
    writer cannot be handed one without the other. The writer resamples it from
    this rate to the vocoder's and brings it to stereo before muxing
    (:func:`_conformed_supplied_audio`).

    **What it must contain: the RECONCILED waveform, never the raw upload.**
    ``generate._prepared_audio`` reconciles a supplied file against the video
    length before encoding it (untouched inside a +/-0.05 s tolerance, trimmed to
    upstream's own latent grid when long, refused when short), and that same
    reconciled waveform is what conditioned the render. Muxing the original file
    instead would reintroduce at the output end exactly the desync ``2eaeb2c``
    fixed at the input end -- a 10 s upload against a 5 s video conditions on 5 s
    and would ship with 10 s of sound.

    ``waveform`` is whatever ``audio.load_waveform`` returns (a torch tensor
    shaped ``[channels, samples]``); it is typed loosely because this module is
    importable with no torch present and :func:`_as_audio_planes` already owns
    the conversion.
    """

    waveform: object
    sampling_rate: int


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


def _describe_track(channels: int, sampling_rate: int) -> str:
    layout = {1: "mono", 2: "stereo"}.get(channels, f"{channels}-channel")
    return f"{layout} at {sampling_rate / 1000:g} kHz"


def _downmixed_to_stereo(planes):
    """More than two channels -> stereo. See ``_DOWNMIX_FRONT``.

    Channels are named by FFmpeg's default layout for their count (6 is 5.1:
    FL FR FC LFE BL BR), the same ``"{n}c"`` convention the writer uses.
    """
    import av
    import numpy as np

    names = [channel.name for channel in av.AudioLayout(f"{planes.shape[0]}c").channels]
    sides = np.zeros((2, planes.shape[1]), dtype=np.float32)
    weights = np.zeros(2, dtype=np.float32)
    for plane, name in zip(planes, names, strict=True):
        if "LFE" in name:
            continue
        weight = 1.0 if name in _DOWNMIX_FRONT else _DOWNMIX_OTHER_WEIGHT
        left, right = "L" in name, "R" in name
        for side, applies in ((0, left or not right), (1, right or not left)):
            if applies:
                sides[side] += weight * plane
                weights[side] += weight
    return np.ascontiguousarray(sides / np.maximum(weights, 1e-9)[:, None])


def _as_stereo(planes):
    import numpy as np

    channels = planes.shape[0]
    if channels == MUXED_AUDIO_CHANNELS:
        return planes
    if channels == 1:
        return np.ascontiguousarray(np.repeat(planes, MUXED_AUDIO_CHANNELS, axis=0))
    return _downmixed_to_stereo(planes)


def _resampled_stereo(planes, *, from_rate: int, to_rate: int):
    """Stereo float planes at another rate, with FFmpeg's resampler.

    The length is pinned to ``round(samples * to_rate / from_rate)``, so the
    duration is kept to within one output sample.
    """
    import av
    import numpy as np

    if from_rate == to_rate:
        return planes
    source = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(planes, dtype=np.float32), format=AUDIO_FRAME_FORMAT, layout="stereo"
    )
    source.sample_rate = from_rate
    resampler = av.AudioResampler(format=AUDIO_FRAME_FORMAT, layout="stereo", rate=to_rate)
    chunks = [
        frame.to_ndarray() for frame in (*resampler.resample(source), *resampler.resample(None))
    ]
    resampled = np.concatenate(chunks, axis=1) if chunks else np.zeros((2, 0), dtype=np.float32)
    wanted = (planes.shape[1] * to_rate + from_rate // 2) // from_rate
    resampled = resampled[:, :wanted]
    if resampled.shape[1] < wanted:
        resampled = np.pad(resampled, ((0, 0), (0, wanted - resampled.shape[1])))
    return np.ascontiguousarray(resampled, dtype=np.float32)


def _conformed_supplied_audio(supplied: SuppliedAudio, *, sampling_rate: int, target: Path):
    """The user's track in the soundtrack's format: ``sampling_rate``, stereo.

    Returns ``(planes, note)``; ``note`` is ``None`` when nothing had to change.
    """
    try:
        source_rate = int(supplied.sampling_rate)
    except (TypeError, ValueError):
        source_rate = 0
    if source_rate <= 0:
        raise EncodeError(
            path=str(target),
            reason=(
                f"the supplied audio's sampling rate {supplied.sampling_rate!r} is not a "
                f"positive number of samples per second"
            ),
        )
    planes = _as_audio_planes(supplied.waveform, target=target)
    channels = planes.shape[0]
    planes = _resampled_stereo(_as_stereo(planes), from_rate=source_rate, to_rate=sampling_rate)
    if channels == MUXED_AUDIO_CHANNELS and source_rate == sampling_rate:
        return planes, None
    note = (
        f"Your audio was converted from {_describe_track(channels, source_rate)} to "
        f"{_describe_track(MUXED_AUDIO_CHANNELS, sampling_rate)} to match the video's "
        f"soundtrack. Its pitch and length are unchanged."
    )
    if channels > MUXED_AUDIO_CHANNELS:
        note += " Its extra channels were mixed down into left and right."
    return planes, note


# ---------------------------------------------------------------------------
# picture check -- is there a usable picture in there?
# ---------------------------------------------------------------------------
#
# Three conditions, each with its own reason:
#
# * flat: no picture. Brightness barely varies across the frame.
# * veiled: a picture under a grey veil. The veil pulls both ends of the tonal
#   range toward grey: no blacks, no bright highlights, almost no colour. A dark,
#   low-key shot keeps its blacks and a bright, overcast one its highlights.
# * static: coloured speckle. Brightness at one point says nothing about the
#   brightness an eighth of a frame away, where any real scene has shapes,
#   light and horizon; and it is strongly coloured, so grey texture (snow,
#   gravel, a starfield) is left alone.
#
# Up to STRUCTURE_SAMPLE_FRAMES evenly spaced frames are resized to one analysis
# size before anything is measured, so a picture gets the same answer at any
# resolution. Each statistic is the median over those frames. Nothing here
# refuses a write: a condition becomes a plain-language hint on the result.

#: Long side, in pixels, that every sampled frame is resized to first.
STRUCTURE_ANALYSIS_SIZE = 256

#: Frames looked at per video, evenly spaced, both ends included.
STRUCTURE_SAMPLE_FRAMES = 16

# Thresholds from a labelled set of 42 outputs, every one looked at, measured
# 2026-11-16: 27 good, 3 flat grey, 9 veiled, 3 colour static. Brackets: the
# worst good and worst bad value at the analysis size. Colourfulness is
# Hasler & Süsstrunk's (15 "slightly", 33 "moderately colourful").
FLAT_MAX_LUMA_STD = 6.0  # luma std, 0..255 (good >= 14.9, grey <= 4.3)
# veiled needs both: no good file has lifted blacks AND dim highlights.
VEILED_MIN_BLACK_LEVEL = 40.0  # luma p1 (veiled >= 57.6; dim-highlight good <= 1.0)
VEILED_MAX_WHITE_LEVEL = 135.0  # luma p99 (veiled <= 115.1; lifted-black good >= 176.8)
VEILED_MAX_COLOURFULNESS = 20.0  # (veiled <= 13.8)
STATIC_MAX_CORRELATION = 0.10  # luma correlation at 1/8 frame (good >= 0.188, static <= 0.033)
STATIC_MIN_COLOURFULNESS = 33.0  # (static >= 62.9)

# The retired structure-ratio gate's names and values, kept importable for older
# callers. No condition reads them.
MIN_STRUCTURE_RATIO = 0.70
MAX_HIGH_FREQUENCY_RATIO = 1.0
LOW_BLOCK_STD = 8.0
LOW_LUMA_BAND = 32.0

#: Correlation lag in analysis pixels: an eighth of the long side.
_CORRELATION_LAG = STRUCTURE_ANALYSIS_SIZE // 8

#: Box side for the report-only block statistics.
_BLOCK = 8

#: Below this luma std a frame has no variation to normalise a ratio by.
_MIN_CONTRAST = 1e-3

_HINT = (
    "If that is not what you asked for, try another seed or prompt. This is an automatic "
    "check and only a hint: your video was saved as usual."
)


@dataclass(frozen=True)
class PictureCondition:
    """One named way a picture goes wrong. ``rule`` is for logs; ``note`` is what the user reads."""

    name: str
    rule: str
    note: str


PICTURE_CONDITIONS = (
    PictureCondition(
        name="flat",
        rule=f"luma std < {FLAT_MAX_LUMA_STD:g}",
        note=f"This picture looks almost completely flat: one even tone with no shapes or "
        f"detail in it. {_HINT}",
    ),
    PictureCondition(
        name="veiled",
        rule=(
            f"luma p1 >= {VEILED_MIN_BLACK_LEVEL:g}, luma p99 < {VEILED_MAX_WHITE_LEVEL:g} and "
            f"colourfulness < {VEILED_MAX_COLOURFULNESS:g}"
        ),
        note=f"This picture is very low in contrast and almost colourless, as if seen through "
        f"a grey haze, so it may be washed out. {_HINT}",
    ),
    PictureCondition(
        name="static",
        rule=(
            f"luma correlation at 1/8 frame < {STATIC_MAX_CORRELATION:g} and colourfulness >= "
            f"{STATIC_MIN_COLOURFULNESS:g}"
        ),
        note=f"This picture looks like coloured speckle with no larger shapes in it, so it may "
        f"be noise rather than a scene. {_HINT}",
    ),
)

_CONDITIONS = {condition.name: condition for condition in PICTURE_CONDITIONS}


@dataclass(frozen=True)
class PictureMeasurements:
    """Medians over the sampled frames, each resized to the analysis size first.

    ``None`` means "could not measure" (no frames, or nothing varies), never
    "fine". The last five are reported for continuity with older runs; no
    condition reads them.
    """

    frames: int
    sampled: int
    flat_frames: int
    luma_std: float | None
    black_level: float | None
    white_level: float | None
    colourfulness: float | None
    long_range_correlation: float | None
    luma_band: float | None
    structure_ratio: float | None
    high_frequency_ratio: float | None
    block_std: float | None
    saturation: float | None


#: The measurement record's older name.
FrameStructure = PictureMeasurements


@dataclass(frozen=True)
class PictureCheck:
    """What :func:`assess_picture` found: the numbers, the conditions that fired, the notes."""

    measurements: PictureMeasurements
    conditions: tuple[str, ...]
    notes: tuple[str, ...]


def _resample_weights(size_in: int, size_out: int):
    """``[size_out, size_in]`` weights: area averages going down, linear going up."""
    import numpy as np

    if size_out < size_in:
        scale = size_in / size_out
        start = np.arange(size_out, dtype=np.float64)[:, None] * scale
        pixel = np.arange(size_in, dtype=np.float64)[None, :]
        overlap = np.minimum(start + scale, pixel + 1.0) - np.maximum(start, pixel)
        return (np.clip(overlap, 0.0, None) / scale).astype(np.float32)
    position = np.clip((np.arange(size_out) + 0.5) * size_in / size_out - 0.5, 0, size_in - 1)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, size_in - 1)
    fraction = (position - low).astype(np.float32)
    weights = np.zeros((size_out, size_in), dtype=np.float32)
    rows = np.arange(size_out)
    np.add.at(weights, (rows, low), 1.0 - fraction)
    np.add.at(weights, (rows, high), fraction)
    return weights


def _analysis_shape(height: int, width: int) -> tuple[int, int]:
    scale = STRUCTURE_ANALYSIS_SIZE / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def _at_analysis_size(frame, weights: dict):
    """One ``[H, W, 3]`` frame as float32 at the analysis size, aspect kept.

    ``weights`` caches the resampling matrices across the frames of one call.
    """
    import numpy as np

    values = np.asarray(frame, dtype=np.float32)
    height, width = values.shape[:2]
    out_height, out_width = _analysis_shape(height, width)

    def matrix(size_in, size_out):
        if (size_in, size_out) not in weights:
            weights[size_in, size_out] = _resample_weights(size_in, size_out)
        return weights[size_in, size_out]

    def rows(array):
        if out_height == height:
            return array
        return np.tensordot(matrix(height, out_height), array, axes=(1, 0))

    def columns(array):
        if out_width == width:
            return array
        return np.tensordot(array, matrix(width, out_width), axes=(1, 1)).transpose(0, 2, 1)

    # Shrinking the axis that ends up smaller first costs fewer multiplications.
    if out_height <= out_width:
        return np.ascontiguousarray(columns(rows(values)))
    return np.ascontiguousarray(rows(columns(values)))


def _block_means(luma):
    """The 8x8 box means; ragged right/bottom edges are dropped, not padded."""
    height, width = luma.shape
    rows, columns = height // _BLOCK, width // _BLOCK
    if rows == 0 or columns == 0:
        return None
    trimmed = luma[: rows * _BLOCK, : columns * _BLOCK]
    return trimmed.reshape(rows, _BLOCK, columns, _BLOCK).mean(axis=(1, 3))


def _laplacian4(luma):
    """``4c - up - down - left - right``, on the interior only."""
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return None
    middle = luma[1:-1, 1:-1]
    return 4.0 * middle - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:]


def _correlation(luma, lag: int):
    """Mean of the horizontal and vertical luma autocorrelation at ``lag`` pixels."""
    import numpy as np

    centred = luma - luma.mean()
    variance = float(np.mean(centred * centred))
    pairs = []
    if centred.shape[1] > lag:
        pairs.append(float(np.mean(centred[:, lag:] * centred[:, :-lag])))
    if centred.shape[0] > lag:
        pairs.append(float(np.mean(centred[lag:] * centred[:-lag])))
    if not pairs or variance <= 0.0:
        return None
    return sum(pairs) / len(pairs) / variance


def _frame_statistics(frame, weights: dict) -> dict:
    import numpy as np

    rgb = _at_analysis_size(frame, weights)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = red * 0.299 + green * 0.587 + blue * 0.114
    luma_std = float(luma.std())
    black, white = (float(value) for value in np.percentile(luma, (1.0, 99.0)))
    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    high, low = rgb.max(axis=-1), rgb.min(axis=-1)
    blocks = _block_means(luma)
    stats = {
        "luma_std": luma_std,
        "luma_band": white - black,
        "black_level": black,
        "white_level": white,
        "colourfulness": float(
            np.hypot(red_green.std(), yellow_blue.std())
            + 0.3 * np.hypot(red_green.mean(), yellow_blue.mean())
        ),
        "long_range_correlation": None,
        "structure_ratio": None,
        "high_frequency_ratio": None,
        "block_std": None if blocks is None else float(blocks.std()),
        "saturation": float(
            np.mean(np.where(high > 0.0, (high - low) / np.maximum(high, 1.0), 0.0))
        ),
    }
    if luma_std >= _MIN_CONTRAST:
        stats["long_range_correlation"] = _correlation(luma, _CORRELATION_LAG)
        if stats["block_std"] is not None:
            stats["structure_ratio"] = stats["block_std"] / luma_std
        laplacian = _laplacian4(luma)
        if laplacian is not None:
            stats["high_frequency_ratio"] = float(laplacian.std()) / luma_std
    return stats


def _sampled_indices(count: int, sample):
    """``sample`` evenly-spaced frame indices, endpoints included."""
    import numpy as np

    if sample is None or sample >= count:
        return list(range(count))
    return [int(index) for index in np.unique(np.linspace(0, count - 1, sample).round())]


def measure_structure(frames, *, sample=STRUCTURE_SAMPLE_FRAMES) -> PictureMeasurements:
    """The numbers :func:`assess_picture` judges, over ``frames`` (``[F, H, W, 3]`` uint8 RGB).

    Reads at most :data:`STRUCTURE_SAMPLE_FRAMES` frames whatever ``sample``
    says (``None`` included), so the work is bounded for any length of video.
    ``frames`` may be an array or any sequence; only the sampled items are read.
    """
    import numpy as np

    count = len(frames)
    limit = STRUCTURE_SAMPLE_FRAMES if sample is None else int(sample)
    limit = max(1, min(limit, STRUCTURE_SAMPLE_FRAMES))
    indices = _sampled_indices(count, limit) if count else []
    weights: dict = {}
    per_frame = [_frame_statistics(frames[index], weights) for index in indices]

    def median(name):
        values = [stats[name] for stats in per_frame if stats[name] is not None]
        return float(np.median(values)) if values else None

    return PictureMeasurements(
        frames=count,
        sampled=len(indices),
        flat_frames=sum(1 for stats in per_frame if stats["luma_std"] < _MIN_CONTRAST),
        luma_std=median("luma_std"),
        black_level=median("black_level"),
        white_level=median("white_level"),
        colourfulness=median("colourfulness"),
        long_range_correlation=median("long_range_correlation"),
        luma_band=median("luma_band"),
        structure_ratio=median("structure_ratio"),
        high_frequency_ratio=median("high_frequency_ratio"),
        block_std=median("block_std"),
        saturation=median("saturation"),
    )


def _fired(measured: PictureMeasurements) -> tuple[str, ...]:
    """At most one condition: flat, else veiled, else static."""

    def below(value, limit):
        return value is not None and value < limit

    def at_least(value, limit):
        return value is not None and value >= limit

    if below(measured.luma_std, FLAT_MAX_LUMA_STD):
        return ("flat",)
    if (
        at_least(measured.black_level, VEILED_MIN_BLACK_LEVEL)
        and below(measured.white_level, VEILED_MAX_WHITE_LEVEL)
        and below(measured.colourfulness, VEILED_MAX_COLOURFULNESS)
    ):
        return ("veiled",)
    if below(measured.long_range_correlation, STATIC_MAX_CORRELATION) and at_least(
        measured.colourfulness, STATIC_MIN_COLOURFULNESS
    ):
        return ("static",)
    return ()


def assess_picture(frames, *, sample=STRUCTURE_SAMPLE_FRAMES) -> PictureCheck:
    """Measure up to 16 frames and name what is wrong with the picture, if anything.

    numpy only, bounded work. A good picture comes back with no conditions and
    no notes.
    """
    measured = measure_structure(frames, sample=sample)
    conditions = _fired(measured)
    return PictureCheck(
        measurements=measured,
        conditions=conditions,
        notes=tuple(_CONDITIONS[name].note for name in conditions),
    )


def _measurements(checked) -> PictureMeasurements:
    return checked.measurements if isinstance(checked, PictureCheck) else checked


def structure_notes(checked: PictureCheck | PictureMeasurements) -> tuple[str, ...]:
    """The notes every result carries: one per condition that fired, none for a good picture.

    Notes only, never a refusal -- by the time these frames exist the user has
    paid for them, and a statistic is a hint, not a verdict.
    """
    if isinstance(checked, PictureCheck):
        return checked.notes
    return tuple(_CONDITIONS[name].note for name in _fired(checked))


#: Older name of :func:`structure_notes`; the offline harness still calls it.
structure_warnings = structure_notes


def structure_failure(checked: PictureCheck | PictureMeasurements) -> str | None:
    """The fired condition with its rule and the numbers behind it, or ``None``.

    For the offline harness and logs, not for the user (the user gets
    :func:`structure_notes`).
    """
    measured = _measurements(checked)
    fired = _fired(measured)
    if not fired:
        return None
    (name,) = fired
    if name == "flat":
        seen = f"luma std {measured.luma_std:.2f}"
    elif name == "veiled":
        seen = (
            f"luma p1 {measured.black_level:.1f}, p99 {measured.white_level:.1f}, "
            f"colourfulness {measured.colourfulness:.1f}"
        )
    else:
        seen = (
            f"correlation {measured.long_range_correlation:.3f}, "
            f"colourfulness {measured.colourfulness:.1f}"
        )
    return f"{name} ({_CONDITIONS[name].rule}): measured {seen}"


def dropped_audio_warning(counterpart_label: str | None = None) -> str:
    """Why a video-only mode's file has no sound, in the UI's words.

    The audio existed: the model is jointly audio-visual and renders it whether
    or not the mode exports it. Discarding it without saying so would leave a
    user who expected sound with silence and no explanation -- which is the
    silent-wrongness shape, just quieter. The mode that keeps the sound is named
    only where one exists (``modes.counterpart_label_for``): video to video has
    none, and a remedy that does not exist is worse than none.
    """
    note = "this mode returns silent video. The sound the model made with it was left out"
    if counterpart_label:
        return f"{note}; {counterpart_label} keeps it."
    return f"{note}."


def _write_video(
    frames,
    audio,
    target: Path,
    *,
    frame_rate,
    sampling_rate,
    include_audio: bool,
    mode: str = "",
    supplied_audio: SuppliedAudio | None = None,
    counterpart_label: str | None = None,
):
    """Encode the frames, muxing the audio track only if the mode exports it.

    Returns ``(notes, planes)``: any warnings the caller should surface, and the
    exact float samples muxed (``None`` when no audio was muxed). Audio present
    with no sample rate raises rather than falling back to a video-only write:
    silently dropping the audio track from a *joint* generation is the exact
    defect this module is written against. Dropping it from a **video-only**
    mode is the opposite case -- declared, intended, and reported rather than
    silent.

    ``supplied_audio`` REPLACES the model's track (:class:`SuppliedAudio`),
    converted from its own rate and channel count to ``sampling_rate`` stereo.
    It applies only where the container is going to carry audio at all -- a
    video-only mode still drops what the model rendered and still says so,
    because substituting there would put sound into a file whose mode declares
    it has none. The substitution happens before the contract check below, so a
    mode that exports audio is satisfied by the supplied track even where the
    model returned nothing.

    The picture is H.264 at :data:`VIDEO_CRF`, converted to yuv420p with the
    Rec. 709 matrix and tagged as such, with the index at the front of the file.
    """
    import av
    from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc

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
    supplied = supplied_audio if include_audio else None
    if supplied is not None:
        audio = supplied.waveform
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
        notes.append(dropped_audio_warning(counterpart_label))
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
        if supplied is not None:
            planes, note = _conformed_supplied_audio(
                supplied, sampling_rate=int(sampling_rate), target=target
            )
            if note is not None:
                notes.append(note)
                _log.warning("%s", note)
        else:
            planes = _as_audio_planes(audio, target=target)

    array = _as_video_frames(frames, target=target)
    _, height, width, _ = array.shape

    # Measured HERE because this is the one place the normalised rgb24 stack
    # already exists: no second decode, no second conversion, ~50 ms for 16
    # frames of 1216x704. WARN only, never a refusal -- see `structure_notes`.
    for note in structure_notes(measure_structure(array, sample=STRUCTURE_SAMPLE_FRAMES)):
        notes.append(note)
        _log.warning("%s", note)

    matrix = Colorspace[VIDEO_COLOUR_MATRIX]
    primaries = ColorPrimaries[VIDEO_COLOUR_STANDARD]
    transfer = ColorTrc[VIDEO_COLOUR_STANDARD]
    colour_range = ColorRange[VIDEO_COLOUR_RANGE]

    container = av.open(str(target), mode="w", container_options=dict(VIDEO_CONTAINER_OPTIONS))
    try:
        video = container.add_stream(
            VIDEO_CODEC, rate=Fraction(float(frame_rate)).limit_denominator()
        )
        video.width = int(width)
        video.height = int(height)
        video.pix_fmt = VIDEO_PIXEL_FORMAT
        video.codec_context.options = {"crf": str(VIDEO_CRF), "preset": VIDEO_PRESET}
        video.codec_context.colorspace = matrix
        video.codec_context.color_primaries = primaries
        video.codec_context.color_trc = transfer
        video.codec_context.color_range = colour_range

        audio_stream = None
        if planes is not None:
            audio_stream = _add_audio_stream(
                container,
                channels=planes.shape[0],
                sampling_rate=sampling_rate,
                codec=MUXED_AUDIO_CODEC,
            )

        for frame in array:
            picture = av.VideoFrame.from_ndarray(frame, format=VIDEO_FRAME_FORMAT).reformat(
                format=VIDEO_PIXEL_FORMAT,
                src_colorspace=matrix,
                dst_colorspace=matrix,
                src_color_range=ColorRange.JPEG,
                dst_color_range=colour_range,
                dst_color_trc=transfer,
                dst_color_primaries=primaries,
            )
            for packet in video.encode(picture):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        if audio_stream is not None:
            _encode_audio(container, audio_stream, planes, sampling_rate=sampling_rate)
    finally:
        container.close()
    return tuple(notes), planes


def _write_audio(audio, target: Path, *, sampling_rate) -> None:
    """Write the audio track alone, preserving its channel count.

    Uncompressed PCM, so what comes back is what the vocoder produced (or, for
    a video+audio run, exactly the samples muxed into the mp4). The sample rate
    must come from ``vocoder.config.output_sampling_rate`` at runtime -- never
    assumed -- and the channel count survives, because the vocoder is stereo
    and a mono downmix is a silent quality regression.
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
    supplied_audio: SuppliedAudio | None = None,
    **extra,
) -> dict[str, Path]:
    """Encode the pipeline output into ``run_dir``; return the artifact path per track.

    Every run gets its own directory, so an artifact is only ever looked up
    inside the run that produced it (PLAN §5.3).

    A mode exporting video and audio gets ``output.mp4`` carrying both streams
    AND ``output.wav``, the same samples that were muxed, so the audio player
    is handed a sound file rather than the video. The mp4 is unchanged by it.

    ``frame_rate`` is the rate the pipeline generated at and ``sampling_rate``
    is ``vocoder.config.output_sampling_rate``. Neither is defaulted: see the
    module docstring for why a second independent assumption about either is a
    silent-wrongness bug rather than a convenience.

    ``exports`` decides the container's contents, not just its name: a mode that
    does not export ``audio`` gets a file with no audio stream. ``warnings`` is
    an optional list the caller may pass to collect anything worth telling the
    user -- the dropped-audio note, and whatever :func:`structure_notes` found
    in the frames. Everything in it is also logged unconditionally, so the
    information survives a caller that does not pass a sink.

    ``supplied_audio`` is the caller naming a waveform to carry INSTEAD of the
    model's -- see :class:`SuppliedAudio`. It is a muxing instruction and
    therefore only meaningful for a mode that writes a video container; handed
    one for an audio-only export this raises rather than ignoring it, because a
    caller that asked for the user's own track and silently received the model's
    would have no way to find out.

    **Nothing measured about the frames can stop this write.** The structure
    detector is a statistical pre-filter and it is wrong about real, murky
    output by design; by the time it runs the user has already spent 35+
    GPU-seconds on these frames. It annotates the result and the file is
    emitted regardless -- a human decides, not a heuristic.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frames = getattr(output, "frames", None)
    audio = getattr(output, "audio", None)
    wants_video = "video" in exports
    target = run_dir / (VIDEO_FILENAME if wants_video else AUDIO_FILENAME)

    if supplied_audio is not None and not wants_video:
        raise EncodeError(
            path=str(target),
            reason=(
                f"mode {mode!r} declares exports={list(exports)} and writes no video container, "
                f"so there is nothing to mux the supplied audio into. Refusing to drop it "
                f"quietly: the caller asked for the user's own waveform and would otherwise "
                f"receive the model's with nothing to say so"
            ),
        )

    try:
        if wants_video:
            notes, muxed = _write_video(
                frames,
                audio,
                target,
                frame_rate=frame_rate,
                sampling_rate=sampling_rate,
                include_audio="audio" in exports,
                mode=mode,
                supplied_audio=supplied_audio,
                counterpart_label=extra.get("counterpart_label"),
            )
            if warnings is not None:
                warnings.extend(notes)
            # The track is in the container exactly when `_write_video` muxed
            # one -- never inferred from the notes, which also carry unrelated
            # warnings. `supplied_audio` counts: it is the track in the file.
            written = {"video": target}
            if muxed is not None:
                track = run_dir / AUDIO_FILENAME
                _write_audio(muxed, track, sampling_rate=sampling_rate)
                written["audio"] = track
        else:
            _write_audio(audio, target, sampling_rate=sampling_rate)
            written = {"audio": target}
    except LTXStudioError:
        raise
    except Exception as exc:
        raise EncodeError(path=str(target), reason=str(exc)) from exc
    return written
