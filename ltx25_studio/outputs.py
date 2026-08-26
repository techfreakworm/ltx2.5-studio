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
wrote **-24.2 dBFS** of real content. 64 dB of separation, one variable. Muxing
that empty track is the same emptiness wearing a stream header. This module
still refuses to *invent* anything: the substitute waveform and its own sample
rate both arrive from the caller, exactly like ``frame_rate`` and
``sampling_rate`` do.

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

**The frames are measured on the way out, and never refused.** A file can be
well-formed, the right length, the right size, full of distinct frames -- and be
colour static. One was, and every content check this project had scored it a
PASS, because distinctness, byte count, geometry and brightness range are all
properties of noise. :func:`measure_structure` asks the one question they cannot:
is there anything at a scale above one pixel? The answer rides back on the
result's warnings channel and **nothing here raises on it**, because by the time
these frames exist the user has already paid 35+ GPU-seconds for them and this
is a heuristic that is wrong about real murky output by design. Emit the file,
attach the note, let a human look.

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
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .errors import EncodeError, LTXStudioError

_log = logging.getLogger(__name__)

__all__ = [
    "AUDIO_FILENAME",
    "AUDIO_ONLY_CODEC",
    "LOW_BLOCK_STD",
    "LOW_LUMA_BAND",
    "MAX_AUDIO_CHANNELS",
    "MAX_HIGH_FREQUENCY_RATIO",
    "MIN_STRUCTURE_RATIO",
    "MUXED_AUDIO_CODEC",
    "STRUCTURE_SAMPLE_FRAMES",
    "VIDEO_CODEC",
    "VIDEO_FILENAME",
    "VIDEO_PIXEL_FORMAT",
    "FrameStructure",
    "SuppliedAudio",
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

#: AAC for the muxed track (MP4's normal audio codec) and uncompressed PCM for
#: the audio-only export, where the point is to hand back exactly what the
#: vocoder produced rather than a re-compressed approximation of it.
MUXED_AUDIO_CODEC = "aac"
AUDIO_ONLY_CODEC = "pcm_s16le"


@dataclass(frozen=True)
class SuppliedAudio:
    """The user's OWN waveform, to be muxed in place of the model's track.

    **A pair, not two arguments, because the two must not drift.** The rate here
    is the *upload's* rate, and it is emphatically not
    ``vocoder.config.output_sampling_rate`` -- on this stack those are 16 kHz and
    48 kHz respectively, and muxing a 16 kHz waveform at 48 kHz pitch-shifts it
    silently, which is precisely the defect class :func:`_required` exists to
    refuse. Carrying the waveform and its rate as one frozen record means the
    writer cannot be handed one without the other.

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


# ---------------------------------------------------------------------------
# structure -- is there a picture in there, or only colour?
# ---------------------------------------------------------------------------
#
# Every content check this project had could be passed by structureless
# garbage. The artifact that proved it had 121 of 121 distinct frames, the
# geometry the request asked for, non-zero bytes and a full luma range -- and
# was colour static. Distinctness, byte count, geometry and brightness range
# are all properties of *noise*, so a checker built from them scores noise as a
# pass and always will.
#
# What separates a picture from static is that a picture has structure at a
# scale ABOVE one pixel and static does not. Two ratios say that, and both are
# divided by the frame's own luma standard deviation so that they measure shape
# rather than contrast -- a dim shot and a bright shot of the same scene land in
# the same place:
#
# * ``structure_ratio`` (SR) = std of the 8x8 box means / std of the pixels.
#   Averaging 8x8 blocks of a real picture keeps most of its variation, because
#   neighbouring pixels agree; averaging 8x8 blocks of noise cancels it. Measured
#   0.918..0.994 across every banked real artifact, 0.530 on the failure.
# * ``high_frequency_ratio`` (HF) = std of the 4-neighbour Laplacian / std of the
#   pixels. The Laplacian is what is left after the picture is subtracted, so it
#   is small on a picture and larger than the signal itself on noise. Measured
#   0.147..0.386 across the banked set, 3.146 on the failure.
#
# **Temporal difference is deliberately NOT the signal here, and this module has
# never carried a temporal refusal -- do not add one.** The offline runner did:
# an absolute ``mean |Δ| < 0.5`` gate. It did reject the operator's failure
# (0.403), but by accident and with no usable margin -- the banked ``r7b``
# artifact, then believed to be a legitimate near-static 9-frame render, measures
# 0.864, only 2.0x above the thing being rejected. A gate whose true and false cases are a factor
# of two apart is a coin toss waiting to happen, and this one has since been
# caught landing wrong: it refused a real 512x960x121 render at 0.4071, and that
# render has a sibling at 0.3859 -- a legitimate artifact BELOW the failure the
# threshold was calibrated against. The populations overlap, so the axis cannot
# separate them, so it is a warning there now. SR's are 1.31x and 1.32x either
# side of 0.70, which is the geometric middle of the two populations, and HF's
# are 2.6x and 3.1x either side of 1.0.
#
# On this path the question does not even arise: **nothing measured about the
# frames may refuse a write.** By the time they exist the user has paid the
# GPU-seconds, and `structure_notes` demotes even the refusal-grade structure
# finding to a note for exactly that reason.
#
# **Saturation is measured and never asserted on.** A deliberately monochrome
# render measures 0.00 and is not a defect; making colour a requirement would
# refuse a legitimate black-and-white shot, which is the false-defect direction
# that has already cost this build days.

#: Below this the 8x8 box means have cancelled out -- there is nothing at the
#: block scale. On its own it is not enough (a legitimately soft, defocused shot
#: could sit low), which is why the refusal needs the noise term too.
MIN_STRUCTURE_RATIO = 0.70

#: Above this the residual after removing the picture is bigger than the picture.
#: Real footage cannot do that; per-pixel noise does it by definition.
MAX_HIGH_FREQUENCY_RATIO = 1.0

#: Absolute, not a ratio: the 8x8 box means barely move at all. WARN only --
#: ``r4a`` in the banked set measures 5.33 and is a real (if murky) render.
LOW_BLOCK_STD = 8.0

#: p99 - p1 of luma. Below this the whole frame lives in a narrow brightness
#: band. WARN only, for the same reason.
LOW_LUMA_BAND = 32.0

#: Frames to measure on the app path. Evenly spaced, median-aggregated; on the
#: banked set the 16-frame estimate tracks the full decode to within 1.6% on
#: every metric, for a twentieth of the work.
STRUCTURE_SAMPLE_FRAMES = 16

#: The side of the box filter. 8 because that is the scale at which "adjacent
#: pixels agree" starts to mean something and still well below any real subject.
_BLOCK = 8

#: A frame whose luma std is below this has no contrast to normalise by, so SR
#: and HF are 0/0. Such frames are counted and EXCLUDED from the ratios rather
#: than folded in as zeros -- "could not measure" must not read as "measured and
#: bad", and the flatness itself is what :data:`LOW_BLOCK_STD` and the runner's
#: own flat-frame check are for.
_MIN_CONTRAST = 1e-3


@dataclass(frozen=True)
class FrameStructure:
    """What the frames contain, as numbers. Medians over the sampled frames.

    ``structure_ratio`` and ``high_frequency_ratio`` are ``None`` when no
    sampled frame had enough contrast (or enough pixels) to measure them. That
    is a third state on purpose: it is neither a pass nor a fail, and every
    caller here treats it as "not checked" rather than "checked and fine".
    """

    frames: int
    sampled: int
    structure_ratio: float | None
    high_frequency_ratio: float | None
    block_std: float | None
    luma_band: float | None
    luma_std: float | None
    saturation: float | None
    flat_frames: int

    def render(self) -> str:
        def number(value, spec=".4f"):
            return "n/a" if value is None else format(value, spec)

        line = (
            f"structure {number(self.structure_ratio)} (>= {MIN_STRUCTURE_RATIO}) · "
            f"high-freq {number(self.high_frequency_ratio)} "
            f"(<= {MAX_HIGH_FREQUENCY_RATIO}) · "
            f"block std {number(self.block_std, '.3f')} · "
            f"luma band {number(self.luma_band, '.2f')} · "
            f"luma std {number(self.luma_std, '.3f')} · "
            f"saturation {number(self.saturation, '.4f')} (reported, never asserted)"
        )
        if self.flat_frames:
            line += f" · {self.flat_frames}/{self.sampled} frame(s) too flat to measure a ratio"
        return f"{line}  [median of {self.sampled} of {self.frames} frames]"


def _luma(frame):
    """0.299R + 0.587G + 0.114B, float32, on one ``[H, W, 3]`` uint8 frame."""
    import numpy as np

    values = np.asarray(frame, dtype=np.float32)
    return values[..., 0] * 0.299 + values[..., 1] * 0.587 + values[..., 2] * 0.114


def _block_means(luma):
    """The 8x8 box means. Ragged right/bottom edges are dropped, not padded.

    Padding would invent pixels and give the edge blocks an artificial variance
    of their own; dropping costs at most seven rows and columns.
    """
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


def _saturation(frame):
    """Mean ``(max - min) / max`` over the pixels. Reported, never asserted on."""
    import numpy as np

    values = np.asarray(frame, dtype=np.float32)
    high = values.max(axis=-1)
    low = values.min(axis=-1)
    return float(np.mean(np.where(high > 0.0, (high - low) / np.maximum(high, 1.0), 0.0)))


def _sampled_indices(count: int, sample):
    """``sample`` evenly-spaced frame indices, endpoints included."""
    import numpy as np

    if sample is None or sample >= count:
        return list(range(count))
    return [int(index) for index in np.unique(np.linspace(0, count - 1, sample).round())]


def measure_structure(frames, *, sample=None) -> FrameStructure:
    """Measure spatial structure over ``frames`` (``[F, H, W, 3]`` uint8 RGB).

    ``sample=None`` measures every frame -- 380 ms for 121 frames of 1216x704,
    which is what the offline runner does. ``sample=16`` is the app path.

    Aggregation is the MEDIAN, not the mean: one legitimately flat frame (a fade
    to black, a cut) must not drag a whole healthy shot toward a warning, and one
    accidentally structured frame must not rescue a stack of static.
    """
    import numpy as np

    count = len(frames)
    if count == 0:
        return FrameStructure(
            frames=0,
            sampled=0,
            structure_ratio=None,
            high_frequency_ratio=None,
            block_std=None,
            luma_band=None,
            luma_std=None,
            saturation=None,
            flat_frames=0,
        )

    indices = _sampled_indices(count, sample)
    ratios: list[float] = []
    high_frequencies: list[float] = []
    block_stds: list[float] = []
    bands: list[float] = []
    luma_stds: list[float] = []
    saturations: list[float] = []
    flat = 0

    for index in indices:
        frame = frames[index]
        luma = _luma(frame)
        luma_std = float(luma.std())
        luma_stds.append(luma_std)
        low, high = (float(value) for value in np.percentile(luma, (1.0, 99.0)))
        bands.append(high - low)
        saturations.append(_saturation(frame))

        blocks = _block_means(luma)
        block_std = None if blocks is None else float(blocks.std())
        if block_std is not None:
            block_stds.append(block_std)

        if luma_std < _MIN_CONTRAST:
            flat += 1
            continue
        if block_std is not None:
            ratios.append(block_std / luma_std)
        laplacian = _laplacian4(luma)
        if laplacian is not None:
            high_frequencies.append(float(laplacian.std()) / luma_std)

    def median(values):
        return float(np.median(values)) if values else None

    return FrameStructure(
        frames=count,
        sampled=len(indices),
        structure_ratio=median(ratios),
        high_frequency_ratio=median(high_frequencies),
        block_std=median(block_stds),
        luma_band=median(bands),
        luma_std=median(luma_stds),
        saturation=median(saturations),
        flat_frames=flat,
    )


def structure_failure(structure: FrameStructure) -> str | None:
    """The one diagnosis strong enough to condemn an artifact, or ``None``.

    **Both** conditions, never either alone. Low SR by itself is a soft or
    defocused shot; high HF by itself is a sharp, grainy one. Only together do
    they say "there is nothing at the block scale AND the pixel-to-pixel
    residual is larger than the signal", which is what structureless colour
    static measures like and what no real render in the banked set comes close
    to.

    Returns a neutral statement of what was measured. The offline runner raises
    on it; the app attaches it to the result as a warning and emits the file
    anyway -- see :func:`structure_notes`.
    """
    ratio = structure.structure_ratio
    high_frequency = structure.high_frequency_ratio
    if ratio is None or high_frequency is None:
        return None
    if ratio >= MIN_STRUCTURE_RATIO or high_frequency <= MAX_HIGH_FREQUENCY_RATIO:
        return None
    return (
        f"structure ratio {ratio:.4f} < {MIN_STRUCTURE_RATIO} AND high-frequency ratio "
        f"{high_frequency:.4f} > {MAX_HIGH_FREQUENCY_RATIO}: the 8x8 box means have cancelled "
        f"out and the pixel-to-pixel residual is larger than the picture. These frames carry no "
        f"spatial structure -- that is colour static, not a scene. Real renders in the banked "
        f"set measure 0.918..0.994 structure and 0.147..0.386 high-frequency"
    )


def structure_warnings(structure: FrameStructure) -> tuple[str, ...]:
    """Things worth a human's attention that must never fail a run.

    Both are absolute measures of how much there is to look at, and a real,
    murky, low-contrast render trips them: the banked ``r4a`` measures 5.33 and
    24.76 here. Refusing on either would destroy a legitimate artifact, so they
    annotate and nothing more.
    """
    notes: list[str] = []
    if structure.block_std is not None and structure.block_std < LOW_BLOCK_STD:
        notes.append(
            f"low detail: 8x8 block std {structure.block_std:.3f} < {LOW_BLOCK_STD}. The frames "
            f"vary very little at the block scale -- worth looking at, not proof of a defect"
        )
    if structure.luma_band is not None and structure.luma_band < LOW_LUMA_BAND:
        notes.append(
            f"low contrast: luma p99-p1 {structure.luma_band:.2f} < {LOW_LUMA_BAND} on a 0..255 "
            f"scale. The whole frame sits in a narrow brightness band -- worth looking at, not "
            f"proof of a defect"
        )
    return tuple(notes)


def structure_notes(structure: FrameStructure) -> tuple[str, ...]:
    """Every finding as a WARNING, refusing nothing. The app's entry point.

    The app must never raise on any of this. By the time these frames exist the
    user has already paid 35+ GPU-seconds for them, and the detector is a
    statistical pre-filter that is wrong about ``r4a``-class output by design.
    A pre-filter may demote an artifact; it may not destroy one. So the file is
    written, the note rides along on the result, and a human decides.
    """
    failure = structure_failure(structure)
    if failure is None:
        return structure_warnings(structure)
    return (
        f"this output may be structureless -- {failure}. The file was written anyway: "
        f"this is a statistical check and it is not a verdict. Please look at it",
        *structure_warnings(structure),
    )


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
    frames,
    audio,
    target: Path,
    *,
    frame_rate,
    sampling_rate,
    include_audio: bool,
    mode: str = "",
    supplied_audio: SuppliedAudio | None = None,
) -> tuple[str, ...]:
    """Encode the frames, muxing the audio track only if the mode exports it.

    Returns any warnings the caller should surface. Audio present with no sample
    rate raises rather than falling back to a video-only write: silently
    dropping the audio track from a *joint* generation is the exact defect this
    module is written against. Dropping it from a **video-only** mode is the
    opposite case -- declared, intended, and reported rather than silent.

    ``supplied_audio`` REPLACES the model's track, waveform and rate together
    (:class:`SuppliedAudio`). It applies only where the container is going to
    carry audio at all -- a video-only mode still drops what the model rendered
    and still says so, because substituting there would put sound into a file
    whose mode declares it has none. The substitution happens before the
    contract check below, so a mode that exports audio is satisfied by the
    supplied track even where the model returned nothing.
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
    if supplied_audio is not None and include_audio:
        # Both values, together, from the same record: see SuppliedAudio for why
        # the rate cannot be left as the vocoder's.
        audio = supplied_audio.waveform
        sampling_rate = supplied_audio.sampling_rate
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

    # Measured HERE because this is the one place the normalised rgb24 stack
    # already exists: no second decode, no second conversion, ~50 ms for 16
    # frames of 1216x704. WARN only, never a refusal -- see `structure_notes`.
    for note in structure_notes(measure_structure(array, sample=STRUCTURE_SAMPLE_FRAMES)):
        notes.append(note)
        _log.warning("%s", note)

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
    supplied_audio: SuppliedAudio | None = None,
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
            notes = _write_video(
                frames,
                audio,
                target,
                frame_rate=frame_rate,
                sampling_rate=sampling_rate,
                include_audio="audio" in exports,
                mode=mode,
                supplied_audio=supplied_audio,
            )
            if warnings is not None:
                warnings.extend(notes)
            # Derived from the same two facts `_write_video` muxes on, NOT from
            # "notes is empty". It used to be the latter, which was correct only
            # while the dropped-audio note was the sole thing that could ever be
            # in there -- the moment a second, unrelated note appeared (the
            # structure warnings), a muxed audio track would have been reported
            # as absent and every audio-exporting mode would have raised
            # EmptyOutputError on a file that was perfectly fine.
            # `supplied_audio` counts here for the same reason the model's track
            # does: it is a track that ends up IN the container. Reading only
            # `audio` would report the audio export as absent on exactly the runs
            # this substitution exists for, and `generate._artifacts_for` would
            # raise EmptyOutputError on a file that carries both streams.
            muxed_audio = "audio" in exports and (audio is not None or supplied_audio is not None)
            written = ("video", "audio") if muxed_audio else ("video",)
        else:
            _write_audio(audio, target, sampling_rate=sampling_rate)
            written = ("audio",)
    except LTXStudioError:
        raise
    except Exception as exc:
        raise EncodeError(path=str(target), reason=str(exc)) from exc
    return dict.fromkeys(written, target)
