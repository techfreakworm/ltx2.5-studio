"""UI inputs -> conditions, and the index conversion nobody sees go wrong.

PLAN §3.1a, §5.7 rows 1, 2 and 8.

**The severe one.** ``LTX2VideoCondition.index`` is a **latent** index, not a
pixel frame number. Upstream computes::

    start_idx = max((latent_start_idx - 1) * temporal_compression_ratio + 1, 0)

Pass a pixel frame where a latent index is expected and the conditioning image
lands eight frames from where the user put it -- guaranteed, silently, every
time. The UI speaks pixel frames because that is what a human means by "frame
40"; the pipeline gets latent indices; this module owns the conversion so
neither side has to remember.

**The other half.** An out-of-range index is *silently skipped* upstream with
only a ``logger.warning``. The user supplies a conditioning image, the image
does nothing at all, and nobody is told. We validate the range ourselves and
raise -- a condition is never quietly dropped, leaving a shorter list than the
user built.

Image and video conditioning are the same mechanism (§3.1a): one list of
conditions, each with a source, an index and a strength.

**The payload, which is where the sameness stops.** Three input kinds reach
conditioning by three different routes, and only two of them arrived decoded.
``gr.Image(type="pil")`` hands over a PIL image and upstream takes it as-is;
``audio.py`` is handed a path and decodes it itself with ``av``; ``gr.Video``
is handed a **path** and nothing anywhere decoded it. So ``modes._conditions``
built a perfectly well-formed ``ConditionSpec`` around a ``str``, and diffusers
refused it at the last possible moment::

    TypeError: Unsupported `frames` type for condition 0: <class 'str'>
    # preprocess_conditions, pipeline_ltx2_condition.py:899 (pinned ac56fa2f)

That is both video-input modes dead and the other nine fine — one missing
decoder, not a conditioning bug. :func:`decode_video_frames` closes it here
rather than in ``modes.py`` for two reasons: ``modes.py`` states as a property
that it imports nothing heavy and stays testable with no media stack present,
and this module is already the one that owns "normalised and range-checked,
before diffusers". The index conversion below and this decode are the same job
on two different fields of the same object.

Integer arithmetic at module level, and the one piece of I/O — decoding a video
condition — imports ``av`` lazily inside the function that needs it, so this
module still imports with no media stack and no GPU present.
"""

from __future__ import annotations

import heapq
import math
import struct
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .constraints import DEFAULT_TEMPORAL_COMPRESSION_RATIO
from .errors import ConstraintError

__all__ = [
    "SHOT_ROW_ANCHOR",
    "ConditionSpec",
    "build_conditions",
    "decode_video_frames",
    "latent_group_for",
    "latent_index_for",
    "latent_to_pixel_index",
    "normalise_condition",
    "pixel_index_for",
    "pixel_to_latent_index",
    "validate_condition_index",
]


#: A shot-list row as ``gr.Dataframe(type="array")`` hands it over:
#: ``(prompt segment, anchor, strength)``. The UI declares the same order in
#: ``ui/fields.SHOT_COLUMNS``; if either moves, the other must move with it, and
#: reading the prompt cell as a frame number is precisely the silent failure the
#: shared declaration exists to stop.
SHOT_ROW_PROMPT = 0
SHOT_ROW_ANCHOR = 1
SHOT_ROW_STRENGTH = 2


@dataclass(frozen=True)
class ConditionSpec:
    """A conditioning input, normalised and range-checked, before diffusers.

    ``index`` is always a **latent** index by the time a spec exists -- the
    conversion happens in :func:`normalise_condition`, so nothing downstream has
    to remember which unit it is holding.
    """

    source: object
    kind: str = "image"
    index: int = 0
    strength: float = 1.0


def pixel_index_for(
    latent_index: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> int:
    """The pixel frame a latent index lands on.

    Quoted from upstream: ``max((latent_start_idx - 1) * ratio + 1, 0)``. Note
    the discontinuity -- latent 0 and 1 map to pixel 0 and 1, and only then does
    the stride become ``ratio``. That kink is where off-by-one bugs live.
    """
    return max((latent_index - 1) * temporal_compression_ratio + 1, 0)


#: Alias with the name the UI layer asks for. Same function, one implementation
#: -- a second copy of ``max((latent - 1) * ratio + 1, 0)`` living in the UI
#: could drift from this one, and a readout that disagrees with the generator is
#: worse than no readout at all.
def latent_to_pixel_index(
    latent_index: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> int:
    """The pixel frame a latent index lands on. See :func:`pixel_index_for`."""
    return pixel_index_for(latent_index, temporal_compression_ratio=temporal_compression_ratio)


def pixel_to_latent_index(
    pixel_frame: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> int:
    """The latent index holding a pixel frame. See :func:`latent_index_for`."""
    return latent_index_for(pixel_frame, temporal_compression_ratio=temporal_compression_ratio)


def latent_index_for(
    pixel_frame: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> int:
    """The latent index holding a given pixel frame. Inverse of the above."""
    if pixel_frame <= 0:
        return 0
    return (pixel_frame - 1) // temporal_compression_ratio + 1


def latent_group_for(
    pixel_frame: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> dict:
    """Which latent a frame lands on, and which frames share it.

    The mapping is **lossy in this direction**: temporal compression is 8:1, so a
    run of frames collapses onto one latent index. A user who types frame 30 is
    anchored at the same place as one who types frame 25, and saying nothing
    about that is the same quiet wrongness as snapping a dimension silently.

    Returns the latent, the span of frames sharing it, and whether the frame the
    user gave is the one that span begins at.

    The arithmetic lives here rather than in the UI for the reason PLAN §5.7 #1
    gives: a second copy would eventually disagree with the real one, and a
    readout that disagrees with what runs is worse than no readout.
    """
    ratio = max(int(temporal_compression_ratio), 1)
    latent = latent_index_for(pixel_frame, temporal_compression_ratio=ratio)
    first = pixel_index_for(latent, temporal_compression_ratio=ratio)
    last = 0 if latent == 0 else ratio * latent
    return {
        "latent": latent,
        "first_frame": first,
        "last_frame": last,
        "exact": pixel_frame == first,
        "shared": last > first,
    }


def validate_condition_index(pixel_frame: int, *, num_frames: int) -> int:
    """Return the index unchanged, or raise naming the index and the bound."""
    if 0 <= pixel_frame < num_frames:
        return pixel_frame
    raise ConstraintError(
        f"conditioning frame {pixel_frame} is outside this clip, which has {num_frames} frames "
        f"(valid range 0 to {num_frames - 1}). Upstream would skip this condition with only a "
        f"log line and generate anyway, so the image would silently have no effect."
    )


def normalise_condition(
    condition,
    *,
    num_frames: int,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> dict:
    """One UI condition -> validated fields ready for ``LTX2VideoCondition``.

    Accepts the ``dict`` the UI sends or any object with the same attributes, so
    the UI never has to construct a diffusers type itself.
    """
    if isinstance(condition, (list, tuple)):
        # A raw shot-list row. Reaching the pipeline unconverted is the bug this
        # function exists to prevent, so it is handled here rather than refused.
        cells = list(condition)

        def cell(position, default=None):
            value = cells[position] if len(cells) > position else None
            return default if value is None else value

        source = cell(SHOT_ROW_PROMPT)
        pixel_frame = int(cell(SHOT_ROW_ANCHOR, 0) or 0)
        strength = float(cell(SHOT_ROW_STRENGTH, 1.0))
    elif isinstance(condition, dict):
        source = condition.get("source", condition.get("frames"))
        pixel_frame = int(condition.get("index", 0) or 0)
        strength = float(condition.get("strength", 1.0))
    else:
        source = getattr(condition, "source", getattr(condition, "frames", None))
        pixel_frame = int(getattr(condition, "index", 0) or 0)
        strength = float(getattr(condition, "strength", 1.0))

    if source is None:
        raise ConstraintError(
            "a conditioning entry has no image or video attached. An empty condition would be "
            "dropped silently and the result would look like the prompt alone."
        )
    validate_condition_index(pixel_frame, num_frames=num_frames)
    return {
        "source": source,
        "index": latent_index_for(
            pixel_frame, temporal_compression_ratio=temporal_compression_ratio
        ),
        "pixel_index": pixel_frame,
        "strength": strength,
    }


def _already_decoded(source) -> bool:
    """True if ``source`` is already a payload ``preprocess_conditions`` accepts.

    Duck-typed rather than ``isinstance``-checked, because naming the accepted
    types means importing them: ``torch`` and ``PIL`` both, on a path that runs
    with no model stack present. The four accepted shapes are a PIL image, a
    list of them, a numpy array and a tensor -- the last two are the ones with
    ``ndim``, the first two are the ones that can ``convert``.

    A payload that is already frames is left alone rather than re-decoded. A
    spec can legitimately be rebuilt -- ``pipelines._with_real_conditions``
    re-normalises anything reaching it that is not already a spec -- and a
    second decode would be handed an array where a path is expected.
    """
    if isinstance(source, (str, Path)):
        return False
    if hasattr(source, "ndim") or hasattr(source, "convert"):
        return True
    if isinstance(source, (list, tuple)) and source:
        return all(hasattr(frame, "convert") for frame in source)
    return False


def _display_matrix(frame):
    """The frame's display matrix (nine int32, FFmpeg's layout), or ``None``."""
    for side_data in getattr(frame, "side_data", None) or ():
        if "DISPLAYMATRIX" in str(getattr(side_data, "type", "")).upper():
            raw = bytes(side_data)
            if len(raw) >= 36:
                return struct.unpack("=9i", raw[:36])
    return None


def _matrix_orientation(matrix) -> tuple[int, bool]:
    """``(turns, mirror)`` for a display matrix, decided as FFmpeg's autorotate does.

    FFmpeg reads the clockwise angle (``get_rotation``), then picks transpose,
    flips or nothing from the signs of the matrix entries; each of its choices
    is written here as counter-clockwise quarter turns followed by an optional
    left-right mirror.
    """
    a, b, c, d = matrix[0], matrix[1], matrix[3], matrix[4]
    scale_x, scale_y = math.hypot(a, c), math.hypot(b, d)
    if not scale_x or not scale_y:
        return 0, False
    theta = -round(-math.degrees(math.atan2(b / scale_y, a / scale_x)))
    theta -= 360 * math.floor(theta / 360 + 0.9 / 360)
    if abs(theta - 90) < 1:
        return (3, True) if c > 0 else (3, False)  # transpose, or turn clockwise
    if abs(theta - 180) < 1:
        hflip, vflip = a < 0, d < 0  # FFmpeg applies each one it finds
        if hflip and vflip:
            return 2, False
        if hflip:
            return 0, True
        return (2, True) if vflip else (0, False)
    if abs(theta - 270) < 1:
        return (1, True) if c < 0 else (1, False)  # anti-transpose, or turn anticlockwise
    if abs(theta) < 1:
        return (2, True) if d < 0 else (0, False)  # mirror upside down, or nothing
    return 0, False


def _orientation(frame, stream) -> tuple[int, bool]:
    """``(counter-clockwise quarter turns, then mirror left-right)`` that show this
    stream's frames as a player does.

    The display matrix wins, mirroring included; an older file may carry only
    a ``rotate`` stream tag, which counts clockwise. Angles that are not a
    multiple of 90 are left alone.
    """
    matrix = _display_matrix(frame)
    if matrix is not None:
        return _matrix_orientation(matrix)
    degrees = int(round(getattr(frame, "rotation", 0) or 0))
    if not degrees:
        tag = next((v for k, v in stream.metadata.items() if k.lower() == "rotate"), None)
        try:
            degrees = -int(round(float(tag))) if tag else 0
        except ValueError:
            degrees = 0
    if degrees % 90:
        return 0, False
    return (degrees // 90) % 4, False


#: FFmpeg's numbers for the HDR transfers phones and cameras tag video with:
#: AVCOL_TRC_SMPTE2084 (PQ, HDR10) and AVCOL_TRC_ARIB_STD_B67 (HLG).
_HDR_TRANSFERS = {16: "PQ", 18: "HLG"}
#: AVCOL_PRI_BT2020, the wide-colour primaries HDR video uses.
_BT2020_PRIMARIES = 9


def _colour(frame) -> tuple[dict, str | None]:
    """``to_ndarray`` arguments that bring an HDR or BT.2020 frame to SDR BT.709, and
    what the clip is; ``({}, None)`` for an SDR clip, which is read unchanged.

    Without destination values PyAV's reformat drops the source transfer and
    primaries, so HLG reference white reads as (189, 189, 189). FFmpeg's
    swscale converts both when it is asked to.
    """
    transfer = _HDR_TRANSFERS.get(int(getattr(frame, "color_trc", 0) or 0))
    wide = int(getattr(frame, "color_primaries", 0) or 0) == _BT2020_PRIMARIES
    if transfer is None and not wide:
        return {}, None
    from av.video.reformatter import ColorPrimaries, ColorTrc

    kind = f"HDR ({transfer})" if transfer else "wide-colour (BT.2020)"
    return {"dst_color_trc": ColorTrc.BT709, "dst_color_primaries": ColorPrimaries.BT709}, kind


def _colour_note(source, kind: str) -> str:
    return (
        f"Your video {Path(str(source)).name} is {kind}. It was converted to standard (SDR) "
        f"colour for the model, so as a reference its colours may look flat or dark. Convert "
        f"it to SDR for a closer match."
    )


def _upright(frame, orientation: tuple[int, bool], colour: dict):
    import numpy as np

    array = frame.to_ndarray(format="rgb24", **colour)
    turns, mirror = orientation
    if not (turns or mirror):
        return array
    if turns:
        array = np.rot90(array, k=turns)
    return np.ascontiguousarray(array[:, ::-1] if mirror else array)


def _upstream_geometry(source_height: int, source_width: int, height: int, width: int):
    """``(scale, new_h, new_w, top, left)`` of upstream's resize to ``(height, width)``.

    Quoted from ``preprocess_conditions`` at the pinned diffusers: scale so the
    longer side fills, ``ceil`` each side, then centre-crop.
    """
    scale = max(height / source_height, width / source_width)
    new_h, new_w = math.ceil(source_height * scale), math.ceil(source_width * scale)
    return scale, new_h, new_w, (new_h - height) // 2, (new_w - width) // 2


def _shrink_with_torch(array, geometry, size):
    """Upstream's own resize: ``F.interpolate``, bilinear, ``align_corners=False``."""
    import torch

    _scale, new_h, new_w, top, left = geometry
    height, width = size
    pixels = torch.from_numpy(array).permute(2, 0, 1)[None].to(torch.float32)
    pixels = torch.nn.functional.interpolate(
        pixels, size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0, :, top : top + height, left : left + width]
    return pixels.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


def _sample_points(count_in: int, count_out: int, first: int, count: int):
    """Where torch's bilinear samples one axis, for outputs ``first`` to ``first + count``."""
    import numpy as np

    x = (np.arange(first, first + count, dtype=np.float64) + 0.5) * (count_in / count_out) - 0.5
    x = np.maximum(x, 0.0)
    low = np.minimum(np.floor(x).astype(np.int64), count_in - 1)
    return low, np.minimum(low + 1, count_in - 1), (x - low).astype(np.float32)


def _shrink_with_numpy(array, geometry, size):
    """The same arithmetic as :func:`_shrink_with_torch`, where torch is not installed."""
    import numpy as np

    _scale, new_h, new_w, top, left = geometry
    height, width = size
    up, down, row_weight = _sample_points(array.shape[0], new_h, top, height)
    left_of, right_of, column_weight = _sample_points(array.shape[1], new_w, left, width)
    row_weight, column_weight = row_weight[:, None, None], column_weight[None, :, None]
    rows = array[up].astype(np.float32) * (1 - row_weight)
    rows += array[down].astype(np.float32) * row_weight
    out = rows[:, left_of] * (1 - column_weight) + rows[:, right_of] * column_weight
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class _Pictures:
    """Decoded frames -> RGB arrays as a player shows them, at most ``size``.

    Orientation and colour are properties of the stream, read off the first
    frame converted; ``kind`` is set when the clip is HDR or wide-colour. With
    ``size``, a picture upstream would shrink is shrunk here, the moment it is
    decoded.
    """

    def __init__(self, stream, size=None):
        self.stream = stream
        self.size = size
        self.orientation = None
        self.colour: dict = {}
        self.kind: str | None = None
        self.shrink = None
        if size is not None:
            try:
                import torch  # noqa: F401 -- the pipeline's own resize, where it exists
            except ImportError:
                self.shrink = _shrink_with_numpy
            else:
                self.shrink = _shrink_with_torch

    def __call__(self, frame):
        if self.orientation is None:
            self.orientation = _orientation(frame, self.stream)
            self.colour, self.kind = _colour(frame)
        array = _upright(frame, self.orientation, self.colour)
        if self.shrink is None:
            return array
        geometry = _upstream_geometry(*array.shape[:2], *self.size)
        return array if geometry[0] >= 1 else self.shrink(array, geometry, self.size)


class _Stack:
    """Frames written into one array as they arrive, so the result is the only copy:
    no list of frames, and no ``np.stack`` copying it again."""

    def __init__(self, capacity, source):
        self.capacity, self.source = capacity, source
        self.array, self.count = None, 0

    def __len__(self):
        return self.count

    def append(self, frame):
        import numpy as np

        if self.array is None:
            self.array = np.empty((self.capacity or 64, *frame.shape), dtype=frame.dtype)
        elif frame.shape != self.array.shape[1:]:
            raise ConstraintError(
                f"{self.source} changes frame size partway through, so its frames cannot be "
                f"stacked into one clip. Re-encode it at a single resolution."
            )
        if self.count == len(self.array):
            grown = np.empty((2 * self.count, *frame.shape), dtype=frame.dtype)
            grown[: self.count] = self.array
            self.array = grown
        self.array[self.count] = frame
        self.count += 1

    def result(self):
        return self.array[: self.count]


def _frame_by_frame(container, stream, picture, frames, *, max_frames):
    for frame in container.decode(stream):
        frames.append(picture(frame))
        if max_frames is not None and len(frames) >= max_frames:
            break


#: The deepest picture reordering H.264 and HEVC allow. A frame's display time
#: is settled once this many later frames have been decoded.
REORDER_DEPTH = 16


def _duration(frame):
    """How long ``frame`` is shown, in seconds, or ``None`` if it does not say."""
    if frame.duration and frame.time_base is not None:
        return Fraction(frame.duration) * frame.time_base
    return None


def _display_times(frames, *, nominal: Fraction):
    """``(seconds from the first frame, frame)`` for decoded frames, in display order.

    A decoder hands frames back in display order, so the k-th frame shows at
    the k-th smallest timestamp. Most containers stamp each frame with its own
    display time and the sort changes nothing; H.264 with B-frames in AVI
    stamps display-order frames with decode-order times, and the sort over
    :data:`REORDER_DEPTH` frames restores them. A frame with no stamp at all is
    shown one frame duration after the one before (``nominal`` only if it has
    no duration). Times never go backwards.
    """
    pending: deque = deque()
    stamps: list = []
    origin = shown = None
    last = (None, None)  # (seconds, frame) of the latest frame placed

    def release():
        nonlocal origin, shown
        seconds = heapq.heappop(stamps)
        origin = seconds if origin is None else origin
        shown = seconds - origin if shown is None else max(shown, seconds - origin)
        return shown, pending.popleft()

    for frame in frames:
        stamp = frame.pts if frame.pts is not None else frame.dts
        if stamp is not None and frame.time_base is not None:
            seconds = Fraction(stamp) * frame.time_base
        elif last[0] is None:
            seconds = Fraction(0)
        else:
            seconds = last[0] + (_duration(last[1]) or _duration(frame) or 1 / nominal)
        last = (seconds, frame)
        heapq.heappush(stamps, seconds)
        pending.append(frame)
        if len(pending) > REORDER_DEPTH:
            yield release()
    while pending:
        yield release()


def _at_rate(container, stream, picture, frames, *, frame_rate: Fraction, max_frames):
    """The frame nearest each ``t = i / frame_rate``, from the first frame on.

    Ties go to the earlier frame. Targets past the last frame's display time
    are not filled, so a short clip yields fewer frames. Only frames that are
    taken are converted.
    """
    limit = float("inf") if max_frames is None else max_frames
    nominal = Fraction(stream.guessed_rate or stream.average_rate or frame_rate)
    previous = None  # (seconds, frame)
    gap = None
    cached = (None, None)

    def take(frame):
        nonlocal cached
        if cached[0] is not frame:
            cached = (frame, picture(frame))
        frames.append(cached[1])

    for when, frame in _display_times(container.decode(stream), nominal=nominal):
        while len(frames) < limit and Fraction(len(frames)) / frame_rate <= when:
            target = Fraction(len(frames)) / frame_rate
            if previous is not None and target - previous[0] <= when - target:
                take(previous[1])
            else:
                take(frame)
        if len(frames) >= limit:
            return
        if previous is not None and when > previous[0]:
            gap = when - previous[0]
        previous = (when, frame)

    if previous is not None:
        last_seconds, last = previous
        shown_for = _duration(last) or gap or 1 / nominal
        while len(frames) < limit and Fraction(len(frames)) / frame_rate < last_seconds + shown_for:
            take(last)


def decode_video_frames(
    source,
    *,
    max_frames: int | None = None,
    frame_rate=None,
    notes: list | None = None,
    size=None,
):
    """A video path -> the RGB frames the pipeline will actually take.

    Returns ``(F, H, W, 3)`` ``uint8``, which is one of the four payload types
    ``preprocess_conditions`` dispatches on. Three things decided that shape:

    * ``av``'s ``to_ndarray(format="rgb24")`` produces it directly, so there is
      no conversion to get wrong. It is also the representation upstream builds
      anyway -- the PIL-list branch is a ``np.stack`` of exactly this.
    * ``uint8`` is **load-bearing**, not incidental. Upstream re-compresses a
      single-frame condition at the model CRF and raises ``ValueError`` if the
      array is not ``uint8`` while doing it. A float payload decodes fine here
      and fails there, on the metered side.
    * ``av`` rather than torchvision or imageio, for the reason ``audio.py``
      already gives at length: it is a required dependency that bundles its own
      FFmpeg, so it works where the system codecs are not ours to install.

    **Timing.** With ``frame_rate`` (the generation's rate), frames are chosen by
    presentation timestamp: frame ``i`` is the source frame nearest to
    ``i / frame_rate`` seconds after the first one. A 60 fps clip therefore
    conditions a 24 fps run at real speed rather than at 0.4x, and a
    variable-frame-rate clip keeps its timing. Without it, frames are read one
    by one in order, which is only right when the clip's rate already matches.

    **Orientation.** The display matrix a phone or an editor stores beside a
    clip is applied, rotation and mirroring both, so the frames are what a
    player shows.

    **Colour.** An HDR clip (PQ or HLG) or a BT.2020 one is converted to SDR
    BT.709 by FFmpeg's swscale rather than read as if it were SDR; with
    ``notes`` (a list), one sentence for the user is appended saying so.

    **Size.** ``size`` is ``(height, width)``: the largest size the pipeline
    will resize this condition to (for a two-stage run, the stage-2 size). Each
    frame larger than that is reduced the moment it is decoded, with upstream's
    own arithmetic (scale so the longer side fills, ``ceil`` each side,
    bilinear with no antialias, centre-crop), so upstream's resize of it is an
    exact identity and only the rounding to whole code values differs, by at
    most half of one. A clip is never held at its own size: a 4K frame is
    25 MB, 12 GB for 481 of them, before upstream makes a float32 copy on the
    device. A picture smaller than ``size`` is left for upstream to enlarge,
    and without ``size`` nothing is resized. Frames go straight into one
    array, never a list stacked afterwards.

    ``max_frames`` bounds the decode, and it is not an optimisation.
    ``trim_conditioning_sequence`` keeps ``min(F, target - start)`` frames and
    takes the **first** of them, so everything past the target is decoded and
    thrown away. That is not a rounding error on a real upload: a ten-minute
    1080p clip is ~15,000 frames, and ``15_000 x 1920 x 1080 x 3`` bytes is
    **~90 GB** of anonymous memory. Under ``auto_duration`` the real target is
    chosen by the duration head after this runs, so the cap is the request's own
    ``num_frames``; a head that picks a longer clip would see a condition
    trimmed shorter than upstream would have trimmed it.

    Raises :class:`ConstraintError` naming the file for anything undecodable.
    The alternative is a bare ``av`` error, or worse a path travelling on to
    become ``Unsupported `frames` type`` inside the forked worker -- where the
    reason we attach is stripped on the way out and the file that caused it is
    no longer named anywhere.
    """
    import av

    rate = None
    if frame_rate is not None:
        try:
            rate = Fraction(str(frame_rate))
        except (TypeError, ValueError):
            rate = None
        if rate is None or rate <= 0:
            raise ConstraintError(
                f"the frame rate {frame_rate!r} is not a positive number, so the video "
                f"condition cannot be matched to the generation's timing."
            )

    if size is not None:
        size = tuple(int(side) for side in size)
        if len(size) != 2 or min(size) <= 0:
            raise ConstraintError(
                f"the condition size {size!r} is not a (height, width) of positive whole numbers."
            )

    frames = _Stack(max_frames, source)
    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise ConstraintError(
                    f"{source} carries no video stream, so there is nothing to condition on. "
                    f"Supply a video file rather than an audio-only or empty one."
                )
            stream = container.streams.video[0]
            picture = _Pictures(stream, size)
            if rate is None:
                _frame_by_frame(container, stream, picture, frames, max_frames=max_frames)
            else:
                _at_rate(container, stream, picture, frames, frame_rate=rate, max_frames=max_frames)
    except ConstraintError:
        raise
    except Exception as exc:  # noqa: BLE001 -- every decode failure has one answer here
        raise ConstraintError(
            f"{source} could not be decoded as video ({type(exc).__name__}: {exc}). "
            f"A video condition must be decoded before it reaches the pipeline; an "
            f"undecodable one travels on as a path and fails the frames-type check inside "
            f"the GPU worker, by which point the file that caused it is unnamed."
        ) from exc

    if not len(frames):
        raise ConstraintError(
            f"{source} declares a video stream and decoded to no frames at all. The file is "
            f"truncated or empty -- a header is not evidence that anything is behind it, and "
            f"an empty condition would leave the run looking like the prompt alone."
        )
    if picture.kind is not None and notes is not None:
        notes.append(_colour_note(source, picture.kind))
    return frames.result()


def build_conditions(
    conditions: Sequence,
    *,
    num_frames: int,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
    frame_rate=None,
    notes: list | None = None,
    size=None,
) -> tuple[ConditionSpec, ...]:
    """Normalise every condition into a range-checked :class:`ConditionSpec`.

    No diffusers, so it runs on every path including the ones with no model
    present. That matters: this is where an out-of-range anchor is caught, and
    upstream would skip it with only a log line and generate anyway.

    A **video** condition is also decoded here, which is the one step that is
    not arithmetic -- see :func:`decode_video_frames` for why it belongs at this
    seam and nowhere later. Everything else about the condition is left exactly
    as it arrived: only the payload type was ever wrong, and a decode that
    quietly reset an anchor or a strength would trade one silent defect for
    another.

    ``frame_rate`` is the generation's rate; a video condition is read at it by
    timestamp. Without it the video is read frame by frame. ``notes`` (a list)
    collects what the decode tells the user, such as an HDR clip's conversion.
    ``size`` is the largest ``(height, width)`` the pipeline resizes a video
    condition to; see :func:`decode_video_frames`.

    The returned list is always the same length as the input. A condition that
    cannot be built raises; it never comes back quietly shorter, because a shot
    that vanishes between the form and the model is the quietest failure there
    is.
    """
    built = []
    for condition in conditions:
        fields = normalise_condition(
            condition,
            num_frames=num_frames,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        kind = getattr(condition, "kind", "image")
        source = fields["source"]
        if kind == "video" and not _already_decoded(source):
            source = decode_video_frames(
                source,
                max_frames=num_frames if num_frames and num_frames > 0 else None,
                frame_rate=frame_rate,
                notes=notes,
                size=size,
            )
        built.append(
            ConditionSpec(
                source=source,
                kind=kind,
                index=fields["index"],
                strength=fields["strength"],
            )
        )
    return tuple(built)
