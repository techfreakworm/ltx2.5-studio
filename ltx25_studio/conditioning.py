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


def _quarter_turns(frame, stream) -> int:
    """Counter-clockwise quarter turns that make this stream's frames upright.

    The display matrix (``frame.rotation``, counter-clockwise degrees) wins; an
    older file may carry only a ``rotate`` stream tag, which counts clockwise.
    Angles that are not a multiple of 90 are left alone.
    """
    degrees = int(round(getattr(frame, "rotation", 0) or 0))
    if not degrees:
        tag = next((v for k, v in stream.metadata.items() if k.lower() == "rotate"), None)
        try:
            degrees = -int(round(float(tag))) if tag else 0
        except ValueError:
            degrees = 0
    if degrees % 90:
        return 0
    return (degrees // 90) % 4


def _upright(frame, turns: int):
    import numpy as np

    array = frame.to_ndarray(format="rgb24")
    return np.ascontiguousarray(np.rot90(array, k=turns)) if turns else array


def _frame_by_frame(container, stream, *, max_frames):
    frames, turns = [], None
    for frame in container.decode(stream):
        if turns is None:
            turns = _quarter_turns(frame, stream)
        frames.append(_upright(frame, turns))
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def _at_rate(container, stream, *, frame_rate: Fraction, max_frames):
    """The frame nearest each ``t = i / frame_rate``, from the first frame on.

    Ties go to the earlier frame. Targets past the last frame's display time
    are not filled, so a short clip yields fewer frames.
    """
    limit = float("inf") if max_frames is None else max_frames
    nominal = stream.average_rate or stream.guessed_rate or frame_rate
    frames: list = []
    turns = origin = None
    previous = None  # (seconds, frame)
    gap = None
    cached = (None, None)

    def take(frame):
        nonlocal cached
        if cached[0] is not frame:
            cached = (frame, _upright(frame, turns))
        frames.append(cached[1])

    for frame in container.decode(stream):
        if turns is None:
            turns = _quarter_turns(frame, stream)
        if frame.pts is None or frame.time_base is None:
            when = Fraction(0) if previous is None else previous[0] + 1 / Fraction(nominal)
        else:
            origin = frame.pts if origin is None else origin
            when = Fraction(frame.pts - origin) * frame.time_base
        while len(frames) < limit and Fraction(len(frames)) / frame_rate <= when:
            target = Fraction(len(frames)) / frame_rate
            if previous is not None and target - previous[0] <= when - target:
                take(previous[1])
            else:
                take(frame)
        if len(frames) >= limit:
            return frames
        if previous is not None and when > previous[0]:
            gap = when - previous[0]
        previous = (when, frame)

    if previous is not None:
        last_seconds, last = previous
        if last.duration and last.time_base is not None:
            shown_for = Fraction(last.duration) * last.time_base
        else:
            shown_for = gap if gap is not None else 1 / Fraction(nominal)
        while len(frames) < limit and Fraction(len(frames)) / frame_rate < last_seconds + shown_for:
            take(last)
    return frames


def decode_video_frames(source, *, max_frames: int | None = None, frame_rate=None):
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

    **Orientation.** The display rotation a phone stores beside a portrait clip
    is applied, so the frames are upright as a player would show them.

    **No resizing, deliberately.** Upstream scales the condition so the longer
    side fills the target and centre-crops to exactly ``(height, width)``.
    Re-deriving that here would be a second copy of somebody else's arithmetic,
    and validating the source geometry against our own dimension rules would
    reject a 1080p upload that upstream handles perfectly well.

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
    import numpy as np

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

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise ConstraintError(
                    f"{source} carries no video stream, so there is nothing to condition on. "
                    f"Supply a video file rather than an audio-only or empty one."
                )
            stream = container.streams.video[0]
            if rate is None:
                frames = _frame_by_frame(container, stream, max_frames=max_frames)
            else:
                frames = _at_rate(container, stream, frame_rate=rate, max_frames=max_frames)
    except ConstraintError:
        raise
    except Exception as exc:  # noqa: BLE001 -- every decode failure has one answer here
        raise ConstraintError(
            f"{source} could not be decoded as video ({type(exc).__name__}: {exc}). "
            f"A video condition must be decoded before it reaches the pipeline; an "
            f"undecodable one travels on as a path and fails the frames-type check inside "
            f"the GPU worker, by which point the file that caused it is unnamed."
        ) from exc

    if not frames:
        raise ConstraintError(
            f"{source} declares a video stream and decoded to no frames at all. The file is "
            f"truncated or empty -- a header is not evidence that anything is behind it, and "
            f"an empty condition would leave the run looking like the prompt alone."
        )
    if len({frame.shape for frame in frames}) > 1:
        raise ConstraintError(
            f"{source} changes frame size partway through, so its frames cannot be stacked "
            f"into one clip. Re-encode it at a single resolution."
        )
    return np.stack(frames)


def build_conditions(
    conditions: Sequence,
    *,
    num_frames: int,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
    frame_rate=None,
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
    timestamp. Without it the video is read frame by frame.

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
