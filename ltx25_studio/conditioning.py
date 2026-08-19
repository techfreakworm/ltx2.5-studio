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

Pure integer arithmetic at module level. ``diffusers`` is imported lazily inside
:func:`build_conditions` so this module stays importable with no GPU present.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .constraints import DEFAULT_TEMPORAL_COMPRESSION_RATIO, image_crf_for
from .errors import ConstraintError

__all__ = [
    "SHOT_ROW_ANCHOR",
    "ConditionSpec",
    "build_conditions",
    "image_crf_for",
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


def build_conditions(
    conditions: Sequence,
    *,
    num_frames: int,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> tuple[ConditionSpec, ...]:
    """Normalise every condition into a range-checked :class:`ConditionSpec`.

    Pure arithmetic -- no diffusers, so it runs on every path including the ones
    with no model present. That matters: this is where an out-of-range anchor is
    caught, and upstream would skip it with only a log line and generate anyway.

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
        built.append(
            ConditionSpec(
                source=fields["source"],
                kind=getattr(condition, "kind", "image"),
                index=fields["index"],
                strength=fields["strength"],
            )
        )
    return tuple(built)
