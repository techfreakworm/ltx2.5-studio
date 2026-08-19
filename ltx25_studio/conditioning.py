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

from .constraints import DEFAULT_TEMPORAL_COMPRESSION_RATIO, image_crf_for
from .errors import ConstraintError

__all__ = [
    "build_conditions",
    "image_crf_for",
    "latent_index_for",
    "normalise_condition",
    "pixel_index_for",
    "validate_condition_index",
]


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


def latent_index_for(
    pixel_frame: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> int:
    """The latent index holding a given pixel frame. Inverse of the above."""
    if pixel_frame <= 0:
        return 0
    return (pixel_frame - 1) // temporal_compression_ratio + 1


def validate_condition_index(pixel_frame: int, *, num_frames: int) -> int:
    """Return the index unchanged, or raise naming the index and the bound."""
    if 0 <= pixel_frame < num_frames:
        return pixel_frame
    raise ConstraintError(
        f"conditioning frame {pixel_frame} is outside this clip, which has {num_frames} frames "
        f"(valid range 0 to {num_frames - 1}). Upstream would skip this condition with only a "
        f"log line and generate anyway, so the image would silently have no effect."
    )


def normalise_condition(condition, *, num_frames: int, temporal_compression_ratio: int) -> dict:
    """One UI condition -> validated fields ready for ``LTX2VideoCondition``.

    Accepts the ``dict`` the UI sends or any object with the same attributes, so
    the UI never has to construct a diffusers type itself.
    """
    if isinstance(condition, dict):
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
    crf: int | None = None,
):
    """Convert UI conditions into ``LTX2VideoCondition`` objects.

    ``diffusers`` is imported here rather than at module scope so the whole
    package stays importable -- and unit-testable -- with no GPU and no weights.

    The returned list is always the same length as the input. If a condition
    cannot be built, this raises; it never returns a quietly shorter list.
    """
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

    built = []
    for condition in conditions:
        fields = normalise_condition(
            condition,
            num_frames=num_frames,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        kwargs = {
            "frames": fields["source"],
            "index": fields["index"],
            "strength": fields["strength"],
        }
        if crf is not None:
            kwargs["crf"] = crf
        built.append(LTX2VideoCondition(**kwargs))
    return built
