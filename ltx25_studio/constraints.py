"""Dimension and frame-count validation, CRF selection, sigma schedules.

PLAN §5.5, §10 OQ-3. Pure functions over numbers -- no torch, no weights, no
I/O -- which is what makes them testable with zero GPU (§9.1).

**Why this module exists.** Upstream does not raise for a bad ``num_frames``.
It computes ``latent_num_frames = (num_frames - 1) // ratio + 1``, which
silently floors. At ratio 8, every value from 113 to 120 produces the *same*
latent count: a user who asks for 120 frames gets 113 and is told nothing. That
is the silent-wrongness path we refuse to inherit.

**The modulus is not the literal 8.** It is ``vae.temporal_compression_ratio``,
passed in as ``temporal_compression_ratio=``. The default exists only so the UI
can validate input before any VAE has been loaded; ``pipelines.py`` asserts the
loaded VAE agrees with that default and raises loudly if upstream ever changes
it (PLAN §5.5). Hardcoding 8 here would violate §5.4's own principle.

Division of labour, held to strictly: ``validate_*`` raises, ``snap_*`` fixes.
Never both, never neither.
"""

from __future__ import annotations

from .errors import AudioVideoDesyncError, ConstraintError, DimensionError, FrameCountError

__all__ = [
    "DEFAULT_FRAME_RATE",
    "DEFAULT_IMAGE_CRF",
    "DEFAULT_LENGTH_TOLERANCE_SECONDS",
    "DEFAULT_SPATIAL_COMPRESSION_RATIO",
    "DEFAULT_TEMPORAL_COMPRESSION_RATIO",
    "DIMENSION_MULTIPLE",
    "DISTILLED_SIGMA_VALUES",
    "LTX2_5_IMAGE_CRF",
    "STAGE_2_DISTILLED_SIGMA_VALUES",
    "LTX2_5_TEXT_ENCODER_TYPES",
    "image_crf_for",
    "is_valid_num_frames",
    "nearest_valid_num_frames",
    "snap_dimension",
    "snap_dimensions",
    "snap_num_frames",
    "validate_audio_video_agreement",
    "validate_dimension",
    "validate_dimensions",
    "validate_num_frames",
]

#: Fallback for ``vae.temporal_compression_ratio``, used only before a VAE is
#: loaded. NOT a constant of the model -- see the module docstring.
DEFAULT_TEMPORAL_COMPRESSION_RATIO = 8

#: Fallback for ``vae.spatial_compression_ratio``, same status as the temporal
#: one above: a pre-load default, not a constant of the model. Verified against
#: the real ``vae/config.json`` (``spatial_compression_ratio: 32``).
DEFAULT_SPATIAL_COMPRESSION_RATIO = 32

#: Every spatial dimension must be a multiple of this. Enforced upstream too
#: (all generation pipelines and the latent-upsample pipeline raise on it).
#: Pass ``multiple=vae.spatial_compression_ratio`` once a VAE is loaded.
DIMENSION_MULTIPLE = DEFAULT_SPATIAL_COMPRESSION_RATIO

#: Frames per second, used before any pipeline exists. Upstream's own
#: ``__call__`` default -- verified at ``pipeline_ltx2_condition.py:1076``,
#: ``frame_rate: float = 24.0``, and the same in the base pipeline. Same status
#: as the two ratios above: a pre-load default, not a constant of the model.
#:
#: **This number has two consumers and they must never disagree.** It is passed
#: to the pipeline, which generates at that rate, and to ``outputs.write_output``,
#: which declares it in the container. If one of them takes this value and the
#: other falls back to upstream's own default, the two agree only by coincidence
#: -- and the day either changes, the video plays at the wrong speed while every
#: layer reports success. That is why the encoder holds no default of its own and
#: raises instead: there is one number, it lives here, and it flows outward
#: (PLAN §5.4).
DEFAULT_FRAME_RATE = 24.0

#: Image-conditioning CRF for LTX-2.5, selected when
#: ``text_encoder.config.model_type`` is a gemma4 variant.
LTX2_5_IMAGE_CRF = 18

#: The generic default, correct for earlier LTX generations and wrong for 2.5.
DEFAULT_IMAGE_CRF = 33

#: Distilled 8-step schedule, CFG=1. Mirrored from
#: ``diffusers.pipelines.ltx2.utils`` so an upstream change fails a test rather
#: than silently changing output quality (PLAN §5.5).
DISTILLED_SIGMA_VALUES = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)

#: Stage-2 refinement schedule (3 steps).
STAGE_2_DISTILLED_SIGMA_VALUES = (0.909375, 0.725, 0.421875)

#: Slack allowed between the audio and video track lengths. Frame counts are
#: derived independently, so exact float equality is not a reasonable bar --
#: but the tolerance must be small and explicit rather than implied.
DEFAULT_LENGTH_TOLERANCE_SECONDS = 0.05

#: The shortest legal clip: one frame, at any compression ratio.
MIN_NUM_FRAMES = 1


#: Text-encoder ``model_type`` values that identify an LTX-2.5 stack.
LTX2_5_TEXT_ENCODER_TYPES = ("gemma4_unified", "gemma4")


def image_crf_for(model_type: str) -> int:
    """CRF for image-conditioning re-compression, derived not hardcoded.

    Upstream selects 18 when ``text_encoder.config.model_type`` is a gemma4
    variant and 33 otherwise. Writing either number down makes the code wrong for
    the other, so it is read from the loaded text encoder at runtime (PLAN §5.4,
    §5.7 row 8).
    """
    return LTX2_5_IMAGE_CRF if model_type in LTX2_5_TEXT_ENCODER_TYPES else DEFAULT_IMAGE_CRF


def _check_ratio(temporal_compression_ratio: int) -> int:
    """Reject a nonsense ratio with our error, never a bare ZeroDivisionError."""
    if temporal_compression_ratio < 1:
        raise ConstraintError(
            f"temporal_compression_ratio must be >= 1, got {temporal_compression_ratio}. "
            f"It comes from vae.temporal_compression_ratio and should never be less than 1."
        )
    return temporal_compression_ratio


# ---------------------------------------------------------------------------
# num_frames
# ---------------------------------------------------------------------------


def is_valid_num_frames(
    num_frames: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> bool:
    """True iff ``num_frames`` is ``k * ratio + 1`` for some ``k >= 0``."""
    _check_ratio(temporal_compression_ratio)
    if num_frames < MIN_NUM_FRAMES:
        return False
    return (num_frames - 1) % temporal_compression_ratio == 0


def nearest_valid_num_frames(
    num_frames: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> tuple[int, int]:
    """The legal frame counts bracketing ``num_frames``, as ``(lower, upper)``.

    A legal value brackets itself. Anything below the minimum clamps to
    ``(1, 1)`` -- there is nothing below one frame to offer.
    """
    ratio = _check_ratio(temporal_compression_ratio)
    if num_frames < MIN_NUM_FRAMES:
        return (MIN_NUM_FRAMES, MIN_NUM_FRAMES)
    if is_valid_num_frames(num_frames, temporal_compression_ratio=ratio):
        return (num_frames, num_frames)
    steps = (num_frames - 1) // ratio
    return (steps * ratio + 1, (steps + 1) * ratio + 1)


def snap_num_frames(
    num_frames: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> int:
    """The nearest legal frame count. Ties resolve upward, never toward zero."""
    lower, upper = nearest_valid_num_frames(
        num_frames, temporal_compression_ratio=temporal_compression_ratio
    )
    if lower == upper:
        return lower
    return lower if (num_frames - lower) < (upper - num_frames) else upper


def validate_num_frames(
    num_frames: int,
    *,
    temporal_compression_ratio: int = DEFAULT_TEMPORAL_COMPRESSION_RATIO,
) -> int:
    """Return ``num_frames`` unchanged, or raise :class:`FrameCountError`.

    This is the check upstream does not do. See the module docstring.
    """
    ratio = _check_ratio(temporal_compression_ratio)
    if is_valid_num_frames(num_frames, temporal_compression_ratio=ratio):
        return num_frames
    raise FrameCountError(
        num_frames=num_frames,
        temporal_compression_ratio=ratio,
        nearest=nearest_valid_num_frames(num_frames, temporal_compression_ratio=ratio),
    )


# ---------------------------------------------------------------------------
# width / height
# ---------------------------------------------------------------------------


def validate_dimension(value: int, *, name: str, multiple: int = DIMENSION_MULTIPLE) -> int:
    """Return ``value`` unchanged, or raise :class:`DimensionError`.

    Zero and negatives are rejected even though they satisfy the modulus: a
    dimension below one tile is not a dimension.
    """
    if value >= multiple and value % multiple == 0:
        return value
    raise DimensionError(name=name, value=value, multiple=multiple)


def validate_dimensions(
    width: int, height: int, *, multiple: int = DIMENSION_MULTIPLE
) -> tuple[int, int]:
    """Validate both axes, naming whichever one is wrong."""
    return (
        validate_dimension(width, name="width", multiple=multiple),
        validate_dimension(height, name="height", multiple=multiple),
    )


def snap_dimension(value: int, *, multiple: int = DIMENSION_MULTIPLE) -> int:
    """The nearest legal dimension. Ties resolve upward; anything below the
    minimum clamps up to ``multiple`` rather than collapsing to zero."""
    if value < multiple:
        return multiple
    lower = (value // multiple) * multiple
    upper = lower + multiple
    return lower if (value - lower) < (upper - value) else upper


def snap_dimensions(
    width: int, height: int, *, multiple: int = DIMENSION_MULTIPLE
) -> tuple[int, int]:
    """Snap both axes."""
    return (
        snap_dimension(width, multiple=multiple),
        snap_dimension(height, multiple=multiple),
    )


# ---------------------------------------------------------------------------
# audio / video agreement
# ---------------------------------------------------------------------------


def validate_audio_video_agreement(
    *,
    video_seconds: float,
    audio_seconds: float,
    tolerance_seconds: float = DEFAULT_LENGTH_TOLERANCE_SECONDS,
) -> None:
    """Raise :class:`AudioVideoDesyncError` if the two tracks disagree.

    The tracks' lengths are derived independently, and supplying pre-encoded
    ``audio_latents`` makes the audio length come from the latent shape while
    ``num_frames`` still governs the video (PLAN §5.7 #11). Emitting a desynced
    file is worse than refusing to emit one.
    """
    if abs(video_seconds - audio_seconds) <= tolerance_seconds:
        return None
    raise AudioVideoDesyncError(video_seconds=video_seconds, audio_seconds=audio_seconds)
