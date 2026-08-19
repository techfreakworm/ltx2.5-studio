"""Spec for ``ltx25_studio/conditioning.py`` — PLAN §3.1a, §5.7 rows 1, 2 and 8.

§5.7 row 1 is the severe one:

    ``LTX2VideoCondition.index`` is a **LATENT** index, not a pixel frame index.
    Upstream computes ``start_idx = max((latent_start_idx - 1) * ratio + 1, 0)``.

Pass a pixel frame number where a latent index is expected and you get an
**off-by-8 guaranteed** — the conditioning image lands eight frames from where
the user put it, and nothing anywhere says so. The UI speaks pixel frames; the
pipeline gets latent indices; ``conditioning.py`` owns the conversion.

Row 2 is the other half: an out-of-range index is **silently skipped** upstream
with only a ``logger.warning``, so a whole conditioning image does nothing at
all. We validate the range ourselves and raise.

All pure integer arithmetic — no torch, no PIL, no weights.
"""

from __future__ import annotations

import pytest

from ltx25_studio import conditioning
from ltx25_studio.errors import ConstraintError, LTXStudioError

#: ``(latent_index, pixel_frame)`` at ratio 8, from upstream's own formula.
BOUNDARY_PAIRS = [(0, 0), (1, 1), (2, 9), (3, 17), (4, 25), (16, 121)]


# --------------------------------------------------------------------------
# latent <-> pixel (PLAN §5.7 row 1)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("latent_index,pixel_frame", BOUNDARY_PAIRS)
def test_pixel_index_matches_the_upstream_formula(latent_index, pixel_frame):
    """``max((latent_start_idx - 1) * ratio + 1, 0)``, quoted from the source.

    If upstream changes this, a test should fail here rather than the
    conditioning quietly landing in the wrong place.
    """
    assert conditioning.pixel_index_for(latent_index) == pixel_frame


@pytest.mark.parametrize("latent_index,pixel_frame", BOUNDARY_PAIRS)
def test_latent_index_is_the_inverse(latent_index, pixel_frame):
    assert conditioning.latent_index_for(pixel_frame) == latent_index


@pytest.mark.parametrize("latent_index,pixel_frame", BOUNDARY_PAIRS)
def test_the_conversion_round_trips(latent_index, pixel_frame):
    assert conditioning.pixel_index_for(conditioning.latent_index_for(pixel_frame)) == pixel_frame


def test_frame_zero_stays_zero():
    """``index=0`` is the default and by far the most common case — every
    image-to-video call uses it. An off-by-one here breaks the headline path."""
    assert conditioning.latent_index_for(0) == 0
    assert conditioning.pixel_index_for(0) == 0


def test_frame_one_is_latent_one_not_latent_zero():
    """The formula's discontinuity: latent 0 and latent 1 map to pixel 0 and
    pixel 1, then the stride jumps to ``ratio``. Off-by-one bait."""
    assert conditioning.latent_index_for(1) == 1
    assert conditioning.pixel_index_for(1) == 1
    assert conditioning.pixel_index_for(2) == 9


def test_the_ratio_is_a_parameter_not_the_literal_eight():
    """Same principle as ``constraints.py`` (§5.4): the 8 is
    ``vae.temporal_compression_ratio``."""
    assert conditioning.pixel_index_for(2, temporal_compression_ratio=4) == 5
    assert conditioning.pixel_index_for(2, temporal_compression_ratio=8) == 9
    assert conditioning.latent_index_for(5, temporal_compression_ratio=4) == 2


# --------------------------------------------------------------------------
# range validation (PLAN §5.7 row 2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pixel_frame", [0, 1, 9, 120])
def test_an_in_range_condition_index_validates(pixel_frame):
    assert conditioning.validate_condition_index(pixel_frame, num_frames=121) == pixel_frame


@pytest.mark.parametrize("pixel_frame", [121, 200, -1])
def test_an_out_of_range_condition_index_raises(pixel_frame):
    """Upstream skips it with a ``logger.warning`` and generates anyway. The user
    supplied an image, the image did nothing, and no one was told. That is
    exactly the silent-failure class §5.3 exists to eliminate."""
    with pytest.raises(ConstraintError):
        conditioning.validate_condition_index(pixel_frame, num_frames=121)


def test_the_range_error_names_the_index_and_the_bound():
    with pytest.raises(ConstraintError) as excinfo:
        conditioning.validate_condition_index(200, num_frames=121)
    message = str(excinfo.value)
    assert "200" in message and "121" in message, message


def test_a_condition_index_is_never_silently_dropped():
    """Whatever the failure, it must be an exception — never a filtered-out
    condition and a shorter list than the user built."""
    with pytest.raises(LTXStudioError):
        conditioning.validate_condition_index(999, num_frames=9)


# --------------------------------------------------------------------------
# CRF selection (PLAN §5.7 row 8, §5.4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["gemma4_unified", "gemma4"])
def test_ltx25_text_encoders_select_crf_18(model_type):
    from ltx25_studio import constraints

    assert constraints.image_crf_for(model_type) == 18


@pytest.mark.parametrize("model_type", ["gemma3", "t5", "llama"])
def test_other_text_encoders_select_the_generic_crf(model_type):
    """"hardcoding either number is wrong for the other" — so it is derived from
    ``text_encoder.config.model_type`` at runtime, not written down."""
    from ltx25_studio import constraints

    assert constraints.image_crf_for(model_type) == 33
