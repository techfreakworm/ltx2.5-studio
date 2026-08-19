"""Spec for ``ltx25_studio/constraints.py`` — PLAN §5.5, §10 OQ-3.

The rule this module exists for: **upstream does not raise, so we must.**
``num_frames`` is fed into ``latent_num_frames = (num_frames - 1) // ratio + 1``,
which silently floors a bad value. 120 frames and 113 frames produce the *same*
latent count; the user asked for one and got the other, with no error anywhere.

The contract:

* ``validate_*`` raises; ``snap_*`` fixes. Never both, never neither.
* The modulus is ``vae.temporal_compression_ratio``, passed in as
  ``temporal_compression_ratio=``. It defaults to 8 only so the UI can validate
  before any VAE exists (§5.4's own principle applied to §5.5).
* Snapping is idempotent, and a legal value snaps to itself.
"""

from __future__ import annotations

import pytest

from ltx25_studio import constraints
from ltx25_studio.errors import DimensionError, FrameCountError, LTXStudioError

LEGAL_FRAME_COUNTS = [1, 9, 17, 25, 49, 97, 113, 121]
ILLEGAL_FRAME_COUNTS = [120, 122, 0, -1, 2, 8, 100, 128]


# --------------------------------------------------------------------------
# num_frames % 8 == 1
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_frames", LEGAL_FRAME_COUNTS)
def test_legal_frame_counts_validate(num_frames):
    assert constraints.validate_num_frames(num_frames) == num_frames


@pytest.mark.parametrize("num_frames", ILLEGAL_FRAME_COUNTS)
def test_illegal_frame_counts_raise_a_typed_error(num_frames):
    with pytest.raises(FrameCountError):
        constraints.validate_num_frames(num_frames)


@pytest.mark.parametrize(
    "num_frames,expected_neighbours",
    [
        (120, ("113", "121")),
        (122, ("121", "129")),
        (100, ("97", "105")),
    ],
)
def test_frame_count_error_names_the_nearest_legal_values(num_frames, expected_neighbours):
    with pytest.raises(FrameCountError) as excinfo:
        constraints.validate_num_frames(num_frames)
    message = str(excinfo.value)
    for neighbour in expected_neighbours:
        assert neighbour in message, f"{num_frames}: message must name {neighbour}; got {message!r}"


@pytest.mark.parametrize("num_frames", [0, -1, -8])
def test_frame_count_error_for_sub_legal_values_names_the_minimum(num_frames):
    with pytest.raises(FrameCountError) as excinfo:
        constraints.validate_num_frames(num_frames)
    assert "1" in str(excinfo.value)


def test_upstream_silently_floors_exactly_where_we_raise():
    """The rationale, as an executable comment (PLAN §10 OQ-3).

    113..120 all collapse to the same ``latent_num_frames`` upstream. A user who
    types 120 gets 113 frames of video and no warning. That is the bug we refuse
    to inherit.
    """
    ratio = 8
    latent_counts = {(n - 1) // ratio + 1 for n in range(113, 121)}
    assert latent_counts == {15}, "upstream floor behaviour changed; re-read §10 OQ-3"

    assert constraints.validate_num_frames(113) == 113
    for silently_floored in range(114, 121):
        with pytest.raises(FrameCountError):
            constraints.validate_num_frames(silently_floored)


@pytest.mark.parametrize("num_frames", LEGAL_FRAME_COUNTS)
def test_is_valid_num_frames_agrees_with_validate(num_frames):
    assert constraints.is_valid_num_frames(num_frames) is True


@pytest.mark.parametrize("num_frames", ILLEGAL_FRAME_COUNTS)
def test_is_valid_num_frames_rejects_illegal(num_frames):
    assert constraints.is_valid_num_frames(num_frames) is False


# --------------------------------------------------------------------------
# the modulus is NOT the literal 8
# --------------------------------------------------------------------------


def test_default_temporal_compression_ratio_is_the_documented_fallback():
    assert constraints.DEFAULT_TEMPORAL_COMPRESSION_RATIO == 8


def test_omitting_the_ratio_behaves_as_ratio_eight():
    assert constraints.validate_num_frames(121) == 121
    with pytest.raises(FrameCountError):
        constraints.validate_num_frames(5)


def test_ratio_four_accepts_a_value_that_ratio_eight_rejects():
    """The decisive proof that 8 is not hardcoded.

    5 frames is legal at ``temporal_compression_ratio=4`` (4*1+1) and illegal at
    8. A validator that hardcodes 8 cannot pass both halves of this test.
    """
    assert constraints.validate_num_frames(5, temporal_compression_ratio=4) == 5
    with pytest.raises(FrameCountError):
        constraints.validate_num_frames(5, temporal_compression_ratio=8)


@pytest.mark.parametrize("num_frames", [1, 5, 9, 117, 121])
def test_ratio_four_legal_values(num_frames):
    assert constraints.validate_num_frames(num_frames, temporal_compression_ratio=4) == num_frames


@pytest.mark.parametrize("num_frames", [4, 120, 122, 0])
def test_ratio_four_illegal_values(num_frames):
    with pytest.raises(FrameCountError):
        constraints.validate_num_frames(num_frames, temporal_compression_ratio=4)


def test_ratio_four_error_names_the_ratio_four_neighbours():
    with pytest.raises(FrameCountError) as excinfo:
        constraints.validate_num_frames(120, temporal_compression_ratio=4)
    message = str(excinfo.value)
    assert "117" in message and "121" in message, message


@pytest.mark.parametrize("num_frames", [1, 33, 65])
def test_ratio_thirtytwo_is_supported_too(num_frames):
    assert constraints.validate_num_frames(num_frames, temporal_compression_ratio=32) == num_frames


@pytest.mark.parametrize("bad_ratio", [0, -1])
def test_a_nonsense_ratio_raises_our_error_not_zerodivisionerror(bad_ratio):
    with pytest.raises(LTXStudioError):
        constraints.validate_num_frames(121, temporal_compression_ratio=bad_ratio)


# --------------------------------------------------------------------------
# nearest / snap for frames
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_frames,expected",
    [(120, (113, 121)), (122, (121, 129)), (121, (121, 121)), (1, (1, 1)), (0, (1, 1)), (-4, (1, 1))],
)
def test_nearest_valid_num_frames(num_frames, expected):
    assert constraints.nearest_valid_num_frames(num_frames) == expected


@pytest.mark.parametrize(
    "num_frames,expected",
    [(120, 121), (122, 121), (114, 113), (0, 1), (-9, 1), (121, 121), (1, 1)],
)
def test_snap_num_frames(num_frames, expected):
    assert constraints.snap_num_frames(num_frames) == expected


@pytest.mark.parametrize("num_frames,expected", [(120, 121), (118, 117), (5, 5)])
def test_snap_num_frames_honours_a_non_default_ratio(num_frames, expected):
    assert constraints.snap_num_frames(num_frames, temporal_compression_ratio=4) == expected


@pytest.mark.parametrize("num_frames", LEGAL_FRAME_COUNTS)
def test_snapping_a_legal_frame_count_returns_it_unchanged(num_frames):
    assert constraints.snap_num_frames(num_frames) == num_frames


@pytest.mark.parametrize("num_frames", ILLEGAL_FRAME_COUNTS + LEGAL_FRAME_COUNTS)
def test_snap_num_frames_is_idempotent(num_frames):
    once = constraints.snap_num_frames(num_frames)
    assert constraints.snap_num_frames(once) == once


@pytest.mark.parametrize("num_frames", ILLEGAL_FRAME_COUNTS + LEGAL_FRAME_COUNTS)
def test_snap_num_frames_always_produces_a_valid_value(num_frames):
    assert constraints.validate_num_frames(constraints.snap_num_frames(num_frames))


# --------------------------------------------------------------------------
# width / height % 32 == 0
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [32, 64, 512, 704, 1024, 1216])
def test_legal_dimensions_validate(value):
    assert constraints.validate_dimension(value, name="width") == value


@pytest.mark.parametrize("value", [1, 31, 33, 700, 1000, 0, -32])
def test_illegal_dimensions_raise(value):
    with pytest.raises(DimensionError):
        constraints.validate_dimension(value, name="width")


def test_dimension_error_names_the_axis_the_value_and_the_multiple():
    with pytest.raises(DimensionError) as excinfo:
        constraints.validate_dimension(700, name="height")
    message = str(excinfo.value)
    assert "height" in message and "700" in message and "32" in message, message


def test_validate_dimensions_checks_both_axes():
    assert constraints.validate_dimensions(704, 480) == (704, 480)
    with pytest.raises(DimensionError):
        constraints.validate_dimensions(700, 480)
    with pytest.raises(DimensionError):
        constraints.validate_dimensions(704, 481)


def test_dimension_multiple_constant():
    assert constraints.DIMENSION_MULTIPLE == 32


# --------------------------------------------------------------------------
# snapping dimensions, in both directions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (700, 704),  # up
        (1000, 992),  # down
        (48, 64),  # tie -> up, never toward zero
        (1, 32),  # clamped to the minimum legal value
        (0, 32),
        (-16, 32),
        (512, 512),  # already legal
    ],
)
def test_snap_dimension(value, expected):
    assert constraints.snap_dimension(value) == expected


@pytest.mark.parametrize("value", [0, 1, 31, 48, 512, 700, 1000, 1216])
def test_snap_dimension_is_idempotent(value):
    once = constraints.snap_dimension(value)
    assert constraints.snap_dimension(once) == once


@pytest.mark.parametrize("value", [0, 1, 31, 48, 512, 700, 1000, 1216])
def test_snap_dimension_always_produces_a_valid_value(value):
    snapped = constraints.snap_dimension(value)
    assert constraints.validate_dimension(snapped, name="width") == snapped


def test_snap_dimensions_snaps_both_axes():
    assert constraints.snap_dimensions(700, 1000) == (704, 992)


def test_snap_dimensions_leaves_legal_values_alone():
    assert constraints.snap_dimensions(704, 480) == (704, 480)


# --------------------------------------------------------------------------
# CRF and sigma schedules (PLAN §5.5)
# --------------------------------------------------------------------------


def test_ltx25_image_conditioning_crf_is_18_not_the_generic_33():
    assert constraints.LTX2_5_IMAGE_CRF == 18
    assert constraints.DEFAULT_IMAGE_CRF == 33
    assert constraints.LTX2_5_IMAGE_CRF != constraints.DEFAULT_IMAGE_CRF


def test_distilled_sigma_schedule_is_the_verified_eight_step_schedule():
    assert list(constraints.DISTILLED_SIGMA_VALUES) == [
        1.0,
        0.99375,
        0.9875,
        0.98125,
        0.975,
        0.909375,
        0.725,
        0.421875,
    ]


def test_stage_2_distilled_sigma_schedule_is_the_verified_three_step_schedule():
    assert list(constraints.STAGE_2_DISTILLED_SIGMA_VALUES) == [0.909375, 0.725, 0.421875]


@pytest.mark.parametrize("name", ["DISTILLED_SIGMA_VALUES", "STAGE_2_DISTILLED_SIGMA_VALUES"])
def test_sigma_schedules_are_strictly_decreasing_and_in_range(name):
    values = list(getattr(constraints, name))
    assert values, f"{name} is empty"
    assert all(0.0 < v <= 1.0 for v in values), values
    assert values == sorted(values, reverse=True), values
    assert len(set(values)) == len(values), values


def test_sigma_schedules_match_the_installed_diffusers():
    """PLAN §5.5: an upstream schedule change must fail a test, not change output quality.

    Skipped until diffusers is installed (it is not a test-time dependency).
    """
    utils = pytest.importorskip("diffusers.pipelines.ltx2.utils")
    assert list(constraints.DISTILLED_SIGMA_VALUES) == list(utils.DISTILLED_SIGMA_VALUES)
    assert list(constraints.STAGE_2_DISTILLED_SIGMA_VALUES) == list(
        utils.STAGE_2_DISTILLED_SIGMA_VALUES
    )


# --------------------------------------------------------------------------
# audio/video length agreement (PLAN §5.5 row 4, §3.1c)
# --------------------------------------------------------------------------


def test_matching_audio_and_video_durations_validate():
    assert constraints.validate_audio_video_agreement(video_seconds=5.0, audio_seconds=5.0) is None


def test_desynced_audio_and_video_raise_rather_than_emitting_a_desynced_file():
    from ltx25_studio.errors import AudioVideoDesyncError

    with pytest.raises(AudioVideoDesyncError) as excinfo:
        constraints.validate_audio_video_agreement(video_seconds=5.0, audio_seconds=4.25)
    message = str(excinfo.value)
    assert "5.0" in message and "4.25" in message, message


def test_sub_frame_rounding_slack_is_tolerated():
    """Frame counts are computed independently (§3.1c); exact float equality is not a
    reasonable bar. A tolerance is — as long as it is small and explicit."""
    assert (
        constraints.validate_audio_video_agreement(
            video_seconds=5.0, audio_seconds=5.01, tolerance_seconds=0.05
        )
        is None
    )
