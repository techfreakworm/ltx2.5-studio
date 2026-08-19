"""Spec for the latent<->pixel index conversion — PLAN §5.7 rows 1 and 2.

``latent_to_pixel_index`` → ``return 0`` left the suite fully green, and
`grep -rn latent_to_pixel_index tests/` returned **0**. So did
``pixel_to_latent_index``.

PLAN calls this conversion "the severe one" and it earns the name. Upstream
computes ``start_idx = max((latent_start_idx - 1) * ratio + 1, 0)``, so a
conditioning image passed a pixel frame where a latent index belongs lands eight
frames from where the user put it — *guaranteed, silently, every time*. There is
no error, no warning, and no way for a user to tell a mis-indexed anchor from a
model that simply ignored their image.

The function is also read twice for two different purposes — the shot-list
readout in the UI and the conditions handed to the pipeline — which is exactly
the shape where a stubbed or drifted implementation stays invisible: a readout
that agrees with a wrong generator looks perfectly consistent.

What kills ``return 0`` and nothing else:

* the **kink**. Latent 0 and 1 both map to small pixel values and only then does
  the stride become the compression ratio. A test that only checks large indices
  misses the discontinuity; a test that only checks index 0 is satisfied by the
  stub.
* **injectivity**. Distinct latents must land on distinct pixels. A constant
  passes every "is it an int" and "is it non-negative" check ever written.
* the **round trip**, in both directions.
"""

from __future__ import annotations

import pytest

from ltx25_studio import conditioning
from ltx25_studio.constraints import DEFAULT_TEMPORAL_COMPRESSION_RATIO
from ltx25_studio.ui import handlers
from ltx25_studio.ui.fields import SHOT_ANCHOR_COLUMN, SHOT_COLUMNS

RATIO = DEFAULT_TEMPORAL_COMPRESSION_RATIO

#: Computed by hand from upstream's own expression, ``max((n - 1) * ratio + 1, 0)``.
#: Written out rather than generated, so a bug in the implementation cannot also
#: be a bug in the expectation.
LATENT_TO_PIXEL = {0: 0, 1: 1, 2: 9, 3: 17, 4: 25, 5: 33}


def test_the_compression_ratio_is_the_one_these_expectations_were_computed_for():
    """If the ratio moves, the hand-computed table below is wrong, not the code."""
    assert RATIO == 8


# --------------------------------------------------------------------------
# the arithmetic, including the discontinuity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("latent", "pixel"), sorted(LATENT_TO_PIXEL.items()))
def test_a_latent_index_maps_to_its_upstream_pixel_frame(latent, pixel):
    """``max((latent - 1) * ratio + 1, 0)``, quoted from upstream."""
    assert conditioning.pixel_index_for(latent) == pixel, (
        f"latent {latent} must land on pixel frame {pixel}; a wrong conversion puts the "
        f"user's conditioning image {RATIO} frames from where they put it, silently "
        f"(PLAN §5.7 row 1)."
    )


def test_the_kink_at_the_start_is_preserved():
    """Latents 0 and 1 are one pixel apart; after that the stride is the ratio.

    "That kink is where off-by-one bugs live" — the module's own docstring. A
    conversion that is merely linear everywhere looks right on big numbers and
    is wrong at exactly the index most users pick: the first frame.
    """
    assert conditioning.pixel_index_for(1) - conditioning.pixel_index_for(0) == 1
    assert conditioning.pixel_index_for(2) - conditioning.pixel_index_for(1) == RATIO
    assert conditioning.pixel_index_for(3) - conditioning.pixel_index_for(2) == RATIO


def test_distinct_latent_indices_land_on_distinct_pixel_frames():
    """Injectivity — the property a constant-returning stub cannot fake."""
    pixels = [conditioning.pixel_index_for(n) for n in range(12)]
    assert len(set(pixels)) == len(pixels), (
        f"the conversion collapses distinct anchors onto the same frame: {pixels}. Every "
        f"shot in a list would anchor to the same place."
    )


def test_the_conversion_is_monotonic():
    """A later latent index can never point at an earlier frame."""
    pixels = [conditioning.pixel_index_for(n) for n in range(12)]
    assert pixels == sorted(pixels), f"non-monotonic conversion: {pixels}"


def test_the_conversion_never_returns_a_negative_frame():
    """``max(..., 0)`` is in upstream's expression and is load-bearing."""
    assert conditioning.pixel_index_for(0) == 0
    assert conditioning.pixel_index_for(-3) == 0


# --------------------------------------------------------------------------
# the round trip, both ways
# --------------------------------------------------------------------------


@pytest.mark.parametrize("latent", range(10))
def test_a_latent_index_survives_a_round_trip(latent):
    assert conditioning.latent_index_for(conditioning.pixel_index_for(latent)) == latent


@pytest.mark.parametrize("pixel", [0, 1, 9, 17, 25, 33])
def test_an_anchor_pixel_frame_survives_a_round_trip(pixel):
    assert conditioning.pixel_index_for(conditioning.latent_index_for(pixel)) == pixel


@pytest.mark.parametrize("pixel", range(1, 40))
def test_every_pixel_frame_maps_into_the_latent_that_contains_it(pixel):
    """The inverse is a *containment*, not a bijection: eight pixels share a latent.

    So the round trip from a pixel is only exact on the boundary frames, and what
    must hold everywhere is that the latent chosen actually covers the frame.
    """
    latent = conditioning.latent_index_for(pixel)
    start = conditioning.pixel_index_for(latent)
    end = conditioning.pixel_index_for(latent + 1)
    assert start <= pixel < end, (
        f"pixel frame {pixel} was mapped to latent {latent}, which covers frames "
        f"[{start}, {end}) — it does not contain the frame the user asked for."
    )


# --------------------------------------------------------------------------
# one implementation, two names
# --------------------------------------------------------------------------


@pytest.mark.parametrize("latent", range(8))
def test_the_alias_the_ui_calls_is_the_same_conversion(latent):
    """``latent_to_pixel_index`` is the name the UI asks for; it must not be a copy.

    The module's docstring is explicit about why: "a second copy of
    ``max((latent - 1) * ratio + 1, 0)`` living in the UI could drift from this
    one, and a readout that disagrees with the generator is worse than no readout
    at all."
    """
    assert conditioning.latent_to_pixel_index(latent) == conditioning.pixel_index_for(latent)


@pytest.mark.parametrize("pixel", [0, 1, 9, 17])
def test_the_other_alias_is_the_same_inverse(pixel):
    assert conditioning.pixel_to_latent_index(pixel) == conditioning.latent_index_for(pixel)


@pytest.mark.parametrize("ratio", [4, 8, 16])
def test_the_ratio_is_a_parameter_rather_than_a_constant(ratio):
    """PLAN §5.4: it is read off the loaded VAE, so it must be overridable here."""
    assert conditioning.pixel_index_for(2, temporal_compression_ratio=ratio) == ratio + 1


# --------------------------------------------------------------------------
# the UI seam: the readout has to move when the anchor moves
# --------------------------------------------------------------------------


def shot_rows(*anchors):
    rows = []
    for anchor in anchors:
        row = [None] * len(SHOT_COLUMNS)
        row[0] = "a shot"
        row[SHOT_ANCHOR_COLUMN] = anchor
        rows.append(row)
    return rows


#: Frame anchors and the latent each lands on, computed by hand from
#: ``(frame - 1) // ratio + 1``. The user types a **frame**; the model wants a
#: latent; the conversion is 8:1 lossy in this direction.
FRAME_TO_LATENT = {0: 0, 1: 1, 5: 1, 20: 3, 40: 5}


@pytest.mark.parametrize(("frame", "latent"), sorted(FRAME_TO_LATENT.items()))
def test_a_frame_anchor_reports_the_latent_it_lands_on(frame, latent):
    """The arithmetic behind the readout, asserted independently of the wording.

    This is the half that kills a stubbed conversion: an expectation computed by
    hand, not by calling the function under test.
    """
    assert conditioning.latent_group_for(frame)["latent"] == latent


@pytest.mark.parametrize("frame", [1, 5, 20, 40])
def test_the_span_of_frames_sharing_a_latent_is_reported_correctly(frame):
    """The lossy direction, made visible: which frames mean the same anchor.

    "Two shots the user thinks are a third of a second apart can be identical."
    A stub that reports one frame per latent hides exactly that.
    """
    group = conditioning.latent_group_for(frame)
    assert group["first_frame"] <= frame <= group["last_frame"], (
        f"frame {frame} is reported as sharing latent {group['latent']} with frames "
        f"{group['first_frame']}–{group['last_frame']}, a span that does not contain it."
    )
    assert conditioning.latent_index_for(group["first_frame"]) == group["latent"]
    assert conditioning.latent_index_for(group["last_frame"]) == group["latent"]


def test_two_frames_in_the_same_span_are_the_same_anchor():
    """Frames 1 and 5 are one third of a second apart and mean the same thing."""
    assert conditioning.latent_group_for(1)["latent"] == conditioning.latent_group_for(5)["latent"]


def test_two_frames_in_different_spans_are_different_anchors():
    """And the conversion must still separate anchors that really are separate.

    A stub collapsing everything onto one latent would make every shot in a list
    anchor to the same frame — silently, since nothing downstream can tell.
    """
    assert conditioning.latent_group_for(1)["latent"] != conditioning.latent_group_for(20)["latent"]


@pytest.mark.parametrize("frame", [1, 20, 40])
def test_the_readout_words_the_conversion_it_was_given_without_redoing_it(frame):
    """The seam: the UI must render ``conditioning``'s answer, not compute its own.

    Asserted as an exact match against the renderer driven with the module's own
    output, so the two cannot drift — "a readout that disagrees with what runs is
    worse than no readout at all".
    """
    from ltx25_studio.ui import render

    expected = render.shot_anchor_readout(
        [{"frame": frame, **conditioning.latent_group_for(frame)}], out_of_range=[]
    )
    assert handlers.on_shots_change(shot_rows(frame), 121) == expected


def test_an_anchor_past_the_end_of_the_clip_is_flagged_in_the_readout():
    """PLAN §5.7 row 2: upstream skips it with a log line and generates anyway.

    Catching it while the user is still typing is far cheaper than catching it
    after a metered run — and if the conversion is stubbed to 0, nothing is ever
    out of range and this warning can never appear.
    """
    readout = handlers.on_shots_change(shot_rows(400), 121)
    assert "⚠️" in readout or "outside" in readout.lower() or "range" in readout.lower(), (
        f"latent anchor 400 lands well past a 121-frame clip and the readout does not warn:"
        f"\n\n{readout}"
    )
