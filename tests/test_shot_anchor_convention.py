"""The shot-anchor convention, as ruled — PLAN §5.7 rows 1 and 2.

**The ruling: the user types a PIXEL FRAME. ``conditioning.py`` converts
pixel → latent.**

This file exists because the two halves of the original instruction could not
both be true: *"the anchor index is a latent index, label it so users are not
misled, **and** let ``conditioning.py`` do the conversion."* If the field is
already a latent index there is nothing to convert; if the conversion runs, the
field must be a frame. The UI implemented the first half, ``conditioning.py``
implemented the second, and each was internally consistent — so neither side's
tests could see the contradiction. Wiring them naively double-converts and moves
every anchor.

A convention that lives in two modules' heads and in neither's tests is not a
convention; it is a coincidence waiting to be corrected. So it is written down
here, once, as executable statements:

* the column is labelled as a **frame**, and does not say "latent";
* a frame the user types reaches the pipeline as the latent index for that
  frame — converted **exactly once**;
* the mapping is **lossy**, and the UI says so instead of snapping silently.

That last point is the nuance the ruling calls out. At 8:1 compression a run of
frames collapses onto one latent, so two shots the user believes are a third of a
second apart can be the same anchor. Both cases are specified: the **exact** one,
where the frame is the first of its span, and the **inexact** one, where it is
not and the readout must say where it actually lands.
"""

from __future__ import annotations

import pytest
from conftest import image_input, inputs_for
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

from ltx25_studio import conditioning
from ltx25_studio import generate as generate_module
from ltx25_studio.constraints import DEFAULT_TEMPORAL_COMPRESSION_RATIO
from ltx25_studio.errors import LTXStudioError
from ltx25_studio.ui import fields, handlers

RATIO = DEFAULT_TEMPORAL_COMPRESSION_RATIO
NUM_FRAMES = 121

#: ``(frame, latent)`` where the frame is the FIRST of that latent's span, so the
#: conversion is exact. Computed by hand from ``(frame - 1) // ratio + 1``.
EXACT_ANCHORS = [(0, 0), (1, 1), (9, 2), (17, 3), (25, 4), (33, 5)]

#: ``(frame, latent, first_frame_of_the_span)`` where the frame is mid-span, so
#: the anchor snaps backwards and the user must be told.
INEXACT_ANCHORS = [(5, 1, 1), (12, 2, 9), (30, 4, 25), (32, 4, 25)]


def shot_row(frame, prompt="a second shot", strength=1.0):
    """One row exactly as ``gr.Dataframe(type="array")`` hands it over."""
    row = [None] * len(fields.SHOT_COLUMNS)
    row[0] = prompt
    row[fields.SHOT_ANCHOR_COLUMN] = frame
    row[2] = strength
    return row


def keyframe(frame, strength=1.0):
    """An image anchored at a pixel ``frame``, as a programmatic caller supplies one.

    End to end, a shot-list ROW is refused before the run (its first cell is
    prompt text, not frames), so the convention is driven with a real keyframe:
    same anchor field, same single conversion in ``conditioning.py``.
    """
    return {"source": image_input(), "index": frame, "strength": strength}


def conditions_reaching_the_pipeline(rows, tmp_path, *, num_frames=NUM_FRAMES):
    pipe = FakePipe()
    generate_module.generate(
        mode="image_text_to_video",
        seed=7,
        **inputs_for("image_text_to_video"),
        num_frames=num_frames,
        conditions=rows,
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    return pipe.calls[-1].get("conditions", ())


# --------------------------------------------------------------------------
# the ruling, stated
# --------------------------------------------------------------------------


def test_the_anchor_column_is_labelled_as_a_frame():
    """Users think in frames. A latent index is an artefact of the VAE.

    The label is half the convention: a column that says "LATENT" makes the
    off-by-8 the user's problem to understand, and contradicts the module that
    converts for them.
    """
    label = fields.SHOT_COLUMNS[fields.SHOT_ANCHOR_COLUMN][0]
    assert "frame" in label.lower(), (
        f"the anchor column is labelled {label!r}, which does not tell the user it takes a "
        f"frame number."
    )
    assert "latent" not in label.lower(), (
        f"the anchor column is labelled {label!r}. conditioning.py converts frame -> latent, "
        f"so a column asking for a latent index would be converted a second time and every "
        f"anchor would move."
    )


def test_the_ui_and_the_converter_read_the_same_column():
    """Two modules, two constants, one column. They have to agree.

    ``fields.SHOT_ANCHOR_COLUMN`` positions the widget; ``conditioning``'s own
    row offset reads the value back out. Nothing forces them to match, and if
    they drift the converter reads the *prompt text* as a frame number — which
    coerces to 0 or raises, several layers from the cause.
    """
    assert conditioning.SHOT_ROW_ANCHOR == fields.SHOT_ANCHOR_COLUMN, (
        f"conditioning reads shot-row column {conditioning.SHOT_ROW_ANCHOR} while the UI puts "
        f"the anchor in column {fields.SHOT_ANCHOR_COLUMN}."
    )


# --------------------------------------------------------------------------
# converted exactly once, all the way to the pipeline
# --------------------------------------------------------------------------


ALL_ANCHORS = EXACT_ANCHORS + [(frame, latent) for frame, latent, _ in INEXACT_ANCHORS]


@pytest.mark.parametrize(("frame", "latent"), ALL_ANCHORS)
def test_a_frame_the_user_types_arrives_as_that_frames_latent(frame, latent, tmp_path):
    """The whole convention in one assertion, end to end.

    Three outcomes are distinguishable on purpose: frame 25 unconverted is 25,
    converted once is 4, converted twice is 1. A test that only checked "it is
    an int" or "it is not the frame" would pass on the double conversion.
    """
    conditions = conditions_reaching_the_pipeline([keyframe(frame)], tmp_path)
    shot = conditions[-1]

    assert shot.index == latent, (
        f"the user typed frame {frame}; the pipeline received index {shot.index}. "
        f"Converted once it is {latent}; unconverted it would be {frame}; converted twice it "
        f"would be {conditioning.latent_index_for(latent)}."
    )


@pytest.mark.parametrize("frame", [17, 25, 33])
def test_the_anchor_is_not_converted_twice(frame, tmp_path):
    """Stated separately, because double conversion is the specific failure the
    contradicted instruction produced — and it looks like a working conversion."""
    conditions = conditions_reaching_the_pipeline([keyframe(frame)], tmp_path)

    once = conditioning.latent_index_for(frame)
    twice = conditioning.latent_index_for(once)
    assert once != twice, f"frame {frame} is a poor probe: one and two conversions agree"
    assert conditions[-1].index == once, (
        f"frame {frame} reached the pipeline as {conditions[-1].index}, which is the "
        f"double-converted value. conditioning.py ran twice on the same anchor."
    )


@pytest.mark.parametrize("frame", [17, 25, 33])
def test_the_anchor_is_converted_at_all(frame, tmp_path):
    """The opposite failure: the raw frame travelling through untouched.

    Upstream would then anchor the shot ~8x further into the clip than the user
    asked, or skip it entirely — with a log line, and a completed metered run.
    """
    conditions = conditions_reaching_the_pipeline([keyframe(frame)], tmp_path)
    assert conditions[-1].index != frame, (
        f"frame {frame} reached the pipeline unconverted. LTX2VideoCondition.index is a "
        f"latent index (PLAN §5.7 row 1)."
    )


@pytest.fixture
def diffusers_condition_type(monkeypatch):
    """Make the lazily-imported diffusers condition type resolvable offline.

    ``pipelines._with_real_conditions`` returns its input unchanged when
    diffusers is absent, so **without this the entire conversion branch is
    unreachable in tests** — and a spec written against it passes no matter what
    the branch does. That is not hypothetical: the first version of the
    idempotence test below was silently vacuous, and survived a mutation that
    deleted the guard it was written to protect.

    A stand-in module rather than the real package: still no GPU, no weights and
    no 72 GB of diffusers, but the branch actually executes.
    """
    import sys
    import types

    class LTX2VideoCondition:
        def __init__(self, *, frames, index, strength, **extra):
            self.frames = frames
            self.index = index
            self.strength = strength

    for name in ("diffusers", "diffusers.pipelines", "diffusers.pipelines.ltx2"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    leaf = types.ModuleType("diffusers.pipelines.ltx2.pipeline_ltx2_condition")
    leaf.LTX2VideoCondition = LTX2VideoCondition
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)
    return LTX2VideoCondition


def test_the_conversion_branch_is_actually_reachable_in_these_tests(diffusers_condition_type):
    """Guards the fixture itself, so the specs below cannot go quietly vacuous.

    If the stand-in ever stops resolving, this fails loudly instead of letting
    every idempotence assertion below pass on an untouched input.
    """
    from ltx25_studio import pipelines

    spec = conditioning.ConditionSpec(source="img.png", kind="image", index=4, strength=1.0)
    converted = pipelines._with_real_conditions({"conditions": [spec], "num_frames": NUM_FRAMES})[
        "conditions"
    ][0]

    assert isinstance(converted, diffusers_condition_type), (
        "the conditions were returned unconverted, so the diffusers branch did not run and "
        "nothing below is really being tested"
    )


@pytest.mark.parametrize("latent", [1, 3, 4, 5])
def test_a_spec_that_already_carries_a_latent_index_is_left_alone(latent, diffusers_condition_type):
    """Idempotence, at the layer where a second conversion could still creep in.

    ``StageRunner`` normalises anything that is not already a ``ConditionSpec``.
    A spec arriving from ``generate()`` has been converted and range-checked, so
    re-normalising it would move the anchor a second time — which is precisely
    the double conversion the contradicted instruction produced.
    """
    from ltx25_studio import pipelines

    already = conditioning.ConditionSpec(source="img.png", kind="image", index=latent, strength=1.0)
    converted = pipelines._with_real_conditions(
        {"conditions": [already], "num_frames": NUM_FRAMES}
    )["conditions"][0]

    assert converted.index == latent, (
        f"a ConditionSpec already carrying latent {latent} was re-converted to "
        f"{converted.index} (a second pass would give "
        f"{conditioning.latent_index_for(latent)}). The anchor moves every time it crosses "
        f"this layer."
    )


def test_re_normalising_really_would_move_the_anchor(diffusers_condition_type):
    """Proves the guard above is load-bearing rather than decorative.

    If converting twice happened to be a no-op, the idempotence tests would pass
    for the wrong reason and tell us nothing.
    """
    assert conditioning.latent_index_for(4) != 4, (
        "converting an already-converted anchor is a no-op for this value, so the idempotence "
        "specs would not detect a second pass"
    )


# --------------------------------------------------------------------------
# the exact case
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("frame", "latent"), EXACT_ANCHORS)
def test_a_frame_at_the_start_of_its_span_is_reported_as_exact(frame, latent):
    """The frame the user typed is the frame the anchor lands on. Nothing to warn about."""
    group = conditioning.latent_group_for(frame)

    assert group["latent"] == latent
    assert group["first_frame"] == frame
    assert group["exact"] is True, (
        f"frame {frame} is the first frame of latent {latent} and should be reported as an "
        f"exact anchor."
    )


@pytest.mark.parametrize(("frame", "_latent"), EXACT_ANCHORS)
def test_an_exact_anchor_is_not_flagged_as_snapping(frame, _latent):
    """A warning that fires on correct input is a warning nobody reads."""
    readout = handlers.on_shots_change([shot_row(frame)], NUM_FRAMES)
    assert "⟵" not in readout, (
        f"frame {frame} lands exactly on its latent and the readout marked it as snapping:"
        f"\n\n{readout}"
    )
    assert "snap" not in readout.lower()


# --------------------------------------------------------------------------
# the inexact case — the lossy half the ruling calls out
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("frame", "latent", "first"), INEXACT_ANCHORS)
def test_a_frame_mid_span_is_reported_as_inexact(frame, latent, first):
    """Frame 30 anchors where frame 25 does. Silently snapping is quiet wrongness."""
    group = conditioning.latent_group_for(frame)

    assert group["latent"] == latent
    assert group["first_frame"] == first
    assert group["exact"] is False, (
        f"frame {frame} snaps back to frame {first} (both are latent {latent}) and must not "
        f"be reported as an exact anchor."
    )
    assert group["shared"] is True


@pytest.mark.parametrize(("frame", "latent", "first"), INEXACT_ANCHORS)
def test_the_readout_says_where_an_inexact_anchor_actually_lands(frame, latent, first):
    """ "Frame 30 → latent 4 (frames 25–32 share this anchor)" is honest.

    Telling the user the rule is weaker than showing them the span, because the
    rule is the part people misread — and finding out after a metered run is the
    expensive way.
    """
    readout = handlers.on_shots_change([shot_row(frame)], NUM_FRAMES)

    assert "⟵" in readout or "snap" in readout.lower(), (
        f"frame {frame} snaps to frame {first} and the readout does not say so:\n\n{readout}"
    )
    assert str(first) in readout, (
        f"the readout does not name frame {first}, which is where frame {frame} actually "
        f"anchors:\n\n{readout}"
    )
    assert str(conditioning.latent_group_for(frame)["last_frame"]) in readout, (
        f"the readout does not show the end of the span sharing latent {latent}:\n\n{readout}"
    )


def test_two_frames_the_user_thinks_are_apart_are_named_as_the_same_anchor():
    """The concrete user-facing consequence, spelled out.

    Frames 25 and 30 are five frames apart on screen and identical to the model.
    The readout has to be the place that says so.
    """
    assert (
        conditioning.latent_group_for(25)["latent"] == conditioning.latent_group_for(30)["latent"]
    )

    readout = handlers.on_shots_change([shot_row(25), shot_row(30)], NUM_FRAMES)
    assert "⟵" in readout or "snap" in readout.lower(), (
        f"two shots anchored five frames apart resolve to the same latent and the readout "
        f"does not mention it:\n\n{readout}"
    )


def test_the_span_is_exactly_the_frames_that_share_the_latent():
    """The reported span must be the truth, not an approximation of it."""
    group = conditioning.latent_group_for(30)
    span = range(group["first_frame"], group["last_frame"] + 1)

    for frame in span:
        assert conditioning.latent_index_for(frame) == group["latent"], (
            f"frame {frame} is inside the reported span but resolves to latent "
            f"{conditioning.latent_index_for(frame)}, not {group['latent']}"
        )
    for frame in (group["first_frame"] - 1, group["last_frame"] + 1):
        if frame < 0:
            continue
        assert conditioning.latent_index_for(frame) != group["latent"], (
            f"frame {frame} is outside the reported span but resolves to the same latent"
        )
    assert len(span) == RATIO, f"a latent should cover {RATIO} frames, span was {len(span)}"


# --------------------------------------------------------------------------
# the range check, under the frame convention
# --------------------------------------------------------------------------


def test_an_anchor_past_the_end_of_the_clip_is_refused_before_the_run(tmp_path):
    """PLAN §5.7 row 2: upstream skips it with a log line and generates anyway.

    Now that the column is a frame count, "outside the clip" means exactly what
    the user's own frame count says — which is the comparison that can finally
    be made.
    """
    with pytest.raises(LTXStudioError):
        conditions_reaching_the_pipeline([keyframe(NUM_FRAMES + 50)], tmp_path)


def test_the_last_legal_frame_is_accepted(tmp_path):
    """The boundary, in the direction that matters: off-by-one here rejects a
    legitimate anchor and the user has no way to know why."""
    conditions = conditions_reaching_the_pipeline([keyframe(NUM_FRAMES - 1)], tmp_path)
    assert conditions[-1].index == conditioning.latent_index_for(NUM_FRAMES - 1)


def test_the_readout_warns_about_a_frame_outside_the_clip():
    """Caught while the user is still typing, not after a metered run."""
    readout = handlers.on_shots_change([shot_row(NUM_FRAMES + 50)], NUM_FRAMES)
    assert "⚠️" in readout or "outside" in readout.lower()
