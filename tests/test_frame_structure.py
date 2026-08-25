"""Spec for the structure detector — ``ltx25_studio/outputs.py``.

**Every content check this project had could be passed by structureless
garbage.** The artifact that proved it came back from the Space with 121 of 121
distinct frames, the geometry the request asked for, non-zero bytes and a full
luma range — and it was colour static. ``scripts/space_matrix.py`` scored
artifacts like it as PASS, and the app inspected the frames not at all: one
``len(frames) == 0`` check and one ``st_size != 0`` check between the model and
the user.

The reason a checker built from those numbers can never work is that
distinctness, byte count, geometry and brightness range are **properties of
noise**. Noise has all of them, in abundance. What noise does not have is
structure at a scale above one pixel, and that is the only thing worth
measuring here.

Two ratios say it, both divided by the frame's own luma std so they describe
shape rather than contrast:

* **SR** — 8x8 box-mean std over pixel std. Real pictures survive box-averaging
  because neighbouring pixels agree; noise cancels.
* **HF** — 4-neighbour Laplacian std over pixel std. The Laplacian is what is
  left once the picture is subtracted, so on static it is *larger* than the
  signal.

The operator's standing ruling is the shape of this whole file: **a statistical
gate is a pre-filter, never a verdict. It demotes artifacts; it never promotes
one.** Three consequences, each with tests below:

1. **Only one conjunction condemns** — low SR *and* high HF together. Either
   alone describes a legitimate render (soft/defocused, or sharp/grainy).
2. **The weaker findings only WARN.** ``block_std`` and ``luma_band`` are
   absolute measures of how much there is to look at, and a real murky render
   trips them: the banked ``r4a`` artifact measures 5.33 and 24.76 and is a
   genuine (if poor) generation. Refusing on those destroys real work.
3. **The app never raises on any of it.** By the time the frames exist the user
   has paid 35+ GPU-seconds for them. The file is written, the note rides on
   ``GenerationResult.warnings``, a human decides.

Measured on the banked set, full decode, this implementation:

    artifact          WxHxN            SR      HF   blk std    band     sat
    run1/r8/repro     256x256x9     0.9259  0.2377   62.142  246.83  0.4868
    r2_704            704x480x49    0.9201  0.2192   63.252  246.22  0.4781
    r3_retry         1216x704x121   0.9199  0.2232   65.113  246.10  0.5131
    r4a              1408x960x25    0.9756  0.3852    5.334   24.76  0.1670   WARN
    r7a               256x256x9     0.9236  0.2029   64.574  248.06  0.4625
    r7b               256x256x9     0.9939  0.1471   45.749  136.16  0.1415
    d1_noauto         512x960x121   0.9183  0.2254   65.688  247.48  0.4896
    r6_lora/r9        256x256x9     0.9271  0.2295   63.460  248.09  0.4944
    THE FAILURE       512x960x121   0.5298  3.1455    1.360   13.00  0.0885   FAIL

Worst real SR 0.9183, failure 0.5298; ``sqrt(0.9183 * 0.5298) = 0.6975``, so the
0.70 gate sits at the geometric middle with 1.31x under the worst real render
and 1.32x over the failure. HF: worst real 0.3852, failure 3.1455, gate 1.0 —
2.6x and 3.1x. **Saturation is never gated on**: half the banked set is nearly
monochrome and a deliberately black-and-white render measures 0.00.

No GPU, no weights, no network: numpy arrays and ``av``, both pinned.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ltx25_studio import generate as generate_module
from ltx25_studio import outputs

np = pytest.importorskip("numpy")

SIZE = 64
FRAMES = 9
SAMPLING_RATE = 48000


# ---------------------------------------------------------------------------
# frame stacks, built to measure like the artifacts they stand in for
# ---------------------------------------------------------------------------


def _grid():
    return np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)


def _grey(planes):
    """``[F, H, W]`` luma -> ``[F, H, W, 3]`` uint8. Saturation exactly 0.0."""
    return np.repeat(np.clip(planes, 0, 255)[..., None], 3, axis=-1).astype(np.uint8)


def structureless_frames(count=FRAMES, seed=11):
    """Colour static. Distinct, full-range, moving — and nothing is in it.

    This is the operator's failure, reduced: every pre-existing check passes on
    it, which is the point of the fixture.
    """
    return np.random.default_rng(seed).integers(0, 256, (count, SIZE, SIZE, 3), dtype=np.uint8)


def _base_pattern():
    rows, columns = _grid()
    return 40 + 90 * np.sin(columns / 11.0) * np.cos(rows / 17.0) + 60 * (rows / SIZE)


def near_static_frames(count=FRAMES):
    """``r7b``-class: real structure, almost no motion. Must never FAIL.

    The banked ``r7b`` — a 9-frame render whose mean frame-to-frame |Δ| is 0.864,
    only 2.0x above the operator's failure at 0.403 — is the artifact that
    disqualified temporal difference as the signal, so it gets a stand-in here.
    (``r7b`` predates the int8 conditioning fix and 898dad2 invalidates its
    content; what it stands for here is the near-static *shape*, and the fixture
    below is synthetic in any case.) The runner's absolute ``mean |Δ|`` gate has since been demoted
    to a warning for exactly this reason, after it was caught refusing a real
    512x960x121 render at 0.4071 whose sibling measures 0.3859 — a legitimate
    artifact BELOW the failure the threshold was calibrated against.
    """
    base = _base_pattern()
    return _grey(np.stack([base + index * 1.4 for index in range(count)]))


def monochrome_frames(count=FRAMES):
    """Saturation 0.00 and perfectly healthy. Must never FAIL, must never WARN."""
    rows, columns = _grid()
    base = _base_pattern()
    frames = []
    for index in range(count):
        distance = (columns - (12 + index * 4)) ** 2 + (rows - SIZE / 2) ** 2
        frames.append(base + 120 * np.exp(-distance / 120.0))
    return _grey(np.stack(frames))


def murky_frames(count=FRAMES):
    """``r4a``-class: a real render with very little contrast. WARN, never FAIL."""
    rows, columns = _grid()
    base = 120 + 6 * np.sin(columns / 9.0) * np.cos(rows / 13.0)
    return _grey(
        np.stack([base + 3 * np.sin((columns + index * 5) / 9.0) for index in range(count)])
    )


def drifting_flat_frames(count=FRAMES):
    """Each frame one flat colour; the brightness ramps across the stack.

    The stack-wide std of this is 62. Its per-frame std is 0. That gap is the
    mislabelled-``pixel_std`` bug, made visible.
    """
    return np.stack(
        [np.full((SIZE, SIZE, 3), 20 + index * 24, dtype=np.uint8) for index in range(count)]
    )


def verdict(frames, **kwargs):
    measured = outputs.measure_structure(frames, **kwargs)
    return measured, outputs.structure_failure(measured), outputs.structure_warnings(measured)


# ---------------------------------------------------------------------------
# the thresholds are the measured ones
# ---------------------------------------------------------------------------


def test_the_thresholds_are_the_ones_the_banked_set_justifies():
    """Pinned, because a threshold nudged without a re-measurement is how a gate
    stops separating the two populations it was fitted to."""
    assert outputs.MIN_STRUCTURE_RATIO == 0.70
    assert outputs.MAX_HIGH_FREQUENCY_RATIO == 1.0
    assert outputs.LOW_BLOCK_STD == 8.0
    assert outputs.LOW_LUMA_BAND == 32.0


def test_the_structure_gate_sits_between_the_two_measured_populations():
    """1.31x under the worst real render, 1.32x over the failure. A gate with a
    factor of two either side is a coin toss; this one is not."""
    worst_real, the_failure = 0.9183, 0.5298
    assert the_failure < outputs.MIN_STRUCTURE_RATIO < worst_real
    assert worst_real / outputs.MIN_STRUCTURE_RATIO > 1.3
    assert outputs.MIN_STRUCTURE_RATIO / the_failure > 1.3


def test_the_high_frequency_gate_sits_between_the_two_measured_populations():
    worst_real, the_failure = 0.3852, 3.1455
    assert worst_real < outputs.MAX_HIGH_FREQUENCY_RATIO < the_failure
    assert outputs.MAX_HIGH_FREQUENCY_RATIO / worst_real > 2.5
    assert the_failure / outputs.MAX_HIGH_FREQUENCY_RATIO > 3.0


# ---------------------------------------------------------------------------
# it catches the thing every other check missed
# ---------------------------------------------------------------------------


def test_colour_static_is_condemned():
    measured, failure, _warnings = verdict(structureless_frames())
    assert failure is not None, measured.render()
    assert measured.structure_ratio < outputs.MIN_STRUCTURE_RATIO
    assert measured.high_frequency_ratio > outputs.MAX_HIGH_FREQUENCY_RATIO
    assert "colour static" in failure


def test_every_pre_existing_check_passes_on_that_same_static():
    """The reason this detector had to exist at all.

    Distinct frames, a full pixel std, and motion two orders of magnitude above
    the level that would even raise an eyebrow. Not one of them is evidence of a
    picture — they are all properties of noise, and noise has them in abundance.
    """
    frames = structureless_frames().astype("float32")
    distinct = len({frame.tobytes() for frame in structureless_frames()})
    per_frame_std = float(np.median([frame.std() for frame in frames]))
    delta = float(np.mean([np.abs(frames[i + 1] - frames[i]).mean() for i in range(FRAMES - 1)]))

    assert distinct == FRAMES, "9 of 9 distinct — exactly what the failure reported"
    assert per_frame_std > 1.0, "nowhere near flat"
    assert delta > 0.5, "and it moves, by a factor of 100 over the near-static warning"


# ---------------------------------------------------------------------------
# and it does not catch anything else — the direction that costs real work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,builder",
    [
        ("near-static, r7b-class", near_static_frames),
        ("monochrome", monochrome_frames),
        ("murky, r4a-class", murky_frames),
    ],
)
def test_a_real_render_is_never_condemned(name, builder):
    measured, failure, _warnings = verdict(builder())
    assert failure is None, f"{name} was refused: {measured.render()}"
    assert measured.structure_ratio >= outputs.MIN_STRUCTURE_RATIO


def test_a_monochrome_render_is_neither_failed_nor_warned():
    """Saturation 0.00 is a legitimate artistic choice, not a defect. Gating on
    colour would refuse a black-and-white render — the false-defect direction
    that has already cost this build days."""
    measured, failure, warnings = verdict(monochrome_frames())
    assert measured.saturation == pytest.approx(0.0, abs=1e-6)
    assert failure is None
    assert warnings == (), measured.render()


def test_saturation_is_measured_and_never_asserted_on():
    """Both directions of the same property, so the check cannot quietly become
    a colour requirement."""
    colourful = outputs.measure_structure(structureless_frames())
    monochrome = outputs.measure_structure(monochrome_frames())
    assert colourful.saturation > monochrome.saturation
    assert "never asserted" in monochrome.render()


def test_a_near_static_render_survives_where_the_temporal_gate_barely_would():
    """``r7b`` measures 0.864 mean |Δ| against the 0.5 the runner used to refuse
    on, and the operator's static measures 0.403 — a factor of 2.0 between what
    was then a pass and a fail. The structure ratio separates the same two
    artifacts by 1.31x and 1.32x either side of its gate, which is why it, not
    motion, is the load-bearing check — and why motion is now only reported."""
    frames = near_static_frames().astype("float32")
    delta = float(np.mean([np.abs(frames[i + 1] - frames[i]).mean() for i in range(FRAMES - 1)]))
    measured, failure, _warnings = verdict(near_static_frames())
    assert delta < 2.0, "this fixture really is nearly still"
    assert failure is None
    assert measured.structure_ratio > 0.9


# ---------------------------------------------------------------------------
# the WARN tier annotates and nothing more
# ---------------------------------------------------------------------------


def test_a_low_contrast_render_warns_on_both_counts_and_fails_on_neither():
    """``r4a`` measures block std 5.33 and luma band 24.76 and is real output.
    This is the tier that exists so those numbers can be said out loud without
    destroying the artifact."""
    measured, failure, warnings = verdict(murky_frames())
    assert failure is None
    assert measured.block_std < outputs.LOW_BLOCK_STD
    assert measured.luma_band < outputs.LOW_LUMA_BAND
    assert any("low detail" in note for note in warnings)
    assert any("low contrast" in note for note in warnings)
    assert all("not proof of a defect" in note for note in warnings)


def test_a_healthy_render_warns_about_nothing():
    """The other direction. A warner that always warns is noise, and noise gets
    ignored — which is how the one warning that mattered would be missed."""
    _measured, failure, warnings = verdict(monochrome_frames())
    assert failure is None and warnings == ()


def test_the_two_warn_thresholds_sit_above_the_real_artifact_that_trips_them():
    """r4a is *supposed* to trip these. That is the tier working, not a bug."""
    assert outputs.LOW_BLOCK_STD > 5.334
    assert outputs.LOW_LUMA_BAND > 24.76
    assert outputs.LOW_BLOCK_STD < 45.749, "r7b, the next lowest real render, is clear of it"
    assert outputs.LOW_LUMA_BAND < 136.16


# ---------------------------------------------------------------------------
# the conjunction, both halves of it
# ---------------------------------------------------------------------------


def test_a_low_structure_ratio_alone_does_not_condemn():
    """A soft or defocused shot loses block-scale variation without being noise.
    Only the noise term alongside it makes the diagnosis."""
    measured = outputs.FrameStructure(
        frames=9,
        sampled=9,
        structure_ratio=0.10,
        high_frequency_ratio=0.30,
        block_std=40.0,
        luma_band=200.0,
        luma_std=60.0,
        saturation=0.4,
        flat_frames=0,
    )
    assert outputs.structure_failure(measured) is None


def test_a_high_frequency_ratio_alone_does_not_condemn():
    """A sharp, grainy, correctly-generated shot. Structure is still there."""
    measured = outputs.FrameStructure(
        frames=9,
        sampled=9,
        structure_ratio=0.95,
        high_frequency_ratio=3.00,
        block_std=40.0,
        luma_band=200.0,
        luma_std=60.0,
        saturation=0.4,
        flat_frames=0,
    )
    assert outputs.structure_failure(measured) is None


def detailed_frames(count=FRAMES):
    """Coarse structure plus a fine ripple — something for both ratios to see."""
    rows, columns = _grid()
    base = (
        128
        + 70 * np.sin(columns / 11.0) * np.cos(rows / 17.0)
        + 25 * np.sin(columns / 1.7) * np.sin(rows / 2.3)
    )
    return _grey(np.stack([base + index * 4 for index in range(count)]))


def test_the_ratios_are_contrast_invariant():
    """Dimming every pixel scales both stds together and must not move the
    ratios — otherwise the gate is a brightness gate wearing a disguise, and a
    dim shot and a bright shot of the same scene land in different places. That
    is the entire reason both numbers are divided by the frame's own luma std.

    SR holds to four decimals. HF drifts by about a tenth, and the cause is
    named rather than tuned away: uint8 quantization adds a fixed ±0.5-level
    error that the Laplacian sees in full while the signal it is divided by has
    shrunk. It moves the ratio the *safe* way (up, toward the gate, on a dim
    frame) and 0.21 against a 1.0 gate is not a number this can reach.
    """
    bright = detailed_frames()
    dim = (bright.astype("float32") * 0.25).astype(np.uint8)
    full = outputs.measure_structure(bright)
    faded = outputs.measure_structure(dim)
    assert faded.luma_std < full.luma_std / 3, "the fixture really is much darker"
    assert faded.structure_ratio == pytest.approx(full.structure_ratio, rel=1e-3)
    assert faded.high_frequency_ratio == pytest.approx(full.high_frequency_ratio, rel=0.30)
    assert outputs.structure_failure(faded) is None


# ---------------------------------------------------------------------------
# "could not measure" is not "measured and fine"
# ---------------------------------------------------------------------------


def test_a_frame_with_no_contrast_yields_no_ratio_rather_than_a_zero():
    """SR and HF are 0/0 on a uniform frame. Reporting 0.0 would be a fabricated
    measurement that happens to sit below the gate; the honest answer is that
    the ratio does not exist, and the flatness itself is what the runner's own
    flat-frame check is for."""
    measured, failure, warnings = verdict(drifting_flat_frames())
    assert measured.structure_ratio is None
    assert measured.high_frequency_ratio is None
    assert measured.flat_frames == FRAMES
    assert failure is None, "unmeasurable must never read as condemned"
    assert warnings, "but it is still worth saying out loud"
    assert "too flat to measure" in measured.render()


def test_an_empty_frame_stack_measures_nothing_and_condemns_nothing():
    measured, failure, warnings = verdict(np.zeros((0, SIZE, SIZE, 3), dtype=np.uint8))
    assert measured.frames == 0 and measured.sampled == 0
    assert failure is None and warnings == ()


def test_frames_too_small_to_hold_a_block_are_not_condemned():
    """A 4x4 frame has no 8x8 block and no Laplacian interior worth the name.
    Refusing it would be refusing for lack of evidence."""
    tiny = np.random.default_rng(3).integers(0, 256, (4, 4, 4, 3), dtype=np.uint8)
    measured, failure, _warnings = verdict(tiny)
    assert measured.structure_ratio is None
    assert failure is None


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_the_sampled_estimate_tracks_the_full_decode():
    """The app measures 16 evenly-spaced frames, the offline runner measures all
    of them. On the banked set the two agree to within 1.6% on every metric;
    they must not diverge here either, or the app and the runner would disagree
    about the same file."""
    frames = np.concatenate([monochrome_frames(), murky_frames(), monochrome_frames()])
    full = outputs.measure_structure(frames)
    sampled = outputs.measure_structure(frames, sample=outputs.STRUCTURE_SAMPLE_FRAMES)
    assert sampled.sampled == outputs.STRUCTURE_SAMPLE_FRAMES
    assert full.sampled == len(frames)
    assert sampled.structure_ratio == pytest.approx(full.structure_ratio, rel=0.05)
    assert sampled.block_std == pytest.approx(full.block_std, rel=0.25)


def test_sampling_spans_the_whole_stack_rather_than_its_head():
    """Evenly spaced, endpoints included. Measuring the first 16 frames of a 121
    frame video would miss a decode that degrades partway through."""
    indices = outputs._sampled_indices(121, 16)
    assert indices[0] == 0 and indices[-1] == 120
    assert len(indices) == 16


def test_a_short_stack_is_measured_whole():
    assert outputs._sampled_indices(9, 16) == list(range(9))


def test_the_aggregate_is_a_median_so_one_odd_frame_cannot_swing_it():
    """A fade to black, a cut, a single dropped frame: the mean would drag the
    whole shot toward a warning, and a single accidentally-structured frame
    would rescue a stack of static. Neither may happen."""
    healthy = monochrome_frames()
    spoiled = np.concatenate([healthy, np.zeros((2, SIZE, SIZE, 3), dtype=np.uint8)])
    measured, failure, warnings = verdict(spoiled)
    assert failure is None and warnings == ()
    assert measured.flat_frames == 2
    assert measured.structure_ratio == pytest.approx(
        outputs.measure_structure(healthy).structure_ratio, rel=1e-6
    )


# ---------------------------------------------------------------------------
# the app: WARN, never refuse
# ---------------------------------------------------------------------------


class StaticPipe:
    """A pipeline that generates colour static and can report a sampling rate."""

    def __init__(self):
        self.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=SAMPLING_RATE))

    def __call__(self, **kwargs):
        return StaticOutput(audio=False)


class StaticOutput:
    """A pipeline output that is colour static, and stereo audio to go with it."""

    def __init__(self, frames=None, audio=True):
        self.frames = structureless_frames() if frames is None else frames
        self.audio = np.zeros((2, SAMPLING_RATE // 10), dtype=np.float32) if audio else None


def test_the_writer_emits_a_structureless_artifact_and_says_so(tmp_path):
    """**The app must never refuse on this.** The user has already paid 35+
    GPU-seconds for these frames, and the detector is a heuristic that is wrong
    about r4a-class output by design. Emit the file, attach the note, let a
    human look."""
    notes: list[str] = []
    written = outputs.write_output(
        StaticOutput(audio=False),
        run_dir=tmp_path / "run",
        mode="text_to_video",
        exports=("video",),
        seed=1,
        frame_rate=24,
        warnings=notes,
    )
    artifact = written["video"]
    assert artifact.exists() and artifact.stat().st_size > 0, "the file must survive the warning"
    assert any("may be structureless" in note for note in notes), notes
    assert any("not a verdict" in note for note in notes), notes


def test_the_writer_stays_silent_about_a_healthy_artifact(tmp_path):
    notes: list[str] = []
    outputs.write_output(
        StaticOutput(frames=monochrome_frames(), audio=False),
        run_dir=tmp_path / "run",
        mode="text_to_video",
        exports=("video",),
        seed=1,
        frame_rate=24,
        warnings=notes,
    )
    assert notes == []


def test_a_structure_warning_does_not_cost_the_run_its_audio_track(tmp_path):
    """A regression that would have been introduced by this very change.

    ``write_output`` used to infer "the audio was muxed" from "the notes list is
    empty" — true only while the dropped-audio note was the one thing that could
    ever be in there. A second, unrelated note would have reported a muxed
    track as absent, and ``generate._artifacts_for`` would have raised
    ``EmptyOutputError`` on a file that was perfectly fine.
    """
    notes: list[str] = []
    written = outputs.write_output(
        StaticOutput(),
        run_dir=tmp_path / "run",
        mode="text_to_audio_video",
        exports=("video", "audio"),
        seed=1,
        frame_rate=24,
        sampling_rate=SAMPLING_RATE,
        warnings=notes,
    )
    assert set(written) == {"video", "audio"}, f"the audio track went missing: {notes}"
    assert any("may be structureless" in note for note in notes)


def test_the_note_reaches_the_result_the_user_is_handed(tmp_path):
    """End to end through the production encoder. A warning that stops at the
    Space's logs is a warning the user never sees — which is most of what it
    was for."""
    result = generate_module.generate(
        mode="text_to_video",
        seed=7,
        width=SIZE,
        height=SIZE,
        pipeline_factory=lambda **kwargs: StaticPipe(),
        encoder=None,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    assert result.output_path.exists()
    assert any("may be structureless" in note for note in result.warnings), result.warnings
