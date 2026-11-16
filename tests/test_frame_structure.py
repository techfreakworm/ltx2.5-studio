"""Spec for the picture check: ``outputs.assess_picture`` and the notes every result carries.

Every finished video passes through a cheap check that tells the creator when the picture is
probably broken. It names three conditions, each from statistics measured on up to 16 evenly
spaced frames after resizing them to one analysis size (so the same picture gives the same answer
at any resolution):

* ``flat``: no picture at all, one even tone (the operator's grey failure);
* ``veiled``: a real picture under a grey veil: no blacks, no bright highlights, almost no
  colour (the audio-to-video path before its fix);
* ``static``: coloured speckle with no shape larger than a blob (the broken Space build).

The check is a hint, never a refusal: the file is always written, a good picture gets no note,
and the note says what was seen without quoting numbers.

Fixtures in ``tests/fixtures/structure/`` are real frames from local runs; their README lists the
sources. The only real colour-static frames are Space outputs that are not copied into the repo,
so static is synthesised below as random colour blobs and checked against what was measured on
those frames (correlation at an eighth of the frame at most 0.033, colourfulness at least 62.9).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ltx25_studio import generate as generate_module
from ltx25_studio import outputs

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "structure"
FRAMES = FIXTURES / "frames.json"
SAMPLING_RATE = 48000
STATIC_SIZE = 256

EXPECTED = {"good": (), "bad_grey": ("flat",), "bad_washed": ("veiled",)}


def fixture_entries():
    return {entry["name"]: entry for entry in json.loads(FRAMES.read_text())["fixtures"]}


def fixture_names():
    return sorted(fixture_entries())


def label_of(name):
    return name.rsplit("_", 1)[0]


def load(name):
    data = base64.b64decode(fixture_entries()[name]["jpeg_base64"])
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


def stack(frame, count=4):
    return np.repeat(frame[None], count, axis=0)


def blob_static(height, width, *, seed=0, token=32, frames=1):
    """Blobby colour static: random colours per 32-px block (the model's latent cell), a finer
    16-px layer on top, softened edges. No two blocks agree, so nothing spans the frame."""
    rng = np.random.default_rng(seed)
    kernel = np.ones(4, np.float32) / 4

    def soften(row):
        return np.convolve(row, kernel, mode="same")

    out = []
    for _ in range(frames):

        def layer(size):
            rows, cols = -(-height // size) + 1, -(-width // size) + 1
            grid = rng.random((rows, cols, 3)).astype(np.float32)
            big = np.repeat(np.repeat(grid, size, 0), size, 1)
            dy, dx = rng.integers(0, size, 2)
            return big[dy : dy + height, dx : dx + width]

        coarse, fine = layer(token), layer(token // 2)
        luma = 255 * (0.7 * coarse[..., 0] + 0.3 * fine[..., 0])
        hue = (coarse[..., 1:] - 0.5) * 2 * 255 * 0.22
        rgb = np.stack(
            [luma + hue[..., 0], luma - 0.5 * hue[..., 0] - 0.5 * hue[..., 1], luma + hue[..., 1]],
            axis=-1,
        )
        for axis in (0, 1):
            rgb = np.apply_along_axis(soften, axis, rgb)
        out.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return np.stack(out)


def resized(frame, long_side):
    height, width = frame.shape[:2]
    scale = long_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(frame).resize(size, Image.BICUBIC))


# ---------------------------------------------------------------------------
# the fixtures
# ---------------------------------------------------------------------------


def test_the_fixtures_are_small_and_each_one_is_documented():
    entries = fixture_entries()
    readme = (FIXTURES / "README.md").read_text()
    assert 0 < len(entries) <= 24
    assert FRAMES.stat().st_size <= 400 * 1024
    binary = {".jpg", ".jpeg", ".png", ".npy", ".npz", ".mp4"}
    assert not [path.name for path in FIXTURES.iterdir() if path.suffix.lower() in binary]
    for name, entry in entries.items():
        assert f"| {name} |" in readme, f"{name} is not in the fixtures README"
        assert entry["label"] == label_of(name) and entry["label"] in EXPECTED, name
        assert entry["source"], name
    assert {label_of(name) for name in entries} == set(EXPECTED)


@pytest.mark.parametrize("name", fixture_names())
def test_every_fixture_is_classified_as_labelled(name):
    check = outputs.assess_picture(stack(load(name)))
    assert check.conditions == EXPECTED[label_of(name)], check.measurements
    assert bool(check.notes) == bool(EXPECTED[label_of(name)])


# ---------------------------------------------------------------------------
# static, synthesised (see the module docstring)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("height,width", [(256, 256), (480, 704), (704, 1216), (960, 512)])
def test_blobby_colour_static_is_named_static(height, width):
    check = outputs.assess_picture(blob_static(height, width, frames=3))
    measured = check.measurements
    assert measured.long_range_correlation < 0.06, (
        "the stand-in must decorrelate like the real frames"
    )
    assert 45 < measured.colourfulness < 100, "and be about as colourful as them"
    assert check.conditions == ("static",)


def test_pixel_noise_is_static_too():
    noise = np.random.default_rng(11).integers(0, 256, (3, 128, 128, 3), dtype=np.uint8)
    assert outputs.assess_picture(noise).conditions == ("static",)


def test_grey_speckle_is_not_called_coloured():
    """Snow, gravel, a starfield: texture with no colour is not the coloured mush."""
    frames = blob_static(256, 256, frames=2)
    grey = np.repeat(frames.mean(axis=-1, keepdims=True), 3, axis=-1).astype(np.uint8)
    assert outputs.assess_picture(grey).conditions == ()


# ---------------------------------------------------------------------------
# the same picture at any resolution gives the same answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["good_7", "good_9", "good_10", "bad_grey_1", "bad_washed_2"])
def test_a_fixture_at_1024_px_measures_like_it_does_at_256(name):
    small = load(name)
    large = resized(small, 1024)
    assert max(large.shape[:2]) == 1024
    at_256 = outputs.assess_picture(stack(small))
    at_1024 = outputs.assess_picture(stack(large))
    assert at_1024.conditions == at_256.conditions
    # Within 1 % of the 0..255 scale or 5 %: bicubic enlargement rings a little at edges.
    for field in ("luma_std", "black_level", "white_level", "colourfulness"):
        assert getattr(at_1024.measurements, field) == pytest.approx(
            getattr(at_256.measurements, field), rel=0.05, abs=2.5
        ), field
    assert at_1024.measurements.long_range_correlation == pytest.approx(
        at_256.measurements.long_range_correlation, abs=0.03
    )


def test_static_at_256_px_and_its_1024_px_enlargement_are_both_static():
    small = blob_static(256, 256)[0]
    large = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)
    assert outputs.assess_picture(stack(small)).conditions == ("static",)
    assert outputs.assess_picture(stack(large)).conditions == ("static",)


# ---------------------------------------------------------------------------
# legitimate pictures that sit near a condition get no note
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["good_9", "good_10", "good_6"])
def test_a_dark_contrasty_frame_yields_no_note(name):
    frame = load(name)
    check = outputs.assess_picture(stack(frame))
    assert float(frame.mean()) < 90, "the fixture really is dark"
    assert check.notes == () and check.conditions == ()


@pytest.mark.parametrize("name", ["good_7", "good_8"])
def test_a_low_texture_sky_and_water_frame_yields_no_note(name):
    check = outputs.assess_picture(stack(load(name)))
    assert check.notes == () and check.conditions == ()


def test_a_dark_low_key_picture_is_not_called_washed_out():
    """The dark portrait at 45 % brightness has no highlights and almost no colour, like a
    veiled frame; what it lacks is the veil's lifted blacks."""
    frame = (load("good_10").astype(np.float32) * 0.45).astype(np.uint8)
    check = outputs.assess_picture(stack(frame))
    measured = check.measurements
    assert measured.white_level < outputs.VEILED_MAX_WHITE_LEVEL
    assert measured.colourfulness < outputs.VEILED_MAX_COLOURFULNESS
    assert measured.black_level < outputs.VEILED_MIN_BLACK_LEVEL
    assert check.conditions == ()


@pytest.mark.parametrize("name", ["good_12", "good_13"])
def test_a_bright_overcast_picture_is_not_called_washed_out(name):
    """The overcast beach is low in contrast, its blacks are lifted and it has little colour,
    like a veiled frame; what it lacks is the veil's dimmed highlights."""
    check = outputs.assess_picture(stack(load(name)))
    measured = check.measurements
    assert measured.black_level >= outputs.VEILED_MIN_BLACK_LEVEL
    assert measured.colourfulness < outputs.VEILED_MAX_COLOURFULNESS
    assert measured.white_level >= outputs.VEILED_MAX_WHITE_LEVEL
    assert check.conditions == () and check.notes == ()


def test_a_dark_black_and_white_night_with_true_blacks_yields_no_note():
    """Dim highlights and no colour, like a veiled frame, and a narrow brightness spread; what
    it has that a veil takes away is true blacks."""
    check = outputs.assess_picture(stack(load("good_15")))
    measured = check.measurements
    assert measured.white_level < outputs.VEILED_MAX_WHITE_LEVEL
    assert measured.colourfulness < outputs.VEILED_MAX_COLOURFULNESS
    assert measured.black_level < outputs.VEILED_MIN_BLACK_LEVEL / 10
    assert measured.luma_std > outputs.FLAT_MAX_LUMA_STD * 2
    assert check.conditions == () and check.notes == ()


def test_a_soft_snowfall_scene_yields_no_note():
    """Soft and low in detail by content: kept out of every condition, not just one."""
    check = outputs.assess_picture(stack(load("good_16")))
    measured = check.measurements
    assert measured.black_level < outputs.VEILED_MIN_BLACK_LEVEL
    assert measured.white_level > outputs.VEILED_MAX_WHITE_LEVEL
    assert measured.long_range_correlation > outputs.STATIC_MAX_CORRELATION
    assert check.conditions == () and check.notes == ()


def test_a_black_and_white_picture_yields_no_note():
    real = load("good_14")
    frame = load("good_1")
    grey = np.repeat(frame.mean(axis=-1, keepdims=True), 3, axis=-1).astype(np.uint8)
    for picture in (real, grey):
        check = outputs.assess_picture(stack(picture))
        assert check.measurements.colourfulness < 5.0
        assert check.conditions == ()


def test_one_odd_frame_cannot_swing_the_answer():
    """A fade to black or a cut: the aggregate is a median over the sampled frames."""
    good = stack(load("good_2"), 8)
    spoiled = np.concatenate([good, np.zeros((2, *good.shape[1:]), dtype=np.uint8)])
    assert outputs.assess_picture(spoiled).conditions == ()
    static = blob_static(256, 256, frames=6)
    rescued = np.concatenate([static, resized(load("good_11"), 256)[None]])
    assert outputs.assess_picture(rescued).conditions == ("static",)


def test_a_flat_video_is_named_flat_however_it_drifts():
    frames = np.stack([np.full((64, 64, 3), 20 + index * 24, dtype=np.uint8) for index in range(9)])
    check = outputs.assess_picture(frames)
    assert check.conditions == ("flat",)
    assert check.measurements.long_range_correlation is None, "no variation, nothing to correlate"


def test_an_empty_stack_measures_nothing_and_says_nothing():
    check = outputs.assess_picture(np.zeros((0, 64, 64, 3), dtype=np.uint8))
    assert check.measurements.frames == 0 and check.measurements.sampled == 0
    assert check.measurements.luma_std is None
    assert check.conditions == () and check.notes == ()


# ---------------------------------------------------------------------------
# the notes
# ---------------------------------------------------------------------------

FORBIDDEN = (
    "luma",
    "band",
    "std",
    "correlation",
    "colourfulness",
    "ratio",
    "threshold",
    "percentile",
    "p99",
    "p1",
    "banked",
    "artifact",
    "dataset",
    "statistic",
    "structure",
    "measured",
    "median",
    "failed",
    "failure",
    "error",
    "broken",
)


def every_note():
    notes = {condition.name: condition.note for condition in outputs.PICTURE_CONDITIONS}
    assert set(notes) == {"flat", "veiled", "static"}
    return notes


@pytest.mark.parametrize("name", ["flat", "veiled", "static"])
def test_a_note_says_what_was_seen_and_that_it_is_a_hint(name):
    note = every_note()[name]
    lowered = note.lower()
    assert not re.search(r"\d", note), f"no numbers or ranges: {note}"
    for word in FORBIDDEN:
        assert not re.search(rf"\b{re.escape(word)}\b", lowered), f"{word!r} in {note}"
    assert "hint" in lowered and "saved" in lowered
    assert "if that is not what you asked for" in lowered


def test_each_condition_says_something_different():
    notes = every_note()
    assert "flat" in notes["flat"].lower()
    assert "contrast" in notes["veiled"].lower() and "colour" in notes["veiled"].lower()
    assert "speckle" in notes["static"].lower()


def test_the_notes_a_result_carries_are_the_checks_own():
    frame = load("bad_washed_1")
    check = outputs.assess_picture(stack(frame))
    assert outputs.structure_notes(check) == check.notes == (every_note()["veiled"],)
    measured = outputs.measure_structure(stack(frame))
    assert outputs.structure_notes(measured) == check.notes
    assert outputs.structure_notes(outputs.assess_picture(stack(load("good_3")))) == ()


# ---------------------------------------------------------------------------
# bounded and torch-free
# ---------------------------------------------------------------------------


class CountingFrames:
    """A 481-frame video that records which frames were read."""

    def __init__(self, frame, count):
        self.frame, self.count, self.read = frame, count, []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.read.append(index)
        return self.frame


def test_a_481_frame_video_is_judged_on_at_most_16_frames():
    frames = CountingFrames(load("good_4"), 481)
    check = outputs.assess_picture(frames)
    assert len(set(frames.read)) <= outputs.STRUCTURE_SAMPLE_FRAMES == 16
    assert 0 in frames.read and 480 in frames.read, "evenly spaced, both ends"
    assert check.measurements.frames == 481 and check.measurements.sampled == 16
    frames.read.clear()
    outputs.measure_structure(frames, sample=None)
    assert len(set(frames.read)) <= 16, "the older name is bounded the same way"


def test_a_large_frame_is_measured_at_the_analysis_size():
    frame = resized(load("good_2"), 1920)
    check = outputs.assess_picture(stack(frame, 2))
    assert outputs.STRUCTURE_ANALYSIS_SIZE == 256
    assert check.conditions == ()


def test_the_check_needs_neither_torch_nor_a_media_stack():
    """numpy only. (PIL is already loaded by gradio through ``errors``; the call adds nothing.)"""
    code = (
        "import sys, numpy\n"
        "from ltx25_studio import outputs\n"
        "def roots(): return {m.split('.')[0] for m in sys.modules}\n"
        "before = roots()\n"
        "outputs.assess_picture(numpy.zeros((2, 64, 64, 3), numpy.uint8))\n"
        "heavy = {'torch', 'diffusers', 'transformers', 'av', 'PIL', 'cv2', 'scipy'}\n"
        "print(sorted((roots() - before) & heavy), sorted(roots() & {'torch', 'diffusers', 'av'}))\n"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[] []", result.stdout + result.stderr


# ---------------------------------------------------------------------------
# thresholds, and the numbers that put them there
# ---------------------------------------------------------------------------

# Measured by this check on the labelled dataset (42 files: 27 good, 3 grey, 9 veiled, 3 static).
WORST_GOOD_LUMA_STD = 14.93
WORST_GREY_LUMA_STD = 4.30
WORST_VEILED_BLACK_LEVEL = 57.6
WORST_VEILED_WHITE_LEVEL = 115.1
# veiled needs both ends of the range pulled in, so a good file is kept out by either end:
# good files with lifted blacks (p1 >= 40) all keep highlights of at least this (the beach),
GOOD_LIFTED_BLACKS_MIN_WHITE_LEVEL = 176.8
# and good files with dim highlights (p99 < 135) all keep blacks of at most this (the night).
GOOD_DIM_HIGHLIGHTS_MAX_BLACK_LEVEL = 1.0
WORST_VEILED_COLOURFULNESS = 13.8
WORST_GOOD_CORRELATION = 0.188
WORST_STATIC_CORRELATION = 0.033
WORST_STATIC_COLOURFULNESS = 62.9


def test_the_thresholds_sit_between_the_measured_classes_with_room_either_side():
    assert WORST_GREY_LUMA_STD * 1.3 < outputs.FLAT_MAX_LUMA_STD < WORST_GOOD_LUMA_STD / 2.4
    assert outputs.VEILED_MIN_BLACK_LEVEL * 1.4 < WORST_VEILED_BLACK_LEVEL
    assert GOOD_DIM_HIGHLIGHTS_MAX_BLACK_LEVEL * 10 < outputs.VEILED_MIN_BLACK_LEVEL
    assert WORST_VEILED_WHITE_LEVEL * 1.15 < outputs.VEILED_MAX_WHITE_LEVEL
    assert outputs.VEILED_MAX_WHITE_LEVEL * 1.3 < GOOD_LIFTED_BLACKS_MIN_WHITE_LEVEL
    assert WORST_VEILED_COLOURFULNESS * 1.4 < outputs.VEILED_MAX_COLOURFULNESS
    assert WORST_STATIC_CORRELATION + 0.05 < outputs.STATIC_MAX_CORRELATION
    assert outputs.STATIC_MAX_CORRELATION + 0.05 < WORST_GOOD_CORRELATION
    assert outputs.STATIC_MIN_COLOURFULNESS * 1.8 < WORST_STATIC_COLOURFULNESS


def test_the_older_names_still_resolve():
    for name in (
        "MIN_STRUCTURE_RATIO",
        "MAX_HIGH_FREQUENCY_RATIO",
        "LOW_BLOCK_STD",
        "LOW_LUMA_BAND",
    ):
        assert isinstance(getattr(outputs, name), float)
    static = outputs.measure_structure(blob_static(256, 256, frames=2))
    assert "static" in outputs.structure_failure(static)
    assert outputs.structure_warnings(static) == outputs.structure_notes(static)
    good = outputs.measure_structure(stack(load("good_5")))
    assert outputs.structure_failure(good) is None
    assert good.structure_ratio is not None and good.high_frequency_ratio is not None


# ---------------------------------------------------------------------------
# the app: a note, never a refusal
# ---------------------------------------------------------------------------


class StaticPipe:
    """A pipeline that generates colour static and can report a sampling rate."""

    def __init__(self):
        self.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=SAMPLING_RATE))

    def __call__(self, **kwargs):
        return StaticOutput(audio=False)


class StaticOutput:
    """A pipeline output that is colour static, with stereo audio to go with it."""

    def __init__(self, frames=None, audio=True):
        self.frames = blob_static(STATIC_SIZE, STATIC_SIZE, frames=9) if frames is None else frames
        self.audio = np.zeros((2, SAMPLING_RATE // 10), dtype=np.float32) if audio else None


def test_the_writer_saves_a_static_video_and_says_what_it_saw(tmp_path):
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
    assert artifact.exists() and artifact.stat().st_size > 0, "the file must survive the note"
    assert notes == [every_note()["static"]]


def test_the_writer_stays_silent_about_a_good_video(tmp_path):
    notes: list[str] = []
    outputs.write_output(
        StaticOutput(frames=stack(load("good_9")[:160, :256], 9), audio=False),
        run_dir=tmp_path / "run",
        mode="text_to_video",
        exports=("video",),
        seed=1,
        frame_rate=24,
        warnings=notes,
    )
    assert notes == []


def test_a_note_does_not_cost_the_run_its_audio_track(tmp_path):
    """``write_output`` once inferred "the audio was muxed" from an empty notes list."""
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
    assert every_note()["static"] in notes


def test_the_note_reaches_the_result_the_user_is_handed(tmp_path):
    from conftest import inputs_for

    result = generate_module.generate(
        mode="text_to_video",
        seed=7,
        **inputs_for("text_to_video"),
        width=STATIC_SIZE,
        height=STATIC_SIZE,
        pipeline_factory=lambda **kwargs: StaticPipe(),
        encoder=None,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    assert result.output_path.exists()
    assert every_note()["static"] in result.warnings, result.warnings
