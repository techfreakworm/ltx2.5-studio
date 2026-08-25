"""Spec for ``scripts/space_matrix.py`` — the live-Space runner, with no Space.

Every test here runs offline: a fake client stands in for ``gradio_client``, and
the media the content checks read are **real containers built with ``av``**, the
same way ``tests/test_verify_streams.py`` does it. Faking the probe would test
the mock; building the file tests the checker.

Four properties are what this file exists for, and each of them is a defect this
project has already paid for once:

**The ceiling refuses rather than reports.** A budget guard that notices
afterwards is an accounting entry, not a guard. The refusal has to happen with
the run unstarted, and the proof is that the fake client was never called.

**A run that produced nothing raises.** ``ui/handlers.on_generate`` turns every
failure into ``(None, None, <markdown>)`` — a success-SHAPED triple. There is no
exception to catch and no falsy return to notice; the only thing that can tell
the difference is a check against what the registry says the mode exports.

**The content checks reject in both directions.** A checker that only ever
passes is indistinguishable from no checker at all, so every rejection test has
an acceptance test beside it. That symmetry is also the guard against the
*inverse* error — this build has reported a false defect from a bad fixture, and
a threshold tuned so tight that a healthy artifact fails is the same mistake
wearing the opposite sign.

**Dry run spends nothing.** Asserted by making ``connect`` explode: if the code
path that dry-runs can reach the network at all, the test fails loudly instead
of quietly costing GPU-seconds.
"""

from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import space_matrix

from ltx25_studio import modes
from ltx25_studio.ui import fields

av = pytest.importorskip("av")


# ---------------------------------------------------------------------------
# real banked artifacts
#
# A synthesised near-static fixture proves the checker is self-consistent. It
# cannot prove the thresholds are right about REAL output, and getting that
# wrong is what put a false defect report in this project twice. So the tests
# that pin the temporal policy read the artifacts the Space actually returned —
# tens of megabytes of mp4, kept out of the repo — and SKIP when they are not
# there. A box that has never seen this build's output cannot honestly assert
# on it in either direction, and a test that quietly passes without its evidence
# is worse than one that says it was not run.
# ---------------------------------------------------------------------------

BANKED = Path(
    os.environ.get(
        "LTX25_BANKED_ARTIFACTS",
        "/private/tmp/claude-502/-Users-techfreakworm--soma-worktrees-ltx25-studio-ltx25-studio"
        "/e132fe51-c395-433b-bf41-2a9e53c8c0cc/scratchpad",
    )
)

#: ``image_to_audio_video``, 512x960x121: a red balloon over a lake at dawn,
#: structure 0.9974, high-freq 0.0559, block std 81.6, luma band 238.0 — and
#: mean |Δ| 0.4071, for which the old gate destroyed it. Near-static because it
#: is conditioned on a still image of that scene, which is the mode working.
FALSELY_REFUSED = "m2/13013c1d91469a64a629b4a91ef03a165b7071818375e2dc63c28125ff57bda0/output.mp4"

#: The same scene, a sibling run, at mean |Δ| 0.3859 — BELOW the 0.403 the old
#: threshold was calibrated against. The two populations overlap.
STILLER_SIBLING = "m2/e9ff3b66ebb44b7ffb68e438dac00133dc2ef72b2d1127449e6a8c0d46c7a82e/output.mp4"


def banked(relative: str) -> Path:
    """A real generated artifact, or a skip that names what is missing."""
    path = BANKED / relative
    if not path.is_file():
        pytest.skip(f"banked artifact not present: {path}")
    return path


# ---------------------------------------------------------------------------
# real media, built here
# ---------------------------------------------------------------------------

FIXTURE_SIZE = 64
FIXTURE_RATE = 24


def _frame(index: int, *, moving: bool, blank: bool) -> np.ndarray:
    """A bright block on a dark field, at a position that depends on the index.

    A block rather than per-pixel noise on purpose: h264 discards noise and
    keeps structure, so a noise-based fixture can decode back flatter than it
    went in and fail a motion check for reasons that have nothing to do with the
    checker.
    """
    canvas = np.zeros((FIXTURE_SIZE, FIXTURE_SIZE, 3), dtype=np.uint8)
    if blank:
        return canvas
    offset = (index * 8) % (FIXTURE_SIZE - 20) if moving else 0
    canvas[10:30, offset : offset + 20] = 255
    return canvas


def write_video(path, *, frames=9, moving=True, blank=False, with_audio=False, silent=False):
    """A real mp4 with exactly the streams and motion asked for."""
    with av.open(str(path), "w") as container:
        video = container.add_stream("h264", rate=FIXTURE_RATE)
        video.width = video.height = FIXTURE_SIZE
        video.pix_fmt = "yuv420p"
        audio = None
        if with_audio:
            audio = container.add_stream("aac", rate=48000, layout="stereo")
            audio.time_base = Fraction(1, 48000)

        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(
                _frame(index, moving=moving, blank=blank), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, FIXTURE_RATE)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        if audio is not None:
            for index in range(8):
                for packet in audio.encode(_audio_frame(index, silent=silent, rate=48000)):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return Path(path)


def write_frames(path, stack, *, rate=FIXTURE_RATE):
    """A real h264 mp4 carrying exactly the ``[F, H, W, 3]`` uint8 stack given.

    ``write_video`` renders one shape of picture; the structure checks need
    several, including one that is deliberately not a picture at all. h264 at
    64x64 preserves per-pixel noise well enough that a static fixture decodes
    back static — verified by the measurements the tests below assert on, which
    are taken from the DECODED file rather than the array that went in.
    """
    stack = np.ascontiguousarray(stack.astype(np.uint8))
    with av.open(str(path), "w") as container:
        video = container.add_stream("h264", rate=rate)
        video.height, video.width = int(stack.shape[1]), int(stack.shape[2])
        video.pix_fmt = "yuv420p"
        for index, frame in enumerate(stack):
            picture = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            picture.pts, picture.time_base = index, Fraction(1, rate)
            for packet in video.encode(picture):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
    return Path(path)


def structureless_stack(frames=9, seed=11):
    """Colour static: distinct, full-range, moving, and empty of any picture."""
    return np.random.default_rng(seed).integers(
        0, 256, (frames, FIXTURE_SIZE, FIXTURE_SIZE, 3), dtype=np.uint8
    )


def _smooth_base():
    rows, columns = np.mgrid[0:FIXTURE_SIZE, 0:FIXTURE_SIZE].astype(np.float32)
    return 40 + 90 * np.sin(columns / 11.0) * np.cos(rows / 17.0) + 60 * (rows / FIXTURE_SIZE)


def near_static_stack(frames=9):
    """``r7b``-class: structured, barely moving. The artifact that disqualifies
    temporal difference as the signal — the banked one sits at mean |Δ| 0.824
    against a 0.5 threshold, a factor of 2.0 from the thing being rejected."""
    base = _smooth_base()
    planes = np.stack([base + index * 1.4 for index in range(frames)])
    return np.repeat(np.clip(planes, 0, 255)[..., None], 3, axis=-1).astype(np.uint8)


def barely_moving_stack(frames=9, step=0.4, scale=1.0):
    """The artifact class the old temporal gate destroyed: structured, coherent,
    and drifting far too slowly to satisfy an absolute ``mean |Δ|`` threshold.

    Decodes at mean |Δ| 0.45 — the same neighbourhood as the real
    ``image_to_audio_video`` render refused at 0.4071 — with structure 0.97 and
    a freeze ratio of 0.0093, nine times clear of the floor. It must WARN and
    pass.

    ``scale`` dims the whole stack without changing the scene or its motion.
    That is the axis an absolute threshold cannot survive: at ``scale=0.35`` the
    measured |Δ| falls six-fold purely because the picture got darker.
    """
    base = _smooth_base() * scale
    planes = np.stack([base + index * step * scale for index in range(frames)])
    return np.repeat(np.clip(planes, 0, 255)[..., None], 3, axis=-1).astype(np.uint8)


def frozen_stack(frames=9):
    """One frame, N times — a real freeze, not a still scene.

    Distinct from :func:`barely_moving_stack` by construction rather than by
    degree: nothing changes between these frames at all, and what survives h264
    is codec noise. That is precisely the case byte identity misses.
    """
    return np.repeat(near_static_stack(1), frames, axis=0)


def murky_stack(frames=9):
    """``r4a``-class: a real render with almost no contrast. WARN, never FAIL."""
    rows, columns = np.mgrid[0:FIXTURE_SIZE, 0:FIXTURE_SIZE].astype(np.float32)
    base = 120 + 6 * np.sin(columns / 9.0) * np.cos(rows / 13.0)
    planes = np.stack([base + 3 * np.sin((columns + index * 5) / 9.0) for index in range(frames)])
    return np.repeat(np.clip(planes, 0, 255)[..., None], 3, axis=-1).astype(np.uint8)


def drifting_flat_stack(frames=9):
    """Each frame one flat colour, brightness ramping across the stack."""
    return np.stack(
        [
            np.full((FIXTURE_SIZE, FIXTURE_SIZE, 3), 20 + index * 24, dtype=np.uint8)
            for index in range(frames)
        ]
    )


def _audio_frame(index: int, *, silent: bool, rate: int, channels: int = 2):
    samples = 1024
    layout = "stereo" if channels == 2 else "mono"
    if silent:
        block = np.zeros((channels, samples), dtype="float32")
    else:
        t = (np.arange(samples) + index * samples) / float(rate)
        tone = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype("float32")
        block = np.stack([tone] * channels)
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(block), format="fltp", layout=layout)
    frame.rate, frame.pts = rate, index * samples
    frame.time_base = Fraction(1, rate)
    return frame


def write_wav(path, *, silent=False, rate=48000, channels=2, blocks=16):
    """A real pcm_s16le wav — the shape ``outputs.write_output`` emits for audio-only modes."""
    with av.open(str(path), "w", format="wav") as container:
        layout = "stereo" if channels == 2 else "mono"
        stream = container.add_stream("pcm_s16le", rate=rate, layout=layout)
        for index in range(blocks):
            for packet in stream.encode(
                _audio_frame(index, silent=silent, rate=rate, channels=channels)
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return Path(path)


# ---------------------------------------------------------------------------
# the fake client and a fake clock
# ---------------------------------------------------------------------------


class FakeClient:
    """Stands in for ``gradio_client.Client``. No network, no GPU, no quota.

    Records every call so a test can assert on the POSITIONS arguments landed in
    — which is the only way to catch a value shifted one slot left, the defect
    ``fields.py`` and ``API_FIELD_KEYS`` both exist to prevent.
    """

    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[dict] = []

    def predict(self, *args, api_name=None):
        self.calls.append({"args": args, "api_name": api_name})
        if not self.results:
            raise AssertionError("the runner called predict more times than the test planned for")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def arg(self, key: str, call: int = 0):
        """One call's value for a named UI field, resolved through the real ordering."""
        return self.calls[call]["args"][1 + space_matrix.API_FIELD_KEYS.index(key)]


class FakeClock:
    """``time.time`` that advances a fixed step per read, so wall clock is exact."""

    def __init__(self, step: float = 0.0):
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


@pytest.fixture
def clock(monkeypatch):
    """Install a deterministic clock into the module's own namespace.

    Patched on ``space_matrix.time`` rather than on the ``time`` module, so the
    substitution cannot leak into anything else the suite is doing.
    """

    def install(step: float):
        fake = FakeClock(step)
        monkeypatch.setattr(space_matrix, "time", SimpleNamespace(time=fake))
        return fake

    return install


@pytest.fixture
def quiet_log():
    lines: list[str] = []
    return lines.append, lines


@pytest.fixture(autouse=True)
def isolated_quota_ledger(tmp_path, monkeypatch):
    """Point the default quota ledger at a throwaway file, for EVERY test here.

    Autouse rather than opt-in: the ledger's default path is a real file in the
    repo that records the operator's actual spend for the day. A test that reads
    it is merely flaky; a test that writes it corrupts the one number standing
    between the next run and the daily budget. Nothing in this file should be one
    forgotten argument away from that.
    """
    path = tmp_path / "quota-ledger.json"
    monkeypatch.setattr(space_matrix, "DEFAULT_QUOTA_LEDGER", str(path))
    return path


# ---------------------------------------------------------------------------
# the endpoint's argument order is DERIVED, not restated
# ---------------------------------------------------------------------------


def test_the_argument_order_is_the_ui_field_order_minus_the_state():
    """``gr.State`` has no ``api_info``, so a client never sees it.

    16 parameters after ``mode_id``, not 17. Getting this wrong shifts nothing
    (``loras`` is last today) — but it would the moment a field is added above
    it, and then every value lands one slot left with no error anywhere.
    """
    without_the_state = tuple(key for key in fields.FIELD_KEYS if key != "loras")
    assert without_the_state == space_matrix.API_FIELD_KEYS
    assert "loras" not in space_matrix.API_FIELD_KEYS
    assert len(space_matrix.API_FIELD_KEYS) == len(fields.FIELD_KEYS) - 1


def test_the_argument_order_matches_the_endpoint_gradio_actually_publishes():
    """Built headlessly and compared against the real API signature.

    This is the check that spans the client boundary: ``fields.py`` guarantees
    the UI's two orderings agree with each other, and nothing but this
    guarantees ours agrees with *theirs*.
    """
    pytest.importorskip("gradio")
    from ltx25_studio.ui import build_ui

    endpoint = build_ui().get_api_info()["named_endpoints"][space_matrix.ENDPOINT]
    parameters = endpoint["parameters"]

    assert len(parameters) == 1 + len(space_matrix.API_FIELD_KEYS)
    assert parameters[0]["component"] == "Radio"  # mode_id

    # `text` is the only kind whose component name is not just its own name.
    component_for_kind = {"text": "textbox"}
    for index, key in enumerate(space_matrix.API_FIELD_KEYS, start=1):
        kind = fields.field_by_key(key).kind
        expected = component_for_kind.get(kind, kind)
        actual = parameters[index]["component"].lower()
        assert actual == expected, (
            f"argument {index} is a {actual}, but API_FIELD_KEYS puts {key!r} "
            f"(a {kind}) there — the client's ordering has drifted from the UI's"
        )

    assert [r["component"] for r in endpoint["returns"]] == ["Video", "Audio", "Markdown"]


def test_predict_args_are_read_out_through_the_key_order(tmp_path):
    """Never appended positionally: a dict, read through ``API_FIELD_KEYS``."""
    config = space_matrix.RunConfig(mode_id="text_to_video", seed=77, prompt="a lighthouse")
    args = space_matrix.predict_args(config)

    assert args[0] == "text_to_video"
    assert len(args) == 1 + len(space_matrix.API_FIELD_KEYS)
    named = dict(zip(space_matrix.API_FIELD_KEYS, args[1:], strict=True))
    assert named["prompt"] == "a lighthouse"
    assert named["seed"] == 77
    assert named["width"] == named["height"] == space_matrix.CHEAPEST_SIZE
    assert named["num_frames"] == space_matrix.CHEAPEST_FRAMES
    assert named["image"] is named["video"] is named["audio"] is None
    # The Dataframe's headers come from the same tuple the component is built
    # from, so a renamed column cannot leave us sending a table it refuses.
    assert named["conditions"]["headers"] == list(fields.SHOT_HEADERS)
    assert named["conditions"]["data"] == []


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


def test_the_default_order_covers_every_registered_mode():
    assert set(space_matrix.DEFAULT_MODE_ORDER) == set(modes.MODES)
    assert len(space_matrix.DEFAULT_MODE_ORDER) == len(modes.MODES)


def test_the_default_order_is_a_dependency_order():
    """Every consumer runs after something that produced its input.

    Not decoration: a plan that puts ``image_to_video`` first spends nothing and
    fails immediately, and a plan that puts it second-to-last spends the whole
    budget before discovering the fixture never arrived.
    """
    available: set[str] = set()
    for mode_id in space_matrix.DEFAULT_MODE_ORDER:
        spec = modes.get_mode(mode_id)
        needs = set(spec.inputs) & fields.MEDIA_KINDS
        assert needs <= available, (
            f"{mode_id} needs {sorted(needs - available)} and nothing before it produces that"
        )
        if "video" in spec.exports:
            available |= {"video", "image"}  # the image fixture is a frame of the video
        if "audio" in spec.exports:
            available.add("audio")


def test_a_named_subset_runs_in_the_order_given():
    plan = space_matrix.build_plan(["video_to_audio", "text_to_video"])
    assert [c.mode_id for c in plan] == ["video_to_audio", "text_to_video"]


def test_an_unknown_mode_names_what_is_registered():
    with pytest.raises(Exception) as caught:
        space_matrix.build_plan(["text_to_hologram"])
    assert "text_to_video" in str(caught.value)


def test_the_cheapest_config_passes_the_real_validators():
    """Uses ``constraints``, not a re-derived modulus."""
    for config in space_matrix.build_plan():
        space_matrix.validate_config(config)
    assert space_matrix.CHEAPEST_SIZE % constraints_multiple() == 0
    assert (space_matrix.CHEAPEST_FRAMES - 1) % 8 == 0


def constraints_multiple() -> int:
    from ltx25_studio import constraints

    return constraints.DIMENSION_MULTIPLE


def test_an_illegal_geometry_is_refused_offline():
    from ltx25_studio.errors import DimensionError, FrameCountError

    with pytest.raises(DimensionError):
        space_matrix.validate_config(space_matrix.RunConfig(mode_id="text_to_video", width=250))
    with pytest.raises(FrameCountError):
        space_matrix.validate_config(space_matrix.RunConfig(mode_id="text_to_video", num_frames=10))


def test_the_registry_overrides_are_reflected_in_the_plan():
    """``min_resolution`` and ``skip_stage_2`` win over the request, so the plan
    must report what will RUN and not what was asked for."""
    audio_only = space_matrix.RunConfig(
        mode_id="text_to_audio", width=704, height=480, enable_stage_2=True
    )
    assert audio_only.effective_width == modes.MIN_RESOLUTION
    assert audio_only.effective_height == modes.MIN_RESOLUTION
    assert audio_only.effective_stage_2 is False

    plain = space_matrix.RunConfig(mode_id="text_to_video", width=704, enable_stage_2=True)
    assert plain.effective_width == 704
    assert plain.effective_stage_2 is True


# ---------------------------------------------------------------------------
# the quota ceiling — it REFUSES, it does not report
# ---------------------------------------------------------------------------


def test_the_default_ceiling_is_eighty_percent_of_the_daily_budget():
    assert space_matrix.DAILY_QUOTA_SECONDS == 2400.0
    assert space_matrix.DEFAULT_QUOTA_CEILING_SECONDS == 1920.0
    assert space_matrix.BILLING_SIZE_MULTIPLIER == 2.0


def test_the_cost_of_a_run_is_wall_clock_times_the_size_multiplier():
    record = space_matrix.RunRecord(
        config=space_matrix.RunConfig(mode_id="text_to_video"), wall_seconds=21.44, artifacts={}
    )
    assert record.gpu_seconds_upper_bound == pytest.approx(42.88)
    rendered = record.render_cost()
    assert "<=" in rendered
    assert "UPPER BOUND" in rendered
    assert "NOT MEASURED BILLING" in rendered


def test_every_cost_figure_is_labelled_an_upper_bound():
    """A bound that loses its label becomes a measurement the moment it is quoted."""
    ledger = space_matrix.Ledger(spent_upper_bound=243.58)
    assert "UPPER BOUND" in ledger.render()
    assert "<=" in ledger.render()

    plan = space_matrix.render_plan(space_matrix.build_plan(["text_to_video"]), ledger, live=False)
    assert "UPPER BOUND" in plan
    assert "ZERO GPU-seconds" in plan


def test_the_ledger_refuses_a_run_that_would_cross_the_ceiling():
    ledger = space_matrix.Ledger(
        ceiling_seconds=150.0, reservation_seconds=50, spent_upper_bound=100.0
    )
    config = space_matrix.RunConfig(mode_id="text_to_video")
    assert ledger.projected_for(config) == pytest.approx(100.0)  # 50s reserved x 2
    assert ledger.would_cross(config) is True
    with pytest.raises(space_matrix.QuotaCeilingReached) as caught:
        ledger.check_before(config)
    assert "text_to_video" in str(caught.value)
    assert "150.0" in str(caught.value)


def test_the_ceiling_stops_the_matrix_before_the_run_is_ever_sent(tmp_path, clock, quiet_log):
    """The proof is the call count: the refused run was never sent to the Space.

    Never "borrow tomorrow's" — a guard that lets one more run through because it
    is nearly within budget is a guard with no threshold at all.
    """
    log, _lines = quiet_log
    clock(step=30.0)  # each predict measures exactly 30s -> 60 GPU-s charged
    video = write_video(tmp_path / "out.mp4")

    client = FakeClient(
        [str(video), None, "✅ Text to video complete."],
        [str(video), None, "✅ Text to video complete."],
    )
    ledger = space_matrix.Ledger(ceiling_seconds=150.0, reservation_seconds=50)
    outcome = space_matrix.run_matrix(
        client,
        space_matrix.build_plan(["text_to_video", "text_to_video"]),
        fixtures=space_matrix.FixtureStore(tmp_path / "fixtures"),
        ledger=ledger,
        log=log,
    )

    # Run 1: 0 + 100 projected <= 150, so it starts and charges 2 x 30s = 60.
    # Run 2: 60 + 100 = 160 > 150, so it is refused with nothing sent.
    assert len(client.calls) == 1
    assert len(outcome.records) == 1
    assert outcome.stopped is True
    assert outcome.stopped_at.mode_id == "text_to_video"
    assert ledger.spent_upper_bound == pytest.approx(60.0)
    assert ledger.spent_upper_bound < ledger.ceiling_seconds


def test_the_stop_summary_reports_rather_than_crashing(tmp_path, clock, quiet_log):
    log, lines = quiet_log
    clock(step=30.0)
    video = write_video(tmp_path / "out.mp4")
    client = FakeClient([str(video), None, "ok"], [str(video), None, "ok"])
    outcome = space_matrix.run_matrix(
        client,
        space_matrix.build_plan(["text_to_video", "text_to_video"]),
        fixtures=space_matrix.FixtureStore(tmp_path / "fixtures"),
        ledger=space_matrix.Ledger(ceiling_seconds=150.0, reservation_seconds=50),
        log=log,
    )
    assert any("STOPPED AT THE QUOTA CEILING" in line for line in lines)
    assert any("Nothing was spent on this run" in line for line in lines)
    summary = space_matrix.render_summary(outcome)
    assert "STOPPED before text_to_video" in summary
    assert "UPPER BOUND" in summary


def test_a_projection_that_undershoots_still_charges_the_observed_wall_clock(tmp_path, clock):
    """The platform re-forks a killed worker, so a run CAN cost more than one
    attempt. The ledger must accumulate what happened, not what was predicted."""
    clock(step=100.0)
    video = write_video(tmp_path / "out.mp4")
    ledger = space_matrix.Ledger(ceiling_seconds=10_000.0, reservation_seconds=50)
    record = space_matrix.run_one(
        FakeClient([str(video), None, "ok"]),
        space_matrix.RunConfig(mode_id="text_to_video"),
        ledger=ledger,
        log=lambda *a: None,
    )
    assert record.projected_gpu_seconds == pytest.approx(100.0)
    assert record.gpu_seconds_upper_bound == pytest.approx(200.0)
    assert ledger.spent_upper_bound == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# fail loudly — a success-shaped result carrying nothing
# ---------------------------------------------------------------------------

SPACE_FAILURE_TEXT = (
    "### ❌ Generation failed — unexpected `Error`\n\n"
    "> 'GPU task aborted'\n\n"
    "This error is not one of ours, which usually means it came from the pipeline "
    "or the platform. The full traceback is in the Space logs."
)


def test_a_run_with_no_artifact_raises_and_quotes_the_spaces_own_words(tmp_path, quiet_log):
    log, _lines = quiet_log
    client = FakeClient([None, None, SPACE_FAILURE_TEXT])
    with pytest.raises(space_matrix.SpaceRunError) as caught:
        space_matrix.run_one(client, space_matrix.RunConfig(mode_id="text_to_video"), log=log)
    message = str(caught.value)
    assert "text_to_video" in message
    assert "no video file" in message
    assert "'GPU task aborted'" in message, "the Space's own error text must survive verbatim"
    assert "success-SHAPED" in message


def test_the_missing_track_named_is_the_one_the_registry_declares(tmp_path, quiet_log):
    """``text_to_audio`` exports audio only — a returned video path is not a pass."""
    log, _lines = quiet_log
    video = write_video(tmp_path / "out.mp4")
    client = FakeClient([str(video), None, SPACE_FAILURE_TEXT])
    with pytest.raises(space_matrix.SpaceRunError) as caught:
        space_matrix.run_one(client, space_matrix.RunConfig(mode_id="text_to_audio"), log=log)
    assert "no audio file" in str(caught.value)
    assert "text_to_audio" in str(caught.value)


def test_an_empty_status_is_itself_reported_as_a_defect(quiet_log):
    log, _lines = quiet_log
    with pytest.raises(space_matrix.SpaceRunError) as caught:
        space_matrix.run_one(
            FakeClient([None, None, ""]), space_matrix.RunConfig(mode_id="text_to_video"), log=log
        )
    assert "NO status text at all" in str(caught.value)


def test_a_zero_byte_artifact_is_not_a_pass(tmp_path, quiet_log):
    """The inverse of the empty-output bug: a path that is real and a file that
    is not. Scoring it as a pass is the same class of mistake."""
    log, _lines = quiet_log
    empty = tmp_path / "output.mp4"
    empty.write_bytes(b"")
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.run_one(
            FakeClient([str(empty), None, "✅ complete"]),
            space_matrix.RunConfig(mode_id="text_to_video"),
            log=log,
        )
    assert "ZERO BYTES" in str(caught.value)
    assert "A path is not an artifact" in str(caught.value)


def test_a_client_exception_still_charges_the_ledger(tmp_path, clock, quiet_log):
    """A call that raised on our side may still have burned a slot on theirs."""
    log, _lines = quiet_log
    clock(step=15.0)
    ledger = space_matrix.Ledger(ceiling_seconds=10_000.0)
    with pytest.raises(space_matrix.SpaceRunError) as caught:
        space_matrix.run_one(
            FakeClient(TimeoutError("read timed out")),
            space_matrix.RunConfig(mode_id="text_to_video"),
            ledger=ledger,
            log=log,
        )
    assert "TimeoutError" in str(caught.value)
    assert ledger.spent_upper_bound == pytest.approx(30.0)


def test_an_unexpected_response_shape_is_refused(quiet_log):
    log, _lines = quiet_log
    with pytest.raises(space_matrix.SpaceRunError) as caught:
        space_matrix.run_one(
            FakeClient("output.mp4"), space_matrix.RunConfig(mode_id="text_to_video"), log=log
        )
    assert "not the (video, audio, status) triple" in str(caught.value)


def test_a_run_that_succeeds_is_not_reported_as_a_failure(tmp_path, quiet_log):
    """The other direction. A checker that only ever raises is not a checker."""
    log, _lines = quiet_log
    video = write_video(tmp_path / "out.mp4")
    record = space_matrix.run_one(
        FakeClient([str(video), None, "✅ Text to video complete. seed `1234`"]),
        space_matrix.RunConfig(mode_id="text_to_video"),
        log=log,
    )
    assert record.artifacts["video"][1] > 0
    assert record.video is not None
    assert record.video.frames == 9
    assert record.audio is None


# ---------------------------------------------------------------------------
# content, not existence — video
# ---------------------------------------------------------------------------


def test_a_moving_video_passes_and_reports_its_numbers(tmp_path):
    content = space_matrix.verify_video_content(
        write_video(tmp_path / "moving.mp4"), expected_frames=9
    )
    assert content.frames == 9
    assert content.distinct_frames >= space_matrix.MIN_DISTINCT_FRAMES
    assert content.spatial_std > space_matrix.MIN_PIXEL_STD
    assert content.mean_abs_delta > space_matrix.LOW_MEAN_FRAME_DELTA
    assert "distinct" in content.render()


def test_a_frozen_video_is_rejected(tmp_path):
    """Nine identical frames: correct codec, correct geometry, correct count,
    ``verify_streams`` exit 0 — and nothing moved."""
    frozen = write_video(tmp_path / "frozen.mp4", moving=False)
    assert verify_streams_ok("text_to_video", frozen), "the structural check passes on this file"
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(frozen, expected_frames=9)
    assert "one frame N times" in str(caught.value)


def test_byte_identity_alone_cannot_catch_a_frozen_render(tmp_path):
    """Why the freeze floor exists at all, and why deleting it is not safe.

    The obvious reading of ``MIN_DISTINCT_FRAMES`` is that it already covers
    frozen video: every frame identical, one distinct frame, refused. It does
    not. **h264 is lossy, so identical frames do not decode back identical** —
    the fixture above goes in as nine copies of one array and comes back as
    *eight distinct frames of nine*, which sails through the distinctness floor.

    So byte identity catches only the lossless case, and every artifact this
    tool ever sees is h264. Without a second, scale-free temporal refusal,
    demoting the old ``mean |Δ|`` gate would have left a dead video with nothing
    at all to fail: a frozen photograph has *perfect* spatial structure, so the
    structure gates cannot see it either.
    """
    from ltx25_studio import outputs

    frames = space_matrix._decode_video_frames(write_video(tmp_path / "frozen.mp4", moving=False))
    distinct = len({frame.tobytes() for frame in frames})

    assert distinct >= space_matrix.MIN_DISTINCT_FRAMES, (
        f"the distinctness floor PASSES this frozen file ({distinct} of {len(frames)} distinct) — "
        f"codec noise, not motion"
    )
    structure = outputs.measure_structure(frames)
    assert outputs.structure_failure(structure) is None, (
        "and the structure gates pass it too: a frozen picture is still a picture"
    )


def test_a_blank_video_is_rejected(tmp_path):
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(write_video(tmp_path / "blank.mp4", blank=True))
    assert "flat" in str(caught.value)


def test_a_short_video_is_rejected_because_upstream_floors_silently(tmp_path):
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(
            write_video(tmp_path / "short.mp4", frames=5), expected_frames=9
        )
    assert "decoded 5 frames, asked for 9" in str(caught.value)


def test_a_file_with_no_video_frames_behind_it_is_rejected(tmp_path):
    """A container that opens cleanly and yields nothing to look at.

    ``verify_streams`` catches the missing-stream case from the registry side;
    this is the same refusal one layer in, so the content checker is not
    dependent on having been called in the right order to be honest.
    """
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(write_wav(tmp_path / "audio-only.wav"))
    assert "ZERO video frames" in str(caught.value)
    assert "not a pass" in str(caught.value)


# ---------------------------------------------------------------------------
# content, not existence — STRUCTURE
#
# The three checks above are all satisfied by NOISE. One artifact proved it:
# 121 of 121 distinct frames, the geometry the request asked for, non-zero
# bytes, a full luma range — colour static, scored PASS by this very tool.
# ---------------------------------------------------------------------------


def test_a_structureless_video_is_rejected(tmp_path):
    """Distinct, moving, full-range, correct geometry — and nothing is in it."""
    static = write_frames(tmp_path / "static.mp4", structureless_stack())
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(static, expected_frames=9)
    message = str(caught.value)
    assert "colour static" in message
    assert "properties of NOISE" in message


def test_every_other_content_check_passes_on_that_same_static(tmp_path):
    """The reason the structure check had to be added at all.

    If any pre-existing check could catch this file, the operator's failure
    would have been caught. None of them can: each one measures something noise
    has in abundance. Asserted here directly, so nobody can later delete the
    structure check believing the older ones cover it.
    """
    static = write_frames(tmp_path / "static.mp4", structureless_stack())
    assert verify_streams_ok("text_to_video", static), "the structural check passes on this file"

    frames = space_matrix._decode_video_frames(static)
    stack = np.stack(frames).astype("float32")
    distinct = len({frame.tobytes() for frame in frames})
    spatial = float(np.median([frame.std() for frame in stack]))
    delta = float(np.mean([np.abs(stack[i + 1] - stack[i]).mean() for i in range(len(frames) - 1)]))

    assert len(frames) == 9, "the frame-count check passes"
    assert distinct >= space_matrix.MIN_DISTINCT_FRAMES, "the distinctness check passes"
    assert spatial > space_matrix.MIN_PIXEL_STD, "the flatness check passes"
    assert delta > space_matrix.LOW_MEAN_FRAME_DELTA, (
        "and the motion measure passes by 100x — it was a FAIL when this artifact came "
        "back, and it scored this file a pass"
    )


def test_the_refusal_names_both_halves_of_the_conjunction(tmp_path):
    """One number alone never condemns, so the message must show both."""
    static = write_frames(tmp_path / "static.mp4", structureless_stack())
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(static)
    message = str(caught.value)
    assert f"< {space_matrix.MIN_STRUCTURE_RATIO}" in message
    assert f"> {space_matrix.MAX_HIGH_FREQUENCY_RATIO}" in message


def test_a_near_static_but_structured_video_is_accepted(tmp_path):
    """``r7b`` must not fail. It is a legitimate near-still render sitting only
    2.0x above the operator's static on the *temporal* gate, which is exactly
    why the structure ratio — 1.31x and 1.32x either side — is the one that
    carries the refusal."""
    content = space_matrix.verify_video_content(
        write_frames(tmp_path / "near-static.mp4", near_static_stack()), expected_frames=9
    )
    assert content.structure.structure_ratio > space_matrix.MIN_STRUCTURE_RATIO
    assert content.structure.high_frequency_ratio < space_matrix.MAX_HIGH_FREQUENCY_RATIO
    assert content.warnings == (), content.render()


# ---------------------------------------------------------------------------
# temporal difference WARNS — the demoted gate
#
# This section is the counterpart of the one above, and it is the same lesson
# with the sign reversed. The structure gates exist because a statistical check
# scored mush a PASS. These exist because a statistical check scored a real
# render a FAIL: `image_to_audio_video` at 512x960x121, a coherent photographic
# scene, refused for mean |Δ| 0.4071 < 0.5 — for being still, while conditioned
# on a still image of that scene.
#
# A temporal measure cannot separate a still scene from a dead one. Both measure
# near zero, the populations were measured OVERLAPPING, and the gate had already
# been observed landing the wrong way up. So it warns now, and these tests are
# what stop it being quietly restored: structure decides, motion is reported.
# ---------------------------------------------------------------------------


def test_a_near_static_render_warns_and_is_not_rejected(tmp_path):
    """The pin. Structurally sound, far too still for the old absolute gate,
    and it must come back as a PASS carrying a note."""
    content = space_matrix.verify_video_content(
        write_frames(tmp_path / "still.mp4", barely_moving_stack()), expected_frames=9
    )
    assert content.mean_abs_delta < space_matrix.LOW_MEAN_FRAME_DELTA, (
        f"the fixture must actually be in the destroyed artifact's neighbourhood, "
        f"not merely near it: {content.mean_abs_delta:.4f}"
    )
    assert content.structure.structure_ratio > space_matrix.MIN_STRUCTURE_RATIO
    assert content.structure.high_frequency_ratio < space_matrix.MAX_HIGH_FREQUENCY_RATIO
    assert any("near-static" in note for note in content.warnings), content.render()
    assert "*** WARNING:" in content.render(), "measured, printed, and not a verdict"


def test_the_near_static_warning_says_it_is_not_a_defect(tmp_path):
    """A warning a reader mistakes for a verdict is how the gate grows back."""
    content = space_matrix.verify_video_content(
        write_frames(tmp_path / "still.mp4", barely_moving_stack()), expected_frames=9
    )
    note = next(note for note in content.warnings if "near-static" in note)
    assert "NOT proof of a defect" in note
    assert "still scene from a dead one" in note


def test_the_verdict_does_not_move_when_the_same_scene_is_merely_dimmer(tmp_path):
    """Why the surviving temporal refusal is a RATIO.

    Two encodes of one scene at one rate of motion, the second dimmed. The
    absolute measure falls six-fold on exposure alone — so an absolute threshold
    decides on how bright a render is, which is not a question about whether it
    is alive. The freeze floor normalises by the frames' own spatial std and
    both files land on the same side of it.
    """
    bright = space_matrix.verify_video_content(
        write_frames(tmp_path / "bright.mp4", barely_moving_stack()), expected_frames=9
    )
    dim = space_matrix.verify_video_content(
        write_frames(tmp_path / "dim.mp4", barely_moving_stack(scale=0.35)), expected_frames=9
    )

    assert dim.mean_abs_delta < bright.mean_abs_delta / 4.0, (
        f"the absolute measure collapses with exposure: {bright.mean_abs_delta:.4f} -> "
        f"{dim.mean_abs_delta:.4f}"
    )
    for content in (bright, dim):
        assert content.relative_delta > space_matrix.MIN_RELATIVE_FRAME_DELTA, content.render()


def test_a_frozen_stack_is_refused_and_the_moving_one_beside_it_is_not(tmp_path):
    """Both directions in one test, on the same picture.

    ``frozen_stack`` and ``barely_moving_stack`` render the identical scene; one
    of them moves a fraction of a level per frame and the other does not move at
    all. The freeze floor has to tell those apart, because that fraction is the
    whole difference between an artifact worth keeping and a dead one.
    """
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(
            write_frames(tmp_path / "frozen.mp4", frozen_stack()), expected_frames=9
        )
    assert "one frame N times" in str(caught.value)

    alive = space_matrix.verify_video_content(
        write_frames(tmp_path / "alive.mp4", barely_moving_stack()), expected_frames=9
    )
    assert alive.relative_delta > space_matrix.MIN_RELATIVE_FRAME_DELTA


def test_the_freeze_floor_is_not_measured_where_there_is_no_contrast(tmp_path):
    """0/0 is not a failure. Below :data:`MIN_PIXEL_STD` there is nothing to
    normalise by, so the ratio reports ``None`` and the flatness check does the
    condemning on its own terms — the same third state ``FrameStructure`` keeps
    for SR and HF, for the same reason: "could not measure" must not read as
    "measured and bad"."""
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(
            write_video(tmp_path / "blank.mp4", blank=True), expected_frames=9
        )
    message = str(caught.value)
    assert "flat" in message
    assert "one frame N times" not in message, "the flat file is diagnosed as flat, not as frozen"
    assert "relative n/a" in message


def test_the_banked_render_the_old_gate_destroyed_now_passes():
    """The artifact itself, not a stand-in.

    ``image_to_audio_video`` at 512x960x121: 121 of 121 distinct frames,
    structure 0.9974, high-freq 0.0559 — a red balloon over a lake at dawn, and
    the old absolute gate refused it at mean |Δ| 0.4071 for being too still.
    """
    content = space_matrix.verify_video_content(banked(FALSELY_REFUSED), expected_frames=121)

    assert content.mean_abs_delta < space_matrix.LOW_MEAN_FRAME_DELTA, (
        f"this is the artifact the old gate refused; if it is no longer under the old "
        f"threshold the test has lost its subject: {content.mean_abs_delta:.4f}"
    )
    assert content.structure.structure_ratio > space_matrix.MIN_STRUCTURE_RATIO
    assert content.structure.high_frequency_ratio < space_matrix.MAX_HIGH_FREQUENCY_RATIO
    assert content.relative_delta > space_matrix.MIN_RELATIVE_FRAME_DELTA
    assert any("near-static" in note for note in content.warnings)


def test_the_legitimate_population_reaches_below_the_threshold_that_refused_it():
    """The measurement that ends the argument.

    The old gate looked defensible because the failure it was calibrated against
    sat at 0.403 and the stillest *known* real render sat at 0.864 — a factor of
    two. It was never a factor of two. A sibling of the destroyed artifact, the
    same scene, structure 0.9973, measures 0.3859: **a legitimate render below
    the failure the threshold was drawn against.** The populations overlap, so
    no threshold on this axis separates them, so this one is a warning.
    """
    stiller = space_matrix.verify_video_content(banked(STILLER_SIBLING), expected_frames=121)

    assert stiller.mean_abs_delta < 0.403, (
        f"a real render measuring {stiller.mean_abs_delta:.4f}, under the 0.403 the old "
        f"threshold was calibrated against"
    )
    assert stiller.structure.structure_ratio > space_matrix.MIN_STRUCTURE_RATIO, (
        "and it is unambiguously a picture"
    )
    assert stiller.relative_delta > space_matrix.MIN_RELATIVE_FRAME_DELTA


def test_a_frozen_copy_of_a_real_frame_is_refused_where_the_render_is_not(tmp_path):
    """The freeze floor against the hardest real pair there is: a genuine
    photographic frame, and that same frame held for the whole clip.

    This is also where byte identity is measurably useless — the frozen re-encode
    comes back with dozens of distinct frames — and where the structure gates are
    blind, because a frozen photograph is still a photograph.
    """
    from ltx25_studio import outputs

    frames = space_matrix._decode_video_frames(banked(FALSELY_REFUSED))
    held = write_frames(tmp_path / "held.mp4", np.stack([frames[0]] * len(frames)))

    decoded = space_matrix._decode_video_frames(held)
    assert len({frame.tobytes() for frame in decoded}) >= space_matrix.MIN_DISTINCT_FRAMES
    assert outputs.structure_failure(outputs.measure_structure(decoded)) is None

    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(held, expected_frames=len(frames))
    assert "one frame N times" in str(caught.value)


def test_a_monochrome_video_is_accepted(tmp_path):
    """Saturation 0.00 is an artistic choice. Half the banked set is nearly
    monochrome and a deliberately black-and-white render measures exactly zero,
    so colour is measured and never gated on."""
    content = space_matrix.verify_video_content(
        write_frames(tmp_path / "mono.mp4", near_static_stack()), expected_frames=9
    )
    assert content.structure.saturation == pytest.approx(0.0, abs=0.01)
    assert "never asserted" in content.render()


# ---------------------------------------------------------------------------
# the WARN tier annotates and never raises
# ---------------------------------------------------------------------------


def test_a_low_contrast_video_warns_and_is_not_rejected(tmp_path):
    """``r4a`` measures block std 5.33 and luma band 24.76 and is real output.
    Refusing on those numbers destroys work a user paid GPU-seconds for; saying
    them out loud costs nothing. **A statistical measure demotes an artifact
    here; it never condemns one on its own.**"""
    content = space_matrix.verify_video_content(
        write_frames(tmp_path / "murky.mp4", murky_stack()), expected_frames=9
    )
    assert content.structure.block_std < space_matrix.LOW_BLOCK_STD
    assert content.structure.luma_band < space_matrix.LOW_LUMA_BAND
    assert any("low detail" in note for note in content.warnings)
    assert any("low contrast" in note for note in content.warnings)


def test_the_warning_is_visible_in_what_the_runner_prints(tmp_path):
    """A warning folded into a paragraph of numbers is a warning nobody reads."""
    content = space_matrix.verify_video_content(
        write_frames(tmp_path / "murky.mp4", murky_stack()), expected_frames=9
    )
    rendered = content.render()
    assert "*** WARNING:" in rendered
    assert "structure" in rendered and "high-freq" in rendered


def test_a_healthy_video_carries_no_warnings(tmp_path):
    """The other direction. A warner that always warns gets ignored."""
    content = space_matrix.verify_video_content(
        write_frames(tmp_path / "good.mp4", near_static_stack()), expected_frames=9
    )
    assert content.warnings == ()
    assert "WARNING" not in content.render()


def test_the_run_records_the_structure_it_measured(tmp_path, quiet_log):
    """Reported on every run, not only on the failing ones — a number that only
    ever appears next to a refusal cannot be sanity-checked against a good run."""
    log, lines = quiet_log
    video = write_video(tmp_path / "out.mp4")
    record = space_matrix.run_one(
        FakeClient([str(video), None, "✅ Text to video complete. seed `1234`"]),
        space_matrix.RunConfig(mode_id="text_to_video"),
        log=log,
    )
    assert record.video.structure is not None
    assert "structure" in "\n".join(lines)


# ---------------------------------------------------------------------------
# spatial std is not stack std — the mislabelled `pixel_std`, fixed
# ---------------------------------------------------------------------------


def test_a_stack_of_flat_frames_that_merely_changes_brightness_is_rejected(tmp_path):
    """The bug in the old ``pixel_std``: it was ``stack.std()`` over the whole
    ``[F, H, W, 3]`` block, documented as a per-frame flatness measure. It is
    not one. It folds TEMPORAL brightness drift into a supposedly spatial
    number, so nine individually blank frames fading from dark to light score
    62 against a threshold of 1.0 and sail through the check written to catch
    exactly them."""
    drifting = write_frames(tmp_path / "drift.mp4", drifting_flat_stack())
    frames = space_matrix._decode_video_frames(drifting)
    stack = np.stack(frames).astype("float32")

    assert float(stack.std()) > 50.0, "the OLD pixel_std passes this file comfortably"

    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_video_content(drifting, expected_frames=9)
    assert "median per-frame std" in str(caught.value)
    assert "flat" in str(caught.value)


def test_both_std_figures_are_reported_and_only_one_is_asserted_on(tmp_path):
    """``stack_std`` keeps its place in the record — it is a real number and
    losing it would lose continuity with every earlier run — but it is labelled
    for what it is and nothing gates on it."""
    content = space_matrix.verify_video_content(
        write_frames(tmp_path / "good.mp4", near_static_stack()), expected_frames=9
    )
    rendered = content.render()
    assert "spatial std" in rendered
    assert "stack std" in rendered and "never asserted" in rendered


def test_the_structure_thresholds_are_the_apps_own(tmp_path):
    """Bound, not copied. A laptop-side gate the deployed app does not share is
    a gate that only ever catches the failures nobody is looking at, and two
    hand-maintained copies of a threshold drift the moment one is retuned."""
    from ltx25_studio import outputs

    assert space_matrix.MIN_STRUCTURE_RATIO is outputs.MIN_STRUCTURE_RATIO
    assert space_matrix.MAX_HIGH_FREQUENCY_RATIO is outputs.MAX_HIGH_FREQUENCY_RATIO
    assert space_matrix.LOW_BLOCK_STD is outputs.LOW_BLOCK_STD
    assert space_matrix.LOW_LUMA_BAND is outputs.LOW_LUMA_BAND


def verify_streams_ok(mode_id, path) -> bool:
    """The structural verdict, so a test can state what it does NOT catch."""
    return verify_streams_module().verify(mode_id, str(path)).ok


def verify_streams_module():
    import verify_streams

    return verify_streams


# ---------------------------------------------------------------------------
# content, not existence — audio
# ---------------------------------------------------------------------------


def test_audible_audio_passes_and_reports_dbfs(tmp_path):
    content = space_matrix.verify_audio_content(
        write_wav(tmp_path / "loud.wav"), expected_sample_rate=48000
    )
    assert content.sample_rate == 48000
    assert content.channels == 2
    assert content.dbfs > space_matrix.SILENT_DBFS
    assert "dBFS" in content.render()


def test_silent_audio_is_rejected(tmp_path):
    """H9: correct codec, correct rate, two channels, exit 0 — and no sound."""
    silent = write_wav(tmp_path / "silent.wav", silent=True)
    assert verify_streams_ok("text_to_audio", silent), "the structural check passes on this file"
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_audio_content(silent, expected_sample_rate=48000)
    assert "SILENT" in str(caught.value)
    assert "dBFS" in str(caught.value)


def test_quiet_audio_is_flagged_but_not_rejected(tmp_path):
    """-55 dBFS is faint and audible. Calling it silence is what turned H9 into a
    false defect report, so the two thresholds must stay far apart."""
    assert space_matrix.SILENT_DBFS < -55.0 < space_matrix.VERY_QUIET_DBFS
    content = space_matrix.AudioContent(
        sample_rate=48000, channels=2, samples=1024, peak=0.01, rms=0.0018, dbfs=-55.0
    )
    assert "VERY QUIET" in content.render()


def test_the_wrong_sample_rate_is_rejected(tmp_path):
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_audio_content(
            write_wav(tmp_path / "16k.wav", rate=16000), expected_sample_rate=48000
        )
    assert "sample rate 16000" in str(caught.value)


def test_a_downmixed_track_is_rejected(tmp_path):
    with pytest.raises(space_matrix.ArtifactContentError) as caught:
        space_matrix.verify_audio_content(
            write_wav(tmp_path / "mono.wav", channels=1), expected_sample_rate=48000
        )
    assert "1 channel(s), expected 2" in str(caught.value)


def test_the_expected_channel_count_comes_from_verify_streams(tmp_path):
    """Not restated here: one constant, read twice."""
    import verify_streams

    assert verify_streams.EXPECTED_AUDIO_CHANNELS == 2


def test_the_audio_check_runs_on_a_muxed_mp4_too(tmp_path):
    """Audio+video modes write ONE container with an AAC track inside it."""
    muxed = write_video(tmp_path / "both.mp4", with_audio=True)
    content = space_matrix.verify_audio_content(muxed, expected_sample_rate=48000)
    assert content.channels == 2
    assert content.dbfs > space_matrix.SILENT_DBFS


def test_a_silent_muxed_track_is_rejected(tmp_path):
    muxed = write_video(tmp_path / "silent-both.mp4", with_audio=True, silent=True)
    with pytest.raises(space_matrix.ArtifactContentError):
        space_matrix.verify_audio_content(muxed, expected_sample_rate=48000)


# ---------------------------------------------------------------------------
# fixture chaining — the right file into the right input
# ---------------------------------------------------------------------------


def test_a_video_run_seeds_both_the_video_and_the_image_fixture(tmp_path, quiet_log):
    log, _lines = quiet_log
    store = space_matrix.FixtureStore(tmp_path / "fixtures")
    record = space_matrix.run_one(
        FakeClient([str(write_video(tmp_path / "out.mp4")), None, "✅ complete"]),
        space_matrix.RunConfig(mode_id="text_to_video"),
        fixtures=store,
        log=log,
    )
    assert set(record.fixtures_saved) == {"video", "image"}
    assert store.has("video")
    assert store.has("image")
    assert store.path_for("image").suffix == ".png"
    assert not store.has("audio"), "text_to_video exports no audio and must seed none"


def test_the_image_fixture_is_a_middle_frame_not_frame_zero(tmp_path):
    """Frame 0 of a conditioned run is the input echoed back, so an image fixture
    taken from there conditions the next run on itself — a degenerate signal the
    model can produce something plausible from while the anchor is wrong."""
    source = write_video(tmp_path / "src.mp4", frames=9)
    index = space_matrix.extract_image_fixture(source, tmp_path / "image.png")
    assert index == 4
    assert (tmp_path / "image.png").stat().st_size > 0


def test_an_audio_only_run_seeds_a_real_wav(tmp_path, quiet_log):
    log, _lines = quiet_log
    store = space_matrix.FixtureStore(tmp_path / "fixtures")
    record = space_matrix.run_one(
        FakeClient([None, str(write_wav(tmp_path / "out.wav")), "✅ complete"]),
        space_matrix.RunConfig(mode_id="text_to_audio"),
        fixtures=store,
        log=log,
    )
    assert record.fixtures_saved == ("audio",)
    assert store.has("audio")
    # A real decodable wav, not a renamed mp4.
    with av.open(str(store.path_for("audio"))) as container:
        assert container.streams.audio


def test_a_muxed_artifact_is_transcoded_rather_than_renamed(tmp_path):
    """Copying an mp4 to ``audio.wav`` would work — ``av`` sniffs content, not
    extension — and would be a small lie that becomes a large one later."""
    muxed = write_video(tmp_path / "both.mp4", with_audio=True)
    dest = space_matrix.extract_audio_fixture(muxed, tmp_path / "audio.wav")
    with av.open(str(dest)) as container:
        assert container.format.name in ("wav", "wave")
        assert not container.streams.video
        assert container.streams.audio[0].codec_context.name.startswith("pcm")


def test_the_chained_fixture_lands_in_the_right_argument_slot(tmp_path, quiet_log):
    """The whole point of chaining: the image fixture must arrive as the IMAGE
    argument, not one slot over."""
    log, _lines = quiet_log
    store = space_matrix.FixtureStore(tmp_path / "fixtures")
    video = write_video(tmp_path / "out.mp4")
    client = FakeClient(
        [str(video), None, "✅ complete"],
        [str(video), None, "✅ complete"],
    )
    space_matrix.run_matrix(
        client,
        space_matrix.build_plan(["text_to_video", "image_to_video"]),
        fixtures=store,
        ledger=space_matrix.Ledger(ceiling_seconds=10_000.0),
        log=log,
    )
    assert len(client.calls) == 2
    image_arg = client.arg("image", call=1)
    assert isinstance(image_arg, dict)
    assert Path(image_arg["path"]) == store.path_for("image")
    # And nothing leaked into the slots this mode does not use.
    assert client.arg("video", call=1) is None
    assert client.arg("audio", call=1) is None


def test_an_audio_input_mode_receives_the_audio_fixture(tmp_path, quiet_log):
    log, _lines = quiet_log
    store = space_matrix.FixtureStore(tmp_path / "fixtures")
    wav = write_wav(tmp_path / "out.wav")
    client = FakeClient(
        [None, str(wav), "✅ complete"],
        [None, str(wav), "✅ complete"],
    )
    space_matrix.run_matrix(
        client,
        space_matrix.build_plan(["text_to_audio", "audio_to_audio"]),
        fixtures=store,
        ledger=space_matrix.Ledger(ceiling_seconds=10_000.0),
        log=log,
    )
    assert Path(client.arg("audio", call=1)["path"]) == store.path_for("audio")
    assert client.arg("image", call=1) is None
    assert client.arg("video", call=1) is None


def test_a_missing_fixture_names_the_run_that_would_have_produced_it(tmp_path):
    store = space_matrix.FixtureStore(tmp_path / "fixtures")
    with pytest.raises(space_matrix.MissingFixtureError) as caught:
        space_matrix.predict_args(space_matrix.RunConfig(mode_id="image_to_video"), fixtures=store)
    message = str(caught.value)
    assert "image_to_video needs a real image input" in message
    assert "text_to_video" in message
    assert "synthesised" in message, "the refusal must say NOT to substitute a fake"


def test_a_fixture_directory_is_never_required_for_a_mode_that_takes_no_file():
    """``--fixture-dir`` is optional for the modes that open the chain."""
    args = space_matrix.predict_args(space_matrix.RunConfig(mode_id="text_to_video"))
    assert args[0] == "text_to_video"


# ---------------------------------------------------------------------------
# --dry-run spends nothing, and it is the default
# ---------------------------------------------------------------------------


@pytest.fixture
def exploding_connect(monkeypatch):
    """Any attempt to reach the network during a dry run fails the test loudly."""
    calls = []

    def _boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("a dry run must never connect to the Space")

    monkeypatch.setattr(space_matrix, "connect", _boom)
    return calls


def test_dry_run_is_the_default(exploding_connect, quiet_log):
    log, lines = quiet_log
    assert space_matrix.main([], log=log) == 0
    assert exploding_connect == []
    assert any("DRY RUN" in line for line in lines)
    assert any("ZERO GPU-seconds were spent" in line for line in lines)


def test_dry_run_prints_the_ordered_plan_with_a_running_total(exploding_connect, quiet_log):
    log, lines = quiet_log
    assert space_matrix.main(["--dry-run"], log=log) == 0
    plan = "\n".join(lines)
    for mode_id in space_matrix.DEFAULT_MODE_ORDER:
        assert mode_id in plan
    assert "cumul" in plan
    assert "ceiling" in plan
    assert "this plan" in plan
    assert "projected" in plan
    assert exploding_connect == []


def test_the_safe_direction_wins_when_both_flags_are_given(exploding_connect, quiet_log):
    log, _lines = quiet_log
    assert space_matrix.main(["--live", "--dry-run"], log=log) == 0
    assert exploding_connect == []


def test_the_dry_run_validates_every_config_and_refuses_an_illegal_plan(quiet_log):
    log, lines = quiet_log
    assert space_matrix.main(["--dry-run", "--width", "250"], log=log) == 2
    assert any("not runnable" in line for line in lines)


def test_the_dry_run_says_where_the_ceiling_would_bind(exploding_connect, quiet_log):
    log, lines = quiet_log
    assert space_matrix.main(["--dry-run", "--quota-ceiling", "250"], log=log) == 0
    plan = "\n".join(lines)
    assert "DOES NOT FIT" in plan
    assert "will be refused, not started" in plan


def test_an_unknown_mode_is_a_usage_error_not_a_crash(quiet_log):
    log, lines = quiet_log
    assert space_matrix.main(["--modes", "not_a_mode"], log=log) == 2
    assert any("not runnable" in line for line in lines)


# ---------------------------------------------------------------------------
# the --live wiring, driven end to end with a fake client
#
# Everything above tests `run_matrix` directly. That leaves the one seam nothing
# had covered: the argv -> plan -> ledger -> client -> summary path inside
# `main`. It is also the only path the operator actually types, so a mistake
# there costs GPU-seconds to discover.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_connect(monkeypatch):
    """Hand ``main`` a fake client instead of a real connection."""

    def install(client):
        seen = {}

        def _connect(space_id, *, token=None, download_dir=None, log=print):
            seen.update(space_id=space_id, token=token, download_dir=download_dir)
            return client

        monkeypatch.setattr(space_matrix, "connect", _connect)
        return seen

    return install


def test_live_runs_the_plan_and_returns_zero(tmp_path, fake_connect, clock, quiet_log):
    log, lines = quiet_log
    clock(step=10.0)
    client = FakeClient([str(write_video(tmp_path / "out.mp4")), None, "✅ complete"])
    seen = fake_connect(client)

    code = space_matrix.main(
        [
            "--live",
            "--modes",
            "text_to_video",
            "--fixture-dir",
            str(tmp_path / "fixtures"),
        ],
        log=log,
    )
    assert code == 0
    assert len(client.calls) == 1
    assert client.calls[0]["api_name"] == space_matrix.ENDPOINT
    assert seen["space_id"] == space_matrix.DEFAULT_SPACE_ID
    summary = "\n".join(lines)
    assert "SUMMARY" in summary
    assert "UPPER BOUND" in summary
    assert "Do not quote it as billed seconds" in summary


def test_live_returns_three_when_the_ceiling_stops_it(tmp_path, fake_connect, clock, quiet_log):
    """Non-zero but orderly: a stop is reported, not raised."""
    log, lines = quiet_log
    clock(step=30.0)
    video = str(write_video(tmp_path / "out.mp4"))
    client = FakeClient([video, None, "ok"], [video, None, "ok"])
    fake_connect(client)

    code = space_matrix.main(
        [
            "--live",
            "--modes",
            "text_to_video,text_to_video",
            "--quota-ceiling",
            "150",
            "--reservation-seconds",
            "50",
            "--fixture-dir",
            str(tmp_path / "fixtures"),
        ],
        log=log,
    )
    assert code == 3
    assert len(client.calls) == 1
    assert any("STOPPED AT THE QUOTA CEILING" in line for line in lines)


def test_a_failed_run_raises_out_of_main_after_printing_the_ledger(
    tmp_path, fake_connect, clock, quiet_log
):
    """The summary still gets printed: a run that blew up still spent quota, and
    losing that number leaves the next session blind to how much of the day is
    gone. Then it raises, because a traceback is the loudest thing available."""
    log, lines = quiet_log
    clock(step=10.0)
    client = FakeClient([None, None, SPACE_FAILURE_TEXT])
    fake_connect(client)

    with pytest.raises(space_matrix.SpaceRunError):
        space_matrix.main(
            ["--live", "--modes", "text_to_video", "--fixture-dir", str(tmp_path / "fixtures")],
            log=log,
        )
    summary = "\n".join(lines)
    assert "SUMMARY" in summary
    assert "20.0 GPU-s" in summary, "the run's bounded cost survives the failure"


def test_the_default_fixture_directory_is_inside_the_ignored_outputs_tree():
    """A fixture directory anywhere else leaves generated media untracked in the
    working tree, and the first ``git add -A`` commits a video."""
    source = (REPO_ROOT / "scripts" / "space_matrix.py").read_text()
    assert 'REPO / "outputs" / "space-fixtures"' in source
    assert "outputs/" in (REPO_ROOT / ".gitignore").read_text()


# ---------------------------------------------------------------------------
# the ceiling spans invocations, because the budget does
#
# The regression this section exists for: the counter used to live entirely in
# one process's memory, so five separate invocations each started at zero and
# each independently announced FITS. Five runs of 1900 would each have printed
# FITS while spending 9500 of a 2400/day budget.
# ---------------------------------------------------------------------------


def test_a_fresh_ledger_starts_at_zero(tmp_path):
    store = space_matrix.PersistentQuota(tmp_path / "ledger.json", day="2026-08-22")
    assert store.spent_upper_bound == 0.0
    assert store.invocations == 0
    assert store.rolled_over_from is None


def test_spend_survives_the_process(tmp_path):
    path = tmp_path / "ledger.json"
    first = space_matrix.PersistentQuota(path, day="2026-08-22")
    first.begin_invocation()
    first.add(43.1)
    first.add(88.8)

    second = space_matrix.PersistentQuota(path, day="2026-08-22")
    assert second.spent_upper_bound == pytest.approx(131.9)
    assert second.invocations == 1


def test_the_ledger_file_states_its_own_unit(tmp_path):
    """Anyone who cats the file must see that it is a bound, not billed seconds."""
    path = tmp_path / "ledger.json"
    space_matrix.PersistentQuota(path, day="2026-08-22").add(43.1)
    payload = json.loads(path.read_text())
    assert payload["utc_day"] == "2026-08-22"
    assert payload["spent_upper_bound"] == pytest.approx(43.1)
    assert "UPPER BOUND" in payload["unit"]
    assert "NOT billed seconds" in payload["unit"]
    assert "no readable ZeroGPU quota endpoint" in payload["note"]


def test_a_new_utc_day_resets_and_says_what_it_dropped(tmp_path):
    path = tmp_path / "ledger.json"
    space_matrix.PersistentQuota(path, day="2026-08-21").add(346.0)

    tomorrow = space_matrix.PersistentQuota(path, day="2026-08-22")
    assert tomorrow.spent_upper_bound == 0.0
    assert tomorrow.rolled_over_from == ("2026-08-21", pytest.approx(346.0))
    rendered = tomorrow.render()
    assert "NEW UTC DAY" in rendered
    assert "346.0" in rendered, "a rollover must say what it dropped, not silently zero"
    assert "rolling 24h window" in rendered, "the unverified reset mechanism must be stated"


def test_the_same_utc_day_does_not_reset(tmp_path):
    path = tmp_path / "ledger.json"
    space_matrix.PersistentQuota(path, day="2026-08-22").add(346.0)
    same = space_matrix.PersistentQuota(path, day="2026-08-22")
    assert same.spent_upper_bound == pytest.approx(346.0)
    assert same.rolled_over_from is None


def test_utc_day_is_utc_not_local():
    from datetime import UTC, datetime

    assert space_matrix.utc_day(datetime(2026, 8, 22, 23, 59, tzinfo=UTC)) == "2026-08-22"
    assert space_matrix.utc_day(datetime(2026, 8, 23, 0, 1, tzinfo=UTC)) == "2026-08-23"


def test_an_unreadable_ledger_is_refused_rather_than_read_as_zero(tmp_path):
    """A corrupt accounting file is not evidence that nothing was spent."""
    path = tmp_path / "ledger.json"
    path.write_text("{ this is not json")
    with pytest.raises(space_matrix.QuotaLedgerUnreadable) as caught:
        space_matrix.PersistentQuota(path, day="2026-08-22")
    assert "not evidence that nothing was spent" in str(caught.value)
    assert "--reset-quota-ledger" in str(caught.value)


def test_reset_starts_the_day_from_zero_on_purpose(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{ this is not json")
    store = space_matrix.PersistentQuota(path, day="2026-08-22", reset=True)
    assert store.spent_upper_bound == 0.0


def test_main_refuses_an_unreadable_ledger_and_recovers_with_the_flag(
    tmp_path, exploding_connect, quiet_log
):
    log, lines = quiet_log
    path = tmp_path / "broken.json"
    path.write_text("nonsense")
    assert space_matrix.main(["--dry-run", "--quota-ledger", str(path)], log=log) == 2
    assert any("could not be read as a quota ledger" in line for line in lines)

    assert (
        space_matrix.main(
            ["--dry-run", "--quota-ledger", str(path), "--reset-quota-ledger"], log=log
        )
        == 0
    )


def test_a_second_invocation_starts_where_the_first_left_off(
    tmp_path, fake_connect, clock, quiet_log
):
    log, lines = quiet_log
    clock(step=10.0)
    video = str(write_video(tmp_path / "out.mp4"))
    ledger_path = tmp_path / "ledger.json"
    argv = [
        "--live",
        "--modes",
        "text_to_video",
        "--quota-ledger",
        str(ledger_path),
        "--fixture-dir",
        str(tmp_path / "fixtures"),
    ]

    fake_connect(FakeClient([video, None, "ok"]))
    assert space_matrix.main(argv, log=log) == 0
    assert json.loads(ledger_path.read_text())["spent_upper_bound"] == pytest.approx(20.0)

    lines.clear()
    fake_connect(FakeClient([video, None, "ok"]))
    assert space_matrix.main(argv, log=log) == 0
    payload = json.loads(ledger_path.read_text())
    assert payload["spent_upper_bound"] == pytest.approx(40.0)
    assert payload["invocations"] == 2
    # And the second invocation SAID it was carrying the first's spend.
    assert any("carried      <= 20.0" in line for line in lines)


def test_separate_invocations_cannot_each_announce_that_they_fit(
    tmp_path, fake_connect, clock, quiet_log
):
    """The headline regression, stated as a test.

    Two invocations under one ceiling: the first fits and spends, the second is
    refused BECAUSE of the first. Before the ledger persisted, the second would
    have started from zero and announced that it fit.
    """
    log, lines = quiet_log
    clock(step=30.0)
    video = str(write_video(tmp_path / "out.mp4"))
    ledger_path = tmp_path / "ledger.json"
    argv = [
        "--live",
        "--modes",
        "text_to_video",
        "--quota-ceiling",
        "150",
        "--reservation-seconds",
        "50",
        "--quota-ledger",
        str(ledger_path),
        "--fixture-dir",
        str(tmp_path / "fixtures"),
    ]

    first = FakeClient([video, None, "ok"])
    fake_connect(first)
    assert space_matrix.main(argv, log=log) == 0
    assert len(first.calls) == 1  # 0 + 100 projected <= 150

    lines.clear()
    second = FakeClient([video, None, "ok"])
    fake_connect(second)
    assert space_matrix.main(argv, log=log) == 3
    assert len(second.calls) == 0, (
        "the second invocation must be refused on the FIRST invocation's spend — "
        "a ceiling that only sees its own process is an inert report"
    )
    assert any("STOPPED AT THE QUOTA CEILING" in line for line in lines)
    # 60 spent, not 120: the refused run was never sent.
    assert json.loads(ledger_path.read_text())["spent_upper_bound"] == pytest.approx(60.0)


def test_the_dry_run_reads_the_ledger_and_never_writes_it(tmp_path, exploding_connect, quiet_log):
    log, lines = quiet_log
    ledger_path = tmp_path / "ledger.json"
    space_matrix.PersistentQuota(ledger_path, day=space_matrix.utc_day()).add(346.0)
    before = ledger_path.read_text()

    assert space_matrix.main(["--dry-run", "--quota-ledger", str(ledger_path)], log=log) == 0
    plan = "\n".join(lines)
    assert "carried      <= 346.0" in plan, "the dry run must judge the plan against today"
    assert "READ, not written" in plan
    assert ledger_path.read_text() == before


def test_the_ledger_is_written_through_per_run_not_at_the_end(
    tmp_path, fake_connect, clock, quiet_log
):
    """A matrix that dies on run 2 has still spent run 1."""
    log, _lines = quiet_log
    clock(step=10.0)
    video = str(write_video(tmp_path / "out.mp4"))
    ledger_path = tmp_path / "ledger.json"
    fake_connect(FakeClient([video, None, "ok"], [None, None, SPACE_FAILURE_TEXT]))

    with pytest.raises(space_matrix.SpaceRunError):
        space_matrix.main(
            [
                "--live",
                "--modes",
                "text_to_video,text_to_video",
                "--quota-ledger",
                str(ledger_path),
                "--fixture-dir",
                str(tmp_path / "fixtures"),
            ],
            log=log,
        )
    # Both runs are on the books: the one that worked and the one that burned a
    # slot and failed.
    assert json.loads(ledger_path.read_text())["spent_upper_bound"] == pytest.approx(40.0)


def test_the_ledger_path_is_not_derived_from_the_fixture_dir(
    tmp_path, fake_connect, clock, quiet_log
):
    """Two invocations with DIFFERENT fixture dirs must share one ledger.

    A ledger under --fixture-dir would reintroduce the whole defect: each
    invocation would keep its own total and each would announce FITS.
    """
    log, _lines = quiet_log
    clock(step=10.0)
    video = str(write_video(tmp_path / "out.mp4"))
    ledger_path = tmp_path / "ledger.json"

    for name in ("fixtures-a", "fixtures-b"):
        fake_connect(FakeClient([video, None, "ok"]))
        assert (
            space_matrix.main(
                [
                    "--live",
                    "--modes",
                    "text_to_video",
                    "--quota-ledger",
                    str(ledger_path),
                    "--fixture-dir",
                    str(tmp_path / name),
                ],
                log=log,
            )
            == 0
        )
    assert json.loads(ledger_path.read_text())["spent_upper_bound"] == pytest.approx(40.0)


def test_the_default_ledger_path_is_fixed_and_inside_the_ignored_outputs_tree():
    source = (REPO_ROOT / "scripts" / "space_matrix.py").read_text()
    assert 'DEFAULT_QUOTA_LEDGER = "outputs/.space-quota-ledger.json"' in source
    assert "outputs/" in (REPO_ROOT / ".gitignore").read_text()
    # Resolved from the module constant, never from a directory flag.
    assert "args.quota_ledger or (REPO / DEFAULT_QUOTA_LEDGER)" in source


def test_the_module_never_writes_the_token_anywhere():
    """The token is read from the environment and handed straight to the client.

    Asserted on the source because the failure mode is a token that reaches a log
    line or a file, and by the time that happens in a run it is already leaked.
    """
    source = (REPO_ROOT / "scripts" / "space_matrix.py").read_text()
    assert "hf_" not in source.replace("hf_gradio", "")
    # Read once, at the single site that builds the client.
    assert source.count('os.environ.get("HF_TOKEN")') == 1
