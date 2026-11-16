"""Spec for ``scripts/space_matrix.py``, the Space harness, with no Space and no network.

Live paths run against ``tests/space_fakes.FakeSpace``: a real HTTP server on 127.0.0.1 that speaks
the Gradio queue protocol and the Hub API's status and log endpoints. Its published signature is
pinned against the real ``build_ui()`` below, and every request it receives is recorded, so a test
can prove what did NOT happen (no GPU call after a quota refusal, no request at all in a dry run).

Media the checks decode are real containers built with ``av``.

Every test runs with a sentinel token in ``HF_TOKEN`` and with the default Space and Hub URLs
pointed at a closed local port, so a forgotten override fails instead of reaching Hugging Face.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import space_matrix
from space_fakes import QUOTA_MESSAGE, FakeSpace, Reply

from ltx25_studio import modes, zerogpu
from ltx25_studio.ui import fields

av = pytest.importorskip("av")

SENTINEL = "hf_" + "SENTINEL" + uuid.uuid4().hex
CLOSED = "http://127.0.0.1:9"
NOW = datetime(2026, 11, 16, 15, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# isolation: no real token, no real host, a fixed clock
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.setattr(space_matrix, "DEFAULT_HUB_URL", CLOSED)
    monkeypatch.setattr(space_matrix, "space_host", lambda space_id: CLOSED)
    monkeypatch.setattr(space_matrix, "_utcnow", lambda: NOW)
    original = space_matrix.Http.open

    def local_only(self, method, url, **kwargs):
        assert url.startswith("http://127.0.0.1:"), f"a test tried to reach {url}"
        return original(self, method, url, **kwargs)

    monkeypatch.setattr(space_matrix.Http, "open", local_only)


@pytest.fixture
def space(tmp_path):
    started: list[FakeSpace] = []

    def make(**kwargs) -> FakeSpace:
        kwargs.setdefault("token", SENTINEL)
        fake = FakeSpace(tmp_path / f"fake{len(started)}", **kwargs).__enter__()
        started.append(fake)
        return fake

    yield make
    for fake in started:
        fake.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# real media
# ---------------------------------------------------------------------------

SIZE = 64
RATE = 24


def write_frames(path, stack, *, rate=RATE, with_audio=False):
    stack = np.ascontiguousarray(stack.astype(np.uint8))
    with av.open(str(path), "w") as container:
        video = container.add_stream("h264", rate=rate)
        video.height, video.width = int(stack.shape[1]), int(stack.shape[2])
        video.pix_fmt = "yuv420p"
        audio = None
        if with_audio:
            audio = container.add_stream("aac", rate=48000, layout="stereo")
            audio.time_base = Fraction(1, 48000)
        for index, frame in enumerate(stack):
            picture = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            picture.pts, picture.time_base = index, Fraction(1, rate)
            for packet in video.encode(picture):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None:
            for index in range(8):
                for packet in audio.encode(_audio_frame(index, silent=False, rate=48000)):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return Path(path)


def moving_stack(frames=9, size=SIZE):
    """A bright block on a dark field that moves: structured, moving, decodes cleanly."""
    stack = np.zeros((frames, size, size, 3), dtype=np.uint8)
    for index in range(frames):
        offset = (index * 4) % (size - 20)
        stack[index, 10:30, offset : offset + 20] = 255
        stack[index, 40:50, 5:15] = (40, 120, 200)
    return stack


def write_video(path, *, frames=9, size=SIZE, with_audio=False):
    return write_frames(path, moving_stack(frames, size), with_audio=with_audio)


def structureless_stack(frames=9, seed=11):
    return np.random.default_rng(seed).integers(0, 256, (frames, SIZE, SIZE, 3), dtype=np.uint8)


def _smooth_base():
    rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return 40 + 90 * np.sin(columns / 11.0) * np.cos(rows / 17.0) + 60 * (rows / SIZE)


def barely_moving_stack(frames=9, step=0.4, scale=1.0):
    base = _smooth_base() * scale
    planes = np.stack([base + index * step * scale for index in range(frames)])
    return np.repeat(np.clip(planes, 0, 255)[..., None], 3, axis=-1).astype(np.uint8)


def frozen_stack(frames=9):
    return np.repeat(barely_moving_stack(1, step=1.4), frames, axis=0)


def murky_stack(frames=9):
    rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    base = 120 + 6 * np.sin(columns / 9.0) * np.cos(rows / 13.0)
    planes = np.stack([base + 3 * np.sin((columns + index * 5) / 9.0) for index in range(frames)])
    return np.repeat(np.clip(planes, 0, 255)[..., None], 3, axis=-1).astype(np.uint8)


def _audio_frame(index, *, silent, rate, channels=2, amplitude=0.4):
    samples = 1024
    layout = "stereo" if channels == 2 else "mono"
    if silent:
        block = np.zeros((channels, samples), dtype="float32")
    else:
        t = (np.arange(samples) + index * samples) / float(rate)
        tone = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype("float32")
        block = np.stack([tone] * channels)
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(block), format="fltp", layout=layout)
    frame.rate, frame.pts, frame.time_base = rate, index * samples, Fraction(1, rate)
    return frame


def write_wav(path, *, silent=False, rate=48000, channels=2, blocks=16, amplitude=0.4):
    with av.open(str(path), "w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout="stereo" if channels == 2 else "mono")
        for index in range(blocks):
            for packet in stream.encode(
                _audio_frame(index, silent=silent, rate=rate, channels=channels, amplitude=amplitude)
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return Path(path)


def write_png(path, size=SIZE):
    from PIL import Image

    Image.fromarray(moving_stack(1, size)[0]).save(str(path))
    return Path(path)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def t2v(label="t2v", **values):
    return {"label": label, "mode": "text_to_video", "prompt": "a tram", "width": SIZE, "height": SIZE,
            "num_frames": 9, "seed": 7, **values}


def run(argv, *, fake=None):
    lines: list[str] = []
    if fake is not None:
        argv = [*argv, "--space-url", fake.url, "--hub-url", fake.url]
    code = space_matrix.main(argv, log=lines.append)
    return code, lines


def live(tmp_path, fake, calls, *extra, ledger=None, out=None, assets=None):
    plan = tmp_path / f"plan-{uuid.uuid4().hex[:6]}.json"
    plan.write_text(json.dumps(calls))
    ledger = ledger or tmp_path / "ledger.jsonl"
    out = out or tmp_path / f"out-{uuid.uuid4().hex[:6]}"
    argv = ["--plan", str(plan), "--live", "--quota-ledger", str(ledger), "--out", str(out), *extra]
    if assets is not None:
        argv += ["--assets-dir", str(assets)]
    code, lines = run(argv, fake=fake)
    return code, lines, out, ledger


def ledger_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def call_records(path):
    return [r for r in ledger_lines(path) if r["kind"] == "call"]


def seed_ledger(path, *, day="2026-11-16", billed=0.0, label="earlier"):
    space_matrix.Ledger(path).append(
        {"kind": "call", "utc_day": day, "label": label, "billed_conservative": billed, "reserved_seconds": 180}
    )


# ---------------------------------------------------------------------------
# the published signature: built by name, checked against the live Space
# ---------------------------------------------------------------------------


def test_the_argument_order_is_the_ui_field_order_minus_the_state():
    """``gr.State`` has no api info, so a client never sees it; everything else keeps its order."""
    without_the_state = tuple(f.key for f in fields.FIELDS if f.kind != "state")
    assert without_the_state == space_matrix.API_FIELD_KEYS
    assert "loras" not in space_matrix.API_FIELD_KEYS
    assert len(space_matrix.API_FIELD_KEYS) == len(fields.FIELD_KEYS) - 1


def test_the_argument_order_matches_the_endpoint_gradio_actually_publishes():
    """Parameter 0 is the mode; then the field spec, append-only, minus the State."""
    pytest.importorskip("gradio")
    from ltx25_studio.ui import build_ui

    endpoint = build_ui().get_api_info()["named_endpoints"][space_matrix.ENDPOINT]
    parameters = endpoint["parameters"]
    assert len(parameters) == 1 + len(space_matrix.API_FIELD_KEYS)
    assert parameters[0]["component"] == "Radio"  # mode_id
    component_for_kind = {"text": "textbox"}
    for index, key in enumerate(space_matrix.API_FIELD_KEYS, start=1):
        kind = fields.field_by_key(key).kind
        assert parameters[index]["component"].lower() == component_for_kind.get(kind, kind), (
            f"argument {index} is a {parameters[index]['component']}, but the field spec puts {key!r} there"
        )
    assert [r["component"] for r in endpoint["returns"]] == ["Video", "Audio", "Markdown"]


def test_the_contract_check_passes_on_the_real_app_and_the_fake_publishes_the_same(tmp_path):
    """The harness's check finds nothing wrong with the real app, and the fake matches the real app."""
    pytest.importorskip("gradio")
    from ltx25_studio.ui import build_ui

    demo = build_ui()
    info, config = demo.get_api_info(), demo.get_config_file()
    real = space_matrix.find_endpoint(config, info, space_matrix.ENDPOINT)
    problems, table = space_matrix.generate_contract_problems(
        real.parameters, input_types=real.input_types, plan_modes=list(modes.MODES)
    )
    assert problems == [], "\n".join(table + problems)

    fake = FakeSpace(tmp_path / "fake")
    mirrored = space_matrix.find_endpoint(fake.config, fake.info, space_matrix.ENDPOINT)
    assert mirrored.input_types == real.input_types
    assert [p["component"] for p in mirrored.parameters] == [p["component"] for p in real.parameters]
    for ours, theirs in zip(mirrored.parameters[1:], real.parameters[1:], strict=True):
        assert theirs["label"].startswith(ours["label"])
    real_lora = space_matrix.find_endpoint(config, info, space_matrix.LORA_ENDPOINT)
    fake_lora = space_matrix.find_endpoint(fake.config, fake.info, space_matrix.LORA_ENDPOINT)
    assert fake_lora.input_types == real_lora.input_types
    assert [p["component"] for p in fake_lora.parameters] == [p["component"] for p in real_lora.parameters]
    route = space_matrix._lora_route(config, info)
    assert route.source_slot is not None and route.upload_slot is not None


def test_the_payload_reaches_the_space_by_name(tmp_path, space):
    """The fake decodes each slot with ITS OWN slot map, so a shifted value shows as a wrong name."""
    fake = space().reply(Reply(video=write_video(tmp_path / "v.mp4")))
    code, lines, _out, _ledger = live(
        tmp_path, fake, [t2v(seed=31, enable_stage_2=False, min_seconds=3.0, max_seconds=4.5)]
    )
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    params = fake.generate_calls[0]["params"]
    assert fake.generate_calls[0]["mode"] == "text_to_video"
    assert params["prompt"] == "a tram"
    assert (params["width"], params["height"], params["num_frames"]) == (SIZE, SIZE, 9)
    assert (params["seed"], params["min_seconds"], params["max_seconds"]) == (31, 3.0, 4.5)
    assert params["image"] is params["video"] is params["audio"] is None
    assert params["loras"] is None, "a State slot is sent empty; the session supplies it"
    assert params["conditions"]["data"] == []
    assert fake.protocol_errors == []


def test_a_field_appended_to_the_spec_is_sent_in_its_own_slot(tmp_path, space, monkeypatch):
    """A decoder choice is being appended. A key the spec has is passed through by name."""
    future = fields.Field(
        key="future_choice", kind="dropdown", label="A later choice", default="a",
        options={"choices": [("A", "a"), ("B", "b")]},
    )
    monkeypatch.setattr(fields, "FIELDS", (*fields.FIELDS, future))
    fake = space().reply(Reply(video=write_video(tmp_path / "v.mp4")))
    code, lines, _out, _ledger = live(tmp_path, fake, [t2v(future_choice="b", decoder="conv")])
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    params = fake.generate_calls[0]["params"]
    assert params["future_choice"] == "b" and params["decoder"] == "conv"
    assert params["enhance_prompt"] is False and params["width"] == SIZE


def test_a_key_the_field_spec_does_not_have_is_refused_before_anything_is_sent(tmp_path, space):
    fake = space()
    code, lines, _out, _ledger = live(tmp_path, fake, [t2v(guidance_scale=3.0)])
    assert code == space_matrix.EXIT_ILLEGAL
    assert any("'guidance_scale' is not a field" in line for line in lines)
    assert fake.requests == []


@pytest.mark.parametrize(
    "patch, needle",
    [
        ("swap_labels", "slot 8"),
        ("insert", "slot"),
        ("kind", "is a number"),
        ("state", "State inputs"),
        ("mode", "does not offer mode"),
    ],
)
def test_a_contract_mismatch_is_refused_with_a_diff_and_nothing_is_spent(tmp_path, space, patch, needle):
    def info_patch(info):
        params = info["named_endpoints"]["/on_generate"]["parameters"]
        if patch == "swap_labels":
            params[7]["label"], params[8]["label"] = params[8]["label"], params[7]["label"]
        elif patch == "insert":
            params.insert(3, {"label": "Decoder", "parameter_name": "x", "component": "Dropdown"})
        elif patch == "kind":
            params[7]["component"] = "Slider"
        elif patch == "mode":
            params[0]["type"]["enum"] = ["image_to_video"]
        return info

    def config_patch(config):
        if patch == "state":
            dependency = config["dependencies"][0]
            inputs = dependency["inputs"]
            state = next(c["id"] for c in config["components"] if c["type"] == "state")
            inputs.remove(state)
            inputs.insert(5, state)
        return config

    fake = space(info_patch=info_patch, config_patch=config_patch)
    code, lines, out, ledger = live(tmp_path, fake, [t2v()])
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_ILLEGAL, text
    assert "CONTRACT MISMATCH" in text and needle in text
    assert fake.joins == [] and "/gradio_api/upload" not in fake.paths()
    assert {r["method"] for r in fake.requests} == {"GET"}
    assert json.loads((out / "contract.json").read_text())["problems"]
    assert not ledger.exists() or call_records(ledger) == []


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


def test_a_plan_file_names_fields_media_loras_and_expectations(tmp_path):
    assets = tmp_path / "assets"
    (assets / "images").mkdir(parents=True)
    write_png(assets / "images" / "tram.png")
    (assets / "adapter.safetensors").write_bytes(b"x" * 16)
    plan = {
        "defaults": {"seed": 7, "num_frames": 121, "expect": {"has_audio": False}},
        "calls": [
            {"label": "i2v", "mode": "image_to_video", "image": "images/tram.png", "strength": 0.6,
             "expect": {"frames": 121}},
            {"label": "lora", "mode": "text_to_video", "prompt": "p",
             "loras": [{"source": "org/style-lora", "weight": 0.8}, {"source": "adapter.safetensors"}]},
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    first, second = space_matrix.load_plan(path, assets_dir=assets)
    assert first.media["image"].path == assets / "images" / "tram.png"
    assert first.value("strength") == 0.6 and first.value("seed") == 7
    assert first.expect == {"has_audio": False, "frames": 121}
    assert [lora.local for lora in second.loras] == [None, assets / "adapter.safetensors"]
    assert second.loras[0].weight == 0.8 and second.loras[1].weight == 1.0


@pytest.mark.parametrize(
    "call, needle",
    [
        ({"label": "a", "mode": "text_to_hologram"}, "text_to_hologram"),
        ({"label": "a", "mode": "image_to_video"}, "needs a image"),
        ({"label": "a", "mode": "text_to_video", "audio": "missing.wav"}, "does not exist"),
        ({"label": "a", "mode": "image_to_video", "image": "x.png", "prompt": "hi"}, "takes no prompt"),
        ({"label": "a", "mode": "text_to_video", "width": 250}, "a"),
        ({"label": "a", "mode": "text_to_video", "num_frames": 10}, "a"),
        ({"label": "a", "mode": "text_to_video", "strength": 2.0}, "outside"),
        ({"label": "a", "mode": "text_to_video", "seed": "seven"}, "must be a number"),
        ({"label": "a", "mode": "text_to_video", "loras": [{"source": "not a repo"}]}, "neither a file"),
        ({"label": "a", "mode": "text_to_video", "expect": {"colour": "red"}}, "expect.colour"),
        ({"label": "a", "mode": "text_to_video", "prompt": "p", "loras": []}, None),
        ({"label": "a b", "mode": "text_to_video"}, "label"),
        ({"label": "a", "mode": "text_to_video", "loras": "org/x"}, "list"),
        ({"label": "a", "mode": "text_to_video", "image": {"from_call": "nope"}}, "not an earlier call"),
    ],
)
def test_an_illegal_call_is_refused_offline(tmp_path, call, needle):
    write_png(tmp_path / "x.png")
    if needle is None:
        space_matrix.parse_plan([call], base=tmp_path)
        return
    with pytest.raises(space_matrix.PlanError) as caught:
        space_matrix.parse_plan([call], base=tmp_path)
    assert needle in str(caught.value)


def test_inputs_are_never_chained_unless_a_call_says_so(tmp_path):
    with pytest.raises(space_matrix.PlanError):
        space_matrix.parse_plan([t2v(), {"label": "i2v", "mode": "image_to_video"}], base=tmp_path)
    calls = space_matrix.parse_plan(
        [t2v(), {"label": "i2v", "mode": "image_to_video", "image": {"from_call": "t2v", "frame": 0}}], base=tmp_path
    )
    assert calls[1].media["image"].chained and calls[1].media["image"].frame == 0
    with pytest.raises(space_matrix.PlanError) as caught:
        space_matrix.parse_plan(
            [{"label": "t2a", "mode": "text_to_audio", "prompt": "p"},
             {"label": "i2v", "mode": "image_to_video", "image": {"from_call": "t2a"}}],
            base=tmp_path,
        )
    assert "exports no video" in str(caught.value)


def test_the_default_plan_covers_every_mode_and_writes_its_chains_out():
    calls = space_matrix.default_plan()
    assert [c.mode for c in calls] == list(space_matrix.DEFAULT_MODE_ORDER)
    assert set(space_matrix.DEFAULT_MODE_ORDER) == set(modes.MODES)
    produced: set[str] = set()
    for call in calls:
        for kind, item in call.media.items():
            assert item.chained, f"{call.label}: the default plan's {kind} must be an explicit chain"
            assert {"image": "video", "video": "video", "audio": "audio"}[kind] in produced
        produced |= set(call.exports)
    assert calls[0].render_size == (space_matrix.CHEAPEST_SIZE, space_matrix.CHEAPEST_SIZE)


def test_an_unknown_mode_and_an_illegal_override_are_usage_errors():
    assert run(["--modes", "not_a_mode"])[0] == space_matrix.EXIT_ILLEGAL
    assert run(["--width", "250"])[0] == space_matrix.EXIT_ILLEGAL
    code, lines = run(["--plan", "x.json", "--width", "704"])
    assert code == space_matrix.EXIT_ILLEGAL and any("plan file sets its own" in line for line in lines)


def test_a_single_call_comes_from_flags(tmp_path):
    write_png(tmp_path / "x.png")
    args = space_matrix._run_parser().parse_args(
        ["--mode", "image_text_to_video", "--image", "x.png", "--assets-dir", str(tmp_path), "--prompt", "p",
         "--width", "704", "--height", "480", "--num-frames", "49", "--enable-stage-2", "--set", "strength=0.5",
         "--lora", "org/style@0.7", "--expect", "frames=49"]
    )
    (call,) = space_matrix.build_calls(args)
    assert call.label == "image_text_to_video" and call.stage_2
    assert call.value("strength") == 0.5 and call.value("num_frames") == 49
    assert call.loras[0].source == "org/style" and call.loras[0].weight == 0.7
    assert call.expected_geometry() == (1408, 960) and call.expected_frames() == 49


def test_only_selects_labels_and_refuses_a_broken_chain(tmp_path):
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps([t2v(), t2v("b"), {"label": "c", "mode": "image_to_video",
                                                   "image": {"from_call": "t2v"}}]))
    code, lines = run(["--plan", str(plan), "--only", "b"])
    assert code == space_matrix.EXIT_OK and any("1. b" in line for line in lines)
    assert run(["--plan", str(plan), "--only", "c"])[0] == space_matrix.EXIT_ILLEGAL


@pytest.mark.parametrize(
    "values, needle",
    [
        ({"width": 1952}, "1920"),
        ({"num_frames": 489}, "481"),
        ({"auto_duration": True, "max_seconds": 25.0}, "20"),
        ({"auto_duration": True, "min_seconds": 5.0, "max_seconds": 5.0}, "greater than"),
    ],
)
def test_the_apps_limits_are_refused_before_anything_is_sent(tmp_path, space, values, needle):
    fake = space()
    code, lines, _out, _ = live(tmp_path, fake, [t2v(**values)])
    assert code == space_matrix.EXIT_ILLEGAL and needle in "\n".join(lines)
    assert fake.requests == []


def test_the_apps_own_gate_runs_on_every_call_offline(tmp_path, space):
    """``generate.prepare_request`` refuses what the Space would refuse, before any GPU is reserved."""
    assets = tmp_path / "assets"
    assets.mkdir()
    write_png(tmp_path / "picture.png").rename(assets / "not-audio.wav")
    fake = space()
    call = {"label": "a2v", "mode": "audio_to_video", "audio": "not-audio.wav", "width": SIZE, "height": SIZE,
            "num_frames": 9}
    code, lines, _out, _ = live(tmp_path, fake, [call], assets=assets)
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_ILLEGAL
    assert "the app would refuse this call before reserving a GPU" in text and "has no audio stream" in text
    assert fake.requests == []


def test_an_audio_mode_takes_its_length_from_the_audio(tmp_path, space):
    from ltx25_studio import constraints

    assets = tmp_path / "assets"
    assets.mkdir()
    write_wav(assets / "speech.wav", blocks=60)
    frames = constraints.largest_num_frames_within(60 * 1024 / 48000, frame_rate=constraints.DEFAULT_FRAME_RATE)
    call = {"label": "a2v", "mode": "audio_to_video", "audio": "speech.wav", "width": SIZE, "height": SIZE,
            "auto_duration": True}
    (parsed,) = space_matrix.parse_plan([call], base=assets)
    assert parsed.expected_frames() == frames
    assert parsed.app_notes and "length set from your audio" in parsed.app_notes[0]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([call]))
    text = "\n".join(run(["--plan", str(plan), "--assets-dir", str(assets)])[1])
    assert "app       length set from your audio" in text and f'"frames": {frames}' in text

    # Chained audio is only measurable once its producer ran: the gate runs again, live, on the real file.
    wav = write_wav(tmp_path / "made.wav", blocks=60)
    both = write_video(tmp_path / "a2v.mp4", frames=frames, with_audio=True)
    fake = space().reply(Reply(audio=wav), Reply(video=both, audio=both))
    chained = [{"label": "t2a", "mode": "text_to_audio", "prompt": "rain"},
               {**call, "audio": {"from_call": "t2a"}}]
    code, lines, out, _ = live(tmp_path, fake, chained)
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    request = json.loads((out / "a2v" / "request.json").read_text())
    assert request["app_frames"] == frames and "length set from your audio" in request["app_notes"][0]
    assert json.loads((out / "a2v" / "verify.json").read_text())["expected"]["frames"] == frames


# ---------------------------------------------------------------------------
# the dry run: the default, and it spends nothing
# ---------------------------------------------------------------------------


def test_the_dry_run_is_the_default_and_contacts_nothing(tmp_path, space):
    fake = space()
    ledger = tmp_path / "ledger.jsonl"
    seed_ledger(ledger, billed=346.0)
    before = ledger.read_text()
    out = tmp_path / "never"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([t2v(), t2v("s2", enable_stage_2=True, enhance_prompt=True)]))
    code, lines = run(["--plan", str(plan), "--quota-ledger", str(ledger), "--out", str(out)], fake=fake)
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_OK
    assert fake.requests == [], "a dry run must not contact the Space or the Hub"
    assert ledger.read_text() == before and not out.exists()
    assert "DRY RUN" in text and "ZERO GPU-seconds were spent" in text
    assert '"seed": 7' in text and '"width": 64' in text, "the resolved request is printed per call"
    assert "<= 346.0 GPU-s already on the ledger" in text
    assert "remaining    1574.0" in text


def test_the_dry_run_prices_each_call_with_the_apps_estimator(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([
        t2v("base"),
        t2v("s2", enable_stage_2=True),
        t2v("lora", loras=[{"source": "org/x"}]),
        t2v("enh", enhance_prompt=True),
        {"label": "t2a", "mode": "text_to_audio", "prompt": "p", "enable_stage_2": True},
    ]))
    calls = space_matrix.load_plan(plan)
    expected = {
        "base": zerogpu.estimate_duration(gpu_duration_seconds=180),
        "s2": zerogpu.estimate_duration(gpu_duration_seconds=180, enable_stage_2=True),
        "lora": zerogpu.estimate_duration(gpu_duration_seconds=180, loras=["x"]),
        "enh": zerogpu.estimate_duration(gpu_duration_seconds=180, enhance_prompt=True),
        "t2a": zerogpu.estimate_duration(gpu_duration_seconds=180),
    }
    for call in calls:
        assert space_matrix.reservation_seconds(call, base_seconds=180) == expected[call.label]
    assert expected["s2"] == 270 and expected["lora"] == 240 and expected["enh"] == 360
    code, lines = run(["--plan", str(plan)])
    text = "\n".join(lines)
    assert "reserve   270 s (base 180, x1.5 stage 2) -> worst case <= 540 GPU-s" in text
    assert "reserve   360 s (base 180, +180 enhancer) -> worst case <= 720 GPU-s" in text

    assert zerogpu.estimate_duration(
        gpu_duration_seconds=180, enable_stage_2=True, loras=["x"], enhance_prompt=True
    ) == 510


def test_the_multiplier_is_the_apps_own_for_the_spaces_size(tmp_path, monkeypatch):
    assert space_matrix.quota_multiplier() == zerogpu.quota_multiplier("xlarge") == 2
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([t2v()]))
    text = "\n".join(run(["--plan", str(plan), "--zerogpu-size", "large"])[1])
    assert "multiplier x1" in text and "worst case <= 180 GPU-s" in text
    original = zerogpu.quota_multiplier
    monkeypatch.setattr(zerogpu, "quota_multiplier", lambda size=None: 3)
    assert "multiplier x3" in "\n".join(run(["--plan", str(plan)])[1])
    monkeypatch.setattr(zerogpu, "quota_multiplier", original)
    code, lines = run(["--plan", str(plan), "--zerogpu-size", "huge"])
    assert code == space_matrix.EXIT_ILLEGAL and any("--zerogpu-size" in line for line in lines)


def test_the_dry_run_says_where_the_ceiling_binds(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([t2v("a"), t2v("b"), t2v("c")]))
    code, lines = run(["--plan", str(plan), "--quota-ceiling", "800"])
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_OK
    assert "DOES NOT FIT: call #3 (c)" in text and "2026-11-17T00:00:00Z" in text


def test_the_safe_direction_wins_when_both_flags_are_given(tmp_path, space):
    fake = space()
    code, _lines = run(["--live", "--dry-run", "--modes", "text_to_video"], fake=fake)
    assert code == space_matrix.EXIT_OK and fake.requests == []


# ---------------------------------------------------------------------------
# the budget is a hard stop
# ---------------------------------------------------------------------------


def test_live_needs_a_ledger_and_an_evidence_directory(tmp_path, space):
    fake = space()
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps([t2v()]))
    assert run(["--plan", str(plan), "--live", "--out", str(tmp_path / "o")], fake=fake)[0] == 2
    assert run(["--plan", str(plan), "--live", "--quota-ledger", str(tmp_path / "l")], fake=fake)[0] == 2
    assert fake.requests == []


def test_a_call_whose_reservation_would_pass_the_ceiling_is_never_sent(tmp_path, space):
    fake = space()
    ledger = tmp_path / "ledger.jsonl"
    seed_ledger(ledger, billed=1600.0)
    code, lines, _out, _ = live(tmp_path, fake, [t2v()], ledger=ledger)
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_CEILING, text
    assert fake.joins == []
    assert "reserves 180 s x 2 = 360.0" in text and "320.0 GPU-s are left today" in text
    assert "rolls over at 2026-11-17T00:00:00Z (in 8h 30m)" in text


def test_a_reservation_that_exactly_meets_the_ceiling_runs(tmp_path, space):
    fake = space().reply(Reply(video=write_video(tmp_path / "v.mp4")))
    ledger = tmp_path / "ledger.jsonl"
    seed_ledger(ledger, billed=1560.0)
    code, lines, _out, _ = live(tmp_path, fake, [t2v()], ledger=ledger)
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    assert len(fake.joins_for("/on_generate")) == 1


def test_the_ceiling_counts_this_invocations_spend_and_the_next_call_is_refused(tmp_path, space):
    video = write_video(tmp_path / "v.mp4")
    # The first call bills its measured window (~0.1 s here); 1559.99 + 360 fits, plus that does not.
    fake = space(step_delay=0.005).reply(Reply(video=video), Reply(video=video))
    ledger = tmp_path / "ledger.jsonl"
    seed_ledger(ledger, billed=1559.99)
    code, lines, _out, _ = live(tmp_path, fake, [t2v("a"), t2v("b")], ledger=ledger)
    assert code == space_matrix.EXIT_CEILING, "\n".join(lines)
    assert len(fake.joins_for("/on_generate")) == 1
    assert [r["label"] for r in call_records(ledger)] == ["earlier", "a"]


def test_the_ledger_survives_invocations_and_is_append_only(tmp_path, space):
    video = write_video(tmp_path / "v.mp4")
    ledger = tmp_path / "ledger.jsonl"
    fake = space().reply(Reply(video=video), Reply(video=video))
    assert live(tmp_path, fake, [t2v("a")], ledger=ledger)[0] == 0
    first = ledger.read_text()
    assert live(tmp_path, fake, [t2v("b")], ledger=ledger)[0] == 0
    assert ledger.read_text().startswith(first), "records are appended, never rewritten"
    kinds = [r["kind"] for r in ledger_lines(ledger)]
    assert kinds == ["invocation", "call", "invocation", "call"]
    spent = space_matrix.Ledger(ledger).spent("2026-11-16")
    assert spent == pytest.approx(sum(r["billed_conservative"] for r in call_records(ledger)))

    seed_ledger(ledger, billed=1600.0)
    code, lines, _out, _ = live(tmp_path, fake, [t2v("c")], ledger=ledger)
    assert code == space_matrix.EXIT_CEILING, "a second invocation is judged on the first's spend"


def test_a_new_utc_day_starts_from_zero_and_says_what_the_last_day_spent(tmp_path, space, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    seed_ledger(ledger, day="2026-11-16", billed=1900.0)
    monkeypatch.setattr(space_matrix, "_utcnow", lambda: datetime(2026, 11, 17, 0, 5, tzinfo=UTC))
    fake = space().reply(Reply(video=write_video(tmp_path / "v.mp4")))
    code, lines, _out, _ = live(tmp_path, fake, [t2v()], ledger=ledger)
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_OK, text
    assert "UTC 2026-11-16, ended at <= 1900.0" in text
    assert call_records(ledger)[-1]["utc_day"] == "2026-11-17"


@pytest.mark.parametrize("content", ["{ not json", '{"utc_day": "2026-11-16", "spent_upper_bound": 3}\n',
                                     '{"kind": "call", "utc_day": "2026-11-16"}\n'])
def test_an_unreadable_ledger_is_refused_rather_than_read_as_zero(tmp_path, space, content):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(content)
    fake = space()
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps([t2v()]))
    code, lines = run(["--plan", str(plan), "--quota-ledger", str(ledger)])
    assert code == space_matrix.EXIT_ILLEGAL and any("not evidence that nothing was spent" in x for x in lines)
    assert live(tmp_path, fake, [t2v()], ledger=ledger)[0] == space_matrix.EXIT_ILLEGAL
    assert fake.requests == []


@pytest.mark.parametrize("kind", ["quota", "quota_raised"])
def test_a_platform_quota_refusal_stops_everything_at_once(tmp_path, space, kind):
    video = write_video(tmp_path / "v.mp4")
    fake = space().reply(Reply(kind=kind), Reply(video=video))
    code, lines, out, ledger = live(tmp_path, fake, [t2v("a"), t2v("b")], "--keep-going")
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_QUOTA, text
    assert len(fake.joins_for("/on_generate")) == 1, "never retried, never continued"
    assert fake.requests[-1]["path"] == "/gradio_api/queue/data", "nothing at all after the refusal"
    record = call_records(ledger)[-1]
    assert record["outcome"] == "quota_refusal" and record["label"] == "a"
    assert "exceeded your Pro ZeroGPU quota" in record["error"] or "ZeroGPU quota" in record["error"]
    assert json.loads((out / "a" / "result.json").read_text())["outcome"] == "quota"
    assert "THE PLATFORM REFUSED ON QUOTA" in text


@pytest.mark.parametrize(
    "text",
    [
        QUOTA_MESSAGE,
        "You have exceeded your free ZeroGPU quota (120s requested vs. 30s left). Try again in 0:01:00.",
        "You have exceeded your ZeroGPU runs limit. ",
        "Space app has reached its GPU limit. Retry later",
        "You have too many ZeroGPU credits allocated to running tasks.",
        "ZeroGPU quota exceeded",
        "<p>You have exceeded your Pro ZeroGPU quota (360s requested vs. 1s left).</p>",
    ],
)
def test_every_quota_wording_is_recognised(text):
    assert space_matrix.quota_refusal(text)


@pytest.mark.parametrize("text", ["Done. Reserves 180 s more of your quota.", "GPU task aborted", "", None])
def test_other_text_is_not_a_quota_refusal(text):
    assert space_matrix.quota_refusal(text) is None


def test_the_ledger_holds_reservation_conservative_and_worker_figures(tmp_path, space):
    fake = space(step_delay=0.01).reply(
        Reply(video=write_video(tmp_path / "v.mp4"), record={"worker_elapsed_seconds": 12.5, "category": None})
    )
    code, lines, out, ledger = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    record = call_records(ledger)[-1]
    timing = json.loads((out / "t2v" / "timing.json").read_text())
    assert record["reserved_seconds"] == 180 and record["multiplier"] == 2.0
    assert record["billed_worker"] == pytest.approx(25.0)
    assert timing["window_seconds"] > 0
    assert record["billed_conservative"] == pytest.approx(timing["window_seconds"] * 2, abs=1e-3)
    assert timing["generate_process_starts_t"] < timing["generate_completed_t"]
    assert space_matrix.Ledger(ledger).spent("2026-11-16") == pytest.approx(record["billed_conservative"])


def test_the_reports_gpu_time_is_read_when_no_record_is_published(tmp_path, space):
    """Today the run record reaches the API only as the report's "Took N s, GPU time" line."""
    gpu = '<dl class="mr-done-grid"><dt>Took</dt><dd><span class="mr-num">41</span> s, GPU time</dd></dl>'
    wall = '<dl class="mr-done-grid"><dt>Took</dt><dd><span class="mr-num">55</span> s, wall-clock</dd></dl>'
    video = write_video(tmp_path / "v.mp4")
    fake = space().reply(Reply(video=video, status=gpu), Reply(video=video, status=wall))
    code, lines, out, ledger = live(tmp_path, fake, [t2v("gpu"), t2v("wall")])
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    first, second = call_records(ledger)[-2:]
    assert first["billed_worker"] == pytest.approx(82.0) and second["billed_worker"] is None
    assert "whole seconds" in json.loads((out / "gpu" / "timing.json").read_text())["worker_seconds_source"]


# ---------------------------------------------------------------------------
# failures: recorded verbatim, stop by default
# ---------------------------------------------------------------------------


def test_a_timeout_is_recorded_charged_its_reservation_and_never_retried(tmp_path, space):
    fake = space().reply(Reply(kind="timeout"))
    code, lines, out, ledger = live(tmp_path, fake, [t2v("a"), t2v("b")], "--call-timeout", "0.4",
                                    "--read-timeout", "0.3")
    assert code == space_matrix.EXIT_CALL_FAILED, "\n".join(lines)
    assert len(fake.joins_for("/on_generate")) == 1
    record = call_records(ledger)[-1]
    assert record["outcome"] == "timeout" and record["billed_conservative"] == pytest.approx(360.0)
    assert json.loads((out / "a" / "result.json").read_text())["outcome"] == "timeout"


def test_a_typed_app_error_is_recorded_verbatim_with_its_category(tmp_path, space):
    card = ('### The model stopped while generating\n\nAny GPU time this run used is billed.\n\n'
            '<details class="mr-detail"><summary>Detail</summary><p>CUDA out of memory at 1408x960</p></details>')
    fake = space().reply(Reply(kind="app_error", status=card, record={"category": "oom"}))
    code, lines, out, ledger = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_CALL_FAILED
    result = json.loads((out / "t2v" / "result.json").read_text())
    assert result["status_markdown"].startswith(card) and result["category"] == "oom"
    assert "CUDA out of memory at 1408x960" in result["detail"]
    assert call_records(ledger)[-1]["category"] == "oom"


def test_a_foreign_error_is_recorded_verbatim(tmp_path, space):
    fake = space().reply(Reply(kind="foreign_error", message="RuntimeError: CUDA error: illegal memory access"))
    code, lines, out, ledger = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_CALL_FAILED
    result = json.loads((out / "t2v" / "result.json").read_text())
    assert result["error"] == "RuntimeError: CUDA error: illegal memory access"
    assert call_records(ledger)[-1]["error"] == "RuntimeError: CUDA error: illegal memory access"


def test_a_success_shaped_reply_with_no_file_is_a_failure(tmp_path, space):
    fake = space().reply(Reply(kind="no_artifact", status="### The run finished but produced no file"))
    code, lines, out, _ = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_CALL_FAILED
    assert "no video file" in json.loads((out / "t2v" / "result.json").read_text())["detail"]


def test_a_failure_stops_the_plan_unless_keep_going(tmp_path, space):
    video = write_video(tmp_path / "v.mp4")
    fake = space().reply(Reply(kind="foreign_error", message="boom"), Reply(video=video))
    code, _lines, _out, _ = live(tmp_path, fake, [t2v("a"), t2v("b")])
    assert code == space_matrix.EXIT_CALL_FAILED and len(fake.joins_for("/on_generate")) == 1
    fake.replies.clear()
    fake.reply(Reply(kind="foreign_error", message="boom"), Reply(video=video))
    code, _lines, out, _ = live(tmp_path, fake, [t2v("a"), t2v("b")], "--keep-going")
    assert code == space_matrix.EXIT_CALL_FAILED and len(fake.joins_for("/on_generate")) == 3
    assert json.loads((out / "summary.json").read_text())["calls"][1]["status"] == "ok"


def test_a_space_that_is_not_running_is_not_called(tmp_path, space):
    fake = space(stage="SLEEPING")
    code, lines, _out, _ = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_ILLEGAL and "SLEEPING" in "\n".join(lines)
    assert fake.paths() == [f"/api/spaces/{fake.space_id}"]


# ---------------------------------------------------------------------------
# LoRA through the public API, in the same session as the run
# ---------------------------------------------------------------------------


def test_loras_are_validated_and_staged_in_the_generate_session(tmp_path, space):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "style.safetensors").write_bytes(b"adapter-bytes")
    fake = space().reply(Reply(video=write_video(tmp_path / "v.mp4")))
    call = t2v(loras=[{"source": "SOLRICKS/LTX-2.5-BTS-Movie-Set"}, {"source": "style.safetensors"}])
    code, lines, out, ledger = live(tmp_path, fake, [call], assets=assets)
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    endpoints = [j["endpoint"] for j in fake.joins]
    assert endpoints == ["/on_validate_lora", "/on_validate_lora", "/on_generate"]
    assert len({j["session_hash"] for j in fake.joins}) == 1, "one session, so the staged State reaches the run"
    assert [e["source"] for e in fake.generate_calls[0]["loras"]] == ["SOLRICKS/LTX-2.5-BTS-Movie-Set",
                                                                     "style.safetensors"]
    hub, upload = fake.lora_calls
    assert hub["params"]["repo_id"] == "SOLRICKS/LTX-2.5-BTS-Movie-Set" and hub["params"]["upload"] is None
    assert fake.uploaded(upload["params"]["upload"]["path"]).read_bytes() == b"adapter-bytes"
    request = json.loads((out / "t2v" / "request.json").read_text())
    assert [v["staged"] for v in request["loras"]] == [True, True]
    assert request["loras"][0]["verdict"].startswith("#### Accepted")
    assert call_records(ledger)[-1]["reserved_seconds"] == 240
    assert fake.protocol_errors == []


def test_a_rejected_lora_stops_the_call_before_any_gpu_time(tmp_path, space):
    fake = space().lora(("reject", "only 26 of 1660 layers resolve"))
    code, lines, out, ledger = live(tmp_path, fake, [t2v(loras=[{"source": "org/wrong"}])])
    assert code == space_matrix.EXIT_CALL_FAILED
    assert fake.joins_for("/on_generate") == []
    record = call_records(ledger)[-1]
    assert record["billed_conservative"] == 0.0 and "only 26 of 1660" in record["error"]
    assert json.loads((out / "t2v" / "result.json").read_text())["loras"][0]["staged"] is False


def test_a_lora_strength_with_no_public_route_is_refused_before_anything_runs(tmp_path, space):
    fake = space()
    code, lines, _out, _ = live(tmp_path, fake, [t2v(loras=[{"source": "org/style", "weight": 0.8}])])
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_ILLEGAL
    assert "no public way to set an adapter's strength" in text and "/on_lora_weight" in text
    assert fake.joins == []


def test_a_lora_strength_is_set_through_a_published_route(tmp_path, space):
    fake = space(lora_weight_route=True).reply(Reply(video=write_video(tmp_path / "v.mp4")))
    code, lines, _out, _ = live(tmp_path, fake, [t2v(loras=[{"source": "org/style", "weight": 0.8}])])
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    assert [j["endpoint"] for j in fake.joins] == ["/on_validate_lora", "/on_lora_weight", "/on_generate"]
    assert fake.generate_calls[0]["loras"] == [{"source": "org/style", "weight": 0.8}]


# ---------------------------------------------------------------------------
# evidence per call
# ---------------------------------------------------------------------------


def test_every_call_leaves_its_evidence(tmp_path, space):
    fake = space(step_delay=0.003).reply(Reply(video=write_video(tmp_path / "v.mp4", with_audio=True),
                                               audio=None))
    fake.replies[0].audio = fake.replies[0].video
    call = {"label": "t2av", "mode": "text_to_audio_video", "prompt": "a captain", "width": SIZE,
            "height": SIZE, "num_frames": 9, "seed": 3, "expect": {"has_audio": True}}
    code, lines, out, _ = live(tmp_path, fake, [call])
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    folder = out / "t2av"
    names = {p.name for p in folder.iterdir()}
    assert {"request.json", "events.jsonl", "result.json", "verify.json", "timing.json"} <= names
    assert any(name.endswith(".mp4") for name in names)
    request = json.loads((folder / "request.json").read_text())
    assert request["fields"]["prompt"] == "a captain" and request["mode"] == "text_to_audio_video"
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    messages = [e.get("msg") for e in events if e.get("msg")]
    for wanted in ("estimation", "process_starts", "log", "progress", "heartbeat", "process_completed"):
        assert wanted in messages
    assert messages.count("progress") == 8
    times = [e["t"] for e in events]
    assert times == sorted(times)
    assert any(e.get("progress_data", [{}])[0].get("desc") == "Denoising · pass 1 of 1" for e in events
               if e.get("msg") == "progress")
    verify = json.loads((folder / "verify.json").read_text())
    assert verify["ok"] and verify["expected"]["geometry"] == [SIZE, SIZE] and verify["decoded_frames"] == 9
    assert {c["name"] for c in verify["content"]["video"]["checks"]} >= {"freeze_floor", "motion"}
    assert "conditions" in verify["content"]["video"]["picture_health"]
    (track,) = verify["content"]["audio"]["tracks"]
    assert {"level_dbfs", "peak", "clipped_samples", "silence_ratio", "spectral_flatness",
            "spectral_centroid_hz", "band_energy_share"} <= set(track)
    result = json.loads((folder / "result.json").read_text())
    assert set(result["artifacts"]) == {"video", "audio"}
    assert result["artifacts"]["video"]["path"] == result["artifacts"]["audio"]["path"], "downloaded once"
    assert json.loads((out / "summary.json").read_text())["calls"][0]["status"] == "ok"


def test_stage_2_is_expected_at_twice_the_request(tmp_path, space):
    big = write_video(tmp_path / "big.mp4", size=2 * SIZE)
    small = write_video(tmp_path / "small.mp4")
    fake = space().reply(Reply(video=big))
    code, lines, out, _ = live(tmp_path, fake, [t2v(enable_stage_2=True)])
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    assert json.loads((out / "t2v" / "verify.json").read_text())["expected"]["geometry"] == [128, 128]
    fake.reply(Reply(video=small))
    code, lines, out, _ = live(tmp_path, fake, [t2v(enable_stage_2=True)])
    assert code == space_matrix.EXIT_CALL_FAILED
    assert any("expected 128x128" in p for p in json.loads((out / "t2v" / "verify.json").read_text())["problems"])


def test_a_frame_count_that_misses_the_expectation_is_a_mismatch(tmp_path, space):
    fake = space().reply(Reply(video=write_video(tmp_path / "v.mp4", frames=17)))
    code, _lines, out, _ = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_CALL_FAILED
    assert "decoded 17 frames, expected 9" in json.loads((out / "t2v" / "verify.json").read_text())["problems"]


def test_a_picture_condition_is_recorded_and_shown_and_never_stops_the_plan(tmp_path, space, monkeypatch):
    """The app's check is recorded as it says it; a fired condition is shown, the frames are looked at."""
    from ltx25_studio import outputs

    checked = outputs.assess_picture(moving_stack())
    monkeypatch.setattr(
        outputs, "assess_picture",
        lambda frames, **kwargs: replace(checked, conditions=("static: a fired test condition",)),
    )
    video = write_video(tmp_path / "v.mp4")
    fake = space().reply(Reply(video=video), Reply(video=video))
    code, lines, out, _ = live(tmp_path, fake, [t2v("a"), t2v("b")])
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    health = json.loads((out / "a" / "verify.json").read_text())["content"]["video"]["picture_health"]
    assert health["conditions"] == ["static: a fired test condition"]
    assert len(fake.joins_for("/on_generate")) == 2
    assert json.loads((out / "summary.json").read_text())["calls"][0]["status"] == "ok"
    assert any("picture: static: a fired test condition" in line for line in lines)


def test_a_chained_input_comes_from_the_named_call_and_says_so(tmp_path, space):
    fake = space().reply(Reply(video=write_video(tmp_path / "v.mp4")), Reply(video=write_video(tmp_path / "w.mp4")))
    calls = [t2v(), {"label": "i2v", "mode": "image_to_video", "width": SIZE, "height": SIZE, "num_frames": 9,
                     "image": {"from_call": "t2v", "frame": 0}}]
    code, lines, out, _ = live(tmp_path, fake, calls)
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    media = json.loads((out / "i2v" / "request.json").read_text())["media"]["image"]
    assert media["chained"] is True and media["from_call"] == "t2v" and media["frame_index"] == 0
    image = fake.generate_calls[1]["params"]["image"]
    assert image["orig_name"] == "image.png" and fake.uploaded(image["path"]).read_bytes()[:4] == b"\x89PNG"
    assert fake.generate_calls[1]["params"]["video"] is None


def test_media_from_the_assets_directory_is_uploaded_into_its_own_slot(tmp_path, space):
    assets = tmp_path / "assets"
    assets.mkdir()
    write_wav(assets / "speech.wav")
    fake = space().reply(Reply(video=write_video(tmp_path / "v.mp4", with_audio=True)))
    fake.replies[0].audio = fake.replies[0].video
    call = {"label": "a2v", "mode": "audio_to_video", "audio": "speech.wav", "width": SIZE, "height": SIZE,
            "num_frames": 9, "noise_scale": 0.4}
    code, lines, out, _ = live(tmp_path, fake, [call], assets=assets)
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    params = fake.generate_calls[0]["params"]
    assert params["audio"]["orig_name"] == "speech.wav" and params["image"] is None and params["noise_scale"] == 0.4
    assert json.loads((out / "a2v" / "request.json").read_text())["media"]["audio"]["chained"] is False


def test_evidence_never_mixes_with_an_earlier_run(tmp_path, space):
    fake = space()
    out = tmp_path / "out"
    (out / "t2v").mkdir(parents=True)
    (out / "t2v" / "request.json").write_text("{}")
    code, lines, _out, _ = live(tmp_path, fake, [t2v()], out=out)
    assert code == space_matrix.EXIT_ILLEGAL and fake.requests == []


# ---------------------------------------------------------------------------
# read-only subcommands: one GET each
# ---------------------------------------------------------------------------


def test_status_is_one_get(space):
    fake = space(stage="RUNNING", sha="f80947294c994370aebdf164df17bacb1f2c270d")
    code, lines = run(["status"], fake=fake)
    text = "\n".join(lines)
    assert code == 0
    assert [(r["method"], r["path"]) for r in fake.requests] == [("GET", f"/api/spaces/{fake.space_id}")]
    assert "stage      RUNNING" in text and "f80947294c99" in text and "zero-a10g" in text


def test_logs_is_one_bounded_get(tmp_path, space):
    fake = space(log_lines=("===== Application Startup =====", "Running on local URL"), logs_hold=True)
    target = tmp_path / "logs" / "run.log"
    code, lines = run(["logs", "--out", str(target), "--max-seconds", "0.3"], fake=fake)
    assert code == 0
    assert [(r["method"], r["path"]) for r in fake.requests] == [("GET", f"/api/spaces/{fake.space_id}/logs/run")]
    assert "Running on local URL" in target.read_text()


def test_info_is_one_get_and_reports_a_mismatch(space):
    fake = space()
    code, lines = run(["info"], fake=fake)
    assert code == 0 and "MATCHES" in "\n".join(lines)
    assert [(r["method"], r["path"]) for r in fake.requests] == [("GET", "/gradio_api/info")]

    def swap(info):
        params = info["named_endpoints"]["/on_generate"]["parameters"]
        params[1]["component"] = "Number"
        return info

    other = space(info_patch=swap)
    code, lines = run(["info"], fake=other)
    assert code == space_matrix.EXIT_ILLEGAL and any("✗" in line for line in lines)


# ---------------------------------------------------------------------------
# the token: read at call time, sent only as the Authorization header, never written down
# ---------------------------------------------------------------------------


def test_the_token_is_read_when_needed_and_falls_back_to_the_cached_file(tmp_path, space, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    assert space_matrix.read_token() is None
    home = tmp_path / "hf-home"
    home.mkdir(exist_ok=True)
    (home / "token").write_text(SENTINEL + "\n")
    assert space_matrix.read_token().reveal() == SENTINEL
    fake = space()
    assert run(["status"], fake=fake)[0] == 0
    assert fake.requests[0]["headers"]["authorization"] == f"Bearer {SENTINEL}"


def test_a_secret_cannot_be_printed_or_pickled():
    secret = space_matrix.Secret(SENTINEL)
    assert SENTINEL not in repr(secret) and SENTINEL not in str(secret) and SENTINEL not in f"{secret}"
    with pytest.raises(TypeError):
        pickle.dumps(secret)


def test_a_rejected_token_leaves_no_trace_in_the_error(tmp_path, space):
    fake = space(token="hf_" + "the_right_one")
    code, lines = run(["status"], fake=fake)
    assert code == 1 and any("HTTP 401" in line for line in lines)
    assert SENTINEL not in "\n".join(lines)


def test_no_token_means_no_live_run(tmp_path, space, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    fake = space()
    code, lines, _out, _ = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_ILLEGAL and fake.requests == []


def test_the_token_appears_nowhere_on_any_path(tmp_path, space):
    """Every path, with a sentinel token: no output line, file or record may contain it."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "a.safetensors").write_bytes(b"x")
    video = write_video(tmp_path / "v.mp4")
    lines: list[str] = []
    outs: list[Path] = []
    ledger = tmp_path / "ledger.jsonl"

    def go(fake, calls, *extra):
        code, got, out, _ = live(tmp_path, fake, calls, *extra, ledger=ledger, assets=assets)
        lines.extend(got)
        outs.append(out)
        return code

    leaky = SENTINEL + " leaked"
    fake = space(log_lines=(f"token={SENTINEL}", "fine"))
    fake.reply(Reply(video=video, status=f"Done. {SENTINEL}"), Reply(kind="timeout"),
               Reply(kind="app_error", message=leaky), Reply(kind="foreign_error", message=leaky),
               Reply(kind="quota", message=QUOTA_MESSAGE + SENTINEL))
    assert go(fake, [t2v("ok", loras=[{"source": "a.safetensors"}, {"source": "org/x"}])]) == 0
    assert go(fake, [t2v("slow")], "--call-timeout", "0.3", "--read-timeout", "0.2") == 5
    assert go(fake, [t2v("typed")]) == 5
    assert go(fake, [t2v("foreign")]) == 5
    assert go(fake, [t2v("quota")]) == 4
    lines.extend(run(["status"], fake=fake)[1])
    lines.extend(run(["info"], fake=fake)[1])
    log_file = tmp_path / "run.log"
    lines.extend(run(["logs", "--out", str(log_file), "--max-seconds", "0.3"], fake=fake)[1])
    mismatch = space(info_patch=lambda info: info["named_endpoints"]["/on_generate"]["parameters"].pop() and info)
    go(mismatch, [t2v("mismatch")])
    lines.extend(run(["--modes", "text_to_video"], fake=fake)[1])

    written = [log_file, ledger, *(p for out in outs if out.exists() for p in out.rglob("*") if p.is_file())]
    for path in written:
        assert SENTINEL not in path.read_bytes().decode("utf-8", "replace"), f"{path} carries the token"
    assert SENTINEL not in "\n".join(lines)
    assert any("<redacted>" in path.read_text(errors="replace") for path in written)
    for request in fake.requests:
        assert request["headers"].get("authorization") == f"Bearer {SENTINEL}"
        assert SENTINEL not in request["raw_path"] and SENTINEL.encode() not in request["body"]
        assert not any(SENTINEL in v for k, v in request["headers"].items() if k != "authorization")


def test_the_source_holds_no_token():
    source = (REPO_ROOT / "scripts" / "space_matrix.py").read_text()
    assert "hf_" not in source.lower().replace("hf_token", "").replace("hf_home", "").replace("hf_hub", "")


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_help_documents_every_flag_and_the_exit_codes():
    parser = space_matrix._run_parser()
    text = parser.format_help()
    for action in parser._actions:
        for flag in action.option_strings:
            assert flag in text
        if action.option_strings and action.dest != "help":
            assert action.help, f"{action.option_strings} has no help"
    for needle in ("0 plan complete", "2 illegal plan", "3 stopped at", "4 platform quota", "5 a call failed",
                   "status", "info", "logs --out FILE"):
        assert needle in text
    for name in space_matrix.SUBCOMMANDS:
        sub = space_matrix._sub_parser(name)
        assert all(a.help for a in sub._actions if a.option_strings and a.dest != "help")


# ---------------------------------------------------------------------------
# content measurements: the harness's frame checks, the app's picture check verbatim, audio as numbers
# ---------------------------------------------------------------------------


def checks(path):
    return {c["name"]: c["verdict"] for c in space_matrix.measure_video(path)["checks"]}


def the_apps_own_record(path):
    from ltx25_studio import outputs

    return outputs.assess_picture(space_matrix._decode_video_frames(path))


def test_a_moving_structured_video_passes_every_frame_check(tmp_path):
    path = write_video(tmp_path / "v.mp4")
    measured = space_matrix.measure_video(path)
    assert measured["decoded_frames"] == 9
    assert {c["name"]: c["verdict"] for c in measured["checks"]} == {
        "flatness": "pass", "distinct_frames": "pass", "freeze_floor": "pass", "motion": "pass",
    }
    assert all(c["threshold"] for c in measured["checks"])


def test_the_picture_check_is_the_apps_own_recorded_verbatim(tmp_path):
    """Measurements, conditions and notes exactly as ``outputs`` returns them; no threshold of its own."""
    for name, stack in (("static", structureless_stack()), ("murky", murky_stack()), ("moving", moving_stack())):
        path = write_frames(tmp_path / f"{name}.mp4", stack)
        checked = the_apps_own_record(path)
        health = space_matrix.measure_video(path)["picture_health"]
        assert health["conditions"] == list(checked.conditions) and health["notes"] == list(checked.notes)
        assert health["measurements"] == dataclasses.asdict(checked.measurements)
    source = (REPO_ROOT / "scripts" / "space_matrix.py").read_text()
    for name in ("MIN_STRUCTURE_RATIO", "MAX_HIGH_FREQUENCY_RATIO", "LOW_BLOCK_STD", "LOW_LUMA_BAND",
                 "FLAT_MAX_LUMA_STD", "VEILED_", "STATIC_M", "structure_failure"):
        assert name not in source, f"the harness must not hold the picture check's {name}"
    assert source.count("outputs.assess_picture(") == 1, "one wrapper is the only call site"


def test_a_frozen_render_fails_the_freeze_floor_that_byte_identity_misses(tmp_path):
    """One frame held for the whole clip: h264 decodes it back as several distinct frames."""
    held = write_frames(tmp_path / "held.mp4", np.repeat(moving_stack(1), 9, axis=0))
    measured = space_matrix.measure_video(held)
    assert measured["distinct_frames"] >= space_matrix.MIN_DISTINCT_FRAMES, "codec noise defeats byte identity"
    verdicts = checks(held)
    assert verdicts["distinct_frames"] == "pass"
    assert verdicts["freeze_floor"] == "fail", "the scale-free floor is what sees the freeze"
    assert checks(write_frames(tmp_path / "smooth.mp4", frozen_stack()))["freeze_floor"] == "fail"
    assert checks(write_frames(tmp_path / "slow.mp4", barely_moving_stack()))["freeze_floor"] == "pass"


def test_a_blank_video_fails_flatness_and_the_freeze_floor_is_not_measured(tmp_path):
    verdicts = checks(write_frames(tmp_path / "blank.mp4", np.zeros((9, SIZE, SIZE, 3))))
    assert verdicts["flatness"] == "fail" and verdicts["freeze_floor"] == "n/a"


def test_colour_static_passes_every_frame_check(tmp_path):
    """Static is distinct, full-range and moving: only the picture check can see it."""
    verdicts = checks(write_frames(tmp_path / "static.mp4", structureless_stack()))
    assert set(verdicts.values()) == {"pass"}


def test_a_near_static_render_only_warns(tmp_path):
    verdicts = checks(write_frames(tmp_path / "slow.mp4", barely_moving_stack()))
    assert verdicts["motion"] == "warn" and verdicts["freeze_floor"] == "pass"


def test_the_verdict_does_not_move_when_the_scene_is_merely_dimmer(tmp_path):
    bright = checks(write_frames(tmp_path / "b.mp4", barely_moving_stack()))
    dim = checks(write_frames(tmp_path / "d.mp4", barely_moving_stack(scale=0.35)))
    assert bright["freeze_floor"] == dim["freeze_floor"] == "pass"


def test_an_undecodable_file_is_a_failed_measurement_not_a_crash(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert space_matrix.measure_video(junk)["checks"][0]["verdict"] == "fail"


def write_samples(path, samples, *, rate=48000):
    """A pcm_s16le wav holding exactly ``samples`` (``[channels, n]`` floats in -1..1)."""
    samples = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    layout = "stereo" if samples.shape[0] == 2 else "mono"
    with av.open(str(path), "w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout=layout)
        for start in range(0, samples.shape[1], 1024):
            block = np.ascontiguousarray(samples[:, start : start + 1024])
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
            frame.rate, frame.pts, frame.time_base = rate, start, Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return Path(path)


def track_of(path):
    (track,) = space_matrix.measure_audio(path)["tracks"]
    return track


def test_audio_is_measured_not_judged(tmp_path):
    rate, n = 48000, 48000
    t = np.arange(n) / rate
    tone = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    noise = np.random.default_rng(3).normal(0.0, 0.1, n)
    burst = np.concatenate([np.zeros(n // 2), np.sign(np.sin(2 * np.pi * 220.0 * t[: n // 2]))])

    tonal = track_of(write_samples(tmp_path / "tone.wav", np.stack([tone, tone])))
    hiss = track_of(write_samples(tmp_path / "hiss.wav", np.stack([noise, noise])))
    clipped = track_of(write_samples(tmp_path / "burst.wav", np.stack([burst, burst])))
    silent = track_of(write_samples(tmp_path / "silent.wav", np.zeros((2, n))))

    assert tonal["sample_rate"] == 48000 and tonal["channels"] == 2 and -12 < tonal["level_dbfs"] < -10
    assert tonal["clipped_samples"] == 0 and tonal["silence_ratio"] == 0.0
    assert tonal["spectral_flatness"] < 0.05 and 400 < tonal["spectral_centroid_hz"] < 480
    assert tonal["band_energy_share"]["250-2000"] > 0.95
    assert hiss["spectral_flatness"] > 0.8, "white noise has a flat spectrum"
    assert hiss["spectral_centroid_hz"] > 8000 and hiss["band_energy_share"]["6000-nyquist"] > 0.6
    assert clipped["clipped_samples"] > 1000 and clipped["peak_dbfs"] > -0.1
    assert 0.45 < clipped["silence_ratio"] < 0.55
    assert silent["level_dbfs"] < space_matrix.SILENT_DBFS and silent["spectral_flatness"] is None
    for track in (tonal, hiss, clipped, silent):
        assert not any(key in track for key in ("verdict", "checks"))


def test_only_digital_silence_fails_a_call_on_audio_content(tmp_path, space):
    """Noise, clipping and a quiet track are reported; silence, a missing track or a wrong rate fail."""
    n = 48000
    hiss = write_samples(tmp_path / "hiss.wav", np.random.default_rng(1).normal(0, 0.02, (2, n)))
    silent = write_samples(tmp_path / "silent.wav", np.zeros((2, n)))
    mono = write_samples(tmp_path / "mono.wav", np.random.default_rng(2).normal(0, 0.1, (1, n)))
    fake = space().reply(Reply(audio=hiss), Reply(audio=silent), Reply(audio=mono))
    t2a = {"mode": "text_to_audio", "prompt": "rain"}
    code, lines, out, _ = live(tmp_path, fake, [{"label": "hiss", **t2a}, {"label": "silent", **t2a},
                                                {"label": "mono", **t2a}], "--keep-going")
    assert code == space_matrix.EXIT_CALL_FAILED
    statuses = {c["label"]: c["status"] for c in json.loads((out / "summary.json").read_text())["calls"]}
    assert statuses == {"hiss": "ok", "silent": "mismatch", "mono": "mismatch"}
    assert any("digital silence" in p for p in json.loads((out / "silent" / "verify.json").read_text())["problems"])
    assert any("audio  " in line and "flatness" in line for line in lines)


# ---------------------------------------------------------------------------
# real Space renders (F42): set LTX25_BANKED_ARTIFACTS_DIR to run these
#
# The folder holds real Space outputs that have been LOOKED AT and are pictures:
#
#   still_render.mp4          a still-image-conditioned run (e.g. image_to_audio_video,
#                             512x960x121) that moves little: the class the old absolute
#                             motion gate refused
#   still_render_sibling.mp4  a second real render of the same kind (another seed or run)
#
# e.g. LTX25_BANKED_ARTIFACTS_DIR=<lead's evidence>/space
# ---------------------------------------------------------------------------

BANKED_ENV = "LTX25_BANKED_ARTIFACTS_DIR"
BANKED_FILES = ("still_render.mp4", "still_render_sibling.mp4")


def banked(name: str) -> Path:
    import os

    folder = os.environ.get(BANKED_ENV)
    if not folder:
        pytest.skip(f"set {BANKED_ENV} to a folder holding {', '.join(BANKED_FILES)} (real Space renders)")
    path = Path(folder).expanduser() / name
    if not path.is_file():
        pytest.skip(f"{BANKED_ENV}={folder} has no {name}")
    return path


@pytest.mark.parametrize("name", BANKED_FILES)
def test_a_real_still_render_passes_the_freeze_floor_and_motion_only_warns(name):
    measured = space_matrix.measure_video(banked(name))
    verdicts = {c["name"]: c["verdict"] for c in measured["checks"]}
    assert verdicts["flatness"] == "pass" and verdicts["freeze_floor"] == "pass"
    assert verdicts["motion"] in ("pass", "warn"), "a still scene is reported, never refused"
    if measured["mean_abs_delta"] < space_matrix.LOW_MEAN_FRAME_DELTA:
        assert verdicts["motion"] == "warn"


def test_a_frozen_copy_of_a_real_frame_fails_the_freeze_floor_where_the_render_does_not(tmp_path):
    source = banked(BANKED_FILES[0])
    frames = space_matrix._decode_video_frames(source)
    held = write_frames(tmp_path / "held.mp4", np.stack([frames[0]] * len(frames)))
    measured = space_matrix.measure_video(held)
    assert measured["distinct_frames"] >= space_matrix.MIN_DISTINCT_FRAMES, "byte identity cannot see it"
    assert checks(held)["freeze_floor"] == "fail" and checks(source)["freeze_floor"] == "pass"


def test_the_example_plan_resolves_against_the_lead_assets_layout(tmp_path):
    """Every call of the shipped example is legal, and every mode appears."""
    example = REPO_ROOT / "scripts" / "space_plan.example.json"
    document = json.loads(example.read_text())
    paths = set()
    for call in document["calls"]:
        for kind in ("image", "video", "audio"):
            if isinstance(call.get(kind), str):
                paths.add(call[kind])
        for lora in call.get("loras", []):
            if lora["source"].endswith(".safetensors"):
                paths.add(lora["source"])
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".wav":
            write_wav(target)
        elif target.suffix == ".mp4":
            write_video(target)
        elif target.suffix == ".png":
            write_png(target)
        else:
            target.write_bytes(b"x")
    calls = space_matrix.load_plan(example, assets_dir=tmp_path)
    assert {c.mode for c in calls} == set(modes.MODES)
    assert any(c.stage_2 for c in calls) and any(c.loras for c in calls) and any(c.value("auto_duration") for c in calls)
