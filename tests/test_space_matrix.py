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
        # A chained input does not exist yet: the app's gate runs on it once it does.
        assert call.gate == (space_matrix.GATE_DEFERRED if call.media else space_matrix.GATE_ACCEPTED), call.label
    assert calls[0].render_size == (space_matrix.CHEAPEST_SIZE, space_matrix.CHEAPEST_SIZE)
    code, lines = run(["--plan", "default"])
    assert code == space_matrix.EXIT_OK
    assert sum("app gate  checked when it exists" in line for line in lines) == sum(bool(c.media) for c in calls)


def test_a_chained_input_waits_for_its_file_and_real_files_still_pass_the_gate(tmp_path, monkeypatch):
    """Deferring a chained input never weakens the check of a file that exists, and a call whose
    inputs all exist always goes through ``generate.prepare_request``."""
    from ltx25_studio import generate

    (tmp_path / "not-a-video.mp4").write_bytes(b"not a video")
    write_video(tmp_path / "clip.mp4")
    seen: list[str] = []
    real = generate.prepare_request
    monkeypatch.setattr(generate, "prepare_request", lambda request: (seen.append(request.mode), real(request))[1])
    v2v = {"label": "v2v", "mode": "video_to_video", "width": SIZE, "height": SIZE, "noise_scale": 0.725}
    calls = space_matrix.parse_plan(
        [t2v(), {**v2v, "video": {"from_call": "t2v"}}, {**v2v, "label": "file", "video": "clip.mp4"}], base=tmp_path)
    assert [c.gate for c in calls] == [space_matrix.GATE_ACCEPTED, space_matrix.GATE_DEFERRED,
                                       space_matrix.GATE_ACCEPTED]
    assert seen == ["text_to_video", "video_to_video"], "the gate ran on every call whose inputs exist"
    assert calls[2].app_frames == 9 and calls[1].expected_frames() is None, "a clip sets the length once it exists"
    with pytest.raises(space_matrix.PlanError) as caught:
        space_matrix.parse_plan([t2v(), {**v2v, "video": {"from_call": "t2v"}},
                                 {**v2v, "label": "file", "video": "not-a-video.mp4"}], base=tmp_path)
    assert "file" in str(caught.value) and "could not be opened as a video" in str(caught.value)
    assert "'v2v'" not in str(caught.value)


def test_a_chained_input_is_gated_on_its_file_before_anything_is_sent(tmp_path, space):
    """Live, the app's gate runs on the produced file; a refusal there sends nothing and bills nothing."""
    short = write_wav(tmp_path / "short.wav", blocks=16)  # 0.34 s
    fake = space().reply(Reply(audio=short))
    calls = [{"label": "t2a", "mode": "text_to_audio", "prompt": "rain"},
             {"label": "a2v", "mode": "audio_to_video", "audio": {"from_call": "t2a"}, "width": SIZE,
              "height": SIZE, "num_frames": 121}]
    code, lines, out, ledger = live(tmp_path, fake, calls)
    assert code == space_matrix.EXIT_CALL_FAILED, "\n".join(lines)
    assert len(fake.joins_for("/on_generate")) == 1 and sum(p == "/gradio_api/upload" for p in fake.paths()) == 0
    record = call_records(ledger)[-1]
    assert record["label"] == "a2v" and record["billed_conservative"] == 0.0 and record["category"] == "input"
    result = json.loads((out / "a2v" / "result.json").read_text())
    assert (result["sent"], result["category"]) == (False, "input") and result["detail"].endswith(
        "Your audio is 0.3 s, too short for Audio to video: the shortest recording it can use is 0.4 s."
    )


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


@pytest.mark.parametrize("device", ["cpu", "mps", None])
def test_the_gate_runs_as_the_space_whatever_this_machine_is(device, monkeypatch):
    """Stage 2 and the diffusion decoder are refused off CUDA. The gate answers for the Space
    (ZeroGPU, CUDA), never for this machine's LTX25_DEVICE, and leaves this machine as it was."""
    from ltx25_studio import device as device_module
    from ltx25_studio import generate

    if device is None:
        monkeypatch.delenv("LTX25_DEVICE", raising=False)
    else:
        monkeypatch.setenv("LTX25_DEVICE", device)
        monkeypatch.setattr(device_module, "host_has_mps", lambda: True)
    before = device_module.current
    stage_2 = t2v("s2", width=704, height=480, num_frames=49, enable_stage_2=True)
    best = t2v("best", width=704, height=480, num_frames=121, decoder="diffusion")
    for call in space_matrix.parse_plan([stage_2, best], base=Path.cwd()):
        assert call.gate == space_matrix.GATE_ACCEPTED
    assert device_module.current is before
    if device is not None:
        (call,) = space_matrix.parse_plan([stage_2], base=Path.cwd())
        with pytest.raises(Exception, match="CUDA"):
            generate.prepare_request(space_matrix._app_request(call, {}))


def test_a_choice_is_sent_by_its_value_whichever_name_the_plan_uses(tmp_path, space):
    # Above the diffusion decoder's smallest size, which the gate checks where its config is cached.
    fake = space().reply(*[Reply(video=write_video(tmp_path / "v.mp4", frames=17, size=256))] * 3)
    shape = {"width": 256, "height": 256, "num_frames": 17}
    calls = [t2v("best", decoder="Best quality", **shape), t2v("fast", decoder="Fast", **shape),
             t2v("value", decoder="conv", **shape)]
    code, lines, _out, _ = live(tmp_path, fake, calls)
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    assert [c["params"]["decoder"] for c in fake.generate_calls] == ["diffusion", "conv", "conv"]
    with pytest.raises(space_matrix.PlanError) as caught:
        space_matrix.parse_plan([t2v(decoder="Turbo")], base=tmp_path)
    assert "'diffusion' (Best quality)" in str(caught.value) and "'conv' (Fast)" in str(caught.value)


def test_a_variation_mode_shows_the_level_the_app_runs(tmp_path):
    """The level comes from the mode's own kwargs; the published noise_scale default (0.5) is not
    the clip's default level, so a plan states the level it means."""
    from ltx25_studio import audio_conditioning

    write_video(tmp_path / "clip.mp4")
    write_wav(tmp_path / "sound.wav", blocks=60)
    v2v = {"label": "v2v", "mode": "video_to_video", "video": "clip.mp4", "width": SIZE, "height": SIZE}
    a2a = {"label": "a2a", "mode": "audio_to_audio", "audio": "sound.wav", "auto_duration": True}
    plan = [{**v2v, "noise_scale": audio_conditioning.DEFAULT_VIDEO_LEVEL}, {**v2v, "label": "v2v-default"},
            {**a2a, "noise_scale": 0.73}, t2v()]
    calls = space_matrix.parse_plan(plan, base=tmp_path)
    assert [c.app_level for c in calls] == [audio_conditioning.DEFAULT_VIDEO_LEVEL, 0.421875, 0.725, None]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    text = "\n".join(run(["--plan", str(path), "--assets-dir", str(tmp_path)])[1])
    assert "level     0.909375 (noise_scale 0.909375; the mode's levels: 0.421875, 0.725, 0.909375)" in text
    assert "level     0.725 (noise_scale 0.73" in text


def test_skip_leaves_calls_out_and_refuses_a_broken_chain(tmp_path):
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps([t2v(), t2v("b"), {"label": "c", "mode": "image_to_video", "width": SIZE,
                                                   "height": SIZE, "image": {"from_call": "t2v"}}]))
    code, lines = run(["--plan", str(plan), "--skip", "b"])
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_OK and "1. t2v" in text and "2. c" in text and ". b " not in text
    assert run(["--plan", str(plan), "--skip", "t2v"])[0] == space_matrix.EXIT_ILLEGAL
    assert run(["--plan", str(plan), "--skip", "nope"])[0] == space_matrix.EXIT_ILLEGAL


def test_a_lora_weight_outside_the_apps_range_is_refused_offline(tmp_path):
    with pytest.raises(space_matrix.PlanError) as caught:
        space_matrix.parse_plan([t2v(loras=[{"source": "org/style", "weight": 2.5}])], base=tmp_path)
    assert "2" in str(caught.value) and "org/style" in str(caught.value)
    (call,) = space_matrix.parse_plan([t2v(loras=[{"source": "org/style", "weight": 2.0}])], base=tmp_path)
    assert call.loras[0].needs_weight_route


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


def test_the_dry_run_schedules_the_days_and_says_how_to_resume(tmp_path):
    """At worst case 5 calls of 360 fit a 1920 day: 7 calls take two days, and the second day's
    command is the same one with --live --resume."""
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([t2v(f"c{i}") for i in range(1, 8)]))
    ledger, out = tmp_path / "ledger.jsonl", tmp_path / "runs"
    code, lines = run(["--plan", str(plan), "--quota-ledger", str(ledger), "--out", str(out)])
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_OK
    assert "2 UTC day(s) at worst case" in text
    assert "day 1  UTC 2026-11-16 (today)" in text and "<= 1800 GPU-s  calls #1-5    c1 .. c5" in text
    assert "day 2  UTC 2026-11-17" in text and "<=  720 GPU-s  calls #6-7    c6 .. c7" in text
    assert "DOES NOT FIT: call #6 (c6)" in text
    assert (f"python scripts/space_matrix.py --plan {plan} --quota-ledger {ledger} --out {out} --live --resume"
            in text)
    assert not out.exists() and not ledger.exists(), "a dry run writes nothing"

    seed_ledger(ledger, billed=1700.0)
    text = "\n".join(run(["--plan", str(plan), "--quota-ledger", str(ledger)])[1])
    assert "day 1  UTC 2026-11-16 (today)  nothing fits in what is left today" in text
    assert "3 UTC day(s)" in text and "DOES NOT FIT: call #1 (c1)" in text and "--out <DIR>" in text
    assert space_matrix.schedule_days([360.0] * 3, spent=0.0, ceiling=700.0) == [[0], [1], [2]]
    assert space_matrix.schedule_days([2000.0, 10.0], spent=0.0, ceiling=1920.0) == [[0], [1]]


def test_the_dry_run_never_reads_the_token(tmp_path, monkeypatch):
    def refuse():
        raise AssertionError("a dry run read the token")

    monkeypatch.setattr(space_matrix, "read_token", refuse)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([t2v(), t2v("lora", loras=[{"source": "org/x"}])]))
    assert run(["--plan", str(plan), "--quota-ledger", str(tmp_path / "l.jsonl")])[0] == space_matrix.EXIT_OK


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
    records = ledger_lines(ledger)
    assert [r["kind"] for r in records] == ["invocation", "call_start", "call"] * 2, (
        "a call is written down at its full reservation before it is sent, then settled"
    )
    for start, settled in ((records[1], records[2]), (records[4], records[5])):
        assert start["call_id"] == settled["call_id"] and start["billed_conservative"] == 360.0
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


def test_a_call_that_was_sent_and_never_settled_counts_at_its_full_reservation(tmp_path, space):
    """A process killed mid-call leaves only its start record: the day counts its whole reservation."""
    ledger = space_matrix.Ledger(tmp_path / "ledger.jsonl")
    start = {"kind": "call_start", "call_id": "k1", "utc_day": "2026-11-16", "label": "killed",
             "billed_conservative": 360.0}
    ledger.append(start)
    assert ledger.spent("2026-11-16") == 360.0 and [r["label"] for r in ledger.unsettled()] == ["killed"]
    ledger.append({**start, "kind": "call", "billed_conservative": 41.5})
    assert ledger.spent("2026-11-16") == 41.5 and ledger.unsettled() == []
    ledger.append({**start, "call_id": "k2", "utc_day": "2026-11-15"})
    assert ledger.previous_day("2026-11-16") == ("2026-11-15", 360.0)

    fake = space()
    killed = tmp_path / "killed.jsonl"
    seed_ledger(killed, billed=1400.0)
    space_matrix.Ledger(killed).append(start)
    code, lines, _out, _ = live(tmp_path, fake, [t2v()], ledger=killed)
    assert code == space_matrix.EXIT_CEILING and fake.joins == [], "1400 + 360 unsettled + 360 > 1920"
    assert any("1 call(s) sent and never settled (killed)" in line for line in lines)


def test_an_interrupted_call_is_charged_and_recorded(tmp_path, space, monkeypatch):
    real = space_matrix._run_events

    def interrupted(ctx, evidence, endpoint, data, session_hash, started, deadline, timing, key):
        if key == "generate":
            ctx.app.join(endpoint.fn_index, data, session_hash)
            raise KeyboardInterrupt
        return real(ctx, evidence, endpoint, data, session_hash, started, deadline, timing, key)

    monkeypatch.setattr(space_matrix, "_run_events", interrupted)
    fake = space()
    with pytest.raises(KeyboardInterrupt):
        live(tmp_path, fake, [t2v("a"), t2v("b")], out=tmp_path / "out")
    records = ledger_lines(tmp_path / "ledger.jsonl")
    assert [r["kind"] for r in records] == ["invocation", "call_start", "call"]
    assert records[2]["outcome"] == "interrupted" and records[2]["billed_conservative"] >= 360.0
    assert json.loads((tmp_path / "out" / "a" / "result.json").read_text())["outcome"] == "interrupted"
    assert len(fake.joins_for("/on_generate")) == 1


def test_a_failure_after_the_run_completed_still_charges_it(tmp_path, space, monkeypatch):
    def broken(self, server_path, dest):
        raise space_matrix.TransportError("the download broke")

    monkeypatch.setattr(space_matrix.SpaceApp, "download", broken)
    fake = space(step_delay=0.01).reply(Reply(video=write_video(tmp_path / "v.mp4")))
    code, lines, out, ledger = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_CALL_FAILED, "\n".join(lines)
    (record,) = call_records(ledger)
    timing = json.loads((out / "t2v" / "timing.json").read_text())
    assert record["billed_conservative"] > 0 and record["outcome"] == "completed"
    assert record["billed_conservative"] == pytest.approx(timing["window_seconds"] * 2, abs=1e-3)
    assert space_matrix.Ledger(ledger).unsettled() == []
    assert "the download broke" in json.loads((out / "t2v" / "result.json").read_text())["detail"]


def test_a_plan_stopped_at_the_ceiling_resumes_the_next_day_without_rerunning(tmp_path, space, monkeypatch):
    video = write_video(tmp_path / "v.mp4")
    fake = space(step_delay=0.005).reply(*[Reply(video=video)] * 3)
    ledger, out = tmp_path / "ledger.jsonl", tmp_path / "runs"
    seed_ledger(ledger, billed=1559.99)
    calls = [t2v("a"), {"label": "b", "mode": "image_to_video", "width": SIZE, "height": SIZE, "num_frames": 9,
                         "image": {"from_call": "a", "frame": 0}}, t2v("c")]
    code, lines, _out, _ = live(tmp_path, fake, calls, ledger=ledger, out=out)
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_CEILING, text
    assert "--live --resume" in text and len(fake.joins_for("/on_generate")) == 1
    first = (out / "a" / "result.json").read_text()

    monkeypatch.setattr(space_matrix, "_utcnow", lambda: datetime(2026, 11, 17, 0, 10, tzinfo=UTC))
    dry = "\n".join(run(["--plan", str(next(tmp_path.glob("plan-*.json"))), "--out", str(out), "--resume"])[1])
    assert "resume       1 call(s) already have an outcome in --out" in dry and "a (ok)" in dry
    assert "1. b" in dry and "2. c" in dry
    code, lines, _out, _ = live(tmp_path, fake, calls, "--resume", ledger=ledger, out=out)
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    assert [c["mode"] for c in fake.generate_calls] == ["text_to_video", "image_to_video", "text_to_video"]
    assert (out / "a" / "result.json").read_text() == first, "a finished call is never re-run"
    assert json.loads((out / "b" / "request.json").read_text())["media"]["image"]["from_call"] == "a"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["skipped"] == [{"label": "a", "outcome": "ok"}]
    assert [c["label"] for c in summary["calls"]] == ["b", "c"]
    assert len(list(out.glob("summary-*.json"))) == 2


def test_resume_never_retries_a_call_that_reached_the_space(tmp_path, space):
    video = write_video(tmp_path / "v.mp4")
    out = tmp_path / "runs"
    fake = space().reply(Reply(kind="foreign_error", message="boom"), Reply(video=video), Reply(video=video))
    calls = [t2v("failed"), t2v("killed"), t2v("unsent"), t2v("fresh")]
    assert live(tmp_path, fake, calls[:1], out=out)[0] == space_matrix.EXIT_CALL_FAILED
    (out / "killed").mkdir()
    (out / "killed" / "result.json").write_text('{"outcome": "sent", "sent": true}')
    (out / "unsent").mkdir()
    (out / "unsent" / "request.json").write_text("{}")
    code, lines, _out, _ = live(tmp_path, fake, calls, "--resume", out=out)
    text = "\n".join(lines)
    assert code == space_matrix.EXIT_OK, text
    assert f"{'failed':<32} skipped: it has an outcome" in text and f"{'killed':<32} skipped" in text
    assert [c["params"]["prompt"] for c in fake.generate_calls].count("a tram") == 3, "failed once, then two"
    assert json.loads((out / "unsent" / "result.json").read_text())["outcome"] == "ok"
    assert len(list(out.glob("unsent.unsent-*"))) == 1, "the earlier unsent attempt is kept, moved aside"
    assert run(["--plan", "default", "--resume"])[0] == space_matrix.EXIT_ILLEGAL, "--resume needs --out"


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


#: The platform's words as ``spaces`` (zero/client.py) raises them, title then message.
LIMIT_TITLE, LIMIT_MESSAGE = "ZeroGPU illegal duration", "The requested GPU duration (1020s) is larger than the maximum allowed"
QUEUE_TITLE, QUEUE_MESSAGE = "ZeroGPU queue timeout", "No GPU was available after 60s. Retry later"


def platform_card(message):
    """A failure card as ``render.failure`` renders a ``gr.Error``: only its message survives."""
    return ("### The run did not finish\n\nNothing ran. No GPU time was used. Press Generate again.\n\n"
            f'<details class="mr-detail" open><summary>Detail</summary><p>&#x27;{message}&#x27;</p></details>')


@pytest.mark.parametrize(
    "text, category",
    [(LIMIT_TITLE, "limit"), (LIMIT_MESSAGE, "limit"), (QUEUE_TITLE, "queue"), (QUEUE_MESSAGE, "queue"),
     (platform_card(LIMIT_MESSAGE), "limit"), (QUOTA_MESSAGE, "quota"),
     ("You have exceeded your ZeroGPU runs limit. ", "quota")],
)
def test_every_platform_refusal_is_named_by_its_category(text, category):
    assert space_matrix.platform_refusal(text)[0] == category


@pytest.mark.parametrize(
    "reply, category",
    [
        (Reply(kind="app_error", status=platform_card(LIMIT_MESSAGE)), "limit"),
        (Reply(kind="app_error", status=platform_card(QUEUE_MESSAGE)), "queue"),
        (Reply(kind="raised", title=LIMIT_TITLE, message="larger than allowed"), "limit"),
        (Reply(kind="app_error", status="### Busy", record={"category": "queue"}), "queue"),
    ],
)
def test_a_limit_or_queue_refusal_bills_nothing_and_stops_like_a_failed_call(tmp_path, space, reply, category):
    video = write_video(tmp_path / "v.mp4")
    fake = space(step_delay=0.01).reply(reply)
    code, lines, out, ledger = live(tmp_path, fake, [t2v("a"), t2v("b")])
    assert code == space_matrix.EXIT_CALL_FAILED, "\n".join(lines)
    assert len(fake.joins_for("/on_generate")) == 1, "stopped, never retried"
    (record,) = call_records(ledger)
    assert record["category"] == category and record["billed_conservative"] == 0.0
    assert record["outcome"] == "platform_refusal" and space_matrix.Ledger(ledger).spent("2026-11-16") == 0.0
    assert json.loads((out / "a" / "timing.json").read_text())["window_seconds"] > 0, "measured, not billed"
    fake.reply(reply, Reply(video=video))
    code, _lines, out, _ = live(tmp_path, fake, [t2v("a"), t2v("b")], "--keep-going")
    assert code == space_matrix.EXIT_CALL_FAILED
    assert [c["status"] for c in json.loads((out / "summary.json").read_text())["calls"]] == ["failed", "ok"]


def test_an_unknown_category_is_recorded_as_it_came_and_billed(tmp_path, space):
    fake = space(step_delay=0.01).reply(Reply(kind="app_error", status="### Odd", record={"category": "brand_new"}))
    code, lines, out, ledger = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_CALL_FAILED
    (record,) = call_records(ledger)
    assert record["category"] == "brand_new" and record["billed_conservative"] > 0
    result = json.loads((out / "t2v" / "result.json").read_text())
    assert result["classified"] == {"category": "brand_new", "source": "the run's structured record"}


def test_a_failure_card_is_classified_by_the_apps_own_classifier(tmp_path, space, monkeypatch):
    """With no structured record and no platform words, ``errors.categorise`` names it."""
    from ltx25_studio import errors

    seen = []
    real = errors.categorise
    monkeypatch.setattr(errors, "categorise", lambda exc: (seen.append(str(exc)), real(exc))[1])
    card = ('### The GPU ran out of memory\n\nAny GPU time this run used is billed.\n\n'
            '<details class="mr-detail"><summary>Detail</summary><p>CUDA out of memory at 1408x960</p></details>')
    fake = space().reply(Reply(kind="app_error", status=card))
    code, _lines, out, ledger = live(tmp_path, fake, [t2v()])
    assert code == space_matrix.EXIT_CALL_FAILED
    assert call_records(ledger)[-1]["category"] == "oom" and any("CUDA out of memory" in s for s in seen)
    classified = json.loads((out / "t2v" / "result.json").read_text())["classified"]
    assert classified["source"] == "errors.categorise on the text that reached the API"


def test_the_ledger_holds_reservation_conservative_and_worker_figures(tmp_path, space):
    """The worker's own GPU time comes only from a structured record, by RunRecord's field name."""
    fake = space(step_delay=0.01).reply(
        Reply(video=write_video(tmp_path / "v.mp4"), record={"elapsed_seconds": 12.5, "category": None})
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


def test_the_report_is_kept_verbatim_and_its_prose_is_never_read_for_figures(tmp_path, space):
    """The status the API returns (the report, its diagnostics) is saved whole; no figure is parsed
    out of its prose, so with no structured record there is no worker figure."""
    gpu = ('<p class="mr-report-head">Done: Text → video. Seed 7.</p>\n<details class="mr-diagnostics">'
           '<summary>Diagnostics</summary><p>Peak GPU memory reserved: 55.91 GiB</p></details>'
           '<dl class="mr-done-grid"><dt>Took</dt><dd><span class="mr-num">41</span> s, GPU time</dd></dl>')
    video = write_video(tmp_path / "v.mp4")
    fake = space().reply(Reply(video=video, status=gpu))
    code, lines, out, ledger = live(tmp_path, fake, [t2v("gpu")])
    assert code == space_matrix.EXIT_OK, "\n".join(lines)
    assert call_records(ledger)[-1]["billed_worker"] is None
    timing = json.loads((out / "gpu" / "timing.json").read_text())
    assert timing["billed_worker"] is None and timing["worker_elapsed_seconds"] is None
    result = json.loads((out / "gpu" / "result.json").read_text())
    assert result["status_markdown"] == gpu and result["record"] is None


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
            "spectral_centroid_hz", "band_energy_share", "band_level_dbfs", "onsets"} <= set(track)
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
    fake.reply(Reply(video=video, status=f"Done. {SENTINEL}", log=f"queue note {SENTINEL}"), Reply(kind="timeout"),
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


def test_the_token_goes_to_the_space_and_the_hub_only_and_never_across_a_redirect(space):
    trusted, elsewhere = space(), space(token=None)
    http = space_matrix.Http(space_matrix.Secret(SENTINEL), trusted=(trusted.url,))
    http.get_json(f"{trusted.url}/config")
    http.get_json(f"{elsewhere.url}/config")
    http.get_json(f"{trusted.url}/redirect?to={elsewhere.url}/gradio_api/info")
    assert trusted.requests[0]["headers"]["authorization"] == f"Bearer {SENTINEL}"
    assert [r["path"] for r in elsewhere.requests] == ["/config", "/gradio_api/info"]
    assert all("authorization" not in r["headers"] for r in elsewhere.requests), "an untrusted host got the token"


def test_a_server_that_echoes_the_token_never_gets_it_written_down(tmp_path, space):
    """An error body carrying the Authorization header back: the error, the traceback, the
    ledger, every evidence file and every printed line hold <redacted> instead."""
    assets = tmp_path / "assets"
    assets.mkdir()
    write_png(assets / "picture.png")
    fake = space(echo_token_on_upload=True)
    call = {"label": "i2v", "mode": "image_to_video", "image": "picture.png", "width": SIZE, "height": SIZE,
            "num_frames": 9}
    code, lines, out, ledger = live(tmp_path, fake, [call], assets=assets)
    assert code == space_matrix.EXIT_CALL_FAILED, "\n".join(lines)
    result = json.loads((out / "i2v" / "result.json").read_text())
    assert "HTTP 500" in result["detail"] and "upload refused for Bearer <redacted>" in result["detail"]
    assert "<redacted>" in result["traceback"] and result["sent"] is False
    written = [ledger, *(p for p in out.rglob("*") if p.is_file())]
    for path in written:
        assert SENTINEL not in path.read_text(errors="replace"), f"{path} carries the token"
    assert SENTINEL not in "\n".join(lines)
    assert call_records(ledger)[-1]["billed_conservative"] == 0.0 and fake.joins_for("/on_generate") == []
    error = space_matrix.HttpError(500, f"{fake.url}/upload?sig={SENTINEL}", f"echo: Bearer {SENTINEL}")
    assert SENTINEL not in str(error) and SENTINEL not in repr(error) and "<redacted>" in str(error)


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


def test_noise_at_a_normal_level_no_longer_looks_like_sound(tmp_path):
    """The level alone cannot tell noise at -19 dBFS from sound; flatness, the bands and the onset
    count can, and are reported beside it. Measurements: nothing here passes or fails a call."""
    rate, n = 48000, 5 * 48000
    t = np.arange(n) / rate
    noise = np.random.default_rng(7).normal(0.0, 10 ** (-19 / 20), n)
    gate = ((t % 0.25) >= 0.13).astype(float)  # 20 bursts of 120 ms, each after 130 ms of silence
    bursts = 0.4 * np.sin(2 * np.pi * 440.0 * t) * gate
    steady = 0.158 * np.sin(2 * np.pi * 440.0 * t)

    hiss = track_of(write_samples(tmp_path / "hiss.wav", np.stack([noise, noise])))
    chimes = track_of(write_samples(tmp_path / "chimes.wav", np.stack([bursts, bursts])))
    tone = track_of(write_samples(tmp_path / "tone.wav", np.stack([steady, steady])))

    assert -19.5 < hiss["level_dbfs"] < -18.5 and -19.5 < tone["level_dbfs"] < -18.5, "the same level"
    assert hiss["spectral_flatness"] > 0.8 and tone["spectral_flatness"] < 0.05
    assert hiss["onsets"] == 0 and tone["onsets"] == 0 and chimes["onsets"] == 20
    assert chimes["onsets_per_second"] == pytest.approx(4.0, abs=0.01)
    assert hiss["band_energy_share"]["6000-nyquist"] > 0.6 and tone["band_energy_share"]["250-2000"] > 0.95
    assert hiss["band_level_dbfs"]["6000-nyquist"] > tone["band_level_dbfs"]["6000-nyquist"] + 40
    line = space_matrix._audio_line({"tracks": [hiss]})
    assert "-19.0 dBFS" in line and "flatness 0." in line and "onsets 0" in line and "6000-nyquist" in line
    assert space_matrix.count_onsets(np.zeros(rate), rate) == 0
    rise = np.concatenate([np.zeros(rate // 2), 0.3 * np.sin(2 * np.pi * 220.0 * t[: rate // 2])])
    assert space_matrix.count_onsets(rise, rate) == 1, "a sound out of silence is one onset"


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


EXAMPLE_PLAN = REPO_ROOT / "scripts" / "space_plan.example.json"

#: The lead's assets as the app's gate reads them (measured with ``av`` on 2026-11-16): an audio
#: file's length, rate and channels, a clip's frames and rate. A plan that names a new file must
#: say here what it is, because its length decides whether a call is legal.
LEAD_ASSETS = {
    "sfx/ocean_waves_gulls_48k_stereo.wav": {"seconds": 5.0, "rate": 48000, "channels": 2},
    "speech/captain_storm_male_48k_stereo.wav": {"seconds": 4.8298, "rate": 48000, "channels": 2},
    "operator/captain_designed.wav": {"seconds": 8.16, "rate": 24000, "channels": 1},
    "video/cabin_panright_704x480_121f.mp4": {"frames": 121},
    "video/tram_pushin_704x480_121f.mp4": {"frames": 121},
}


def example_media() -> set[str]:
    document = json.loads(EXAMPLE_PLAN.read_text())
    return {call[kind] for call in document["calls"] for kind in ("image", "video", "audio")
            if isinstance(call.get(kind), str)}


def write_lead_assets(root: Path) -> Path:
    """Stand-ins with the lengths, rates and channels of the lead's files (images: any picture)."""
    for relative in example_media():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        facts = LEAD_ASSETS.get(relative)
        if target.suffix == ".png":
            write_png(target)
        elif target.suffix == ".mp4":
            write_frames(target, moving_stack(facts["frames"]))
        else:
            n = int(round(facts["seconds"] * facts["rate"]))
            t = np.arange(n) / facts["rate"]
            tone = 0.3 * np.sin(2 * np.pi * 220.0 * t)
            write_samples(target, np.stack([tone] * facts["channels"]), rate=facts["rate"])
    return root


def test_the_example_plan_is_the_leads_matrix_and_every_call_passes_the_apps_gate(tmp_path):
    """Every call goes through ``generate.prepare_request`` on files with the lead's lengths, and
    the plan is the matrix the lead asked for."""
    assert all(p.endswith(".png") or p in LEAD_ASSETS for p in example_media()), "describe new files in LEAD_ASSETS"
    from ltx25_studio import audio_conditioning

    calls = {c.label: c for c in space_matrix.load_plan(EXAMPLE_PLAN, assets_dir=write_lead_assets(tmp_path))}
    assert {c.mode for c in calls.values()} == set(modes.MODES)
    assert {c.gate for c in calls.values()} == {space_matrix.GATE_ACCEPTED}, "no input is chained"
    assert not any("strength" in c.values for c in calls.values()), "no mode reads strength"

    eleven = [c for label, c in calls.items() if label.startswith("m")]
    assert {c.mode for c in eleven} == set(modes.MODES)
    for call in eleven:
        assert not call.stage_2 and (call.min_resolution or call.render_size in ((704, 480), (480, 704)))
    v2v = calls["m04_video_to_video"]
    assert v2v.app_level == audio_conditioning.DEFAULT_VIDEO_LEVEL and v2v.value("prompt")
    assert v2v.app_frames == 121, "the clip sets the length"
    assert calls["m10_audio_to_audio"].app_level == 0.725
    a2v = [c for c in eleven if c.mode == "audio_to_video"]
    assert {(c.render_size, "image" in c.media) for c in a2v} == {((704, 480), False), ((480, 704), True)}
    assert all(c.value("prompt") for c in a2v)

    upscaled = [c for c in calls.values() if c.stage_2]
    assert {c.mode for c in upscaled} == {"text_to_audio_video", "audio_to_video"}
    for call in upscaled:
        assert call.render_size == (704, 480) and call.expected_geometry() == (1408, 960)
    assert calls["x2_m05_text_to_audio_video"].expected_frames() == 49

    best, fast = calls["dec_m01_best_quality"], calls["dec_m01_fast"]
    assert (best.value("decoder"), fast.value("decoder")) == ("diffusion", "conv"), "sent by value"
    assert {k: v for k, v in best.values.items() if k != "decoder"} == {
        k: v for k, v in fast.values.items() if k != "decoder"}, "the same request and seed"

    picks = calls["len_m05_512x960_model_picks"]
    assert (picks.mode, picks.render_size, picks.value("min_seconds"), picks.value("max_seconds")) == (
        "text_to_audio_video", (512, 960), 5.0, 10.0)
    assert picks.app_frames is None, "the model chooses the length"
    (lora,) = calls["lora_m01_bts_w080"].loras
    assert (calls["lora_m01_bts_w080"].mode, lora.source, lora.weight) == (
        "text_to_video", "SOLRICKS/LTX-2.5-BTS-Movie-Set", 0.8)
    assert calls["enh_m05_text_to_audio_video"].runs("enhance_prompt") is True
    operator = calls["op_m11_operator_audio"]
    assert operator.media["audio"].path.name == "captain_designed.wav" and operator.mode == "audio_to_video"
    assert operator.app_frames == 193, "8.16 s of audio sets 193 frames"
    for call in calls.values():
        if "audio" in call.media and call.value("auto_duration"):
            assert any("length set from your audio" in note for note in call.app_notes), call.label


def test_the_example_plan_against_the_real_assets():
    """Opt-in: LTX25_SPACE_ASSETS_DIR=<the lead's assets folder>."""
    import os

    folder = os.environ.get("LTX25_SPACE_ASSETS_DIR")
    if not folder:
        pytest.skip("set LTX25_SPACE_ASSETS_DIR to the lead's assets folder to gate the example plan on its files")
    calls = space_matrix.load_plan(EXAMPLE_PLAN, assets_dir=Path(folder).expanduser())
    assert {c.gate for c in calls} == {space_matrix.GATE_ACCEPTED}
