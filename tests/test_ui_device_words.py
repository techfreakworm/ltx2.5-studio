"""One device rule for every quota sentence (F75).

The lead's real-weights session on a Mac read "Reserves 180 s of GPU time", "Needs 360 s of quota
free to start", "Billed for GPU time used × 2", "you pay for that GPU time" and "Any GPU time this
run used is billed": none of it is true off ZeroGPU. Where the app is not on ZeroGPU nothing the
page shows speaks of reserving, quota, billing or paying; on ZeroGPU every sentence stays as it
was (the other UI tests pin them under the stand-in ``spaces`` module, as here).

The platform is resolved as ``app.py`` resolves it (``device.current()``): the stand-in platform
package in ``sys.modules`` for ZeroGPU; ``LTX25_DEVICE`` and the CUDA probe for everything else.
Every string is taken from where the page gets it: the built Blocks walked through all eleven
modes, ``handlers.review`` with each lever on, the start step, ``handlers.run`` through the real
``generate()`` (GPU seams faked), and the finish step's reports.
"""

from __future__ import annotations

import re
import sys
import types
from dataclasses import replace

import pytest
from conftest import inputs_for

from ltx25_studio import config, device, errors, modes, zerogpu
from ltx25_studio import generate as generate_module
from ltx25_studio.ui import handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

MODE_IDS = tuple(modes.MODES)
#: The words of a meter (the brief's list): none may appear off ZeroGPU.
METER_WORDS = ("quota", "reserv", "billed", "bill", "pay")


def visible(markup) -> str:
    """Text a user sees: closed <details> bodies and tags removed, entities read."""
    import html

    markup = re.sub(r"<details(?![^>]*\bopen\b)[^>]*>.*?</details>", "", str(markup), flags=re.S)
    return html.unescape(" ".join(re.sub(r"<[^>]+>", " ", markup).split()))


def meter_words(texts) -> list[tuple[str, str]]:
    found = []
    for text in texts:
        words = visible(text).lower()
        found += [(word, visible(text)[:160]) for word in METER_WORDS if word in words]
    return found


# --------------------------------------------------------------------------
# the platforms
# --------------------------------------------------------------------------

LOCAL = ("cpu", "mps", "cuda")


@pytest.fixture
def no_examples(tmp_path, monkeypatch):
    """No shipped manifest: these tests read the page's own words only."""
    from ltx25_studio.ui import showcase

    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))


@pytest.fixture
def platform(monkeypatch, no_examples):
    """``platform("cpu" | "mps" | "cuda" | "zerogpu")``: this process as app.py resolves it."""

    def on(name: str) -> None:
        monkeypatch.delitem(sys.modules, "spaces", raising=False)
        monkeypatch.delenv(config.ENV_DEVICE, raising=False)
        monkeypatch.setattr(device, "host_has_mps", lambda: name == "mps")
        monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: name == "cuda")
        if name == "zerogpu":
            module = types.ModuleType("spaces")
            module.GPU = lambda *a, **k: lambda f: f
            monkeypatch.setitem(sys.modules, "spaces", module)
        elif name in ("cpu", "mps"):
            monkeypatch.setenv(config.ENV_DEVICE, name)
        found = handlers.run_platform()
        expected = ("cuda", True) if name == "zerogpu" else (name, False)
        assert found == expected, f"the stand-in platform resolved as {found}"

    return on


@pytest.fixture
def staged_lora(tmp_path):
    from lora_fixtures import good_adapter

    path = good_adapter(tmp_path)
    return [{"source": "org/adapter", "path": str(path), "weight": 0.8, "content_key": "k"}]


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """``generate()`` is the real one; only the GPU seams are fakes."""
    from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

    real = generate_module.generate

    def generate(**params):
        return real(
            pipeline_factory=factory_returning(FakePipe()),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path / "runs",
            retention_bytes=10**9,
            **params,
        )

    monkeypatch.setattr(generate_module, "generate", generate)
    return generate


def form_values(mode_id: str, **overrides) -> dict:
    return {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id) | overrides


LEVERS = {
    "plain": {},
    "upscale": {"enable_stage_2": True},
    "enhance": {"enhance_prompt": True, "prompt": "a harbour at dawn"},
    "lora": {"loras": "staged"},
    "all": {"enable_stage_2": True, "enhance_prompt": True, "prompt": "a harbour", "loras": "staged"},
}


def lever_values(mode_id: str, lever: str, staged: list) -> dict:
    values = form_values(mode_id)
    for key, value in LEVERS[lever].items():
        if key == "prompt" and not modes.accepts(modes.get_mode(mode_id), "text"):
            continue
        values[key] = staged if value == "staged" else value
    return values


def page_strings(demo) -> list[str]:
    """Everything the built page holds for a user to read, in every mode: values, labels,
    info lines, placeholders and choices of every component, the switch run between modes."""
    from test_ui_form import Form

    form = Form(demo)
    held: list[str] = []

    def collect() -> None:
        for block in demo.blocks.values():
            props = form.props[block._id]
            if props.get("visible") is False:
                continue
            words = (form.get(block), props.get("label"), props.get("info"), props.get("placeholder"))
            held.extend(text for text in words if isinstance(text, str))
            held.extend(str(choice) for choice in props.get("choices") or ())

    collect()
    for mode_id in MODE_IDS:
        form.pick_mode(mode_id)
        collect()
    return held


# --------------------------------------------------------------------------
# off ZeroGPU: no meter anywhere
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", LOCAL)
def test_off_zerogpu_the_built_page_speaks_of_no_meter_in_any_mode(platform, name):
    platform(name)
    from ltx25_studio.ui import build_ui

    demo = build_ui()
    held = page_strings(demo)
    assert meter_words(held) == [], meter_words(held)[:6]
    identity = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "identity")
    assert "How quota works" not in identity.value
    assert "mr-quota" not in identity.value, "the disclosure is not on the page at all"


@pytest.mark.parametrize("name", LOCAL)
@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_off_zerogpu_the_cost_panel_says_where_it_runs_with_each_lever(
    platform, staged_lora, name, mode_id
):
    platform(name)
    where = {"cpu": "on the CPU", "mps": "on this Mac's GPU", "cuda": "on the GPU"}[name]
    for lever in LEVERS:
        facts = handlers.review(mode_id, lever_values(mode_id, lever, staged_lora))
        texts = [
            render.cost_panel(facts),
            render.commit_line(facts),
            render.readiness_reason(facts),
            render.readiness_reason(facts, running=True),
        ]
        assert meter_words(texts) == [], (lever, meter_words(texts))
        panel = visible(render.cost_panel(facts))
        for gone in ("Reserves", "Needs", "Billed"):
            assert gone not in panel, (lever, gone, panel)
        assert not re.search(r"\+\d+ s\b", panel), (lever, "a lamp names reserved seconds", panel)
        if facts.state != "refused":
            assert f"Runs {where}" in panel and "nothing is metered" in panel, (lever, panel)
            assert render.commit_line(facts).endswith(where) or facts.state == "incomplete"


@pytest.mark.parametrize("name", LOCAL)
def test_off_zerogpu_the_running_and_done_states_speak_of_no_meter(platform, wired, name):
    platform(name)
    from ltx25_studio.ui import blocks

    demo = blocks.build_ui()
    run = next(f for f in demo.fns.values() if f.api_name == "on_generate")
    start = demo.fns[run.trigger_after]
    texts = []
    for mode_id in MODE_IDS:
        mode = modes.get_mode(mode_id)
        started = start.fn(mode_id, True, "a harbour at dawn")
        texts += [v.get("value") if isinstance(v, dict) else v for v in started]
        texts.append(render.monitor_caption(kind="running", mode=mode))
    values = form_values("text_to_audio_video")
    outcome = handlers.run("text_to_audio_video", *(values[k] for k in FIELD_KEYS))
    assert outcome.error is None, outcome.status
    facts = handlers.run_facts(outcome)
    texts += [render.run_summary(facts), outcome.status, render.took_words(facts)]
    texts += [render.monitor_caption(kind=k, mode=modes.get_mode("text_to_video"), seed=1) for k in ("run", "earlier", "failed", "refused", "unshown")]
    found = meter_words([t for t in texts if isinstance(t, str)])
    assert found == [], found[:6]


def _raising(monkeypatch, exc):
    def generate(**_params):
        raise exc

    monkeypatch.setattr(generate_module, "generate", generate)


@pytest.mark.parametrize("name", LOCAL)
@pytest.mark.parametrize("category", [*errors.CATEGORIES, "a-category-the-ui-has-never-seen"])
@pytest.mark.parametrize("gpu_used", [True, False])
def test_off_zerogpu_every_failure_card_has_no_billing_sentence(
    platform, monkeypatch, name, category, gpu_used
):
    platform(name)
    exc = errors.DecodeError(mode="text_to_video", reason="the words of the failure")
    exc.category, exc.gpu_used = category, gpu_used
    _raising(monkeypatch, exc)
    values = form_values("text_to_video")
    outcome = handlers.run("text_to_video", *(values[k] for k in FIELD_KEYS))
    assert outcome.error is exc and outcome.status.startswith("### ")
    assert meter_words([outcome.status]) == [], outcome.status
    assert "the words of the failure" in outcome.status, "the detail is kept"


@pytest.mark.parametrize("name", LOCAL)
def test_off_zerogpu_a_refusal_lost_run_empty_file_and_unshown_result_bill_nothing(
    platform, monkeypatch, wired, tmp_path, name
):
    platform(name)
    values = form_values("text_to_video", width=700)
    refused = handlers.run("text_to_video", *(values[k] for k in FIELD_KEYS))
    assert refused.refused and "Nothing ran." in refused.status
    lost = handlers.unshown_report(None, "the server's message")
    good_values = form_values("text_to_audio_video")
    finished = handlers.run("text_to_audio_video", *(good_values[k] for k in FIELD_KEYS))
    assert finished.error is None, finished.status
    unshown = handlers.unshown_report(finished, "the server's message")
    monkeypatch.setattr(
        generate_module,
        "generate",
        lambda **_p: replace(finished.result, video_path=None, audio_path=None),
    )
    empty = handlers.run("text_to_audio_video", *(good_values[k] for k in FIELD_KEYS))
    assert empty.status.startswith("### The run finished but produced no file")
    texts = [refused.status, lost, unshown, empty.status]
    assert meter_words(texts) == [], meter_words(texts)


# --------------------------------------------------------------------------
# on ZeroGPU: the Space's words, unchanged
# --------------------------------------------------------------------------


def test_on_zerogpu_the_panel_still_shows_reserves_needs_and_billed(platform, staged_lora):
    platform("zerogpu")
    from ltx25_studio.ui import build_ui

    facts = handlers.review("text_to_audio_video", lever_values("text_to_audio_video", "all", staged_lora))
    panel = visible(render.cost_panel(facts))
    for words in ("Reserves", "of GPU time", "Needs", "of quota free to start", "Billed", "billed too"):
        assert words in panel, words
    lit = re.findall(r"(Upscale|Enhance|LoRA) \+(\d+) s", panel)
    assert [name for name, _s in lit] == ["Upscale", "Enhance", "LoRA"], panel
    line = render.commit_line(facts).replace("\u00a0", " ")
    assert line.endswith(f"needs {facts.reserve * facts.multiplier} s quota")
    demo = build_ui()
    identity = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "identity")
    assert "How quota works" in identity.value
    frame = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "matrix-frame")
    for note in render.COLUMN_NOTES.values():
        assert note in frame.value
    enhance = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "enhance")
    upscale = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "upscale")
    assert f"Reserves {zerogpu.ENHANCE_DURATION_SECONDS} s more" in enhance.info
    assert "Reserves" in upscale.info
    exc = errors.DecodeError(mode="text_to_video", reason="x")
    assert "Any GPU time this run used is billed." in render.failure(exc)


def test_on_zerogpu_upscale_is_never_parked(platform):
    platform("zerogpu")
    facts = handlers.review("text_to_audio_video", form_values("text_to_audio_video"))
    assert facts.upscale_refused is None


# --------------------------------------------------------------------------
# Upscale where this machine cannot run it: parked with the backend's own reason
# --------------------------------------------------------------------------


def _backend_refuses_upscale(mode_id: str, values: dict) -> str | None:
    """What the backend says about this request with Upscale on (its own gate)."""
    request = generate_module.GenerationRequest(mode=mode_id, **(values | {"enable_stage_2": True}))
    try:
        generate_module.prepare_request(request)
    except device.StageTwoNeedsCudaError as exc:
        return str(exc)
    except Exception:  # noqa: BLE001 -- another refusal is not the device's
        return None
    return None


@pytest.mark.parametrize("name", ["cpu", "mps"])
@pytest.mark.parametrize(
    ("width", "height", "frames"),
    [(704, 480, 121), (960, 544, 49), (64, 64, 9), (32, 32, 9), (64, 64, 17), (96, 64, 9)],
)
def test_upscale_is_parked_exactly_where_the_backend_refuses_it(platform, name, width, height, frames):
    platform(name)
    values = form_values("text_to_video", width=width, height=height, num_frames=frames)
    facts = handlers.review("text_to_video", values)
    refused = _backend_refuses_upscale("text_to_video", values)
    assert (facts.upscale_refused is not None) == (refused is not None), (width, height, frames)
    if refused:
        where = {"cpu": "the CPU", "mps": "this Mac's GPU"}[name]
        assert facts.upscale_refused == f"Upscale ×2 needs a CUDA GPU; this machine runs on {where}."


def test_on_the_cpu_the_built_page_parks_upscale_unticked_with_the_reason(platform):
    platform("cpu")
    from test_ui_form import Form

    from ltx25_studio.ui import build_ui

    form = Form(build_ui())
    upscale = form.fields["enable_stage_2"]
    form.pick_mode("text_to_video")
    form.user(form.fields["prompt"], "a tram")
    assert not form.interactive(upscale) and form.field("enable_stage_2") is False
    assert "needs a CUDA GPU; this machine runs on the CPU" in form.props[upscale._id]["info"]
    form.user(form.fields["width"], 64)
    form.user(form.fields["height"], 64)
    form.user(form.fields["num_frames"], 9)
    assert form.interactive(upscale), "a size this machine can upscale frees the control"
    form.user(upscale, True)
    assert form.field("enable_stage_2") is True
    form.user(form.fields["width"], 704)
    assert not form.interactive(upscale) and form.field("enable_stage_2") is False, (
        "a size it cannot upscale parks it again, unticked: Generate is never pressed into a refusal"
    )
