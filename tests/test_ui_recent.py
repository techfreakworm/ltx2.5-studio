"""Recent runs can show an earlier, paid output again (review round 2, F16).

After a successful run (S6) and a failed one (S7), "Show" on the earlier row puts that run's
own file back on the monitor, captioned as an earlier run, and nothing runs. Once the run's
folder has been evicted the button goes. ``/on_generate`` is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import inputs_for
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder
from test_ui_form import Form

from ltx25_studio import generate as generate_module
from ltx25_studio import modes
from ltx25_studio.ui import blocks, handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


class Request:
    session_hash = "recent-runs"


@pytest.fixture
def form():
    from ltx25_studio.ui import build_ui

    return Form(build_ui())


def _run_and_finish(form: Form, outcome: handlers.RunOutcome) -> None:
    """What the page does after /on_generate: the real finish step, with this outcome."""
    blocks._remember(Request(), outcome)
    form.run(form.fn("_finish"), request=Request())


_REAL_GENERATE = generate_module.generate


def _real_outcome(tmp_path, monkeypatch) -> handlers.RunOutcome:
    real = _REAL_GENERATE
    monkeypatch.setattr(
        generate_module,
        "generate",
        lambda **params: real(
            pipeline_factory=factory_returning(FakePipe()),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path,
            retention_bytes=10**9,
            **params,
        ),
    )
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_audio_video")
    outcome = handlers.run("text_to_audio_video", *(values[k] for k in FIELD_KEYS))
    assert outcome.error is None and outcome.video, outcome.status
    return outcome


def _show_buttons(form: Form) -> list:
    """The rows' Show buttons, in row order (the finish step lists every row's controls)."""
    finish = form.fn("_finish")
    return [c for c in finish.outputs if isinstance(c, gr.Button) and c.value == "Show"]


def test_show_puts_the_earlier_run_back_on_the_monitor_and_runs_nothing(
    form, tmp_path, monkeypatch
):
    s6 = _real_outcome(tmp_path, monkeypatch)
    _run_and_finish(form, s6)
    failed = handlers.RunOutcome(
        video=None,
        audio=None,
        status=render.failure(RuntimeError("boom")),
        mode_id="text_to_audio_video",
        params=s6.params,
        error=RuntimeError("boom"),
    )
    _run_and_finish(form, failed)
    video, caption = form.by_id("video-out"), form.by_id("monitor-caption")
    assert not form.visible(video), "after S7 the monitor is idle"

    calls = []
    monkeypatch.setattr(generate_module, "generate", lambda **params: calls.append(params))
    show = _show_buttons(form)[0]
    assert form.visible(show), "the S6 row offers Show"
    form.click(show)

    assert calls == [], "Show ran a generation"
    assert form.visible(video) and form.get(video) == s6.video
    seed = s6.result.seed
    mode = modes.get_mode("text_to_audio_video")
    assert form.get(caption) == render.monitor_caption(kind="earlier", mode=mode, seed=seed)
    assert "Earlier run" in form.get(caption)
    form.pick_mode("text_to_audio")
    assert form.get(video) == s6.video, "a mode change does not hide the earlier run"


def test_show_goes_once_the_run_folder_is_evicted(form, tmp_path, monkeypatch):
    s6 = _real_outcome(tmp_path, monkeypatch)
    _run_and_finish(form, s6)
    show = _show_buttons(form)[0]
    assert form.visible(show)
    for path in Path(s6.video).parent.iterdir():
        path.unlink()
    form.click(show)
    assert not form.visible(show), "Show stays on a run whose file is gone"
    _run_and_finish(form, _real_outcome(tmp_path / "next", monkeypatch))
    rows = _show_buttons(form)
    assert form.visible(rows[0]) and not form.visible(rows[1]), "the evicted row keeps no Show"


def test_on_generate_keeps_its_contract():
    from ltx25_studio.ui import build_ui

    demo = build_ui()
    endpoint = demo.get_api_info()["named_endpoints"]["/on_generate"]
    assert [r["component"] for r in endpoint["returns"]] == ["Video", "Audio", "Markdown"]
    assert len(endpoint["parameters"]) == 1 + len([k for k in FIELD_KEYS if k != "loras"])
