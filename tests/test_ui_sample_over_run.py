"""A run stays on the monitor, and every mode's sample can still be seen (review round 4, M4).

After one successful run no mode's sample could be seen again in the session: the monitor kept
the run when the mode changed (AC-06, right), and the jump to the sample showed only while the
monitor held a sample. Now, while the monitor holds a run (yours or an earlier one) from another
mode than the selected one, and that mode has a sample, the control under the purpose line reads
"See an example of <mode>" at every width (class ``mr-over-run``, which the stylesheet shows on a
desktop); a click puts that sample on the monitor with its usual caption and the monitor follows
later switches with their own samples; while a sample covers the run the same control reads
"Back to your run" and puts it back.

Driven through the built page's own steps (:class:`test_ui_form.Form`): a real run through
``handlers.run`` (GPU seams faked), the real finish step, the real switch and the real click.
"""

from __future__ import annotations

import json

import pytest
from conftest import inputs_for
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder
from test_ui_form import Form

from ltx25_studio import generate as generate_module
from ltx25_studio import modes
from ltx25_studio.ui import blocks, handlers, render, showcase
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

#: The modes the stand-in manifest has samples for: the run's own (05), a sound-only one (10)
#: and Audio to video (11). Text to video (01) has none.
SAMPLES = {
    "text_to_audio_video": "samples/t2av.mp4",
    "audio_to_audio": "samples/a2a.m4a",
    "audio_to_video": "samples/a2v.mp4",
}
OVER_RUN = "mr-over-run"


class Request:
    session_hash = "sample-over-run"


@pytest.fixture
def show_dir(tmp_path, monkeypatch):
    root = tmp_path / "examples"
    (root / "samples").mkdir(parents=True)
    entries = {}
    for mode_id, relative in SAMPLES.items():
        (root / relative).write_bytes(b"sample")
        entries[mode_id] = {"sample": {"file": relative, "caption": "Made earlier.", "seed": 3}}
    (root / showcase.MANIFEST).write_text(json.dumps({"modes": entries}), encoding="utf-8")
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(root))
    return root


@pytest.fixture
def form(show_dir):
    return Form(blocks.build_ui())


@pytest.fixture
def wired(tmp_path, monkeypatch):
    real = generate_module.generate
    monkeypatch.setattr(
        generate_module,
        "generate",
        lambda **params: real(
            pipeline_factory=factory_returning(FakePipe()),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path / "runs",
            retention_bytes=10**9,
            **params,
        ),
    )


def a_run(form: Form, mode_id: str = "text_to_audio_video") -> handlers.RunOutcome:
    """A run of ``mode_id`` through the real handler, then the page's real finish step."""
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id) | {"seed": 5}
    outcome = handlers.run(mode_id, *(values[k] for k in FIELD_KEYS))
    assert outcome.error is None and outcome.video, outcome.status
    form.values[form.by_id("run-status")._id] = outcome.status
    blocks._remember(Request(), outcome)
    form.run(form.fn("_finish"), request=Request())
    return outcome


def sample_path(show_dir, mode_id: str) -> str:
    return str((show_dir / SAMPLES[mode_id]).resolve())


def jump_reads(form: Form) -> tuple[str | None, bool, bool]:
    """The control under the purpose line: (its words, shown, carries mr-over-run)."""
    jump = form.by_id("sample-jump")
    classes = form.props[jump._id].get("elem_classes") or ()
    return (form.get(jump) if form.visible(jump) else None, form.visible(jump), OVER_RUN in classes)


def on_monitor(form: Form) -> str | None:
    video, audio = form.by_id("video-out"), form.by_id("audio-out")
    if form.visible(video) and form.get(video):
        return str(form.get(video))
    if form.visible(audio) and form.get(audio):
        return str(form.get(audio))
    return None


def caption(form: Form) -> str:
    return form.get(form.by_id("monitor-caption"))


def test_run_then_switch_then_the_example_then_back_to_the_run(form, wired, show_dir):
    run = a_run(form)
    label = modes.get_mode("text_to_audio_video").label
    assert on_monitor(form) == str(run.video)
    assert caption(form) == f"**Your run.** {label}, seed {run.result.seed}."
    assert jump_reads(form) == (None, False, False), "the run is the selected mode's own"

    form.pick_mode("audio_to_audio")
    assert on_monitor(form) == str(run.video), "the run stays when the mode changes (AC-06)"
    assert caption(form).startswith("**Your run.**")
    assert jump_reads(form) == ("See an example of Audio to audio", True, True)

    form.click(form.by_id("sample-jump"))
    assert on_monitor(form) == sample_path(show_dir, "audio_to_audio")
    assert not form.visible(form.by_id("video-out")), "a sound sample plays in the sound player"
    assert caption(form).startswith("**Example output, not your run.** Audio to audio")
    assert jump_reads(form) == (render.BACK_TO_RUN, True, True)
    assert render.BACK_TO_RUN == "Back to your run"

    form.pick_mode("audio_to_video")
    assert on_monitor(form) == sample_path(show_dir, "audio_to_video"), "later switches follow"
    assert caption(form).startswith("**Example output, not your run.** Audio to video")
    assert jump_reads(form) == (render.BACK_TO_RUN, True, True), "the run is still one click away"

    form.click(form.by_id("sample-jump"))
    assert on_monitor(form) == str(run.video)
    assert caption(form) == f"**Your run.** {label}, seed {run.result.seed}."
    assert jump_reads(form) == ("See an example of Audio to video", True, True)

    form.pick_mode("text_to_audio_video")
    assert on_monitor(form) == str(run.video)
    assert jump_reads(form) == (None, False, False)


def test_a_mode_without_a_sample_offers_nothing_over_the_run(form, wired):
    run = a_run(form)
    form.pick_mode("text_to_video")
    assert on_monitor(form) == str(run.video)
    assert jump_reads(form) == (None, False, False)


def test_an_earlier_run_shown_from_recent_runs_works_the_same_way(form, wired, show_dir):
    first = a_run(form)
    second = a_run(form)
    show = [b for b in form.fn("_finish").outputs if isinstance(b, gr.Button) and b.value == "Show"]
    form.click(show[1])
    label = modes.get_mode("text_to_audio_video").label
    assert on_monitor(form) == str(first.video) != str(second.video)
    assert caption(form) == f"**Earlier run.** {label}, seed {first.result.seed}."

    form.pick_mode("audio_to_video")
    assert jump_reads(form) == ("See an example of Audio to video", True, True)
    form.click(form.by_id("sample-jump"))
    assert on_monitor(form) == sample_path(show_dir, "audio_to_video")
    form.click(form.by_id("sample-jump"))
    assert on_monitor(form) == str(first.video)
    assert caption(form) == f"**Earlier run.** {label}, seed {first.result.seed}."


def test_a_run_finished_while_another_mode_is_selected_keeps_its_own_mode(form, wired):
    """The run's caption, its Recent row and the monitor state name the mode that ran, not the
    one selected when it ended."""
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_audio_video")
    outcome = handlers.run("text_to_audio_video", *(values[k] for k in FIELD_KEYS))
    form.pick_mode("audio_to_audio")
    blocks._remember(Request(), outcome)
    form.run(form.fn("_finish"), request=Request())
    label = modes.get_mode("text_to_audio_video").label
    assert caption(form).startswith(f"**Your run.** {label}, ")
    recent = next(c for c in form.fn("_finish").outputs if isinstance(c, gr.State)
                  and isinstance(form.get(c), list) and form.get(c))
    assert form.get(recent)[0]["mode"] == "text_to_audio_video"
    assert jump_reads(form) == ("See an example of Audio to audio", True, True)


def test_back_to_a_run_whose_file_is_gone_is_not_offered(form, wired, show_dir):
    from pathlib import Path

    run = a_run(form)
    form.pick_mode("audio_to_audio")
    form.click(form.by_id("sample-jump"))
    assert jump_reads(form)[0] == render.BACK_TO_RUN
    Path(run.video).unlink()
    form.pick_mode("audio_to_video")
    assert jump_reads(form) == (render.SAMPLE_JUMP, True, False), (
        "the run's file was evicted: only the plain jump to the sample is left"
    )
    assert on_monitor(form) == sample_path(show_dir, "audio_to_video")
    form.click(form.by_id("sample-jump"))
    assert on_monitor(form) == sample_path(show_dir, "audio_to_video"), "nothing to go back to"


def test_with_no_run_the_jump_is_the_phones_plain_jump_to_the_sample(form):
    assert jump_reads(form) == (render.SAMPLE_JUMP, True, False)
    form.pick_mode("text_to_video")
    assert jump_reads(form) == (None, False, False)
    form.pick_mode("audio_to_audio")
    assert jump_reads(form) == (render.SAMPLE_JUMP, True, False)


def test_a_new_run_drops_the_covered_run_and_hides_the_control(form, wired):
    a_run(form)
    form.pick_mode("audio_to_audio")
    form.click(form.by_id("sample-jump"))
    start = form.fn("_start")
    out = start.fn("audio_to_audio", False, "", Request())
    got = dict(zip(start.outputs, out, strict=True))
    assert got[form.by_id("sample-jump")]["visible"] is False
    state = next(v for c, v in got.items() if isinstance(c, gr.State) and isinstance(v, dict))
    assert state == {"kind": "running", "mode": "audio_to_audio"}


def test_the_click_passes_gradios_own_processing(show_dir, tmp_path):
    """The step as Gradio's queue runs it: the sample reaches the player as a file it serves,
    and the control's words and class reach the page."""
    import asyncio

    from gradio.state_holder import SessionState

    demo = blocks.build_ui()
    step = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "_on_jump")
    monitor = next(c for c in step.inputs if isinstance(c, gr.State))
    run_file = tmp_path / "output.mp4"
    run_file.write_bytes(b"run")
    state = SessionState(demo)
    state[monitor._id] = blocks._run_state("text_to_audio_video", 5, run_file, run_file)
    out = asyncio.run(
        demo.process_api(
            block_fn=step,
            inputs=["audio_to_video", None],
            state=state,
            request=gr.Request(session_hash="jump"),
            session_hash="jump",
            in_event_listener=True,
        )
    )["data"]
    got = {getattr(c, "elem_id", None): v for c, v in zip(step.outputs, out, strict=True)}
    assert got["video-out"]["value"]["path"] == sample_path(show_dir, "audio_to_video")
    assert got["monitor-caption"].startswith("**Example output, not your run.**")
    assert got["sample-jump"]["value"] == render.BACK_TO_RUN
    assert OVER_RUN in got["sample-jump"]["elem_classes"]
    assert state[monitor._id]["covered"]["video"] == run_file


def test_the_click_brings_the_monitor_into_view_below_1024px(form):
    jump = form.by_id("sample-jump")
    (listener,) = form.listeners(jump, "click")
    js = listener.js
    assert "scrollIntoView" in js and "getElementById('monitor')" in js
    assert "innerWidth >= 1024" in js and "block: 'center'" in js
    assert "return args" in js, "the step's inputs pass through to the handler"
    assert listener.api_visibility == "private"


def test_the_words_live_beside_the_sample_jump():
    assert render.SAMPLE_JUMP == "See an example output"
    assert render.sample_of(modes.get_mode("audio_to_audio")) == "See an example of Audio to audio"
