"""A failure card belongs to its run (review round 3, R4).

After a mode switch the mode-01 "width=700" refusal stayed under Generate in Audio to audio, red
and open, beside the panel's own (different) refusal; the enhancer card outlived the mode too. Now
a switch, or a Restore, to another mode collapses the card to one line naming its run, and a
refusal is captioned "Refused before running." while "This run failed." is kept for runs that used
the GPU.

Driven through the built page's own steps (:class:`test_ui_form.Form`): the status the page holds
is what ``/on_generate`` returned, the finish step runs as the page runs it, then the real switch.
"""

from __future__ import annotations

import re

import pytest
from conftest import inputs_for
from test_ui_form import Form

from ltx25_studio import errors, modes
from ltx25_studio.ui import blocks, handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


class Request:
    session_hash = "failure-card"


@pytest.fixture
def form():
    from ltx25_studio.ui import build_ui

    return Form(build_ui())


def visible(markup: str) -> str:
    markup = re.sub(r"<details(?![^>]*\bopen\b)[^>]*>.*?</details>", "", str(markup), flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", markup).split())


def finished(form: Form, outcome: handlers.RunOutcome) -> None:
    """What the page does once ``/on_generate`` returned ``outcome``: the status holds its
    report, then the real finish step runs."""
    form.values[form.by_id("run-status")._id] = outcome.status
    blocks._remember(Request(), outcome)
    form.run(form.fn("_finish"), request=Request())


def refused_in_01(form: Form) -> handlers.RunOutcome:
    """Mode 01 at width 700, run through the real handler: refused before dispatch."""
    form.pick_mode("text_to_video")
    form.user(form.fields["prompt"], "A tram climbs a cobbled street")
    form.user(form.fields["width"], 700)
    outcome = handlers.run("text_to_video", *form.form_values())
    assert outcome.refused and "width=700" in outcome.status, outcome.status
    return outcome


def enhancer_failed_in_05(form: Form) -> handlers.RunOutcome:
    form.pick_mode("text_to_audio_video")
    form.user(form.fields["prompt"], "a harbour at dawn")
    exc = errors.PromptEnhancementError(repo_id="org/enhancer", reason="it could not be fetched")
    return handlers.RunOutcome(
        video=None,
        audio=None,
        status=render.failure(exc),
        mode_id="text_to_audio_video",
        params=dict(zip(FIELD_KEYS, form.form_values(), strict=True)),
        error=exc,
    )


def test_an_input_refusal_in_01_then_a_switch_to_10_leaves_one_line_naming_01(form):
    finished(form, refused_in_01(form))
    status, caption = form.by_id("run-status"), form.by_id("monitor-caption")
    assert form.get(status).startswith("### Refused before running")
    assert form.get(caption) == render.monitor_caption(kind="refused")
    assert form.get(caption) == "**Refused before running.** The report below says why."

    form.pick_mode("audio_to_audio")

    held = form.get(status)
    assert "###" not in held and "<details" not in held, f"the card outlived its mode: {held!r}"
    assert "width" not in held.lower(), held
    assert held.startswith("**Last run:** Text to video, ") and held.count("\n") == 0, held
    assert held.endswith(". Refused before running: fix the input."), held
    assert re.match(r"\*\*Last run:\*\* Text to video, \d\d:\d\d\. ", held), held
    assert form.get(caption) != render.monitor_caption(kind="refused"), "the caption kept it"


def test_an_enhancer_failure_in_05_then_a_switch_to_02_leaves_one_line_naming_05(form):
    finished(form, enhancer_failed_in_05(form))
    status, caption = form.by_id("run-status"), form.by_id("monitor-caption")
    assert form.get(status).startswith("### The prompt enhancer is unavailable")
    assert form.get(caption) == render.monitor_caption(kind="failed"), (
        "a run that used the GPU keeps 'This run failed.'"
    )

    form.pick_mode("image_to_video")

    held = form.get(status)
    label = modes.get_mode("text_to_audio_video").label
    assert "Turn Enhance off" not in held and "###" not in held, held
    assert held.startswith(f"**Last run:** {label}, ") and held.endswith(
        ". The prompt enhancer is unavailable."
    ), held


def test_the_line_is_written_once_and_the_card_stays_in_its_own_mode(form):
    finished(form, refused_in_01(form))
    status = form.by_id("run-status")
    card = form.get(status)
    form.pick_mode("text_to_video")  # no switch at all: the card stays as it is
    assert form.get(status) == card
    form.pick_mode("image_to_video")
    line = form.get(status)
    form.pick_mode("audio_to_audio")
    form.pick_mode("text_to_video")
    assert form.get(status) == line, "a later switch rewrote the line or brought the card back"


def test_a_restore_into_another_mode_collapses_the_card_too(form, tmp_path, monkeypatch):
    from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

    from ltx25_studio import generate as generate_module

    real = generate_module.generate
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
    values = {k: v for k, v in zip(FIELD_KEYS, form.form_values(), strict=True)}
    values |= inputs_for("text_to_audio_video")
    good = handlers.run("text_to_audio_video", *(values[k] for k in FIELD_KEYS))
    assert good.error is None, good.status
    finished(form, good)
    finished(form, refused_in_01(form))
    restore = next(
        b
        for b in form.demo.blocks.values()
        if isinstance(b, gr.Button) and b.value == "Restore settings"
    )
    form.click(restore)
    assert form.get(form.radio) == "text_to_audio_video"
    held = form.get(form.by_id("run-status"))
    assert held.startswith("**Last run:** Text to video, "), held


def test_a_success_report_is_left_alone_by_a_switch(form, tmp_path, monkeypatch):
    status = form.by_id("run-status")
    report = "<p class='mr-report-head'>Done: Text to video with sound. Seed 3.</p>"

    class Result:
        seed = 3
        warnings = ()
        run_dir = tmp_path

    video = tmp_path / "output.mp4"
    video.write_bytes(b"x")
    outcome = handlers.RunOutcome(
        video=str(video),
        audio=str(video),
        status=report,
        mode_id="text_to_audio_video",
        params=dict(zip(FIELD_KEYS, form.form_values(), strict=True)),
        result=Result(),
    )
    finished(form, outcome)
    form.pick_mode("audio_to_audio")
    assert form.get(status) == report


def test_a_new_run_forgets_the_last_failure(form):
    finished(form, refused_in_01(form))
    start = form.fn("_start")
    out = start.fn("text_to_video", False, "a tram", Request())
    failure_state = start.outputs[-1]
    assert isinstance(failure_state, gr.State) and out[-1] is None


@pytest.mark.parametrize(
    ("category", "gpu_used", "refused", "kind"),
    [
        ("input", False, True, "refused"),
        ("input", False, False, "refused"),
        ("quota", False, False, "refused"),
        ("input", True, False, "failed"),
        ("enhancer", True, False, "failed"),
        ("bug", True, False, "failed"),
    ],
)
def test_the_caption_says_refused_exactly_when_nothing_ran(category, gpu_used, refused, kind):
    exc = errors.DecodeError(mode="text_to_video", reason="x")
    exc.category, exc.gpu_used = category, gpu_used
    outcome = handlers.RunOutcome(
        video=None, audio=None, status="### x", mode_id="text_to_video", error=exc, refused=refused
    )
    assert handlers.ran_nothing(outcome) is (kind == "refused")
    assert not handlers.ran_nothing(None)


def test_the_headline_is_read_from_a_card_and_nothing_else():
    assert render.card_headline("### Not enough GPU quota for this run\n\nNothing ran.") == (
        "Not enough GPU quota for this run"
    )
    assert render.card_headline("**Running: Text to video.** Loading…") is None
    assert render.card_headline("") is None and render.card_headline(None) is None
    assert visible(render.lost_run()).startswith("### The run ended without a result")
    assert render.card_headline(render.lost_run()) == "The run ended without a result"
