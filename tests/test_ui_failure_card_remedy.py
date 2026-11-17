"""A failure card does not outlive Show, Restore or its own remedy (review round 4, minor 2).

R4 collapsed a failure card to one line naming its run when the mode changed. Within one mode
the card stayed red and open: in the round-4 captures Enhance was already unticked and "The prompt
enhancer is unavailable. Turn Enhance off ..." was still under Generate, through Show, Restore,
LoRA staging and Upscale. Now the card collapses to the same one line when Show puts a run on the
monitor, when Restore writes a run's settings, and when a control its step names changes (any
field, for a refusal whose detail names the field); a card whose step names no control (press
Generate again) stays until the next run.

Driven through the built page's own steps (:class:`test_ui_form.Form`).
"""

from __future__ import annotations

import re

import pytest
from conftest import inputs_for
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder
from test_ui_failure_card import enhancer_failed_in_05, finished, refused_in_01
from test_ui_form import Form

from ltx25_studio import errors
from ltx25_studio import generate as generate_module
from ltx25_studio.ui import blocks, handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

LINE = re.compile(r"\*\*Last run:\*\* .+, \d\d:\d\d\. .+\.$")


@pytest.fixture
def form(tmp_path, monkeypatch):
    from ltx25_studio.ui import showcase

    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    return Form(blocks.build_ui())


def status(form: Form) -> str:
    return form.get(form.by_id("run-status"))


def card_of(form: Form, exc: BaseException, mode_id: str) -> handlers.RunOutcome:
    return handlers.RunOutcome(
        video=None,
        audio=None,
        status=render.failure(exc),
        mode_id=mode_id,
        params=dict(zip(FIELD_KEYS, form.form_values(), strict=True)),
        error=exc,
    )


def test_unticking_enhance_collapses_the_enhancer_card(form):
    outcome = enhancer_failed_in_05(form)
    form.user(form.fields["enhance_prompt"], True)  # on, as it was when the run failed
    finished(form, outcome)
    assert status(form).startswith("### The prompt enhancer is unavailable")
    form.user(form.fields["prompt"], "a harbour at dawn, wide")
    assert status(form).startswith("###"), "a field its step does not name leaves it"
    form.user(form.fields["enhance_prompt"], False)
    assert LINE.match(status(form)), status(form)
    assert status(form).endswith(". The prompt enhancer is unavailable.")


def test_fixing_a_refused_input_collapses_its_card(form):
    finished(form, refused_in_01(form))
    assert status(form).startswith("### Refused before running")
    form.user(form.fields["width"], 704)
    assert LINE.match(status(form)), status(form)


def test_a_card_whose_step_is_press_generate_again_stays_until_the_next_run(form):
    form.pick_mode("text_to_video")
    exc = errors.DecodeError(mode="text_to_video", reason="something broke")
    exc.category, exc.gpu_used = errors.BUG, True
    finished(form, card_of(form, exc, "text_to_video"))
    held = status(form)
    assert held.startswith("### Something went wrong in this app")
    form.user(form.fields["width"], 640)
    form.user(form.fields["prompt"], "a different prompt")
    assert status(form) == held


def test_a_quota_card_collapses_when_a_lever_it_names_changes(form):
    form.pick_mode("text_to_video")
    form.user(form.fields["prompt"], "a tram")
    exc = errors.DecodeError(mode="text_to_video", reason="not enough quota")
    exc.category, exc.gpu_used = errors.QUOTA, False
    finished(form, card_of(form, exc, "text_to_video"))
    assert status(form).startswith("### Not enough GPU quota")
    form.user(form.fields["prompt"], "a tram at night")
    assert status(form).startswith("###")
    form.user(form.fields["enhance_prompt"], True)
    assert LINE.match(status(form)), status(form)


def _good_run(form: Form, tmp_path, monkeypatch) -> None:
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
    values = dict(zip(FIELD_KEYS, form.form_values(), strict=True)) | inputs_for(
        "text_to_audio_video"
    )
    good = handlers.run("text_to_audio_video", *(values[k] for k in FIELD_KEYS))
    assert good.error is None, good.status
    finished(form, good)


def test_show_collapses_the_card(form, tmp_path, monkeypatch):
    form.pick_mode("text_to_audio_video")
    _good_run(form, tmp_path, monkeypatch)
    finished(form, enhancer_failed_in_05(form))
    show = next(
        b for b in form.fn("_finish").outputs if isinstance(b, gr.Button) and b.value == "Show"
    )
    form.click(show)
    assert LINE.match(status(form)), status(form)
    assert form.get(form.by_id("monitor-caption")).startswith("**Earlier run.**")


def test_restore_in_the_same_mode_collapses_the_card(form, tmp_path, monkeypatch):
    form.pick_mode("text_to_audio_video")
    _good_run(form, tmp_path, monkeypatch)
    finished(form, enhancer_failed_in_05(form))
    restore = next(
        b
        for b in form.demo.blocks.values()
        if isinstance(b, gr.Button) and b.value == "Restore settings"
    )
    form.click(restore)
    assert form.get(form.radio) == "text_to_audio_video"
    assert LINE.match(status(form)), status(form)
