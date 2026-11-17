""""Use this prompt" turns Enhance off and says so (review round 4, minor 9).

After a run with Enhance on, the Done card shows the prompt the model was sent and offers "Use
this prompt". It wrote that prompt into the prompt box and left Enhance on, so the next run
enhanced the enhanced prompt again (and on the Space reserved its seconds again): the next run
did not use "this prompt". Now the same step turns Enhance off, and the button says what it did.
In a mode that takes no prompt nothing hidden is written: the prompt is parked, and comes back
with the prompt box.

Driven through the built page's own steps (:class:`test_ui_form.Form`): a run through
``handlers.run`` and the real ``generate()`` (the pipeline and the enhancer faked), the real
finish step, the real click.
"""

from __future__ import annotations

import pytest
from conftest import inputs_for
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder
from test_ui_form import Form

from ltx25_studio import generate as generate_module
from ltx25_studio.ui import blocks, handlers, render, showcase
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

TYPED = "a dog runs on a beach"


class Request:
    session_hash = "use-this-prompt"


@pytest.fixture
def form(tmp_path, monkeypatch):
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    real = generate_module.generate
    monkeypatch.setattr(
        generate_module,
        "generate",
        lambda **params: real(
            pipeline_factory=factory_returning(FakePipe()),
            encoder=one_muxed_file_encoder,
            enhancer=lambda prompt, image=None: "A REWRITTEN " + prompt,
            output_root=tmp_path / "runs",
            retention_bytes=10**9,
            **params,
        ),
    )
    return Form(blocks.build_ui())


def enhanced_run(form: Form) -> handlers.RunOutcome:
    form.pick_mode("text_to_audio_video")
    form.user(form.fields["prompt"], TYPED)
    form.user(form.fields["enhance_prompt"], True)
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_audio_video")
    values |= {"prompt": TYPED, "enhance_prompt": True}
    outcome = handlers.run("text_to_audio_video", *(values[k] for k in FIELD_KEYS))
    assert outcome.error is None, outcome.status
    form.values[form.by_id("run-status")._id] = outcome.status
    blocks._remember(Request(), outcome)
    form.run(form.fn("_finish"), request=Request())
    return outcome


def use_prompt_button(form: Form):
    return next(
        c
        for c in form.fn("_finish").outputs
        if isinstance(c, gr.Button) and c.value == "Use this prompt"
    )


def test_use_this_prompt_writes_it_and_turns_enhance_off_saying_so(form):
    enhanced_run(form)
    button = use_prompt_button(form)
    assert form.visible(button) and form.get(button) == "Use this prompt"
    assert form.field("enhance_prompt") is True

    form.click(button)
    assert form.field("prompt") == "A REWRITTEN " + TYPED
    assert form.field("enhance_prompt") is False, "the next run sends this prompt as it is"
    assert form.get(button) == "Copied to your prompt; Enhance is off" == render.PROMPT_USED
    assert not form.interactive(button)


def test_the_next_run_offers_the_button_afresh(form):
    enhanced_run(form)
    form.click(use_prompt_button(form))
    enhanced_run(form)
    button = use_prompt_button(form)
    assert form.get(button) == "Use this prompt" and form.interactive(button)


def test_in_a_mode_that_takes_no_prompt_it_is_parked_not_written_hidden(form):
    enhanced_run(form)
    button = use_prompt_button(form)
    form.pick_mode("image_to_video")
    assert not form.visible(form.fields["prompt"])
    form.click(button)
    assert form.field("prompt") == "", "no hidden value is ever sent"
    assert form.get(button) == "Copied to your prompt; Enhance is off"
    form.pick_mode("text_to_audio_video")
    assert form.field("prompt") == "A REWRITTEN " + TYPED
    assert form.field("enhance_prompt") is False
