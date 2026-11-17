"""Arrow keys on a LoRA strength slider reach the run (review round 4, minor 1).

The strength was wired to the slider's ``release`` only, and Gradio 6.24's slider sends
``release`` on pointer-up over the range or on leaving its number box: arrow keys on the range
send ``input`` and ``change``. So a keyboard user set 0.7 on the slider while the accordion, the
cost panel's LoRA row and the run kept 1.0. Now ``input`` is wired too (``always_last``), and the
value it sends is held to the slider's own range, read off the page: a value typed into the
number box passes through numbers outside it ("0" on the way to "0.7"), which Gradio would refuse
before any handler with an error on screen.

Driven through the built page's own steps (:class:`test_ui_form.Form`), with ``Form.key``: the
keyboard's events, no ``release``.
"""

from __future__ import annotations

import pytest
from lora_fixtures import good_adapter
from test_ui_form import Form

from ltx25_studio.ui import blocks, handlers, render, showcase

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


@pytest.fixture
def form(tmp_path, monkeypatch):
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    return Form(blocks.build_ui())


def staged_form(form: Form, tmp_path) -> list:
    """One adapter checked and staged through the handler, its row shown as the page shows it."""
    _verdict, staged, _summary = handlers.on_validate_lora("", str(good_adapter(tmp_path)), [])
    assert len(staged) == 1
    form.values[form.fields["loras"]._id] = staged
    for name in ("_lora_rows", "_lora_label", "_review"):
        form.run(form.fn(name))
    return staged


def strength_sliders(form: Form) -> list:
    return [
        c
        for c in form.fn("_lora_rows").outputs
        if isinstance(c, gr.Slider) and str(c.label).startswith("Strength of adapter")
    ]


def lora_row_of_the_panel(form: Form) -> str:
    import re

    panel = re.sub(r"<[^>]+>", "", form.get(form.by_id("cost"))).replace(" ", " ")
    return re.search(r"LoRA(.*?)$", panel).group(1)


def test_an_arrow_key_on_the_strength_reaches_the_staged_adapter_and_the_run(form, tmp_path):
    staged_form(form, tmp_path)
    slider = strength_sliders(form)[0]
    assert form.visible(slider)
    form.key(slider, 0.7)
    (entry,) = form.field("loras")
    assert entry["weight"] == 0.7, "the keyboard's value reached the staged adapter"
    accordion = form.by_id("lora")
    assert form.props[accordion._id]["label"] == render.lora_summary_label([entry])
    assert "at 0.7" in form.props[accordion._id]["label"]
    assert "at 0.7" in lora_row_of_the_panel(form)


def test_the_keyboards_event_is_wired_last_wins_and_held_to_the_sliders_own_range(form):
    for index, slider in enumerate(strength_sliders(form)):
        (listener,) = form.listeners(slider, "input")
        assert listener.trigger_mode == "always_last"
        assert listener.api_visibility == "private"
        assert slider.elem_id == f"lora-strength-{index}"
        js = listener.js
        assert f"#lora-strength-{index} input[type=\"range\"]" in js
        assert "Math.min(Math.max(" in js and "range.min" in js and "range.max" in js
        (released,) = form.listeners(slider, "release")
        assert released.fn.__name__ == listener.fn.__name__, "the pointer still works"


def test_the_pointer_path_still_sets_the_strength(form, tmp_path):
    staged_form(form, tmp_path)
    form.user(strength_sliders(form)[0], 1.25)
    assert form.field("loras")[0]["weight"] == 1.25
