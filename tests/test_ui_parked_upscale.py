"""Off CUDA, a parked Upscale promises no size (review round 4, minor 7).

Where this machine cannot run Upscale ×2 for the request, the control is parked: unticked,
disabled, with the backend's reason as its line. Its label still read "Upscale ×2: returns
1408 × 960" over "Upscale ×2 needs a CUDA GPU; this machine runs on the CPU.", and the cost panel
kept an unlit Upscale lamp. P12's rule (a size only when it will run) now holds here too: while
parked the label is the bare "Upscale ×2" and the panel has no Upscale lamp; at a size this
machine can upscale, both come back.

Driven through the built page's own steps (:class:`test_ui_form.Form`), the first paint
included.
"""

from __future__ import annotations

import re
import sys

import pytest
from test_ui_form import Form

from ltx25_studio import config
from ltx25_studio.ui import blocks, handlers, render, showcase

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


@pytest.fixture
def on_the_cpu(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "spaces", raising=False)
    monkeypatch.setenv(config.ENV_DEVICE, "cpu")
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    assert handlers.run_platform() == ("cpu", False)


def first_paint(form: Form) -> None:
    for f in form.fns:
        if any(event == "load" for _target, event in (f.targets or [])):
            form.run(f)


def lamps(form: Form) -> list[str]:
    return re.findall(r'class="mr-lamp mr-lamp-cost[^"]*">([^<]*)<', form.get(form.by_id("cost")))


def test_the_parked_upscale_names_no_size_and_the_panel_drops_its_lamp(on_the_cpu):
    form = Form(blocks.build_ui())
    upscale = form.fields["enable_stage_2"]
    first_paint(form)
    assert not form.interactive(upscale)
    assert form.props[upscale._id]["label"] == "Upscale ×2", "a parked control promises no size"
    assert form.props[upscale._id]["info"] == render.upscale_needs_cuda("cpu")
    assert not any(lamp.startswith("Upscale") for lamp in lamps(form)), lamps(form)

    form.pick_mode("text_to_video")
    form.user(form.fields["prompt"], "a tram")
    assert form.props[upscale._id]["label"] == "Upscale ×2"
    assert not any(lamp.startswith("Upscale") for lamp in lamps(form)), lamps(form)

    for key, value in (("width", 64), ("height", 64), ("num_frames", 9)):
        form.user(form.fields[key], value)
    assert form.interactive(upscale), "a size this machine can upscale frees the control"
    assert form.props[upscale._id]["label"] == "Upscale ×2: returns 128 × 128"
    assert "Upscale" in lamps(form)

    form.user(form.fields["width"], 704)
    assert not form.interactive(upscale)
    assert form.props[upscale._id]["label"] == "Upscale ×2", "parked again, no size again"


def test_the_review_writes_the_upscale_label_and_the_size_hints_do_not(on_the_cpu):
    """The review decides the label with the parking, so no second writer can race it."""
    demo = blocks.build_ui()
    upscale = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "upscale")
    writers = {f.fn.__name__ for f in demo.fns.values() if f.fn and upscale in (f.outputs or [])}
    assert "_review" in writers and "_hints" not in writers, writers
