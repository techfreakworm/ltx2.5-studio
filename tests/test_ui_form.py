"""A stand-in for the page: the built Blocks' own event functions, run as the browser runs them.

:class:`Form` holds every component's value and visible/interactive state. A user action
sets a value and fires that component's ``input`` listeners, then its ``change`` listeners
when the value changed; a listener's outputs are written back, a value an output changed
fires that component's ``change`` listeners, and ``.then`` steps follow. Nothing here
re-implements a handler: every function run is one ``build_ui`` wired.

The other UI state tests import :class:`Form`; the tests below pin the harness itself.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from ltx25_studio.ui.fields import FIELD_KEYS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

_PROPS = ("visible", "interactive", "label", "info", "placeholder", "choices", "elem_classes")


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return type(a) is type(b) and bool(a == b)
    except Exception:  # noqa: BLE001 -- an image compares by identity
        return False


class Form:
    """The page's component values, driven through the real event functions."""

    def __init__(self, demo: gr.Blocks):
        self.demo = demo
        self.fns = list(demo.fns.values())
        self.values: dict[int, Any] = {}
        self.props: dict[int, dict] = {}
        for block in demo.blocks.values():
            self.props[block._id] = {k: getattr(block, k, None) for k in _PROPS}
            if hasattr(block, "value"):
                value = block.value
                self.values[block._id] = (
                    copy.deepcopy(value) if isinstance(value, (dict, list)) else value
                )
        generate = next(f for f in self.fns if f.api_name == "on_generate")
        self.radio = generate.inputs[0]
        self.fields = dict(zip(FIELD_KEYS, generate.inputs[1:], strict=True))
        self.ran: list[str] = []

    # -- lookups ----------------------------------------------------------------------------
    def by_id(self, elem_id: str):
        return next(b for b in self.demo.blocks.values() if getattr(b, "elem_id", None) == elem_id)

    def fn(self, name: str):
        return next(f for f in self.fns if getattr(f.fn, "__name__", "") == name)

    def get(self, component) -> Any:
        return self.values.get(component._id)

    def field(self, key: str) -> Any:
        return self.get(self.fields[key])

    def form_values(self) -> tuple:
        """The ``FIELD_KEYS`` values, in order: what ``/on_generate`` would be sent."""
        return tuple(self.get(self.fields[key]) for key in FIELD_KEYS)

    def visible(self, component) -> bool:
        return self.props[component._id].get("visible") is not False

    def interactive(self, component) -> bool:
        return self.props[component._id].get("interactive") is not False

    def listeners(self, component, event: str) -> list:
        return [f for f in self.fns if (component._id, event) in (f.targets or [])]

    # -- what a user does ---------------------------------------------------------------------
    def user(self, component, value: Any) -> None:
        """The user sets a control: ``input`` (and a slider's ``release``) listeners, then
        ``change`` if the value changed."""
        old = self.values.get(component._id)
        self.values[component._id] = value
        for event in ("input", "release"):
            for f in self.listeners(component, event):
                self.run(f)
        if not _same(old, value):
            for f in self.listeners(component, "change"):
                self.run(f)

    def copy(self) -> Form:
        """An independent page in the same state (values replaced, never mutated in place)."""
        twin = copy.copy(self)
        twin.values = dict(self.values)
        twin.props = {key: dict(props) for key, props in self.props.items()}
        twin.ran = []
        return twin

    def pick_mode(self, mode_id: str) -> None:
        self.user(self.radio, mode_id)

    def click(self, component) -> None:
        for f in self.listeners(component, "click"):
            self.run(f)

    # -- what the page does with a listener ---------------------------------------------------
    def run(self, f, depth: int = 0, *, request: Any = None) -> None:
        """Run one listener; ``request`` stands in for the ``gr.Request`` Gradio injects."""
        assert depth < 12, "an event cascade did not settle"
        if f.fn is None:
            return
        self.ran.append(getattr(f.fn, "__name__", repr(f.fn)))
        leading = (request,) if request is not None else ()
        out = f.fn(*leading, *[self.values.get(c._id) for c in f.inputs])
        outs = [out] if len(f.outputs) == 1 else list(out)
        changed = []
        for component, value in zip(f.outputs, outs, strict=True):
            if isinstance(value, dict) and value.get("__type__") == "update":
                self.props[component._id].update({k: v for k, v in value.items() if k in _PROPS})
                if "value" not in value:
                    continue
                value = value["value"]
            if not _same(self.values.get(component._id), value):
                changed.append(component)
            self.values[component._id] = value
        for component in changed:
            for g in self.listeners(component, "change"):
                self.run(g, depth + 1)
        for g in self.fns:
            if g.trigger_after == f._id and not g.trigger_only_on_failure:
                self.run(g, depth + 1)


@pytest.fixture(scope="module")
def demo():
    from ltx25_studio.ui import build_ui

    return build_ui()


def test_the_form_starts_from_the_built_values(demo):
    form = Form(demo)
    assert form.get(form.radio) == "text_to_audio_video"
    assert form.field("width") == 704 and form.field("height") == 480
    assert form.field("decoder") == "auto"


def test_a_mode_change_runs_the_real_switch_and_its_cascade(demo):
    form = Form(demo)
    form.user(form.fields["prompt"], "a lighthouse in fog")
    form.pick_mode("image_to_video")
    assert "_switch" in form.ran and "_review" in form.ran
    assert form.field("prompt") == "" and not form.visible(form.fields["prompt"])
    form.pick_mode("text_to_audio_video")
    assert form.field("prompt") == "a lighthouse in fog"
