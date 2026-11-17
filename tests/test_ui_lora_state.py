"""Staging an adapter survives Gradio's own state tracking (#20; review round 4, after minor 2).

The adapter list is a ``gr.State`` (the field ``loras``). Minor 2 put a change listener on every
form field, so that a failure card collapses when a control its step names changes, and the
adapter list got one too. Gradio hashes a State that has a change listener before and after every
event that writes it (``Blocks.process_api``: ``get_state_ids_to_track``, then
``utils.deep_hash``), and a staged adapter carries the validator's report, a frozen dataclass
that holds dicts, which cannot be hashed. So once an adapter passed its check, every event that
writes the list failed on the page: Check and add and the upload ("Error" where the accepted card
belongs, nothing staged, the file left in the drop zone), Remove, Remove all and the strength. The
handlers' own tests call them directly and never met Gradio's tracking.

Driven here as the page drives them: each step through Gradio's own ``process_api`` with one
session state, then the listeners the page runs next (a step's ``.then`` listeners when it
completes, its ``.failure`` listeners when it fails, and the change listeners of any State Gradio
reports changed). The adapter list carries no change listener now; its own events collapse a card
whose step names the adapters, when they changed the list.
"""

from __future__ import annotations

import asyncio
import re

import pytest
from lora_fixtures import good_adapter, wrong_dimension_adapter
from test_ui_gradio_refusal import payload

from ltx25_studio import errors, lora
from ltx25_studio.ui import blocks, render, showcase

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

LINE = re.compile(r"\*\*Last run:\*\* .+, \d\d:\d\d\. .+\.$")


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """The page, with the Hub standing in as two adapters on disk (one made for this model, one
    2B one) and the LoRA cache in a temporary folder."""
    monkeypatch.setattr(lora, "lora_cache_dir", lambda: tmp_path / "lora-cache")
    hub = {
        "org/fits": good_adapter(tmp_path / "hub"),
        "org/2b": wrong_dimension_adapter(tmp_path / "hub"),
    }
    monkeypatch.setattr(lora, "fetch_lora", lambda source: hub[source])
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    return blocks.build_ui()


class Page:
    """One browser session on ``demo``: Gradio's own steps, on one session state."""

    def __init__(self, demo):
        from gradio.state_holder import SessionState

        self.demo = demo
        self.state = SessionState(demo)
        self.values: dict = {}  # what the page holds, by component id

    def button(self, label: str, index: int = 0):
        found = [
            b for b in self.demo.blocks.values() if isinstance(b, gr.Button) and b.value == label
        ]
        return found[index]

    def by_elem_id(self, elem_id: str):
        return next(b for b in self.demo.blocks.values() if getattr(b, "elem_id", None) == elem_id)

    def fn(self, name: str):
        return next(f for f in self.demo.fns.values() if getattr(f.fn, "__name__", "") == name)

    def listeners(self, block_id: int, event: str) -> list:
        return [f for f in self.demo.fns.values() if (block_id, event) in (f.targets or [])]

    def fire(self, component, event: str, value=None) -> list[tuple[str, str]]:
        """The user acts on ``component`` (now holding ``value``, if given). Returns each step's
        name with "ok" or its exception's class."""
        if value is not None:
            self.values[component._id] = value
        queue, log = self.listeners(component._id, event), []
        while queue:
            step = queue.pop(0)
            name = getattr(step.fn, "__name__", repr(step.fn))
            try:
                result = asyncio.run(
                    self.demo.process_api(
                        block_fn=step,
                        inputs=payload(step, self.values),
                        state=self.state,
                        request=gr.Request(session_hash="lora-state"),
                        session_hash="lora-state",
                        in_event_listener=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- the page runs the failure listeners alone
                log.append((name, type(exc).__name__))
                queue += [
                    f
                    for f in self.demo.fns.values()
                    if f.trigger_after == step._id and f.trigger_only_on_failure
                ]
                continue
            log.append((name, "ok"))
            for block, value_out in zip(step.outputs, result["data"], strict=True):
                if block.stateful:
                    continue
                if isinstance(value_out, dict) and value_out.get("__type__") == "update":
                    if "value" in value_out:
                        self.values[block._id] = value_out["value"]
                else:
                    self.values[block._id] = value_out
            queue += [
                f
                for f in self.demo.fns.values()
                if f.trigger_after == step._id and not f.trigger_only_on_failure
            ]
            for state_id in result.get("changed_state_ids") or []:
                queue += self.listeners(state_id, "change")
        return log


def ok(log) -> bool:
    return bool(log) and all(outcome == "ok" for _name, outcome in log)


def adapters(page: Page) -> list:
    """The adapter list as the session holds it."""
    check = next(f for f in page.demo.fns.values() if f.api_name == "on_validate_lora")
    return page.state[next(o for o in check.outputs if o.stateful)._id]


def check_and_add(page: Page, repo_id: str) -> list:
    check = page.button("Check and add")
    page.values[page.listeners(check._id, "click")[0].inputs[0]._id] = repo_id
    return page.fire(check, "click")


def test_an_adapter_is_checked_reweighed_removed_and_cleared_through_gradios_own_steps(demo):
    page = Page(demo)
    log = check_and_add(page, "org/fits")
    assert ok(log), log
    assert [entry["weight"] for entry in adapters(page)] == [1.0]
    assert str(page.values[page.by_elem_id("lora-verdict")._id]).startswith("#### Accepted: ")

    strength = page.by_elem_id("lora-strength-0")
    for event, weight in (("release", 0.7), ("input", 0.5)):  # the pointer, then the keyboard
        log = page.fire(strength, event, weight)
        assert ok(log), (event, log)
        assert [entry["weight"] for entry in adapters(page)] == [weight], event

    log = page.fire(page.button("Remove"), "click")
    assert ok(log), log
    assert adapters(page) == []

    assert ok(check_and_add(page, "org/fits"))
    log = page.fire(page.button("Remove all"), "click")
    assert ok(log), log
    assert adapters(page) == []


def test_no_state_on_the_page_carries_a_change_listener(demo):
    """Gradio would hash such a State before and after every event that writes it, and the
    adapter list holds the validator's report, which cannot be hashed."""
    states = {b._id: b for b in demo.blocks.values() if getattr(b, "stateful", False)}
    watched = sorted(
        block_id
        for f in demo.fns.values()
        for block_id, event in (f.targets or [])
        if event == "change" and block_id in states
    )
    assert watched == []


def test_a_card_naming_the_adapters_collapses_when_an_adapter_event_changes_the_list(demo):
    """Minor 2 for the adapter list, from its own events: "Remove the adapter, or try a different
    one" is acted on when the list changes; a rejected adapter changes nothing."""
    page = Page(demo)
    exc = errors.DecodeError(mode="text_to_video", reason="the adapter does not attach")
    exc.category, exc.gpu_used = errors.ADAPTER, True
    card = render.failure(exc)
    headline = render.card_headline(card)
    assert render.step_names(headline, "loras")
    status = page.by_elem_id("run-status")
    last_failure = page.fn("_card_acted_on").inputs[0]
    page.values[status._id] = card
    page.state[last_failure._id] = {"mode": "text_to_video", "at": "08:06", "headline": headline}

    assert ok(check_and_add(page, "org/2b"))
    assert adapters(page) == []
    assert page.values[status._id] == card, "a rejected adapter left the list as it was"
    assert page.state[last_failure._id] is not None

    assert ok(check_and_add(page, "org/fits"))
    assert LINE.match(str(page.values[status._id])), page.values[status._id]
    assert str(page.values[status._id]).endswith(f". {headline}.")
    assert page.state[last_failure._id] is None
