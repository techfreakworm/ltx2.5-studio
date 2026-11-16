"""The memory card (F73): this machine's memory check declined a step before it started.

The session's refused run read "Something went wrong in this app. Any GPU time this run used is
billed. Press Generate again." The backend now files such a refusal under its own category,
"memory", with no GPU time used. The card is keyed by the category's string, so nothing here
imports the backend's class: the UI is handed an exception carrying ``category = "memory"`` and
``gpu_used = False``, as ``errors.categorise`` reads them.
"""

from __future__ import annotations

import html
import re
import sys

import pytest
from conftest import inputs_for

from ltx25_studio import config, device, errors, zerogpu
from ltx25_studio import generate as generate_module
from ltx25_studio.ui import blocks, handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

HEADLINE = "This machine does not have enough free memory for this run"
STEP = "Close other apps, or make the run smaller, then press Generate again."
#: The detail the backend writes, with the numbers to change (the session's refusal).
DETAIL = (
    "Not enough memory for 'load transformer': 67.3 GB is already in use and this step needs "
    "about 25.1 GB more, 92.4 GB in all, which is over the 90.0 GB limit by 2.4 GB. Nothing "
    "was loaded. Close other apps, pick a smaller size or fewer frames, or raise "
    "LTX25_MEMORY_LIMIT_GB if you have measured that this machine can take it."
)


class MemoryRefusal(errors.LTXStudioError):
    """Stands for the backend's refusal: only the category and gpu_used the UI reads."""

    category = "memory"
    gpu_used = False


def visible(markup: str) -> str:
    markup = re.sub(r"<details(?![^>]*\bopen\b)[^>]*>.*?</details>", "", markup, flags=re.S)
    return html.unescape(" ".join(re.sub(r"<[^>]+>", " ", markup).split()))


@pytest.fixture
def on_the_cpu(monkeypatch):
    monkeypatch.delitem(sys.modules, "spaces", raising=False)
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: False)
    monkeypatch.setattr(device, "host_has_mps", lambda: False)
    monkeypatch.setenv(config.ENV_DEVICE, "cpu")
    assert handlers.run_platform() == ("cpu", False)


@pytest.mark.parametrize("on_zerogpu", [False, True], ids=["off ZeroGPU", "on ZeroGPU"])
def test_the_memory_card_names_the_machine_one_step_and_opens_its_numbers(on_zerogpu):
    exc = MemoryRefusal(DETAIL)
    assert errors.categorise(exc) == errors.Failure("memory", False)
    card = render.failure(exc, on_zerogpu=on_zerogpu)
    head, _, body = card.partition("\n")
    assert head == f"### {HEADLINE}"
    words = visible(card)
    assert STEP in words
    assert '<details class="mr-detail" open>' in card, "the detail opens by itself"
    assert DETAIL in words, "the numbers to change are shown, word for word"
    assert "Something went wrong" not in words and "billed" not in words.lower()
    if on_zerogpu:
        assert "Nothing ran. No GPU time was used." in words
    else:
        assert words.startswith(f"### {HEADLINE} {STEP}"), "off ZeroGPU: no quota line"


def test_a_memory_refusal_from_the_run_reaches_the_page_as_the_memory_card(
    on_the_cpu, monkeypatch
):
    """Through the real handler: ``generate()`` raises the refusal, the status is the card,
    and the finish step captions it as refused before running."""
    pytest.importorskip("gradio", reason="gradio is a declared dev dependency")
    from test_ui_form import Form

    exc = MemoryRefusal(DETAIL)

    def generate(**_params):
        raise exc

    monkeypatch.setattr(generate_module, "generate", generate)
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_video")
    outcome = handlers.run("text_to_video", *(values[k] for k in FIELD_KEYS))
    assert outcome.error is exc and not outcome.refused
    assert outcome.status.startswith(f"### {HEADLINE}\n\n{STEP}")
    assert handlers.ran_nothing(outcome)

    form = Form(blocks.build_ui())

    class Request:
        session_hash = "memory-card"

    form.values[form.by_id("run-status")._id] = outcome.status
    blocks._remember(Request(), outcome)
    form.run(form.fn("_finish"), request=Request())
    assert form.get(form.by_id("monitor-caption")) == render.monitor_caption(kind="refused")
    assert form.get(form.by_id("generate-btn")) == "Generate"
