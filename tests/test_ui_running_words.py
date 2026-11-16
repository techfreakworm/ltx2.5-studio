"""While a run is in progress, the report says what is happening, where it runs.

On ZeroGPU a run first reserves GPU time from the user's quota, and the sentence says so.
Anywhere else there is no quota and no reservation, yet the lead's run on a Mac read
"Reserving GPU time from your quota" through 208 s of local work. There the sentence says
only what happens: loading, sampling and decoding on this machine, named by the device that
runs it, in the words the finished run's "Took" line uses.

Driven through the page's own start step, with this process resolved as ``app.py``
resolves it (``device.current()``): a stand-in platform package in ``sys.modules`` for
ZeroGPU, ``LTX25_DEVICE`` and the CUDA probe for everything else.
"""

from __future__ import annotations

import sys
import types

import pytest

from ltx25_studio import config, device, modes, zerogpu
from ltx25_studio.ui import blocks, handlers, render

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

MODE = "text_to_audio_video"
LABEL = modes.get_mode(MODE).label
SPACE_SENTENCE = "Reserving GPU time from your quota, then loading, sampling and decoding."
#: Words that belong to ZeroGPU's quota and reservation, never to a local run.
QUOTA_WORDS = ("quota", "reserv", "billed")


@pytest.fixture(scope="module")
def start():
    """The page's start step: the report it shows the instant Generate is pressed."""
    demo = blocks.build_ui()
    run = next(f for f in demo.fns.values() if f.api_name == "on_generate")
    step = demo.fns[run.trigger_after]
    status = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "run-status")

    def started(*, enhance: bool = False, prompt: str = "a harbour at dawn") -> str:
        out = step.fn(MODE, enhance, prompt)
        return out[step.outputs.index(status)]["value"]

    return started


@pytest.fixture
def local(monkeypatch):
    """This process off the Space (no platform package, no CUDA), on ``LTX25_DEVICE``."""
    monkeypatch.delitem(sys.modules, "spaces", raising=False)
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: False)
    monkeypatch.setattr(device, "host_has_mps", lambda: True)

    def on(requested: str | None) -> None:
        if requested is None:
            monkeypatch.delenv(config.ENV_DEVICE, raising=False)
        else:
            monkeypatch.setenv(config.ENV_DEVICE, requested)

    return on


@pytest.fixture
def space(monkeypatch):
    """This process on ZeroGPU: the platform package is imported (``app.py`` does it first)."""
    module = types.ModuleType("spaces")
    module.GPU = lambda *a, **k: lambda f: f
    monkeypatch.setitem(sys.modules, "spaces", module)
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)


def test_on_zerogpu_the_sentence_stays(start, space):
    assert start() == f"**Running: {LABEL}.** {SPACE_SENTENCE}"
    assert "inside this run's reservation" in start(enhance=True)


@pytest.mark.parametrize(
    ("requested", "where"),
    [("cpu", "on the CPU"), ("mps", "on this Mac's GPU"), (None, "on this machine")],
)
def test_off_zerogpu_it_says_only_what_happens_on_this_machine(start, local, requested, where):
    local(requested)
    text = start()
    assert text == f"**Running: {LABEL}.** Loading, sampling and decoding {where}."
    for word in QUOTA_WORDS:
        assert word not in text.lower(), f"a local run's report says {word!r}: {text}"


def test_a_cuda_box_that_is_not_a_space_has_no_quota_either(start, local, monkeypatch):
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: True)
    local(None)
    text = start()
    assert text.endswith("Loading, sampling and decoding on the GPU.")
    assert "quota" not in text.lower()


def test_off_zerogpu_enhancing_names_the_rewrite_and_no_reservation(start, local):
    local("cpu")
    text = start(enhance=True)
    assert text.startswith(f"**Enhancing your prompt: {LABEL}.**")
    assert "Gemma rewrites it on this machine first" in text
    for word in QUOTA_WORDS:
        assert word not in text.lower(), f"a local run's report says {word!r}: {text}"


def test_the_finished_run_and_the_running_one_name_the_device_alike():
    for name in ("cpu", "mps", "cuda"):
        took = render.took_words(
            render.RunFacts(
                mode=modes.get_mode(MODE),
                seed=1,
                returned_size=None,
                seconds=None,
                stage_2=False,
                loras=(),
                took=3.0,
                took_is_gpu=True,
                device=name,
            )
        )
        running = render.running(modes.get_mode(MODE), on_zerogpu=False, device=name)
        assert running.endswith(took.split(" s ", 1)[1] + "."), (name, took, running)


def test_a_caller_that_does_not_say_where_gets_the_spaces_words():
    """The page always says (``run_platform``); a bare call keeps the Space's sentence."""
    assert SPACE_SENTENCE in handlers.on_generate_start(MODE)
