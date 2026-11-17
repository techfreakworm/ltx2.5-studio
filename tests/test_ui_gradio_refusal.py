"""A request Gradio refuses before any handler runs ends with Generate back and a card (#5, F28).

Two ways Gradio could fail the generation step while preparing its inputs, before any of the
app's code ran: a seed typed below the field's ``minimum=-1``, and an upload whose file Gradio's
cache clean-up has removed. The failure step that brings Generate back took the same inputs, so
it failed the same way, and the page stayed on "Generating…" with no card.

Now no typed number field declares a range (the app's own gate refuses a seed below -1, in the
page's words), and a last failure listener that takes no inputs follows each failure step, so
nothing Gradio does while preparing inputs can stop Generate coming back with a card.

Driven as the page drives it: each step through Gradio's own ``process_api``, then the listeners
Gradio's page runs next (a step's ``.then`` listeners when it completes, its ``.failure``
listeners alone when it fails).
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

import pytest

from ltx25_studio.ui import blocks, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


@pytest.fixture
def demo():
    return blocks.build_ui()


def visible(markup) -> str:
    markup = re.sub(r"<details(?![^>]*\bopen\b)[^>]*>.*?</details>", "", str(markup), flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", markup).split())


def detail(markup) -> str:
    found = re.search(r"<summary>Detail</summary><p>(.*?)</p>", str(markup), flags=re.S)
    return found.group(1) if found else ""


def generation_step(demo):
    return next(f for f in demo.fns.values() if f.api_name == "on_generate")


def by_elem_id(demo, elem_id):
    return next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == elem_id)


def page_values(demo, mode_id: str, **overrides) -> dict:
    """The page's values by component id: the field defaults, the mode, ``overrides``."""
    run = generation_step(demo)
    values = {spec.key: spec.default for spec in FIELDS} | overrides
    by_id = {run.inputs[0]._id: mode_id}
    by_id |= {c._id: values[k] for c, k in zip(run.inputs[1:], FIELD_KEYS, strict=True)}
    by_id[by_elem_id(demo, "length-source")._id] = render.LENGTH_MATCH
    return by_id


def payload(step, values: dict) -> list:
    """What the page sends for ``step``'s inputs. A ``gr.State`` sends nothing (Gradio reads it
    from the session); a file the page already holds is sent as the page holds it."""
    sent = []
    for block in step.inputs:
        if block.stateful:
            sent.append(None)
            continue
        value = values.get(block._id, block.value)
        if isinstance(value, dict) and "meta" in value:
            sent.append(value)
            continue
        value = block.postprocess(value)
        sent.append(value.model_dump() if hasattr(value, "model_dump") else value)
    return sent


def press_generate(demo, values: dict, session: str):
    """Generate, as Gradio's page runs it. Returns ``(applied, log, state)``: what the page
    holds for each component it was sent (an update merges into what it held: one without a
    value keeps the value), each step's name with "ok" or the exception's class, and the
    session state."""
    from gradio.state_holder import SessionState

    state = SessionState(demo)
    start = demo.fns[generation_step(demo).trigger_after]
    queue, applied, log = [start], {}, []
    while queue:
        step = queue.pop(0)
        name = getattr(step.fn, "__name__", repr(step.fn))
        try:
            data = asyncio.run(
                demo.process_api(
                    block_fn=step,
                    inputs=payload(step, values),
                    state=state,
                    request=gr.Request(session_hash=session),
                    session_hash=session,
                    in_event_listener=True,
                )
            )["data"]
        except Exception as exc:  # noqa: BLE001 -- the page runs the failure listeners alone
            log.append((name, type(exc).__name__))
            queue += [
                f
                for f in demo.fns.values()
                if f.trigger_after == step._id and f.trigger_only_on_failure
            ]
            continue
        log.append((name, "ok"))
        for component, value in zip(step.outputs, data, strict=True):
            sent = value if isinstance(value, dict) and value.get("__type__") == "update" else None
            now = dict(applied.get(component._id) or {})
            now.update(
                {k: v for k, v in sent.items() if k != "__type__"} if sent else {"value": value}
            )
            applied[component._id] = now
        queue += [
            f
            for f in demo.fns.values()
            if f.trigger_after == step._id and not f.trigger_only_on_failure
        ]
    return applied, log, state


def held(applied: dict, component) -> object:
    return (applied.get(component._id) or {}).get("value")


def running_state(demo, state) -> bool:
    start = demo.fns[generation_step(demo).trigger_after]
    running = next(c for c in start.outputs if isinstance(c, gr.State) and c.value is False)
    return state[running._id]


def test_no_typed_number_field_declares_a_range_and_the_sliders_keep_theirs(demo):
    fields = dict(zip(FIELD_KEYS, generation_step(demo).inputs[1:], strict=True))
    numbers = {key: c for key, c in fields.items() if isinstance(c, gr.Number)}
    assert set(numbers) == {"width", "height", "num_frames", "min_seconds", "max_seconds", "seed"}
    for key, number in numbers.items():
        assert number.minimum is None and number.maximum is None, key
    for key in ("strength", "noise_scale"):
        assert isinstance(fields[key], gr.Slider) and fields[key].maximum is not None, key


def test_a_seed_below_minus_one_comes_back_refused_in_the_pages_words(demo):
    values = page_values(demo, "text_to_video", prompt="A tram climbs a cobbled street", seed=-2)
    applied, log, state = press_generate(demo, values, "seed-below")

    assert log[:3] == [("_start", "ok"), ("_on_generate", "ok"), ("_finish", "ok")], log
    assert held(applied, by_elem_id(demo, "generate-btn")) == "Generate"
    assert running_state(demo, state) is False
    card = held(applied, by_elem_id(demo, "run-status"))
    assert visible(card).startswith("### Refused before running: fix the input"), card
    assert "seed=-2 is out of range" in str(card)
    assert held(applied, by_elem_id(demo, "monitor-caption")) == render.monitor_caption(
        kind="refused"
    )


def test_a_generation_step_gradio_cannot_prepare_still_gives_the_page_back(demo):
    gone = Path(tempfile.gettempdir()) / "ltx25-uib-an-upload-the-cache-removed.png"
    assert not gone.exists()
    upload = {
        "path": str(gone),
        "url": None,
        "orig_name": "harbour.png",
        "size": None,
        "mime_type": None,
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }
    values = page_values(demo, "image_to_video", image=upload)
    applied, log, state = press_generate(demo, values, "upload-gone")

    names = [name for name, _outcome in log]
    assert names[:3] == ["_start", "_on_generate", "_finish_failed"], log
    assert log[1][1] != "ok" and log[2][1] != "ok", "Gradio refused the inputs of both"
    assert log[-1] == ("_finish_bare", "ok"), log

    generate = applied[by_elem_id(demo, "generate-btn")._id]
    assert generate["value"] == "Generate" and generate["interactive"] is True
    assert running_state(demo, state) is False
    assert applied[by_elem_id(demo, "monitor-idle")._id]["visible"] is True
    for player in ("video-out", "audio-out"):
        update = applied[by_elem_id(demo, player)._id]
        assert update["visible"] is False and update["value"] is None, player
    card = held(applied, by_elem_id(demo, "run-status"))
    words = visible(card)
    assert words.startswith("### Your inputs could not be read"), words
    assert "Nothing ran." in words and "add it again, then press Generate" in words
    assert "image" in detail(card), "the Detail carries Gradio's own message"
    assert held(applied, by_elem_id(demo, "monitor-caption")) == render.monitor_caption(
        kind="refused"
    )


def test_the_last_listener_takes_no_inputs_and_follows_every_failure_step(demo):
    run = generation_step(demo)
    failure_steps = [
        f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "_finish_failed"
    ]
    assert len(failure_steps) == 2, "one after the run's step, one after the finish step"
    for step in failure_steps:
        (last,) = [f for f in demo.fns.values() if f.trigger_after == step._id]
        assert last.trigger_only_on_failure and last.fn.__name__ == "_finish_bare"
        assert last.inputs == [], "nothing Gradio prepares can fail it"
        assert last.api_visibility == "private"
        assert by_elem_id(demo, "generate-btn") in last.outputs
    assert run.inputs, "the generation step itself keeps its inputs"


@pytest.mark.parametrize("on_zerogpu", [True, False])
def test_the_card_says_nothing_ran_in_the_platforms_words(on_zerogpu):
    card = render.inputs_unread("Could not preprocess input", hours=6, on_zerogpu=on_zerogpu)
    words = visible(card)
    assert words.startswith("### Your inputs could not be read")
    assert ("No GPU time was used." in words) is on_zerogpu
    assert "more than 6 hours ago" in words
    if not on_zerogpu:
        for meter in ("quota", "reserv", "bill", "pay"):
            assert meter not in words.lower(), meter
    assert render.card_headline(card) == "Your inputs could not be read"
