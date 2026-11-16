"""Whatever ends a run, the page comes back (the lead's run on real weights, 2026-11-16).

That run finished and wrote its file; then Gradio refused to serve it (the output folder was
outside the working directory and the temp dir) while processing the step's outputs, after
the handler had returned. Gradio 6.24's page then runs only that step's ``.failure``
listeners, never its ``.then`` ones (in its ``Blocks`` script an ``error`` status throws, and
the catch dispatches the failure triggers alone). The listener wired there re-sent the same
file, failed the same way, and the page stayed on "Generating…" for good.

Here the steps run through Gradio's own ``process_api``, as its queue runs them, with the
result file outside the working directory and the temp dir: the generation step raises
Gradio's own ``InvalidPathError``, and the step the page runs next -- the real ``.failure``
listener -- passes Gradio's output processing and gives the page back: Generate on, the
monitor idle and holding no file, and a card that says the run finished but its result could
not be shown, what it used, what to do, and the server's message as its Detail.
"""

from __future__ import annotations

import asyncio
import re
import tempfile

import pytest
from conftest import inputs_for
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

from ltx25_studio import generate as generate_module
from ltx25_studio import modes
from ltx25_studio.ui import blocks, handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

MODE = "text_to_audio_video"


def visible(markup: str) -> str:
    markup = re.sub(r"<details(?![^>]*\bopen\b)[^>]*>.*?</details>", "", markup, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", markup).split())


def detail(markup: str) -> str:
    found = re.search(r"<summary>Detail</summary><p>(.*?)</p>", markup, flags=re.S)
    return found.group(1) if found else ""


@pytest.fixture
def demo():
    return blocks.build_ui()


def generation_step(demo):
    return next(f for f in demo.fns.values() if f.api_name == "on_generate")


def listeners_after(demo, step, *, failure: bool) -> list:
    return [
        f
        for f in demo.fns.values()
        if f.trigger_after == step._id and bool(f.trigger_only_on_failure) == failure
    ]


def by_elem_id(demo, elem_id):
    return next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == elem_id)


def form_values() -> dict:
    return {spec.key: spec.default for spec in FIELDS} | inputs_for(MODE)


def payload(step, values: dict) -> list:
    """What the page sends for ``step``'s inputs: each value as its component sends it.
    A ``gr.State`` sends nothing; Gradio reads it from the session."""
    sent = []
    for block in step.inputs:
        if block.stateful:
            sent.append(None)
            continue
        value = block.postprocess(values.get(block._id, block.value))
        sent.append(value.model_dump() if hasattr(value, "model_dump") else value)
    return sent


def run_step(demo, step, values: dict, state, session: str) -> dict:
    """One step, as Gradio's queue runs it (``route_utils.call_process_api``)."""
    return asyncio.run(
        demo.process_api(
            block_fn=step,
            inputs=payload(step, values),
            state=state,
            request=gr.Request(session_hash=session),
            session_hash=session,
            in_event_listener=True,
        )
    )


def page_values(demo, step) -> dict:
    """The page's values by component id, for the generation step and the steps after it."""
    values = form_values()
    by_id = {step.inputs[0]._id: MODE}
    by_id |= {c._id: values[k] for c, k in zip(step.inputs[1:], FIELD_KEYS, strict=True)}
    return by_id


@pytest.fixture
def outside(tmp_path, monkeypatch):
    """A working directory and a temp dir of the test's own, and an output folder in
    neither: the setup of the lead's run. Gradio checks both at call time."""
    cwd, temp, out = tmp_path / "cwd", tmp_path / "temp", tmp_path / "elsewhere" / "outputs"
    for folder in (cwd, temp):
        folder.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    real = generate_module.generate
    monkeypatch.setattr(
        generate_module,
        "generate",
        lambda **params: real(
            pipeline_factory=factory_returning(FakePipe()),
            encoder=one_muxed_file_encoder,
            output_root=out,
            retention_bytes=10**9,
            **params,
        ),
    )
    return out


def launched(demo):
    """Gradio checks a returned file's folder only once the app has launched."""
    demo.has_launched = True
    return demo


def files_in(data) -> list:
    """Every file a step's processed outputs carry (Gradio sends a file as a dict with a
    ``path``)."""
    if isinstance(data, dict):
        own = [data["path"]] if "path" in data else []
        return own + [f for value in data.values() for f in files_in(value)]
    if isinstance(data, (list, tuple)):
        return [f for value in data for f in files_in(value)]
    return []


# --------------------------------------------------------------------------
# the wiring
# --------------------------------------------------------------------------


def test_a_failed_step_is_followed_by_its_own_listener_and_so_is_the_finish_step(demo):
    run = generation_step(demo)
    button = by_elem_id(demo, "generate-btn")
    (finish,) = listeners_after(demo, run, failure=False)
    (failed,) = listeners_after(demo, run, failure=True)
    assert finish.fn.__name__ == "_finish" and failed.fn.__name__ == "_finish_failed"
    assert not finish.trigger_only_on_success, "`.then`: it runs whenever the step completes"
    assert failed.outputs == finish.outputs and button in failed.outputs
    (after_finish,) = listeners_after(demo, finish, failure=True)
    assert after_finish.fn is failed.fn, "a finish step that fails must not leave the page"
    for step in (failed, after_finish):
        assert step.api_visibility == "private" and step.inputs == finish.inputs


# --------------------------------------------------------------------------
# the lead's run, through Gradio's own processing
# --------------------------------------------------------------------------


def test_a_result_gradio_will_not_serve_ends_with_generate_back_and_a_card(demo, outside):
    from gradio.exceptions import InvalidPathError
    from gradio.state_holder import SessionState

    demo = launched(demo)
    run = generation_step(demo)
    (failed,) = listeners_after(demo, run, failure=True)
    start = demo.fns[run.trigger_after]
    state, session = SessionState(demo), "lead-run"
    values = page_values(demo, run)

    run_step(demo, start, values, state, session)
    with pytest.raises(InvalidPathError) as raised:
        run_step(demo, run, values, state, session)
    assert "allowed_paths" in str(raised.value) and str(outside) in str(raised.value)
    written = list(outside.glob("run-*/output.mp4"))
    assert written, "the run itself finished and wrote its file"

    values |= {failed.inputs[1]._id: [], failed.inputs[2]._id: render.LENGTH_MATCH}
    out = run_step(demo, failed, values, state, session)["data"]
    got = {c._id: v for c, v in zip(failed.outputs, out, strict=True)}

    assert files_in(out) == [], "the step after a failed one sent a file"
    generate = got[by_elem_id(demo, "generate-btn")._id]
    assert generate["value"] == "Generate" and generate["interactive"] is True
    running = next(c for c in failed.outputs if isinstance(c, gr.State) and c.value is False)
    assert state[running._id] is False
    assert got[by_elem_id(demo, "monitor-idle")._id]["visible"] is True
    for player in ("video-out", "audio-out"):
        update = got[by_elem_id(demo, player)._id]
        assert update["visible"] is False and update["value"] is None, player
    caption = got[by_elem_id(demo, "monitor-caption")._id]
    assert "Running" not in caption and caption == render.monitor_caption(kind="unshown")

    status = got[by_elem_id(demo, "run-status")._id]
    assert status["visible"] is True
    card = status["value"]
    words = visible(card)
    assert words.startswith("### The run finished, but its result could not be shown")
    assert re.search(r"It used \d+ s", words), "what the run used is stated"
    # The run finished and its file is kept: it is listed in Recent runs, and Show (which
    # costs nothing) is the step, not a paid rerun (review round 3, section 9 (a)).
    assert (
        "Its file is kept: press Show in Recent runs. If that fails too, report it with the "
        "detail below." in words
    )
    assert "Press Generate again" not in words
    recent = next(c for c in failed.outputs if isinstance(c, gr.State) and isinstance(state[c._id], list))
    assert [str(entry["video"]) for entry in state[recent._id]] == [str(written[0])]
    show = next(c for c in failed.outputs if isinstance(c, gr.Button) and c.value == "Show")
    assert got[show._id]["visible"] is True
    assert str(outside) not in words, "no server path outside the Detail"
    assert "allowed_paths" in detail(card), "the Detail carries the server's message"
    assert "Cannot move" in detail(card)


def test_with_the_folder_main_allows_gradio_serves_the_same_run(outside, demo):
    """What ``app.main()`` passes for this folder (``served_outputs``) is what Gradio's own
    check needs: the same run's file now reaches the page and the finish step shows it.
    ``outside`` comes first so the demo's cache is under the test's own temp dir."""
    from gradio.state_holder import SessionState

    demo = launched(demo)
    demo.allowed_paths = blocks.served_outputs(outside)["allowed_paths"]
    run = generation_step(demo)
    (finish,) = listeners_after(demo, run, failure=False)
    state, session = SessionState(demo), "allowed"
    values = page_values(demo, run)
    run_step(demo, demo.fns[run.trigger_after], values, state, session)
    out = run_step(demo, run, values, state, session)["data"]
    assert files_in(out), "the run's file reached the page"
    assert blocks._recall_failure(gr.Request(session_hash=session)) is None

    values |= {finish.inputs[1]._id: [], finish.inputs[2]._id: render.LENGTH_MATCH}
    done = run_step(demo, finish, values, state, session)["data"]
    got = {c._id: v for c, v in zip(finish.outputs, done, strict=True)}
    video = got[by_elem_id(demo, "video-out")._id]
    assert video["visible"] is True and video["autoplay"] is True and files_in(video)
    assert got[by_elem_id(demo, "generate-btn")._id]["value"] == "Generate"
    assert got[by_elem_id(demo, "monitor-caption")._id].startswith("**Your run.**")


def test_when_the_handler_itself_raises_the_page_still_comes_back(demo, outside, monkeypatch):
    from gradio.state_holder import SessionState

    def broken(*args, **kwargs):
        raise RuntimeError("the handler broke after the run")

    monkeypatch.setattr(handlers, "run", broken)
    demo = launched(demo)
    run = generation_step(demo)
    (failed,) = listeners_after(demo, run, failure=True)
    state, session = SessionState(demo), "handler-raised"
    values = page_values(demo, run)
    run_step(demo, demo.fns[run.trigger_after], values, state, session)
    with pytest.raises(RuntimeError):
        run_step(demo, run, values, state, session)

    values |= {failed.inputs[1]._id: [], failed.inputs[2]._id: render.LENGTH_MATCH}
    out = run_step(demo, failed, values, state, session)["data"]
    got = {c._id: v for c, v in zip(failed.outputs, out, strict=True)}
    assert got[by_elem_id(demo, "generate-btn")._id]["value"] == "Generate"
    assert got[by_elem_id(demo, "monitor-caption")._id] == render.monitor_caption(kind="failed")
    card = got[by_elem_id(demo, "run-status")._id]["value"]
    assert visible(card).startswith("### The run ended without a result")
    assert detail(card) == "the handler broke after the run"


def test_a_new_run_never_reads_an_earlier_runs_outcome_or_message(demo):
    run = generation_step(demo)
    start = demo.fns[run.trigger_after]
    request = gr.Request(session_hash="earlier")
    outcome = handlers.RunOutcome(video=None, audio=None, status="### old", mode_id=MODE)
    blocks._remember(request, outcome)
    blocks._note_failure("earlier", RuntimeError("an old message"))
    start.fn(MODE, False, "a harbour", request)
    assert blocks._recall(request) is None and blocks._recall_failure(request) is None


# --------------------------------------------------------------------------
# the report, by what is known about the run
# --------------------------------------------------------------------------


class _Result:
    seed = 7
    warnings = ()
    run_dir = "unused"
    record = None


def _finished(tmp_path, **kwargs) -> handlers.RunOutcome:
    video = tmp_path / "output.mp4"
    video.write_bytes(b"x")
    return handlers.RunOutcome(
        video=str(video),
        audio=str(video),
        status="Done",
        mode_id=MODE,
        params=form_values() | {"width": 704, "height": 480, "num_frames": 49},
        result=_Result(),
        took=208.0,
        **kwargs,
    )


def test_the_report_says_what_the_run_used_where_it_ran(tmp_path, monkeypatch):
    outcome = _finished(tmp_path)
    monkeypatch.setattr(handlers, "_run_device", lambda result: ("mps", False))
    local = handlers.unshown_report(outcome, "the server said so")
    assert "It used 208 s" in visible(local)
    for word in ("quota", "billed", "reserv"):
        assert word not in visible(local).lower(), f"a local run's card says {word!r}"
    assert detail(local) == "the server said so"

    monkeypatch.setattr(handlers, "_run_device", lambda result: ("cuda", True))
    space = visible(handlers.unshown_report(outcome, "the server said so"))
    assert "It used 208 s" in space and "which is billed" in space


def test_the_report_without_a_server_message_says_none_came():
    facts = render.RunFacts(
        mode=modes.get_mode(MODE),
        seed=1,
        returned_size=(704, 480),
        seconds=2.0,
        stage_2=False,
        loras=(),
        took=40.0,
        took_is_gpu=True,
        device="cuda",
        on_zerogpu=True,
        quota_multiplier=2,
    )
    card = render.result_not_shown(facts, None)
    assert "40 s of GPU time, about 80 s of your quota, which is billed" in visible(card)
    assert detail(card) == render.NO_SERVER_MESSAGE
    assert detail(render.lost_run()) == render.NO_SERVER_MESSAGE


def test_a_run_whose_own_report_is_a_failure_keeps_that_report(tmp_path):
    failed = handlers.RunOutcome(
        video=None,
        audio=None,
        status=render.failure(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")),
        mode_id=MODE,
        error=RuntimeError("oom"),
    )
    assert not handlers.has_result(failed)
    assert handlers.unshown_report(failed, "a later server error") == failed.status
    assert handlers.has_result(_finished(tmp_path))
    assert visible(handlers.unshown_report(None, "gone")).startswith(
        "### The run ended without a result"
    )
