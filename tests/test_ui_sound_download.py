"""A video + sound result offers its sound track as a quiet download (#6, F30).

Every video + sound run also writes its sound as its own WAV, and the public endpoint returns
it, but the page offered no way to save the sound alone. Now the Done card's action row, beside
Reuse seed, offers "Download the sound (WAV)" when the run returned both files, and only then:
not for a sound-only or a silent run, not for one file that carries both tracks.

The runs go through ``handlers.run`` and the real ``generate()`` with the production writer (only
the pipeline is a fake), then the page's real finish step.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from conftest import inputs_for
from test_encoder_contract import FRAME_SIZE, VocoderPipe
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder
from test_ui_form import Form

from ltx25_studio import generate as generate_module
from ltx25_studio.ui import blocks, handlers, render, showcase
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


class Request:
    session_hash = "sound-download"


@pytest.fixture
def form(tmp_path, monkeypatch):
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    return Form(blocks.build_ui())


def wire(monkeypatch, root: Path, *, encoder=None, pipe=None) -> None:
    real = generate_module.generate
    monkeypatch.setattr(
        generate_module,
        "generate",
        lambda **params: real(
            pipeline_factory=factory_returning(pipe or VocoderPipe()),
            encoder=encoder,
            output_root=root,
            retention_bytes=10**9,
            **params,
        ),
    )


def finished(form: Form, mode_id: str) -> handlers.RunOutcome:
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id)
    values |= {"width": FRAME_SIZE, "height": FRAME_SIZE, "num_frames": 9}
    outcome = handlers.run(mode_id, *(values[k] for k in FIELD_KEYS))
    assert outcome.error is None, outcome.status
    form.values[form.by_id("run-status")._id] = outcome.status
    blocks._remember(Request(), outcome)
    form.run(form.fn("_finish"), request=Request())
    return outcome


def offered(form: Form) -> str | None:
    """The sound download's file while it is shown, else None."""
    sound = form.by_id("sound-download")
    return str(form.get(sound)) if form.visible(sound) and form.get(sound) else None


def test_a_video_with_sound_offers_its_wav_beside_reuse_seed(form, tmp_path, monkeypatch):
    wire(monkeypatch, tmp_path / "runs")
    outcome = finished(form, "text_to_audio_video")
    assert Path(outcome.video).suffix == ".mp4" and Path(outcome.audio).suffix == ".wav"
    assert offered(form) == str(outcome.audio)
    sound = form.by_id("sound-download")
    assert isinstance(sound, gr.DownloadButton) and sound.label == render.SOUND_DOWNLOAD
    assert render.SOUND_DOWNLOAD == "Download the sound (WAV)"
    assert "mr-quiet" in (form.props[sound._id].get("elem_classes") or ())
    row = form.by_id("report-actions")
    assert sound in row.children and form.by_id("report-actions").children[0].value == "Reuse seed"


@pytest.mark.parametrize("mode_id", ["text_to_audio", "text_to_video"])
def test_a_sound_only_or_a_silent_run_offers_no_separate_sound(form, tmp_path, monkeypatch, mode_id):
    wire(monkeypatch, tmp_path / "runs")
    finished(form, "text_to_audio_video")
    assert offered(form)
    outcome = finished(form, mode_id)
    assert (outcome.video is None) or (outcome.audio is None)
    assert offered(form) is None


def test_one_file_carrying_both_tracks_is_not_offered_twice(form, tmp_path, monkeypatch):
    wire(monkeypatch, tmp_path / "runs", encoder=one_muxed_file_encoder, pipe=FakePipe())
    outcome = finished(form, "text_to_audio_video")
    assert outcome.video == outcome.audio
    assert offered(form) is None


def test_a_new_run_and_a_failure_take_it_away(form, tmp_path, monkeypatch):
    wire(monkeypatch, tmp_path / "runs")
    finished(form, "text_to_audio_video")
    start = form.fn("_start")
    said = start.fn("text_to_audio_video", False, "a harbour", Request())
    out = dict(zip(start.outputs, said, strict=True))
    assert out[form.by_id("sound-download")]["visible"] is False
    failed = handlers.RunOutcome(
        video=None,
        audio=None,
        status=render.failure(RuntimeError("boom")),
        mode_id="text_to_audio_video",
        error=RuntimeError("boom"),
    )
    blocks._remember(Request(), failed)
    form.run(form.fn("_finish"), request=Request())
    assert offered(form) is None


def test_the_wav_reaches_the_page_through_gradios_own_processing(tmp_path, monkeypatch):
    from gradio.state_holder import SessionState

    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    monkeypatch.chdir(tmp_path)
    wire(monkeypatch, tmp_path / "runs")
    demo = blocks.build_ui()
    finish = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "_finish")
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_audio_video")
    values |= {"width": FRAME_SIZE, "height": FRAME_SIZE, "num_frames": 9}
    outcome = handlers.run("text_to_audio_video", *(values[k] for k in FIELD_KEYS))
    blocks._remember(gr.Request(session_hash="wav"), outcome)
    sent = ["text_to_audio_video", None, render.LENGTH_MATCH]
    for component, key in zip(finish.inputs[3:], FIELD_KEYS, strict=True):
        if component.stateful:
            sent.append(None)
            continue
        value = component.postprocess(values[key])
        sent.append(value.model_dump() if hasattr(value, "model_dump") else value)
    out = asyncio.run(
        demo.process_api(
            block_fn=finish,
            inputs=sent,
            state=SessionState(demo),
            request=gr.Request(session_hash="wav"),
            session_hash="wav",
            in_event_listener=True,
        )
    )["data"]
    got = {getattr(c, "elem_id", None): v for c, v in zip(finish.outputs, out, strict=True)}
    sound = got["sound-download"]
    assert sound["visible"] is True and sound["value"]["orig_name"] == "output.wav"
    assert Path(sound["value"]["path"]).read_bytes() == Path(outcome.audio).read_bytes()
