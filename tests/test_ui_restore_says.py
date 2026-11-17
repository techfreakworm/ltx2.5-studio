"""Restore settings says what it did, the seed included (review round 4, minor 10).

Restore writes the run's own settings back, its seed setting among them (usually -1, a new
seed each run), and keeps the current media. On a desktop nothing moved and nothing said so:
the frame before and after a Restore was the same, and "Reuse seed" then "Restore settings" on
one row silently went back to a random seed. Now a quiet line in Recent runs says what was
restored, the seed in words, and which of the current media stay; the next run takes it away.

Driven through the built page's own steps (:class:`test_ui_form.Form`).
"""

from __future__ import annotations

import pytest
from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM, inputs_for
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder
from test_ui_failure_card import finished
from test_ui_form import Form

from ltx25_studio import generate as generate_module
from ltx25_studio.ui import blocks, handlers, render, showcase
from ltx25_studio.ui.fields import FIELD_KEYS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


@pytest.fixture
def form(tmp_path, monkeypatch):
    from ltx25_studio import audio as audio_module

    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    # The recording loads without torch (the loader alone is stubbed, as elsewhere).
    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )
    real = generate_module.generate
    monkeypatch.setattr(
        generate_module,
        "generate",
        lambda **params: real(
            pipeline_factory=factory_returning(FakePipe()),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path / "runs",
            retention_bytes=10**9,
            **params,
        ),
    )
    return Form(blocks.build_ui())


def run_in(form: Form, mode_id: str, **overrides) -> handlers.RunOutcome:
    form.pick_mode(mode_id)
    values = dict(zip(FIELD_KEYS, form.form_values(), strict=True)) | inputs_for(mode_id)
    values |= overrides
    outcome = handlers.run(mode_id, *(values[k] for k in FIELD_KEYS))
    assert outcome.error is None, outcome.status
    finished(form, outcome)
    return outcome


def row_buttons(form: Form, label: str) -> list:
    return [b for b in form.fn("_finish").outputs if isinstance(b, gr.Button) and b.value == label]


def note(form: Form) -> str | None:
    line = form.by_id("recent-note")
    return form.get(line) if form.visible(line) else None


def test_restore_says_the_seed_is_random_again_and_which_media_stay(form):
    run_in(form, "text_to_audio_video")
    assert note(form) is None
    form.click(row_buttons(form, "Reuse seed")[0])
    assert form.field("seed") >= 0
    form.click(row_buttons(form, "Restore settings")[0])
    assert form.field("seed") == -1, "the run's own seed setting is written back"
    said = note(form)
    assert said is not None and said.startswith("Settings from ")
    assert "the seed is random again, as it was for that run" in said
    assert "stays" not in said, "a mode with no media keeps none"


def test_restore_of_a_fixed_seed_names_it_and_the_media_that_stay(form):
    run_in(form, "audio_to_video", seed=42)
    form.click(row_buttons(form, "Restore settings")[0])
    said = note(form)
    assert said.endswith(
        "restored, with that run's seed, 42. Your current audio and image stay as they are."
    )
    assert form.field("seed") == 42


def test_the_next_run_takes_the_line_away(form):
    run_in(form, "text_to_audio_video")
    form.click(row_buttons(form, "Restore settings")[0])
    assert note(form)
    start = form.fn("_start")
    out = dict(
        zip(start.outputs, start.fn("text_to_audio_video", False, "a tram", None), strict=True)
    )
    assert out[form.by_id("recent-note")]["visible"] is False


def test_the_words():
    assert render.restored_line(at="08:06", seed=None, media=()) == (
        "Settings from 08:06 restored; the seed is random again, as it was for that run."
    )
    assert render.restored_line(at="08:06", seed=7, media=("image",)) == (
        "Settings from 08:06 restored, with that run's seed, 7. Your current image stays as it is."
    )
