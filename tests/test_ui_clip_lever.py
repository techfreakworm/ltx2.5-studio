"""Video to video's lever: how much to change the clip (review round 2, §6(a); ruling 7).

The mode is a variation of the user's clip (``modes`` ``video_conditioning="variation"``):
the clip is noised to a level and denoised along the schedule's tail. The UI offers the
backend's own three levels as "A little / Moderately / A lot" (default "A lot", the only
level at which the prompt acts), writing ``noise_scale`` through a strict zip as Audio to
sound does. The strength slider is gone from the page (its API slot stays), the length is
always the clip's, the report names the choice, and the cost panel does not move.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder
from test_ui_form import Form

from ltx25_studio import audio_conditioning, modes
from ltx25_studio import generate as generate_module
from ltx25_studio.ui import handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

CLIP_MODES = [m for m, s in modes.MODES.items() if getattr(s, "video_conditioning", None)]


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    import av
    import numpy as np

    path = tmp_path_factory.mktemp("clip") / "two-seconds.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for index in range(48):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), index * 5 % 255, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


@pytest.fixture
def form():
    from ltx25_studio.ui import build_ui

    return Form(build_ui())


@pytest.fixture
def pipes(monkeypatch, tmp_path):
    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )
    real = generate_module.generate
    made = []

    def generate(**params):
        pipe = FakePipe()
        made.append(pipe)
        return real(
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path / str(len(made)),
            retention_bytes=10**9,
            **params,
        )

    monkeypatch.setattr(generate_module, "generate", generate)
    return made


def test_the_registry_has_a_clip_variation_for_this_lever_to_drive():
    assert CLIP_MODES == ["video_to_video"]


def test_the_choices_are_the_backends_own_levels_and_a_lot_is_the_default():
    levels = [level for _name, level in render.CLIP_CHOICES]
    assert levels == list(audio_conditioning.VIDEO_VARIATION_LEVELS)
    assert dict(render.CLIP_CHOICES)[render.CLIP_DEFAULT] == audio_conditioning.DEFAULT_VIDEO_LEVEL
    assert [name for name, _ in render.CLIP_CHOICES] == ["A little", "Moderately", "A lot"]


@pytest.mark.parametrize("mode_id", CLIP_MODES)
def test_the_mode_shows_the_change_choice_and_none_of_the_levers_it_does_not_read(form, mode_id):
    form.pick_mode(mode_id)
    choice = form.by_id("lever-clip")
    assert form.visible(choice) and form.get(choice) == render.CLIP_DEFAULT
    assert form.props[choice._id]["label"] == "How much to change your clip"
    assert form.props[choice._id]["info"] == (
        "The prompt steers the result only at “A lot”; at the lower settings your clip is "
        "re-rendered almost as it is."
    )
    assert not form.visible(form.fields["strength"]), "strength is not read by a variation"
    assert not form.visible(form.by_id("lever-noise"))
    for length_control in (form.by_id("length-source"), form.by_id("seconds")):
        assert not form.visible(length_control), "the length is always the clip's"
    assert not form.visible(form.fields["num_frames"])
    assert "strength" in FIELD_KEYS, "the API slot stays"


@pytest.mark.parametrize("mode_id", CLIP_MODES)
def test_each_choice_changes_what_reaches_the_pipeline(form, pipes, clip, mode_id):
    form.pick_mode(mode_id)
    form.user(form.fields["video"], clip)
    reached = {}
    for name, level in render.CLIP_CHOICES:
        form.user(form.by_id("lever-clip"), name)
        outcome = handlers.run(mode_id, *form.form_values())
        assert outcome.error is None, outcome.status
        sent = pipes[-1].calls[-1]
        variation = sent.get("video_variation")  # modes.VariationClip: the clip and its level
        assert variation is not None, (name, sorted(sent))
        reached[name] = variation.level
        assert reached[name] == pytest.approx(level), name
        facts = handlers.run_facts(outcome)
        shown = " ".join(render.run_summary(facts).replace("<", " <").split())
        assert f"{name} (" in shown, "the report names the choice"
    assert len(set(reached.values())) == 3


def test_the_choice_does_not_move_the_cost_panel(form, clip):
    form.pick_mode("video_to_video")
    form.user(form.fields["video"], clip)
    panels = set()
    for name, _level in render.CLIP_CHOICES:
        form.user(form.by_id("lever-clip"), name)
        values = dict(zip(FIELD_KEYS, form.form_values(), strict=True))
        panels.add(render.cost_panel(handlers.review("video_to_video", values)))
    assert len(panels) == 1


def test_the_clip_and_the_sound_keep_their_own_choices(form):
    form.pick_mode("audio_to_audio")
    form.user(form.by_id("lever-noise"), "Ignore my audio (new sound)")
    assert form.field("noise_scale") == 1.0
    form.pick_mode("video_to_video")
    assert form.get(form.by_id("lever-clip")) == render.CLIP_DEFAULT
    assert form.field("noise_scale") == dict(render.CLIP_CHOICES)[render.CLIP_DEFAULT]
    form.user(form.by_id("lever-clip"), "A little")
    form.pick_mode("audio_to_audio")
    assert form.field("noise_scale") == 1.0, "the sound's own choice came back"


def test_restore_brings_a_clip_runs_choice_back(form, pipes, clip):
    from ltx25_studio.ui import blocks

    form.pick_mode("video_to_video")
    form.user(form.fields["video"], clip)
    form.user(form.by_id("lever-clip"), "Moderately")
    outcome = handlers.run("video_to_video", *form.form_values())
    entry = blocks._recent_entry(
        "video_to_video", outcome.result, outcome.params, handlers.run_facts(outcome), outcome
    )
    form.user(form.by_id("lever-clip"), "A lot")
    form.pick_mode("text_to_video")
    form.values[form.fn("_finish").inputs[1]._id] = [entry]
    restore = next(
        b
        for b in form.demo.blocks.values()
        if isinstance(b, gr.Button) and b.value == "Restore settings"
    )
    form.click(restore)
    assert form.get(form.radio) == "video_to_video"
    assert form.get(form.by_id("lever-clip")) == "Moderately"
    assert form.field("noise_scale") == dict(render.CLIP_CHOICES)["Moderately"]
