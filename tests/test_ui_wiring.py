"""The built Blocks: contracts, privacy and the Generate chain (AC-19, AC-32, R11, R12, R13).

Built headlessly with real Gradio: no server, no GPU, no weights.
"""

from __future__ import annotations

import pytest

from ltx25_studio import modes
from ltx25_studio.ui import handlers
from ltx25_studio.ui.fields import FIELD_KEYS, field_by_key

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


@pytest.fixture(scope="module")
def demo():
    from ltx25_studio.ui import build_ui

    return build_ui()


def by_elem_id(demo, elem_id):
    return next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == elem_id)


def dependency(demo, api_name):
    return next(f for f in demo.fns.values() if f.api_name == api_name)


def test_only_the_two_published_endpoints_are_public(demo):
    public = sorted(f.api_name for f in demo.fns.values() if f.api_visibility != "private")
    assert public == ["on_generate", "on_validate_lora"]
    assert sorted(demo.get_api_info()["named_endpoints"]) == ["/on_generate", "/on_validate_lora"]


def test_on_generate_keeps_its_signature_and_returns(demo):
    endpoint = demo.get_api_info()["named_endpoints"]["/on_generate"]
    params = endpoint["parameters"]
    api_keys = [k for k in FIELD_KEYS if field_by_key(k).kind != "state"]
    assert params[0]["component"] == "Radio"
    assert len(params) == 1 + len(api_keys)
    names = {"text": "textbox"}
    for index, key in enumerate(api_keys, start=1):
        kind = field_by_key(key).kind
        assert params[index]["component"].lower() == names.get(kind, kind), key
    assert [r["component"] for r in endpoint["returns"]] == ["Video", "Audio", "Markdown"]


def test_on_validate_lora_keeps_its_inputs_and_outputs(demo):
    fn = dependency(demo, "on_validate_lora")
    assert [type(c).__name__ for c in fn.inputs] == ["Textbox", "File", "State"]
    assert [type(c).__name__ for c in fn.outputs] == ["Markdown", "State", "Markdown"]


def test_text_to_video_with_sound_is_selected_on_first_load(demo):
    radio = by_elem_id(demo, "mode-matrix")
    assert radio.value == "text_to_audio_video"
    assert [value for _label, value in radio.choices] == list(modes.MODES)
    labels = dict((value, label) for label, value in radio.choices)
    assert all(labels[m] == modes.get_mode(m).label for m in modes.MODES), (
        "each cell's accessible name is the mode's full label"
    )


def test_the_hidden_api_slots_render_no_visible_control(demo):
    fn = dependency(demo, "on_generate")
    slots = dict(zip(["mode", *FIELD_KEYS], fn.inputs, strict=True))
    for key in ("negative_prompt", "conditions", "noise_scale"):
        assert slots[key].visible is False, f"{key} must keep its slot but show nothing"
    assert not any(
        isinstance(b, gr.Dataframe) and b.visible is not False for b in demo.blocks.values()
    ), "there is no shot-list table anywhere"


def test_uploads_are_pruned_from_gradios_cache(demo):
    assert demo.delete_cache is not None
    frequency, age = demo.delete_cache
    assert 0 < frequency <= age


def test_generate_is_disabled_by_the_first_step_and_restored_by_a_final_one(demo):
    button = by_elem_id(demo, "generate-btn")
    run = dependency(demo, "on_generate")
    start = demo.fns[run.trigger_after]
    finish = next(
        f for f in demo.fns.values() if f.trigger_after == run._id and button in f.outputs
    )
    assert button in start.outputs and start.api_visibility == "private"
    assert not finish.trigger_only_on_success and not finish.trigger_only_on_failure
    # Gradio 6.24 runs `.then` listeners only when the step completes; a step
    # Gradio itself fails fires `.failure` listeners alone. Both must bring
    # Generate back.
    on_failure = [
        f for f in demo.fns.values()
        if f.trigger_after == run._id and f.trigger_only_on_failure and button in f.outputs
    ]
    assert on_failure, "nothing re-enables Generate when Gradio fails the generation step"
    assert on_failure[0].fn is finish.fn
    assert run.show_progress_on and by_elem_id(demo, "video-out") in run.show_progress_on

    start_values = start.fn("text_to_audio_video", False)
    first = start_values[start.outputs.index(button)]
    assert first["interactive"] is False


@pytest.mark.parametrize("succeeded", [True, False])
def test_the_final_step_brings_generate_back_on_success_and_on_failure(demo, succeeded, tmp_path):
    from conftest import inputs_for

    from ltx25_studio.ui import blocks

    button = by_elem_id(demo, "generate-btn")
    run = dependency(demo, "on_generate")
    finish = next(
        f for f in demo.fns.values() if f.trigger_after == run._id and button in f.outputs
    )

    class Request:
        session_hash = f"test-{succeeded}"

    values = {k: field_by_key(k).default for k in FIELD_KEYS} | inputs_for("text_to_audio_video")
    if succeeded:
        video = tmp_path / "output.mp4"
        video.write_bytes(b"x")

        class Result:
            seed = 5
            warnings = ()
            run_dir = tmp_path

        outcome = handlers.RunOutcome(
            video=str(video),
            audio=str(video),
            status="ok",
            mode_id="text_to_audio_video",
            params=values,
            result=Result(),
        )
    else:
        outcome = handlers.RunOutcome(
            video=None,
            audio=None,
            status="### failed",
            mode_id="text_to_audio_video",
            params=values,
            error=RuntimeError("boom"),
        )
    blocks._remember(Request(), outcome)
    out = finish.fn(Request(), "text_to_audio_video", [], *(values[k] for k in FIELD_KEYS))
    update = out[finish.outputs.index(button)]
    assert update["value"] == "Generate"
    assert update["interactive"] is True, "a ready form gets Generate back"
    audio_out = by_elem_id(demo, "audio-out")
    video_out = by_elem_id(demo, "video-out")
    if succeeded:
        assert out[finish.outputs.index(video_out)]["visible"] is True
        assert out[finish.outputs.index(audio_out)]["visible"] is False, (
            "a video with sound is one player, not a video player and an audio player"
        )


def test_every_ui_only_event_is_private(demo):
    for fn in demo.fns.values():
        if fn.api_name not in ("on_generate", "on_validate_lora"):
            assert fn.api_visibility == "private", fn.api_name
