"""What each mode shows, and that what is hidden never reaches a run.

* A field is shown exactly when the registry says the mode uses its kind
  (``modes.accepts``), and marked required exactly when it is required
  (``modes.requires``) -- the predicate ``generate.scope_request_to_mode``
  scopes requests with, so the two sides cannot drift (B10, AC-13b).
* Selecting a mode never writes a value into a field (B7).
* A value the selected mode does not use is parked, not submitted; switching
  back brings the user's value back (AC-13, AC-13b).
* The inert levers are gone (B6, AC-15).
"""

from __future__ import annotations

import pytest
from conftest import (
    STUB_SAMPLING_RATE,
    STUB_WAVEFORM,
    audio_file,
    image_input,
    inputs_for,
    video_file,
)
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

from ltx25_studio import generate as generate_module
from ltx25_studio import modes
from ltx25_studio.ui import handlers
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS, PARKED_FIELDS, TOGGLEABLE_FIELDS

MODE_IDS = tuple(modes.MODES)

#: Which input kind each user field carries.
KIND_OF = {
    "prompt": "text",
    "enhance_prompt": "text",
    "image": "image",
    "video": "video",
    "audio": "audio",
}


def defaults() -> dict:
    return {spec.key: spec.default for spec in FIELDS}


# --------------------------------------------------------------------------
# one acceptance predicate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_a_field_is_shown_exactly_when_the_registry_says_the_mode_uses_it(mode_id):
    spec = modes.get_mode(mode_id)
    view = handlers.mode_view(mode_id)
    for key, kind in KIND_OF.items():
        assert view.active[key] == modes.accepts(spec, kind), (
            f"{mode_id}: the {key!r} control is {'shown' if view.active[key] else 'hidden'} but "
            f"modes.accepts says {modes.accepts(spec, kind)}"
        )
    assert view.prompt_required == modes.requires(spec, "text")
    assert view.prompt_label == ("Prompt" if modes.requires(spec, "text") else "Prompt (optional)")


@pytest.mark.parametrize(
    "mode_id", ["video_to_video", "video_to_audio", "audio_to_audio", "audio_to_video"]
)
def test_the_modes_with_an_optional_prompt_show_one_marked_optional(mode_id):
    view = handlers.mode_view(mode_id)
    assert view.active["prompt"] and not view.prompt_required
    assert view.prompt_placeholder.startswith("Optional.")
    assert view.active["enhance_prompt"], "Enhance belongs beside every prompt box"


@pytest.mark.parametrize("mode_id", ["image_to_video", "image_to_audio_video"])
def test_the_image_only_modes_show_no_prompt_and_no_enhance(mode_id):
    view = handlers.mode_view(mode_id)
    assert not view.active["prompt"]
    assert not view.active["enhance_prompt"]


# --------------------------------------------------------------------------
# no inert levers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_the_negative_prompt_and_the_shot_list_are_shown_nowhere(mode_id):
    view = handlers.mode_view(mode_id)
    assert not view.active["negative_prompt"], "guidance is fixed at 1.0; it can never act"
    assert not view.active["conditions"], "a shot row with text is refused before every run"


def test_the_clip_strength_lever_is_offered_only_for_video_to_video():
    shown = {m for m in MODE_IDS if handlers.mode_view(m).active["strength"]}
    assert shown == {"video_to_video"}


def test_the_noise_lever_is_offered_only_for_audio_to_sound():
    shown = {m for m in MODE_IDS if handlers.mode_view(m).active["noise_scale"]}
    assert shown == {"audio_to_audio"}


def test_let_the_model_pick_appears_only_where_no_input_has_a_length():
    shown = {m for m in MODE_IDS if handlers.mode_view(m).active["auto_duration"]}
    assert shown == {
        "text_to_video",
        "image_to_video",
        "image_text_to_video",
        "text_to_audio_video",
        "image_to_audio_video",
        "image_text_to_audio_video",
        "text_to_audio",
    }
    timed = {m for m in MODE_IDS if handlers.mode_view(m).timed_input}
    assert timed == {"video_to_video", "video_to_audio", "audio_to_audio", "audio_to_video"}


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_pinned_modes_show_a_read_only_line_instead_of_size_controls(mode_id):
    spec = modes.get_mode(mode_id)
    view = handlers.mode_view(mode_id)
    if spec.min_resolution:
        assert view.pinned_line and "256 × 256" in view.pinned_line
        assert not view.active["width"] and not view.active["height"]
        assert not view.sizes_picture
    else:
        assert view.pinned_line is None
        assert view.active["width"] and view.active["height"]
    assert view.upscale == (not spec.skip_stage_2)
    assert view.active["enable_stage_2"] == (not spec.skip_stage_2)


# --------------------------------------------------------------------------
# selecting a mode never writes into a field
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_selecting_a_mode_writes_no_value_into_any_field(mode_id):
    _purpose, updates = handlers.on_mode_change(mode_id)
    assert len(updates) == len(TOGGLEABLE_FIELDS)
    assert all(update.value is None for update in updates), (
        f"{mode_id} wrote a value into a field; the user's own values must never be replaced"
    )


def test_leaving_a_pinned_mode_leaves_the_users_size_alone():
    values = defaults()
    after, _ = handlers.mode_switch("text_to_video", "audio_to_audio", values, {})
    after, _ = handlers.mode_switch("audio_to_audio", "image_text_to_video", after, {})
    assert (after["width"], after["height"]) == (704, 480)
    assert after["enable_stage_2"] is False


# --------------------------------------------------------------------------
# parking
# --------------------------------------------------------------------------


def test_a_prompt_is_parked_where_it_is_not_used_and_comes_back():
    values = defaults() | {"prompt": "a lighthouse in fog", "enhance_prompt": True}
    hidden, parked = handlers.mode_switch("text_to_audio_video", "image_to_video", values, {})
    assert hidden["prompt"] == "" and hidden["enhance_prompt"] is False
    assert parked == {"prompt": "a lighthouse in fog", "enhance_prompt": True}

    back, parked = handlers.mode_switch("image_to_video", "text_to_audio_video", hidden, parked)
    assert back["prompt"] == "a lighthouse in fog" and back["enhance_prompt"] is True
    assert parked == {}


def test_media_is_parked_and_restored_by_the_mode_that_takes_it():
    clip = video_file()
    values = defaults() | {"video": clip, "strength": 0.4}
    hidden, parked = handlers.mode_switch("video_to_video", "text_to_video", values, {})
    assert hidden["video"] is None and hidden["strength"] == 0.8
    back, _ = handlers.mode_switch("text_to_video", "video_to_video", hidden, parked)
    assert back["video"] == clip and back["strength"] == 0.4


def test_a_value_shared_by_both_modes_stays_put():
    image = image_input()
    values = defaults() | {"image": image, "prompt": "p"}
    after, parked = handlers.mode_switch(
        "image_text_to_video", "image_text_to_audio_video", values, {}
    )
    assert after["image"] is image and after["prompt"] == "p" and parked == {}


SWITCHES = [(a, b) for a in MODE_IDS for b in MODE_IDS if a != b]


@pytest.mark.parametrize(("previous", "mode_id"), SWITCHES)
def test_every_field_a_mode_does_not_use_is_back_at_its_default(previous, mode_id):
    stale = defaults() | {
        "prompt": "STALE",
        "negative_prompt": "STALE",
        "image": image_input(),
        "video": video_file(),
        "audio": audio_file(),
        "conditions": [["a shot", 9, 0.5]],
        "strength": 0.3,
        "noise_scale": 0.9,
        "enhance_prompt": True,
        "auto_duration": True,
    }
    after, _ = handlers.mode_switch(previous, mode_id, stale, {})
    view = handlers.mode_view(mode_id)
    for spec in PARKED_FIELDS:
        if not view.active[spec.key]:
            assert after[spec.key] == spec.default, (
                f"{previous} -> {mode_id}: hidden {spec.key!r} would be submitted as "
                f"{after[spec.key]!r}"
            )


# --------------------------------------------------------------------------
# AC-13: a stale form runs exactly like a clean one, through the real generate()
# --------------------------------------------------------------------------


@pytest.fixture
def capture(monkeypatch, tmp_path):
    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )
    real = generate_module.generate
    runs = []

    def go(mode_id, values):
        pipe = FakePipe()
        record = {"pipe": pipe, "encoder": {}, "result": None}

        def encoder(output, **kwargs):
            record["encoder"].update(kwargs)
            return one_muxed_file_encoder(output, **kwargs)

        def wired(**params):
            record["result"] = real(
                pipeline_factory=factory_returning(pipe),
                encoder=encoder,
                enhancer=lambda prompt, image=None: "ENHANCED " + prompt,
                output_root=tmp_path / f"run-{len(runs)}",
                retention_bytes=10**9,
                **params,
            )
            return record["result"]

        monkeypatch.setattr(generate_module, "generate", wired)
        try:
            record["status"] = handlers.on_generate(mode_id, *(values[k] for k in FIELD_KEYS))[2]
        finally:
            monkeypatch.setattr(generate_module, "generate", real)
        runs.append(record)
        return record

    return go


def _sent(record) -> dict:
    kwargs = dict(record["pipe"].calls[-1]) if record["pipe"].calls else {}
    out = {}
    for key, value in kwargs.items():
        if key in ("generator", "progress"):
            continue
        if key == "conditions":
            out[key] = tuple((c.kind, c.index, c.strength) for c in value)
        elif key == "audio_latents":
            out[key] = tuple(getattr(value, "shape", ()))
        else:
            out[key] = value
    return out


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_a_form_full_of_stale_values_runs_exactly_like_a_clean_one(mode_id, capture):
    required = inputs_for(mode_id)
    clean = defaults() | {"seed": 7} | required

    stale = defaults() | {
        "prompt": "STALE PROMPT",
        "negative_prompt": "STALE NEGATIVE",
        "image": image_input(),
        "video": video_file(),
        "audio": audio_file(),
        "conditions": [["a stale shot", 9, 0.5]],
        "strength": 0.3,
        "noise_scale": 0.9,
        "enhance_prompt": True,
        "seed": 7,
        "decoder": "conv",
    }
    # The user filled every field in some mode that takes them all, then picked
    # this one and supplied what it needs.
    previous = next(m for m in MODE_IDS if m != mode_id)
    after, _ = handlers.mode_switch(previous, mode_id, stale, {})
    view = handlers.mode_view(mode_id)
    for key in KIND_OF:
        if view.active[key]:
            # A field the mode shows is not stale: the user can see it. Give the
            # clean form the same visible value so only hidden values differ.
            after[key] = clean.get(key, after[key])
    after |= required
    if view.active["decoder"]:
        clean["decoder"] = after["decoder"]  # shown, so the user's choice is not stale

    base, leak = capture(mode_id, clean), capture(mode_id, after)
    assert leak["result"] is not None, leak["status"]
    assert _sent(leak) == _sent(base), f"a hidden value changed what {mode_id} sent"
    assert len(leak["pipe"].audio_vae.encoded) == len(base["pipe"].audio_vae.encoded)
    assert ("supplied_audio" in leak["encoder"]) == ("supplied_audio" in base["encoder"])
    assert not any(
        "was supplied, but" in note or "so it was ignored" in note
        for note in leak["result"].warnings
    ), (
        f"a UI-driven run of {mode_id} told the user about a value they cannot see: "
        f"{leak['result'].warnings}"
    )


def test_the_levers_follow_the_registrys_own_attributes():
    """Strength only where the mode leaves the clip's strength to the user; the
    change choice only where the mode varies the audio rather than holding it."""
    for mode_id in MODE_IDS:
        spec = modes.get_mode(mode_id)
        view = handlers.mode_view(mode_id)
        assert view.active["strength"] == (
            modes.accepts(spec, "video") and spec.video_strength is None
        )
        assert view.active["noise_scale"] == (spec.audio_conditioning == "variation")


def test_the_change_choices_write_the_levels_the_audio_module_supports():
    from ltx25_studio import audio_conditioning
    from ltx25_studio.ui import render

    levels = [level for _name, level in render.NOISE_CHOICES]
    assert levels == list(audio_conditioning.VARIATION_LEVELS)
    assert all(audio_conditioning.snap_level(level) == level for level in levels)


def test_the_decoder_choice_is_offered_only_where_a_picture_comes_back():
    for mode_id in MODE_IDS:
        spec = modes.get_mode(mode_id)
        assert handlers.mode_view(mode_id).active["decoder"] == ("video" in spec.exports)
    values = defaults() | {"decoder": "diffusion"}
    hidden, parked = handlers.mode_switch("text_to_video", "text_to_audio", values, {})
    assert hidden["decoder"] == "auto", "a sound-only run is sent with the default decoder"
    back, _ = handlers.mode_switch("text_to_audio", "text_to_video", hidden, parked)
    assert back["decoder"] == "diffusion"
