"""A run consumes only what its mode declares (F01, F04).

The UI keeps the value of a field it hides when the mode changes, and
``/on_generate`` forwards every ``FIELD_KEYS`` value. So a request routinely
carries a stale image, video, audio file, prompt or enhance flag from a mode the
user has left. These specs drive the REAL handler into the REAL ``generate()``,
through a dispatch shaped like ``app.run_generation`` that records the GPU
reservation, down to a fake pipe -- and require that a value the mode does not
take changes nothing about the run except one note saying it was ignored.

What "takes" means is read off the registry: ``inputs`` (required) plus
``optional_inputs``. Nothing here names a mode in a rule.
"""

from __future__ import annotations

import pytest
from conftest import (
    INPUT_PROMPT,
    audio_file,
    image_input,
    inputs_for,
    video_file,
)
from test_exports_contract import MODE_IDS, FakePipe, factory_returning, one_muxed_file_encoder

from ltx25_studio import generate as generate_module
from ltx25_studio import modes, zerogpu
from ltx25_studio.ui import handlers
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS


def accepted_kinds(spec) -> set[str]:
    return set(spec.inputs) | set(getattr(spec, "optional_inputs", ()))


#: One non-default value per field the UI can leave behind in a hidden widget.
def stale_values() -> dict:
    return {
        "prompt": "STALE PROMPT",
        "negative_prompt": "STALE NEGATIVE",
        "image": image_input(),
        "video": video_file(),
        "audio": audio_file(),
        "conditions": [["a stale shot", 9, 0.5]],
        "strength": 0.3,
        "noise_scale": 0.9,
        "enhance_prompt": True,
    }


#: Which input kind each field belongs to. ``conditions`` hang on visual media.
FIELD_KINDS = {
    "prompt": {"text"},
    "negative_prompt": {"text"},
    "enhance_prompt": {"text"},
    "image": {"image"},
    "video": {"video"},
    "audio": {"audio"},
    "conditions": {"image", "video"},
    "strength": {"video"},
    "noise_scale": {"audio"},
}

#: Fields whose drop must be announced: the user put something there.
ANNOUNCED = {"prompt", "negative_prompt", "image", "video", "audio"}


def _disallowed_pairs():
    pairs = []
    for mode_id in MODE_IDS:
        accepted = accepted_kinds(modes.get_mode(mode_id))
        for field_name, kinds in FIELD_KINDS.items():
            if not kinds & accepted:
                pairs.append((mode_id, field_name))
    return pairs


DISALLOWED = _disallowed_pairs()


def values_for(mode_id: str, **overrides) -> tuple:
    """Positional values in ``FIELD_KEYS`` order, as the browser sends them."""
    values = {spec.key: spec.default for spec in FIELDS}
    values["seed"] = 7
    values.update(inputs_for(mode_id))
    values.update(overrides)
    return tuple(values[key] for key in FIELD_KEYS)


class Run:
    def __init__(self):
        self.pipe = FakePipe()
        self.encoder_kwargs: dict = {}
        self.reservations: list[int] = []
        self.enhancer_calls: list[tuple] = []
        self.result = None
        self.status = ""

    @property
    def sent(self) -> dict:
        assert self.pipe.calls, f"the pipeline never ran. Status:\n{self.status}"
        return self.pipe.calls[-1]


@pytest.fixture
def run(monkeypatch, tmp_path, audio_input_ready):
    """Real handler -> real generate -> recording dispatch -> FakePipe."""
    real = generate_module.generate
    counter = iter(range(1000))

    def go(mode_id: str, **overrides) -> Run:
        record = Run()

        def dispatch(**kwargs):
            # Shaped like app.run_generation: the estimator sees exactly the
            # kwargs the decorated function is called with.
            record.reservations.append(zerogpu.estimate_duration(**kwargs))
            token = generate_module._IN_GPU_WORKER.set(True)
            try:
                return real(**kwargs)
            finally:
                generate_module._IN_GPU_WORKER.reset(token)

        def encoder(output, **kwargs):
            record.encoder_kwargs.update(kwargs)
            return one_muxed_file_encoder(output, **kwargs)

        def enhancer(prompt, image=None):
            record.enhancer_calls.append((prompt, image))
            return "ENHANCED " + prompt

        def wired(**params):
            record.result = real(
                pipeline_factory=factory_returning(record.pipe),
                encoder=encoder,
                enhancer=enhancer,
                output_root=tmp_path / f"run-{next(counter)}",
                retention_bytes=10**9,
                **params,
            )
            return record.result

        monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
        monkeypatch.setattr(generate_module, "generate", wired)
        try:
            _video, _audio, record.status = handlers.on_generate(
                mode_id, *values_for(mode_id, **overrides)
            )
        finally:
            monkeypatch.setattr(generate_module, "generate", real)
        return record

    return go


def normalised(kwargs: dict) -> dict:
    """The kwargs, with per-run objects reduced to what they mean."""
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


def condition_kinds(kwargs: dict) -> list[str]:
    return [c.kind for c in kwargs.get("conditions", ())]


# --------------------------------------------------------------------------
# the property: a value the mode does not take changes nothing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode_id", "field_name"), DISALLOWED, ids=[f"{m}-{f}" for m, f in DISALLOWED]
)
def test_a_value_the_mode_does_not_take_cannot_change_the_run(mode_id, field_name, run):
    base = run(mode_id)
    leak = run(mode_id, **{field_name: stale_values()[field_name]})

    assert base.result is not None, base.status
    assert leak.result is not None, (
        f"{mode_id} refused a run over a {field_name!r} it does not take, which the user "
        f"cannot see or clear:\n{leak.status}"
    )
    assert normalised(leak.sent) == normalised(base.sent), (
        f"a stale {field_name!r} changed what {mode_id} sent to the pipeline"
    )
    assert len(leak.pipe.audio_vae.encoded) == len(base.pipe.audio_vae.encoded), (
        f"a stale {field_name!r} changed the audio encodes in {mode_id}"
    )
    assert ("supplied_audio" in leak.encoder_kwargs) == ("supplied_audio" in base.encoder_kwargs)
    assert leak.enhancer_calls == base.enhancer_calls, (
        f"a stale {field_name!r} changed what the enhancer was asked in {mode_id}"
    )
    assert leak.reservations == base.reservations, (
        f"a stale {field_name!r} changed the GPU reservation for {mode_id}: "
        f"{base.reservations} -> {leak.reservations}"
    )

    new_notes = [note for note in leak.result.warnings if note not in base.result.warnings]
    if field_name in ANNOUNCED:
        assert len(new_notes) == 1, (
            f"dropping a stale {field_name!r} in {mode_id} must be said exactly once: {new_notes}"
        )
        label = modes.get_mode(mode_id).label
        assert label in new_notes[0], f"the note does not name the mode: {new_notes[0]!r}"
    else:
        assert len(new_notes) <= 1, new_notes


# --------------------------------------------------------------------------
# the five regressions, by name
# --------------------------------------------------------------------------


def test_text_to_video_never_encodes_a_stale_audio_upload(run):
    leak = run("text_to_video", audio=audio_file())

    assert not leak.pipe.audio_vae.encoded, "a hidden upload was encoded into a text_to_video run"
    assert "audio_latents" not in leak.sent


def test_text_to_audio_video_never_muxes_a_stale_upload_as_yours(run):
    leak = run("text_to_audio_video", audio=audio_file())

    assert leak.result is not None, leak.status
    assert "supplied_audio" not in leak.encoder_kwargs, (
        "a hidden upload replaced the model's soundtrack in text_to_audio_video"
    )
    assert not any("audio YOU supplied" in note for note in leak.result.warnings), (
        f"the run told the user their own audio is in the file: {leak.result.warnings}"
    )


def test_audio_to_video_never_conditions_on_a_stale_video(run):
    leak = run("audio_to_video", video=video_file())

    assert "video" not in condition_kinds(leak.sent), (
        f"a hidden video became a condition in audio_to_video: {condition_kinds(leak.sent)}"
    )


def test_a_stale_enhance_flag_neither_reserves_nor_runs_the_enhancer(run):
    base = run("image_to_video")
    leak = run("image_to_video", enhance_prompt=True, prompt="STALE PROMPT")

    assert leak.reservations == base.reservations, (
        f"a hidden enhance flag changed the reservation: {base.reservations} -> {leak.reservations}"
    )
    assert leak.enhancer_calls == [], "image_to_video takes no prompt and the enhancer ran anyway"


def test_the_enhancer_never_sees_a_stale_image(run):
    leak = run("text_to_video", enhance_prompt=True, image=image_input())

    assert len(leak.enhancer_calls) == 1, leak.status
    prompt, image = leak.enhancer_calls[0]
    assert prompt == INPUT_PROMPT
    assert image is None, "text_to_video takes no image, and the enhancer was shown one"


# --------------------------------------------------------------------------
# optional inputs (F04)
# --------------------------------------------------------------------------


def test_video_to_video_sends_a_prompt_when_one_is_given(run):
    leak = run("video_to_video", prompt="slow dolly in, golden hour")
    assert leak.sent["prompt"] == "slow dolly in, golden hour"


def test_video_to_video_sends_an_empty_prompt_when_none_is_given(run):
    base = run("video_to_video")
    assert base.sent["prompt"] == ""


@pytest.mark.parametrize(
    "mode_id", ["video_to_video", "video_to_audio", "audio_to_audio", "audio_to_video"]
)
def test_the_modes_with_an_optional_prompt_accept_one(mode_id):
    spec = modes.get_mode(mode_id)
    assert "text" in getattr(spec, "optional_inputs", ()), (
        f"{mode_id} has always sent a prompt; it must be declared optional so the UI can show it"
    )


def test_audio_to_video_takes_an_optional_first_frame_image(run):
    leak = run("audio_to_video", image=image_input())

    conditions = leak.sent.get("conditions", ())
    assert [(c.kind, c.index, c.strength) for c in conditions] == [("image", 0, 1.0)], (
        f"an image supplied to audio_to_video must become exactly one first-frame condition: "
        f"{[(c.kind, c.index, c.strength) for c in conditions]}"
    )


@pytest.mark.parametrize("mode_id", ["image_to_video", "image_to_audio_video"])
def test_the_image_only_modes_stay_promptless(mode_id):
    spec = modes.get_mode(mode_id)
    assert "text" not in accepted_kinds(spec), (
        f"{mode_id} is 'one condition, empty prompt'; it must not accept a prompt"
    )


# --------------------------------------------------------------------------
# the scoping function itself
# --------------------------------------------------------------------------


def scope(request, spec):
    return generate_module.scope_request_to_mode(request, spec)


def test_scoping_leaves_defaults_unannounced():
    """A field the user never touched is not news."""
    spec = modes.get_mode("image_to_video")
    request = generate_module.GenerationRequest(mode=spec.id, image=image_input())

    scoped, notes = scope(request, spec)

    assert notes == ()
    assert scoped.image is request.image


def test_scoping_is_idempotent():
    """The worker re-runs it on the request the web side already scoped."""
    spec = modes.get_mode("text_to_video")
    request = generate_module.GenerationRequest(
        mode=spec.id, prompt="p", image=image_input(), audio=audio_file(), strength=0.2
    )

    once, notes = scope(request, spec)
    twice, again = scope(once, spec)

    assert notes and again == ()
    assert twice == once


def test_scoping_switches_off_stage_2_where_the_registry_skips_it():
    """Otherwise the estimator reserves 1.5x for a pass that never runs."""
    spec = modes.get_mode("text_to_audio")
    request = generate_module.GenerationRequest(mode=spec.id, prompt="p", enable_stage_2=True)

    scoped, _notes = scope(request, spec)

    assert scoped.enable_stage_2 is False


def test_scoping_resets_strength_and_noise_scale_where_they_do_nothing():
    spec = modes.get_mode("text_to_video")
    request = generate_module.GenerationRequest(
        mode=spec.id, prompt="p", strength=0.2, noise_scale=0.9
    )

    scoped, _notes = scope(request, spec)
    defaults = generate_module.GenerationRequest(mode=spec.id)

    assert scoped.strength == defaults.strength
    assert scoped.noise_scale == defaults.noise_scale


def test_an_empty_shot_table_normalises_to_no_conditions():
    spec = modes.get_mode("image_to_video")
    for empty in ([], (), {"headers": ["a", "b", "c"], "data": []}, [["", None, None]]):
        request = generate_module.GenerationRequest(
            mode=spec.id, image=image_input(), conditions=empty
        )
        scoped, notes = scope(request, spec)
        assert scoped.conditions == (), empty
        assert notes == (), empty


def test_a_registry_helper_says_what_a_mode_accepts_and_requires():
    """The UI renders from this rather than re-deriving it."""
    spec = modes.get_mode("audio_to_video")
    assert modes.accepts(spec, "audio") and modes.requires(spec, "audio")
    assert modes.accepts(spec, "text") and not modes.requires(spec, "text")
    assert modes.accepts(spec, "image") and not modes.requires(spec, "image")
    assert not modes.accepts(spec, "video")
