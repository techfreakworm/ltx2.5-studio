"""The cost panel states exact figures, from the modules that compute them.

* the reservation is ``zerogpu.estimate_duration`` for the same request --
  stage 2, Enhance and staged LoRAs included -- and each extra is named with
  its seconds (B1, AC-07);
* "needs N s free" is the reservation times the quota multiplier (AC-08);
* a request the backend would refuse shows "won't run", both problems with
  their nearest legal values, no figure, and Generate stays off with the same
  words the server would use (B3, AC-09, readiness);
* the sound-only modes say they pay for a thrown-away video (AC-11).
"""

from __future__ import annotations

import re

import pytest
from conftest import image_input, inputs_for

from ltx25_studio import generate as generate_module
from ltx25_studio import modes, zerogpu
from ltx25_studio.errors import MissingInputError
from ltx25_studio.ui import handlers, render
from ltx25_studio.ui.fields import FIELDS

MODE_IDS = tuple(modes.MODES)


def form(mode_id: str, **overrides) -> dict:
    values = {spec.key: spec.default for spec in FIELDS}
    values |= inputs_for(mode_id)
    values |= overrides
    return values


@pytest.fixture
def base_60(monkeypatch):
    """The verdict's arithmetic is at a 60 s base; pin the config to it."""
    monkeypatch.setenv("LTX25_GPU_DURATION_SECONDS", "60")
    assert zerogpu.estimate_duration() == 60


@pytest.fixture
def staged_lora(tmp_path):
    """A real (sparse) adapter: staged files are read before dispatch, so a placeholder
    would be refused rather than costed."""
    from lora_fixtures import good_adapter

    path = good_adapter(tmp_path)
    return [{"source": "org/adapter", "path": str(path), "weight": 0.8, "content_key": "k"}]


# --------------------------------------------------------------------------
# AC-07: the reservation ladder, each extra named
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("upscale", "enhance", "lora", "expected", "named"),
    [
        (False, False, False, 60, []),
        (True, False, False, 90, [("Upscale ×2", 30)]),
        (True, True, False, 270, [("Upscale ×2", 30), ("Enhance", 180)]),
        (True, True, True, 330, [("Upscale ×2", 30), ("LoRA", 60), ("Enhance", 180)]),
        (False, False, True, 120, [("LoRA", 60)]),
    ],
)
def test_the_reservation_is_the_estimators_with_each_extra_named(
    base_60, staged_lora, upscale, enhance, lora, expected, named
):
    values = form(
        "image_text_to_video",
        enable_stage_2=upscale,
        enhance_prompt=enhance,
        loras=staged_lora if lora else [],
    )
    facts = handlers.review("image_text_to_video", values)

    assert facts.state == "ready", facts.reason
    assert facts.reserve == expected
    prepared, _spec, _notes = generate_module.prepare_request(
        generate_module.GenerationRequest(mode="image_text_to_video", **values)
    )
    assert facts.reserve == zerogpu.estimate_duration(request=prepared), (
        "the panel and the live dispatch's estimator disagree"
    )
    assert list(facts.extras) == named
    panel = render.cost_panel(facts)
    for label, seconds in named:
        assert f"{seconds} s for {label}" in panel, panel


def test_each_lever_refreshes_the_panel(base_60, staged_lora):
    """The panel is a function of every lever; flipping one changes its words."""
    plain = render.cost_panel(handlers.review("image_text_to_video", form("image_text_to_video")))
    for lever in ({"enable_stage_2": True}, {"enhance_prompt": True}, {"loras": staged_lora}):
        changed = render.cost_panel(
            handlers.review("image_text_to_video", form("image_text_to_video", **lever))
        )
        assert changed != plain, f"{lever} did not move the cost panel"


def test_upscale_states_the_returned_size(base_60):
    facts = handlers.review("image_text_to_video", form("image_text_to_video", enable_stage_2=True))
    assert facts.returned_size == (1408, 960)
    assert "1408" in render.cost_panel(facts) and "upscaled ×2" in render.cost_panel(facts)


# --------------------------------------------------------------------------
# AC-08: every mode, in words
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_every_mode_states_size_length_return_reservation_quota_and_billing(mode_id):
    facts = handlers.review(mode_id, form(mode_id))
    assert facts.state == "ready", facts.reason
    panel = re.sub(r"<[^>]+>", "", render.cost_panel(facts))
    for words in (
        "Renders",
        "Returns",
        "Reserves",
        "of GPU time",
        "of quota free to start",
        "GPU time used × ",
        "runs that fail on the GPU are billed too",
        "LoRA",
    ):
        assert words in panel, f"{mode_id}: {words!r} missing from the cost panel:\n{panel}"
    assert f"{facts.reserve * handlers.quota_multiplier()} s of quota free" in panel


def test_the_quota_needed_is_the_reservation_times_the_multiplier(base_60):
    facts = handlers.review("text_to_audio_video", form("text_to_audio_video"))
    assert facts.multiplier == handlers.quota_multiplier() == 2
    assert "120 s of quota free to start" in re.sub(r"<[^>]+>", "", render.cost_panel(facts))


@pytest.mark.parametrize("mode_id", ["text_to_audio", "video_to_audio", "audio_to_audio"])
def test_the_sound_only_modes_say_the_video_is_rendered_and_thrown_away(mode_id):
    facts = handlers.review(mode_id, form(mode_id))
    panel = render.cost_panel(facts)
    assert facts.discards_video and facts.render_size == (256, 256)
    assert "256 × 256 video is rendered and thrown away; you pay for" in panel
    assert "sound only" in panel


def test_audio_to_video_names_the_audio_encode():
    facts = handlers.review("audio_to_video", form("audio_to_video"))
    assert facts.encodes_audio
    assert "The file carries your own audio." in render.cost_panel(facts)


# --------------------------------------------------------------------------
# AC-09 and readiness: refused requests
# --------------------------------------------------------------------------


def test_an_illegal_size_and_length_wont_run_and_shows_no_figure():
    values = form("text_to_video", width=700, height=256, num_frames=120)
    facts = handlers.review("text_to_video", values)

    assert facts.state == "refused"
    assert facts.reserve is None
    problems = " ".join(facts.problems)
    assert "Width 700" in problems and "704" in problems
    assert "120 frames" in problems and "113" in problems and "121" in problems
    panel = render.cost_panel(facts)
    assert "Won’t run" in panel and "Reserves" not in panel and "quota free" not in panel
    reason = render.readiness_reason(facts)
    assert "Width 700" in reason and "120 frames" in reason


def test_a_missing_input_disables_generate_with_the_servers_own_words():
    values = form("image_to_video", image=None)
    facts = handlers.review("image_to_video", values)
    with pytest.raises(MissingInputError) as refused:
        generate_module.prepare_request(
            generate_module.GenerationRequest(mode="image_to_video", **values)
        )
    assert facts.state == "incomplete"
    assert render.readiness_reason(facts) == str(refused.value)
    assert facts.reserve is not None, "an incomplete form still shows what the run will cost"


def test_enhance_with_an_empty_prompt_is_refused_and_adds_nothing():
    values = form("video_to_video", enhance_prompt=True, prompt="")
    facts = handlers.review("video_to_video", values)
    assert facts.state == "refused" and facts.reserve is None
    assert "Enhance is on but the prompt is empty" in render.readiness_reason(facts)


def test_a_ready_request_has_no_reason_line():
    facts = handlers.review("image_to_video", form("image_to_video", image=image_input()))
    assert facts.state == "ready" and render.readiness_reason(facts) == ""


def test_a_running_request_keeps_generate_off_with_a_reason():
    facts = handlers.review("image_to_video", form("image_to_video"))
    assert "in progress" in render.readiness_reason(facts, running=True)


# --------------------------------------------------------------------------
# AC-10: refused before any GPU dispatch, through /on_generate
# --------------------------------------------------------------------------


def test_an_illegal_api_request_is_refused_before_any_dispatch(monkeypatch):
    dispatched = []
    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", lambda **kw: dispatched.append(kw))
    called = []
    real = generate_module.generate
    monkeypatch.setattr(generate_module, "generate", lambda **kw: called.append(kw) or real(**kw))
    from ltx25_studio.ui.fields import FIELD_KEYS

    values = form("text_to_video", width=700, num_frames=120)
    video, audio, status = handlers.on_generate("text_to_video", *(values[k] for k in FIELD_KEYS))

    assert (video, audio) == (None, None)
    assert status.startswith("### "), "a refusal renders the loud panel"
    assert "Nothing ran. No GPU time was used." in status
    assert dispatched == [] and called == [], "the GPU-decorated function must never be called"


def test_the_decoder_choice_does_not_move_the_cost_panel():
    """The reservation does not depend on the decoder, so the panel must not say it does."""
    base = render.cost_panel(handlers.review("text_to_video", form("text_to_video")))
    for choice in ("diffusion", "conv"):
        assert (
            render.cost_panel(
                handlers.review("text_to_video", form("text_to_video", decoder=choice))
            )
            == base
        )
