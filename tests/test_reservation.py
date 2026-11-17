"""The GPU reservation follows each run's size and length (issue #21).

Every Space run used to reserve 180 s whatever it was (x1.5 with Upscale, +60 s with an
adapter, +180 s with Enhance), and ZeroGPU wants twice the reservation free on xlarge: a
visitor with less than 360 s of quota could not start a 25-second default run, and nothing
stopped a run too big to finish inside its reservation. The estimate is now
``PRE + steps x t(N1) + [Upscale: UPSAMPLE + 3 x t(N2)] + post + [Enhance] + [adapter]``
from the Space's measurements, times a 1.3 margin, rounded up to 5 s, never below 45 s;
and on the Space a request past the measured envelope is refused before any GPU time.

The measured windows below are the Space log's schedule -> release, in seconds: the six
deploy-2 measurement runs (M1-M6, 2026-11-17, mode 05, seed 7) and the 24 deploy-1 calls
(2026-11-16). Each run goes through the real gate: ``generate()`` prepares the request (the
length a recording or a clip sets, the pinned size of a sound-only mode) and dispatches it,
and the estimator is called with the dispatch's own keyword arguments, as ``spaces`` calls it.
"""

from __future__ import annotations

import logging
import math
import re
import sys
import types
from dataclasses import replace

import pytest
from conftest import audio_file, image_input, inputs_for, video_file
from test_user_words import offences

from ltx25_studio import config, constraints, preflight, zerogpu
from ltx25_studio import device as device_module
from ltx25_studio import generate as generate_module
from ltx25_studio.ui import handlers, render
from ltx25_studio.ui.fields import FIELDS

GenerationRequest = generate_module.GenerationRequest

T2V = "text_to_video"
T2AV = "text_to_audio_video"
PROMPT = "a weathered sea captain at the wheel in a storm"

#: The operator's levers, by the names the Space's settings use.
OLD_VARIABLE = "LTX25_GPU_DURATION_SECONDS"
FLOOR_VARIABLE = "LTX25_GPU_DURATION_FLOOR_SECONDS"
#: The margin over every measured window, and the smallest reservation.
MARGIN = 1.3
SMALLEST = 45


@pytest.fixture
def on_the_space(monkeypatch):
    """This process as the Space runs it: the platform package is imported (``app.py`` does it
    first), so ``device.current()`` answers ZeroGPU, CUDA."""
    module = types.ModuleType("spaces")
    module.GPU = lambda *a, **k: lambda f: f
    monkeypatch.setitem(sys.modules, "spaces", module)
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)
    monkeypatch.delenv(OLD_VARIABLE, raising=False)
    monkeypatch.delenv(FLOOR_VARIABLE, raising=False)
    assert device_module.current().on_zerogpu


@pytest.fixture
def dispatched(monkeypatch):
    """The GPU entrypoint, recording the reservation ``spaces`` would ask for: the estimator is
    called with exactly the keyword arguments the dispatch receives."""
    reserved: list[int] = []

    def dispatch(**kwargs):
        reserved.append(zerogpu.estimate_duration(**kwargs))
        return "dispatched"

    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
    return reserved


def staged_adapter(tmp_path) -> list[dict]:
    """A real (sparse) adapter, staged as the page stages one."""
    from lora_fixtures import good_adapter

    return [{"source": "org/adapter", "path": str(good_adapter(tmp_path)), "weight": 1.0}]


def reserve_for(dispatched, tmp_path, **fields) -> int:
    """The reservation for one request, through the gate and the dispatch."""
    fields = {key: value(tmp_path) if callable(value) else value for key, value in fields.items()}
    assert generate_module.generate(**fields) == "dispatched"
    return dispatched[-1]


# --------------------------------------------------------------------------
# every measured run fits its reservation with the 1.3 margin
# --------------------------------------------------------------------------

#: ``(label, measured window in seconds, the request as sent)``.
MEASURED = (
    # deploy 2 measurement runs (MEASUREMENTS.md), mode 05
    ("M1 704x480x121 Upscale", 52, {"mode": T2AV, "num_frames": 121, "enable_stage_2": True}),
    ("M2 704x480x481", 65, {"mode": T2AV, "num_frames": 481}),
    ("M3 1408x960x121", 69, {"mode": T2AV, "width": 1408, "height": 960}),
    ("M4 1280x704x241", 88, {"mode": T2AV, "width": 1280, "height": 704, "num_frames": 241}),
    ("M5 704x480x241 Upscale", 100, {"mode": T2AV, "num_frames": 241, "enable_stage_2": True}),
    (
        "M6 960x544x241 Upscale",
        154,
        {"mode": T2AV, "width": 960, "height": 544, "num_frames": 241, "enable_stage_2": True},
    ),
    # deploy 1, the 24 calls (timings.md)
    ("m01", 32.3, {"mode": T2V}),
    ("m04 clip 121 frames", 17.0, {"mode": "video_to_video", "video": lambda _: video_file(121)}),
    ("m05", 24.0, {"mode": T2AV}),
    (
        "m10 recording 4.73 s",
        22.2,
        {"mode": "audio_to_audio", "audio": lambda _: audio_file(4.73), "auto_duration": True},
    ),
    (
        "m11 first frame 480x704",
        24.7,
        {
            "mode": "audio_to_video",
            "width": 480,
            "height": 704,
            "image": lambda _: image_input(),
            "audio": lambda _: audio_file(4.72),
            "auto_duration": True,
        },
    ),
    ("x2 m05 49 frames", 33.4, {"mode": T2AV, "num_frames": 49, "enable_stage_2": True}),
    ("dec m01 fast", 19.6, {"mode": T2V, "decoder": "conv"}),
    ("x1 m05 49 frames", 37.4, {"mode": T2AV, "num_frames": 49}),
    (
        "len m05 512x960 model picks 5-10 s",
        28.2,
        {
            "mode": T2AV,
            "width": 512,
            "height": 960,
            "auto_duration": True,
            "min_seconds": 5.0,
            "max_seconds": 10.0,
        },
    ),
    ("enh m05", 51.6, {"mode": T2AV, "enhance_prompt": True}),
    (
        "op m11 512x960 recording 193 frames",
        36.5,
        {
            "mode": "audio_to_video",
            "width": 512,
            "height": 960,
            "audio": lambda _: audio_file(193 / 24 + 0.01),
            "auto_duration": True,
        },
    ),
    ("lora m01", 36.4, {"mode": T2V, "loras": staged_adapter}),
    ("m02", 21.9, {"mode": "image_to_video", "image": lambda _: image_input()}),
    ("m03", 17.8, {"mode": "image_text_to_video", "image": lambda _: image_input()}),
    ("m06", 23.7, {"mode": "image_to_audio_video", "image": lambda _: image_input()}),
    ("m07", 28.2, {"mode": "image_text_to_audio_video", "image": lambda _: image_input()}),
    ("m08", 13.5, {"mode": "text_to_audio"}),
    ("m09", 22.4, {"mode": "video_to_audio", "video": lambda _: video_file(121)}),
    (
        "m11 recording 4.72 s",
        23.1,
        {"mode": "audio_to_video", "audio": lambda _: audio_file(4.72), "auto_duration": True},
    ),
    (
        "x2 m11 49 frames",
        29.4,
        {
            "mode": "audio_to_video",
            "audio": lambda _: audio_file(2.05),
            "num_frames": 49,
            "enable_stage_2": True,
        },
    ),
    (
        "m11 seed 21",
        21.7,
        {"mode": "audio_to_video", "audio": lambda _: audio_file(4.72), "auto_duration": True},
    ),
    ("m09 seed 19", 10.5, {"mode": "video_to_audio", "video": lambda _: video_file(121)}),
    ("m05 seed 12", 22.9, {"mode": T2AV}),
    ("lora twin, no adapter", 22.2, {"mode": T2V}),
)


@pytest.mark.parametrize(("label", "window", "fields"), MEASURED, ids=[row[0] for row in MEASURED])
def test_every_measured_space_run_is_reserved_the_margin_over_its_window(
    label, window, fields, on_the_space, dispatched, tmp_path
):
    reserved = reserve_for(dispatched, tmp_path, prompt=PROMPT, **fields)

    assert reserved >= MARGIN * window, (
        f"{label}: reserves {reserved} s for a run measured at {window} s; the margin is x{MARGIN}"
    )


def test_the_measured_points_are_the_ones_the_constants_name():
    """The documented fit, read back: a change to a coefficient must be a change to this table."""
    assert zerogpu.RESERVATION_MARGIN == MARGIN
    assert zerogpu.MIN_RESERVATION_SECONDS == SMALLEST
    assert zerogpu.PRE_SECONDS == 25 and zerogpu.UPSAMPLE_SECONDS == 3
    assert zerogpu.step_seconds(5_280) == pytest.approx(1.47, abs=0.01), "M1 stage 1: 1.49 s/step"
    assert zerogpu.step_seconds(21_120) == pytest.approx(5.15, abs=0.01), "M3: 5.14 s/step"
    assert zerogpu.step_seconds(40_920) == pytest.approx(11.11, abs=0.01), "M5 stage 2: 11.12"
    assert zerogpu.step_seconds(63_240) == pytest.approx(19.67, abs=0.01), "M6 stage 2: 19.96"


# --------------------------------------------------------------------------
# the default run, and what each extra adds
# --------------------------------------------------------------------------


def form(mode_id: str, **overrides) -> dict:
    values = {spec.key: spec.default for spec in FIELDS}
    values |= inputs_for(mode_id)
    values |= overrides
    return values


def test_the_default_run_needs_less_than_200_s_of_quota_to_start(
    on_the_space, dispatched, tmp_path
):
    """Mode 05 at the page's defaults (704 x 480, 121 frames, no extras)."""
    reserved = reserve_for(dispatched, tmp_path, mode=T2AV, prompt=PROMPT)

    assert reserved <= 95
    assert reserved * zerogpu.quota_multiplier("xlarge") < 200
    facts = handlers.review(T2AV, form(T2AV))
    assert facts.state == "ready", facts.reason
    assert facts.reserve == reserved, "the panel and the dispatch disagree on the default run"
    panel = re.sub(r"<[^>]+>", "", render.cost_panel(facts)).replace("\u00a0", " ")
    assert f"Reserves{reserved} s of GPU time" in panel
    assert f"Needs{reserved * 2} s of quota free to start" in panel


@pytest.mark.parametrize(
    ("width", "height", "frames"),
    [(256, 256, 9), (704, 480, 121), (1280, 704, 241), (960, 544, 241)],
)
def test_enhance_and_an_adapter_each_add_their_stated_seconds_at_any_size(width, height, frames):
    request = GenerationRequest(
        mode=T2AV, prompt=PROMPT, width=width, height=height, num_frames=frames
    )
    plain = zerogpu.estimate_duration(request=request)

    enhanced = zerogpu.estimate_duration(request=replace(request, enhance_prompt=True))
    adapted = zerogpu.estimate_duration(request=replace(request, loras=({"source": "a"},)))
    two = zerogpu.estimate_duration(
        request=replace(request, loras=({"source": "a"}, {"source": "b"}))
    )
    both = zerogpu.estimate_duration(
        request=replace(request, enhance_prompt=True, loras=({"source": "a"},))
    )

    assert enhanced == plain + zerogpu.ENHANCE_DURATION_SECONDS
    assert adapted == two == plain + zerogpu.LORA_DURATION_SECONDS, (
        "one flat allowance for any adapters"
    )
    assert both == plain + zerogpu.ENHANCE_DURATION_SECONDS + zerogpu.LORA_DURATION_SECONDS


def test_the_stated_seconds_are_the_measured_extras_with_the_margin():
    """Enhance added 27 s of window and an adapter 5-15 s on the Space; the model allows 60 s and
    30 s, and the reservation adds those x1.3, rounded up to 5 s."""
    assert zerogpu.ENHANCE_WINDOW_SECONDS == 60 and zerogpu.LORA_WINDOW_SECONDS == 30
    assert zerogpu.ENHANCE_DURATION_SECONDS == 80 == 5 * math.ceil(60 * 1.3 / 5)
    assert zerogpu.LORA_DURATION_SECONDS == 40 == 5 * math.ceil(30 * 1.3 / 5)


def upscale_seconds(width, height, frames) -> int:
    request = GenerationRequest(mode=T2AV, width=width, height=height, num_frames=frames)
    return zerogpu.estimate_duration(
        request=replace(request, enable_stage_2=True)
    ) - zerogpu.estimate_duration(request=request)


def test_upscale_adds_more_for_a_larger_or_longer_video():
    default = upscale_seconds(704, 480, 121)
    assert default == 45
    assert upscale_seconds(704, 480, 241) > default, "longer"
    assert upscale_seconds(960, 544, 241) > upscale_seconds(704, 480, 241), "larger"
    assert upscale_seconds(704, 480, 49) < default, "shorter"


def test_the_reservation_grows_with_size_and_length():
    def reserve(width, height, frames):
        return zerogpu.estimate_duration(
            request=GenerationRequest(mode=T2AV, width=width, height=height, num_frames=frames)
        )

    assert reserve(704, 480, 121) == 60
    assert reserve(704, 480, 241) > reserve(704, 480, 121) < reserve(1408, 960, 121)
    assert reserve(704, 480, 481) > reserve(704, 480, 241)
    assert reserve(64, 64, 9) == SMALLEST, "never below 45 s"


@pytest.mark.parametrize(
    ("overrides", "upscale"),
    [
        ({"enable_stage_2": True}, 45),
        ({"width": 1280, "height": 704, "num_frames": 241}, None),
        ({"width": 960, "height": 544, "num_frames": 241, "enable_stage_2": True}, 145),
    ],
    ids=["704x480x121 Upscale", "1280x704x241", "960x544x241 Upscale"],
)
def test_the_cost_panel_reserves_what_the_dispatch_reserves_for_this_size(
    overrides, upscale, on_the_space, dispatched, tmp_path
):
    """The panel's Reserves and its Upscale seconds are this request's, not the default run's."""
    values = form(T2AV, **overrides)
    facts = handlers.review(T2AV, values)
    dispatched_reserve = reserve_for(dispatched, tmp_path, mode=T2AV, **values)

    assert facts.state == "ready", facts.reason
    assert facts.reserve == dispatched_reserve
    assert dict(facts.extras).get("Upscale ×2") == upscale
    assert facts.base + sum(seconds for _label, seconds in facts.extras) == facts.reserve


# --------------------------------------------------------------------------
# the request is read as the run will use it
# --------------------------------------------------------------------------


def test_a_sound_only_mode_is_reserved_at_its_pinned_size_and_never_upscaled():
    pinned = zerogpu.estimate_duration(request=GenerationRequest(mode="text_to_audio"))
    asked = GenerationRequest(mode="text_to_audio", width=1920, height=1088, enable_stage_2=True)

    assert zerogpu.estimate_duration(request=asked) == pinned
    assert pinned == zerogpu.estimate_duration(
        request=GenerationRequest(mode=T2V, width=256, height=256)
    )


def test_a_model_picked_length_is_reserved_at_its_longest_end():
    picked = GenerationRequest(mode=T2AV, auto_duration=True, min_seconds=5.0, max_seconds=10.0)
    _shortest, longest = constraints.duration_head_frame_range(
        min_seconds=5.0, max_seconds=10.0, frame_rate=24.0
    )

    assert longest == 233
    assert zerogpu.estimate_duration(request=picked) == zerogpu.estimate_duration(
        request=GenerationRequest(mode=T2AV, num_frames=longest)
    )


def test_a_recording_sets_the_length_it_is_reserved_for(on_the_space, dispatched, tmp_path):
    """Match my audio: the gate measures the recording, and the reservation follows it."""
    short = reserve_for(
        dispatched, tmp_path, mode="audio_to_video", audio=audio_file(2.05), auto_duration=True
    )
    long = reserve_for(
        dispatched, tmp_path, mode="audio_to_video", audio=audio_file(15.0), auto_duration=True
    )

    assert short == zerogpu.estimate_duration(
        request=GenerationRequest(mode="audio_to_video", num_frames=49)
    )
    assert long == zerogpu.estimate_duration(
        request=GenerationRequest(mode="audio_to_video", num_frames=held(15.0))
    )
    assert long > short


def held(seconds: float) -> int:
    return constraints.largest_num_frames_within(seconds, frame_rate=24.0)


def test_a_clip_sets_the_length_it_is_reserved_for(on_the_space, dispatched, tmp_path):
    """Video to video takes its length from the clip, whatever the frame count says."""
    reserved = reserve_for(
        dispatched, tmp_path, mode="video_to_video", video=video_file(241), num_frames=9
    )
    measured = zerogpu.estimate_duration(
        request=GenerationRequest(
            mode="video_to_video", num_frames=241, extras={"length_source": "video"}
        )
    )

    assert reserved == measured
    assert reserved > reserve_for(
        dispatched, tmp_path, mode="video_to_video", video=video_file(49), num_frames=241
    )


def test_a_length_still_to_come_from_an_input_is_reserved_at_the_longest_it_can_be():
    """Before the recording or the clip is there, the run may be as long as the app allows."""
    longest = zerogpu.estimate_duration(
        request=GenerationRequest(mode="audio_to_video", num_frames=constraints.MAX_NUM_FRAMES)
    )

    assert (
        zerogpu.estimate_duration(
            request=GenerationRequest(mode="audio_to_video", auto_duration=True)
        )
        == longest
    )
    assert zerogpu.estimate_duration(
        request=GenerationRequest(mode="video_to_video", num_frames=121)
    ) == zerogpu.estimate_duration(
        request=GenerationRequest(
            mode="video_to_video",
            num_frames=constraints.MAX_NUM_FRAMES,
            extras={"length_source": "video"},
        )
    )


@pytest.mark.parametrize(("level", "steps"), [(0.421875, 1), (0.725, 2), (0.909375, 3)])
def test_a_clip_variation_is_reserved_for_the_steps_it_runs(level, steps):
    """A variation denoises only the tail of the schedule from its level."""

    def reserve(mode, **fields):
        return zerogpu.estimate_duration(
            request=GenerationRequest(
                mode=mode,
                width=1280,
                height=704,
                num_frames=241,
                extras={"length_source": "video"},
                **fields,
            )
        )

    varied = reserve("video_to_video", noise_scale=level)
    full = reserve(T2V)
    shape = zerogpu.run_shape(
        request=GenerationRequest(
            mode="video_to_video",
            width=1280,
            height=704,
            num_frames=241,
            noise_scale=level,
            extras={"length_source": "video"},
        )
    )

    assert shape.steps == steps
    assert varied < full


# --------------------------------------------------------------------------
# beyond the measured envelope: refused before any GPU time, on the Space only
# --------------------------------------------------------------------------


def refusal(**fields) -> str:
    with pytest.raises(constraints.BeyondEnvelopeError) as refused:
        generate_module.prepare_request(GenerationRequest(prompt=PROMPT, **fields))
    return str(refused.value)


def runs(**fields) -> None:
    generate_module.prepare_request(GenerationRequest(prompt=PROMPT, **fields))


def test_the_envelope_limits_are_inclusive_and_the_next_value_is_past_them():
    assert constraints.within_envelope(tokens=64_000, pixel_frames=520_000_000)
    assert not constraints.within_envelope(tokens=64_001, pixel_frames=1)
    assert not constraints.within_envelope(tokens=1, pixel_frames=520_000_001)


def test_one_pass_at_the_token_limit_runs_and_the_next_length_is_refused(on_the_space):
    # 50 x 40 x 32 = 64,000 tokens exactly; the next legal length is 33 latent frames.
    runs(mode=T2V, width=1600, height=1280, num_frames=249)

    message = refusal(mode=T2V, width=1600, height=1280, num_frames=257)

    assert message == (
        "This size and length is more than this Space has been measured to finish. At "
        "1600 × 1280, up to 10.4 s fits. Make it smaller or shorter."
    )


def test_one_pass_past_the_token_limit_is_refused_below_the_pixel_limit(on_the_space):
    """59 x 35 x 31 = 64,015 tokens, 509,608,960 pixel-frames: the token limit alone refuses."""
    assert "At 1888 × 1120, up to 9.7 s fits" in refusal(
        mode=T2V, width=1888, height=1120, num_frames=241
    )


def test_upscale_at_its_token_limit_runs_and_one_step_wider_is_refused(on_the_space):
    # The second pass: 80 x 40 x 20 = 64,000 tokens; 82 x 40 x 20 = 65,600 (513,884,160 pixel-frames).
    runs(mode=T2V, width=1280, height=640, num_frames=153, enable_stage_2=True)

    message = refusal(mode=T2V, width=1312, height=640, num_frames=153, enable_stage_2=True)

    assert message == (
        "This size and length is more than this Space has been measured to finish. At "
        "1312 × 640 with Upscale ×2, up to 6.0 s fits (20.0 s without Upscale ×2). Make it "
        "smaller or shorter."
    )
    assert "with Upscale ×2" in refusal(
        mode=T2V, width=1280, height=640, num_frames=161, enable_stage_2=True
    )


def test_the_pixel_limit_refuses_the_first_length_past_it(on_the_space, monkeypatch):
    """Within 481 frames the token limits always bind first (pixel-frames per token stay below
    520,000,000 / 64,000), so the pixel limit is checked here with the token limit lifted."""
    monkeypatch.setattr(constraints, "ENVELOPE_MAX_TOKENS", 10**9)
    runs(mode=T2V, width=1920, height=1120, num_frames=241)  # 518,246,400 pixel-frames

    assert "At 1920 × 1120, up to 10.0 s fits" in refusal(
        mode=T2V, width=1920, height=1120, num_frames=249
    )


def test_a_model_picked_length_is_checked_at_its_longest_end(on_the_space):
    message = refusal(
        mode=T2AV, width=1920, height=1088, auto_duration=True, min_seconds=5.0, max_seconds=20.0
    )
    assert "At 1920 × 1088, up to 10.0 s fits" in message


def test_the_measured_runs_are_inside_the_envelope(on_the_space):
    runs(mode=T2AV, width=960, height=544, num_frames=241, enable_stage_2=True)  # M6
    runs(mode=T2AV, width=1280, height=704, num_frames=241)  # M4
    runs(mode=T2AV, width=704, height=480, num_frames=481)  # M2
    runs(mode=T2AV, width=1920, height=1088, num_frames=241)


def test_the_refusal_is_in_the_pages_words():
    sentences = [
        str(
            constraints.BeyondEnvelopeError(
                width=1920, height=1088, frame_rate=24.0, longest=241, longest_without_upscale=None
            )
        ),
        str(
            constraints.BeyondEnvelopeError(
                width=1920, height=1088, frame_rate=24.0, longest=49, longest_without_upscale=241
            )
        ),
    ]
    for sentence in sentences:
        assert offences(sentence) == [], sentence


def test_the_refusal_survives_the_trip_back_from_the_gpu_worker():
    import pickle

    refused = constraints.BeyondEnvelopeError(
        width=1312, height=640, frame_rate=24.0, longest=145, longest_without_upscale=481
    )
    back = pickle.loads(pickle.dumps(refused))

    assert type(back) is constraints.BeyondEnvelopeError and str(back) == str(refused)
    assert (back.longest, back.longest_without_upscale) == (145, 481)


def test_generate_refuses_it_before_any_gpu_time(on_the_space, dispatched):
    with pytest.raises(constraints.BeyondEnvelopeError) as refused:
        generate_module.generate(mode=T2V, prompt=PROMPT, width=1920, height=1088, num_frames=481)

    assert dispatched == [], "the GPU entrypoint must never be called"
    assert refused.value.gpu_used is False


def test_the_panel_says_it_wont_run_in_the_same_words(on_the_space):
    values = form(T2V, width=1600, height=1280, num_frames=257)
    facts = handlers.review(T2V, values)
    with pytest.raises(constraints.BeyondEnvelopeError) as refused:
        generate_module.prepare_request(GenerationRequest(mode=T2V, **values))

    assert facts.state == "refused" and facts.reserve is None
    panel = render.cost_panel(facts)
    assert "Won’t run" in panel and "Reserves" not in panel
    assert str(refused.value) in panel.replace("&#x27;", "'")
    assert render.readiness_reason(facts) == str(refused.value)


def test_off_the_space_the_envelope_is_not_applied(monkeypatch):
    monkeypatch.delitem(sys.modules, "spaces", raising=False)
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: True)
    assert not device_module.current().on_zerogpu

    runs(mode=T2V, width=1920, height=1088, num_frames=481)


# --------------------------------------------------------------------------
# the old lever, and the new floor
# --------------------------------------------------------------------------

REQUESTS = (
    GenerationRequest(mode=T2AV),
    GenerationRequest(mode=T2AV, width=960, height=544, num_frames=241, enable_stage_2=True),
    GenerationRequest(mode="text_to_audio", enhance_prompt=True, prompt=PROMPT),
)


def test_the_variables_keep_the_names_the_space_settings_use():
    assert config.ENV_GPU_DURATION_SECONDS == OLD_VARIABLE
    assert config.ENV_GPU_DURATION_FLOOR_SECONDS == FLOOR_VARIABLE


def test_the_old_duration_variable_changes_no_reservation(monkeypatch):
    monkeypatch.delenv(OLD_VARIABLE, raising=False)
    monkeypatch.delenv(FLOOR_VARIABLE, raising=False)
    before = [zerogpu.estimate_duration(request=request) for request in REQUESTS]

    for value in ("180", "60", "600"):
        monkeypatch.setenv(OLD_VARIABLE, value)
        assert [zerogpu.estimate_duration(request=request) for request in REQUESTS] == before


def startup_log(caplog) -> list[str]:
    module = types.ModuleType("spaces")
    module.GPU = lambda *a, **k: lambda f: f
    runtime_config = config.load_config()
    resolution = device_module.resolve(module, runtime_config)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ltx25_studio"):
        preflight.run(runtime_config, strict=False, spaces_module=module, resolution=resolution)
    return "\n".join(record.getMessage() for record in caplog.records).splitlines()


def test_the_old_duration_variable_is_logged_once_at_start_up_as_ignored(monkeypatch, caplog):
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: pytest.fail("CUDA probed on ZeroGPU"))
    monkeypatch.delenv(FLOOR_VARIABLE, raising=False)
    monkeypatch.setenv(OLD_VARIABLE, "180")

    named = [line for line in startup_log(caplog) if OLD_VARIABLE + "=" in line]

    assert len(named) == 1, named
    assert "180" in named[0] and "ignored" in named[0] and "size and length" in named[0]
    assert FLOOR_VARIABLE in named[0], "the line names the lever that is left"

    monkeypatch.delenv(OLD_VARIABLE)
    assert not [line for line in startup_log(caplog) if OLD_VARIABLE in line]


def test_the_floor_variable_raises_every_reservation_to_at_least_it(monkeypatch):
    monkeypatch.delenv(FLOOR_VARIABLE, raising=False)
    unfloored = [zerogpu.estimate_duration(request=request) for request in REQUESTS]
    assert unfloored == [60, 240, 125]

    monkeypatch.setenv(FLOOR_VARIABLE, "150")

    assert [zerogpu.estimate_duration(request=request) for request in REQUESTS] == [150, 240, 150]
    assert config.load_config().gpu_duration_floor_seconds == 150


def test_the_floor_is_unset_by_default(monkeypatch):
    monkeypatch.delenv(FLOOR_VARIABLE, raising=False)
    assert config.load_config().gpu_duration_floor_seconds is None


@pytest.mark.parametrize("value", ["0", "-5", "ninety", "1.5"])
def test_a_floor_that_is_not_a_whole_number_of_seconds_is_refused_at_start_up(monkeypatch, value):
    monkeypatch.setenv(FLOOR_VARIABLE, value)

    with pytest.raises(config.ConfigError) as refused:
        config.load_config()
    assert FLOOR_VARIABLE in str(refused.value)
