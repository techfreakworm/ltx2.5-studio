"""The failure cards and the cost words at the point of choice (review round 2: F3, F4, minors 5, 7, 8).

Categories reach the UI as strings (``errors.categorise``); a card is keyed on the string, so
the backend can add one without the UI waiting, and an unknown one gets the generic card.
"""

from __future__ import annotations

import re

import pytest
from conftest import inputs_for

from ltx25_studio import errors, modes, zerogpu
from ltx25_studio.ui import handlers, render
from ltx25_studio.ui.fields import FIELDS

MODE_IDS = tuple(modes.MODES)


def card(category: str, gpu_used: bool, message: str = "detail words") -> str:
    exc = errors.DecodeError(mode="text_to_video", reason=message)
    exc.category = category
    exc.gpu_used = gpu_used
    return render.failure(exc)


def visible(markup: str) -> str:
    markup = re.sub(r"<details(?![^>]*\bopen\b)[^>]*>.*?</details>", "", markup, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", markup).split())


ALL = [
    ("quota", False),
    ("limit", False),
    ("queue", False),
    ("timeout", True),
    ("timeout", False),
    ("oom", True),
    ("input", False),
    ("input", True),
    ("adapter", True),
    ("enhancer", True),
    ("bug", True),
    ("a-category-the-ui-has-never-seen", True),
]


@pytest.mark.parametrize(("category", "gpu_used"), ALL)
def test_every_card_has_a_headline_a_quota_line_one_step_and_the_detail(category, gpu_used):
    text = card(category, gpu_used)
    head, _, body = text.partition("\n")
    assert head.startswith("### ") and len(head) > 8
    quota_line = (
        "Any GPU time this run used is billed."
        if gpu_used
        else "Nothing ran. No GPU time was used."
    )
    assert quota_line in body
    assert "<summary>Detail</summary>" in text and "detail words" in text


@pytest.mark.parametrize(("category", "gpu_used"), ALL)
def test_no_card_contradicts_itself_or_sends_the_user_the_wrong_way(category, gpu_used):
    words = visible(card(category, gpu_used)).lower()
    assert not ("ran out of gpu time" in words and "nothing ran" in words), words
    if category in ("quota", "limit", "queue"):
        assert "smaller" not in words and "shorter" not in words, (
            f"the {category} card advises size or length, which changes no reservation: {words}"
        )


def test_the_quota_card_names_the_knobs_that_change_the_reservation():
    text = visible(card("quota", False))
    assert text.startswith("### Not enough GPU quota")
    for knob in ("Upscale", "Enhance", "LoRA", "cost panel", "daily quota"):
        assert knob in text, knob


def test_a_run_longer_than_the_platform_allows_has_its_own_card():
    text = visible(card("limit", False))
    assert "more GPU time than the platform allows in one run" in text
    assert "Turn off Enhance or Upscale to reserve less." in text


@pytest.mark.parametrize(
    "failure",
    [
        ("queue", False),
        ("timeout", False),  # a queue timeout, before the backend files it under "queue"
    ],
)
def test_a_queue_timeout_says_no_gpu_was_free_and_nothing_ran(failure):
    text = visible(card(*failure))
    assert text.startswith("### No GPU was free in time")
    assert (
        "Nothing ran. No GPU time was used." in text and "Press Generate again in a minute." in text
    )


def test_the_platforms_own_words_reach_the_right_card():
    quota = RuntimeError("You have exceeded your GPU quota (300s requested vs. 112s left).")
    assert visible(render.failure(quota)).startswith("### Not enough GPU quota")
    queued = RuntimeError("ZeroGPU queue timeout")
    assert visible(render.failure(queued)).startswith("### No GPU was free in time")
    illegal = RuntimeError("ZeroGPU illegal duration: 900s requested")
    found = errors.categorise(illegal).category
    expected = "This run asks for more GPU time" if found == "limit" else "Not enough GPU quota"
    assert visible(render.failure(illegal)).startswith("### " + expected)


def test_a_timeout_on_the_gpu_is_billed_and_points_at_the_run_itself():
    text = visible(card("timeout", True))
    assert text.startswith("### The run ran out of GPU time")
    assert "Any GPU time this run used is billed." in text


def test_the_input_and_quota_cards_open_their_detail():
    for category in ("input", "quota", "limit"):
        assert '<details class="mr-detail" open>' in card(category, False), category
    assert '<details class="mr-detail" open>' not in card("bug", True)


# --------------------------------------------------------------------------
# F4: Enhance states its cold start, and mentions an image only where there is one
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode_id", [m for m in MODE_IDS if modes.accepts(modes.get_mode(m), "text")]
)
def test_the_enhance_label_states_the_reservation_and_the_cold_start(mode_id):
    mode = modes.get_mode(mode_id)
    info = render.enhance_info(mode, zerogpu.ENHANCE_DURATION_SECONDS)
    assert f"Reserves {zerogpu.ENHANCE_DURATION_SECONDS} s more" in info
    assert "up to about 2 minutes on a cold start" in info
    assert "GPU" in info
    assert ("image" in info) == modes.accepts(mode, "image"), (
        f"{mode_id}: the label {'mentions' if 'image' in info else 'leaves out'} an image"
    )


def test_the_enhance_label_follows_the_mode(tmp_path):
    from test_ui_form import Form

    from ltx25_studio.ui import build_ui

    form = Form(build_ui())
    enhance = form.fields["enhance_prompt"]
    for mode_id in ("text_to_video", "image_text_to_audio_video", "audio_to_video"):
        form.pick_mode(mode_id)
        expected = render.enhance_info(modes.get_mode(mode_id), zerogpu.ENHANCE_DURATION_SECONDS)
        assert form.props[enhance._id]["info"] == expected, mode_id


# --------------------------------------------------------------------------
# minor 5: lever copy says what it does, in figures the estimator gives
# --------------------------------------------------------------------------


def test_the_upscale_label_says_how_many_seconds_it_adds():
    info = next(spec for spec in FIELDS if spec.key == "enable_stage_2").info
    extra = zerogpu.estimate_duration(enable_stage_2=True) - zerogpu.estimate_duration()
    assert f"Reserves {extra} s more." in info
    assert "1.5 times" not in info


def test_the_change_choice_in_audio_to_sound_names_its_own_ends():
    info = render.FIELD_COPY["noise_scale"][1]
    assert "lower keeps more" not in info
    assert "A little" in info and "A lot" in info


def test_a_clip_mode_without_its_clip_says_the_length_is_the_clips():
    for mode_id in [m for m in MODE_IDS if modes.accepts(modes.get_mode(m), "video")]:
        values = {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id) | {"video": None}
        facts = handlers.review(mode_id, values, length_source=render.LENGTH_MATCH)
        panel = visible(render.cost_panel(facts))
        assert "the length of your clip" in panel, (mode_id, panel)
        assert "121 frames" not in panel


# --------------------------------------------------------------------------
# minor 7: the phone's commit bar gets a line short enough to read whole
# --------------------------------------------------------------------------


def test_the_commit_bar_names_a_missing_input_in_a_few_words():
    for mode_id in MODE_IDS:
        values = {spec.key: spec.default for spec in FIELDS}
        facts = handlers.review(mode_id, values)
        line = render.commit_line(facts)
        assert line.startswith("Needs "), (mode_id, line)
        assert len(line) <= 40, line


def test_the_commit_bar_counts_the_problems_of_a_request_that_wont_run():
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_video")
    one = render.commit_line(handlers.review("text_to_video", values | {"width": 700}))
    two = render.commit_line(
        handlers.review("text_to_video", values | {"width": 700, "num_frames": 120})
    )
    assert one.startswith("Won’t run: width") and len(one) <= 50, one
    assert two == "Won’t run: 2 problems (width, frames)", two


# --------------------------------------------------------------------------
# minor 8: the Advanced hints agree with the limits the panel enforces
# --------------------------------------------------------------------------


def test_the_advanced_hints_respect_the_limits():
    from ltx25_studio import constraints

    dims, frames = handlers.on_shape_change(2048, 480, 489)
    assert "runs" not in dims and str(constraints.MAX_DIMENSION) in dims, dims
    assert "legal length" not in frames and str(constraints.MAX_NUM_FRAMES) in frames, frames
    dims, frames = handlers.on_shape_change(1920, 480, 481)
    assert "runs" in dims and "legal length" in frames
