"""The words on screen: true, plain and in the user's terms (B3, M3, M6, AC-17, AC-24, AC-28).

* a failure has a plain category headline (never a Python class name), says
  whether quota was used, gives one next step and keeps the verbatim detail
  in a collapsible block; it never says "unexpected" or "not one of ours";
* a success report is the seed, the backend's notes as plain text and a
  collapsed Diagnostics block, with no post-run cost reminder;
* no visible string carries a plan reference, a file name, builder jargon or
  a snake_case mode id.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import pytest

from ltx25_studio import errors, modes
from ltx25_studio.ui import handlers, render
from ltx25_studio.ui.fields import FIELDS

MODE_IDS = tuple(modes.MODES)
FORBIDDEN = ("PLAN §", ".py", "kwarg", "upstream", "latent", "Upstream", "Latent")
CLASS_NAMES = tuple(
    name
    for name in dir(errors)
    if name.endswith("Error") and isinstance(getattr(errors, name), type)
)


def visible(markup: str) -> str:
    """Text a user sees: closed <details> bodies and tags removed."""
    markup = re.sub(r"<details(?![^>]*\bopen\b)[^>]*>.*?</details>", "", markup, flags=re.S)
    return re.sub(r"<[^>]+>", " ", markup)


def assert_clean(text: str, where: str) -> None:
    for word in FORBIDDEN:
        assert word not in text, f"{where}: {word!r} in {text!r}"
    for mode_id in MODE_IDS:
        assert mode_id not in text, f"{where}: snake_case id {mode_id!r} in {text!r}"


FAILURES = [
    errors.DecodeError(mode="text_to_video", reason="the pipeline raised OutOfMemoryError()"),
    errors.ModelLoadError(component="transformer", reason="disk full"),
    errors.EncodeError(path="/tmp/run", reason="no muxer"),
    errors.LoRANotAttachedError(
        source="a.safetensors", adapter_name="user_lora_0", components=("transformer",)
    ),
    errors.PromptEnhancementError(
        repo_id="google/gemma", reason="the enhancer could not be fetched"
    ),
    RuntimeError("You have exceeded your GPU quota (300s requested vs. 112s left)."),
]


@pytest.mark.parametrize("exc", FAILURES, ids=lambda e: type(e).__name__)
def test_a_gpu_side_failure_has_a_plain_headline_a_quota_line_a_step_and_the_detail(exc):
    text = render.failure(exc)
    headline = text.split("\n", 1)[0]
    assert headline.startswith("### ")
    assert type(exc).__name__ not in headline
    assert "Any GPU time this run used is billed." in text
    assert "press generate again" in text.lower() or "report" in text.lower()
    assert "<details" in text and "<summary>Detail</summary>" in text
    assert str(exc).strip().split("(")[0][:30] in text.replace("&#x27;", "'").replace("&quot;", '"')
    lowered = text.lower()
    assert "unexpected" not in lowered and "not one of ours" not in lowered


def test_a_refusal_says_nothing_ran_and_keeps_its_detail_open():
    exc = errors.MissingInputError(mode="Image to video", missing=("image",))
    text = render.failure(exc, refused=True)
    assert text.startswith("### Refused before running")
    assert "Nothing ran. No GPU time was used." in text
    assert '<details class="mr-detail" open>' in text
    assert str(exc) in text


def test_the_rendering_survives_the_fork_round_trip():
    exc = errors.DecodeError(mode="text_to_video", reason="boom")
    assert render.failure(pickle.loads(pickle.dumps(exc))) == render.failure(exc)


class Result:
    seed = 1234
    run_dir = Path("outputs/run-abc")
    warnings = (
        "stage 2 returned 1408x960, twice the size you set.",
        "length set from your audio: 5.00 s → 121 frames.",
    )


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_a_success_reads_as_success_with_no_cost_reminder(mode_id):
    mode = modes.get_mode(mode_id)
    text = render.success(Result(), mode)
    assert "Seed 1234" in text
    assert "Reminder" not in text and "⚠" not in text and "> " not in text
    assert '<details class="mr-diagnostics"><summary>Diagnostics</summary>' in text
    shown = visible(text)
    assert "Stage 2 returned 1408x960" in shown, "notes are plain text, capitalised"
    assert "outputs/run-abc" not in shown, "the run folder lives in Diagnostics"
    assert_clean(shown, f"success({mode_id})")


def test_the_run_summary_lists_what_you_got_in_order():
    facts = render.RunFacts(
        mode=modes.get_mode("text_to_audio_video"),
        seed=99,
        returned_size=(1408, 960),
        seconds=5.04,
        stage_2=True,
        loras=(("style.safetensors", 0.8),),
        took=41.2,
    )
    text = visible(render.run_summary(facts))
    order = [text.index(word) for word in ("Mode", "Size", "Length", "Seed", "LoRA", "Took")]
    assert order == sorted(order)
    assert "upscaled ×2" in text and "99" in text and "0.8" in text and "41" in text


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_no_visible_string_carries_builder_jargon(mode_id):
    mode = modes.get_mode(mode_id)
    view = handlers.mode_view(mode_id)
    strings = [
        view.purpose,
        view.prompt_label,
        view.prompt_placeholder,
        *view.captions.values(),
        view.pinned_line or "",
        render.monitor_idle(mode),
        render.running(mode),
        render.running(mode, enhancing=True),
        render.mode_name(mode),
        render.monitor_caption(kind="run", mode=mode, seed=1),
    ]
    from conftest import inputs_for

    values = {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id)
    facts = handlers.review(mode_id, values)
    strings += [render.cost_panel(facts), render.commit_line(facts)]
    for text in strings:
        assert_clean(visible(text), mode_id)


def test_field_labels_and_infos_are_clean():
    for spec in FIELDS:
        assert_clean(spec.label + " " + spec.info, spec.key)
    for _label, info in render.FIELD_COPY.values():
        assert "PLAN" not in info


def test_the_copy_the_verdict_required_is_in_place():
    assert render.TAGLINE == (
        "Video and sound from text, images, clips or audio. Bring your own LoRA."
    )
    quota = render.quota_explainer(multiplier=2, example_reserve=60)
    assert "a 60 s reservation needs 120 s of quota free to start" in quota
    assert "Runs that fail on the GPU are billed too." in quota
    assert render.media_caption(modes.get_mode("audio_to_video"), "audio") == (
        "The sound to picture; it is also the soundtrack of the result."
    )
    assert render.media_caption(modes.get_mode("image_to_video"), "image") == (
        "Used as the first frame."
    )
    enhance = next(spec for spec in FIELDS if spec.key == "enhance_prompt")
    assert "Reserves 180 s more" in enhance.info and "GPU" in enhance.info
    assert "frames" in render.frame_hint(120, ratio=8, lower=113, upper=121, valid=False)
    assert "113" in render.frame_hint(120, ratio=8, lower=113, upper=121, valid=False)
