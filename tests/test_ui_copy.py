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
    lowered_step = text.lower()
    assert (
        "press generate again" in lowered_step
        or "report" in lowered_step
        or "turn enhance off to run with your own prompt" in lowered_step
    )
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


def test_the_start_message_says_enhancing_only_when_the_prompt_will_be_enhanced():
    assert "Enhancing your prompt" in handlers.on_generate_start("text_to_video", True, "a tram")
    assert "Enhancing" not in handlers.on_generate_start("text_to_video", True, "   ")
    assert "Enhancing" not in handlers.on_generate_start("image_to_video", True, "stale")
    assert "Enhancing" not in handlers.on_generate_start("text_to_video", False, "a tram")


@pytest.mark.parametrize(
    ("category", "headline"),
    [
        ("quota", "Not enough GPU quota"),
        ("timeout", "ran out of GPU time"),
        ("out_of_memory", "ran out of memory"),
        ("adapter", "adapter doesn't fit"),
        ("enhancer", "enhancer is unavailable"),
        ("bug", "went wrong in this app"),
    ],
)
def test_a_failure_the_backend_categorised_reads_its_category(category, headline):
    exc = errors.DecodeError(mode="text_to_video", reason="x")
    exc.category = category
    text = render.failure(exc)
    assert headline in text.split("\n", 1)[0]
    assert "Any GPU time this run used is billed." in text
    exc.gpu_used = False
    assert "Nothing ran. No GPU time was used." in render.failure(exc)


def test_run_facts_read_the_results_record_first_and_fall_back_to_the_request():
    params = {spec.key: spec.default for spec in FIELDS} | {
        "width": 704,
        "height": 480,
        "num_frames": 121,
        "enable_stage_2": True,
        "loras": [{"source": "/tmp/a/style.safetensors", "weight": 0.8}],
    }

    class Bare:
        seed = 9

    bare = handlers.RunOutcome(
        video="v",
        audio="v",
        status="",
        mode_id="text_to_audio_video",
        params=params,
        result=Bare(),
        took=12.4,
    )
    facts = handlers.run_facts(bare)
    assert facts.returned_size == (1408, 960) and facts.stage_2 and facts.seed == 9
    assert facts.loras == (("style.safetensors", 0.8),) and not facts.took_is_gpu
    assert "wall-clock" in render.run_summary(facts)

    class Record:
        width, height, seconds, stage_2 = 704, 480, 4.0, False
        worker_seconds, prompt_sent, seed = 33.0, "an enhanced prompt", 10
        adapters = ({"source": "org/style", "weight": 1.2, "modules_attached": 12},)

    class Recorded:
        seed = 9
        record = Record()

    facts = handlers.run_facts(
        handlers.RunOutcome(
            video="v",
            audio="v",
            status="",
            mode_id="text_to_audio_video",
            params=params,
            result=Recorded(),
            took=12.4,
        )
    )
    assert facts.returned_size == (704, 480) and not facts.stage_2 and facts.seconds == 4.0
    assert facts.seed == 10 and facts.loras == (("org/style", 1.2),)
    assert facts.took == 33.0 and facts.took_is_gpu
    summary = render.run_summary(facts)
    assert "GPU time" in summary and "an enhanced prompt" in summary


def test_the_decoder_choices_say_what_each_does_and_the_report_names_the_one_that_ran():
    from ltx25_studio.config import DECODER_CHOICES

    assert [value for _l, value, _w in render.DECODER_OPTIONS] == list(DECODER_CHOICES)
    assert render.decoder_line("auto") == (
        "Uses the model's own decoder when this machine can run it."
    )
    assert render.decoder_line("diffusion") == (
        "The model's diffusion decoder: finer detail, takes longer."
    )
    assert render.decoder_line("conv") == "The quick decoder: slightly softer."
    facts = render.RunFacts(
        mode=modes.get_mode("text_to_video"),
        seed=1,
        returned_size=(704, 480),
        seconds=5.0,
        stage_2=False,
        loras=(),
        took=3.0,
        decoder="diffusion",
    )
    assert "diffusion decoder" in render.run_summary(facts)
    no_record = render.RunFacts(
        mode=modes.get_mode("text_to_video"),
        seed=1,
        returned_size=(704, 480),
        seconds=5.0,
        stage_2=False,
        loras=(),
        took=3.0,
    )
    assert "Decoder" not in render.run_summary(no_record), "never claim a decoder nobody reported"
