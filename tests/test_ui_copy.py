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
#: Measured on real weights: in Audio to video the mouth does not follow the recorded words,
#: while speech the model writes itself (Image + text to video with sound) is in time.
NOT_LIP_SYNCED = (
    "Speech is not lip-synced. For a face that speaks in time, use Image + text to video "
    "with sound and put the line in quotes."
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
        mode.label,
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
        "The sound to picture; it is also the soundtrack of the result. " + NOT_LIP_SYNCED
    )
    assert render.media_caption(modes.get_mode("image_to_video"), "image") == (
        "Used as the first frame."
    )
    enhance = next(spec for spec in FIELDS if spec.key == "enhance_prompt")
    assert "Reserves 180 s more" in enhance.info and "GPU" in enhance.info
    assert "frames" in render.frame_hint(120, ratio=8, lower=113, upper=121, valid=False)
    assert "113" in render.frame_hint(120, ratio=8, lower=113, upper=121, valid=False)


def _texts(form, block) -> list[str]:
    """Every string the page holds for ``block``: its value and its words."""
    props = form.props[block._id]
    held = [form.get(block), props.get("label"), props.get("info"), props.get("placeholder")]
    held += [str(choice) for choice in props.get("choices") or ()]
    return [text for text in held if isinstance(text, str)]


def test_the_lip_sync_line_is_under_the_audio_in_audio_to_video_and_nowhere_else():
    """Through the built page's own mode switch: the line shows, visibly, under the audio
    input in Audio to video, is held by no other component there, and by none in any other
    mode; the mode it sends people to is named by its label."""
    pytest.importorskip("gradio", reason="gradio is a declared dev dependency")
    from test_ui_form import Form

    from ltx25_studio.ui import build_ui

    assert modes.get_mode("image_text_to_audio_video").label in NOT_LIP_SYNCED
    demo = build_ui()
    form = Form(demo)
    blocks = sorted(demo.blocks.values(), key=lambda block: block._id)
    captions = [b for b in blocks if "mr-caption" in (getattr(b, "elem_classes", None) or ())]
    audio_caption = next(b for b in captions if b._id > form.fields["audio"]._id)

    def holders() -> list:
        return [b for b in blocks if any(NOT_LIP_SYNCED in t for t in _texts(form, b))]

    assert holders() == [], "the built page, before any mode is picked"
    for mode_id in MODE_IDS:
        form.pick_mode(mode_id)
        if mode_id == "audio_to_video":
            assert holders() == [audio_caption]
            assert form.visible(audio_caption) and form.visible(form.fields["audio"])
        else:
            assert holders() == [], mode_id
    form.pick_mode("audio_to_audio")
    assert holders() == [], "switching away from Audio to video takes the line away"
    assert form.get(audio_caption) == render.media_caption(
        modes.get_mode("audio_to_audio"), "audio"
    )


def test_the_start_message_says_enhancing_only_when_the_prompt_will_be_enhanced():
    assert "Enhancing your prompt" in handlers.on_generate_start("text_to_video", True, "a tram")
    assert "Enhancing" not in handlers.on_generate_start("text_to_video", True, "   ")
    assert "Enhancing" not in handlers.on_generate_start("image_to_video", True, "stale")
    assert "Enhancing" not in handlers.on_generate_start("text_to_video", False, "a tram")


@pytest.mark.parametrize(
    ("category", "headline"),
    [
        (errors.QUOTA, "Not enough GPU quota"),
        (errors.TIMEOUT, "ran out of GPU time"),
        (errors.OOM, "ran out of memory"),
        (errors.ADAPTER, "adapter doesn't fit"),
        (errors.ENHANCER, "enhancer is unavailable"),
        (errors.BUG, "went wrong in this app"),
    ],
)
def test_a_failure_reads_the_category_errors_categorise_gives_it(category, headline):
    exc = errors.DecodeError(mode="text_to_video", reason="x")
    exc.category = category
    exc.gpu_used = True
    text = render.failure(exc)
    assert headline in text.split("\n", 1)[0]
    assert "Any GPU time this run used is billed." in text
    exc.gpu_used = False
    assert "Nothing ran. No GPU time was used." in render.failure(exc)


def test_platform_and_torch_errors_are_categorised_by_errors_categorise():
    quota = RuntimeError("You have exceeded your GPU quota (300s requested vs. 112s left).")
    text = render.failure(quota)
    assert text.startswith("### Not enough GPU quota")
    assert "Nothing ran. No GPU time was used." in text, "a quota refusal used no GPU time"
    assert "300s requested vs. 112s left" in text and '<details class="mr-detail" open>' in text
    oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert render.failure(oom).startswith("### The GPU ran out of memory")
    assert render.failure(RuntimeError("no idea")).startswith("### Something went wrong")


def test_an_input_problem_raised_inside_the_run_does_not_claim_nothing_ran():
    exc = errors.ConstraintError("the decoder cannot run here")
    exc.gpu_used = True
    text = render.failure(exc)
    assert text.startswith("### The run was refused") and "billed" in text
    exc.gpu_used = False
    assert render.failure(exc).startswith("### Refused before running")


def test_run_facts_read_the_results_record_first_and_fall_back_to_the_request():
    from ltx25_studio import generate as generate_module

    params = {spec.key: spec.default for spec in FIELDS} | {
        "width": 704,
        "height": 480,
        "num_frames": 121,
        "enable_stage_2": True,
        "prompt": "a tram",
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
    assert facts.length_source == generate_module.LENGTH_FROM_REQUEST
    assert facts.prompt_sent is None and facts.decoder is None
    shown = " ".join(visible(render.run_summary(facts)).split())
    assert "Took 12 s" in shown, "the page's own clock, unlabelled"

    record = generate_module.RunRecord(
        mode="text_to_audio_video",
        requested_width=704,
        requested_height=480,
        width=704,
        height=480,
        frames=97,
        frame_rate=24.0,
        seconds=4.04,
        seed=10,
        stage_2=False,
        decoder="diffusion",
        audio_level=None,
        length_source=generate_module.LENGTH_FROM_MODEL,
        prompt="a red vintage tram climbs a cobbled street at golden hour",
        adapters=(("org/style", 1.2, 12),),
        elapsed_seconds=33.0,
        peak_vram_gib=41.5,
    )

    class Recorded:
        seed = 9
        warnings = ()
        run_dir = "outputs/run-x"

    Recorded.record = record
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
    assert facts.returned_size == (704, 480) and not facts.stage_2 and facts.seconds == 4.04
    assert facts.seed == 10 and facts.loras == (("org/style", 1.2, 12),)
    assert facts.took == 33.0 and facts.took_is_gpu and facts.decoder == "diffusion"
    assert facts.length_source == generate_module.LENGTH_FROM_MODEL
    assert facts.prompt_sent == "a red vintage tram climbs a cobbled street at golden hour"
    summary = " ".join(visible(render.run_summary(facts)).split())
    for words in (
        "Took 33 s",
        "golden hour",
        "chosen by the model",
        "on 12 layers",
        "Decoder Best quality",
    ):
        assert words in summary, words
    report = render.success(Recorded(), modes.get_mode("text_to_audio_video"))
    assert "Peak GPU memory reserved: 41.50 GiB" in report
    assert "41.50" not in visible(report), "VRAM lives in the collapsed Diagnostics block"


def test_the_decoder_choices_say_what_each_does_and_the_report_names_the_one_that_ran():
    from ltx25_studio.config import DECODER_CHOICES

    assert [value for _l, value, _w in render.DECODER_OPTIONS] == list(DECODER_CHOICES)
    for label, value, words in render.DECODER_OPTIONS:
        assert render.decoder_line(value) == words
        assert words.endswith(".") and len(words) < 80, label
    assert render.decoder_line("conv", upscaling=True) == render.UPSCALE_DECODER_LINE
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
    assert "Best quality" in render.run_summary(facts)
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
