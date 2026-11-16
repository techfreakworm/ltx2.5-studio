"""What the page says after a run, in the UI's own words (review round 2: F2, minors 1, 3, 4, 25, 26).

The backend owns the notes (``result.warnings``) and rewords them at the source; the UI shows
them as they come and does not translate them. What the UI owns is tested here: the report
head and its Diagnostics block, the "What you got" card, the monitor caption, Recent runs,
the cost title and every other place a mode or a decoder is named.

No test here copies a registry label into a literal: labels are read from the registry.
"""

from __future__ import annotations

import re

import pytest
from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM, image_input, inputs_for
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

from ltx25_studio import generate as generate_module
from ltx25_studio import modes
from ltx25_studio.ui import chrome, handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

MODE_IDS = tuple(modes.MODES)
#: Words the user never reads after a run (the brief's list for the UI's own strings).
JARGON = ("muxed", "latent", "stage 2", "stage-2", "kwarg", "upstream", "VRAM", "num_frames")
PATHS = re.compile(r"(/Users/|/var/|/tmp/|/home/|/data/|/private/)")
REPR = re.compile(r"'[A-Za-z_][^'\s]*'")


def visible(markup: str) -> str:
    markup = re.sub(r"<details(?![^>]*\bopen\b)[^>]*>.*?</details>", "", markup, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", markup).split())


def diagnostics(markup: str) -> str:
    found = re.search(r"<details class=\"mr-diagnostics\">(.*?)</details>", markup, flags=re.S)
    return found.group(1) if found else ""


def assert_ui_words(text: str, where: str) -> None:
    for mode_id in MODE_IDS:
        assert mode_id not in text, f"{where}: mode id {mode_id!r} in {text!r}"
    for word in JARGON:
        assert word.lower() not in text.lower(), f"{where}: {word!r} in {text!r}"
    assert not PATHS.search(text), f"{where}: a server path in {text!r}"
    assert not REPR.search(text), f"{where}: Python repr quotes in {text!r}"
    for word in ("paid", "billed"):
        assert word not in text.lower(), f"{where}: {word!r} after the run: {text!r}"


@pytest.fixture
def wired(monkeypatch, tmp_path):
    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )
    real = generate_module.generate

    def generate(**params):
        return real(
            pipeline_factory=factory_returning(FakePipe()),
            encoder=one_muxed_file_encoder,
            enhancer=lambda prompt, image=None: "A REWRITTEN " + prompt,
            output_root=tmp_path,
            retention_bytes=10**9,
            **params,
        )

    monkeypatch.setattr(generate_module, "generate", generate)


def run(mode_id: str, **overrides) -> handlers.RunOutcome:
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id) | {"seed": 4}
    values |= overrides
    outcome = handlers.run(mode_id, *(values[k] for k in FIELD_KEYS))
    assert outcome.error is None, outcome.status
    return outcome


RUNS = [(m, {}) for m in MODE_IDS] + [
    ("text_to_audio_video", {"enable_stage_2": True}),
    ("image_text_to_audio_video", {"enhance_prompt": True, "image": image_input()}),
]


@pytest.mark.parametrize(("mode_id", "overrides"), RUNS, ids=lambda v: str(v)[:40])
def test_what_the_ui_itself_says_after_a_run_is_in_the_users_words(wired, mode_id, overrides):
    outcome = run(mode_id, **overrides)
    mode = modes.get_mode(mode_id)
    facts = handlers.run_facts(outcome)
    report = render.success(outcome.result, mode)
    head = re.search(r'<p class="mr-report-head">(.*?)</p>', report).group(1)
    strings = {
        "report head": head,
        "what you got": visible(render.run_summary(facts)),
        "monitor caption": render.monitor_caption(kind="run", mode=mode, seed=facts.seed),
        "recent runs": render.recent_run_label(
            mode=mode, seed=facts.seed, size=facts.returned_size, seconds=facts.seconds, at="10:00"
        ),
        "running": render.running(mode, enhancing=bool(overrides.get("enhance_prompt"))),
    }
    for where, text in strings.items():
        assert_ui_words(text, f"{mode_id} {where}")
    assert head.startswith(f"Done: {mode.label}.")


def test_a_cuda_shaped_run_keeps_its_vram_figures_inside_diagnostics(wired, monkeypatch):
    """On CUDA the probe records VRAM laps; none of it may show above Diagnostics."""
    GiB = 1024**3

    class FakeCuda:
        def __init__(self):
            self.peak = 0

        def reset_peak_memory_stats(self):
            self.peak = 0

        def get_device_properties(self, index):
            return type("Props", (), {"total_memory": 95 * GiB})()

        def current_device(self):
            return 0

        def memory_allocated(self):
            return 30 * GiB

        def max_memory_allocated(self):
            return 41 * GiB

        def max_memory_reserved(self):
            return 44 * GiB

    original = generate_module._VramProbe.__init__

    def on_cuda(self):
        original(self)
        self._cuda = FakeCuda()

    monkeypatch.setattr(generate_module._VramProbe, "__init__", on_cuda)
    outcome = run("text_to_audio_video")
    record = outcome.result.record
    assert record.peak_vram_gib and record.diagnostics, (
        "the fake CUDA allocator did not reach the record"
    )
    report = render.success(outcome.result, modes.get_mode("text_to_audio_video"))
    # Placement, whatever the backend's words: every diagnostics line (the VRAM laps
    # among them) is inside the collapsed block, none above it, and so is the peak figure.
    shown = visible(report)
    inside = diagnostics(report)
    for line in record.diagnostics:
        assert render._e(line) in inside, f"a diagnostics line is missing: {line}"
        assert render._e(line) not in shown and line not in shown, (
            f"shown above Diagnostics: {line}"
        )
    peak = f"{record.peak_vram_gib:.2f}"
    assert peak in inside and peak not in shown, shown
    assert_ui_words(visible(render.run_summary(handlers.run_facts(outcome))), "what you got")


# --------------------------------------------------------------------------
# minor 1: one name per mode, one name per decoder choice
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_every_place_the_ui_names_a_mode_uses_the_registrys_label(mode_id):
    mode = modes.get_mode(mode_id)
    view = handlers.mode_view(mode_id)
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id)
    facts = handlers.review(mode_id, values)
    assert view.name == mode.label
    assert view.purpose.startswith(f"**{mode.label}.**")
    assert f">{render._e(mode.label)}</h2>" in render.cost_panel(facts)
    assert mode.label in render.monitor_idle(mode)
    assert render.monitor_caption(kind="run", mode=mode, seed=1).startswith(
        f"**Your run.** {mode.label}"
    )
    assert mode.label in render.recent_run_label(
        mode=mode, seed=1, size=None, seconds=None, at="9:00"
    )
    assert "→" not in render.cost_panel(facts) + view.purpose


def test_the_arrow_form_names_are_gone():
    assert not hasattr(render, "mode_name")


def test_the_report_names_the_decoder_with_the_choices_own_words():
    labels = {value: label for label, value, _words in render.DECODER_OPTIONS}
    for decoder in ("diffusion", "conv"):
        facts = render.RunFacts(
            mode=modes.get_mode("text_to_video"),
            seed=1,
            returned_size=(704, 480),
            seconds=5.0,
            stage_2=False,
            loras=(),
            took=3.0,
            decoder=decoder,
        )
        shown = visible(render.run_summary(facts))
        assert f"Decoder {labels[decoder]}" in shown, shown
    lines = " ".join(render.decoder_line(v) for v in labels) + render.UPSCALE_DECODER_LINE
    for other_name in ("quick decoder", "standard", "convolutional", "diffusion decoder"):
        assert other_name not in lines, other_name


# --------------------------------------------------------------------------
# minor 4: "Took" is labelled by the device that ran it
# --------------------------------------------------------------------------


def _took(device, *, zerogpu=False, gpu_time=True) -> str:
    facts = render.RunFacts(
        mode=modes.get_mode("text_to_video"),
        seed=1,
        returned_size=(704, 480),
        seconds=5.0,
        stage_2=False,
        loras=(),
        took=27.0,
        took_is_gpu=gpu_time,
        device=device,
        on_zerogpu=zerogpu,
        quota_multiplier=2,
    )
    return re.search(r"Took (.*?)$", visible(render.run_summary(facts))).group(1)


def test_took_says_where_the_run_ran():
    assert _took("cpu") == "27 s on the CPU"
    assert _took("mps") == "27 s on this Mac's GPU"
    assert _took("cuda") == "27 s on the GPU"
    assert _took("cuda", zerogpu=True) == "27 s of GPU time, about 54 s of your quota"
    assert _took(None, gpu_time=False) == "27 s"


def test_the_run_facts_name_this_processs_device(wired, monkeypatch):
    monkeypatch.setenv("LTX25_DEVICE", "cpu")
    facts = handlers.run_facts(run("text_to_video"))
    assert facts.device == "cpu" and not facts.on_zerogpu
    assert "on the CPU" in visible(render.run_summary(facts))


# --------------------------------------------------------------------------
# minor 3 and minor 26: the LoRA bay
# --------------------------------------------------------------------------


def test_the_lora_copy_agrees_with_itself_and_shows_a_file_name_not_a_path(tmp_path):
    from lora_fixtures import good_adapter

    from ltx25_studio import lora

    heading = render.lora_heading("12 GB")
    assert "Whether it fits LTX-2.5 is confirmed when you generate" not in heading
    report = lora.validate_lora(str(good_adapter(tmp_path)), upload=True)
    accepted = render.lora_accepted("film-grain.safetensors", report)
    assert "6.5" not in accepted and "Checks 1" not in accepted
    assert "strength 1.0" not in accepted, "the card goes stale once the strength is set"

    upload = tmp_path / "var" / "folders" / "models--upload--abc" / "style.safetensors"
    exc = lora.LoRAError(f"{upload} is not a LoRA adapter")
    rejected = render.lora_rejected(str(upload), exc)
    assert str(upload) not in rejected and "style.safetensors is not a LoRA adapter" in rejected


def test_the_strength_range_is_the_lora_modules_own():
    """A staged row's slider is given the module's range when it is shown; the package
    itself stays lazy until an adapter is checked (``test_module_reachability``)."""
    pytest.importorskip("gradio")
    from ltx25_studio.lora import staging
    from ltx25_studio.ui import build_ui

    demo = build_ui()
    rows = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "_lora_rows")
    staged = [{"source": "org/a", "weight": 0.8}, {"source": "org/b", "weight": 1.0}]
    out = rows.fn(staged)
    sliders = [
        update
        for component, update in zip(rows.outputs, out, strict=True)
        if str(getattr(component, "label", "")).startswith("Strength of adapter")
        and update.get("visible")
    ]
    assert len(sliders) == 2
    for update in sliders:
        assert update["maximum"] == staging.MAX_WEIGHT
        assert 0 < update["minimum"] < staging.DEFAULT_WEIGHT, "more than 0, as the module requires"
    assert [u["value"] for u in sliders] == [0.8, 1.0]


# --------------------------------------------------------------------------
# minor 25: accessible names say what is on screen
# --------------------------------------------------------------------------


def test_the_corner_number_is_not_read_as_part_of_a_cells_name():
    css = chrome._css()
    numbers = set(re.findall(r'content: "(\d\d)";', css))
    assert len(numbers) == len(MODE_IDS)
    assert all(f'content: "{n}" / "";' in css for n in numbers)


def test_the_idle_monitor_reads_its_mode_name():
    idle = render.monitor_idle(modes.get_mode("audio_to_video"))
    assert 'role="img"' not in idle and "Empty monitor" not in idle
    assert modes.get_mode("audio_to_video").label in idle
