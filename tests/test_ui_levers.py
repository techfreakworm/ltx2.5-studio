"""No inert levers (review round 2, F6; AC-15).

The decoder choice is a lever only while Upscale ×2 is off: stage 2 always finishes with the
model's diffusion decoder (``pipelines.StageRunner``), so under Upscale the choice is shown
but cannot be changed, and the line under it says why. Proved through the real handlers and,
for the reason itself, through a real ``StageRunner`` over fake stages.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_decode_policy import OK, runner_on
from test_ui_form import Form

from ltx25_studio import modes
from ltx25_studio.ui import render

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

PICTURE_MODES = [m for m, s in modes.MODES.items() if "video" in s.exports and not s.skip_stage_2]


@pytest.fixture(scope="module")
def demo():
    from ltx25_studio.ui import build_ui

    return build_ui()


def test_stage_2_finishes_with_the_diffusion_decoder_whatever_the_choice():
    """Why the choice is not a lever under Upscale: the real stage runner ignores it."""
    ran = set()
    for policy in ("auto", "diffusion", "conv"):
        runner, _pipeline, _built = runner_on("cuda")
        runner(prompt="a", video_decoder=policy, enable_stage_2=True, **OK)
        ran.add(runner.last_run["decoder"])
    assert ran == {"diffusion"}
    single = set()
    for policy in ("diffusion", "conv"):
        runner, _pipeline, _built = runner_on("cuda")
        runner(prompt="a", video_decoder=policy, **OK)
        single.add(runner.last_run["decoder"])
    assert single == {"diffusion", "conv"}, "without Upscale the choice is a real lever"


@pytest.mark.parametrize("mode_id", PICTURE_MODES)
def test_the_decoder_choice_cannot_be_changed_while_upscale_is_on(demo, mode_id):
    form = Form(demo)
    decoder, line = form.fields["decoder"], form.by_id("decoder-line")
    form.pick_mode(mode_id)
    form.user(form.fields["decoder"], "conv")
    assert form.interactive(decoder) and form.get(line) == render.decoder_line("conv")

    form.user(form.fields["enable_stage_2"], True)
    assert not form.interactive(decoder), f"{mode_id}: the decoder is a dead lever under Upscale"
    assert form.get(line) == render.UPSCALE_DECODER_LINE

    form.user(form.fields["enable_stage_2"], False)
    assert form.interactive(decoder) and form.get(line) == render.decoder_line("conv")


def test_upscale_carried_into_another_mode_keeps_the_decoder_locked(demo):
    form = Form(demo)
    form.user(form.fields["enable_stage_2"], True)
    form.pick_mode("text_to_audio")  # no picture: no decoder, no upscale
    assert not form.visible(form.fields["decoder"])
    form.pick_mode("image_text_to_video")
    assert form.field("enable_stage_2") is True
    assert not form.interactive(form.fields["decoder"])
    assert form.get(form.by_id("decoder-line")) == render.UPSCALE_DECODER_LINE


# --------------------------------------------------------------------------
# every visible control changes what runs (F6's acceptance, AC-15)
# --------------------------------------------------------------------------


def fingerprint(value):
    """A deterministic summary of what reached the pipeline (content, never identity)."""
    import hashlib

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(fingerprint(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), fingerprint(v)) for k, v in value.items()))
    if hasattr(value, "tobytes") and hasattr(value, "size") and hasattr(value, "mode"):
        return ("image", value.size, hashlib.sha1(value.tobytes()).hexdigest())
    if hasattr(value, "shape") and hasattr(value, "tobytes"):
        return ("array", tuple(value.shape), hashlib.sha1(value.tobytes()).hexdigest())
    if hasattr(value, "initial_seed"):
        return ("generator", value.initial_seed())
    if hasattr(value, "__dict__"):
        fields = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return (type(value).__name__, fingerprint(fields))
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return (type(value).__name__, fingerprint(dataclasses.asdict(value)))
    return type(value).__name__


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """``handlers.run`` reaches the real ``generate()``; the pipeline, the encoder and the
    adapter attach are the seams. Each audio file loads as its own waveform."""
    import numpy as np
    from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

    from ltx25_studio import audio as audio_module
    from ltx25_studio import generate as generate_module
    from ltx25_studio import lora

    def load(source):
        waveform = np.zeros((2, round(121 / 24.0 * 16000)), dtype=np.float32)
        waveform[:, ::97] = 0.01 * (1 + len(str(source)) % 7)
        return waveform, 16000

    monkeypatch.setattr(audio_module, "load_waveform", load)
    monkeypatch.setattr(lora, "lora_cache_dir", lambda: tmp_path / "lora-cache")
    seen = {"pipes": [], "adapters": []}

    def attach(pipe, request):
        seen["adapters"].append(
            tuple((e.get("source"), e.get("weight")) for e in getattr(request, "loras", ()) or ())
        )
        return ()

    monkeypatch.setattr(generate_module, "_apply_loras", attach)
    real = generate_module.generate

    def generate(**params):
        pipe = FakePipe()
        seen["pipes"].append(pipe)
        return real(
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            enhancer=lambda prompt, image=None: "ENHANCED " + prompt,
            output_root=tmp_path / f"run-{len(seen['pipes'])}",
            retention_bytes=10**9,
            **params,
        )

    monkeypatch.setattr(generate_module, "generate", generate)
    return seen


def what_runs(form: Form, seen: dict):
    from ltx25_studio.ui import handlers

    seen["adapters"].append(())
    outcome = handlers.run(form.get(form.radio), *form.form_values())
    assert outcome.error is None, outcome.status
    pipe = seen["pipes"][-1]
    sent = {k: v for k, v in pipe.calls[-1].items() if k != "progress"}
    mels = [fingerprint(m) for m in getattr(pipe.audio_vae, "mels", [])]
    return fingerprint(sent), tuple(mels), seen["adapters"][-1], outcome.result.seed


def _media(tmp_path):
    from conftest import audio_file, video_file
    from PIL import Image

    other_audio = tmp_path / "second-sound-file.wav"
    other_audio.write_bytes(Path(audio_file()).read_bytes())
    return {
        "image": (
            Image.new("RGB", (64, 64), (40, 40, 40)),
            Image.new("RGB", (64, 64), (200, 30, 30)),
        ),
        # long enough for stage 2's decoder (17 frames at least)
        "video": (video_file(frames=25), video_file(frames=41)),
        "audio": (audio_file(), str(other_audio)),
    }


def _baseline(demo, mode_id, *, upscale, auto, media, tmp_path):
    from lora_fixtures import good_adapter

    from ltx25_studio.ui import handlers

    form = Form(demo)
    form.pick_mode(mode_id)
    view = handlers.mode_view(mode_id)
    for kind in ("image", "video", "audio"):
        if view.active[kind]:
            form.user(form.fields[kind], media[kind][0])
    if view.active["prompt"]:
        form.user(form.fields["prompt"], "a quiet harbour at dawn")
    form.user(form.fields["seed"], 7)
    if upscale:
        form.user(form.fields["enable_stage_2"], True)
    if auto:
        form.user(form.fields["auto_duration"], True)
    _v, staged, _s = handlers.on_validate_lora("", str(good_adapter(tmp_path / "a")), [])
    form.values[form.fields["loras"]._id] = staged
    form.run(form.fn("_lora_rows"))
    return form


def _controls(form: Form):
    """Every control the page shows and lets the user change, with a new value for it."""
    by = form.by_id
    rows = [c for c in form.fn("_lora_rows").outputs if isinstance(c, gr.Slider)]
    candidates = [
        ("prompt", form.fields["prompt"], lambda v: (v or "") + " at night"),
        ("enhance", form.fields["enhance_prompt"], lambda v: not v),
        ("width", form.fields["width"], lambda v: int(v) + 32),
        ("height", form.fields["height"], lambda v: int(v) + 32),
        ("frames", form.fields["num_frames"], lambda v: 97 if int(v) != 97 else 121),
        ("shortest", form.fields["min_seconds"], lambda v: float(v) + 1.0),
        ("longest", form.fields["max_seconds"], lambda v: float(v) + 1.0),
        ("model picks", form.fields["auto_duration"], lambda v: not v),
        ("upscale", form.fields["enable_stage_2"], lambda v: not v),
        ("seed", form.fields["seed"], lambda v: int(v) + 1),
        ("decoder", form.fields["decoder"], lambda v: "conv" if v != "conv" else "diffusion"),
        ("strength", form.fields["strength"], lambda v: 0.4),
        ("shape", by("shape"), lambda v: "16:9" if v != "16:9" else "1:1"),
        ("seconds", by("seconds"), lambda v: 3.0 if float(v or 0) != 3.0 else 4.0),
        ("sound change", by("lever-noise"), lambda v: "A lot" if v != "A lot" else "A little"),
        ("clip change", by("lever-clip"), lambda v: "A little" if v != "A little" else "A lot"),
        ("adapter strength", rows[0], lambda v: 0.6),
    ]
    for kind in ("image", "video", "audio"):
        candidates.append((kind, form.fields[kind], kind))
    return [
        (name, component, change)
        for name, component, change in candidates
        if form.visible(component) and form.interactive(component)
    ]


CASES = [
    (mode_id, upscale, auto)
    for mode_id, spec in modes.MODES.items()
    for upscale in ((False, True) if not spec.skip_stage_2 else (False,))
    for auto in (
        (False, True)
        if not (modes.accepts(spec, "video") or modes.requires(spec, "audio"))
        else (False,)
    )
]


@pytest.mark.parametrize(("mode_id", "upscale", "auto"), CASES)
def test_every_visible_control_changes_what_reaches_the_pipeline(
    demo, wired, tmp_path, mode_id, upscale, auto
):
    media = _media(tmp_path)
    form = _baseline(demo, mode_id, upscale=upscale, auto=auto, media=media, tmp_path=tmp_path)
    base = what_runs(form, wired)
    assert what_runs(form, wired) == base, "the fingerprint is not deterministic"

    decoder = form.fields["decoder"]
    if upscale and form.visible(decoder):
        assert not form.interactive(decoder), "under Upscale the decoder choice is not a lever"

    controls = _controls(form)
    assert controls, mode_id
    inert = []
    for name, component, change in controls:
        twin = form.copy()
        if isinstance(change, str):  # a media kind: the other file
            twin.user(component, media[change][1])
        elif name == "seconds" and form.visible(form.by_id("length-source")):
            twin.user(twin.by_id("length-source"), "Set the length")
            twin.user(component, change(twin.get(component)))
        else:
            twin.user(component, change(twin.get(component)))
        if what_runs(twin, wired) == base:
            inert.append(name)
    assert not inert, f"{mode_id} (upscale={upscale}, model picks={auto}): inert {inert}"

    # and the adapter's own Remove takes it out of the run
    twin = form.copy()
    remove = next(
        c for c in form.fn("_lora_rows").outputs if isinstance(c, gr.Button) and c.value == "Remove"
    )
    twin.click(remove)
    assert what_runs(twin, wired)[2] == (), "Remove left the adapter in the run"
