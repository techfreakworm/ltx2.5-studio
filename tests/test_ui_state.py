"""State integrity through the real handlers (review round 2: F7, F8, F9, minors 9, 11-13).

Every test drives the built Blocks' own event functions through :class:`test_ui_form.Form`,
the way the page does, and where it matters runs the result through ``handlers.run`` and
the real ``generate()`` with the GPU seams faked.
"""

from __future__ import annotations

import pytest
from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM, image_input, inputs_for
from PIL import Image
from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder
from test_ui_form import Form

from ltx25_studio import generate as generate_module
from ltx25_studio import modes
from ltx25_studio.ui import chrome, handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

MODE_IDS = tuple(modes.MODES)


@pytest.fixture(scope="module")
def demo():
    from ltx25_studio.ui import build_ui

    return build_ui()


@pytest.fixture
def form(demo):
    return Form(demo)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """``handlers.run`` reaches the real ``generate()``; only the GPU seams are fakes."""
    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )
    real = generate_module.generate
    pipes = []

    def generate(**params):
        pipe = FakePipe()
        pipes.append(pipe)
        return real(
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path / f"run-{len(pipes)}",
            retention_bytes=10**9,
            **params,
        )

    monkeypatch.setattr(generate_module, "generate", generate)
    return pipes


def run_form(form: Form) -> handlers.RunOutcome:
    return handlers.run(form.get(form.radio), *form.form_values())


# --------------------------------------------------------------------------
# F7: Restore settings goes through the mode switch
# --------------------------------------------------------------------------


def _recent_entry(mode_id: str, **overrides):
    """A Recent runs entry made the way ``_finish`` makes one, from a real run."""
    from ltx25_studio.ui import blocks

    values = {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id) | overrides
    outcome = handlers.run(mode_id, *(values[k] for k in FIELD_KEYS))
    assert outcome.error is None, outcome.status
    facts = handlers.run_facts(outcome)
    return blocks._recent_entry(mode_id, outcome.result, outcome.params, facts, outcome)


def test_restore_parks_what_the_restored_mode_does_not_take(form, wired):
    """In 06 with an image (and a prompt "X" parked), restore a 05 run whose prompt was "Y"."""
    entry = _recent_entry("text_to_audio_video", prompt="Y", seed=11)
    form.user(form.fields["prompt"], "X")
    form.pick_mode("image_to_audio_video")
    image = image_input()
    form.user(form.fields["image"], image)
    recent = form.fn("_finish").inputs[1]  # _finish(request, mode, recent, *fields)
    form.values[recent._id] = [entry]
    restore = next(
        b
        for b in form.demo.blocks.values()
        if isinstance(b, gr.Button) and b.value == "Restore settings"
    )
    form.click(restore)

    assert form.get(form.radio) == "text_to_audio_video"
    assert form.field("image") is None, "the image 05 does not take would be submitted"
    assert not form.visible(form.fields["image"])
    assert form.field("prompt") == "Y" and form.field("seed") == 11

    outcome = run_form(form)
    assert outcome.error is None, outcome.status
    assert not any(
        "was supplied, but" in n or "was ignored" in n for n in outcome.result.warnings
    ), (
        f"a run straight after the restore told the user about a value they cannot see: "
        f"{outcome.result.warnings}"
    )

    form.pick_mode("image_to_audio_video")
    assert form.field("image") is image, "returning to 06 did not bring the image back"


def test_restore_never_runs_a_second_switch_that_could_undo_it(form, wired):
    """The radio's own change after a restore sees nothing to move (previous == mode)."""
    entry = _recent_entry("image_to_video", image=image_input(), seed=3)
    form.user(form.fields["prompt"], "kept for later")
    recent = form.fn("_finish").inputs[1]
    form.values[recent._id] = [entry]
    restore = next(
        b
        for b in form.demo.blocks.values()
        if isinstance(b, gr.Button) and b.value == "Restore settings"
    )
    form.ran.clear()
    form.click(restore)
    assert form.get(form.radio) == "image_to_video"
    assert form.field("prompt") == "", "02 takes no prompt: it must be parked"
    form.pick_mode("text_to_audio_video")
    assert form.field("prompt") == "kept for later"


# --------------------------------------------------------------------------
# F8: the Shape chip names the size that will render, or nothing
# --------------------------------------------------------------------------


def test_a_custom_size_checks_no_chip_after_a_mode_switch(form):
    shape = form.by_id("shape")
    form.pick_mode("text_to_video")
    form.user(form.fields["height"], 256)
    assert form.get(shape) is None, "704 × 256 is not a preset"
    form.pick_mode("image_text_to_video")
    assert (form.field("width"), form.field("height")) == (704, 256)
    assert form.get(shape) is None, "the chip claimed a shape the render does not have"


def test_match_input_then_a_mode_without_it_checks_the_chip_of_the_actual_size(form):
    shape = form.by_id("shape")
    form.pick_mode("image_to_audio_video")
    form.user(shape, render.MATCH_INPUT)
    form.user(form.fields["image"], Image.new("RGB", (1024, 768)))
    size = (form.field("width"), form.field("height"))
    assert size == handlers.match_size(1024, 768)
    form.pick_mode("text_to_audio_video")
    assert (form.field("width"), form.field("height")) == size
    assert form.get(shape) == handlers.preset_for(*size), (
        f"{size} is checked as {form.get(shape)!r}"
    )


# --------------------------------------------------------------------------
# F9: required media first, in every mode, from the registry
# --------------------------------------------------------------------------


def _layout_order(demo) -> list[int]:
    order = []

    def walk(node):
        order.append(node["id"])
        for child in node.get("children", []) or []:
            walk(child)

    walk(demo.get_config_file()["layout"])
    return order


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_every_mode_shows_its_required_media_before_its_optional_media(demo, mode_id):
    form = Form(demo)
    spec = modes.get_mode(mode_id)
    view = handlers.mode_view(mode_id)
    order = _layout_order(demo)
    shown = sorted(
        (kind for kind in ("image", "video", "audio") if view.active[kind]),
        key=lambda kind: order.index(form.fields[kind]._id),
    )
    required = [k for k in shown if modes.requires(spec, k)]
    assert shown[: len(required)] == required, (
        f"{mode_id}: media inputs in page order {shown}; the required ones {required} must lead"
    )
    if shown:
        assert modes.requires(spec, shown[0]), f"{mode_id}: the first media input is optional"


def test_mode_11_asks_for_its_audio_first_and_offers_the_image_as_an_optional_first_frame(form):
    form.pick_mode("audio_to_video")
    order = _layout_order(form.demo)
    assert order.index(form.fields["audio"]._id) < order.index(form.fields["image"]._id)
    image = form.fields["image"]
    assert form.props[image._id]["label"] == "Image (optional)"
    assert form.props[form.fields["audio"]._id]["label"] == "Audio"
    caption = render.media_caption(modes.get_mode("audio_to_video"), "image")
    assert caption.startswith("Optional.") and "first frame" in caption


def test_the_media_order_follows_the_registry_and_refuses_a_contradiction():
    from types import SimpleNamespace

    def spec(inputs, optional=()):
        return SimpleNamespace(inputs=inputs, optional_inputs=optional)

    assert handlers.media_order() == ("audio", "video", "image")
    assert handlers.media_order([spec(("image",), ("video",))]).index("image") < (
        handlers.media_order([spec(("image",), ("video",))]).index("video")
    )
    with pytest.raises(RuntimeError, match="no single order"):
        handlers.media_order([spec(("image",), ("video",)), spec(("video",), ("image",))])


def test_no_css_reorders_the_media_inputs():
    """DOM order = visual order = focus order: no ``order`` on a media input or its lever."""
    import re

    css = chrome._css()
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if re.search(r"#(in-image|in-video|in-audio|lever-strength|lever-noise)\b", selector):
            assert not re.search(r"(^|[;\s])order\s*:", body), selector


def test_the_endpoints_field_order_is_unchanged():
    assert FIELD_KEYS == (
        "prompt",
        "negative_prompt",
        "image",
        "video",
        "audio",
        "conditions",
        "width",
        "height",
        "auto_duration",
        "num_frames",
        "min_seconds",
        "max_seconds",
        "enable_stage_2",
        "seed",
        "strength",
        "noise_scale",
        "loras",
        "enhance_prompt",
        "decoder",
    )


# --------------------------------------------------------------------------
# minor 9: the length slider keeps to its own step
# --------------------------------------------------------------------------


def test_the_seconds_box_reads_to_its_step(form):
    seconds = form.by_id("seconds")
    assert seconds.step == 0.1 and form.get(seconds) == 5.0
    for frames, shown in ((121, 5.0), (241, 10.0), (113, 4.7)):
        assert handlers.seconds_for_frames(frames) == shown
        assert handlers.frames_for_seconds(shown) == frames
    form.user(seconds, 10.0)
    assert form.field("num_frames") == 241 and form.get(seconds) == 10.0


# --------------------------------------------------------------------------
# minor 11: a clip's frame count never follows the user into a text mode
# --------------------------------------------------------------------------


def test_match_my_clip_does_not_hand_its_frame_count_to_the_next_mode(form, tmp_path):
    from fractions import Fraction

    import av
    import numpy as np

    clip = tmp_path / "three-seconds.mp4"
    with av.open(str(clip), "w") as container:
        stream = container.add_stream("h264", rate=24)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(72):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    form.pick_mode("video_to_audio")
    form.user(form.fields["video"], str(clip))
    matched = handlers.frames_matching_clip(handlers.clip_seconds(str(clip)))
    assert form.field("num_frames") == matched != 121
    form.pick_mode("text_to_audio_video")
    assert form.field("num_frames") == 121, "the clip's frame count followed into a text mode"
    form.user(form.by_id("seconds"), 4.0)
    form.pick_mode("video_to_audio")
    form.pick_mode("text_to_video")
    assert form.field("num_frames") == handlers.frames_for_seconds(4.0), "the user's own length"


# --------------------------------------------------------------------------
# minor 12: what Advanced shows is what counts
# --------------------------------------------------------------------------


def test_advanced_shows_frames_or_the_model_pick_range_never_both(form):
    frames, low, high = (form.fields[k] for k in ("num_frames", "min_seconds", "max_seconds"))
    form.pick_mode("text_to_video")
    assert form.visible(frames) and not form.visible(low) and not form.visible(high)
    form.user(form.fields["auto_duration"], True)
    assert not form.visible(frames) and form.visible(low) and form.visible(high)
    form.pick_mode("audio_to_audio")
    assert not form.visible(low) and not form.visible(high)
    form.pick_mode("text_to_audio_video")
    assert form.field("auto_duration") is True, "the user's own toggle came back"
    assert not form.visible(frames) and form.visible(low)


def test_a_hidden_size_or_range_is_neither_validated_nor_sent(form, wired):
    form.pick_mode("text_to_video")
    form.user(form.fields["width"], 700)
    assert (
        handlers.review(
            "text_to_video", dict(zip(FIELD_KEYS, form.form_values(), strict=True))
        ).state
        == "refused"
    )
    form.pick_mode("text_to_audio")  # sizes are pinned: the box is hidden
    assert form.field("width") == 704, "a hidden, illegal width would be validated and sent"
    form.user(form.fields["prompt"], "rain on a tin roof")
    facts = handlers.review("text_to_audio", dict(zip(FIELD_KEYS, form.form_values(), strict=True)))
    assert facts.state == "ready", facts.problems
    form.pick_mode("text_to_video")
    assert form.field("width") == 700, "the user's own width came back"

    form.user(form.fields["auto_duration"], True)
    form.user(form.fields["max_seconds"], 1.0)  # shorter than the shortest: refused
    form.pick_mode("audio_to_video")
    assert (form.field("min_seconds"), form.field("max_seconds")) == (2.0, 5.0)
    form.pick_mode("text_to_video")
    assert form.field("max_seconds") == 1.0


# --------------------------------------------------------------------------
# minor 13: a request that won't run lights no cost lamp
# --------------------------------------------------------------------------


def test_the_refused_state_lights_no_cost_lamp(tmp_path):
    from lora_fixtures import good_adapter

    adapter = good_adapter(tmp_path)
    values = (
        {spec.key: spec.default for spec in FIELDS}
        | inputs_for("text_to_video")
        | {
            "width": 700,
            "enable_stage_2": True,
            "loras": [{"source": "org/a", "path": str(adapter), "weight": 1.0, "content_key": "k"}],
        }
    )
    panel = render.cost_panel(handlers.review("text_to_video", values))
    assert "Won’t run" in panel
    assert "mr-lamp-cost is-lit" not in panel, "a lamp is lit for a run that costs nothing"
    ready = render.cost_panel(handlers.review("text_to_video", values | {"width": 704}))
    assert ready.count("mr-lamp-cost is-lit") == 2, "Upscale and LoRA light once it would run"
