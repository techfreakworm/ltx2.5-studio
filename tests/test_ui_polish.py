"""Review round 3's polish, bounded to P1, P3, P6, P9, P12, P13 (P16 is in test_ui_rail.py).

The pixels and the geometry are measured in the browser by the capture rig (C, P3, P6, P9,
P12, P13, P16); these tests pin the rules, the words and the wiring they depend on.
"""

from __future__ import annotations

import re

import pytest
from conftest import inputs_for

from ltx25_studio import modes
from ltx25_studio.ui import blocks, chrome, handlers, render
from ltx25_studio.ui.fields import FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

NBSP = " "


def narrow_rules(css: str) -> str:
    start = css.index("@media (max-width: 1023px) {")
    return css[start : css.index("\n}\n", start)]


# P1: Gradio's "Runs" link stays hidden below 1024 px too
def test_the_footers_run_history_stays_hidden_on_narrow_screens():
    css = chrome._css()
    assert "footer .run-history, footer .run-history + .divider { display: none !important; }" in css
    narrow = narrow_rules(css)
    shown = re.search(r"footer :is\(button, a\)([^{]*)\{", narrow).group(1)
    assert ":not(.run-history)" in shown, "the narrow rule (0,1,2) re-showed the run history"


# P3: locked decoder chips are drawn as locked
def test_a_chip_that_cannot_be_changed_is_drawn_locked():
    css = chrome._css()
    rule = re.search(r"\.mr-chips label:has\(input:disabled\) \{([^}]*)\}", css).group(1)
    for words in ("background: transparent !important", "border-style: dashed !important", "color: var(--mr-dust) !important"):
        assert words in rule, words
    checked = css.index(".mr-chips label.selected, .mr-chips label:has(input:checked)")
    assert css.index(".mr-chips label:has(input:disabled)") > checked, "it must win over 'checked'"


def test_under_upscale_the_decoder_chips_are_locked():
    from test_ui_form import Form

    form = Form(blocks.build_ui())
    form.pick_mode("text_to_video")
    decoder = form.fields["decoder"]
    assert form.interactive(decoder)
    form.user(form.fields["enable_stage_2"], True)
    if form.field("enable_stage_2"):  # off CUDA at this size Upscale is parked, unticked
        assert not form.interactive(decoder)


# P6: the empty audio zone is as tall as the image and video zones; its copy is balanced
def test_the_empty_audio_zone_is_200_px_and_its_copy_balanced():
    from ltx25_studio.ui import components

    css = chrome._css()
    assert components.MEDIA_HEIGHT == 200
    assert "#commission #in-audio .audio-container > button { height: 148px !important;" in css
    assert "#commission #in-audio .audio-container > button .wrap { min-height: 0 !important; }" in css
    drop = re.search(r"\.wrap:has\(> \.or\)::after \{([^}]*)\}", css).group(1)
    assert "text-wrap: balance" in drop


# P9: sizes, units and seeds do not break across lines
SPLIT_FIGURES = re.compile(r"\d ×|× \d|\d s\b|seed \d|\d frames")


def _figures_hold(text: str) -> list[str]:
    return SPLIT_FIGURES.findall(re.sub(r"<[^>]+>", "", text))


def test_sizes_units_and_seeds_carry_no_break_spaces():
    mode = modes.get_mode("text_to_audio_video")
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_audio_video")
    facts = handlers.review("text_to_audio_video", values)
    run = render.RunFacts(
        mode=mode, seed=2577797786, returned_size=(704, 480), seconds=2.04, stage_2=False,
        loras=(), took=3.0, took_is_gpu=True, device="cpu",
    )
    texts = {
        "cost panel": render.cost_panel(facts),
        "figure line": render.figure_line(facts),
        "done card": render.run_summary(run),
        "took": render.took_words(run),
        "recent run": render.recent_run_label(mode=mode, seed=2577797786, size=(704, 480), seconds=2.0, at="20:17"),
        "your run": render.monitor_caption(kind="run", mode=mode, seed=12),
        "earlier run": render.monitor_caption(kind="earlier", mode=mode, seed=12),
        "sample": render.monitor_caption(kind="sample", mode=mode, seed=12, settings={"size": "704 × 480", "seconds": 5}),
        "matches clip": render.match_clip_line(clip_seconds=2.0, frames=49, fps=24, audio=False, cap_seconds=20),
        "pinned": render.pinned_line(mode, 256),
    }
    for where, text in texts.items():
        assert _figures_hold(text) == [], (where, text)
    assert f"704{NBSP}×{NBSP}480" in texts["recent run"] and f"seed{NBSP}2577797786" in texts["recent run"]
    assert f"2.0{NBSP}s" in texts["recent run"]


def test_the_space_panel_figures_hold_together_too(monkeypatch):
    import sys
    import types

    from ltx25_studio import config

    module = types.ModuleType("spaces")
    module.GPU = lambda *a, **k: lambda f: f
    monkeypatch.setitem(sys.modules, "spaces", module)
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_audio_video")
    facts = handlers.review("text_to_audio_video", values | {"enable_stage_2": True})
    panel = render.cost_panel(facts)
    assert _figures_hold(panel) == [], panel
    assert _figures_hold(render.commit_line(facts)) == []


# P12: the Upscale label names a returned size only for a size the model takes
@pytest.mark.parametrize(
    ("width", "height", "label"),
    [
        (704, 480, f"Upscale ×2: returns 1408{NBSP}×{NBSP}960"),
        (700, 480, "Upscale ×2"),
        (704, 0, "Upscale ×2"),
        (2048, 480, "Upscale ×2"),
        (None, 480, "Upscale ×2"),
        ("x", 480, "Upscale ×2"),
    ],
)
def test_the_upscale_label_gives_a_size_only_for_a_legal_one(width, height, label):
    assert blocks._upscale_label(width, height) == label


def test_the_page_drops_the_upscale_size_at_an_illegal_width():
    from test_ui_form import Form

    form = Form(blocks.build_ui())
    upscale = form.fields["enable_stage_2"]
    form.pick_mode("text_to_video")
    form.user(form.fields["width"], 700)
    assert form.props[upscale._id]["label"] == "Upscale ×2"
    form.user(form.fields["width"], 704)
    assert form.props[upscale._id]["label"].startswith("Upscale ×2: returns 1408")


# P13: one "Running" line; the caption says whose run it is
def test_while_running_the_caption_names_the_run_and_the_card_its_state():
    demo = blocks.build_ui()
    run = next(f for f in demo.fns.values() if f.api_name == "on_generate")
    start = demo.fns[run.trigger_after]
    caption = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "monitor-caption")
    status = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "run-status")
    out = start.fn("text_to_audio_video", False, "a harbour at dawn")
    said_caption = out[start.outputs.index(caption)]
    said_status = out[start.outputs.index(status)]["value"]
    label = modes.get_mode("text_to_audio_video").label
    assert said_caption == f"**Your run.** {label}."
    assert (said_caption + said_status).count("Running") == 1
    assert said_status.startswith(f"**Running: {label}.**")
