"""The desktop rail on short windows (review round 3, R1 and R2).

At 1366 x 657 and 1280 x 680 the stuck Generate covered the cost panel's Reserves and Needs
rows, with a 4 px band of covered text showing under it; a failure card landed out of sight;
Show changed a monitor scrolled away. The geometry and the pixels are checked in the browser
by the capture rig (R1a-R1d, R2_*); these tests pin the markup, the stylesheet, the script and
the wiring it depends on, and the words of the figure line.
"""

from __future__ import annotations

import re

import pytest

from ltx25_studio import modes
from ltx25_studio.ui import blocks, chrome, handlers, render

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


@pytest.fixture(scope="module")
def demo():
    return blocks.build_ui()


def desktop_rules(css: str) -> str:
    start = css.index("@media (min-width: 1024px) {\n  /* Sized to fit the viewport")
    return css[start : css.index("\n}\n", start)]


def test_generate_is_sticky_at_both_edges_flush_with_the_rails_own_edges():
    rules = desktop_rules(chrome._css())
    commit = re.search(r"#rail > #commit \{([^}]*)\}", rules).group(1)
    assert "position: sticky" in commit and "top: -4px" in commit and "bottom: -4px" in commit
    assert "padding: 8px 0 8px" in commit, "the band under the button is the button's own ink"


def test_a_row_passing_under_generate_runs_into_an_ink_fade():
    rules = desktop_rules(chrome._css())
    for edge, pseudo, direction in (("top", "::before", "to bottom"), ("bottom", "::after", "to top")):
        rule = re.search(rf"\n  #rail > #commit{pseudo} \{{([^}}]*)\}}", rules).group(1)
        assert f"{edge}: -12px" in rule and f"linear-gradient({direction}" in rule, pseudo
    shared = re.search(r"#rail > #commit::before, #rail > #commit::after \{([^}]*)\}", rules).group(1)
    assert "height: 12px" in shared and "pointer-events: none" in shared and "display: none" in shared
    assert "#rail.mr-under-commit > #commit::before, #rail.mr-over-commit > #commit::after" in rules


def test_the_monitor_gives_up_height_and_the_figure_stands_above_generate():
    rules = desktop_rules(chrome._css())
    assert "#rail.mr-monitor-fit > #monitor { aspect-ratio: auto; height: var(--mr-monitor-h); }" in rules
    assert "#rail.mr-figure-under #commit-line { display: block; }" in rules
    assert "#commit-line .mr-commit-phone { display: none; }" in rules
    assert "#commit-line .mr-commit-desk { display: none; }" in chrome._css(), "the phone shows its own"
    assert "min-height: var(--mr-caption-min, 38px)" in rules


def test_the_script_fits_the_rail_shades_its_edges_and_follows_the_report():
    script = chrome.head_html()
    for words in (
        "const MONITOR_FLOOR = 180;",
        "'mr-monitor-fit'",
        "'mr-figure-under'",
        "'mr-under-commit'",
        "'mr-over-commit'",
        "--mr-rail-offset",
        "Math.max(12, top - window.scrollY)",
        "scrollIntoView({ block: 'nearest'",
        "rail.addEventListener('scroll', shade, { passive: true })",
    ):
        assert words in script, words
    rules = desktop_rules(chrome._css())
    assert "scroll-margin-top: calc(var(--mr-commit-h, 80px) + 12px)" in rules


@pytest.mark.parametrize("on_zerogpu", [True, False])
def test_the_figure_line_is_the_panels_figure_in_one_line(on_zerogpu):
    facts = render.CostFacts(
        mode_name="Text to video with sound",
        state="incomplete",
        render_size=(704, 480),
        frames=121,
        seconds=121 / 24,
        reserve=180,
        multiplier=2,
        missing=("text",),
        on_zerogpu=on_zerogpu,
        device="cpu",
    )
    figure = render.figure_line(facts).replace("\u00a0", " ")
    tail = "needs 360 s quota" if on_zerogpu else "on the CPU"
    assert figure == f"704 × 480, 5.0 s, {tail}"
    assert render.commit_line(facts) == "Needs a prompt", "the phone keeps its reason"
    markup = render.commit_markup(facts).replace("\u00a0", " ")
    assert '<span class="mr-commit-phone">Needs a prompt</span>' in markup
    assert f'<span class="mr-commit-desk">{figure}</span>' in markup
    refused = render.CostFacts(mode_name="x", state="refused", problems=("Width 700 won't run",))
    assert render.figure_line(refused) == "", "a request that won't run has no figure"


def test_the_review_writes_both_lines_into_the_commit(demo):
    from test_ui_form import Form

    form = Form(demo)
    form.pick_mode("text_to_video")
    form.pick_mode("text_to_audio_video")
    held = form.get(form.by_id("commit-line"))
    facts = handlers.review("text_to_audio_video", dict(zip(form.fields, form.form_values(), strict=True)))
    assert held == render.commit_markup(facts)
    assert render.figure_line(facts) in held and render.commit_line(facts) in held


def _click(demo, button):
    return next(f for f in demo.fns.values() if (button._id, "click") in (f.targets or []))


def test_generate_and_show_bring_the_monitor_into_view_at_every_width(demo):
    js = blocks._SCROLL_TO_MONITOR
    assert "innerWidth < 1024" not in js, "it used to act below 1024 px only"
    assert "block: wide ? 'nearest' : 'start'" in js and "behavior: 'instant'" in js
    generate = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "generate-btn")
    assert _click(demo, generate).js == js
    shows = [b for b in demo.blocks.values() if isinstance(b, gr.Button) and b.value == "Show"]
    assert len(shows) == blocks.RECENT_LIMIT
    for show in shows:
        assert _click(demo, show).js == js


def test_restore_brings_the_purpose_line_into_view(demo):
    restores = [
        b for b in demo.blocks.values() if isinstance(b, gr.Button) and b.value == "Restore settings"
    ]
    assert len(restores) == blocks.RECENT_LIMIT
    for restore in restores:
        assert _click(demo, restore).js == blocks._SCROLL_TO_PURPOSE
    assert "getElementById('purpose')" in blocks._SCROLL_TO_PURPOSE
    assert "wide ? 'nearest' : 'center'" in blocks._SCROLL_TO_PURPOSE


def _returns_row(facts) -> str:
    panel = re.sub(r"<[^>]+>", "", render.cost_panel(facts)).replace(" ", " ")
    return re.search(r"Returns(.*?)(?:Runs|Reserves)", panel).group(1)


def test_the_line_names_what_the_run_returns_in_every_mode(demo):
    """Round 4, M1: the line beside Generate (the phone bar's ready line too) names what the
    run returns, as the panel's Returns row does: the returned size in a video mode, and
    "Sound" in a sound-only mode, never the 256 x 256 picture it renders and throws away."""
    from conftest import inputs_for

    from ltx25_studio.ui.fields import FIELDS

    for mode_id in modes.MODES:
        values = {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id)
        facts = handlers.review(mode_id, values)
        figure = render.figure_line(facts).replace(" ", " ")
        if facts.state == "ready":
            assert render.commit_line(facts).replace(" ", " ") == figure, mode_id
        if "video" in modes.get_mode(mode_id).exports:
            size = f"{facts.returned_size[0]} × {facts.returned_size[1]}"
            assert figure.startswith(f"{size}, "), (mode_id, figure)
            assert _returns_row(facts).startswith(size), (mode_id, _returns_row(facts))
        else:
            assert figure.startswith("Sound, "), (mode_id, figure)
            assert "×" not in figure, (mode_id, figure)
            assert _returns_row(facts).startswith("sound only"), mode_id


@pytest.fixture
def space(monkeypatch):
    """The Space-shaped build: the stand-in platform package, as app.py resolves it."""
    import sys
    import types

    from ltx25_studio import config

    module = types.ModuleType("spaces")
    module.GPU = lambda *a, **k: lambda f: f
    monkeypatch.setitem(sys.modules, "spaces", module)
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)


def test_with_upscale_the_line_names_the_upscaled_size_on_the_space(space):
    from conftest import inputs_for

    from ltx25_studio.ui.fields import FIELDS

    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_audio_video")
    facts = handlers.review("text_to_audio_video", values | {"enable_stage_2": True})
    assert facts.on_zerogpu and facts.stage_2 and facts.returned_size == (1408, 960)
    figure = render.figure_line(facts).replace(" ", " ")
    need = facts.reserve * facts.multiplier
    assert figure == f"1408 × 960 upscaled, 5.0 s, needs {need} s quota"
    assert "704 × 480" not in figure
    assert _returns_row(facts).startswith("1408 × 960 video with sound, upscaled ×2")


def test_with_upscale_off_zerogpu_the_bar_names_the_upscaled_size(monkeypatch):
    """A CUDA machine of one's own runs Upscale at the default size; and the tablet and phone
    capture (P3, the tiny checkpoint on the CPU): 64 x 64 upscaled returns 128 x 128."""
    import sys

    from conftest import inputs_for

    from ltx25_studio import config, zerogpu
    from ltx25_studio.ui.fields import FIELDS

    monkeypatch.delitem(sys.modules, "spaces", raising=False)
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: True)
    assert handlers.run_platform() == ("cuda", False)
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_audio_video")
    facts = handlers.review("text_to_audio_video", values | {"enable_stage_2": True})
    assert facts.state == "ready" and facts.stage_2, facts.reason
    bar = render.commit_line(facts).replace(" ", " ")
    assert bar == "1408 × 960 upscaled, 5.0 s, on the GPU"

    p3 = render.CostFacts(
        mode_name="Text to video with sound",
        state="ready",
        render_size=(64, 64),
        returned_size=(128, 128),
        stage_2=True,
        frames=9,
        seconds=9 / 24,
        returns="audio+video",
        on_zerogpu=False,
        device="cpu",
    )
    bar = render.commit_line(p3).replace(" ", " ")
    assert bar == "128 × 128 upscaled, 0.4 s, on the CPU" and "64 × 64" not in bar


def test_the_sound_only_modes_say_sound_with_their_length(space):
    from conftest import inputs_for

    from ltx25_studio.ui.fields import FIELDS

    for mode_id in ("text_to_audio", "video_to_audio", "audio_to_audio"):
        values = {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id)
        facts = handlers.review(mode_id, values)
        figure = render.figure_line(facts).replace(" ", " ")
        need = f"needs {facts.reserve * facts.multiplier} s quota"
        assert figure.startswith("Sound, ") and figure.endswith(need), (mode_id, figure)
        assert "256" not in figure, (mode_id, figure)
    unset = render.CostFacts(
        mode_name="Audio to audio",
        state="ready",
        render_size=(256, 256),
        returns="audio",
        returned_size=(256, 256),
        length_from_audio=True,
        on_zerogpu=False,
        device="cpu",
    )
    assert render.figure_line(unset) == "Sound, the length of your audio, on the CPU"
