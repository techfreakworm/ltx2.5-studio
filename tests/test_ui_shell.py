"""The shell's rules and rail script after review round 4 (M4's desktop half, M2 and the minors
that live in ``chrome.py``).

The geometry is measured in the browser by the round-4 rig; these tests pin the rules and the
wiring it depends on.
"""

from __future__ import annotations

import re

import pytest

from ltx25_studio.ui import chrome, render


def test_the_example_jump_shows_at_every_width_while_the_monitor_holds_a_run_from_another_mode():
    """M4: the jump to a mode's sample is hidden at 1024 px and wider, where the monitor sits beside
    the form, except while the monitor holds a run from another mode: then the behaviour gives it
    ``mr-over-run`` and it is the only way to see that mode's sample."""
    css = chrome._css()
    assert "@media (min-width: 1024px) { #sample-jump { display: none !important; } }" in css
    shown = re.search(r"@media \(min-width: 1024px\) \{ #sample-jump\.mr-over-run \{([^}]*)\} \}", css)
    assert shown, "no desktop rule for the jump over a run"
    assert "display: flex !important" in shown.group(1), "the button's own display (Gradio's flex)"


def _function(script: str, opening: str) -> str:
    start = script.index(opening)
    return script[start : script.index("\n  };\n", start)]


def test_the_figure_line_is_decided_on_every_rail_scroll_from_where_the_figure_row_is():
    """M2: the line used to be decided once, from the rail's layout at scroll 0, so a scrolled
    rail left a stuck Generate with no cost beside it; and it showed while the Needs row was in
    view whenever the panel's outer box ran a few px under Generate (layout m1)."""
    script = chrome.head_html()
    shade = _function(script, "const shade = () => {")
    assert "figureLine(rail, commit, cost);" in shade, "shade() runs on every rail scroll"
    assert "rail.addEventListener('scroll', shade, { passive: true })" in script
    fit = _function(script, "const fit = () => {")
    assert "rail.classList.toggle('mr-figure-under'" not in fit, "not from the scroll-0 sum any more"
    rule = _function(script, "const figureLine = (rail, commit, cost) => {")
    for words in (
        # the row the line repeats, wholly visible clear of Generate, or the line shows
        "FIGURE_ROWS.includes(dt.textContent.trim())",
        "top >= r.top + slack && bottom <= r.bottom - slack",
        "(bottom <= kTop - slack || top >= kBottom + slack)",
        "const want = !clear;",
        "rail.classList.toggle('mr-figure-under', want);",
        # judged from Generate's box without the line; a shown line goes only past the slack
        "const kTop = footed ? k.top + lineH : k.top, kBottom = footed ? k.bottom : k.bottom - lineH;",
        "const slack = shown ? FIGURE_SLACK : 0;",
        # off the rail's foot an arriving line takes its room from above: the button and the
        # report stay, and a rail at its end stays at its end
        "if (want && !footed && button) rail.scrollTop += button.getBoundingClientRect().top - before;",
    ):
        assert words in rule, words
    assert "const FIGURE_ROWS = ['Needs', 'Runs'];" in script
    assert "const FIGURE_SLACK = 6;" in script


@pytest.mark.parametrize("on_zerogpu, label", [(True, "Needs"), (False, "Runs")])
def test_the_rows_the_script_looks_for_are_the_cost_panels_own(on_zerogpu, label):
    facts = render.CostFacts(
        mode_name="Text to video with sound",
        state="ready",
        render_size=(704, 480),
        frames=121,
        seconds=121 / 24,
        reserve=180,
        multiplier=2,
        on_zerogpu=on_zerogpu,
        device="cpu",
    )
    assert f"<dt>{label}</dt>" in render.cost_panel(facts)
    assert f"'{label}'" in chrome.head_html()


def test_restore_brings_the_cost_into_view_on_desktop():
    """M2: Restore settings, the click built to change the settings, rewrote the cost out of
    sight on desktop (its own script brings in the purpose line, already in view there)."""
    script = chrome.head_html()
    restore = script[script.index("const RESTORE = ") :]
    restore = restore[: restore.index("\n  });\n")]
    assert "const RESTORE = 'Restore settings';" in restore
    assert "closest('#recent button')" in restore and "!wide()" in restore
    assert "byId('cost')?.scrollIntoView({ block: 'nearest'" in restore
    gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")
    from ltx25_studio.ui import blocks

    demo = blocks.build_ui()
    labels = [b.value for b in demo.blocks.values() if isinstance(b, gr.Button)]
    assert "Restore settings" in labels, "the script finds the button by its words"
