"""Every video player on the page is usable without a hovering pointer (review round 4, M3).

Gradio's video player (6.24.0, pinned) shows its control bar to a hovering pointer only
(``.controls{opacity:0}``, shown by ``.wrap:hover``), names Play "play-pause-replay-button" and
toggles playback on ANY key pressed on it, Tab included. The page shows two of them: the
monitor's (``#video-out``) and a loaded clip's preview (``#in-video``). The app does not patch
Gradio: the stylesheet shows a player's bar while keyboard focus is inside it, while its video is
paused and always on a touch screen, in the well colour; the script names the controls and lets
only Enter and Space act on them. Folded in: the corner buttons' focus rings (P17) and the glyphs
inside the 44 px phone targets (N7).

The geometry, the pixels, the accessibility snapshot and the keys are checked in the browser by
the round-4 rig (M3_*, M3_in_video_*); these tests pin the rules and the wiring they depend on.
"""

from __future__ import annotations

import re

import pytest

from ltx25_studio.ui import chrome

WELL = "rgba(14, 18, 22, 0.86)"
#: Gradio's video player, wherever it is: its root holds the picture (.mirror-wrap) and the bar
PLAYER = ".wrap:has(> .mirror-wrap)"


def _block(css: str, opening: str) -> str:
    start = css.index(opening)
    return css[start : css.index("\n}\n", start)]


def _rule(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r" \{([^}]*)\}", css)
    assert match, f"no rule for {selector!r}"
    return match.group(1)


def test_every_players_bar_is_painted_in_the_well_not_gradios_grey():
    rule = _rule(chrome._css(), f"{PLAYER} > .controls")
    assert f"background: {WELL} !important" in rule


def test_every_players_bar_shows_to_keyboard_focus_and_while_its_video_is_paused():
    rule = _rule(
        chrome._css(),
        f"{PLAYER}:has(:focus-visible) > .controls, .wrap.mr-paused > .controls",
    )
    assert "opacity: 1 !important" in rule


def test_every_players_bar_always_shows_on_a_touch_screen():
    touch = _block(chrome._css(), "@media (hover: none) {")
    assert f"{PLAYER} > .controls {{ opacity: 1 !important; }}" in touch


def test_the_script_tracks_the_pause_and_names_the_controls_of_every_player():
    script = chrome.head_html()
    for words in (
        "wrap.classList.toggle('mr-paused', paused)",
        "const name = paused ? 'Play' : 'Pause';",
        "full.setAttribute('aria-label', 'Full screen')",
        "['play', 'pause', 'ended', 'emptied', 'loadedmetadata'].forEach",
        "video.closest('.wrap')",
        # every Gradio player on the page, not only the monitor's
        "document.querySelectorAll('.mirror-wrap > video')",
    ):
        assert words in script, words
    assert "#video-out video" not in script
    ready = script[script.index("ready(() => {") :]
    assert "players();" in ready, "every new player (a new file on the monitor, a clip loaded) is wired"


def test_only_enter_and_space_act_on_any_players_controls():
    script = chrome.head_html()
    # Gradio's Play toggles on any keydown (Tab included) and its full screen on any keypress
    # (delegated or on the element): a capture listener on the document runs before either
    assert "document.addEventListener('keydown', playerKeys, true);" in script
    assert "document.addEventListener('keypress', playerKeys, true);" in script
    keys = script[script.index("const playerKeys = (event) => {") :]
    keys = keys[: keys.index("};")]
    assert "event.target.closest('[data-mr-control]')" in keys, "whichever player the control is in"
    assert "(event.key === 'Enter' || event.key === ' ') && !event.repeat" in keys
    assert "event.stopPropagation();" in keys, "any other key never reaches Gradio's handler"
    assert keys.count("event.preventDefault();") == 2, "Space never scrolls under the bar"
    assert "control.click()" in keys, "Space still presses full screen (its keypress never comes)"


def test_a_players_corner_buttons_are_inset_so_their_focus_rings_are_whole():
    """P17: the monitor and a media zone clip at their edges; the ring is 4 px outside."""
    rule = _rule(chrome._css(), ":is(#video-out, #in-video) .icon-button-wrapper")
    assert "top: 8px !important" in rule and "right: 8px !important" in rule


def test_the_input_clips_bar_keeps_the_players_white_glyphs():
    """The media zone dims its own buttons to dust; that rule reached the clip's volume button,
    which then stood dust between a white Play and a white Full screen."""
    rule = _rule(chrome._css(), "#commission #in-video .wrap > .controls button")
    assert "color: #fff !important" in rule


def test_on_a_phone_every_players_controls_are_44_px_targets_and_their_glyphs_keep_their_size():
    narrow = _block(chrome._css(), "@media (max-width: 1023px) {")
    targets = _rule(narrow, f'{PLAYER} > .controls :is(button, [role="button"])')
    assert "min-width: 44px !important" in targets and "min-height: 44px !important" in targets
    glyphs = _rule(narrow, f'{PLAYER} > .controls :is(button, [role="button"]) svg')
    assert "width: 20px !important" in glyphs and "height: 20px !important" in glyphs


def test_the_two_players_the_inset_names_are_every_video_the_page_builds():
    """The bar's rules and the script find a player by Gradio's own markup; the corner inset and
    the input clip's glyph colour name the two blocks. A third gr.Video must be added to them."""
    gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")
    from ltx25_studio.ui import blocks

    videos = {b.elem_id for b in blocks.build_ui().blocks.values() if isinstance(b, gr.Video)}
    assert videos == {"video-out", "in-video"}
