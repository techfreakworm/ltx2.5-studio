"""The monitor's player is usable without a hovering pointer (review round 4, M3).

Gradio's video player (6.24.0, pinned) shows its control bar to a hovering pointer only
(``.controls{opacity:0}``, shown by ``.wrap:hover``), names Play "play-pause-replay-button" and
toggles playback on ANY key pressed on it, Tab included. The app does not patch Gradio: the
stylesheet shows the bar while keyboard focus is inside the player, while the video is paused
and always on a touch screen, in the well colour; the script names the controls and lets only
Enter and Space act on them. Folded in: the download corner's focus ring (P17) and the glyphs
inside the 44 px phone targets (N7).

The geometry, the pixels, the accessibility snapshot and the keys are checked in the browser by
the round-4 rig (M3_*); these tests pin the rules and the wiring they depend on.
"""

from __future__ import annotations

import re

from ltx25_studio.ui import chrome

WELL = "rgba(14, 18, 22, 0.86)"


def _block(css: str, opening: str) -> str:
    start = css.index(opening)
    return css[start : css.index("\n}\n", start)]


def _rule(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r" \{([^}]*)\}", css)
    assert match, f"no rule for {selector!r}"
    return match.group(1)


def test_the_bar_is_painted_in_the_well_not_gradios_grey():
    rule = _rule(chrome._css(), "#video-out .wrap > .controls")
    assert f"background: {WELL} !important" in rule


def test_the_bar_shows_to_keyboard_focus_and_while_the_video_is_paused():
    rule = _rule(
        chrome._css(),
        "#video-out .wrap:has(:focus-visible) > .controls, #video-out .wrap.mr-paused > .controls",
    )
    assert "opacity: 1 !important" in rule


def test_the_bar_always_shows_on_a_touch_screen():
    css = chrome._css()
    touch = _block(css, "@media (hover: none) {")
    assert "#video-out .wrap > .controls { opacity: 1 !important; }" in touch


def test_the_script_tracks_the_pause_and_names_the_controls_from_the_videos_own_events():
    script = chrome.head_html()
    for words in (
        "wrap.classList.toggle('mr-paused', paused)",
        "const name = paused ? 'Play' : 'Pause';",
        "full.setAttribute('aria-label', 'Full screen')",
        "['play', 'pause', 'ended', 'emptied', 'loadedmetadata'].forEach",
        "video.closest('.wrap')",
    ):
        assert words in script, words
    ready = script[script.index("ready(() => {") :]
    assert "players();" in ready, "every new player (a new file on the monitor) is wired"


def test_only_enter_and_space_act_on_the_bars_controls():
    script = chrome.head_html()
    # Gradio's Play toggles on any keydown (Tab included) and its full screen on any keypress
    # (delegated or on the element): a capture listener on the document runs before either
    assert "document.addEventListener('keydown', playerKeys, true);" in script
    assert "document.addEventListener('keypress', playerKeys, true);" in script
    keys = script[script.index("const playerKeys = (event) => {") :]
    keys = keys[: keys.index("};")]
    assert "(event.key === 'Enter' || event.key === ' ') && !event.repeat" in keys
    assert "event.stopPropagation();" in keys, "any other key never reaches Gradio's handler"
    assert keys.count("event.preventDefault();") == 2, "Space never scrolls under the bar"
    assert "control.click()" in keys, "Space still presses full screen (its keypress never comes)"


def test_the_download_corner_is_inset_so_its_focus_ring_is_whole():
    """P17: the monitor (and the video block) clip at their edges; the ring is 4 px outside."""
    rule = _rule(chrome._css(), "#video-out .icon-button-wrapper")
    assert "top: 8px !important" in rule and "right: 8px !important" in rule


def test_on_a_phone_the_bars_controls_are_44_px_targets_and_their_glyphs_keep_their_size():
    narrow = _block(chrome._css(), "@media (max-width: 1023px) {")
    targets = re.search(r"#monitor button, ([^{]*)\{([^}]*)\}", narrow)
    assert targets and '#monitor .controls [role="button"]' in targets.group(1)
    assert "min-width: 44px !important" in targets.group(2)
    assert "min-height: 44px !important" in targets.group(2)
    glyphs = _rule(narrow, '#monitor .controls :is(button, [role="button"]) svg')
    assert "width: 20px !important" in glyphs and "height: 20px !important" in glyphs
