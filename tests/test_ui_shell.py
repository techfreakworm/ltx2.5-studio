"""The shell's rules and rail script after review round 4 (M4's desktop half, M2 and the minors
that live in ``chrome.py``).

The geometry is measured in the browser by the round-4 rig; these tests pin the rules and the
wiring it depends on.
"""

from __future__ import annotations

import re

from ltx25_studio.ui import chrome


def test_the_example_jump_shows_at_every_width_while_the_monitor_holds_a_run_from_another_mode():
    """M4: the jump to a mode's sample is hidden at 1024 px and wider, where the monitor sits beside
    the form, except while the monitor holds a run from another mode: then the behaviour gives it
    ``mr-over-run`` and it is the only way to see that mode's sample."""
    css = chrome._css()
    assert "@media (min-width: 1024px) { #sample-jump { display: none !important; } }" in css
    shown = re.search(r"@media \(min-width: 1024px\) \{ #sample-jump\.mr-over-run \{([^}]*)\} \}", css)
    assert shown, "no desktop rule for the jump over a run"
    assert "display: flex !important" in shown.group(1), "the button's own display (Gradio's flex)"
