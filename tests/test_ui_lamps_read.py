"""A lamp's state is not in colour alone (review round 4, minor 12, the panel's part).

The cost lamps marked lit and unlit with a class only, and named their row on a plain ``div``,
which assistive tech skips: a screen-reader user heard every lamp as if it applied, and off
ZeroGPU a lit and an unlit lamp carry the same words ("Upscale", "LoRA"). Now an unlit lamp,
a cost that does not apply, is hidden from assistive tech, and the row is a named group, so
what is read is the costs that apply.
"""

from __future__ import annotations

import re

from conftest import inputs_for

from ltx25_studio.ui import handlers, render
from ltx25_studio.ui.fields import FIELDS


def lamps(panel: str) -> list[tuple[str, bool, bool]]:
    """``(words, lit, hidden from assistive tech)`` per cost lamp."""
    found = re.findall(r'<span class="mr-lamp mr-lamp-cost (is-lit|is-off)"([^>]*)>([^<]*)<', panel)
    return [(words, state == "is-lit", 'aria-hidden="true"' in rest) for state, rest, words in found]


def test_only_the_costs_that_apply_are_read_and_the_row_is_a_named_group():
    values = {spec.key: spec.default for spec in FIELDS} | inputs_for("text_to_audio_video")
    plain = render.cost_panel(handlers.review("text_to_audio_video", values))
    assert '<div class="mr-lamps" role="group" aria-label="Costs that apply">' in plain
    found = lamps(plain)
    assert found and all(not lit and hidden for _words, lit, hidden in found), found

    lit_values = values | {"enhance_prompt": True, "prompt": "a harbour at dawn"}
    enhanced = lamps(render.cost_panel(handlers.review("text_to_audio_video", lit_values)))
    assert [w for w, lit, hidden in enhanced if lit and not hidden][0].startswith("Enhance")
    for _words, lit, hidden in enhanced:
        assert hidden is not lit, "a lit lamp is read; an unlit one is not"
