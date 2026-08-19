"""Orchestration: mode -> stage 1 -> optional stage 2 -> outputs.

PLAN §5.3 lives here. This module owns the never-return-an-empty-output
invariant: any generation that completes without producing a file raises a
specific, actionable error. It never returns a falsy artifact path.

That is not a hypothetical. The predecessor's outage hid for a week because a
mid-graph load failure returned an empty video path and the UI rendered it as a
silent form reset.

Must not import from ``ltx25_studio.ui`` (§7.2).
"""
