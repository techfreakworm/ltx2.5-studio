"""Reusable widgets: the condition list, the LoRA picker, constraint hints.

PLAN §3.1(a) — one condition list serves both image and video conditioning, each
entry carrying a source, an ``index`` and a ``strength``; §3.3 builds multishot
on the same list. §5.5 — dimensions snap to /32 and the snapped value is shown,
never silently substituted.
"""
