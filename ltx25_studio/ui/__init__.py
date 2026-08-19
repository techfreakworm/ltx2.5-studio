"""Gradio layer. May import anything; nothing else may import it (PLAN §7.2).

``build_ui`` is resolved lazily so that importing this package does not import
Gradio. That keeps ``fields.py``, ``render.py`` and ``handlers.py`` -- which
hold every decision this layer makes -- readable and testable on a bare
interpreter, which is the point of §7.2's "no business logic in handlers" rule.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_ui"]


def __getattr__(name: str) -> Any:
    if name == "build_ui":
        from .blocks import build_ui

        return build_ui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
