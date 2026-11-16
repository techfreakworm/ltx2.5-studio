"""The staged-adapter entry: what the UI keeps between Validate and Generate.

One dict per adapter, in ``GenerationRequest.loras``, crossing the ZeroGPU fork
by pickle. Built with :func:`staged_entry`; keys:

``source``
    What the user typed or uploaded; the display name.
``path``
    The local file the GPU worker loads (the fetched file for a Hub id).
``weight``
    LoRA strength, 0 < weight <= 2, default 1.0.
``report``
    The :class:`~ltx25_studio.lora.validate.LoRAValidation` that accepted it.
``content_key``
    sha256 of the whole adapter file (the Hub's LFS hash); the same adapter has
    the same key under any file name, so a second copy can be refused, while two
    adapters that share a layout but not their weights have different keys.

Stdlib only.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

from ..errors import LoRAError
from .fetch import mark_used

__all__ = [
    "DEFAULT_WEIGHT",
    "MAX_WEIGHT",
    "LoRAWeightError",
    "StagedLoRA",
    "check_weight",
    "find_duplicate",
    "staged_entry",
]

DEFAULT_WEIGHT = 1.0
MAX_WEIGHT = 2.0


class LoRAWeightError(LoRAError):
    """The requested LoRA strength is outside (0, 2]."""

    def __init__(self, *, source, weight):
        self.source = source
        self.weight = weight
        super().__init__(
            f"LoRA strength {weight!r} for {source} is out of range: it must be more than 0 "
            f"and at most {MAX_WEIGHT:g}. Most adapters are made for a strength between 0.5 "
            f"and 1.2."
        )


class StagedLoRA(TypedDict):
    source: str
    path: str
    weight: float
    report: Any
    content_key: str


def check_weight(weight, *, source="the adapter") -> float:
    """``weight`` as a float in (0, 2], or :class:`LoRAWeightError`."""
    try:
        value = float(weight)
    except (TypeError, ValueError):
        raise LoRAWeightError(source=source, weight=weight) from None
    if not math.isfinite(value) or not 0.0 < value <= MAX_WEIGHT:
        raise LoRAWeightError(source=source, weight=weight)
    return value


def staged_entry(source, report, *, weight=DEFAULT_WEIGHT) -> StagedLoRA:
    """The entry to append to the staged list for an accepted adapter.

    Stamps the adapter in use, so the LoRA cache does not evict it while the
    session still has it staged.
    """
    entry = StagedLoRA(
        source=str(source),
        path=str(report.path),
        weight=check_weight(weight, source=source),
        report=report,
        content_key=str(getattr(report, "content_key", "") or ""),
    )
    mark_used(entry["path"])
    return entry


def find_duplicate(staged, entry) -> dict | None:
    """The already-staged entry holding the same adapter as ``entry``, if any."""
    key = entry.get("content_key")
    if not key:
        return None
    for existing in staged or ():
        if isinstance(existing, dict) and existing.get("content_key") == key:
            return existing
    return None
