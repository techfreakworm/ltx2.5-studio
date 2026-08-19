"""Thin adapters: UI event -> one call into ``generate``/``lora`` -> UI update.

PLAN §7.2: no business logic lives here. Every handler marshals inputs, calls
exactly one function, and marshals outputs. That rule is what keeps the app
testable without a GPU and without Gradio.

Two things follow from taking that rule seriously, and both are deliberate:

**This module does not import Gradio.** Handlers return plain values -- paths,
strings, tuples of booleans -- and ``blocks.py`` is the only place that turns a
boolean into a ``gr.update(visible=...)``. So every decision a handler makes can
be asserted on directly, with no Blocks context and no browser.

**No handler ever returns a no-op update.** PLAN §5.3's week-long outage was a
handler returning ``gr.update()`` for an empty path -- indistinguishable, on
screen, from a form quietly resetting itself. Every return path below produces
a concrete value for every output and a **non-empty** status string. If you are
editing this file and find yourself wanting to "leave that one alone", that is
the bug reintroducing itself.

The heavy imports (``generate``, ``lora``) are deferred into the call bodies so
this module stays importable -- and the whole UI stays buildable -- without
torch, diffusers or safetensors present. Startup fail-fast is ``preflight.py``'s
job (PLAN §5.6), not an import side effect here.
"""

from __future__ import annotations

from typing import Any

from .. import constraints, modes
from . import render
from .fields import FIELD_KEYS, TOGGLEABLE_FIELDS

__all__ = [
    "export_flags",
    "on_auto_duration_toggle",
    "on_clear_loras",
    "on_generate",
    "on_generate_start",
    "on_mode_change",
    "on_shape_change",
    "on_snap",
    "on_validate_lora",
]


# ---------------------------------------------------------------------------
# input coercion
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    """Gradio numbers arrive as ``float``, or as ``None`` when the box is empty."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# mode selection
# ---------------------------------------------------------------------------


def on_mode_change(mode_id: str) -> tuple[str, tuple[bool, ...]]:
    """Briefing text plus one visibility flag per toggleable field, in spec order.

    The flags come from ``TOGGLEABLE_FIELDS``, which ``blocks.py`` also uses to
    build the ``outputs=`` list -- one tuple, read twice, so the two orderings
    cannot drift apart (see ``fields.py``).
    """
    try:
        mode = modes.get_mode(mode_id)
    except Exception as exc:  # noqa: BLE001 -- an unknown mode must still render
        return render.failure(exc), tuple(True for _ in TOGGLEABLE_FIELDS)
    return (
        render.mode_briefing(mode),
        tuple(bool(spec.visible_when(mode)) for spec in TOGGLEABLE_FIELDS),
    )


def export_flags(mode_id: str) -> tuple[bool, bool]:
    """Whether ``(video, audio)`` players apply to this mode, from the registry.

    A player left on screen for a track the mode never exports is a small lie
    that becomes a large one the moment it still holds the previous run's
    output.
    """
    try:
        exports = modes.get_mode(mode_id).exports
    except Exception:  # noqa: BLE001 -- show both rather than hide the failure
        return (True, True)
    return ("video" in exports, "audio" in exports)


# ---------------------------------------------------------------------------
# live constraint guidance -- PLAN §5.5
# ---------------------------------------------------------------------------


def on_shape_change(width: Any, height: Any, num_frames: Any) -> tuple[str, str]:
    """Warn about illegal dimensions and frame counts without changing anything.

    Guidance and mutation are kept apart on purpose: ``constraints.py`` holds
    that same split (``validate_*`` raises, ``snap_*`` fixes), and a control that
    silently rewrites what you typed is its own kind of dishonesty.
    """
    return (_dimension_hint(width, height), _frame_hint(num_frames))


def _dimension_hint(width: Any, height: Any) -> str:
    w, h = _as_int(width), _as_int(height)
    if w is None or h is None:
        return "⚠️ Width and height must both be numbers."
    return render.dimension_hint(
        w,
        h,
        multiple=constraints.DIMENSION_MULTIPLE,
        snapped=constraints.snap_dimensions(w, h),
    )


def _frame_hint(num_frames: Any) -> str:
    frames = _as_int(num_frames)
    if frames is None:
        return "⚠️ The frame count must be a number."
    ratio = constraints.DEFAULT_TEMPORAL_COMPRESSION_RATIO
    lower, upper = constraints.nearest_valid_num_frames(frames)
    return render.frame_hint(
        frames,
        ratio=ratio,
        lower=lower,
        upper=upper,
        valid=constraints.is_valid_num_frames(frames),
    )


def on_snap(width: Any, height: Any, num_frames: Any) -> tuple[int, int, int, str, str]:
    """Snap all three to the nearest legal value and re-render the guidance.

    PLAN §5.5 asks the UI to *show* the snapped value rather than substitute it
    behind the user's back, which is why snapping is a button they press and not
    something that happens on blur.
    """
    w = constraints.snap_dimension(_as_int(width) or constraints.DIMENSION_MULTIPLE)
    h = constraints.snap_dimension(_as_int(height) or constraints.DIMENSION_MULTIPLE)
    frames = constraints.snap_num_frames(_as_int(num_frames) or 1)
    return (w, h, frames, _dimension_hint(w, h), _frame_hint(frames))


def on_auto_duration_toggle(auto: Any) -> tuple[bool, bool, bool]:
    """Visibility for (num_frames, min_seconds, max_seconds).

    PLAN §3.6: auto-duration works by *omitting* ``num_frames`` and passing a
    seconds range, so the two controls are genuinely exclusive rather than
    merely redundant.
    """
    chosen = bool(auto)
    return (not chosen, chosen, chosen)


# ---------------------------------------------------------------------------
# LoRA -- PLAN §4
# ---------------------------------------------------------------------------


def _validator():
    """Resolve the validator entry point.

    Tolerant about *where* it lives because ``lora/`` is being written in
    parallel; intolerant about it being missing, because a LoRA panel that
    quietly does nothing is precisely the failure PLAN §4.2 is about.
    """
    from .. import lora

    candidates = [lora]
    try:
        from ..lora import validate as validate_module

        candidates.append(validate_module)
    except ImportError:  # pragma: no cover -- module layout still in flux
        pass

    for module in candidates:
        for name in ("validate_lora", "validate"):
            found = getattr(module, name, None)
            if callable(found):
                return found
    raise NotImplementedError(
        "no LoRA validator found: expected a callable `validate_lora` on "
        "`ltx25_studio.lora` or `ltx25_studio.lora.validate` (PLAN §4.3)."
    )


def on_validate_lora(
    repo_id: str | None, upload: Any, applied: list[dict[str, Any]] | None
) -> tuple[str, list[dict[str, Any]], str]:
    """Validate one adapter and, if it passes stages 1-6, stage it for the run.

    The verdict is *exception or no exception*, which is the one part of the
    contract that cannot drift as ``lora/`` settles. The detail panel reads the
    report tolerantly because PLAN §4.5 measured real adapters and most carry no
    metadata at all -- "not declared" is the expected reading, not a failure.

    Rejection messages are surfaced **verbatim**. ``errors.py`` wrote them to be
    actionable (which prefixes were found, which keys are unpaired, which two
    frame counts are legal); flattening one to "invalid file" would throw away
    the only part a user can act on.
    """
    staged = list(applied or [])
    source = (upload or "") or (repo_id or "").strip()
    if not source:
        return (
            "⚠️ Give me either a Hub repo id or an uploaded `.safetensors` file first.",
            staged,
            render.applied_loras(staged),
        )
    source = str(source)

    try:
        report = _validator()(source)
    except Exception as exc:  # noqa: BLE001 -- every failure must reach the user
        return (render.lora_rejected(source, exc), staged, render.applied_loras(staged))

    staged.append({"source": source, "report": report})
    return (render.lora_accepted(source, report), staged, render.applied_loras(staged))


def on_clear_loras() -> tuple[str, list[dict[str, Any]], str]:
    """Drop every staged adapter and say so."""
    return (render.lora_idle(), [], render.applied_loras([]))


# ---------------------------------------------------------------------------
# generation -- PLAN §5.3
# ---------------------------------------------------------------------------


def on_generate_start(mode_id: str) -> str:
    """Fires the instant the button is pressed, so the UI is never silent."""
    try:
        return render.running(modes.get_mode(mode_id))
    except Exception as exc:  # noqa: BLE001
        return render.failure(exc)


def on_generate(mode_id: str, *values: Any) -> tuple[Any, Any, str]:
    """Run one generation and marshal the result.

    Returns ``(video, audio, status)``. **Every** path returns a concrete value
    for all three and a non-empty status. There is no branch that leaves an
    output untouched, because an untouched output is what a silent reset looks
    like (PLAN §5.3).

    ``strict=True`` on the zip is load-bearing: ``FIELD_KEYS`` also built the
    ``inputs=`` list this tuple arrives from, so a mismatch is impossible by
    construction -- and if that ever stops being true, this raises instead of
    handing the wrong value to the wrong parameter.
    """
    params = dict(zip(FIELD_KEYS, values, strict=True))

    try:
        mode = modes.get_mode(mode_id)
    except Exception as exc:  # noqa: BLE001
        return (None, None, render.failure(exc))

    try:
        from .. import generate as generate_module

        result = generate_module.generate(mode=mode_id, **params)
    except Exception as exc:  # noqa: BLE001 -- typed or not, the user sees it
        return (None, None, render.failure(exc))

    video_path = getattr(result, "video_path", None)
    audio_path = getattr(result, "audio_path", None)

    # A success-shaped result carrying an empty path is a bug, not an outcome.
    # Treat it as loudly as an exception rather than rendering the empty value.
    missing = [
        track
        for track, path in (("video", video_path), ("audio", audio_path))
        if track in mode.exports and not path
    ]
    if missing:
        return (None, None, render.empty_output_bug(mode, missing))

    return (
        video_path if "video" in mode.exports else None,
        audio_path if "audio" in mode.exports else None,
        render.success(result, mode),
    )
