"""Presentation only: turn objects and exceptions into Markdown a human reads.

PLAN §7.2 says the UI is the one layer allowed to render a typed exception as a
string. This module is that layer, isolated from Gradio so it can be read and
tested on a bare interpreter -- it imports nothing but the error hierarchy.

**Everything here exists to make failure loud.** PLAN §5.3's post-mortem is of a
bug in exactly this seam: generation returned an empty path and the UI rendered
it with ``event.video_path if event.video_path else gr.update()``, a no-op update
that looks identical to a form resetting itself. Nobody saw an error because
nobody rendered one. It hid for a week.

So the rules this module is written to:

* every failure path produces a **non-empty** string naming what failed;
* a validator's own message is surfaced **verbatim** -- the messages in
  ``errors.py`` were written to be actionable, and flattening one to "invalid
  file" throws away the only part the user could act on;
* "no adapter applied" is stated positively rather than shown as blankness,
  because an empty panel and a working-but-inert LoRA look the same;
* a success-shaped result carrying an empty path gets the loudest message in
  the file, because that is a bug in *us*.
"""

from __future__ import annotations

from typing import Any

from ..errors import (
    LoRAConventionError,
    LoRADimensionError,
    LoRAError,
    LoRAFormatError,
    LoRAHeaderError,
    LoRAInContextError,
    LoRAModuleNotFoundError,
    LoRANotAttachedError,
    LoRAPairingError,
    LoRATooLargeError,
    LoRATruncatedError,
    LoRAVersionError,
    LTXStudioError,
)

__all__ = [
    "AUDIO_VISUAL_BANNER",
    "ZEROGPU_BANNER",
    "applied_loras",
    "dimension_hint",
    "empty_output_bug",
    "failure",
    "frame_hint",
    "lora_accepted",
    "lora_idle",
    "lora_rejected",
    "lora_stage_of",
    "mode_briefing",
    "running",
    "success",
]


# ---------------------------------------------------------------------------
# standing banners -- the honest cost of the model, always on screen
# ---------------------------------------------------------------------------

#: PLAN §3.1(b). Stated where the user picks a mode, not buried in a README.
AUDIO_VISUAL_BANNER = (
    "**LTX-2.5 is jointly audio-visual and has no video-off switch.** Every run renders "
    "*both* tracks and spends full video compute, whichever one you export. The four audio "
    "modes below are honest about this rather than pretending to be cheap."
)

#: PLAN §8.6. Quota is the resource users actually run out of.
ZEROGPU_BANNER = (
    "Running on **ZeroGPU**: each generation reserves a GPU slot and is **metered against a "
    "daily quota**, charged at 2× because this model needs the `xlarge` tier. Failed runs cost "
    "quota too. A timeout is never retried automatically — retrying reuses an expired proxy "
    "token and reports the wrong cause (PLAN §5.8 #5)."
)


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------


def mode_briefing(mode: Any) -> str:
    """Describe one mode, including its cost note **verbatim** from the registry.

    PLAN §3.2 makes this a requirement rather than a nicety: modes 8-11 still
    render video internally, and "a user who does not know that will read the
    runtime as a bug".
    """
    takes = ", ".join(mode.inputs) or "nothing"
    gives = " + ".join(mode.exports)
    lines = [
        f"### {mode.label}",
        f"**Takes** {takes} · **Returns** {gives}",
    ]

    cost_note = getattr(mode, "cost_note", None)
    if cost_note:
        lines += ["", f"> ⚠️ **What this actually costs:** {cost_note}"]

    mitigations = []
    if getattr(mode, "min_resolution", False):
        mitigations.append(
            "runs at the **minimum legal resolution** — the resolution controls are greyed "
            "out and show the size that will actually be used, because the video is rendered "
            "and discarded and paying for pixels nobody sees would be waste"
        )
    if getattr(mode, "skip_stage_2", False):
        mitigations.append("**skips stage-2 refinement** for the same reason")
    if mitigations:
        lines += ["", "Mitigations applied automatically: " + "; ".join(mitigations) + "."]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# live constraint guidance -- PLAN §5.5
# ---------------------------------------------------------------------------


def _legal_frames_near(value: int, ratio: int, count: int = 4) -> str:
    """A few legal frame counts around ``value``, so the rule is concrete."""
    steps = max((value - 1) // ratio, 0)
    first = max(steps - count // 2, 0)
    return ", ".join(str(k * ratio + 1) for k in range(first, first + count + 1))


def frame_hint(num_frames: Any, *, ratio: int, lower: int, upper: int, valid: bool) -> str:
    """Guidance for the frame count, naming the silent-floor behaviour.

    The caller has already asked ``constraints.py`` for the verdict; this only
    words it. Upstream does not raise here -- it computes
    ``(num_frames - 1) // ratio + 1`` and floors -- so this hint is genuinely
    the only warning a user gets before the fact.
    """
    if valid:
        return (
            f"✅ **{num_frames} frames** is legal. Nearby legal values: "
            f"{_legal_frames_near(int(num_frames), ratio)}."
        )
    return (
        f"⚠️ **{num_frames} is not a legal frame count.** It must be k×{ratio} + 1. "
        f"Upstream does **not** reject this — it silently floors it, so you would get "
        f"**{lower}** frames and no warning at all. Legal values either side: **{lower}** or "
        f"**{upper}**. Press *Snap to legal values* to fix it."
    )


def dimension_hint(width: Any, height: Any, *, multiple: int, snapped: tuple[int, int]) -> str:
    """Guidance for width/height. Upstream *does* raise on these, so the cost of
    getting it wrong is a failed run rather than a silently wrong one."""
    bad = []
    if not (isinstance(width, int) and width >= multiple and width % multiple == 0):
        bad.append(f"width **{width}** → **{snapped[0]}**")
    if not (isinstance(height, int) and height >= multiple and height % multiple == 0):
        bad.append(f"height **{height}** → **{snapped[1]}**")
    if not bad:
        return f"✅ {width}×{height} — both divisible by {multiple}."
    return (
        f"⚠️ Every spatial dimension must be a multiple of **{multiple}**; upstream raises "
        f"otherwise. {' · '.join(bad)}. Press *Snap to legal values*."
    )


# ---------------------------------------------------------------------------
# LoRA -- PLAN §4.3
# ---------------------------------------------------------------------------

#: Which validator stage each error means. PLAN §4.3 numbers them; surfacing the
#: number turns "it was rejected" into "it was rejected *here*, for *this*".
#: The half-numbered stages are the gates ``validate.py`` added between the
#: planned ones -- they are named as they are so the mapping stays legible
#: against §4.3's table rather than silently renumbering it.
_LORA_STAGES: tuple[tuple[type, str, str], ...] = (
    (LoRATooLargeError, "1", "size bound"),
    (LoRAFormatError, "3", "format"),
    (LoRAHeaderError, "3", "header parse"),
    (LoRATruncatedError, "3", "header parse"),
    (LoRAVersionError, "4", "version"),
    (LoRAConventionError, "5", "key convention"),
    (LoRAInContextError, "5.5", "in-context adapter"),
    (LoRAPairingError, "6", "low-rank pairing"),
    (LoRADimensionError, "6.5", "shape fit"),
    (LoRAModuleNotFoundError, "6.5", "target modules"),
    (LoRANotAttachedError, "7", "attach assertion"),
)


def lora_stage_of(exc: BaseException) -> tuple[str, str] | None:
    """The validator stage an exception came from, or ``None`` if unmapped."""
    for error_type, number, name in _LORA_STAGES:
        if isinstance(exc, error_type):
            return number, name
    return None


def lora_idle() -> str:
    """The resting state. Says *no adapter is applied* rather than showing blank.

    An empty panel and an adapter that loaded but attached to nothing look the
    same to a user. PLAN §4.2 is about exactly that class of confusion, so the
    UI never expresses "nothing applied" as absence.
    """
    return (
        "**No LoRA applied.** Generation will use the base LTX-2.5 weights.\n\n"
        "Paste a Hub repo id or upload a `.safetensors` adapter, then press *Validate*. "
        "Nothing is downloaded in full to validate it — only the file header is read."
    )


def _shown(value: Any, absent: str = "not declared") -> str:
    if value is None or value == "":
        return f"_{absent}_"
    return f"`{value}`"


def lora_accepted(source: str, report: Any) -> str:
    """Render an accepted adapter's detected properties.

    Read tolerantly on purpose. PLAN §4.5 measured three real public LTX LoRAs
    and **none** carried ``model_version``; two had no ``__metadata__`` at all.
    "not declared" is therefore the honest, expected reading for most real
    uploads — not a defect, and not something to hide.
    """
    warnings = tuple(getattr(report, "warnings", ()) or ())
    lines = [
        f"### ✅ Accepted — `{source}`",
        "",
        "| | |",
        "|---|---|",
        f"| key convention | {_shown(getattr(report, 'convention', None), 'undetected')} |",
        f"| `model_version` | {_shown(getattr(report, 'model_version', None))} |",
        f"| rank | {_shown(getattr(report, 'rank', None))} |",
        f"| alpha | {_shown(getattr(report, 'alpha', None))} |",
        f"| tensors | {_shown(getattr(report, 'num_tensors', None), 'unknown')} |",
    ]
    if getattr(report, "targets_connectors", None) is not None:
        touches = "yes" if report.targets_connectors else "no"
        lines.append(f"| touches the connectors | `{touches}` |")
    lines.append("")
    lines.append(
        "Stages 1–6 passed. **Stage 7 — the check that the adapter actually attaches to more "
        "than zero modules — runs at generation time**, against real weights. That is the check "
        "upstream does not do, and the one that catches an adapter built for another "
        "architecture."
    )
    if warnings:
        lines += ["", "**Warnings:**"] + [f"- {w}" for w in warnings]
    return "\n".join(lines)


def lora_rejected(source: str, exc: BaseException) -> str:
    """Render a rejection, naming the stage and quoting the message verbatim."""
    stage = lora_stage_of(exc)
    heading = f"### ❌ Rejected — `{source}`"
    if stage is not None:
        number, name = stage
        heading += f"\n\n**Failed at stage {number} — {name}.**"
    detail = str(exc).strip() or exc.__class__.__name__
    body = [heading, "", f"> {detail}", "", "**No adapter has been applied.**"]
    if not isinstance(exc, LoRAError):
        body.insert(1, f"\n_(unexpected `{exc.__class__.__name__}` — please report this)_")
    return "\n".join(body)


def applied_loras(applied: list[Any]) -> str:
    """The running list of adapters that will be used by the next generation."""
    if not applied:
        return lora_idle()
    names = "\n".join(f"- `{a.get('source', a)}`" for a in applied)
    return f"**{len(applied)} adapter(s) will be applied to the next run:**\n{names}"


# ---------------------------------------------------------------------------
# generation outcomes -- PLAN §5.3
# ---------------------------------------------------------------------------


def running(mode: Any) -> str:
    """Shown the moment the button is pressed, so the UI is never silent."""
    return (
        f"⏳ **Running — {mode.label}.** Reserving a ZeroGPU slot, then loading, denoising and "
        f"decoding. This is metered against your daily quota."
    )


def success(result: Any, mode: Any) -> str:
    """Confirm a run and say where its artifacts live.

    PLAN §5.3 gives each run its own output subdirectory precisely so a run can
    be correlated with its files. That is only useful if the UI shows which one.
    """
    lines = [f"✅ **{mode.label}** complete."]
    details = []
    seed = getattr(result, "seed", None)
    if seed is not None:
        details.append(f"seed `{seed}`")
    run_dir = getattr(result, "run_dir", None)
    if run_dir:
        details.append(f"run directory `{run_dir}`")
    if details:
        lines.append(" · ".join(details))
    cost_note = getattr(mode, "cost_note", None)
    if cost_note:
        lines += ["", f"> ⚠️ Reminder: {cost_note}"]
    return "\n\n".join(lines)


def empty_output_bug(mode: Any, missing: list[str]) -> str:
    """A run reported success and produced no file. This is a bug in *us*.

    PLAN §5.3 is explicit that a success path capable of returning a falsy
    artifact is a bug regardless of current reachability. The UI's job when it
    sees one is to be as loud as possible -- never to render the empty value and
    let it look like nothing happened.
    """
    tracks = " and ".join(missing)
    return (
        f"### 🛑 Internal error — empty output\n\n"
        f"**{mode.label}** reported success but produced no {tracks} file. That combination is "
        f"supposed to be impossible: PLAN §5.3 requires an empty artifact to raise at the "
        f"source rather than reach this layer.\n\n"
        f"**Nothing above is from this run.** Please report this — it is the exact failure "
        f"shape that hid for a week in the predecessor app."
    )


def failure(exc: BaseException) -> str:
    """Render any generation failure visibly and verbatim.

    Never returns an empty string and never returns a no-op: there is no code
    path in which a user presses Generate and sees nothing change.
    """
    detail = str(exc).strip() or "no detail was attached to the error."
    if isinstance(exc, LTXStudioError):
        return f"### ❌ Generation failed\n\n**{exc.__class__.__name__}**\n\n> {detail}"
    return (
        f"### ❌ Generation failed — unexpected `{exc.__class__.__name__}`\n\n"
        f"> {detail}\n\n"
        f"This error is not one of ours, which usually means it came from the pipeline or the "
        f"platform. The full traceback is in the Space logs."
    )
