"""Every string a user reads, and the markup that carries it.

Gradio-free: this module turns facts handed to it by ``handlers.py`` into
Markdown or small HTML fragments. It does no arithmetic beyond wording, so a
figure on screen cannot disagree with the module that computed it.

Rules the copy is written to (the UI brief, section 7.7):

* plain words for someone spending GPU quota; no internal ids, file names or
  exception class names as headlines;
* every failure path produces a non-empty message that says what happened,
  whether quota was used and one next step, with the verbatim detail kept;
* a validator's or the backend's own message is quoted verbatim, never
  flattened;
* "no LoRA" is said in words, never shown as blankness.
"""

from __future__ import annotations

import html
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import audio_conditioning, constraints, errors, modes
from ..errors import (
    LoRAConventionError,
    LoRADimensionError,
    LoRAError,
    LoRAFetchError,
    LoRAFormatError,
    LoRAHeaderError,
    LoRAInContextError,
    LoRAModuleNotFoundError,
    LoRANotAttachedError,
    LoRAPairingError,
    LoRATooLargeError,
    LoRATruncatedError,
    LoRAVersionError,
)

__all__ = [
    "COLUMN_NOTES",
    "COLUMNS",
    "COST_COLUMN",
    "FIELD_COPY",
    "LOCAL_COLUMN_NOTES",
    "NOISE_CHOICES",
    "READY",
    "ROWS",
    "SHAPE_PRESETS",
    "CostFacts",
    "RunFacts",
    "applied_loras",
    "card_headline",
    "column_notes",
    "commit_line",
    "commit_markup",
    "cost_panel",
    "dimension_hint",
    "empty_output_bug",
    "failure",
    "figure_line",
    "frame_hint",
    "inputs_unread",
    "last_run_line",
    "lora_accepted",
    "lora_duplicate",
    "lora_idle",
    "lora_nothing_to_validate",
    "lora_rejected",
    "lora_stage_of",
    "lora_summary_label",
    "lora_validating",
    "lost_run",
    "media_caption",
    "monitor_caption",
    "monitor_idle",
    "pinned_line",
    "prompt_label",
    "prompt_placeholder",
    "purpose_line",
    "quota_explainer",
    "readiness_reason",
    "recent_run_label",
    "result_not_shown",
    "run_summary",
    "running",
    "sample_of",
    "shot_anchor_readout",
    "step_names",
    "success",
    "upscale_info",
    "upscale_needs_cuda",
    "where_it_runs",
]


def _e(value: Any) -> str:
    """Escape anything that is shown inside HTML markup."""
    return html.escape(str(value), quote=True)


def _num(value: Any) -> str:
    """A figure, set in the numeral face."""
    return f'<span class="mr-num">{_e(value)}</span>'


def _figure(value: Any) -> str:
    """A cost figure (the reservation, the quota needed): tungsten in the cost panel."""
    return f'<span class="mr-num mr-cost-figure">{_e(value)}</span>'


def _size(width: Any, height: Any) -> str:
    return f"{int(width)}\u00a0×\u00a0{int(height)}"


def _size_html(size: tuple[int, int]) -> str:
    """A size with each figure in the numeral face and a plain multiplication sign."""
    return f"{_num(int(size[0]))}\u00a0×\u00a0{_num(int(size[1]))}"


def _secs_html(value: float) -> str:
    return f"{_num(f'{value:.1f}')}\u00a0s"


def _seconds(value: float) -> str:
    return f"{value:.1f}\u00a0s"


def _two_places(value: float) -> str:
    """Two decimals, half up (0.725 -> 0.73), as a person rounds."""
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _and(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# the matrix: what you start from x what you get back
# ---------------------------------------------------------------------------

#: Rows, keyed by the set of inputs a mode REQUIRES. Placement is derived from
#: the registry in ``chrome.py``; only the words live here.
ROWS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"text"}), "Text"),
    (frozenset({"image"}), "Image"),
    (frozenset({"image", "text"}), "Image + text"),
    (frozenset({"video"}), "Video"),
    (frozenset({"audio"}), "Audio"),
)

#: Columns, keyed by the set of tracks a mode exports.
COLUMNS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"audio", "video"}), "Video + sound"),
    (frozenset({"video"}), "Video (silent)"),
    (frozenset({"audio"}), "Sound only"),
)

#: Stated once, in words, under the column they belong to (on ZeroGPU).
COLUMN_NOTES: dict[str, str] = {
    "Video (silent)": "The sound is rendered, then dropped. Same GPU time as with sound.",
    "Sound only": "Renders a 256\u00a0×\u00a0256 video that is thrown away; you pay for that GPU time.",
}
#: The same notes where no meter runs: what happens, and what it takes.
LOCAL_COLUMN_NOTES: dict[str, str] = {
    "Video (silent)": "The sound is rendered, then dropped. The run takes as long as with sound.",
    "Sound only": (
        "Renders a 256\u00a0×\u00a0256 video that is thrown away, and the run takes the time to render it."
    ),
}
#: The column whose runs render a video nobody gets back: marked as a cost on every platform.
COST_COLUMN = "Sound only"


def column_notes(*, on_zerogpu: bool = True) -> dict[str, str]:
    """The column notes in this platform's words."""
    return COLUMN_NOTES if on_zerogpu else LOCAL_COLUMN_NOTES


#: Where a run's time is spent, by device (the Done card's "Took" uses the same words).
_ON_DEVICE = {"cpu": "on the CPU", "mps": "on this Mac's GPU", "cuda": "on the GPU"}


def where_it_runs(device: str | None) -> str:
    """``device`` in the page's words: "on the CPU", "on this Mac's GPU", "on the GPU"."""
    return _ON_DEVICE.get(str(device or ""), "on this machine")

#: One line per mode, shown under the matrix for the selected cell.
_PURPOSE: dict[str, str] = {
    "text_to_audio_video": "A clip with matching sound, from your words.",
    "text_to_video": "A silent clip from your words.",
    "text_to_audio": "Sound from your words: a voice, music, effects or ambience.",
    "image_to_audio_video": "Your image, brought to life, with sound.",
    "image_to_video": "Your image, brought to life, without sound.",
    "image_text_to_audio_video": "Your image, moving the way you describe, with sound.",
    "image_text_to_video": "Your image, moving the way you describe, without sound.",
    "video_to_video": "Your clip, re-rendered. Describe a new look if you like.",
    "video_to_audio": "A soundtrack for your clip.",
    "audio_to_video": (
        "A picture for your sound. Your audio stays as the soundtrack. Speech is not lip-synced."
    ),
    "audio_to_audio": "Your sound, transformed.",
}


def purpose_line(mode: Any) -> str:
    """The selected cell, in words, under the matrix. A mode has one name everywhere:
    the registry's ``label``, the same words as its cell."""
    purpose = _PURPOSE.get(mode.id, "")
    return f"**{mode.label}.** {purpose}".rstrip()


# ---------------------------------------------------------------------------
# field copy -- labels and info lines, keyed by the field spec in fields.py
# ---------------------------------------------------------------------------

#: ``key -> (label, info)``. Info lines become accessible descriptions, so
#: each is at most two short sentences.
FIELD_COPY: dict[str, tuple[str, str]] = {
    "prompt": ("Prompt", ""),
    "negative_prompt": ("Negative prompt", "Not used: this model always runs at guidance 1.0."),
    "image": ("Image", "Used as the first frame."),
    "video": ("Video clip", "The clip to work from."),
    "audio": ("Audio", "The sound to work from."),
    "conditions": ("Shot list", ""),
    "width": ("Width", "Multiples of 32."),
    "height": ("Height", "Multiples of 32."),
    "auto_duration": (
        "Let the model pick the length",
        "Between the shortest and longest seconds in Advanced.",
    ),
    "num_frames": ("Frames", "Lengths come in steps of 8 frames: 113, 121, 129…"),
    "min_seconds": ("Shortest (seconds)", "Used when the model picks the length."),
    "max_seconds": (
        "Longest (seconds)",
        f"Used when the model picks the length; {constraints.MAX_SECONDS:g} s at most.",
    ),
    "enable_stage_2": (
        "Upscale ×2",
        "Upsamples and refines after the first pass, for twice the width and height. "
        "Reserves {upscale_seconds} s more at the default size and length; when it is on, "
        "the cost panel shows what it adds.",
    ),
    "seed": ("Seed", "-1 picks a new seed each run. The seed used is in the run report."),
    "strength": ("Stick to the source clip", "Higher keeps more of your clip. 0.99 at most."),
    "noise_scale": (
        "How much to change your sound",
        "A little keeps the most of your sound and A lot the least; the model rebuilds the "
        "rest. Ignore my audio makes new sound, as if none were given.",
    ),
    "loras": ("LoRA adapters", ""),
    "decoder": ("Decoder", ""),
    "enhance_prompt": (
        "Enhance my prompt",
        "Gemma rewrites your prompt in detail on the GPU before the run. Reserves "
        "{enhance_seconds} s more; uses a few seconds when loaded, up to about 2 minutes on a "
        "cold start.",
    ),
}


def prompt_label(mode: Any, *, required: bool) -> str:
    return "Prompt" if required else "Prompt (optional)"


_PLACEHOLDERS: dict[str, str] = {
    "text_to_audio_video": (
        "Describe the shot and what is heard: subject, action, camera, light, voices, sound."
    ),
    "text_to_video": "Describe the shot: subject, action, camera movement, light.",
    "text_to_audio": "Describe the sound: a voice and what it says, music, effects or ambience.",
    "image_text_to_audio_video": (
        "Describe what happens in your image, and what is heard: motion, camera, sound."
    ),
    "image_text_to_video": "Describe what happens in your image: motion, camera, light.",
    "video_to_video": "Optional. A new look for the clip, for example: as a watercolour painting.",
    "video_to_audio": (
        "Optional. What the sound should be, for example: footsteps on gravel, distant traffic."
    ),
    "audio_to_audio": (
        "Optional. What to turn it into, for example: the same words, whispered in a cave."
    ),
    "audio_to_video": (
        "Optional. What the picture should show, for example: a captain at the wheel in a storm."
    ),
}


def prompt_placeholder(mode: Any) -> str:
    return _PLACEHOLDERS.get(mode.id, "Describe the shot.")


def enhance_info(mode: Any, enhance_seconds: int, *, on_zerogpu: bool = True) -> str:
    """The Enhance line for ``mode``: on ZeroGPU what it reserves and its cold start;
    elsewhere where it runs and that its first use loads it. The image clause only where the
    mode takes an image (the enhancer reads one, ``enhance.py``)."""
    if modes.requires(mode, "image"):
        reads = ", and reads your image"
    elif modes.accepts(mode, "image"):
        reads = ", and reads your image if you add one"
    else:
        reads = ""
    if not on_zerogpu:
        return (
            f"Gemma rewrites your prompt in detail on this machine before the run{reads}. The "
            f"first run with it also loads Gemma, so it takes longer."
        )
    return (
        f"Gemma rewrites your prompt in detail on the GPU before the run{reads}. Reserves "
        f"{enhance_seconds} s more; uses a few seconds when loaded, up to about 2 minutes on a "
        f"cold start."
    )


#: What Upscale ×2 does, on every platform.
UPSCALE_DOES = "Upsamples and refines after the first pass, for twice the width and height."


def upscale_label(size: tuple[int, int] | None) -> str:
    """The Upscale label, with the size it returns for a size the model takes (``size``);
    without one for a size that won't run."""
    base = FIELD_COPY["enable_stage_2"][0]
    return f"{base}: returns {_size(size[0] * 2, size[1] * 2)}" if size else base


def upscale_info(*, upscale_seconds: int, on_zerogpu: bool = True, refused: str | None = None) -> str:
    """The Upscale line: what it does, with what it reserves on ZeroGPU; or, where this
    machine cannot run it for this request, why (the control is then parked)."""
    if refused:
        return refused
    if not on_zerogpu:
        return UPSCALE_DOES
    return (
        f"{UPSCALE_DOES} Reserves {upscale_seconds} s more at the default size and length; "
        f"when it is on, the cost panel shows what it adds."
    )


def upscale_needs_cuda(device: str | None) -> str:
    """Why Upscale ×2 is parked here: the backend's refusal, in the page's device words."""
    return f"Upscale ×2 needs a CUDA GPU; this machine runs {where_it_runs(device)}."


_CAPTIONS: dict[tuple[str, str], str] = {
    ("video_to_video", "video"): "The clip to re-render.",
    ("video_to_audio", "video"): "The clip to make sound for.",
    ("audio_to_audio", "audio"): "The sound to transform.",
    ("audio_to_video", "audio"): (
        "The sound to picture; it is also the soundtrack of the result. Speech is not "
        "lip-synced. For a face that speaks in time, use Image + text to video with sound and "
        "write the words in quotes in your prompt; the model voices them itself."
    ),
    # Under the first-frame picture, above the prompt: the two remedies it names.
    ("audio_to_video", "image"): (
        "Optional. Used as the first frame. An old-sounding recording can come out as a period "
        "film: say “in colour” in the prompt, or give a first-frame picture."
    ),
}


def media_caption(mode: Any, kind: str) -> str:
    """The one line under a media input, written for the selected mode."""
    return _CAPTIONS.get((mode.id, kind), FIELD_COPY[kind][1])


#: The four choices that write ``noise_scale`` in Audio → sound.
#: The words for the levels ``audio_conditioning`` supports, lowest first; the
#: values are the module's own, so the choice writes a level it can run.
NOISE_CHOICES: tuple[tuple[str, float], ...] = tuple(
    zip(
        ("A little", "Moderately", "A lot", "Ignore my audio (new sound)"),
        audio_conditioning.VARIATION_LEVELS,
        strict=True,
    )
)

#: The three choices that write ``noise_scale`` in Video → video: the clip-variation
#: levels ``audio_conditioning`` supports, lowest first. "A lot" is the mode's default
#: and the only level at which the prompt visibly acts (measured on real weights).
CLIP_CHOICES: tuple[tuple[str, float], ...] = tuple(
    zip(
        ("A little", "Moderately", "A lot"),
        audio_conditioning.VIDEO_VARIATION_LEVELS,
        strict=True,
    )
)
CLIP_DEFAULT = "A lot"

#: The clip choice's label and its line. It is not a field of its own: it writes
#: ``noise_scale``, as the change choice in Audio → sound does.
CLIP_LEVER = (
    "How much to change your clip",
    "The prompt steers the result only at “A lot”; at the lower settings your clip is "
    "re-rendered almost as it is.",
)
#: The decoder choices: ``(label, value, what it does)``, named by what a user sees (the
#: Space's A/B on the same latents: the diffusion decoder a little softer and steadier from
#: frame to frame, the standard one sharper frame by frame with more shimmer). The values are
#: ``config.DECODER_CHOICES``; the reservation does not depend on the choice. Auto is the
#: vendor's default; off an NVIDIA GPU it always takes the standard decoder (``pipelines``).
DECODER_OPTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "Auto",
        "auto",
        "Diffusion decoder on an NVIDIA GPU that can run it, standard decoder elsewhere.",
    ),
    (
        "Steadier (diffusion decoder)",
        "diffusion",
        "A little softer, steadier from frame to frame; takes a few seconds longer.",
    ),
    (
        "Sharper (standard decoder)",
        "conv",
        "Sharper frame by frame, with a little more shimmer; faster.",
    ),
)
#: What each decoder that ran is called in the report: the choice's own label.
DECODER_NAMES = {value: label for label, value, _words in DECODER_OPTIONS}


#: Under Upscale ×2 the decoder choice is not a lever (``pipelines.StageRunner``).
UPSCALE_DECODER_LINE = (
    "Upscale ×2 always finishes with the diffusion decoder, whichever you pick here."
)


def decoder_line(value: Any, *, upscaling: bool = False) -> str:
    """The selected decoder, in one line; under Upscale ×2, the decoder it always uses."""
    if upscaling:
        return UPSCALE_DECODER_LINE
    return next((words for _l, v, words in DECODER_OPTIONS if v == value), DECODER_OPTIONS[0][2])


#: Shape presets: ``(name, width, height)``. All multiples of 32.
SHAPE_PRESETS: tuple[tuple[str, int, int], ...] = (
    ("3:2", 704, 480),
    ("16:9", 960, 544),
    ("9:16", 544, 960),
    ("1:1", 640, 640),
    ("4:3", 640, 480),
)
MATCH_INPUT = "Match input"
LENGTH_MATCH = "Match my clip"
LENGTH_SET = "Set the length"


def pinned_line(mode: Any, size: int) -> str:
    return f"Renders at {size}\u00a0×\u00a0{size} with no upscale, because the picture is thrown away."


def match_clip_line(
    *, clip_seconds: float | None, frames: int | None, fps: float, audio: bool, cap_seconds: float
) -> str:
    """What "Match my clip" will do, stated before the run."""
    if audio:
        if clip_seconds is None:
            return f"Matches your audio's length, up to {cap_seconds:g} s."
        line = f"Matches your {clip_seconds:.1f}\u00a0s audio: {frames}\u00a0frames."
    else:
        if clip_seconds is None:
            return f"Matches your clip's length, up to {cap_seconds:g} s."
        line = f"Matches your {clip_seconds:.1f}\u00a0s clip: {frames}\u00a0frames."
    if clip_seconds is not None and clip_seconds > cap_seconds:
        line += f" Trimmed to the first {frames / fps:.1f} s, the longest this model makes."
    return line


def match_input_line(
    *, width: int, height: int, source_width: int, source_height: int, what: str
) -> str:
    return f"Matches your {what} ({_size(source_width, source_height)}): {_size(width, height)}."


# ---------------------------------------------------------------------------
# identity and quota
# ---------------------------------------------------------------------------

TAGLINE = "Video and sound from text, images, clips or audio. Bring your own LoRA."


def quota_explainer(*, multiplier: int, example_reserve: int) -> str:
    """The "How quota works" disclosure, with the numbers this app actually uses."""
    double = {1: "counts once", 2: "counts double"}.get(multiplier, f"counts {multiplier} times")
    return (
        "Each run reserves GPU time on Hugging Face ZeroGPU from your daily quota. This model "
        f"needs the xlarge GPU, which {double}: a {example_reserve} s reservation needs "
        f"{example_reserve * multiplier} s of quota free to start, and you are billed for the "
        f"GPU time actually used, times {multiplier}. Runs that fail on the GPU are billed too. "
        f"A run that runs out of GPU time is not retried for you. {_CARDS['timeout'][1]}"
    )


# ---------------------------------------------------------------------------
# live constraint guidance, beside the exact fields in Advanced
# ---------------------------------------------------------------------------


def frame_hint(num_frames: Any, *, ratio: int, lower: int, upper: int, valid: bool) -> str:
    """Guidance for the frame count. ``constraints.py`` has already decided."""
    if valid:
        return f"{num_frames} frames runs ({int(num_frames) / 24:.2f} s)."
    return (
        f"{num_frames} frames won't run. Use {lower} ({lower / 24:.1f} s) or "
        f"{upper} ({upper / 24:.1f} s), or press Snap."
    )


def frame_limit_hint(num_frames: Any, *, limit: int, seconds: float) -> str:
    """A frame count past the longest the model makes."""
    return (
        f"{num_frames} frames is longer than this model makes: {limit} frames "
        f"({seconds:.1f} s) at most. Press Snap."
    )


def dimension_limit_hint(over: Sequence[tuple[str, Any]], *, limit: int) -> str:
    """A width or height past the largest the model takes."""
    named = _and([f"{name} {value}" for name, value in over])
    return f"{named} is over the {limit} limit: use {limit} or less. Press Snap."


def dimension_hint(width: Any, height: Any, *, multiple: int, snapped: tuple[int, int]) -> str:
    """Guidance for width and height. ``constraints.py`` has already decided."""
    bad = []
    if not (isinstance(width, int) and width >= multiple and width % multiple == 0):
        bad.append(f"width {width} → {snapped[0]}")
    if not (isinstance(height, int) and height >= multiple and height % multiple == 0):
        bad.append(f"height {height} → {snapped[1]}")
    if not bad:
        return f"{width} × {height} runs: both are multiples of {multiple}."
    return f"Sizes must be multiples of {multiple}: {', '.join(bad)}. Press Snap."


def shot_anchor_readout(anchors: list[dict], *, out_of_range: list[int]) -> str:
    """Where each shot-list frame anchor lands. Kept for the API's shot list;
    the UI no longer shows a shot list (rows with text are refused)."""
    if not anchors:
        return (
            "_No shots yet._ Add a row to compose a multishot prompt. The anchor column takes a "
            "**frame number** — the readout here will show which latent it lands on, and which "
            "frames share it."
        )
    lines = ["| shot | anchor (frame) | lands on latent | frames sharing it |", "|---|---|---|---|"]
    inexact = []
    for n, anchor in enumerate(anchors, 1):
        span = (
            f"{anchor['first_frame']}–{anchor['last_frame']}"
            if anchor.get("shared")
            else str(anchor["first_frame"])
        )
        mark = "" if anchor.get("exact") else " ⟵"
        lines.append(f"| {n} | `{anchor['frame']}` | **{anchor['latent']}** | {span}{mark} |")
        if not anchor.get("exact"):
            inexact.append(anchor)
    body = "\n".join(lines)
    if inexact:
        first = inexact[0]
        body += (
            f"\n\n⟵ Temporal compression is 8:1, so these anchors snap. Frame "
            f"`{first['frame']}` anchors at the same place as frame `{first['first_frame']}` — "
            f"frames {first['first_frame']}–{first['last_frame']} all share latent "
            f"**{first['latent']}**."
        )
    if out_of_range:
        body += (
            f"\n\n⚠️ Anchor(s) {', '.join(str(v) for v in out_of_range)} fall outside the clip, "
            f"so a run with them is refused before it starts. Move the anchor inside the clip "
            f"or raise the frame count."
        )
    return body


# ---------------------------------------------------------------------------
# the cost panel -- what this run renders, returns and reserves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostFacts:
    """Everything the cost panel states, computed by ``handlers.review``."""

    mode_name: str
    #: ``ready`` | ``incomplete`` (an input is missing) | ``refused`` (won't run)
    state: str
    reason: str | None = None
    problems: tuple[str, ...] = ()
    render_size: tuple[int, int] | None = None
    pinned: bool = False
    frames: int | None = None
    seconds: float | None = None
    auto_seconds: tuple[float, float] | None = None
    length_from_audio: bool = False
    returns: str = "video"
    returned_size: tuple[int, int] | None = None
    stage_2: bool = False
    stage_2_offered: bool = True
    enhance: bool = False
    enhance_offered: bool = False
    reserve: int | None = None
    base: int | None = None
    extras: tuple[tuple[str, int], ...] = ()
    multiplier: int = 2
    loras: tuple[tuple[str, float], ...] = ()
    discards_video: bool = False
    encodes_audio: bool = False
    seed: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: The input kinds a ``MissingInputError`` named (``incomplete``).
    missing: tuple[str, ...] = ()
    #: ``(what, short form)`` per problem, for the phone's commit bar (``refused``).
    short_problems: tuple[tuple[str, str], ...] = ()
    #: The length comes from a clip not loaded yet: ``"clip"`` (or None).
    length_pending: str | None = None
    #: Where it runs. On ZeroGPU the panel states the reservation, the quota and the
    #: billing; anywhere else, the device, and that nothing is metered.
    on_zerogpu: bool = True
    device: str | None = None
    #: Why Upscale ×2 cannot run here for this request (the backend's refusal), or None.
    upscale_refused: str | None = None


def _returns_words(facts: CostFacts) -> str:
    size = _size_html(facts.returned_size) if facts.returned_size else ""
    if facts.returns == "audio":
        return "sound only (a WAV file)"
    if facts.returns == "audio+video":
        return f"{size} video with sound" if size else "video with sound"
    return f"{size} silent video" if size else "silent video"


def _lamp(text: str, *, lit: bool, kind: str = "cost") -> str:
    """A lamp; an unlit one is a cost that does not apply, so it is not read out (its state
    is not in colour alone)."""
    state = "is-lit" if lit else "is-off"
    hidden = "" if lit else ' aria-hidden="true"'
    return f'<span class="mr-lamp mr-lamp-{kind} {state}"{hidden}>{_e(text)}</span>'


def _lamps(facts: CostFacts) -> str:
    """A lamp lights only for a cost that applies; a request that won't run costs nothing,
    so in the refused state every lamp is dark. The seconds a lever reserves are named only
    where a reservation exists (ZeroGPU)."""
    extras = dict(facts.extras) if facts.on_zerogpu else {}
    live = facts.state != "refused"
    lamps = []
    # Parked where this machine cannot run it: no lamp for a lever that is not offered here.
    if facts.stage_2_offered and not facts.upscale_refused:
        seconds = extras.get("Upscale ×2")
        lamps.append(
            _lamp(f"Upscale +{seconds}\u00a0s" if seconds else "Upscale", lit=live and facts.stage_2)
        )
    if facts.enhance_offered:
        seconds = extras.get("Enhance")
        lamps.append(
            _lamp(f"Enhance +{seconds}\u00a0s" if seconds else "Enhance", lit=live and facts.enhance)
        )
    seconds = extras.get("LoRA")
    lamps.append(_lamp(f"LoRA +{seconds}\u00a0s" if seconds else "LoRA", lit=live and bool(facts.loras)))
    if facts.discards_video:
        lamps.append(_lamp("Video thrown away", lit=True))
    if facts.encodes_audio:
        lamps.append(_lamp("Audio encode", lit=True))
    return (
        '<div class="mr-lamps" role="group" aria-label="Costs that apply">'
        + "".join(lamps)
        + "</div>"
    )


def _length_words(facts: CostFacts) -> str:
    if facts.length_pending == "clip":
        return "the length of your clip"
    if facts.auto_seconds is not None:
        low, high = facts.auto_seconds
        return f"length chosen by the model, {_num(f'{low:g}–{high:g}')}\u00a0s"
    if facts.length_from_audio and facts.frames is None:
        return "the length of your audio"
    if facts.frames is not None and facts.seconds is not None:
        return f"{_secs_html(facts.seconds)} ({_num(facts.frames)}\u00a0frames)"
    return "length not set"


def weight_text(weight: Any) -> str:
    """A LoRA strength as people write it: 1.0, 0.8, 0.85."""
    text = f"{float(weight):.2f}"
    return text[:-1] if text.endswith("0") else text


def _lora_words(loras: Sequence[tuple]) -> str:
    """Adapters and strengths; ``(name, weight[, layers attached])`` each."""
    if not loras:
        return "none, base model only"
    parts = []
    for name, weight, *attached in loras:
        part = f"{_e(name)} at {_num(weight_text(weight))}"
        if attached:
            part += f", on {_num(attached[0])} layers"
        parts.append(part)
    return "; ".join(parts)


def cost_panel(facts: CostFacts) -> str:
    """The signature element: a labelled grid with cost lamps, at Generate."""
    head = (
        '<div class="mr-cost-head">'
        f'<h2 class="mr-cost-title">{_e(facts.mode_name)}</h2>'
        f"{_lamps(facts)}</div>"
    )
    if facts.state == "refused":
        items = "".join(f"<li>{_e(p)}</li>" for p in facts.problems) or (
            f"<li>{_e(facts.reason or 'This request would be refused.')}</li>"
        )
        held = (
            " Nothing is reserved for a request that would be refused."
            if facts.on_zerogpu
            else " Generate stays off until this is fixed."
        )
        body = (
            '<div class="mr-wont-run" role="note">'
            '<p class="mr-wont-run-head"><span class="mr-lamp mr-lamp-cut is-lit">Won’t run</span>'
            f"{held}</p>"
            f"<ul>{items}</ul></div>"
        )
        return f'<section class="mr-cost is-refused" aria-label="Cost of this run">{head}{body}</section>'

    rows = []
    if facts.render_size:
        renders = f"{_size_html(facts.render_size)}, {_length_words(facts)}"
    else:
        renders = _length_words(facts)
    rows.append(("Renders", renders))
    returns = _returns_words(facts)
    if facts.stage_2:
        returns += ", upscaled ×2"
    rows.append(("Returns", returns))
    if not facts.on_zerogpu:
        rows.append(("Runs", f"{_e(where_it_runs(facts.device))}; nothing is metered"))
    else:
        if facts.reserve is not None:
            reserve = f"{_figure(facts.reserve)}\u00a0s of GPU time"
            if facts.extras:
                parts = [f"{facts.base}\u00a0s base"] + [f"{s}\u00a0s for {n}" for n, s in facts.extras]
                reserve += f' <span class="mr-sub">({_e(" + ".join(parts))})</span>'
            rows.append(("Reserves", reserve))
            rows.append(
                ("Needs", f"{_figure(facts.reserve * facts.multiplier)}\u00a0s of quota free to start")
            )
        rows.append(
            (
                "Billed",
                f"for GPU time used ×\u00a0{facts.multiplier}; runs that fail on the GPU are billed too",
            )
        )
    rows.append(("LoRA", _lora_words(facts.loras)))
    if facts.seed is not None and facts.seed >= 0:
        rows.append(("Seed", f"{_num(facts.seed)} (fixed)"))
    grid = "".join(f"<dt>{_e(label)}</dt><dd>{value}</dd>" for label, value in rows)
    notes = ""
    if facts.discards_video:
        spent = (
            "you pay for that GPU time."
            if facts.on_zerogpu
            else "the run takes the time to render it."
        )
        notes += (
            f'<p class="mr-cost-note">A 256\u00a0×\u00a0256 video is rendered and thrown away; {spent}</p>'
        )
    if facts.encodes_audio:
        notes += (
            '<p class="mr-cost-note">Your audio is encoded first, then a full video render. '
            "The file carries your own audio.</p>"
        )
    pending = ""
    if facts.state == "incomplete" and facts.reason:
        pending = f'<p class="mr-cost-pending">{_e(facts.reason)}</p>'
    return (
        f'<section class="mr-cost is-{_e(facts.state)}" aria-label="Cost of this run">'
        f'{head}<dl class="mr-cost-grid">{grid}</dl>{notes}{pending}</section>'
    )


#: How the commit bar names a missing input.
_MISSING_WORDS = {
    "text": "a prompt",
    "image": "an image",
    "video": "a clip",
    "audio": "an audio file",
}


def commit_line(facts: CostFacts) -> str:
    """The phone's commit bar: the cost in one short line, or why Generate is off.

    Short enough to read whole on a phone; the cost panel holds the full reasons.
    """
    if facts.state == "refused":
        shorts = facts.short_problems
        if len(shorts) > 1:
            return f"Won’t run: {len(shorts)} problems ({', '.join(what for what, _ in shorts)})"
        if shorts:
            return f"Won’t run: {shorts[0][1]}"
        return "Won’t run. The cost panel says why."
    if facts.state == "incomplete":
        if facts.missing:
            return "Needs " + _and([_MISSING_WORDS.get(kind, kind) for kind in facts.missing])
        return "Not ready. The cost panel says why."
    return figure_line(facts)


def figure_line(facts: CostFacts) -> str:
    """The cost panel's figure in one line: what the run returns (as the Returns row: the
    returned size, "upscaled" under Upscale ×2, or "Sound" where only sound comes back), its
    length, and the quota needed (on ZeroGPU) or where it runs. Empty for a request that
    won't run (it has no figure)."""
    if facts.state == "refused":
        return ""
    parts = []
    if facts.returns == "audio":
        parts.append("Sound")
    else:
        size = facts.returned_size or (None if facts.stage_2 else facts.render_size)
        if size:
            parts.append(_size(*size) + ("\u00a0upscaled" if facts.stage_2 else ""))
    if facts.auto_seconds is not None:
        parts.append(f"{facts.auto_seconds[0]:g}–{facts.auto_seconds[1]:g}\u00a0s")
    elif facts.seconds is not None:
        parts.append(_seconds(facts.seconds))
    elif facts.length_pending == "clip":
        parts.append("the length of your clip")
    elif facts.length_from_audio:
        parts.append("the length of your audio")
    if not facts.on_zerogpu:
        parts.append(where_it_runs(facts.device))
    elif facts.reserve is not None:
        parts.append(f"needs {facts.reserve * facts.multiplier}\u00a0s quota")
    return ", ".join(parts)


def commit_markup(facts: CostFacts) -> str:
    """The line in the commit: below 1024 px the phone bar's (:func:`commit_line`); on a
    desktop the panel's figure (:func:`figure_line`), shown above Generate when the panel
    runs under it. The stylesheet shows the one for the width."""
    return (
        f'<span class="mr-commit-phone">{_e(commit_line(facts))}</span>'
        f'<span class="mr-commit-desk">{_e(figure_line(facts))}</span>'
    )


def readiness_reason(facts: CostFacts, *, running: bool = False) -> str:
    """The line under Generate whenever it is disabled. Empty when ready."""
    if running:
        return "A run is in progress. Generate comes back when it finishes."
    if facts.state == "ready":
        return ""
    if facts.state == "refused" and facts.problems:
        return " ".join(facts.problems)
    return facts.reason or "This request would be refused."


# ---------------------------------------------------------------------------
# the monitor and the status line under it
# ---------------------------------------------------------------------------

READY = "Ready. Nothing is spent until you press Generate."

#: Below 1024 px the monitor sits under the inputs: a jump to the mode's sample.
SAMPLE_JUMP = "See an example output"
#: While the monitor holds a run from another mode, at every width: the selected mode's sample.
SAMPLE_OF = "See an example of {mode}"
#: While a sample covers a run, the same control puts the run back.
BACK_TO_RUN = "Back to your run"


def sample_of(mode: Any) -> str:
    """The control that puts ``mode``'s sample on a monitor holding a run of another mode."""
    return SAMPLE_OF.format(mode=mode.label)


def monitor_idle(mode: Any) -> str:
    """The monitor with nothing to play: a designed state, not a stock player.

    Read as its words (the mode's name, where the result will play); the crop marks are
    decoration and hidden from assistive tech.
    """
    marks = "".join(
        f'<span class="mr-crop {corner}" aria-hidden="true"></span>'
        for corner in ("tl", "tr", "bl", "br")
    )
    return (
        f'<div class="mr-monitor-idle">{marks}'
        f'<p class="mr-idle-mode">{_e(mode.label)}</p>'
        '<p class="mr-idle-hint">Your result plays here.</p></div>'
    )


#: A size, a length in seconds or a seed written with ordinary spaces (manifest text).
_BREAKABLE_FIGURES = re.compile(r"(\d) ?× ?(\d)|(\d) (s\b)|(seed) (\d)")


def _held(text: str) -> str:
    """``text`` with its sizes, seconds and seeds held together by no-break spaces."""

    def hold(match: re.Match) -> str:
        if match.group(1):
            return f"{match.group(1)}\u00a0×\u00a0{match.group(2)}"
        if match.group(3):
            return f"{match.group(3)}\u00a0{match.group(4)}"
        return f"{match.group(5)}\u00a0{match.group(6)}"

    return _BREAKABLE_FIGURES.sub(hold, text)


def sample_settings_words(settings: Any) -> str:
    """A sample's settings in words: free text as given, or a mapping said in words; its
    sizes, seconds and seeds do not break across lines."""
    if not isinstance(settings, dict):
        return _held(str(settings or ""))
    words = []
    if settings.get("size"):
        words.append(_held(str(settings["size"])))
    elif settings.get("width") and settings.get("height"):
        words.append(_size(settings["width"], settings["height"]))
    if settings.get("seconds"):
        words.append(f"{float(settings['seconds']):g}\u00a0s")
    if settings.get("upscale"):
        words.append("upscaled ×2")
    if settings.get("enhance"):
        words.append("prompt enhanced")
    if settings.get("auto_duration", settings.get("auto")):
        words.append("length chosen by the model")
    return ", ".join(words)


def monitor_caption(
    *, kind: str, mode: Any = None, seed: Any = None, settings: Any = "", caption: str = ""
) -> str:
    """The line under the monitor saying whose output is on it."""
    if kind == "sample":
        extra = f" {caption}" if caption else ""
        said = sample_settings_words(settings)
        bits = ", ".join(b for b in (said, f"seed\u00a0{seed}" if seed is not None else "") if b)
        detail = f" {mode.label}" + (f", {bits}" if bits else "") + "."
        return f"**Example output, not your run.**{detail}{extra}"
    if kind == "run":
        return f"**Your run.** {mode.label}, seed\u00a0{seed}."
    if kind == "earlier":
        return f"**Earlier run.** {mode.label}, seed\u00a0{seed}."
    if kind == "running":
        return f"**Your run.** {mode.label}."
    if kind == "failed":
        return "**This run failed.** The report below says why."
    if kind == "refused":
        return "**Refused before running.** The report below says why."
    if kind == "unshown":
        return "**The result could not be shown.** The report below says why."
    return READY


def card_headline(report: str | None) -> str | None:
    """The headline of a failure card (its ``### `` line), or None for any other report."""
    first = str(report or "").lstrip().split("\n", 1)[0]
    return (first[4:].strip() or None) if first.startswith("### ") else None


def last_run_line(*, mode: Any, at: str, headline: str) -> str:
    """A failure card collapsed to one line once the page moves to another mode: whose run
    it was, when, and its headline."""
    return f"**Last run:** {mode.label}, {at}. {headline}."


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

#: The validator stage each error comes from, and the plain headline for it.
_LORA_STAGES: tuple[tuple[type, str, str], ...] = (
    (LoRAFetchError, "0", "fetch"),
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

_LORA_HEADLINES: tuple[tuple[type, str], ...] = (
    (LoRAFetchError, "Rejected: the adapter could not be fetched from the Hub"),
    (LoRATooLargeError, "Rejected: the file is larger than this app accepts"),
    (LoRAFormatError, "Rejected: this is not a .safetensors adapter"),
    (LoRAHeaderError, "Rejected: the file header could not be read"),
    (LoRATruncatedError, "Rejected: the file is cut short"),
    (LoRAVersionError, "Rejected: this adapter was made for a different model version"),
    (LoRAConventionError, "Rejected: the adapter's layer names are not ones this app knows"),
    (LoRAInContextError, "Rejected: this is an in-context (IC) LoRA, which this app can't apply"),
    (LoRAPairingError, "Rejected: the adapter's weights are incomplete"),
    (LoRADimensionError, "Rejected: the adapter's shapes don't fit LTX-2.5"),
    (LoRAModuleNotFoundError, "Rejected: the adapter targets layers LTX-2.5 doesn't have"),
    (LoRANotAttachedError, "Rejected: the adapter attached to nothing"),
)


def lora_stage_of(exc: BaseException) -> tuple[str, str] | None:
    """The validator stage an exception came from, or ``None`` if unmapped."""
    for error_type, number, name in _LORA_STAGES:
        if isinstance(exc, error_type):
            return number, name
    return None


def lora_no_trigger() -> str:
    """Under the LoRA row in the modes that take no prompt."""
    return (
        "This mode has no prompt, so adapters that need a trigger word will not show their "
        "effect. Use an image + text mode for those."
    )


def lora_heading(cap: str, *, on_zerogpu: bool = True) -> str:
    if not on_zerogpu:
        return (
            f"Add a LoRA from the Hub or upload a .safetensors file (up to {cap}). It is "
            "checked against LTX-2.5's layers before anything runs. The last check, that it "
            "attaches, runs when you generate; if that fails, the run stops there."
        )
    return (
        f"Add a LoRA from the Hub or upload a .safetensors file (up to {cap}). It is checked "
        "against LTX-2.5's layers before any GPU time is used. The last check, that it "
        "attaches, runs when you generate; if that fails, the run stops and the time used is "
        "billed."
    )


def lora_nothing_to_validate() -> str:
    return "Type a Hub repo id such as org/name, or upload a .safetensors file first."


def lora_validating(source: str, *, hub: bool) -> str:
    if hub:
        return (
            f"Fetching {source} from the Hub. Its size is checked against the limit before "
            f"anything downloads; the file is cached, so checking it again is instant."
        )
    return f"Reading the header of {source}…"


def lora_idle() -> str:
    return "No LoRA: runs use the base model."


def shown_name(source: str) -> str:
    """A local file by its name; a Hub id as typed."""
    text = str(source)
    return os.path.basename(text) if os.path.isabs(text) else text


def lora_accepted(source: str, report: Any) -> str:
    """An accepted adapter: a roll-edged card, its detected properties below."""
    name = shown_name(source)
    rows = [
        ("Layer names", getattr(report, "convention", None) or "not detected"),
        ("Model version", getattr(report, "model_version", None) or "not declared"),
        ("Rank", getattr(report, "rank", None) or "not declared"),
        ("Alpha", getattr(report, "alpha", None) or "not declared"),
    ]
    targeted = getattr(report, "modules_targeted", None)
    resolved = getattr(report, "modules_resolved", None)
    if targeted:
        rows.append(("Fits LTX-2.5", f"{resolved} of {targeted} layers"))
    if getattr(report, "hub_revision", None):
        rows.append(("Hub commit", str(report.hub_revision)[:12]))
    lines = [
        f"#### Accepted: {name}",
        "",
        "Every check before a run passed: the file reads cleanly and every layer fits its "
        "target in LTX-2.5. Staged for the next run; set its strength below (adapters are "
        "often made for 0.5 to 1.2).",
        "",
        "| | |",
        "|---|---|",
        *(f"| {label} | {value} |" for label, value in rows),
    ]
    notes = tuple(getattr(report, "warnings", ()) or ())
    if notes:
        lines += ["", "Good to know:", *(f"- {note}" for note in notes)]
    return "\n".join(lines)


def lora_duplicate(source: str, existing: dict) -> str:
    name = shown_name(source)
    shown = shown_name(existing.get("source", ""))
    return (
        f"#### Already staged: {name}\n\n"
        f"This is the same adapter as {shown}, which is already staged. Set its strength "
        f"instead of adding it twice."
    )


def lora_too_many(limit: int) -> str:
    return (
        f"#### Not added: {limit} adapters are already staged\n\nRemove one before adding another."
    )


def _lora_extra_headline(exc: BaseException) -> str | None:
    """Headlines for errors defined inside the LoRA package (imported only once it is loaded)."""
    from ..lora.manifest import LoRAManifestDriftError
    from ..lora.staging import LoRAWeightError

    if isinstance(exc, LoRAManifestDriftError):
        return "Not checked: this app's layer table is out of date, not your file"
    if isinstance(exc, LoRAWeightError):
        return "Not staged: that strength is out of range"
    return None


def lora_rejected(source: str, exc: BaseException) -> str:
    """A rejection: plain headline, the verbatim validator message underneath."""
    name = shown_name(source)
    headline = (_lora_extra_headline(exc) if isinstance(exc, LoRAError) else None) or next(
        (text for error_type, text in _LORA_HEADLINES if isinstance(exc, error_type)),
        "Rejected: the adapter could not be checked",
    )
    detail = _without_paths(str(exc).strip(), name) or "No detail was attached to the error."
    return f"### {headline}\n\n{name} was not staged. Nothing was spent.\n\n> {detail}"


def _without_paths(text: str, name: str) -> str:
    """The validator's words with every server path to the file shown as its name.

    The upload is checked from an app-owned copy, so a message quotes that copy's path;
    the user knows the file by its name.
    """
    import re

    return re.sub(r"(?:[A-Za-z]:)?[/\\][^\s'\"]*[/\\]" + re.escape(name), name, text)


def applied_loras(applied: list[Any]) -> str:
    """The staged list in one line, for the bay and the API."""
    if not applied:
        return lora_idle()
    parts = [
        f"{shown_name(a.get('source', a))} at {weight_text(a.get('weight', 1.0))}" for a in applied
    ]
    return f"Staged for the next run: {'; '.join(parts)}."


def lora_summary_label(applied: Sequence[dict]) -> str:
    """The always-visible LoRA row's label."""
    if not applied:
        return "LoRA: none. Add an adapter"
    weights = _and([weight_text(a.get("weight", 1.0)) for a in applied])
    noun = "adapter" if len(applied) == 1 else "adapters"
    return f"LoRA: {len(applied)} {noun} at {weights}"


# ---------------------------------------------------------------------------
# generation outcomes
# ---------------------------------------------------------------------------


def running(
    mode: Any, *, enhancing: bool = False, on_zerogpu: bool = True, device: str | None = None
) -> str:
    """Shown the moment Generate is pressed, so the page is never silent.

    On ZeroGPU a run first reserves GPU time from the user's quota. Anywhere else there is
    no quota and no reservation: it loads, samples and decodes on this machine, on
    ``device`` when known (the words of :func:`took_words`).
    """
    if not on_zerogpu:
        if enhancing:
            return (
                f"**Enhancing your prompt: {mode.label}.** Gemma rewrites it on this machine "
                f"first, then generation starts. The prompt actually sent is in the report."
            )
        return f"**Running: {mode.label}.** Loading, sampling and decoding {where_it_runs(device)}."
    if enhancing:
        return (
            f"**Enhancing your prompt: {mode.label}.** Gemma rewrites it on the GPU first, "
            f"inside this run's reservation, then generation starts. The prompt actually sent "
            f"is in the report."
        )
    return (
        f"**Running: {mode.label}.** Reserving GPU time from your quota, then loading, "
        f"sampling and decoding."
    )


@dataclass(frozen=True)
class RunFacts:
    """What the UI knows about a finished run, beyond the backend's result."""

    mode: Any
    seed: int
    returned_size: tuple[int, int] | None
    seconds: float | None
    stage_2: bool
    #: ``(name, weight)``, or ``(name, weight, layers attached)`` from the run record.
    loras: tuple[tuple, ...]
    took: float
    #: True when ``took`` is the worker's own GPU time, not the page's wall clock.
    took_is_gpu: bool = False
    prompt_sent: str | None = None
    #: "requested" | "model" | "audio": where the length came from.
    length_source: str | None = None
    #: A model-picked length at an end of the user's range: "shortest", "longest" or "only"
    #: (the range allows one length). There the range may have moved the model's figure.
    length_bound: str | None = None
    #: The level actually used to change the user's sound (Audio -> sound).
    audio_level: float | None = None
    #: The level actually used to change the user's clip (Video -> video).
    video_level: float | None = None
    #: The decoder that ran: "diffusion" or "conv".
    decoder: str | None = None
    #: The device the run's time was measured on ("cuda", "mps", "cpu"), when known.
    device: str | None = None
    #: On ZeroGPU the GPU time is drawn from the user's quota at ``quota_multiplier``.
    on_zerogpu: bool = False
    quota_multiplier: int = 1


def took_words(facts: RunFacts) -> str:
    """ "Took N s", labelled by the device that ran it; on ZeroGPU, with the quota it drew. A
    run of under half a second took "under 1 s", never "0 s"."""
    seconds = round(float(facts.took))
    under = seconds < 1
    spent = f"under {_num(1)}\u00a0s" if under else f"{_num(seconds)}\u00a0s"
    if not facts.took_is_gpu:
        return spent
    if facts.on_zerogpu:
        multiplier = int(facts.quota_multiplier)
        quota = (
            f"under {_num(multiplier)}\u00a0s"
            if under
            else f"about {_num(seconds * multiplier)}\u00a0s"
        )
        return f"{spent} of GPU time, {quota} of your quota"
    where = _ON_DEVICE.get(str(facts.device or ""))
    return f"{spent} {where}" if where else spent


#: Beside Reuse seed when a run returned its picture and its sound as two files.
SOUND_DOWNLOAD = "Download the sound (WAV)"
#: Beside Reuse seed when the prompt sent differs from the one typed (it was enhanced); once
#: pressed, what it did: the prompt is in the box and Enhance is off, so the next run sends
#: it as it is.
USE_PROMPT = "Use this prompt"
PROMPT_USED = "Copied to your prompt; Enhance is off"


def run_summary(facts: RunFacts) -> str:
    """The head of the success report: what you got, in one block."""
    rows = [("Mode", _e(facts.mode.label))]
    if facts.returned_size and "video" in facts.mode.exports:
        size = _size_html(facts.returned_size)
        rows.append(("Size", size + (", upscaled ×2" if facts.stage_2 else ", no upscale")))
    origin = {
        "model": ", chosen by the model",
        "audio": ", matched to your audio",
        "video": ", matched to your clip",
    }.get(facts.length_source or "", "")
    if facts.length_source == "model" and facts.length_bound:
        # Only a length strictly inside the range is the model's own figure for certain.
        origin = {
            "shortest": ", the shortest you allowed",
            "longest": ", the longest you allowed",
            "only": ", the only length you allowed",
        }.get(facts.length_bound, origin)
    if facts.seconds is not None:
        rows.append(("Length", _secs_html(facts.seconds) + origin))
    elif origin:
        rows.append(("Length", origin.lstrip(", ").capitalize()))
    for used, choices in ((facts.audio_level, NOISE_CHOICES), (facts.video_level, CLIP_CHOICES)):
        if used is not None:
            named = next((n for n, v in choices if abs(v - used) < 1e-6), None)
            level = _num(_two_places(used))
            rows.append(("Change", f"{_e(named)} ({level})" if named else level))
    rows.append(("Seed", _num(facts.seed)))
    rows.append(("LoRA", _lora_words(facts.loras)))
    if facts.decoder:
        rows.append(("Decoder", _e(DECODER_NAMES.get(facts.decoder, facts.decoder))))
    rows.append(("Took", took_words(facts)))
    grid = "".join(f"<dt>{label}</dt><dd>{value}</dd>" for label, value in rows)
    sent = ""
    if facts.prompt_sent:
        sent = (
            '<div class="mr-prompt-sent"><p class="mr-sub">Prompt sent</p>'
            f"<blockquote>{_e(facts.prompt_sent)}</blockquote></div>"
        )
    return (
        '<section class="mr-done" aria-label="What you got">'
        '<p class="mr-done-head"><span class="mr-lamp mr-lamp-roll is-lit">Done</span></p>'
        f'<dl class="mr-done-grid">{grid}</dl>{sent}</section>'
    )


def success(result: Any, mode: Any) -> str:
    """The report Markdown ``/on_generate`` returns for a finished run.

    A pure function of the result and the mode: the seed that was really used,
    every note the run returned as plain text, then a collapsed Diagnostics
    block. No post-run cost reminder.
    """
    seed = getattr(result, "seed", None)
    head = f"Done: {mode.label}." + (f" Seed {seed}." if seed is not None else "")
    lines = [f'<p class="mr-report-head">{_e(head)}</p>']
    notes = [str(n) for n in (getattr(result, "warnings", ()) or ()) if str(n).strip()]
    if notes:
        lines.append('<div class="mr-notes">')
        lines += [f"<p>{_e(n[0].upper() + n[1:])}</p>" for n in notes]
        lines.append("</div>")
    record = getattr(result, "record", None)
    diagnostics = [str(line) for line in (getattr(record, "diagnostics", None) or ())]
    peak = getattr(record, "peak_vram_gib", None)
    if peak is not None:
        diagnostics.append(f"Peak GPU memory reserved: {peak:.2f} GiB")
    run_dir = getattr(result, "run_dir", None)
    if run_dir and not any("run folder" in line.lower() for line in diagnostics):
        diagnostics.append(f"Run folder: {run_dir}")
    diagnostics.append(f"Mode id: {mode.id}")
    lines.append(
        '<details class="mr-diagnostics"><summary>Diagnostics</summary>'
        + "".join(f"<p>{_e(d)}</p>" for d in diagnostics)
        + "</details>"
    )
    return "\n".join(lines)


#: The Detail of a card for a run the page could not show, when the server raised nothing.
NO_SERVER_MESSAGE = "The server reported no error; this page could not show what it sent."


def _detail(text: str | None) -> str:
    return (
        '<details class="mr-detail"><summary>Detail</summary>'
        f"<p>{_e((text or '').strip() or NO_SERVER_MESSAGE)}</p></details>"
    )


def lost_run(detail: str | None = None, *, on_zerogpu: bool = True) -> str:
    """The generation step ended without a result this page can show.

    ``detail`` is the server's own message, when it raised one. The billing sentence is
    ZeroGPU's alone.
    """
    billed = "Any GPU time this run used is billed. " if on_zerogpu else ""
    return (
        "### The run ended without a result\n\n"
        f"{billed}Press Generate again. If it happens again, report it with the detail "
        "below.\n\n" + _detail(detail)
    )


def inputs_unread(detail: str | None = None, *, hours: int, on_zerogpu: bool = True) -> str:
    """The page could not hand the run its inputs, so nothing ran.

    Gradio reads every input before any of this app's code runs; an upload its cache
    clean-up removed (after ``hours``) is the cause a creator meets. ``detail`` is Gradio's
    own message. The GPU sentence is ZeroGPU's alone.
    """
    used = "Nothing ran. No GPU time was used." if on_zerogpu else "Nothing ran."
    return (
        "### Your inputs could not be read\n\n"
        f"{used} If you added an image, clip or audio more than {hours} hours ago, the app has "
        "cleared its copy: add it again, then press Generate.\n\n" + _detail(detail)
    )


def result_not_shown(facts: RunFacts, detail: str | None, *, kept: bool = False) -> str:
    """A run that finished, whose result the page could not show.

    What it used, by the device that ran it (billed on ZeroGPU), one next step, and
    ``detail``: the server's own message, when it raised one. When its file is kept the
    run is in Recent runs, and Show (which costs nothing) is the step; running again is
    the step only when the file is gone.
    """
    billed = ", which is billed" if facts.on_zerogpu else ""
    step = (
        "Its file is kept: press Show in Recent runs. If that fails too, report it with the "
        "detail below."
        if kept
        else "Press Generate again. If it happens again, report it with the detail below."
    )
    return (
        "### The run finished, but its result could not be shown\n\n"
        f"It used {took_words(facts)}{billed}. {step}\n\n" + _detail(detail)
    )


def empty_output_bug(mode: Any, missing: list[str], *, on_zerogpu: bool = True) -> str:
    """A run reported success and produced no file. That is a bug in this app."""
    tracks = " and ".join(missing)
    billed = " The GPU time it used is billed." if on_zerogpu else ""
    return (
        "### The run finished but produced no file\n\n"
        f"{mode.label} reported success without a {tracks} file, which this app treats as a "
        f"bug.{billed} Nothing shown here is from this run. Report it, with the time of the "
        "run, so it can be fixed."
    )


#: ``errors.categorise`` category -> ``(headline, next step)``. Keyed on the category's
#: string, so a category the backend adds ("limit", "queue") renders without an import,
#: and one the UI has never seen gets :data:`_GENERIC_CARD`.
_CARDS: dict[str, tuple[str, str]] = {
    "quota": (
        "Not enough GPU quota for this run",
        "To reserve less, turn off Upscale, Enhance or LoRA (the cost panel shows what each "
        "adds), or wait for your daily quota to refill.",
    ),
    "limit": (
        "This run asks for more GPU time than the platform allows in one run",
        "Turn off Enhance or Upscale to reserve less.",
    ),
    "queue": ("No GPU was free in time", "Press Generate again in a minute."),
    "timeout": (
        "The run ran out of GPU time",
        "Make the video shorter or smaller, or turn off Upscale, then press Generate again.",
    ),
    "oom": (
        "The GPU ran out of memory",
        "Make the run smaller: a lower size, fewer frames or Upscale off.",
    ),
    "input": (
        "The run was refused: fix the input",
        "Change what the detail below names, then press Generate again.",
    ),
    "adapter": (
        "An adapter doesn't fit this model",
        "Remove the adapter, or try a different one, and press Generate again.",
    ),
    "enhancer": (
        "The prompt enhancer is unavailable",
        "Turn Enhance off to run with your own prompt.",
    ),
    "bug": (
        "Something went wrong in this app",
        "Press Generate again. If it happens again, report it with the detail below.",
    ),
    # This machine's memory check declined a step before it started; the detail has the
    # numbers (in use, needed, the limit) and what to change.
    "memory": (
        "This machine does not have enough free memory for this run",
        "Close other apps, or make the run smaller, then press Generate again.",
    ),
}
_GENERIC_CARD = (
    "The run did not finish",
    "Press Generate again. If it happens again, report it with the detail below.",
)
#: The fields each card's step names: changing one acts on the step, and the card becomes
#: its one-line record (:func:`last_run_line`). ``"*"`` is any field (a refusal's detail
#: names the field). A card not listed names no control (press Generate again): it stays
#: until the next run.
_STEP_FIELDS: dict[str, tuple[str, ...]] = {
    "quota": ("enable_stage_2", "enhance_prompt", "loras"),
    "limit": ("enhance_prompt", "enable_stage_2"),
    "timeout": ("num_frames", "width", "height", "enable_stage_2"),
    "oom": ("width", "height", "num_frames", "enable_stage_2"),
    "input": ("*",),
    "adapter": ("loras",),
    "enhancer": ("enhance_prompt",),
    "memory": ("width", "height", "num_frames", "enable_stage_2"),
}
#: Cards whose headline is not a ``_CARDS`` one.
_HEADLINE_STEP_FIELDS: dict[str, tuple[str, ...]] = {
    "Refused before running: fix the input": ("*",),
    "Your inputs could not be read": ("image", "video", "audio"),
}


def step_names(headline: str | None, key: str) -> bool:
    """Whether the step of the failure card headed ``headline`` names the field ``key``."""
    fields = _HEADLINE_STEP_FIELDS.get(headline or "")
    if fields is None:
        category = next((c for c, (title, _step) in _CARDS.items() if title == headline), "")
        fields = _STEP_FIELDS.get(category, ())
    return "*" in fields or key in fields


#: Cards about the platform's meter. Off ZeroGPU there is none, so such a failure (it
#: cannot come from a local run) gets the generic card.
_METER_CARDS = ("quota", "limit")
#: Cards whose detail opens by itself: the detail holds the numbers to change.
_OPEN_DETAIL = ("input", "quota", "limit", "memory")


def failure(exc: BaseException, *, refused: bool = False, on_zerogpu: bool = True) -> str:
    """Any failure, rendered loudly: headline, quota line, one next step, detail.

    The category and whether GPU time was used come from ``errors.categorise``.
    ``refused`` is set when this page turned the request down before calling
    the backend at all, which is known to have used no GPU time. The quota line is
    ZeroGPU's; where nothing is metered a card says only that nothing ran, when it was
    refused before running.
    """
    detail = str(exc).strip() or "No detail was attached to the error."
    found = errors.categorise(exc)
    category = str(found.category)
    gpu_used = found.gpu_used and not refused
    if category == "timeout" and not gpu_used:
        # A queue timeout: nothing ran, so it is not "out of GPU time".
        category = "queue"
    headline, step = _CARDS.get(category, _GENERIC_CARD)
    if not on_zerogpu and category in _METER_CARDS:
        headline, step = _GENERIC_CARD
    if category == "input" and not gpu_used:
        headline = "Refused before running: fix the input"
    if on_zerogpu:
        quota = (
            "Any GPU time this run used is billed."
            if gpu_used
            else "Nothing ran. No GPU time was used."
        )
    else:
        quota = "Nothing ran." if refused or (category == "input" and not gpu_used) else ""
    opened = " open" if category in _OPEN_DETAIL else ""
    return (
        f"### {headline}\n\n"
        f"{quota + ' ' if quota else ''}{step}\n\n"
        f'<details class="mr-detail"{opened}><summary>Detail</summary>'
        f"<p>{_e(detail)}</p></details>"
    )


def recent_run_label(
    *,
    mode: Any,
    seed: int,
    size: tuple[int, int] | None,
    seconds: float | None,
    at: str,
) -> str:
    bits = [mode.label]
    if size and "video" in mode.exports:
        bits.append(_size(*size))
    if seconds is not None:
        bits.append(_seconds(seconds))
    bits.append(f"seed\u00a0{seed}")
    return f"**{at}** {', '.join(bits)}"


def example_label() -> str:
    return "Fills the form. Nothing runs until you press Generate."
