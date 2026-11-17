"""Thin adapters: a UI event, one call into the backend, plain values back.

**No Gradio here.** Handlers return plain values and small dataclasses;
``blocks.py`` is the only place that turns one into a ``gr.update``. So every
decision this layer makes is testable on a bare interpreter.

**No handler returns a no-op.** Every path produces a concrete value for every
output and a non-empty status: an untouched output is what a silent reset looks
like.

**The rules live in the backend.** What a mode accepts and requires comes from
``modes.accepts`` / ``modes.requires``; whether a request would be refused comes
from ``generate.prepare_request`` -- the same pre-dispatch check ``generate()``
runs -- and the reservation from ``zerogpu.estimate_duration``. This module only
asks them and marshals the answers.

Heavy imports are deferred into call bodies so the module stays importable
without torch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .. import constraints, modes
from . import render, uploads
from .fields import FIELD_KEYS, FIELDS, PARKED_FIELDS, SHOT_ANCHOR_COLUMN, TOGGLEABLE_FIELDS

__all__ = [
    "MAX_STAGED_LORAS",
    "FieldUpdate",
    "ModeView",
    "RunOutcome",
    "clip_seconds",
    "example_values",
    "export_flags",
    "frames_for_seconds",
    "has_result",
    "legal_size",
    "match_size",
    "media_size",
    "mode_switch",
    "mode_view",
    "on_auto_duration_toggle",
    "on_clear_loras",
    "on_generate",
    "on_generate_start",
    "on_lora_weight",
    "on_mode_change",
    "on_remove_lora",
    "on_shape_change",
    "on_shots_change",
    "on_snap",
    "on_validate_lora",
    "on_validate_start",
    "ordered_seconds",
    "preset_for",
    "quota_multiplier",
    "ran_nothing",
    "review",
    "run",
    "run_platform",
    "seconds_for_frames",
    "unshown_report",
]

#: The bay pre-builds this many adapter rows.
MAX_STAGED_LORAS = 3

FRAME_RATE = constraints.DEFAULT_FRAME_RATE


# ---------------------------------------------------------------------------
# input coercion
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    """Gradio numbers arrive as ``float``, or as ``None`` when the box is empty."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _defaults() -> dict[str, Any]:
    return {spec.key: spec.default for spec in FIELDS}


# ---------------------------------------------------------------------------
# what the selected mode shows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldUpdate:
    """What one field's control should look like for the selected mode.

    ``value`` is always ``None`` here: the mode selector never writes a value
    into a field the user owns. Moving hidden values aside is
    :func:`mode_switch`'s job, and it only ever restores the user's own.
    """

    visible: bool = True
    interactive: bool = True
    value: Any = None


@dataclass(frozen=True)
class ModeView:
    """Everything the page shows differently for one mode."""

    mode_id: str
    name: str
    purpose: str
    active: dict[str, bool]
    prompt_required: bool
    prompt_label: str
    prompt_placeholder: str
    captions: dict[str, str]
    timed_input: bool
    timed_by_audio: bool
    pinned_line: str | None
    match_input: bool
    sizes_picture: bool
    upscale: bool
    #: The change choice for the user's sound (Audio -> sound) or clip (Video -> video).
    varies_sound: bool = False
    varies_clip: bool = False
    #: The length is always the clip's (a clip variation): no length choice is offered.
    length_from_clip: bool = False


def mode_view(mode_id: str) -> ModeView:
    """The page's shape for ``mode_id``, read off the registry."""
    mode = modes.get_mode(mode_id)
    active = {
        spec.key: spec.visible_when is None or bool(spec.visible_when(mode)) for spec in FIELDS
    }
    captions = {
        kind: render.media_caption(mode, kind)
        for kind in ("image", "video", "audio")
        if modes.accepts(mode, kind)
    }
    timed = modes.accepts(mode, "video") or modes.requires(mode, "audio")
    return ModeView(
        mode_id=mode.id,
        name=mode.label,
        purpose=render.purpose_line(mode),
        active=active,
        prompt_required=modes.requires(mode, "text"),
        prompt_label=render.prompt_label(mode, required=modes.requires(mode, "text")),
        prompt_placeholder=render.prompt_placeholder(mode),
        captions=captions,
        timed_input=timed,
        timed_by_audio=modes.requires(mode, "audio"),
        pinned_line=(
            render.pinned_line(mode, modes.MIN_RESOLUTION) if mode.min_resolution else None
        ),
        match_input=not mode.min_resolution
        and (modes.accepts(mode, "image") or modes.accepts(mode, "video")),
        sizes_picture=not mode.min_resolution,
        upscale=not mode.skip_stage_2,
        varies_sound=mode.audio_conditioning == "variation",
        varies_clip=getattr(mode, "video_conditioning", None) == "variation",
        length_from_clip=getattr(mode, "video_conditioning", None) == "variation",
    )


#: Tie-break for :func:`media_order` when the registry leaves two kinds unordered.
_MEDIA_PREFERENCE = ("audio", "video", "image")


def media_order(specs: Any = None) -> tuple[str, ...]:
    """The order the media inputs are built in, read off the registry.

    In every mode the inputs it requires come before the ones it may take, so
    the first media input a mode shows is one it needs. The page is built once,
    so one order must hold for every mode: each mode adds "required before
    optional" constraints and this is their topological order. A registry no
    single order can satisfy raises here, at build time.
    """
    specs = tuple(modes.MODES.values()) if specs is None else tuple(specs)
    after: dict[str, set[str]] = {kind: set() for kind in _MEDIA_PREFERENCE}
    for spec in specs:
        required = [k for k in _MEDIA_PREFERENCE if modes.requires(spec, k)]
        optional = [k for k in _MEDIA_PREFERENCE if modes.accepts(spec, k) and k not in required]
        for kind in optional:
            after[kind].update(required)
    order: list[str] = []
    while len(order) < len(_MEDIA_PREFERENCE):
        ready = [k for k in _MEDIA_PREFERENCE if k not in order and after[k] <= set(order)]
        if not ready:
            raise RuntimeError(
                "no single order of the media inputs puts every mode's required inputs before "
                f"its optional ones: {sorted(set(_MEDIA_PREFERENCE) - set(order))}"
            )
        order.append(ready[0])
    return tuple(order)


def media_label(mode: Any, kind: str) -> str:
    """A media input's label: the kind, marked optional where the mode only may take it."""
    from .fields import field_by_key

    base = field_by_key(kind).label
    return base if modes.requires(mode, kind) else f"{base} (optional)"


def decoder_view(mode_id: str, upscale: Any, choice: Any = "auto") -> tuple[bool, bool, str]:
    """``(visible, interactive, line)`` for the decoder choice.

    Upscale ×2 always finishes with the model's diffusion decoder, whatever the
    choice (``pipelines.StageRunner``), so while it is on the choice is not a
    lever: it is shown but cannot be changed, and the line says why.
    """
    mode = modes.get_mode(mode_id)
    visible = "video" in mode.exports
    upscaling = bool(upscale) and not mode.skip_stage_2
    return visible, not upscaling, render.decoder_line(choice, upscaling=upscaling)


def length_fields_view(mode_id: str, auto: Any) -> dict[str, bool]:
    """Which exact-length fields Advanced shows: the frame count while the length is set,
    the shortest and longest seconds while the model picks it."""
    view = mode_view(mode_id)
    picking = bool(auto)
    return {
        "num_frames": not picking and not view.length_from_clip,
        "min_seconds": view.active["min_seconds"] and picking,
        "max_seconds": view.active["max_seconds"] and picking,
    }


def on_mode_change(mode_id: str) -> tuple[str, tuple[FieldUpdate, ...]]:
    """The purpose line plus one :class:`FieldUpdate` per toggleable field."""
    try:
        view = mode_view(mode_id)
    except Exception as exc:  # noqa: BLE001 -- an unknown mode must still render
        return render.failure(exc), tuple(FieldUpdate() for _ in TOGGLEABLE_FIELDS)
    return view.purpose, tuple(FieldUpdate(visible=view.active[f.key]) for f in TOGGLEABLE_FIELDS)


def export_flags(mode_id: str) -> tuple[bool, bool]:
    """Whether ``(video, audio)`` players apply to this mode, from the registry."""
    try:
        exports = modes.get_mode(mode_id).exports
    except Exception:  # noqa: BLE001 -- show both rather than hide the failure
        return (True, True)
    return ("video" in exports, "audio" in exports)


def mode_switch(
    previous_mode: str | None, mode_id: str, values: dict[str, Any], parked: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move hidden values aside and bring back the user's own.

    For every field that holds the user's own input: when the new mode does not
    use it, a non-default value is parked and the field goes back to its
    default, so a run never submits a value the user cannot see. When a mode
    that uses it is selected again, the parked value comes back. A field that
    was not in use and has nothing parked returns to its default.

    Returns ``(values, parked)`` -- new dicts; the inputs are not modified.
    """
    parked = dict(parked or {})
    out = dict(values)
    defaults = _defaults()
    new = modes.get_mode(mode_id)
    old = modes.get_mode(previous_mode) if previous_mode in modes.MODES else None
    for spec in PARKED_FIELDS:
        now = bool(spec.visible_when(new))
        before = bool(spec.visible_when(old)) if old is not None else now
        value = out.get(spec.key, spec.default)
        if not now:
            if before and not _is_default(value, defaults[spec.key]):
                parked[spec.key] = value
            out[spec.key] = _fresh(defaults[spec.key])
        elif not before:
            out[spec.key] = parked.pop(spec.key, _fresh(defaults[spec.key]))
    return out, parked


def _is_default(value: Any, default: Any) -> bool:
    if value is None:
        return default is None
    if isinstance(default, list):
        return not value or value == default or (isinstance(value, dict) and not value.get("data"))
    try:
        return bool(value == default)
    except Exception:  # noqa: BLE001 -- an image compares by identity
        return False


def _fresh(default: Any) -> Any:
    return list(default) if isinstance(default, list) else default


# ---------------------------------------------------------------------------
# shape and length
# ---------------------------------------------------------------------------


def preset_for(width: Any, height: Any) -> str | None:
    """The preset name for an exact size, or ``None`` for a custom one."""
    w, h = _as_int(width), _as_int(height)
    for name, pw, ph in render.SHAPE_PRESETS:
        if (pw, ph) == (w, h):
            return name
    return None


def preset_size(name: str) -> tuple[int, int] | None:
    for preset, w, h in render.SHAPE_PRESETS:
        if preset == name:
            return (w, h)
    return None


def legal_size(width: Any, height: Any) -> tuple[int, int] | None:
    """``(width, height)`` when the backend takes this size (its own validators), else None."""
    w, h = _as_int(width), _as_int(height)
    if w is None or h is None:
        return None
    try:
        for name, value in (("width", w), ("height", h)):
            constraints.validate_dimension_limit(value, name=name)
            constraints.validate_dimension(value, name=name)
    except constraints.ConstraintError:
        return None
    return (w, h)


def match_size(source_width: int, source_height: int) -> tuple[int, int]:
    """The legal size with the input's aspect ratio and about the default's pixel count."""
    _name, base_w, base_h = render.SHAPE_PRESETS[0]
    aspect = source_width / source_height
    height = (base_w * base_h / aspect) ** 0.5
    width = height * aspect
    limit = constraints.MAX_DIMENSION
    width, height = min(width, limit), min(height, limit)
    return constraints.snap_dimensions(round(width), round(height))


def media_size(image: Any = None, video: Any = None) -> tuple[int, int] | None:
    """Pixel size of an image (PIL) or the first video stream, read in the web process."""
    if image is not None and hasattr(image, "size"):
        w, h = image.size
        return int(w), int(h)
    if video:
        try:
            import av

            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                return int(stream.width), int(stream.height)
        except Exception:  # noqa: BLE001 -- an unreadable clip has no size to match
            return None
    return None


def clip_seconds(path: Any, *, kind: str = "video") -> float | None:
    """A clip's length in seconds as the backend plans the run: an audio file by its sound,
    a video by its picture (``generate._audio_seconds`` / ``generate._video_seconds``), so
    "Matches your N s clip" says what will run. ``None`` for a file the backend refuses
    (the review names the problem)."""
    if not path:
        return None
    from .. import generate as generate_module

    measure = generate_module._audio_seconds if kind == "audio" else generate_module._video_seconds
    try:
        seconds = measure(path)
    except constraints.ConstraintError:
        return None
    return float(seconds) if seconds else None


#: The page's step for the shortest and longest seconds (their boxes step by 0.1).
SECONDS_STEP = 0.1


def ordered_seconds(shortest: Any, longest: Any, *, edited: str) -> tuple[Any, Any]:
    """Shortest and Longest as the page keeps them, for the model-chosen length.

    Longest stops at the longest clip the model makes (``constraints.MAX_SECONDS``) and
    Shortest stays below Longest; when they cross, the field the user did not just edit
    (``edited`` is ``"shortest"`` or ``"longest"``) moves by one step. So neither of the
    backend's two refusals in field names ("max_seconds=25.0 is above the maximum",
    "max_seconds must be greater than min_seconds") can be reached from the page.
    """
    low, high = _as_float(shortest), _as_float(longest)
    top = float(constraints.MAX_SECONDS)
    if high is not None:
        high = min(high, top)
    if low is not None:
        low = min(low, round(top - SECONDS_STEP, 1))
    if low is not None and high is not None and low >= high:
        if edited == "longest":
            low = round(high - SECONDS_STEP, 1)
        else:
            high = round(min(low + SECONDS_STEP, top), 1)
    return (shortest if low is None else low), (longest if high is None else high)


def frames_for_seconds(seconds: Any) -> int:
    """The legal frame count nearest to ``seconds`` at the generation frame rate."""
    value = _as_float(seconds) or 0.0
    frames = round(value * FRAME_RATE)
    frames = min(max(frames, constraints.MIN_NUM_FRAMES), constraints.MAX_NUM_FRAMES)
    return constraints.snap_num_frames(frames)


def seconds_for_frames(frames: Any) -> float:
    """Seconds for the length slider, to its 0.1 s step (121 frames read 5.0, not 5.04)."""
    count = _as_int(frames) or 0
    return round(count / FRAME_RATE, 1)


def frames_matching_clip(seconds: float | None) -> int | None:
    if seconds is None:
        return None
    return constraints.largest_num_frames_within(seconds, frame_rate=FRAME_RATE)


def example_values(example: Any, mode_id: str) -> dict[str, Any]:
    """What clicking an example writes, and nothing more.

    Only the fields the example names are written: its prompt, its media, and
    each setting it lists. Everything else on the form -- the user's other
    media, staged adapters, settings the example does not mention -- stays as
    it is. Keys are field keys, plus ``"preset"``, ``"seconds"``,
    ``"noise_choice"`` and ``"length_source"`` for the controls that write them.
    """
    view = mode_view(mode_id)
    settings = dict(getattr(example, "settings", {}) or {})
    out: dict[str, Any] = {}
    if getattr(example, "prompt", "") and view.active["prompt"]:
        out["prompt"] = example.prompt
    if "enhance" in settings and view.active["enhance_prompt"]:
        out["enhance_prompt"] = bool(settings["enhance"])
    for kind, path in (getattr(example, "media", {}) or {}).items():
        if view.active.get(kind):
            out[kind] = str(path)
    if view.sizes_picture:
        size = None
        if "width" in settings and "height" in settings:
            size = (int(settings["width"]), int(settings["height"]))
        elif "shape" in settings:
            size = preset_size(str(settings["shape"]))
        if size:
            out["width"], out["height"] = size
            out["preset"] = preset_for(*size)
    if "seconds" in settings and not view.length_from_clip:
        frames = frames_for_seconds(settings["seconds"])
        out["num_frames"] = frames
        out["seconds"] = seconds_for_frames(frames)
        if view.active["auto_duration"]:
            out["auto_duration"] = False
        if view.timed_input:
            out["length_source"] = render.LENGTH_SET
    auto = settings.get("auto_duration", settings.get("auto"))
    if auto is not None and view.active["auto_duration"]:
        out["auto_duration"] = bool(auto)
    if "upscale" in settings and view.upscale:
        out["enable_stage_2"] = bool(settings["upscale"])
    if "seed" in settings:
        out["seed"] = int(settings["seed"])
    if "strength" in settings and view.active["strength"]:
        out["strength"] = float(settings["strength"])
    named = settings.get("noise")
    if view.varies_sound and named in dict(render.NOISE_CHOICES):
        out["noise_scale"] = dict(render.NOISE_CHOICES)[named]
        out["noise_choice"] = named
    if view.varies_clip and named in dict(render.CLIP_CHOICES):
        out["noise_scale"] = dict(render.CLIP_CHOICES)[named]
        out["clip_choice"] = named
    return out


# ---------------------------------------------------------------------------
# live constraint guidance, in Advanced
# ---------------------------------------------------------------------------


def on_shape_change(width: Any, height: Any, num_frames: Any) -> tuple[str, str]:
    """Hints for the exact size and frame count, naming the legal values."""
    return (_dimension_hint(width, height), _frame_hint(num_frames))


def _snapped_size(w: int, h: int) -> tuple[int, int]:
    limit = constraints.MAX_DIMENSION
    return tuple(min(value, limit) for value in constraints.snap_dimensions(w, h))


def _dimension_hint(width: Any, height: Any) -> str:
    w, h = _as_int(width), _as_int(height)
    if w is None or h is None:
        return "Width and height must both be numbers."
    over = [
        (name, value)
        for name, value in (("Width", w), ("Height", h))
        if value > constraints.MAX_DIMENSION
    ]
    if over:
        return render.dimension_limit_hint(over, limit=constraints.MAX_DIMENSION)
    return render.dimension_hint(
        w, h, multiple=constraints.DIMENSION_MULTIPLE, snapped=_snapped_size(w, h)
    )


def _frame_hint(num_frames: Any) -> str:
    frames = _as_int(num_frames)
    if frames is None:
        return "The frame count must be a number."
    if frames > constraints.MAX_NUM_FRAMES:
        return render.frame_limit_hint(
            frames,
            limit=constraints.MAX_NUM_FRAMES,
            seconds=constraints.MAX_NUM_FRAMES / FRAME_RATE,
        )
    lower, upper = constraints.nearest_valid_num_frames(frames)
    return render.frame_hint(
        frames,
        ratio=constraints.DEFAULT_TEMPORAL_COMPRESSION_RATIO,
        lower=lower,
        upper=upper,
        valid=constraints.is_valid_num_frames(frames),
    )


def on_snap(width: Any, height: Any, num_frames: Any) -> tuple[int, int, int, str, str]:
    """Snap all three to the nearest legal value; the user presses for it."""
    w, h = _snapped_size(
        _as_int(width) or constraints.DIMENSION_MULTIPLE,
        _as_int(height) or constraints.DIMENSION_MULTIPLE,
    )
    frames = min(constraints.snap_num_frames(_as_int(num_frames) or 1), constraints.MAX_NUM_FRAMES)
    return (w, h, frames, _dimension_hint(w, h), _frame_hint(frames))


def on_auto_duration_toggle(auto: Any) -> tuple[bool, bool, bool]:
    """Visibility for (num_frames, min_seconds, max_seconds) in a mode where the model may
    pick the length; :func:`length_fields_view` is the per-mode answer."""
    chosen = bool(auto)
    return (not chosen, chosen, chosen)


def on_shots_change(rows: Any, num_frames: Any) -> str:
    """Where each shot-list frame anchor lands (the API's shot list).

    ``conditioning.py`` owns the conversion and the range check.
    """
    from .. import conditioning

    frames = _as_int(num_frames) or 0
    anchors: list[dict] = []
    out_of_range: list[int] = []
    for row in rows or []:
        cells = row if isinstance(row, (list, tuple)) else ()
        frame = _as_int(cells[SHOT_ANCHOR_COLUMN] if len(cells) > SHOT_ANCHOR_COLUMN else None)
        if frame is None:
            continue
        group = conditioning.latent_group_for(frame)
        anchors.append({"frame": frame, **group})
        if frames and not 0 <= frame < frames:
            out_of_range.append(frame)
    return render.shot_anchor_readout(anchors, out_of_range=out_of_range)


# ---------------------------------------------------------------------------
# readiness and cost -- asked of the backend, never restated
# ---------------------------------------------------------------------------


def quota_multiplier() -> int:
    """Quota seconds per GPU second, for the configured ZeroGPU size."""
    from .. import zerogpu

    return int(zerogpu.quota_multiplier())


def _shape_problems(
    width: Any, height: Any, num_frames: Any, auto: bool, *, sizes: bool = True
) -> list[tuple[str, str, str]]:
    """Every size and length problem as ``(what, short form, full sentence)``, each with
    its nearest legal values. ``sizes`` is False where the mode renders at a pinned size:
    its width and height are not its input."""
    problems = []
    w, h = _as_int(width), _as_int(height)
    if sizes and (w is None or h is None):
        problems.append(("size", "size not a number", "Width and height must both be numbers."))
    elif sizes:
        for name, value in (("Width", w), ("Height", h)):
            try:
                constraints.validate_dimension_limit(value, name=name.lower())
                constraints.validate_dimension(value, name=name.lower())
            except constraints.ConstraintError:
                snapped = min(constraints.snap_dimension(value), constraints.MAX_DIMENSION)
                problems.append(
                    (
                        name.lower(),
                        f"{name.lower()} {value}, use {snapped}",
                        f"{name} {value} won't run: use {snapped} (multiples of "
                        f"{constraints.DIMENSION_MULTIPLE}, up to {constraints.MAX_DIMENSION}).",
                    )
                )
    if not auto:
        frames = _as_int(num_frames)
        if frames is None:
            problems.append(("frames", "frames not a number", "The frame count must be a number."))
        elif frames > constraints.MAX_NUM_FRAMES:
            problems.append(
                (
                    "frames",
                    f"{frames} frames, use {constraints.MAX_NUM_FRAMES}",
                    f"{frames} frames is longer than this model makes: use at most "
                    f"{constraints.MAX_NUM_FRAMES} ({constraints.MAX_SECONDS:g} s).",
                )
            )
        elif not constraints.is_valid_num_frames(frames):
            lower, upper = constraints.nearest_valid_num_frames(frames)
            problems.append(
                (
                    "frames",
                    f"{frames} frames, use {lower} or {upper}",
                    f"{frames} frames won't run: use {lower} ({lower / FRAME_RATE:.1f} s) or "
                    f"{upper} ({upper / FRAME_RATE:.1f} s).",
                )
            )
    return problems


def _request(mode_id: str, values: dict[str, Any]):
    from .. import generate as generate_module

    return generate_module.GenerationRequest(mode=mode_id, **values)


def _upscale_refusal(mode: Any, request: Any) -> str | None:
    """Why Upscale ×2 cannot run on this machine for ``request``, or None.

    The backend's own gate decides: ``generate._checked_stage_2_size``, the step of
    ``prepare_request`` that refuses a stage-2 request this device cannot decode, asked with
    Upscale on. Called on its own because ``prepare_request`` stops at the first problem (a
    missing prompt), and the control is parked before the form is complete.
    """
    if mode.skip_stage_2 or request is None:
        return None
    from dataclasses import replace

    from .. import device as device_module
    from .. import generate as generate_module

    try:
        generate_module._checked_stage_2_size(replace(request, enable_stage_2=True), mode)
    except device_module.StageTwoNeedsCudaError as exc:
        return render.upscale_needs_cuda(getattr(exc, "device", None))
    except Exception:  # noqa: BLE001 -- any other refusal is the panel's to state
        return None
    return None


def _staged_names(loras: Any) -> tuple[tuple[str, float], ...]:
    out = []
    for entry in loras or ():
        if isinstance(entry, dict):
            source = str(entry.get("source", ""))
            out.append((render.shown_name(source), float(entry.get("weight", 1.0))))
    return tuple(out)


def review(
    mode_id: str, values: dict[str, Any], *, length_source: str | None = None
) -> render.CostFacts:
    """What this request would render, return and reserve -- or why it won't run.

    Readiness is the backend's own answer: ``generate.prepare_request`` is the
    check ``generate()`` runs before it reserves a GPU, so the reason shown under
    a disabled Generate is the refusal the server would give, word for word.

    ``length_source`` is the page's Match my clip / Set the length choice: with
    Match and no clip loaded yet, the length is "the length of your clip", not
    the frame count the form happens to hold.
    """
    from .. import generate as generate_module
    from .. import pipelines, zerogpu
    from ..errors import MissingInputError

    mode = modes.get_mode(mode_id)
    name = mode.label
    multiplier = quota_multiplier()
    device_name, on_zerogpu = run_platform()
    platform = {"on_zerogpu": on_zerogpu, "device": device_name}
    try:
        request = _request(mode_id, values)
    except Exception as exc:  # noqa: BLE001
        return render.CostFacts(mode_name=name, state="refused", reason=str(exc), **platform)

    # A mode that renders at its pinned size does not take the size boxes.
    shape = _shape_problems(
        request.width,
        request.height,
        request.num_frames,
        bool(request.auto_duration),
        sizes=not mode.min_resolution,
    )
    problems = [sentence for _what, _short, sentence in shape]

    state, reason, prepared, notes, missing = "ready", None, request, (), ()
    try:
        prepared, _spec, notes = generate_module.prepare_request(request)
    except MissingInputError as exc:
        state, reason = "incomplete", str(exc)
        missing = tuple(getattr(exc, "missing", ()) or ())
        try:
            prepared, _ = generate_module.scope_request_to_mode(request, mode)
        except Exception:  # noqa: BLE001
            prepared = request
    except Exception as exc:  # noqa: BLE001 -- every refusal is shown, verbatim
        state, reason = "refused", str(exc)
    if problems:
        state = "refused"

    upscale_refused = _upscale_refusal(mode, prepared)
    if state == "refused":
        return render.CostFacts(
            mode_name=name,
            state="refused",
            reason=reason,
            problems=tuple(problems) or ((reason,) if reason else ()),
            short_problems=tuple((what, short) for what, short, _sentence in shape),
            stage_2_offered=not mode.skip_stage_2,
            enhance_offered=modes.accepts(mode, "text"),
            multiplier=multiplier,
            loras=_staged_names(request.loras),
            upscale_refused=upscale_refused,
            **platform,
        )

    stage_2 = generate_module.stage_2_enabled(mode, prepared)
    if mode.min_resolution:
        render_size = (modes.MIN_RESOLUTION, modes.MIN_RESOLUTION)
    else:
        render_size = (int(prepared.width), int(prepared.height))
    scale = pipelines.STAGE_2_SPATIAL_SCALE if stage_2 else 1
    returned = (render_size[0] * scale, render_size[1] * scale)

    auto = bool(prepared.auto_duration)
    from_audio = auto and modes.requires(mode, "audio")
    frames = None if auto else int(prepared.num_frames)
    # A clip mode matching its clip, with no clip loaded yet: the length is the clip's.
    pending = None
    if (
        modes.accepts(mode, "video")
        and request.video is None
        and (length_source in (None, render.LENGTH_MATCH) and not auto)
    ):
        pending, frames = "clip", None
    seconds = None if frames is None else frames / float(prepared.frame_rate)

    enhance = bool(prepared.enhance_prompt) and bool((prepared.prompt or "").strip())
    loras = _staged_names(prepared.loras)
    # The reservation, and each extra's share of it, from the estimator alone.
    steps = [
        ("Upscale ×2", {"enable_stage_2": stage_2}),
        ("LoRA", {"loras": tuple(prepared.loras or ())}),
        ("Enhance", {"enhance_prompt": enhance}),
    ]
    so_far: dict[str, Any] = {
        "request": prepared,
        "enable_stage_2": False,
        "loras": (),
        "enhance_prompt": False,
    }
    base = zerogpu.estimate_duration(**so_far)
    previous = base
    extras = []
    for label, lever in steps:
        so_far.update(lever)
        total = zerogpu.estimate_duration(**so_far)
        if total != previous:
            extras.append((label, total - previous))
        previous = total
    reserve = zerogpu.estimate_duration(
        request=prepared,
        enable_stage_2=stage_2,
        loras=tuple(prepared.loras or ()),
        enhance_prompt=enhance,
    )

    exports = set(mode.exports)
    returns = "audio+video" if exports == {"audio", "video"} else next(iter(exports))
    seed = _as_int(values.get("seed"))
    return render.CostFacts(
        mode_name=name,
        state=state,
        reason=reason,
        render_size=render_size,
        pinned=bool(mode.min_resolution),
        frames=frames,
        seconds=seconds,
        auto_seconds=(
            (float(prepared.min_seconds), float(prepared.max_seconds))
            if auto and not from_audio
            else None
        ),
        length_from_audio=from_audio,
        returns=returns,
        returned_size=returned,
        stage_2=stage_2,
        stage_2_offered=not mode.skip_stage_2,
        enhance=enhance,
        enhance_offered=modes.accepts(mode, "text"),
        reserve=reserve,
        base=base,
        extras=tuple(extras),
        multiplier=multiplier,
        loras=loras,
        discards_video="video" not in exports,
        encodes_audio=modes.requires(mode, "audio") and "video" in exports,
        seed=seed,
        notes=tuple(notes),
        missing=missing,
        length_pending=pending,
        upscale_refused=upscale_refused,
        **platform,
    )


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


def on_validate_start(repo_id: str | None, upload: Any) -> str:
    """Shown the instant a check starts. Same source precedence as the check."""
    import os

    if upload:
        return render.lora_validating(os.path.basename(str(upload)), hub=False)
    source = (repo_id or "").strip()
    if not source:
        return render.lora_nothing_to_validate()
    return render.lora_validating(source, hub=True)


def on_validate_lora(
    repo_id: str | None, upload: Any, applied: list[dict[str, Any]] | None
) -> tuple[str, list[dict[str, Any]], str]:
    """Check one adapter and stage it for the next run.

    An upload is copied into an app-owned folder first, so Gradio's cache
    clean-up cannot remove it between the check and Generate. A Hub id goes
    through ``lora.fetch_lora`` here, in the web process, so no download ever
    lands inside the billed GPU window. The same adapter (by content) is never
    staged twice. Rejections quote the validator verbatim.
    """
    staged = list(applied or [])
    source = str(upload) if upload else (repo_id or "").strip()
    if not source:
        return (render.lora_nothing_to_validate(), staged, render.applied_loras(staged))

    from .. import lora

    copy = None
    try:
        if upload:
            copy = uploads.own_copy(source)
            report = lora.validate_lora(str(copy), upload=True)
        else:
            report = lora.validate_lora(source, fetcher=lora.fetch_lora, upload=False)
        entry = lora.staged_entry(source, report, weight=lora.staging.DEFAULT_WEIGHT)
    except Exception as exc:  # noqa: BLE001 -- every failure must reach the user
        _discard(copy)
        return (render.lora_rejected(source, exc), staged, render.applied_loras(staged))

    existing = lora.find_duplicate(staged, entry)
    if existing is not None:
        _discard(copy)
        return (render.lora_duplicate(source, existing), staged, render.applied_loras(staged))
    if len(staged) >= MAX_STAGED_LORAS:
        _discard(copy)
        return (render.lora_too_many(MAX_STAGED_LORAS), staged, render.applied_loras(staged))
    staged.append(dict(entry))
    return (render.lora_accepted(source, report), staged, render.applied_loras(staged))


def _discard(copy: Any) -> None:
    if copy is not None:
        uploads.discard(copy)


def on_clear_loras() -> tuple[str, list[dict[str, Any]], str]:
    """Drop every staged adapter and say so."""
    return (render.lora_idle(), [], render.applied_loras([]))


def on_remove_lora(staged: list[dict] | None, index: int) -> list[dict]:
    """Remove one staged adapter by position."""
    entries = list(staged or [])
    if 0 <= index < len(entries):
        entries.pop(index)
    return entries


def on_lora_weight(staged: list[dict] | None, index: int, weight: Any) -> list[dict]:
    """Set one staged adapter's strength, through the LoRA module's own check."""
    from .. import lora

    entries = [dict(e) for e in (staged or [])]
    if 0 <= index < len(entries):
        entries[index]["weight"] = lora.check_weight(weight, source=entries[index]["source"])
    return entries


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def run_platform() -> tuple[str | None, bool]:
    """``(device, on ZeroGPU)`` for a run started in this process, as :func:`run_facts`
    labels a finished one."""
    return _run_device(None)


def on_generate_start(
    mode_id: str,
    enhance_prompt: Any = False,
    prompt: Any = None,
    *,
    device: str | None = None,
    on_zerogpu: bool = True,
) -> str:
    """Shown the instant Generate is pressed, so the page is never silent.

    Says the prompt is being enhanced only when it will be: the mode takes a
    prompt and, when the prompt is passed, it is not empty. ``device`` and
    ``on_zerogpu`` say where the run will run (the page passes
    :func:`run_platform`); without them the words are the Space's, a
    reservation from the user's quota.
    """
    try:
        mode = modes.get_mode(mode_id)
    except Exception as exc:  # noqa: BLE001
        return render.failure(exc)
    enhancing = bool(enhance_prompt) and modes.accepts(mode, "text")
    if prompt is not None:
        enhancing = enhancing and bool(str(prompt).strip())
    return render.running(mode, enhancing=enhancing, on_zerogpu=on_zerogpu, device=device)


@dataclass
class RunOutcome:
    """One run, as the UI needs it after ``/on_generate`` has returned."""

    video: Any
    audio: Any
    status: str
    mode_id: str
    params: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: BaseException | None = None
    refused: bool = False
    took: float = 0.0
    #: The request as ``generate.prepare_request`` scoped it (lengths matched
    #: to an audio input are filled in here).
    prepared: Any = None

    def as_tuple(self) -> tuple[Any, Any, str]:
        return (self.video, self.audio, self.status)


def has_result(outcome: RunOutcome | None) -> bool:
    """True for a run that finished with a file to show."""
    return (
        outcome is not None
        and outcome.result is not None
        and outcome.error is None
        and (outcome.video is not None or outcome.audio is not None)
    )


def ran_nothing(outcome: RunOutcome | None) -> bool:
    """True for a run refused before anything ran: by this page before dispatch, or by the
    backend or the platform with no GPU time used (``errors.categorise``)."""
    if outcome is None or outcome.error is None:
        return False
    from .. import errors

    return bool(outcome.refused) or not errors.categorise(outcome.error).gpu_used


def unshown_report(outcome: RunOutcome | None, server_message: str | None) -> str:
    """The report for a run whose step Gradio failed: the handler raised, or Gradio could
    not process or deliver what it returned.

    A run with a result: it finished, what it used, and the server's message. A run whose
    own report is a failure or a refusal: that report, which already says whether GPU time
    was used. No outcome at all: it ended without a result.
    """
    if outcome is None:
        return render.lost_run(server_message, on_zerogpu=run_platform()[1])
    if not has_result(outcome):
        return outcome.status
    return render.result_not_shown(run_facts(outcome), server_message, kept=file_kept(outcome))


def file_kept(outcome: RunOutcome | None) -> bool:
    """True when the run's result file is still on disk (Recent runs can show it)."""
    from pathlib import Path

    path = outcome and (outcome.video or outcome.audio)
    return bool(path) and Path(str(path)).exists()


def run(mode_id: str, *values: Any, progress: Any = None) -> RunOutcome:
    """Run one generation and marshal the result.

    Every path returns a concrete value for each output and a non-empty
    status. A request the backend would refuse is refused here, before
    ``generate()`` is called, with the backend's own words -- so "nothing ran"
    is known, not guessed.
    """
    params = dict(zip(FIELD_KEYS, values, strict=True))
    started = time.monotonic()
    on_zerogpu = run_platform()[1]

    def done(**kwargs) -> RunOutcome:
        return RunOutcome(mode_id=mode_id, params=params, took=time.monotonic() - started, **kwargs)

    try:
        mode = modes.get_mode(mode_id)
    except Exception as exc:  # noqa: BLE001
        return done(
            video=None,
            audio=None,
            status=render.failure(exc, refused=True, on_zerogpu=on_zerogpu),
            error=exc,
            refused=True,
        )

    from .. import generate as generate_module

    try:
        prepared, _spec, _notes = generate_module.prepare_request(_request(mode_id, params))
    except Exception as exc:  # noqa: BLE001 -- refused before any dispatch
        return done(
            video=None,
            audio=None,
            status=render.failure(exc, refused=True, on_zerogpu=on_zerogpu),
            error=exc,
            refused=True,
        )

    try:
        result = generate_module.generate(mode=mode_id, progress=progress, **params)
    except Exception as exc:  # noqa: BLE001 -- typed or not, the user sees it
        return done(
            video=None,
            audio=None,
            status=render.failure(exc, on_zerogpu=on_zerogpu),
            error=exc,
        )

    video_path = getattr(result, "video_path", None)
    audio_path = getattr(result, "audio_path", None)
    missing = [
        track
        for track, path in (("video", video_path), ("audio", audio_path))
        if track in mode.exports and not path
    ]
    if missing:
        return done(
            video=None,
            audio=None,
            status=render.empty_output_bug(mode, missing, on_zerogpu=on_zerogpu),
            result=result,
        )

    return done(
        video=video_path if "video" in mode.exports else None,
        audio=audio_path if "audio" in mode.exports else None,
        status=render.success(result, mode),
        result=result,
        prepared=prepared,
    )


def fact(result: Any, name: str, fallback: Any = None) -> Any:
    """One named fact about a finished run, from the result's structured record.

    The backend is adding ``result.record``; until a field is there, the caller's
    ``fallback`` -- worked out from the request and the wall clock -- is used.
    Every run fact the report shows is read through here, so the switch-over is
    this one function.
    """
    value = getattr(getattr(result, "record", None), name, None)
    return fallback if value is None else value


def run_facts(outcome: RunOutcome) -> render.RunFacts:
    """What the report's "what you got" block states, for a finished run.

    Each fact is read from ``result.record`` (``generate.RunRecord``) and falls
    back to the request and the page's wall clock when a result has none.
    """
    from .. import generate as generate_module
    from .. import pipelines

    mode = modes.get_mode(outcome.mode_id)
    result, params = outcome.result, outcome.params
    request = outcome.prepared
    asked_stage_2 = bool(params.get("enable_stage_2")) and not mode.skip_stage_2
    if mode.min_resolution:
        size = (modes.MIN_RESOLUTION, modes.MIN_RESOLUTION)
    else:
        size = (_as_int(params.get("width")) or 0, _as_int(params.get("height")) or 0)
    scale = pipelines.STAGE_2_SPATIAL_SCALE if asked_stage_2 else 1
    auto = bool(getattr(request, "auto_duration", params.get("auto_duration")))
    frames = None if auto else getattr(request, "num_frames", params.get("num_frames"))
    rate = float(getattr(request, "frame_rate", FRAME_RATE) or FRAME_RATE)

    width = fact(result, "width", size[0] * scale)
    height = fact(result, "height", size[1] * scale)
    adapters = fact(result, "adapters", None)
    if adapters:
        loras = tuple(
            (render.shown_name(str(label)), float(weight), int(attached))
            for label, weight, attached in adapters
        )
    else:
        loras = _staged_names(params.get("loras"))
    asked_auto = bool(params.get("auto_duration"))
    if asked_auto and modes.requires(mode, "audio"):
        origin = generate_module.LENGTH_FROM_AUDIO
    elif asked_auto:
        origin = generate_module.LENGTH_FROM_MODEL
    else:
        origin = generate_module.LENGTH_FROM_REQUEST
    device, on_zerogpu = _run_device(result)
    view = mode_view(mode.id)
    varies_audio = view.varies_sound
    level = fact(result, "audio_level", _as_float(params.get("noise_scale")))
    clip_level = fact(result, "video_level", None) if view.varies_clip else None
    sent = fact(result, "prompt", None)
    typed = str(params.get("prompt") or "").strip()
    elapsed = fact(result, "elapsed_seconds", None)
    length_source = fact(result, "length_source", origin)
    bound = None
    if length_source == generate_module.LENGTH_FROM_MODEL:
        bound = _length_bound(
            fact(result, "frames", None),
            getattr(request, "min_seconds", params.get("min_seconds")),
            getattr(request, "max_seconds", params.get("max_seconds")),
            rate,
        )
    return render.RunFacts(
        mode=mode,
        seed=int(fact(result, "seed", getattr(result, "seed", 0))),
        returned_size=(int(width), int(height)) if width and height else None,
        seconds=fact(result, "seconds", (int(frames) / rate) if frames else None),
        stage_2=bool(fact(result, "stage_2", asked_stage_2)),
        loras=loras,
        took=float(elapsed if elapsed is not None else outcome.took),
        took_is_gpu=elapsed is not None,
        prompt_sent=sent if sent and str(sent).strip() != typed else None,
        length_source=length_source,
        length_bound=bound,
        audio_level=level if varies_audio else None,
        video_level=clip_level,
        decoder=fact(result, "decoder", None),
        device=device,
        on_zerogpu=on_zerogpu,
        quota_multiplier=quota_multiplier(),
    )


def _length_bound(frames: Any, shortest: Any, longest: Any, rate: float) -> str | None:
    """Where a model-picked length of ``frames`` sits in the user's range: "shortest" or
    "longest" when it is the frame count the duration head gives at that end (where the range
    may have raised or lowered the model's own figure), "only" when the range allows one
    length, None strictly inside (the model's figure, for certain) or when unknown. The ends
    are the pipeline's own (``constraints.duration_head_frame_range``)."""
    count, low, high = _as_int(frames), _as_float(shortest), _as_float(longest)
    if count is None or low is None or high is None:
        return None
    first, last = constraints.duration_head_frame_range(
        min_seconds=low, max_seconds=high, frame_rate=float(rate)
    )
    if count == first == last:
        return "only"
    if count == first:
        return "shortest"
    if count == last:
        return "longest"
    return None


def _run_device(result: Any) -> tuple[str | None, bool]:
    """``(device, on ZeroGPU)`` for a finished run.

    The record's own ``device`` when the backend reports one; otherwise this
    process's resolution, which is the one the GPU entrypoint runs with.
    """
    recorded = fact(result, "device", None)
    try:
        from .. import device as device_module

        resolution = device_module.current()
    except Exception:  # noqa: BLE001 -- an unresolvable device leaves the time unlabelled
        return (str(recorded) if recorded else None), False
    return (str(recorded) if recorded else resolution.device), bool(resolution.on_zerogpu)


def on_generate(mode_id: str, *values: Any, progress: Any = None) -> tuple[Any, Any, str]:
    """``/on_generate``'s contract: ``(video, audio, status)``, never a no-op.

    ``progress`` is keyword-only: ``mode_id`` plus the ``FIELD_KEYS`` block is
    the endpoint's positional signature, and a progress handle in a positional
    slot would shift every field one place.
    """
    return run(mode_id, *values, progress=progress).as_tuple()
