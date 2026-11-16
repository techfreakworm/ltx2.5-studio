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
from . import render
from .fields import FIELD_KEYS, FIELDS, PARKED_FIELDS, SHOT_ANCHOR_COLUMN, TOGGLEABLE_FIELDS

__all__ = [
    "MAX_STAGED_LORAS",
    "FieldUpdate",
    "ModeView",
    "RunOutcome",
    "clip_seconds",
    "export_flags",
    "frames_for_seconds",
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
    "preset_for",
    "quota_multiplier",
    "review",
    "run",
    "seconds_for_frames",
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
        name=render.mode_name(mode),
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
    )


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
    """A clip's length in seconds, measured with ``av``. ``None`` when unreadable."""
    if not path:
        return None
    if kind == "audio":
        from .. import generate as generate_module

        try:
            return generate_module._audio_seconds(path)
        except Exception:  # noqa: BLE001 -- the review names the problem
            return None
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return container.duration / av.time_base
            frames = sum(1 for _ in container.decode(stream))
            rate = float(stream.average_rate or FRAME_RATE)
            return frames / rate
    except Exception:  # noqa: BLE001
        return None


def frames_for_seconds(seconds: Any) -> int:
    """The legal frame count nearest to ``seconds`` at the generation frame rate."""
    value = _as_float(seconds) or 0.0
    frames = round(value * FRAME_RATE)
    frames = min(max(frames, constraints.MIN_NUM_FRAMES), constraints.MAX_NUM_FRAMES)
    return constraints.snap_num_frames(frames)


def seconds_for_frames(frames: Any) -> float:
    count = _as_int(frames) or 0
    return round(count / FRAME_RATE, 2)


def frames_matching_clip(seconds: float | None) -> int | None:
    if seconds is None:
        return None
    return constraints.largest_num_frames_within(seconds, frame_rate=FRAME_RATE)


# ---------------------------------------------------------------------------
# live constraint guidance, in Advanced
# ---------------------------------------------------------------------------


def on_shape_change(width: Any, height: Any, num_frames: Any) -> tuple[str, str]:
    """Hints for the exact size and frame count, naming the legal values."""
    return (_dimension_hint(width, height), _frame_hint(num_frames))


def _dimension_hint(width: Any, height: Any) -> str:
    w, h = _as_int(width), _as_int(height)
    if w is None or h is None:
        return "Width and height must both be numbers."
    return render.dimension_hint(
        w, h, multiple=constraints.DIMENSION_MULTIPLE, snapped=constraints.snap_dimensions(w, h)
    )


def _frame_hint(num_frames: Any) -> str:
    frames = _as_int(num_frames)
    if frames is None:
        return "The frame count must be a number."
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
    w = constraints.snap_dimension(_as_int(width) or constraints.DIMENSION_MULTIPLE)
    h = constraints.snap_dimension(_as_int(height) or constraints.DIMENSION_MULTIPLE)
    frames = constraints.snap_num_frames(_as_int(num_frames) or 1)
    return (w, h, frames, _dimension_hint(w, h), _frame_hint(frames))


def on_auto_duration_toggle(auto: Any) -> tuple[bool, bool, bool]:
    """Visibility for (num_frames, min_seconds, max_seconds)."""
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
    """How many seconds of quota one second of GPU time costs.

    ``zerogpu`` does not export this yet; until it does, the configured size
    decides it the way the platform does (xlarge counts double).
    """
    from .. import zerogpu

    exported = getattr(zerogpu, "QUOTA_MULTIPLIER", None)
    if exported is not None:
        return int(exported)
    try:
        from ..config import load_config

        size = load_config().zerogpu_size
    except Exception:  # noqa: BLE001 -- the documented size for this model
        size = "xlarge"
    return 2 if size == "xlarge" else 1


def _shape_problems(width: Any, height: Any, num_frames: Any, auto: bool) -> list[str]:
    """Every size and length problem, each with its nearest legal values."""
    problems = []
    w, h = _as_int(width), _as_int(height)
    if w is None or h is None:
        problems.append("Width and height must both be numbers.")
    else:
        for name, value in (("Width", w), ("Height", h)):
            try:
                constraints.validate_dimension(value, name=name.lower())
                constraints.validate_dimension_limit(value, name=name.lower())
            except constraints.ConstraintError:
                snapped = min(constraints.snap_dimension(value), constraints.MAX_DIMENSION)
                problems.append(
                    f"{name} {value} won't run: use {snapped} (multiples of "
                    f"{constraints.DIMENSION_MULTIPLE}, up to {constraints.MAX_DIMENSION})."
                )
    if not auto:
        frames = _as_int(num_frames)
        if frames is None:
            problems.append("The frame count must be a number.")
        elif not constraints.is_valid_num_frames(frames):
            lower, upper = constraints.nearest_valid_num_frames(frames)
            problems.append(
                f"{frames} frames won't run: use {lower} ({lower / FRAME_RATE:.1f} s) or "
                f"{upper} ({upper / FRAME_RATE:.1f} s)."
            )
        elif frames > constraints.MAX_NUM_FRAMES:
            problems.append(
                f"{frames} frames is longer than this model makes: use at most "
                f"{constraints.MAX_NUM_FRAMES} ({constraints.MAX_SECONDS:g} s)."
            )
    return problems


def _request(mode_id: str, values: dict[str, Any]):
    from .. import generate as generate_module

    return generate_module.GenerationRequest(mode=mode_id, **values)


def _staged_names(loras: Any) -> tuple[tuple[str, float], ...]:
    out = []
    for entry in loras or ():
        if isinstance(entry, dict):
            source = str(entry.get("source", ""))
            out.append((render.shown_name(source), float(entry.get("weight", 1.0))))
    return tuple(out)


def review(mode_id: str, values: dict[str, Any]) -> render.CostFacts:
    """What this request would render, return and reserve -- or why it won't run.

    Readiness is the backend's own answer: ``generate.prepare_request`` is the
    check ``generate()`` runs before it reserves a GPU, so the reason shown under
    a disabled Generate is the refusal the server would give, word for word.
    """
    from .. import generate as generate_module
    from .. import pipelines, zerogpu
    from ..errors import MissingInputError

    mode = modes.get_mode(mode_id)
    name = render.mode_name(mode)
    multiplier = quota_multiplier()
    try:
        request = _request(mode_id, values)
    except Exception as exc:  # noqa: BLE001
        return render.CostFacts(mode_name=name, state="refused", reason=str(exc))

    problems = []
    if mode.min_resolution:
        # The pinned modes render at the registry's size whatever the boxes say;
        # the backend still validates the boxes, so a bad value is still named.
        pass
    problems = _shape_problems(
        request.width, request.height, request.num_frames, bool(request.auto_duration)
    )

    state, reason, prepared, notes = "ready", None, request, ()
    try:
        prepared, _spec, notes = generate_module.prepare_request(request)
    except MissingInputError as exc:
        state, reason = "incomplete", str(exc)
        try:
            prepared, _ = generate_module.scope_request_to_mode(request, mode)
        except Exception:  # noqa: BLE001
            prepared = request
    except Exception as exc:  # noqa: BLE001 -- every refusal is shown, verbatim
        state, reason = "refused", str(exc)
    if problems:
        state = "refused"

    if state == "refused":
        return render.CostFacts(
            mode_name=name,
            state="refused",
            reason=reason,
            problems=tuple(problems) or ((reason,) if reason else ()),
            stage_2_offered=not mode.skip_stage_2,
            enhance_offered=modes.accepts(mode, "text"),
            multiplier=multiplier,
            loras=_staged_names(request.loras),
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
    seconds = None if frames is None else frames / float(prepared.frame_rate)

    enhance = bool(prepared.enhance_prompt) and bool((prepared.prompt or "").strip())
    loras = _staged_names(prepared.loras)
    # The reservation, and each extra's share of it, from the estimator alone.
    steps = [
        ("Upscale ×2", {"enable_stage_2": stage_2}),
        ("LoRA", {"loras": tuple(prepared.loras or ())}),
        ("Enhance", {"enhance_prompt": enhance}),
    ]
    base = zerogpu.estimate_duration()
    so_far: dict[str, Any] = {}
    previous = base
    extras = []
    for label, lever in steps:
        so_far.update(lever)
        total = zerogpu.estimate_duration(**so_far)
        if total != previous:
            extras.append((label, total - previous))
        previous = total
    reserve = zerogpu.estimate_duration(
        enable_stage_2=stage_2, loras=tuple(prepared.loras or ()), enhance_prompt=enhance
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

    try:
        from .. import lora

        if upload:
            from . import uploads

            local = uploads.own_copy(source)
            report = lora.validate_lora(str(local), upload=True)
        else:
            report = lora.validate_lora(source, fetcher=lora.fetch_lora, upload=False)
        entry = lora.staged_entry(source, report)
    except Exception as exc:  # noqa: BLE001 -- every failure must reach the user
        return (render.lora_rejected(source, exc), staged, render.applied_loras(staged))

    existing = lora.find_duplicate(staged, entry)
    if existing is not None:
        return (render.lora_duplicate(source, existing), staged, render.applied_loras(staged))
    if len(staged) >= MAX_STAGED_LORAS:
        return (render.lora_too_many(MAX_STAGED_LORAS), staged, render.applied_loras(staged))
    staged.append(dict(entry))
    return (render.lora_accepted(source, report), staged, render.applied_loras(staged))


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


def on_generate_start(mode_id: str, enhance_prompt: Any = False) -> str:
    """Shown the instant Generate is pressed, so the page is never silent."""
    try:
        return render.running(modes.get_mode(mode_id), enhancing=bool(enhance_prompt))
    except Exception as exc:  # noqa: BLE001
        return render.failure(exc)


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

    def as_tuple(self) -> tuple[Any, Any, str]:
        return (self.video, self.audio, self.status)


def run(mode_id: str, *values: Any, progress: Any = None) -> RunOutcome:
    """Run one generation and marshal the result.

    Every path returns a concrete value for each output and a non-empty
    status. A request the backend would refuse is refused here, before
    ``generate()`` is called, with the backend's own words -- so "nothing ran"
    is known, not guessed.
    """
    params = dict(zip(FIELD_KEYS, values, strict=True))
    started = time.monotonic()

    def done(**kwargs) -> RunOutcome:
        return RunOutcome(mode_id=mode_id, params=params, took=time.monotonic() - started, **kwargs)

    try:
        mode = modes.get_mode(mode_id)
    except Exception as exc:  # noqa: BLE001
        return done(
            video=None,
            audio=None,
            status=render.failure(exc, refused=True),
            error=exc,
            refused=True,
        )

    from .. import generate as generate_module

    try:
        generate_module.prepare_request(_request(mode_id, params))
    except Exception as exc:  # noqa: BLE001 -- refused before any dispatch
        return done(
            video=None,
            audio=None,
            status=render.failure(exc, refused=True),
            error=exc,
            refused=True,
        )

    try:
        result = generate_module.generate(mode=mode_id, progress=progress, **params)
    except Exception as exc:  # noqa: BLE001 -- typed or not, the user sees it
        return done(video=None, audio=None, status=render.failure(exc), error=exc)

    video_path = getattr(result, "video_path", None)
    audio_path = getattr(result, "audio_path", None)
    missing = [
        track
        for track, path in (("video", video_path), ("audio", audio_path))
        if track in mode.exports and not path
    ]
    if missing:
        return done(
            video=None, audio=None, status=render.empty_output_bug(mode, missing), result=result
        )

    return done(
        video=video_path if "video" in mode.exports else None,
        audio=audio_path if "audio" in mode.exports else None,
        status=render.success(result, mode),
        result=result,
    )


def on_generate(mode_id: str, *values: Any, progress: Any = None) -> tuple[Any, Any, str]:
    """``/on_generate``'s contract: ``(video, audio, status)``, never a no-op.

    ``progress`` is keyword-only: ``mode_id`` plus the ``FIELD_KEYS`` block is
    the endpoint's positional signature, and a progress handle in a positional
    slot would shift every field one place.
    """
    return run(mode_id, *values, progress=progress).as_tuple()
