"""The one declarative input spec. Everything else in ``ui/`` derives from it.

A single ordered tuple of :class:`Field` builds four things, so they cannot
drift apart:

1. the component pool          (``components.render_fields``)
2. the ``inputs=`` list        (``[pool[k] for k in FIELD_KEYS]``, in ``blocks.py``)
3. the handler's unpacking     (``dict(zip(FIELD_KEYS, values, strict=True))``)
4. per-mode activity           (``handlers.mode_view`` walks ``TOGGLEABLE_FIELDS``)

``FIELD_KEYS`` is the positional signature of ``/on_generate``: append only,
and never change a field's component kind.

**What a mode uses comes from the registry, once.** A field is active in a
mode exactly when ``modes.accepts(spec, kind)`` says the mode uses that kind of
input -- the same predicate ``generate.scope_request_to_mode`` scopes requests
with -- so the UI and the backend cannot disagree about what a mode takes.

Imports no Gradio: data plus predicates. Labels and info lines live in
``render.FIELD_COPY`` with every other user-facing string.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import modes
from .render import DECODER_OPTIONS, FIELD_COPY

__all__ = [
    "FIELDS",
    "FIELD_KEYS",
    "MEDIA_KINDS",
    "PARKED_FIELDS",
    "SHOT_ANCHOR_COLUMN",
    "SHOT_COLUMNS",
    "SHOT_HEADERS",
    "TOGGLEABLE_FIELDS",
    "Field",
    "field_by_key",
]


@dataclass(frozen=True)
class Field:
    """One user input, described once.

    ``key`` is the name of the matching ``GenerationRequest`` field: the handler
    forwards the marshalled values straight through as keyword arguments.
    """

    key: str
    kind: str
    label: str
    info: str = ""
    group: str = "sampling"
    default: Any = None
    options: dict[str, Any] = field(default_factory=dict)
    #: True when the field contributes to a run of this mode. ``None`` means it
    #: always does (or, for ``gr.State``, that it has no visibility at all).
    visible_when: Callable[[Any], bool] | None = None
    #: The value is the user's, and is moved aside rather than submitted while
    #: the selected mode does not use it (``handlers.mode_switch``).
    parks: bool = False


# ---------------------------------------------------------------------------
# activity predicates -- each reads only the registry entry
# ---------------------------------------------------------------------------


def _accepts(kind: str) -> Callable[[Any], bool]:
    """The registry's own acceptance predicate, for one input kind."""

    def predicate(mode: Any) -> bool:
        return modes.accepts(mode, kind)

    predicate.__name__ = f"accepts_{kind}"
    return predicate


def _never(mode: Any) -> bool:
    """A field kept for the API whose control this UI does not offer."""
    return False


def _has_timed_input(mode: Any) -> bool:
    """A clip or an audio file sets the length ("Match my clip")."""
    return modes.accepts(mode, "video") or modes.requires(mode, "audio")


def _model_may_pick_length(mode: Any) -> bool:
    """ "Let the model pick the length" only where no input has a length."""
    return not _has_timed_input(mode)


def _varies_clip(mode: Any) -> bool:
    """The user's clip is the starting point of a variation, not a condition."""
    return getattr(mode, "video_conditioning", None) == "variation"


def _video_lever(mode: Any) -> bool:
    """The clip-strength lever: a video CONDITION whose strength the mode does not fix.
    A variation does not read strength (its slot stays, for the API)."""
    return modes.accepts(mode, "video") and mode.video_strength is None and not _varies_clip(mode)


def _varies_input(mode: Any) -> bool:
    """The change-level choice: a sound or a clip the mode varies rather than holds fixed."""
    return mode.audio_conditioning == "variation" or _varies_clip(mode)


def _sizes_the_picture(mode: Any) -> bool:
    """False where the registry pins the size because the picture is thrown away."""
    return not mode.min_resolution


def _returns_a_picture(mode: Any) -> bool:
    """The decoder choice: only where a picture comes back."""
    return "video" in mode.exports


def _offers_upscale(mode: Any) -> bool:
    return not mode.skip_stage_2


# ---------------------------------------------------------------------------
# the spec -- order is the endpoint's positional signature
# ---------------------------------------------------------------------------

#: Field kinds that carry a user-supplied asset rather than a number or string.
MEDIA_KINDS = frozenset({"image", "video", "audio"})

#: The shot-list columns. The UI no longer offers a shot list (rows that carry
#: text are refused before dispatch), but ``/on_generate`` keeps the slot and
#: ``conditioning.py`` reads rows in this column order.
SHOT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Prompt segment", "str"),
    ("Anchor (frame)", "number"),
    ("Strength", "number"),
)
SHOT_HEADERS: tuple[str, ...] = tuple(name for name, _ in SHOT_COLUMNS)
SHOT_DATATYPES: tuple[str, ...] = tuple(kind for _, kind in SHOT_COLUMNS)
SHOT_ANCHOR_COLUMN: int = SHOT_HEADERS.index("Anchor (frame)")


def _copy(key: str) -> dict[str, str]:
    """A field's label and info line; any figure in it comes from the estimator."""
    label, info = FIELD_COPY[key]
    if "{" in info:
        from .. import zerogpu

        info = info.format(
            enhance_seconds=zerogpu.ENHANCE_DURATION_SECONDS,
            upscale_seconds=zerogpu.estimate_duration(enable_stage_2=True)
            - zerogpu.estimate_duration(),
        )
    return {"label": label, "info": info}


FIELDS: tuple[Field, ...] = (
    Field(
        key="prompt",
        kind="text",
        group="prompt",
        default="",
        options={"lines": 3, "max_lines": 8},
        visible_when=_accepts("text"),
        parks=True,
        **_copy("prompt"),
    ),
    Field(
        key="negative_prompt",
        kind="text",
        group="prompt",
        default="",
        options={"lines": 1},
        visible_when=_never,
        parks=True,
        **_copy("negative_prompt"),
    ),
    Field(
        key="image",
        kind="image",
        group="media",
        visible_when=_accepts("image"),
        parks=True,
        **_copy("image"),
    ),
    Field(
        key="video",
        kind="video",
        group="media",
        visible_when=_accepts("video"),
        parks=True,
        **_copy("video"),
    ),
    Field(
        key="audio",
        kind="audio",
        group="media",
        visible_when=_accepts("audio"),
        parks=True,
        **_copy("audio"),
    ),
    Field(
        key="conditions",
        kind="dataframe",
        group="media",
        default=[],
        options={
            "headers": list(SHOT_HEADERS),
            "datatype": list(SHOT_DATATYPES),
            "column_count": len(SHOT_COLUMNS),
            "row_count": 0,
            "type": "array",
        },
        visible_when=_never,
        parks=True,
        **_copy("conditions"),
    ),
    Field(
        key="width",
        kind="number",
        group="shape",
        default=704,
        options={"step": 32, "precision": 0},
        visible_when=_sizes_the_picture,
        parks=True,
        **_copy("width"),
    ),
    Field(
        key="height",
        kind="number",
        group="shape",
        default=480,
        options={"step": 32, "precision": 0},
        visible_when=_sizes_the_picture,
        parks=True,
        **_copy("height"),
    ),
    Field(
        key="auto_duration",
        kind="checkbox",
        group="shape",
        default=False,
        visible_when=_model_may_pick_length,
        parks=True,
        **_copy("auto_duration"),
    ),
    Field(
        key="num_frames",
        kind="number",
        group="shape",
        default=121,
        options={"step": 8, "precision": 0},
        **_copy("num_frames"),
    ),
    Field(
        key="min_seconds",
        kind="number",
        group="shape",
        default=2.0,
        options={"step": 0.1},
        visible_when=_model_may_pick_length,
        parks=True,
        **_copy("min_seconds"),
    ),
    Field(
        key="max_seconds",
        kind="number",
        group="shape",
        default=5.0,
        options={"step": 0.1},
        visible_when=_model_may_pick_length,
        parks=True,
        **_copy("max_seconds"),
    ),
    Field(
        key="enable_stage_2",
        kind="checkbox",
        group="shape",
        default=False,
        visible_when=_offers_upscale,
        **_copy("enable_stage_2"),
    ),
    Field(
        key="seed",
        kind="number",
        group="sampling",
        default=-1,
        options={"minimum": -1, "step": 1, "precision": 0},
        **_copy("seed"),
    ),
    Field(
        key="strength",
        kind="slider",
        group="media",
        default=0.8,
        options={"minimum": 0.0, "maximum": 0.99, "step": 0.01},
        visible_when=_video_lever,
        parks=True,
        **_copy("strength"),
    ),
    Field(
        key="noise_scale",
        kind="slider",
        group="media",
        default=0.5,
        options={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
        visible_when=_varies_input,
        parks=True,
        **_copy("noise_scale"),
    ),
    # gr.State: no label, no visibility.
    Field(key="loras", kind="state", group="lora", default=[], **_copy("loras")),
    # Appended, never inserted: FIELD_KEYS is the endpoint's positional order.
    Field(
        key="enhance_prompt",
        kind="checkbox",
        group="prompt",
        default=False,
        visible_when=_accepts("text"),
        parks=True,
        **_copy("enhance_prompt"),
    ),
    # Appended: the decoder that turns the result into frames. Sent as "auto"
    # where no picture comes back.
    Field(
        key="decoder",
        kind="radio",
        group="sampling",
        default="auto",
        options={"choices": [(label, value) for label, value, _w in DECODER_OPTIONS]},
        visible_when=_returns_a_picture,
        parks=True,
        **_copy("decoder"),
    ),
)

#: The ordering contract. Built once, read everywhere.
FIELD_KEYS: tuple[str, ...] = tuple(f.key for f in FIELDS)

#: Fields whose activity depends on the mode, in spec order.
TOGGLEABLE_FIELDS: tuple[Field, ...] = tuple(f for f in FIELDS if f.visible_when is not None)

#: Fields whose value is the user's and is parked while the mode does not use it.
PARKED_FIELDS: tuple[Field, ...] = tuple(f for f in FIELDS if f.parks)


def field_by_key(key: str) -> Field:
    """Look up one field, raising rather than returning ``None`` on a typo."""
    for spec in FIELDS:
        if spec.key == key:
            return spec
    raise KeyError(f"no UI field named {key!r}. Declared fields: {', '.join(FIELD_KEYS)}")


def _assert_unique(keys: Sequence[str]) -> None:
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate UI field keys: {', '.join(duplicates)}")


_assert_unique(FIELD_KEYS)
