"""The **one** declarative input spec. Everything else in ``ui/`` derives from it.

PLAN §3.2 asks for 11 modes over 4 workflows without 11 sets of duplicated
widgets, and PLAN §7.2 asks for a registry that is data rather than branching.
This module is how the UI honours both: a single ordered tuple of :class:`Field`
descriptions, from which four separate things are generated **by construction**
rather than maintained in parallel:

1. the component pool          (``components.build_pool`` iterates ``FIELDS``)
2. the ``inputs=`` list        (``[pool[k] for k in FIELD_KEYS]``)
3. the handler's unpacking     (``dict(zip(FIELD_KEYS, values, strict=True))``)
4. per-mode visibility         (``handlers.visibility_flags`` iterates ``TOGGLEABLE_FIELDS``)

**Why this shape, specifically.** The predecessor kept three hand-maintained
``if``/``elif`` chains -- one listing a mode's input keys, one collecting its
values, one rendering its widgets -- and joined them with
``zip(keys, values, strict=False)``. When the chains drifted, ``strict=False``
did not complain; it silently dropped the tail and handed the wrong value to the
wrong parameter. Generation still "worked". The output was just wrong.

Here the ordering used to *build* the inputs list and the ordering used to
*unpack* it are the same tuple, read twice, so they cannot drift -- and every
``zip`` in this package passes ``strict=True`` so that if they somehow did, it
raises instead of mis-assigning.

Adding an input is one entry in ``FIELDS``. There is no second place.

This module imports no Gradio and no torch: it is data plus predicates, so it
can be read and tested on a bare interpreter. The Gradio mapping lives in
``components.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FIELDS",
    "FIELD_KEYS",
    "MEDIA_KINDS",
    "TOGGLEABLE_FIELDS",
    "Field",
    "field_by_key",
]


@dataclass(frozen=True)
class Field:
    """One user input, described once.

    ``key`` is deliberately the name of the corresponding ``GenerationRequest``
    field (see ``tests/conftest.py::request_fields``), because the handler
    forwards the marshalled dict straight through as keyword arguments. A field
    whose key does not exist on the request is a loud ``TypeError`` at generate
    time, not a silently ignored control.
    """

    key: str
    kind: str
    label: str
    info: str = ""
    group: str = "sampling"
    default: Any = None
    options: dict[str, Any] = field(default_factory=dict)
    #: ``None`` means "never toggled by the mode selector" -- either because the
    #: field applies everywhere, or because it has no visibility at all (a
    #: ``gr.State``, which cannot take a ``visible=`` update and would raise if
    #: one were sent to it).
    visible_when: Callable[[Any], bool] | None = None


# ---------------------------------------------------------------------------
# visibility predicates
#
# Each reads only the mode registry entry (PLAN §3.3 / tests/test_modes.py), so
# adding a mode to `modes.py` gives it a correct UI with no edit here.
# ---------------------------------------------------------------------------


def _accepts(*names: str) -> Callable[[Any], bool]:
    """True when the mode declares any of ``names`` among its inputs."""

    wanted = frozenset(names)

    def predicate(mode: Any) -> bool:
        return bool(wanted & set(mode.inputs))

    return predicate


def _takes_a_prompt(mode: Any) -> bool:
    """Modes that do not declare a text input send an empty prompt.

    PLAN §3.3 row 2 is explicit about this: image-to-video is "one condition,
    **empty prompt**". Showing a prompt box whose contents the builder discards
    would be a lie in the UI, so the box is hidden rather than ignored.
    """
    return "text" in mode.inputs


def _is_conditionable(mode: Any) -> bool:
    """True for modes that have visual media to hang a shot on.

    PLAN §3.1(a): image- and video-conditioning are the *same* mechanism -- one
    ``LTX2VideoCondition`` list carrying frames, an index and a strength.

    Note this keys on the **declared inputs**, not on ``mode.pipeline``, and the
    difference is real: the audio modes (§3.3 rows 10 and 11) also run through
    the ``condition`` workflow, but they enter via ``audio_latents`` and carry no
    frames at all. Keying on the workflow would offer them a shot list with
    nothing to put in it.
    """
    return bool({"image", "video"} & set(mode.inputs))


def _controls_resolution(mode: Any) -> bool:
    """False where PLAN §3.2's mandatory mitigation fixes the resolution.

    Modes whose video is discarded run at the **minimum legal resolution** --
    that is not a preference, it is the mitigation that stops an audio export
    costing a full-resolution render. Offering a resolution control the backend
    overrides would be exactly the kind of dead toggle §3.5 refuses to ship.
    """
    return not mode.min_resolution


def _offers_stage_2(mode: Any) -> bool:
    """False where the registry already forces stage 2 off (PLAN §3.2)."""
    return not mode.skip_stage_2


# ---------------------------------------------------------------------------
# the spec
#
# Order here is the order the widgets read on screen AND the order values are
# marshalled in. One tuple, both jobs -- that is the whole point.
# ---------------------------------------------------------------------------

#: Field kinds that carry a user-supplied asset rather than a number or string.
MEDIA_KINDS = frozenset({"image", "video", "audio"})

FIELDS: tuple[Field, ...] = (
    # -- what the model is told to make -------------------------------------
    Field(
        key="prompt",
        kind="textarea",
        label="Prompt",
        info="What to generate. LTX-2.5 responds to camera, lighting and motion language.",
        group="prompt",
        default="",
        options={"lines": 4, "placeholder": "a lighthouse in heavy fog, slow dolly in, cinematic"},
        visible_when=_takes_a_prompt,
    ),
    Field(
        key="negative_prompt",
        kind="text",
        label="Negative prompt",
        info="Optional. Ignored when the distilled schedule runs at guidance 1.0.",
        group="prompt",
        default="",
        options={"lines": 2, "placeholder": "blurry, warped hands, jpeg artifacts"},
        visible_when=_takes_a_prompt,
    ),
    # -- what the model is given ---------------------------------------------
    Field(
        key="image",
        kind="image",
        label="Conditioning image",
        info="Anchored at frame 0 unless a shot list below says otherwise.",
        group="media",
        visible_when=_accepts("image"),
    ),
    Field(
        key="video",
        kind="video",
        label="Conditioning video",
        info="Same mechanism as the image route — a multi-frame condition (PLAN §3.1a).",
        group="media",
        visible_when=_accepts("video"),
    ),
    Field(
        key="audio",
        kind="audio",
        label="Input audio",
        info="Encoded to audio latents before the call — there is no audio= kwarg (PLAN §3.1c).",
        group="media",
        visible_when=_accepts("audio"),
    ),
    Field(
        key="conditions",
        kind="dataframe",
        label="Shot list — prompt segment · anchor (latent index) · strength",
        info="",
        group="media",
        default=[],
        # Gradio 6 renamed these: `row_count=(0, "dynamic")` and
        # `col_count=(3, "fixed")` are Gradio 5 idioms that now warn. The 6.x
        # spelling is a plain int for the initial count, with the min/max pair
        # moved to `row_limits`/`column_limits` -- which 6.24 accepts but does
        # not yet implement, so passing them only produces a warning. Omitted.
        options={
            "headers": ["Prompt segment", "Anchor (LATENT index)", "Strength"],
            "datatype": ["str", "number", "number"],
            "column_count": 3,
            "row_count": 0,
            "type": "array",
        },
        visible_when=_is_conditionable,
    ),
    # -- shape ----------------------------------------------------------------
    Field(
        key="width",
        kind="number",
        label="Width",
        info="Must be a multiple of 32 — upstream raises otherwise.",
        group="shape",
        default=704,
        options={"minimum": 32, "step": 32, "precision": 0},
        visible_when=_controls_resolution,
    ),
    Field(
        key="height",
        kind="number",
        label="Height",
        info="Must be a multiple of 32 — upstream raises otherwise.",
        group="shape",
        default=480,
        options={"minimum": 32, "step": 32, "precision": 0},
        visible_when=_controls_resolution,
    ),
    Field(
        key="auto_duration",
        kind="checkbox",
        label="Let the model choose the length (duration head)",
        info="Omits num_frames and passes a seconds range instead (PLAN §3.6).",
        group="shape",
        default=False,
    ),
    Field(
        key="num_frames",
        kind="number",
        label="Frames",
        info="Must be k×8 + 1. Upstream SILENTLY FLOORS anything else — this UI is your only warning.",
        group="shape",
        default=121,
        options={"minimum": 1, "step": 8, "precision": 0},
    ),
    Field(
        key="min_seconds",
        kind="number",
        label="Min seconds",
        info="Only used when the duration head is choosing.",
        group="shape",
        default=2.0,
        options={"minimum": 0.1, "step": 0.1},
    ),
    Field(
        key="max_seconds",
        kind="number",
        label="Max seconds",
        info="Only used when the duration head is choosing.",
        group="shape",
        default=5.0,
        options={"minimum": 0.1, "step": 0.1},
    ),
    # -- sampling -------------------------------------------------------------
    Field(
        key="seed",
        kind="number",
        label="Seed",
        info="-1 draws a fresh seed per run. The seed actually used is reported after the run.",
        group="sampling",
        default=-1,
        options={"minimum": -1, "step": 1, "precision": 0},
    ),
    Field(
        key="strength",
        kind="slider",
        label="Conditioning strength",
        info="Below 1.0 or the input is pinned and nothing is generated (PLAN §3.3 row 4).",
        group="sampling",
        default=0.8,
        options={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
        visible_when=_accepts("image", "video"),
    ),
    Field(
        key="noise_scale",
        kind="slider",
        label="Audio noise scale",
        info="img2img-style strength over your audio: 0.0 keeps it, 1.0 discards it entirely.",
        group="sampling",
        default=0.5,
        options={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
        visible_when=_accepts("audio"),
    ),
    Field(
        key="guidance_scale",
        kind="slider",
        label="Guidance scale",
        info="The distilled schedule is trained for 1.0. Raising it fights the schedule.",
        group="sampling",
        default=1.0,
        options={"minimum": 1.0, "maximum": 10.0, "step": 0.1},
    ),
    Field(
        key="num_inference_steps",
        kind="number",
        label="Steps",
        info="8 is the distilled schedule's own step count (PLAN §5.5).",
        group="sampling",
        default=8,
        options={"minimum": 1, "maximum": 50, "step": 1, "precision": 0},
    ),
    Field(
        key="enable_stage_2",
        kind="checkbox",
        label="Stage 2 — upscale and refine",
        info="Higher quality, more GPU-seconds. On a quota-metered Space that is a real cost.",
        group="sampling",
        default=False,
        visible_when=_offers_stage_2,
    ),
    # -- carried state --------------------------------------------------------
    Field(
        key="loras",
        kind="state",
        label="Applied LoRA adapters",
        info="",
        group="lora",
        default=[],
        # No visible_when: gr.State has no `visible` property and sending it one
        # raises. See the Field docstring.
    ),
)

#: The ordering contract. Built once, read everywhere -- see the module docstring.
FIELD_KEYS: tuple[str, ...] = tuple(f.key for f in FIELDS)

#: Only these take a ``visible=`` update when the mode changes, in this order.
TOGGLEABLE_FIELDS: tuple[Field, ...] = tuple(f for f in FIELDS if f.visible_when is not None)


def field_by_key(key: str) -> Field:
    """Look up one field, raising rather than returning ``None`` on a typo."""
    for spec in FIELDS:
        if spec.key == key:
            return spec
    raise KeyError(f"no UI field named {key!r}. Declared fields: {', '.join(FIELD_KEYS)}")


def _assert_unique(keys: Sequence[str]) -> None:
    """A duplicated key would make the pool smaller than the key list and
    quietly desynchronise the marshalling. Catch it at import, not at runtime."""
    seen: set[str] = set()
    duplicates = sorted({k for k in keys if k in seen or seen.add(k)})  # type: ignore[func-returns-value]
    if duplicates:
        raise RuntimeError(f"duplicate UI field keys: {', '.join(duplicates)}")


_assert_unique(FIELD_KEYS)
