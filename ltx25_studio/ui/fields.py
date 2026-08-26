"""The **one** declarative input spec. Everything else in ``ui/`` derives from it.

PLAN §3.2 asks for 11 modes over 4 workflows without 11 sets of duplicated
widgets, and PLAN §7.2 asks for a registry that is data rather than branching.
This module is how the UI honours both: a single ordered tuple of :class:`Field`
descriptions, from which four separate things are generated **by construction**
rather than maintained in parallel:

1. the component pool          (``components.render_fields`` builds one per ``Field``)
2. the ``inputs=`` list        (``[pool[k] for k in FIELD_KEYS]``, in ``blocks.py``)
3. the handler's unpacking     (``dict(zip(FIELD_KEYS, values, strict=True))``)
4. per-mode visibility         (``handlers.on_mode_change`` walks ``TOGGLEABLE_FIELDS``)

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
    "SHOT_ANCHOR_COLUMN",
    "SHOT_COLUMNS",
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
    #: Greyed-out rather than hidden. For a value the backend *overrides*: hiding
    #: it reads as a broken app and leaves the cost explanation nothing to attach
    #: to, whereas a disabled control showing the value that will actually be used
    #: answers "what ran?" -- which matters most when comparing runtimes.
    interactive_when: Callable[[Any], bool] | None = None
    #: Value to force when ``interactive_when`` is False, so a disabled control
    #: shows the real number instead of a stale one the run will ignore.
    forced_value: Callable[[Any], Any] | None = None


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
    costing a full-resolution render, and it is not user-overridable.

    So the control is **disabled, not hidden**, and forced to the value that will
    actually be used. Hiding it would read as a broken app and would leave the
    cost explanation with nothing to attach to; showing a stale 704 next to a run
    that rendered 256 would be worse still.
    """
    return not mode.min_resolution


def _forced_resolution(mode: Any) -> Any:
    """``modes.MIN_RESOLUTION`` when the mitigation applies, else no change.

    Read from the registry rather than restated here -- the number the UI shows
    and the number ``build_kwargs`` sends are then the same constant by
    construction, not by coincidence.
    """
    if not mode.min_resolution:
        return None
    from .. import modes

    return modes.MIN_RESOLUTION


def _offers_stage_2(mode: Any) -> bool:
    """False where the registry already forces stage 2 off (PLAN §3.2).

    ``generate.stage_2_enabled()`` is ``request.enable_stage_2 and not
    spec.skip_stage_2`` -- the registry wins over the request. So for those modes
    the checkbox cannot change the outcome, and the UI says so by disabling it
    rather than by letting the user tick a box that the backend then ignores.
    """
    return not mode.skip_stage_2


def _forced_stage_2(mode: Any) -> Any:
    """Pin the box to the value the backend will actually use."""
    return None if not mode.skip_stage_2 else False


# ---------------------------------------------------------------------------
# the spec
#
# Order here is the order the widgets read on screen AND the order values are
# marshalled in. One tuple, both jobs -- that is the whole point.
# ---------------------------------------------------------------------------

#: Field kinds that carry a user-supplied asset rather than a number or string.
MEDIA_KINDS = frozenset({"image", "video", "audio"})

#: The shot-list columns, declared once. The Dataframe headers AND the index the
#: anchor readout reads both come from here, so reordering the table cannot leave
#: the readout pointing at the wrong column -- it would otherwise read prompt text
#: as a frame number and silently render nothing.
#:
#: The anchor is a **frame**, not a latent index. Users think in frames; a latent
#: index is an artefact of the VAE's 8:1 temporal compression and means nothing
#: to someone generating a video. ``conditioning.py`` converts frame -> latent,
#: which is the single place PLAN §5.7 row 1 assigns that arithmetic to.
SHOT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Prompt segment", "str"),
    ("Anchor (frame)", "number"),
    ("Strength", "number"),
)
SHOT_HEADERS: tuple[str, ...] = tuple(name for name, _ in SHOT_COLUMNS)
SHOT_DATATYPES: tuple[str, ...] = tuple(kind for _, kind in SHOT_COLUMNS)
SHOT_ANCHOR_COLUMN: int = SHOT_HEADERS.index("Anchor (frame)")

FIELDS: tuple[Field, ...] = (
    # -- what the model is told to make -------------------------------------
    Field(
        key="prompt",
        kind="text",
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
        # The anchor is a FRAME, per the ruled convention in
        # tests/test_shot_anchor_convention.py: the user types a pixel frame and
        # `conditioning.py` converts it to a latent index exactly once. This
        # label used to say "latent index", which instructed users to type the
        # one thing that would land their anchor in the wrong place.
        label="Shot list — prompt segment · anchor (frame) · strength",
        info="",
        group="media",
        default=[],
        # Gradio 6 renamed these: `row_count=(0, "dynamic")` and
        # `col_count=(3, "fixed")` are Gradio 5 idioms that now warn. The 6.x
        # spelling is a plain int for the initial count, with the min/max pair
        # moved to `row_limits`/`column_limits` -- which 6.24 accepts but does
        # not yet implement, so passing them only produces a warning. Omitted.
        options={
            "headers": list(SHOT_HEADERS),
            "datatype": list(SHOT_DATATYPES),
            "column_count": len(SHOT_COLUMNS),
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
        interactive_when=_controls_resolution,
        forced_value=_forced_resolution,
    ),
    Field(
        key="height",
        kind="number",
        label="Height",
        info="Must be a multiple of 32 — upstream raises otherwise.",
        group="shape",
        default=480,
        options={"minimum": 32, "step": 32, "precision": 0},
        interactive_when=_controls_resolution,
        forced_value=_forced_resolution,
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
        key="enable_stage_2",
        kind="checkbox",
        label="Stage 2 — upscale and refine",
        info="The 2.5 quality path: a latent upsample plus three more denoise steps. Real extra GPU-seconds on a quota-metered Space, so it is opt-in.",
        group="sampling",
        default=False,
        interactive_when=_offers_stage_2,
        forced_value=_forced_stage_2,
    ),
    Field(
        key="seed",
        kind="number",
        label="Seed",
        info=(
            "-1 draws a fresh seed per run; the seed actually used is reported after the run. "
            "Same seed + same settings reproduces a run exactly."
        ),
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
        info=(
            "img2img-style strength over your audio: 0.0 keeps it as-is. At 1.0 your upload "
            "contributes NOTHING — the run is a full-price generation of fresh audio."
        ),
        group="sampling",
        default=0.5,
        options={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
        visible_when=_accepts("audio"),
    ),
    # NOTE: "Guidance scale" and "Steps" were here and are deliberately gone.
    # The distilled recipe fixes both -- `guidance_scale=1.0`, and an explicit
    # sigma schedule rather than a step count. A user raising guidance fights
    # the schedule and gets worse output with no way to know why, and the step
    # count became inert the moment the sigmas were always sent. An advertised
    # lever attached to nothing is a defect; one attached to HARM is worse.
    # `pipelines.distilled_kwargs()` owns the whole sampling set.
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
    # -- APPENDED, and it belongs here rather than next to the prompt ---------
    # This tuple is the endpoint's positional signature: `blocks.py` builds
    # `inputs=` from it, `scripts/space_matrix.py` builds its `predict` args
    # from it, and any external client that has ever called `/on_generate` was
    # handed today's order. Inserting a field in the middle shifts every value
    # after it one slot to the left for those callers -- prompt into
    # negative_prompt, width into height -- which generates something plausible
    # and wrong with no error anywhere. Appending cannot do that. The layout
    # position is decided separately, in `blocks.py`, which renders this control
    # directly under the prompt where it belongs on screen.
    Field(
        key="enhance_prompt",
        kind="checkbox",
        label="Enhance my prompt (Gemma)",
        info=(
            "Rewrites your line into a full LTX-2.5 caption first, using the model's own "
            "system prompt — image-aware when you supply a conditioning frame. It runs on the "
            "GPU inside this run's reservation, so it adds a second or two of quota rather "
            "than a separate wait. The prompt actually sent is reported with the run."
        ),
        group="prompt",
        default=False,
        # Same predicate as the prompt box itself: a mode that discards the
        # prompt has nothing to enhance, and a checkbox offering to improve a
        # field that is not on screen is the inert-lever shape again.
        visible_when=_takes_a_prompt,
    ),
)

#: The ordering contract. Built once, read everywhere -- see the module docstring.
FIELD_KEYS: tuple[str, ...] = tuple(f.key for f in FIELDS)

#: Only these take an update when the mode changes, in this order. A field with
#: neither predicate is never touched by the selector; a ``gr.State`` has neither
#: and must not be, since it has no ``visible``/``interactive`` property at all.
TOGGLEABLE_FIELDS: tuple[Field, ...] = tuple(
    f for f in FIELDS if f.visible_when is not None or f.interactive_when is not None
)


def field_by_key(key: str) -> Field:
    """Look up one field, raising rather than returning ``None`` on a typo."""
    for spec in FIELDS:
        if spec.key == key:
            return spec
    raise KeyError(f"no UI field named {key!r}. Declared fields: {', '.join(FIELD_KEYS)}")


def _assert_unique(keys: Sequence[str]) -> None:
    """A duplicated key would make the pool smaller than the key list and
    quietly desynchronise the marshalling. Catch it at import, not at runtime."""
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate UI field keys: {', '.join(duplicates)}")


_assert_unique(FIELD_KEYS)
