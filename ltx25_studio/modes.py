"""The 11-mode registry: mode metadata as data, plus per-mode kwarg builders.

PLAN §3.1, §3.2, §7.2.

**Data, not branching.** Adding a mode means adding an entry here; it must never
mean adding an ``if`` to ``generate.py``. That is what the ``build_kwargs``
callable on each entry buys us, and it is the property the architecture specs
enforce.

**Eleven modes, four workflows.** The UI's eleven modes are a presentation layer
over upstream's four -- ``text2video``, ``image2video``, ``condition``,
``in_context`` -- plus the audio-latent input route and the audio-only export.
Describing them as eleven independent code paths would be dishonest and would
produce a far worse registry.

**The expensive truth (PLAN §3.1b).** The model is jointly audio-visual and has
no video-off switch: every call renders video and returns ``(video, audio)``. So
the modes that export audio only still pay full video compute. The registry
carries that as a ``cost_note`` the UI can show, plus the two mitigations --
minimum resolution and skipping the stage-2 upscale. A regression here breaks
nothing visible; it just quietly makes those modes cost several times what they
should, which is why it is data rather than a comment.

Imports nothing heavy: the builders return plain kwargs and light condition
specs, so ``conditioning.py`` owns the actual ``LTX2VideoCondition``
construction and this module stays testable with no diffusers present.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from . import audio_conditioning
from .conditioning import ConditionSpec
from .errors import ConstraintError, UnknownModeError

__all__ = [
    "IMAGE_CONDITION_STRENGTH",
    "KEPT_VIDEO_STRENGTH",
    "MAX_VIDEO_CONDITION_STRENGTH",
    "MIN_RESOLUTION",
    "MODES",
    "PIPELINE_KINDS",
    "ConditionSpec",
    "ModeSpec",
    "accepted_kinds",
    "accepts",
    "condition_kind",
    "counterpart_label_for",
    "get_mode",
    "modes_for_pipeline",
    "requires",
]

#: Upstream's workflow vocabulary, from ``_workflow_map`` in
#: ``diffusers/modular_pipelines/ltx2/modular_blocks_ltx25.py``. Ours must be a
#: subset: inventing a parallel vocabulary is how a registry entry and the
#: pipeline it names drift apart.
PIPELINE_KINDS = ("text2video", "condition")

#: Resolution used by modes whose video is generated and then discarded. There
#: is no minimum enforced upstream -- the only rule is divisibility by 32 -- so
#: this is chosen as the smallest size we are willing to ship, not a limit.
MIN_RESOLUTION = 256

#: A still image anchors its frame completely.
IMAGE_CONDITION_STRENGTH = 1.0

#: ``video_to_video`` conditioning stays below 1.0: at full strength the input
#: frames are pinned and nothing is generated (``video_to_audio`` wants exactly
#: that; see :data:`KEPT_VIDEO_STRENGTH`).
#:
#: ⚠️ **This ceiling never binds in practice, and it is not the lever anyone
#: wants.** ``_conditions`` takes ``min(request.strength,
#: MAX_VIDEO_CONDITION_STRENGTH)`` and ``GenerationRequest.strength`` defaults to
#: **0.8**, so lowering this constant changes nothing at all.
#:
#: Recorded because it has been proposed as a fix twice. When `video_to_audio`
#: came back at −55 dBFS this was named as the culprit; it was not, and neither
#: was condition strength. ✅ The cause was the distilled recipe reaching only
#: the stage-2 path, and with that fixed H9 runs healthy at the **unchanged**
#: default of 0.8. A sweep showing 0.8 → −55.0 and 0.5 → −5.1 measured a real
#: interaction with the broken guidance — inside a pipeline that no longer
#: exists. ``test_the_global_ceiling_is_not_the_lever`` fails if this stops
#: being true.
MAX_VIDEO_CONDITION_STRENGTH = 0.99

#: The video is kept exactly and only sound is generated (``video_to_audio``):
#: every video token is told it is clean.
KEPT_VIDEO_STRENGTH = 1.0


@dataclass(frozen=True)
class ModeSpec:
    """One UI mode: what it takes, what it returns, and what it costs."""

    id: str
    label: str
    #: Which upstream workflow executes it.
    pipeline: str
    #: Input kinds the mode REQUIRES. A run missing one is refused.
    inputs: tuple[str, ...]
    #: Tracks written to disk. A mode exporting only audio still renders video.
    exports: tuple[str, ...]
    build_kwargs: Callable[[object], dict]
    #: Plain-language warning for modes that cost more than their name implies.
    cost_note: str | None = None
    #: Render at :data:`MIN_RESOLUTION` -- for modes whose video is discarded.
    min_resolution: bool = False
    #: Skip the stage-2 spatial upscale. Only ever correct when the video is
    #: discarded; applying it to a mode that returns video halves its quality.
    skip_stage_2: bool = False
    #: The sibling mode that keeps the audio this one discards, or None where
    #: there is none. Carried as DATA so a message can name it without anyone
    #: importing the registry into the writer or transforming mode ids as
    #: strings -- the latter being the branch-on-identifiers pattern the
    #: architecture spec forbids.
    audio_counterpart: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    #: Input kinds the mode can use but does not need. Anything in neither
    #: tuple is dropped from the request before the run (generate.py).
    optional_inputs: tuple[str, ...] = ()
    #: How supplied audio is treated: ``"frozen"`` (held exactly) or
    #: ``"variation"`` (noised to the request's level); None without audio input.
    audio_conditioning: str | None = None
    #: A video condition strength the mode fixes, overriding the request's.
    video_strength: float | None = None


def accepts(spec: ModeSpec, kind: str) -> bool:
    """True when the mode uses ``kind`` at all, required or optional."""
    return kind in spec.inputs or kind in spec.optional_inputs


def requires(spec: ModeSpec, kind: str) -> bool:
    """True when a run of this mode cannot start without ``kind``."""
    return kind in spec.inputs


def accepted_kinds(spec: ModeSpec) -> frozenset[str]:
    """Every input kind the mode uses, required or optional."""
    return frozenset(spec.inputs) | frozenset(spec.optional_inputs)


# ---------------------------------------------------------------------------
# builders -- pure functions of the request, no branching in generate.py
# ---------------------------------------------------------------------------

_AUDIO_ONLY_NOTE = (
    "This mode exports audio only, but the model is jointly audio-visual and has no "
    "video-off switch: it still renders video internally and you pay for it. We run at the "
    "minimum resolution and skip the stage-2 upscale to keep the cost down."
)

_AUDIO_TO_VIDEO_NOTE = (
    "This mode encodes your audio before generating, so it pays the audio encode on top of "
    "a normal video render. Your audio is held fixed while the picture is generated to it, "
    "so the file carries your audio, length-matched to the video."
)


def _base_kwargs(request, *, min_resolution: bool) -> dict:
    """Kwargs every mode passes, with no ``None`` values.

    A ``None`` in the kwargs is how an unset UI field becomes a confusing
    upstream ``TypeError`` several layers away, so unset fields are omitted
    rather than passed along empty.
    """
    width = MIN_RESOLUTION if min_resolution else getattr(request, "width", None)
    height = MIN_RESOLUTION if min_resolution else getattr(request, "height", None)
    candidates = {
        # Passed to the PIPELINE as well as to the encoder. It looks redundant
        # only because both values are 24.0 today: without this the rate the
        # model generates at and the rate the container declares are two
        # independent numbers that happen to agree, and the first change to
        # either plays the video at the wrong speed with no error at all.
        "frame_rate": getattr(request, "frame_rate", None),
        "prompt": getattr(request, "prompt", None),
        "negative_prompt": getattr(request, "negative_prompt", None),
        "width": width,
        "height": height,
        "num_frames": getattr(request, "num_frames", None),
        # guidance_scale and num_inference_steps are NOT sent from here.
        # `pipelines.distilled_kwargs()` owns the whole sampling recipe, because
        # the distilled schedule and its guidance settings are one set that must
        # not be assembled from two places -- which is how the sigmas came to be
        # applied on only one branch. See PLAN 5.5.
    }
    return {key: value for key, value in candidates.items() if value is not None}


def _with_audio(request, kwargs: dict, conditioning: str) -> dict:
    """Attach the audio-latent route and the audio stream's own noise level.

    There is no ``audio=`` argument on any LTX2 pipeline (PLAN §3.1c): audio
    enters as pre-encoded ``audio_latents``. ``noise_scale`` is NOT sent: on the
    condition pipeline it is also the video's initial noise level, so the video
    is sampled as in every other mode and the audio's level travels as
    ``audio_noise_level`` (consumed by ``pipelines.StageRunner``; see
    ``audio_conditioning``).
    """
    latents = getattr(request, "audio_latents", None)
    if latents is not None:
        kwargs["audio_latents"] = latents
    if conditioning == "frozen":
        kwargs["audio_noise_level"] = audio_conditioning.FROZEN
    else:
        noise_scale = getattr(request, "noise_scale", None)
        kwargs["audio_noise_level"] = audio_conditioning.snap_level(
            1.0 if noise_scale is None else noise_scale
        )
    return kwargs


def condition_kind(entry) -> str:
    """The input kind a programmatic condition entry carries (image unless it says)."""
    if isinstance(entry, Mapping):
        return str(entry.get("kind", "image"))
    return str(getattr(entry, "kind", "image"))


def _unaccepted(request, accepted: frozenset[str]) -> list[str]:
    """Every field on ``request`` that carries a kind outside ``accepted``."""
    found = []
    for kind in ("image", "video", "audio"):
        if kind not in accepted and getattr(request, kind, None) is not None:
            found.append(kind)
    if "audio" not in accepted and getattr(request, "audio_latents", None) is not None:
        found.append("audio_latents")
    if "text" not in accepted:
        for name in ("prompt", "negative_prompt"):
            if str(getattr(request, name, "") or "").strip():
                found.append(name)
    for entry in getattr(request, "conditions", ()) or ():
        if isinstance(entry, (list, tuple)):
            found.append("conditions (a shot-list row)")
        elif condition_kind(entry) not in accepted:
            found.append(f"conditions ({condition_kind(entry)})")
    return found


def _refuse_unaccepted(request, accepted: frozenset[str], label: str) -> None:
    """Defence in depth behind ``generate.scope_request_to_mode``.

    The scoped request never trips this. A builder called with an unscoped one
    must fail rather than quietly use a value its mode does not take.
    """
    found = _unaccepted(request, accepted)
    if found:
        raise ConstraintError(
            f"{label} was handed {', '.join(found)}, which it does not use. The request must "
            f"be scoped to the mode before its arguments are built."
        )


def _conditions(
    request, accepted: frozenset[str], video_strength: float | None = None
) -> tuple[ConditionSpec, ...]:
    """Build the condition list. Image and video share one mechanism (§3.1a)."""
    specs: list[ConditionSpec] = []
    image = getattr(request, "image", None)
    if image is not None and "image" in accepted:
        specs.append(
            ConditionSpec(source=image, kind="image", index=0, strength=IMAGE_CONDITION_STRENGTH)
        )
    video = getattr(request, "video", None)
    if video is not None and "video" in accepted:
        strength = getattr(request, "strength", None)
        if strength is None:
            strength = MAX_VIDEO_CONDITION_STRENGTH
        strength = min(float(strength), MAX_VIDEO_CONDITION_STRENGTH)
        if video_strength is not None:
            strength = float(video_strength)
        specs.append(ConditionSpec(source=video, kind="video", index=0, strength=strength))
    specs.extend(
        entry
        for entry in getattr(request, "conditions", ()) or ()
        if condition_kind(entry) in accepted
    )
    return tuple(specs)


def _text_builder(*, min_resolution: bool = False) -> Callable:
    def build(request, accepted: frozenset[str], fields: dict) -> dict:
        return _base_kwargs(request, min_resolution=min_resolution)

    return build


def _condition_builder(*, min_resolution: bool = False) -> Callable:
    def build(request, accepted: frozenset[str], fields: dict) -> dict:
        kwargs = _base_kwargs(request, min_resolution=min_resolution)
        conditions = _conditions(request, accepted, fields.get("video_strength"))
        if conditions:
            kwargs["conditions"] = conditions
        if fields.get("audio_conditioning"):
            _with_audio(request, kwargs, fields["audio_conditioning"])
        return kwargs

    return build


def _mode(builder: Callable, **fields) -> ModeSpec:
    """A registry entry whose builder knows, from the entry itself, what it accepts."""
    if fields.get("audio_conditioning") not in (None, "frozen", "variation"):
        raise ValueError(f"unknown audio_conditioning {fields['audio_conditioning']!r}")
    accepted = frozenset(fields["inputs"]) | frozenset(fields.get("optional_inputs", ()))
    label = fields["label"]

    def build_kwargs(request) -> dict:
        _refuse_unaccepted(request, accepted, label)
        kwargs = builder(request, accepted, fields)
        if "text" not in accepted:
            # PLAN §3.2 row 2: one condition, empty prompt.
            kwargs["prompt"] = ""
        return kwargs

    return ModeSpec(build_kwargs=build_kwargs, **fields)


_SPECS = (
    # -- 1-7: first class ----------------------------------------------------
    _mode(
        _text_builder(),
        id="text_to_video",
        label="Text to video",
        pipeline="text2video",
        inputs=("text",),
        exports=("video",),
        audio_counterpart="text_to_audio_video",
    ),
    _mode(
        _condition_builder(),
        id="image_to_video",
        label="Image to video",
        pipeline="condition",
        inputs=("image",),
        exports=("video",),
        audio_counterpart="image_to_audio_video",
    ),
    _mode(
        _condition_builder(),
        id="image_text_to_video",
        label="Image + text to video",
        pipeline="condition",
        inputs=("image", "text"),
        exports=("video",),
        audio_counterpart="image_text_to_audio_video",
    ),
    _mode(
        _condition_builder(),
        id="video_to_video",
        label="Video to video",
        pipeline="condition",
        inputs=("video",),
        optional_inputs=("text",),
        exports=("video",),
    ),
    _mode(
        _text_builder(),
        id="text_to_audio_video",
        label="Text to video with sound",
        pipeline="text2video",
        inputs=("text",),
        exports=("audio", "video"),
    ),
    _mode(
        _condition_builder(),
        id="image_to_audio_video",
        label="Image to video with sound",
        pipeline="condition",
        inputs=("image",),
        exports=("audio", "video"),
    ),
    _mode(
        _condition_builder(),
        id="image_text_to_audio_video",
        label="Image + text to video with sound",
        pipeline="condition",
        inputs=("image", "text"),
        exports=("audio", "video"),
    ),
    # -- 8-10: audio only, and they still render video -----------------------
    _mode(
        _text_builder(min_resolution=True),
        id="text_to_audio",
        label="Text to audio",
        pipeline="text2video",
        inputs=("text",),
        exports=("audio",),
        cost_note=_AUDIO_ONLY_NOTE,
        min_resolution=True,
        skip_stage_2=True,
    ),
    _mode(
        _condition_builder(min_resolution=True),
        video_strength=KEPT_VIDEO_STRENGTH,
        id="video_to_audio",
        label="Video to audio",
        pipeline="condition",
        inputs=("video",),
        optional_inputs=("text",),
        exports=("audio",),
        cost_note=_AUDIO_ONLY_NOTE,
        min_resolution=True,
        skip_stage_2=True,
    ),
    _mode(
        _condition_builder(min_resolution=True),
        audio_conditioning="variation",
        id="audio_to_audio",
        label="Audio to audio",
        pipeline="condition",
        inputs=("audio",),
        optional_inputs=("text",),
        exports=("audio",),
        cost_note=_AUDIO_ONLY_NOTE,
        min_resolution=True,
        skip_stage_2=True,
    ),
    # -- 11: exports video, so it gets none of the audio-only mitigations -----
    #
    # The audio is frozen: its latents are the encoded upload, unchanged, and the
    # transformer is told at every step that they are clean, so the picture is
    # generated to the sound. The file therefore carries the user's own audio
    # (`generate._muxes_supplied_audio`, derived from `exports`): it is the very
    # waveform the picture was generated to, length-matched to the video.
    _mode(
        _condition_builder(),
        audio_conditioning="frozen",
        id="audio_to_video",
        label="Audio to video",
        pipeline="condition",
        inputs=("audio",),
        # The image is an optional first frame (index 0, strength 1.0). To be
        # verified on real weights before it is kept.
        optional_inputs=("text", "image"),
        exports=("audio", "video"),
        cost_note=_AUDIO_TO_VIDEO_NOTE,
    ),
)

#: The registry, in PLAN §3.2 table order. Read-only by construction.
MODES: Mapping[str, ModeSpec] = MappingProxyType({spec.id: spec for spec in _SPECS})


def get_mode(mode_id: str) -> ModeSpec:
    """Look up a mode, or raise naming what IS registered."""
    try:
        return MODES[mode_id]
    except KeyError:
        raise UnknownModeError(mode=mode_id, known=tuple(MODES)) from None


def modes_for_pipeline(pipeline: str) -> Sequence[ModeSpec]:
    """Every mode executed by one upstream workflow."""
    return tuple(spec for spec in MODES.values() if spec.pipeline == pipeline)


def counterpart_label_for(mode_id: str) -> str | None:
    """Human label of the mode that keeps this one's audio, if there is one.

    Resolved here because the registry is the only thing that knows the
    relationship. Callers hand the label onward and stay ignorant of the
    registry, which is what keeps a message able to name a mode without the
    writer importing modes or munging identifiers.
    """
    counterpart = get_mode(mode_id).audio_counterpart
    return get_mode(counterpart).label if counterpart else None
