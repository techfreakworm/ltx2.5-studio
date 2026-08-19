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

from .conditioning import ConditionSpec
from .errors import UnknownModeError

__all__ = [
    "IMAGE_CONDITION_STRENGTH",
    "MAX_VIDEO_CONDITION_STRENGTH",
    "AUDIO_EXPORT_CONDITION_STRENGTH",
    "MIN_RESOLUTION",
    "MODES",
    "PIPELINE_KINDS",
    "ConditionSpec",
    "ModeSpec",
    "counterpart_label_for",
    "get_mode",
    "modes_for_pipeline",
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

#: Video conditioning must stay below 1.0: at full strength the input frames are
#: pinned and nothing is generated.
#:
#: ⚠️ **This ceiling never binds in practice** and is not the lever anyone wants.
#: ``_conditions`` takes ``min(request.strength, MAX_VIDEO_CONDITION_STRENGTH)``
#: and ``GenerationRequest.strength`` defaults to **0.8**, so lowering this
#: constant changes nothing. The value that matters is the per-mode cap below.
MAX_VIDEO_CONDITION_STRENGTH = 0.99

#: Condition strength for **modes that export audio**, which cannot use the
#: video default. ✅ Measured on hinode, `video_to_audio`, seed 1234 throughout:
#:
#: ===========  ==========  ==========
#: strength     audio       verdict
#: ===========  ==========  ==========
#: 0.99         -38.6 dBFS  near-silent
#: **0.80**     **-55.0**   near-silent -- the old default
#: **0.50**     **-5.1**    **healthy** -- matches text_to_audio's -5.2
#: 0.20         -11.5 dBFS  healthy
#: 0.05         -13.2 dBFS  healthy
#: ===========  ==========  ==========
#:
#: ⚠️ **This is a cliff, not a curve. Do not interpolate.** 0.99 is *louder*
#: than 0.80, and 0.50 is louder than 0.20 -- the ordering is not monotonic, so
#: a value chosen between measured points ("0.6 seems safer") is a guess across
#: a discontinuity. **0.5 is used because it was measured, not because it is
#: middling.** Widening this range means running the sweep again.
#:
#: **The mechanism.** Strength sets where in the sigma schedule the run starts,
#: so at 0.8 only the last ~20% of steps execute. The video stream does not
#: care -- it is anchored by the condition. **The audio stream has nothing
#: anchoring it and simply never gets denoised**, staying near its
#: initialisation. That is what -55 dBFS is.
AUDIO_EXPORT_CONDITION_STRENGTH = 0.5


@dataclass(frozen=True)
class ModeSpec:
    """One UI mode: what it takes, what it returns, and what it costs."""

    id: str
    label: str
    #: Which upstream workflow executes it.
    pipeline: str
    #: Input kinds the UI must collect.
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


# ---------------------------------------------------------------------------
# builders -- pure functions of the request, no branching in generate.py
# ---------------------------------------------------------------------------

_AUDIO_ONLY_NOTE = (
    "This mode exports audio only, but the model is jointly audio-visual and has no "
    "video-off switch: it still renders video internally and you pay for it. We run at the "
    "minimum resolution and skip the stage-2 upscale to keep the cost down."
)

_AUDIO_TO_VIDEO_NOTE = (
    "This mode encodes your audio into latents before generating, so it pays the audio "
    "encode on top of a normal video render."
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


def _with_audio(request, kwargs: dict) -> dict:
    """Attach the audio-latent route.

    There is no ``audio=`` argument on any LTX2 pipeline (PLAN §3.1c). Audio
    enters as pre-encoded ``audio_latents`` combined with ``noise_scale``, which
    behaves like img2img strength: 1.0 discards the input entirely.
    """
    latents = getattr(request, "audio_latents", None)
    if latents is not None:
        kwargs["audio_latents"] = latents
    noise_scale = getattr(request, "noise_scale", None)
    if noise_scale is not None:
        kwargs["noise_scale"] = noise_scale
    return kwargs


def _conditions(request, *, cap: float | None = None) -> tuple[ConditionSpec, ...]:
    """Build the condition list. Image and video share one mechanism (§3.1a).

    ``cap`` is the mode's own ceiling on video condition strength, applied on top
    of :data:`MAX_VIDEO_CONDITION_STRENGTH`. See
    :data:`AUDIO_EXPORT_CONDITION_STRENGTH` for why audio-exporting modes need
    one and why its value must not be interpolated.
    """
    specs: list[ConditionSpec] = []
    image = getattr(request, "image", None)
    if image is not None:
        specs.append(ConditionSpec(source=image, kind="image", index=0))
    video = getattr(request, "video", None)
    if video is not None:
        strength = getattr(request, "strength", None)
        if strength is None:
            strength = MAX_VIDEO_CONDITION_STRENGTH
        specs.append(
            ConditionSpec(
                source=video,
                kind="video",
                index=0,
                strength=min(
                    float(strength),
                    MAX_VIDEO_CONDITION_STRENGTH if cap is None else float(cap),
                ),
            )
        )
    specs.extend(getattr(request, "conditions", ()) or ())
    return tuple(specs)


def _text_builder(*, min_resolution: bool = False, audio: bool = False) -> Callable:
    def build(request) -> dict:
        kwargs = _base_kwargs(request, min_resolution=min_resolution)
        return _with_audio(request, kwargs) if audio else kwargs

    return build


def _condition_builder(
    *,
    keep_prompt: bool = True,
    min_resolution: bool = False,
    audio: bool = False,
    condition_strength: float | None = None,
) -> Callable:
    """Build the kwargs for a conditioned mode.

    ``condition_strength`` caps the video condition for **this mode only**. A
    single global default cannot serve every mode: `video_to_video` needs a high
    strength for fidelity to its input, and that same value silences the audio
    branch. **Where a mode both takes a video condition and exports audio, the
    audio requirement is the binding one** -- a quiet video is a quality
    complaint, a silent audio track is a broken mode.
    """

    def build(request) -> dict:
        kwargs = _base_kwargs(request, min_resolution=min_resolution)
        if not keep_prompt:
            # PLAN §3.2 row 2 is explicit: one condition, empty prompt.
            kwargs["prompt"] = ""
        conditions = _conditions(request, cap=condition_strength)
        if conditions:
            kwargs["conditions"] = conditions
        if audio:
            _with_audio(request, kwargs)
        return kwargs

    return build


_SPECS = (
    # -- 1-7: first class ----------------------------------------------------
    ModeSpec(
        id="text_to_video",
        label="Text to video",
        pipeline="text2video",
        inputs=("text",),
        exports=("video",),
        build_kwargs=_text_builder(),
        audio_counterpart="text_to_audio_video",
    ),
    ModeSpec(
        id="image_to_video",
        label="Image to video",
        pipeline="condition",
        inputs=("image",),
        exports=("video",),
        build_kwargs=_condition_builder(keep_prompt=False),
        audio_counterpart="image_to_audio_video",
    ),
    ModeSpec(
        id="image_text_to_video",
        label="Image + text to video",
        pipeline="condition",
        inputs=("image", "text"),
        exports=("video",),
        build_kwargs=_condition_builder(),
        audio_counterpart="image_text_to_audio_video",
    ),
    ModeSpec(
        id="video_to_video",
        label="Video to video",
        pipeline="condition",
        inputs=("video",),
        exports=("video",),
        build_kwargs=_condition_builder(),
    ),
    ModeSpec(
        id="text_to_audio_video",
        label="Text to video with sound",
        pipeline="text2video",
        inputs=("text",),
        exports=("audio", "video"),
        build_kwargs=_text_builder(),
    ),
    ModeSpec(
        id="image_to_audio_video",
        label="Image to video with sound",
        pipeline="condition",
        inputs=("image",),
        exports=("audio", "video"),
        build_kwargs=_condition_builder(keep_prompt=False),
    ),
    ModeSpec(
        id="image_text_to_audio_video",
        label="Image + text to video with sound",
        pipeline="condition",
        inputs=("image", "text"),
        exports=("audio", "video"),
        build_kwargs=_condition_builder(),
    ),
    # -- 8-10: audio only, and they still render video -----------------------
    ModeSpec(
        id="text_to_audio",
        label="Text to audio",
        pipeline="text2video",
        inputs=("text",),
        exports=("audio",),
        build_kwargs=_text_builder(min_resolution=True),
        cost_note=_AUDIO_ONLY_NOTE,
        min_resolution=True,
        skip_stage_2=True,
    ),
    ModeSpec(
        id="video_to_audio",
        label="Video to audio",
        pipeline="condition",
        inputs=("video",),
        exports=("audio",),
        build_kwargs=_condition_builder(
            min_resolution=True, condition_strength=AUDIO_EXPORT_CONDITION_STRENGTH
        ),
        cost_note=_AUDIO_ONLY_NOTE,
        min_resolution=True,
        skip_stage_2=True,
    ),
    ModeSpec(
        id="audio_to_audio",
        label="Audio to audio",
        pipeline="condition",
        inputs=("audio",),
        exports=("audio",),
        build_kwargs=_condition_builder(min_resolution=True, audio=True),
        cost_note=_AUDIO_ONLY_NOTE,
        min_resolution=True,
        skip_stage_2=True,
    ),
    # -- 11: exports video, so it gets none of the audio-only mitigations -----
    ModeSpec(
        id="audio_to_video",
        label="Audio to video",
        pipeline="condition",
        inputs=("audio",),
        exports=("video",),
        build_kwargs=_condition_builder(audio=True),
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
