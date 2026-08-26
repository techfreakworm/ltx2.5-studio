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
    "encode on top of a normal video render. The file comes back with sound, and that "
    "sound is YOUR audio, muxed back in — because supplying audio suppresses the model's "
    "own audio generation, so the track it would otherwise contribute is silent. Your "
    "upload is length-matched to the video first, so what you get back is the same "
    "waveform that steered the render."
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


def _conditions(request) -> tuple[ConditionSpec, ...]:
    """Build the condition list. Image and video share one mechanism (§3.1a)."""
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
                strength=min(float(strength), MAX_VIDEO_CONDITION_STRENGTH),
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
    *, keep_prompt: bool = True, min_resolution: bool = False, audio: bool = False
) -> Callable:
    def build(request) -> dict:
        kwargs = _base_kwargs(request, min_resolution=min_resolution)
        if not keep_prompt:
            # PLAN §3.2 row 2 is explicit: one condition, empty prompt.
            kwargs["prompt"] = ""
        conditions = _conditions(request)
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
        build_kwargs=_condition_builder(min_resolution=True),
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
    #
    # ⛔ **It used to export video ONLY, and that was the defect.** A user who
    # conditioned on audio got back an mp4 with ZERO audio streams -- confirmed
    # twice on the live Space. The generated track was discarded by
    # `outputs._write_video`'s `include_audio` branch and reported through
    # `dropped_audio_warning`, which told the user to "use the matching
    # audio+video mode to keep it". **There is no such mode.** `audio_counterpart`
    # is None here, so the one mitigation the drop had pointed at nothing.
    #
    # ⛔ **RULED, THEN OVERTURNED BY MEASUREMENT.** The six reasons below chose
    # the model's generated track, and they are kept verbatim because five of
    # them are still true and the sixth is the cost the new ruling accepts. What
    # they could not weigh is a fact that did not exist when they were written.
    #
    # ✅ **A controlled discriminator on the live Space: supplying audio
    # SUPPRESSES the model's own audio generation.** Same pipeline, same
    # 512x960x121 geometry, same seed, one variable -- whether an audio input
    # was given. Both files carried a well-formed 48 kHz stereo AAC track:
    #
    #     audio_to_video       (audio IN)   video+aac ✓   -88.3 dBFS   silence
    #     text_to_audio_video  (no audio)   video+aac ✓   -24.2 dBFS   content
    #
    # 64 dB of separation. So on exactly the runs reason 1-6 govern, "the track
    # that cannot silently mislead" is digital silence -- the same emptiness the
    # original defect had, now wearing a stream header, which is arguably worse
    # because it looks like a track until you play it.
    #
    # ✅ **RULED (this mode ONLY): mux the audio the user SUPPLIED.** He provided
    # it, he expects it back, and the model's track is empty anyway. Reason 3 was
    # a true statement about the code -- the reconciled waveform was a local in
    # `generate._prepared_audio` and did not survive to export time -- and it is
    # now false by construction: `_prepared_audio` returns it as an
    # `outputs.SuppliedAudio`, captured after the reconcile and before the
    # encode, carrying the UPLOAD's sample rate so reason 4 cannot bite either.
    # Reason 5 is the price and it is paid knowingly: the file will look
    # synchronised whether or not the render followed the audio. Reasons 1, 2 and
    # 6 are unchanged and simply lost to a bigger fact.
    #
    # ⚠️ **Scope.** `audio_to_audio` (mode 10) and `video_to_audio` (mode 9) are
    # NOT changed: they are audio-export modes whose artifact IS the model's
    # track, and they are outside the ruling. Whether the same suppression makes
    # mode 10's export silent is a real question that nothing has measured; it is
    # recorded as open rather than answered by extension.
    #
    # The scope is enforced by `generate._muxes_supplied_audio`, which derives it
    # from `exports` -- a video container declared to carry sound -- rather than
    # from a flag on this spec, because a flag is the thing whoever adds the next
    # audio-input mode forgets.
    #
    # ⛔ The superseded ruling, kept in full:
    #
    # ~~**RULED: the muxed track is the audio the model GENERATED, not the audio
    # the user supplied.** Both were defensible; the generated one is the one
    # that cannot silently mislead.~~ Six reasons, in the order that decided it:
    #
    # 1. **`audio_to_audio` already answers this question.** Mode 10 takes the
    #    same audio input, through the same `_condition_builder(audio=True)`, on
    #    the same `condition` pipeline, and returns the model's track -- nobody
    #    argues it should hand the upload back. Mode 11 is that same joint render
    #    with the video kept instead of discarded. Returning a DIFFERENT track
    #    for one than for the other would make "the audio in this file" mean two
    #    things across two modes that differ only in which track survives.
    #
    # 2. **Alignment is by construction for one and not the other.** The
    #    generated audio comes off the same joint latent timeline as the frames.
    #    The upload does not: `audio.reconcile_waveform_with_video` returns it
    #    UNTOUCHED inside a +/-0.05 s tolerance, and when it does trim, it snaps
    #    to `seconds_for_latent_frames(...)` -- upstream's latent grid -- not to
    #    `num_frames / frame_rate`. The upload is never guaranteed video-length.
    #
    # 3. **The reconciled waveform does not exist at export time.** It is a local
    #    in `generate._prepared_audio`, consumed into `audio_latents` and gone.
    #    What survives is `request.audio`: the ORIGINAL, UNTRIMMED file. Muxing
    #    that would reintroduce at the output end exactly the desync `2eaeb2c`
    #    fixed at the input end -- a 10 s upload against a 5 s video conditions
    #    on 5 s and would ship with 10 s of sound.
    #
    # 4. **Rates.** The upload carries the file's own rate; the writer is handed
    #    `vocoder.config.output_sampling_rate` (48000 against the VAE's 16000 --
    #    see `conftest`'s note on that trap). Muxing the upload at the vocoder's
    #    rate pitch-shifts it silently, which is the defect class
    #    `outputs._required` exists to refuse, and carrying a second rate would
    #    double the writer's independent rate assumptions.
    #
    # 5. **It cannot mask a conditioning failure.** Muxing the upload back would
    #    make every mode-11 file look and sound synchronised whether or not the
    #    model actually followed the audio -- the user's own track papering over
    #    a render that ignored it. The generated track cannot do that: if the
    #    model ignored the conditioning, you hear it.
    #
    # 6. **Nothing is lost either way.** The user still holds their file and can
    #    mux it over the video losslessly. The generated track, once dropped, is
    #    gone and was paid for in GPU-seconds -- which is what the drop warning
    #    always said, while pointing at a sibling mode that does not exist.
    #
    # The user is TOLD which track they got, every run, through the warnings
    # channel -- `generate._prepared_audio`. Silence is what the defect was, and
    # that has not changed: the note now says the audio is his own AND says why,
    # because "here is your audio back" without the suppression finding invites
    # the next person to re-derive the six reasons and flip it again.
    ModeSpec(
        id="audio_to_video",
        label="Audio to video",
        pipeline="condition",
        inputs=("audio",),
        exports=("audio", "video"),
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
