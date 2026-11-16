"""Spec for ``ltx25_studio/modes.py`` — PLAN §3.2, §7.2.

The registry is **data, not branching**. Adding a mode means adding an entry;
it must never mean adding an ``if`` to ``generate.py``. That is what the
``build_kwargs`` callable on each entry buys us, and
``test_generate_does_not_branch_on_mode_identifiers`` (in ``test_generate.py``)
is what keeps it honest.

The identifiers below are this suite's ruling: PLAN §3.2 numbers and labels the
modes but does not give machine names, so these are fixed here as the contract.

The expensive part, stated plainly (PLAN §3.1b): the model is jointly
audio-visual and has no video-off switch. Modes 8-10 export audio only and still
pay full video compute. The registry carries that fact — a cost note, minimum
resolution, and skip-stage-2 — because a regression here does not break anything
visible, it just quietly makes those modes cost several times what they should.
"""

from __future__ import annotations

import pytest

from ltx25_studio import modes
from ltx25_studio.errors import UnknownModeError

#: PLAN §3.2, in table order.
EXPECTED_MODE_IDS = (
    "text_to_video",  # 1
    "image_to_video",  # 2
    "image_text_to_video",  # 3
    "video_to_video",  # 4
    "text_to_audio_video",  # 5
    "image_to_audio_video",  # 6
    "image_text_to_audio_video",  # 7
    "text_to_audio",  # 8  ⚠ still pays full video compute
    "video_to_audio",  # 9  ⚠
    "audio_to_audio",  # 10 ⚠
    "audio_to_video",  # 11 ⚠ pays audio encode on top
)

#: Modes 8-10: the ones that render video the user never sees.
AUDIO_ONLY_MODE_IDS = ("text_to_audio", "video_to_audio", "audio_to_audio")

#: Modes 1-7 are first-class; 8-11 carry a warning (PLAN §3.2).
FIRST_CLASS_MODE_IDS = EXPECTED_MODE_IDS[:7]
WARNED_MODE_IDS = EXPECTED_MODE_IDS[7:]

#: The four real workflows, quoted from upstream's ``_workflow_map`` in
#: ``diffusers/modular_pipelines/ltx2/modular_blocks_ltx25.py`` (PLAN §3.1).
#: 11 UI modes are a presentation layer over these — "not 11 code paths".
UPSTREAM_WORKFLOWS = ("text2video", "image2video", "condition", "in_context")

#: PLAN §3.3 workflow column: rows 1, 5, 8.
TEXT2VIDEO_MODE_IDS = ("text_to_video", "text_to_audio_video", "text_to_audio")

#: PLAN §3.3 workflow column: rows 2, 3, 4, 6, 7, 9, 10, 11.
CONDITION_MODE_IDS = (
    "image_to_video",
    "image_text_to_video",
    "video_to_video",
    "image_to_audio_video",
    "image_text_to_audio_video",
    "video_to_audio",
    "audio_to_audio",
    "audio_to_video",
)

EXPECTED_EXPORTS = {
    "text_to_video": ("video",),
    "image_to_video": ("video",),
    "image_text_to_video": ("video",),
    "video_to_video": ("video",),
    "text_to_audio_video": ("audio", "video"),
    "image_to_audio_video": ("audio", "video"),
    "image_text_to_audio_video": ("audio", "video"),
    "text_to_audio": ("audio",),
    "video_to_audio": ("audio",),
    "audio_to_audio": ("audio",),
    # Mode 11 conditions on audio and used to export video ONLY, which returned
    # an mp4 with zero audio streams to a user who had just uploaded a track.
    # It now exports both; the muxed track is the model's own, and `modes.py`
    # carries the ruling and its six reasons at the spec.
    "audio_to_video": ("audio", "video"),
}

EXPECTED_INPUTS = {
    "text_to_video": ("text",),
    "image_to_video": ("image",),
    "image_text_to_video": ("image", "text"),
    "video_to_video": ("video",),
    "text_to_audio_video": ("text",),
    "image_to_audio_video": ("image",),
    "image_text_to_audio_video": ("image", "text"),
    "text_to_audio": ("text",),
    "video_to_audio": ("video",),
    "audio_to_audio": ("audio",),
    "audio_to_video": ("audio",),
}

#: What each mode can use without needing it. Modes 04/09/10/11 always sent a
#: prompt the UI hid; it is declared so the UI can show it. Mode 11's image is
#: an optional first frame, to be verified on real weights.
EXPECTED_OPTIONAL_INPUTS = {
    "text_to_video": (),
    "image_to_video": (),
    "image_text_to_video": (),
    "video_to_video": ("text",),
    "text_to_audio_video": (),
    "image_to_audio_video": (),
    "image_text_to_audio_video": (),
    "text_to_audio": (),
    "video_to_audio": ("text",),
    "audio_to_audio": ("text",),
    "audio_to_video": ("text", "image"),
}

REQUIRED_FIELDS = (
    "id",
    "label",
    "pipeline",
    "inputs",
    "optional_inputs",
    "exports",
    "cost_note",
    "min_resolution",
    "skip_stage_2",
    "build_kwargs",
)


# --------------------------------------------------------------------------
# completeness
# --------------------------------------------------------------------------


def test_all_eleven_modes_are_registered():
    assert len(modes.MODES) == 11


def test_mode_identifiers_match_the_plan_in_order():
    assert tuple(modes.MODES) == EXPECTED_MODE_IDS


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_mode_is_reachable_by_identifier(mode_id):
    assert modes.get_mode(mode_id).id == mode_id


def test_an_unknown_mode_raises_and_lists_the_real_ones():
    with pytest.raises(UnknownModeError) as excinfo:
        modes.get_mode("text_to_hologram")
    message = str(excinfo.value)
    assert "text_to_hologram" in message
    assert "text_to_video" in message, f"the error must list what IS available; got {message!r}"


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_mode_has_a_human_label(mode_id):
    label = modes.get_mode(mode_id).label
    assert isinstance(label, str) and label.strip()
    assert label != mode_id, f"{mode_id}: the label is for humans, not a repeat of the id"


def test_labels_are_unique():
    labels = [modes.get_mode(m).label for m in modes.MODES]
    assert len(set(labels)) == len(labels)


# --------------------------------------------------------------------------
# the registry is data
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_mode_carries_every_required_field(mode_id, field):
    assert hasattr(modes.get_mode(mode_id), field), f"{mode_id} is missing {field!r}"


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_every_mode_has_a_callable_kwarg_builder(mode_id):
    assert callable(modes.get_mode(mode_id).build_kwargs)


def accepted_request(mode_id, make_request, **overrides):
    """A request carrying everything ``mode_id`` accepts, and nothing it does not.

    The builders refuse a value their mode does not take, so "feed every mode
    every medium" is no longer a request any builder will build.
    """
    spec = modes.get_mode(mode_id)
    accepted = set(spec.inputs) | set(spec.optional_inputs)
    fields = {kind: object() for kind in ("image", "video", "audio") if kind in accepted}
    if "audio" in accepted:
        fields["audio_latents"] = object()
    if "text" not in accepted:
        fields.update(prompt="", negative_prompt="")
    fields.update(overrides)
    return make_request(mode=mode_id, **fields)


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_no_dead_modes_every_builder_produces_kwargs(mode_id, make_request):
    """PLAN §7.2: a registered mode that no code path can execute is a lie in the
    UI. Every builder must accept a request and return pipeline kwargs."""
    request = accepted_request(mode_id, make_request)
    kwargs = modes.get_mode(mode_id).build_kwargs(request)
    assert isinstance(kwargs, dict) and kwargs, f"{mode_id}: builder returned {kwargs!r}"


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_every_mode_declares_a_known_pipeline_kind(mode_id):
    assert modes.get_mode(mode_id).pipeline in modes.PIPELINE_KINDS


def test_pipeline_kinds_use_the_upstream_workflow_vocabulary():
    """PLAN §3.1: the workflow names are upstream's, not ours. Inventing a
    parallel vocabulary is how a registry entry and the pipeline it names drift
    apart."""
    assert set(modes.PIPELINE_KINDS) <= set(UPSTREAM_WORKFLOWS), (
        f"unknown workflows: {set(modes.PIPELINE_KINDS) - set(UPSTREAM_WORKFLOWS)}"
    )


@pytest.mark.parametrize("mode_id", TEXT2VIDEO_MODE_IDS)
def test_text2video_modes_declare_the_text2video_workflow(mode_id):
    assert modes.get_mode(mode_id).pipeline == "text2video"


@pytest.mark.parametrize("mode_id", CONDITION_MODE_IDS)
def test_condition_modes_declare_the_condition_workflow(mode_id):
    assert modes.get_mode(mode_id).pipeline == "condition"


def test_every_mode_is_accounted_for_by_a_workflow():
    assert set(TEXT2VIDEO_MODE_IDS) | set(CONDITION_MODE_IDS) == set(EXPECTED_MODE_IDS)


# --------------------------------------------------------------------------
# inputs and exports
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id,expected", sorted(EXPECTED_INPUTS.items()))
def test_mode_declares_its_inputs(mode_id, expected):
    assert set(modes.get_mode(mode_id).inputs) == set(expected)


@pytest.mark.parametrize("mode_id,expected", sorted(EXPECTED_OPTIONAL_INPUTS.items()))
def test_mode_declares_its_optional_inputs(mode_id, expected):
    assert set(modes.get_mode(mode_id).optional_inputs) == set(expected)


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_no_input_is_both_required_and_optional(mode_id):
    spec = modes.get_mode(mode_id)
    assert not set(spec.inputs) & set(spec.optional_inputs)


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
@pytest.mark.parametrize("kind", ["text", "image", "video", "audio"])
def test_accepts_and_requires_read_the_registry(mode_id, kind):
    spec = modes.get_mode(mode_id)
    assert modes.requires(spec, kind) == (kind in EXPECTED_INPUTS[mode_id])
    assert modes.accepts(spec, kind) == (
        kind in EXPECTED_INPUTS[mode_id] or kind in EXPECTED_OPTIONAL_INPUTS[mode_id]
    )


@pytest.mark.parametrize("mode_id,expected", sorted(EXPECTED_EXPORTS.items()))
def test_mode_declares_what_it_exports(mode_id, expected):
    assert set(modes.get_mode(mode_id).exports) == set(expected)


def test_every_mode_exports_something():
    for mode_id in modes.MODES:
        assert modes.get_mode(mode_id).exports, mode_id


# --------------------------------------------------------------------------
# the honest cost of the audio modes (PLAN §3.1b, §3.2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", FIRST_CLASS_MODE_IDS)
def test_first_class_modes_carry_no_cost_warning(mode_id):
    assert modes.get_mode(mode_id).cost_note is None


@pytest.mark.parametrize("mode_id", WARNED_MODE_IDS)
def test_warned_modes_carry_a_cost_note(mode_id):
    note = modes.get_mode(mode_id).cost_note
    assert isinstance(note, str) and note.strip(), (
        f"{mode_id}: PLAN §3.2 requires the UI be able to say, in words, what this costs"
    )


@pytest.mark.parametrize("mode_id", AUDIO_ONLY_MODE_IDS)
def test_audio_only_modes_say_they_still_render_video(mode_id):
    note = modes.get_mode(mode_id).cost_note
    assert "video" in note.lower(), (
        f"{mode_id}: a user who does not know it still renders video reads the runtime "
        f"as a bug; got {note!r}"
    )


@pytest.mark.parametrize("mode_id", AUDIO_ONLY_MODE_IDS)
def test_audio_only_modes_request_minimum_resolution(mode_id):
    assert modes.get_mode(mode_id).min_resolution is True


@pytest.mark.parametrize("mode_id", AUDIO_ONLY_MODE_IDS)
def test_audio_only_modes_skip_stage_two(mode_id):
    assert modes.get_mode(mode_id).skip_stage_2 is True


@pytest.mark.parametrize("mode_id", [m for m in EXPECTED_MODE_IDS if m not in AUDIO_ONLY_MODE_IDS])
def test_video_producing_modes_do_not_skip_stage_two(mode_id):
    """The mitigation is for modes whose video is thrown away. Applying it to a
    mode that returns video would silently halve its quality."""
    spec = modes.get_mode(mode_id)
    assert spec.skip_stage_2 is False
    assert spec.min_resolution is False


def test_audio_to_video_is_not_treated_as_an_audio_only_mode():
    """§3.2 row 11 exports video; it pays audio encode *on top*, so it gets a cost
    note but none of the audio-only mitigations."""
    spec = modes.get_mode("audio_to_video")
    assert "video" in spec.exports
    assert spec.min_resolution is False
    assert spec.skip_stage_2 is False


def test_the_cost_note_agrees_with_which_track_the_run_will_actually_return():
    """The annunciator is where the old ruling survived longest, so pin it here.

    ``cost_note`` is shown before the button is pressed and ``generate``'s
    warnings after -- two surfaces describing one fact, which is exactly the
    shape that drifts. This asserts them against the same predicate rather than
    against each other's wording: if ``_muxes_supplied_audio`` says the container
    carries the user's own upload, the sentence the user reads beforehand has to
    say so too. Before the flip it said the precise opposite.
    """
    from ltx25_studio import generate as generate_module

    spec = modes.get_mode("audio_to_video")
    assert generate_module._muxes_supplied_audio(spec.exports), (
        "mode 11 is the mode the supplied-audio ruling is about; if this is False the "
        "ruling has been disconnected from the mode it was made for"
    )
    note = spec.cost_note.lower()
    assert "your audio" in note, (
        f"the annunciator does not tell the user the sound will be his own: {spec.cost_note!r}"
    )
    assert "held fixed" in note, (
        f"the annunciator states the behaviour and not the reason for it: {spec.cost_note!r}"
    )
    assert "suppress" not in note, f"the annunciator repeats the unproven claim: {note!r}"
    assert "model's own track" not in note, (
        f"the annunciator still carries the superseded ruling's promise: {spec.cost_note!r}"
    )


# --------------------------------------------------------------------------
# what the builders actually build
# --------------------------------------------------------------------------


def test_text_to_video_builds_a_plain_prompt_call(make_request):
    kwargs = modes.get_mode("text_to_video").build_kwargs(
        make_request(mode="text_to_video", prompt="a lighthouse in fog")
    )
    assert kwargs["prompt"] == "a lighthouse in fog"
    assert kwargs["width"] == 704
    assert kwargs["height"] == 480
    assert kwargs["num_frames"] == 121
    assert "conditions" not in kwargs, "a text-only call must not carry a condition list"


def test_image_to_video_builds_a_condition_list(make_request):
    """PLAN §3.1a: image and video conditioning are the same mechanism."""
    kwargs = modes.get_mode("image_to_video").build_kwargs(
        make_request(mode="image_to_video", image=object(), prompt="")
    )
    assert kwargs["conditions"], "an image mode must produce at least one condition"


def test_image_to_video_sends_an_empty_prompt(make_request):
    """§3.2 row 2 is explicit: one condition, empty prompt."""
    kwargs = modes.get_mode("image_to_video").build_kwargs(
        make_request(mode="image_to_video", image=object(), prompt="", negative_prompt="")
    )
    assert kwargs["prompt"] == ""


def test_image_to_video_refuses_a_prompt_rather_than_dropping_it(make_request):
    """The prompt used to be overwritten here, silently. Dropping it, with a
    note, is ``generate.scope_request_to_mode``'s job; a builder handed one
    anyway was called on an unscoped request, which is a bug to surface."""
    from ltx25_studio.errors import ConstraintError

    with pytest.raises(ConstraintError) as caught:
        modes.get_mode("image_to_video").build_kwargs(
            make_request(mode="image_to_video", image=object(), prompt="ignored")
        )
    assert "prompt" in str(caught.value)


#: A non-default value for every field a builder could wrongly consume.
FOREIGN = {
    "text": {"prompt": "a stale prompt"},
    "image": {"image": object()},
    "video": {"video": object()},
    "audio": {"audio": object()},
}


@pytest.mark.parametrize(
    ("mode_id", "kind"),
    [
        (m, k)
        for m in EXPECTED_MODE_IDS
        for k in FOREIGN
        if k not in EXPECTED_INPUTS[m] and k not in EXPECTED_OPTIONAL_INPUTS[m]
    ],
)
def test_a_builder_refuses_a_kind_its_mode_does_not_take(mode_id, kind, make_request):
    """Defence in depth: a builder called directly must not resurrect K1."""
    from ltx25_studio.errors import ConstraintError

    request = accepted_request(mode_id, make_request, **FOREIGN[kind])
    with pytest.raises(ConstraintError):
        modes.get_mode(mode_id).build_kwargs(request)


@pytest.mark.parametrize(
    "mode_id", [m for m in EXPECTED_MODE_IDS if "audio" not in EXPECTED_INPUTS[m]]
)
def test_a_builder_refuses_audio_latents_where_audio_is_not_taken(mode_id, make_request):
    from ltx25_studio.errors import ConstraintError

    request = accepted_request(mode_id, make_request, audio_latents=object())
    with pytest.raises(ConstraintError):
        modes.get_mode(mode_id).build_kwargs(request)


def test_a_builder_refuses_a_shot_row(make_request):
    from ltx25_studio.errors import ConstraintError

    request = make_request(
        mode="image_to_video", image=object(), prompt="", conditions=(["a shot", 9, 0.5],)
    )
    with pytest.raises(ConstraintError):
        modes.get_mode("image_to_video").build_kwargs(request)


def test_audio_to_video_turns_an_optional_image_into_one_first_frame_condition(make_request):
    kwargs = modes.get_mode("audio_to_video").build_kwargs(
        make_request(mode="audio_to_video", audio=object(), image="first-frame")
    )
    assert [(c.source, c.kind, c.index, c.strength) for c in kwargs["conditions"]] == [
        ("first-frame", "image", 0, modes.IMAGE_CONDITION_STRENGTH)
    ]


def test_image_text_to_video_keeps_the_prompt(make_request):
    kwargs = modes.get_mode("image_text_to_video").build_kwargs(
        make_request(mode="image_text_to_video", image=object(), prompt="golden hour")
    )
    assert kwargs["prompt"] == "golden hour"
    assert kwargs["conditions"]


def test_video_to_video_conditions_at_less_than_full_strength(make_request):
    """§3.2 row 4: ``LTX2VideoCondition(frames=<video>, strength<1.0)`` — at
    strength 1.0 the input is pinned and nothing is generated."""
    kwargs = modes.get_mode("video_to_video").build_kwargs(
        make_request(mode="video_to_video", video=object(), strength=0.7)
    )
    conditions = kwargs["conditions"]
    assert conditions
    strengths = [
        getattr(c, "strength", c.get("strength") if isinstance(c, dict) else None)
        for c in conditions
    ]
    assert all(s is not None and s < 1.0 for s in strengths), strengths


@pytest.mark.parametrize("mode_id,level", [("audio_to_audio", 0.421875), ("audio_to_video", 0.0)])
def test_audio_input_modes_pass_latents_and_their_own_audio_level(mode_id, level, make_request):
    """PLAN §3.1c: there is no ``audio=`` kwarg. Audio enters as ``audio_latents``
    with the audio stream's own level; ``noise_scale`` would also set the video's
    initial noise on the condition pipeline, so it is not sent."""
    kwargs = modes.get_mode(mode_id).build_kwargs(
        make_request(mode=mode_id, audio=object(), audio_latents=object(), noise_scale=0.3)
    )
    assert "audio_latents" in kwargs
    assert kwargs["audio_noise_level"] == level
    assert "noise_scale" not in kwargs
    assert "audio" not in kwargs, "there is no `audio=` argument on any LTX2 pipeline"


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_no_builder_passes_a_none_valued_kwarg(mode_id, make_request):
    """A ``None`` in the kwargs is how an unset UI field becomes a confusing
    upstream TypeError several layers away."""
    request = accepted_request(mode_id, make_request)
    kwargs = modes.get_mode(mode_id).build_kwargs(request)
    none_valued = [k for k, v in kwargs.items() if v is None]
    assert not none_valued, f"{mode_id} builds None for {none_valued}"


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_builders_are_pure(mode_id, make_request):
    """Same request in, same kwargs out — and the request is not mutated."""
    request = accepted_request(mode_id, make_request)
    before = vars(request).copy()
    first = modes.get_mode(mode_id).build_kwargs(request)
    second = modes.get_mode(mode_id).build_kwargs(request)
    assert set(first) == set(second)
    scalars = (str, int, float, bool, type(None))
    for key, value in first.items():
        if isinstance(value, scalars):
            assert second[key] == value, f"{mode_id}: {key} differs between calls"
    assert vars(request) == before, f"{mode_id}'s builder mutated the request"


def test_the_global_ceiling_is_not_the_lever():
    """MAX_VIDEO_CONDITION_STRENGTH never binds, so lowering it fixes nothing.

    Pinned because it has been proposed as a fix twice: ``min(request.strength,
    0.99)`` with a request default of 0.8 never reaches 0.99. Kept after the
    `video_to_audio` silence turned out to be the distilled recipe rather than
    condition strength at all -- the constant was wrong both times, and this
    test is what stops it being reached for a third.
    """
    assert modes.MAX_VIDEO_CONDITION_STRENGTH > 0.8, (
        "the global ceiling now binds at the request default; if that is intended the "
        "comment saying it never binds needs to go"
    )
