"""Spec for the stage-2 refinement chain — PLAN §3.2, §7.4.

**Three mutations, all of them green against the suite as it stood:**

* ``stage_2_enabled`` → ``return True`` — the refinement chain runs on **every**
  generation, for every user, on a quota-metered Space. A latent upsample plus
  three more denoise steps, silently, forever. Nothing in the app would say so;
  the runs would just cost more and nobody would know why.
* ``stage_2_enabled`` → ``return False`` — the headline 2.5 quality path is dead.
  The checkbox still ticks, the status still says success, and the output is
  quietly the stage-1 result. This is §5.3's silent-wrongness shape applied to a
  feature rather than to a file.
* drop the registry override — the user's toggle beats ``skip_stage_2``, so the
  audio-only modes pay for upscaling a frame that is generated and thrown away.

`grep -rn 'stage_2_enabled' tests/` returned **0**, and so did ``enable_stage_2``.

The asymmetry is what makes this worth specifying rather than eyeballing: *on*
and *off* are not equally bad in the same direction. Forced-on costs money on
every run; forced-off costs quality on every run; and the registry override
exists because for four modes the user's answer is simply wrong — the video is
discarded, so refining it is pure quota.

Specified at two levels on purpose. The unit level pins the decision; the
``generate()`` level pins that the decision reaches the pipeline, because a
correct predicate whose result is never passed on is the same outage with better
unit tests.
"""

from __future__ import annotations

import pytest
from test_exports_contract import (
    MODE_IDS,
    FakePipe,
    factory_returning,
    one_muxed_file_encoder,
)

from ltx25_studio import generate as generate_module
from ltx25_studio import modes, zerogpu
from ltx25_studio.ui import fields, handlers

STAGE_2_KWARG = "enable_stage_2"

#: The four modes whose video is generated and discarded (PLAN §3.2 rows 8-11).
SKIP_MODES = tuple(m for m in MODE_IDS if modes.MODES[m].skip_stage_2)
HONOURS_MODES = tuple(m for m in MODE_IDS if not modes.MODES[m].skip_stage_2)


def pipeline_kwargs_for(mode_id, tmp_path, **overrides):
    """Run one generation and return the kwargs the pipeline was actually called with."""
    pipe = FakePipe()
    generate_module.generate(
        mode=mode_id,
        seed=7,
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
        **overrides,
    )
    assert pipe.calls, "the pipeline was never called"
    return pipe.calls[-1]


def a_request(mode="text_to_video", **overrides):
    return generate_module.GenerationRequest(mode=mode, **overrides)


# --------------------------------------------------------------------------
# the registry still says what these specs assume
# --------------------------------------------------------------------------


def test_the_registry_marks_exactly_the_discarded_video_modes_as_skip():
    """PLAN §3.2: the mitigation applies to the modes whose video is thrown away.

    Mode 11 (``audio_to_video``) exports video, so it gets none of the audio-only
    mitigations — a mistake here would halve the quality of a mode that returns
    video, which is why it is asserted rather than assumed.
    """
    assert SKIP_MODES == ("text_to_audio", "video_to_audio", "audio_to_audio")
    assert "audio_to_video" in HONOURS_MODES


# --------------------------------------------------------------------------
# mutation 1: stage_2_enabled -> return True
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_stage_2_is_off_unless_the_user_asks_for_it(mode_id):
    """Opt-in, for every mode. ``return True`` here bills every user for it."""
    spec = modes.get_mode(mode_id)
    assert generate_module.stage_2_enabled(spec, a_request(mode=mode_id)) is False, (
        f"mode {mode_id!r} runs the stage-2 chain by default. It is a real extra "
        f"GPU-second cost on a quota-metered Space and the user must own that trade "
        f"(PLAN §7.4)."
    )


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_a_default_run_never_sends_the_stage_2_flag_to_the_pipeline(mode_id, tmp_path):
    """The decision has to stay off all the way to the call, not just in the predicate."""
    kwargs = pipeline_kwargs_for(mode_id, tmp_path)
    assert STAGE_2_KWARG not in kwargs, (
        f"mode {mode_id!r} passed {STAGE_2_KWARG}={kwargs[STAGE_2_KWARG]!r} to the pipeline "
        f"on a default request. Every generation would pay for the refinement chain."
    )


def test_the_default_run_is_not_billed_for_stage_2():
    """The quota consequence, stated as the number the platform reserves on.

    ``estimate_duration`` inflates the reservation by half when stage 2 is on.
    Forcing the chain on would inflate every reservation, which is the part a
    user actually feels.
    """
    base = zerogpu.estimate_duration()
    refined = zerogpu.estimate_duration(enable_stage_2=True)
    assert refined > base, (
        "the duration estimator does not account for stage 2 at all, so a refined run can "
        "time out against a reservation sized for a plain one"
    )
    assert zerogpu.estimate_duration(enable_stage_2=False) == base


# --------------------------------------------------------------------------
# mutation 2: stage_2_enabled -> return False
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", HONOURS_MODES)
def test_asking_for_stage_2_turns_it_on(mode_id):
    """The feature has to be reachable, or the checkbox is decoration."""
    spec = modes.get_mode(mode_id)
    assert generate_module.stage_2_enabled(spec, a_request(mode=mode_id, enable_stage_2=True)), (
        f"mode {mode_id!r} refuses stage 2 even when the user asks for it. The 2.5 quality "
        f"path is dead and nothing reports it (PLAN §7.4)."
    )


@pytest.mark.parametrize("mode_id", HONOURS_MODES)
def test_asking_for_stage_2_reaches_the_pipeline(mode_id, tmp_path):
    """A correct predicate whose answer is dropped on the floor is the same outage."""
    kwargs = pipeline_kwargs_for(mode_id, tmp_path, enable_stage_2=True)
    assert kwargs.get(STAGE_2_KWARG) is True, (
        f"mode {mode_id!r} did not pass {STAGE_2_KWARG}=True to the pipeline despite the "
        f"user asking. The run silently returns the stage-1 result."
    )


# --------------------------------------------------------------------------
# mutation 3: the registry override is dropped
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", SKIP_MODES)
def test_the_registry_beats_the_user_where_the_video_is_discarded(mode_id):
    """PLAN §7.4: "Honouring the user's toggle there would be honouring a mistake."

    These modes render video and throw it away. Upscaling a frame nobody will
    ever see is pure quota — so the registry wins, and the UI disables the box
    rather than letting it lie.
    """
    spec = modes.get_mode(mode_id)
    assert (
        generate_module.stage_2_enabled(spec, a_request(mode=mode_id, enable_stage_2=True)) is False
    ), (
        f"mode {mode_id!r} has skip_stage_2=True — its video is generated and discarded — but "
        f"the user's toggle overrode the registry. That is quota spent upscaling a frame "
        f"nobody will see."
    )


@pytest.mark.parametrize("mode_id", SKIP_MODES)
def test_a_skip_mode_never_sends_the_stage_2_flag_however_hard_the_user_asks(mode_id, tmp_path):
    kwargs = pipeline_kwargs_for(mode_id, tmp_path, enable_stage_2=True)
    assert STAGE_2_KWARG not in kwargs, (
        f"mode {mode_id!r} passed {STAGE_2_KWARG} to the pipeline despite skip_stage_2=True."
    )


# --------------------------------------------------------------------------
# the UI must not offer a control the backend will ignore
# --------------------------------------------------------------------------


def _stage_2_update(mode_id):
    """The ``FieldUpdate`` the mode selector produces for the stage-2 checkbox."""
    toggleable = list(fields.TOGGLEABLE_FIELDS)
    index = next(i for i, spec in enumerate(toggleable) if spec.key == STAGE_2_KWARG)
    _briefing, updates = handlers.on_mode_change(mode_id)
    return updates[index]


@pytest.mark.parametrize("mode_id", SKIP_MODES)
def test_the_stage_2_checkbox_is_disabled_where_the_registry_overrides_it(mode_id):
    """A tickable box the backend ignores is a small lie with a quota-shaped cost.

    The user ticks it, pays nothing extra, gets nothing extra, and has no way to
    tell which of those three happened.
    """
    update = _stage_2_update(mode_id)
    assert update.interactive is False, (
        f"mode {mode_id!r} leaves the stage-2 checkbox interactive even though the registry "
        f"forces the chain off. The control cannot change the outcome."
    )
    assert update.value is False, (
        f"mode {mode_id!r} does not pin the stage-2 checkbox to the value the backend will "
        f"actually use ({update.value!r})."
    )


@pytest.mark.parametrize("mode_id", HONOURS_MODES)
def test_the_stage_2_checkbox_stays_usable_where_the_choice_is_real(mode_id):
    """The inverse: disabling it everywhere would kill the feature just as dead."""
    assert _stage_2_update(mode_id).interactive is True, (
        f"mode {mode_id!r} honours the stage-2 toggle in generate() but the UI disables it, "
        f"so the quality path is unreachable from the app."
    )


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_the_ui_and_the_backend_agree_about_who_decides(mode_id):
    """One registry, read by both sides. This is the seam, stated directly."""
    spec = modes.get_mode(mode_id)
    ui_offers_a_choice = _stage_2_update(mode_id).interactive
    backend_honours_a_choice = generate_module.stage_2_enabled(
        spec, a_request(mode=mode_id, enable_stage_2=True)
    )
    assert ui_offers_a_choice == backend_honours_a_choice, (
        f"mode {mode_id!r}: the UI {'offers' if ui_offers_a_choice else 'refuses'} the stage-2 "
        f"choice while the backend {'honours' if backend_honours_a_choice else 'ignores'} it."
    )
