"""Spec for the seam between ``generate.py`` and ``ui/handlers.py`` — PLAN §5.3, §7.2.

**The seam nothing tested.** ``generate()`` decides what a success looks like.
``on_generate`` decides what a success looks like. When the two disagreed, both
sides' tests still passed: ``generate()``'s tests asserted the artifact existed,
the UI's tests asserted the guard fires on a missing track, and *nobody drove one
into the other*. The result was that every successful generation in the three
flagship modes rendered as "🛑 Internal error — empty output".

So this file drives the real handler with the real ``generate()``, faking only
the two GPU-shaped seams PLAN §7.3 put there for it (``pipeline_factory`` and
``encoder``). Nothing between the layers is stubbed — that is the entire point.

Both directions are specified, because a guard that never fires is as broken as
one that fires wrongly:

* **it must stay silent** — for every mode, a result that satisfies its
  ``exports`` contract renders success, with the media slots the mode declared;
* **it must fire** — for every mode and every track that mode declares, a
  success-shaped result with that track missing renders the loud error and
  hands back nothing.

The values tuple is built from ``FIELD_KEYS``, which is the same tuple
``blocks.py`` builds the ``inputs=`` list from, so this exercises the real
marshalling (``strict=True``) rather than a convenient subset of it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import STUB_WAVEFORM
from test_exports_contract import (
    MODE_IDS,
    MODE_TRACK_PAIRS,
    TRACKS,
    FakePipe,
    _pair_ids,
    factory_returning,
    one_muxed_file_encoder,
)

from ltx25_studio import generate as generate_module
from ltx25_studio import modes
from ltx25_studio.ui import handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

#: The first line of the loud error, taken from the renderer itself so this
#: cannot drift if the wording is edited.
EMPTY_OUTPUT_HEADLINE = render.empty_output_bug(
    modes.get_mode("text_to_video"), ["video"]
).splitlines()[0]

MEDIA_KINDS = ("image", "video", "audio")


def values_for(mode_id: str, **overrides) -> tuple:
    """One positional value per ``FIELD_KEYS`` entry, as Gradio would send them."""
    values = {spec.key: spec.default for spec in FIELDS}
    values["seed"] = 7
    # Hand the mode the media it declares it takes, so the request that reaches
    # the registry's kwarg builder is the shape the UI would really produce.
    for kind in modes.get_mode(mode_id).inputs:
        if kind in MEDIA_KINDS:
            values[kind] = f"conditioning-input.{kind}"
    values.update(overrides)
    return tuple(values[key] for key in FIELD_KEYS)


class Wiring:
    """Records what crossed the seam, so assertions can name the real objects."""

    def __init__(self):
        self.params: list[dict] = []
        self.results: list = []
        self.pipes: list = []

    @property
    def result(self):
        assert self.results, "on_generate never called generate()"
        return self.results[-1]


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Point ``on_generate`` at the REAL ``generate()``, GPU seams faked."""
    real = generate_module.generate
    wiring = Wiring()

    # `values_for` hands the audio modes a path-shaped string, exactly as the UI
    # hands a real filepath. Nothing used to read it -- that was the bug. Now the
    # loader does, so it is stubbed; everything downstream of it (resample, mel,
    # encode, shape assertion) runs for real against the fake audio VAE.
    from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM

    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )

    def wired_generate(**params):
        wiring.params.append(params)
        pipe = FakePipe()
        wiring.pipes.append(pipe)
        result = real(
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path,
            retention_bytes=10**9,
            **params,
        )
        wiring.results.append(result)
        return result

    monkeypatch.setattr(generate_module, "generate", wired_generate)
    return wiring


def _returns(monkeypatch, result):
    """Make ``on_generate`` see exactly this result, unchanged."""
    monkeypatch.setattr(generate_module, "generate", lambda **params: result)


# --------------------------------------------------------------------------
# direction 1: the guard must stay silent on a good run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_a_run_that_satisfies_its_exports_contract_renders_success(mode_id, wired):
    """The blocker, from the UI's side: a good run must not read as a bug."""
    _video, _audio, status = handlers.on_generate(mode_id, *values_for(mode_id))

    assert EMPTY_OUTPUT_HEADLINE not in status, (
        f"mode {mode_id!r} completed successfully and satisfied its exports contract, but the "
        f"handler rendered the internal-error banner. generate.py and ui/handlers.py disagree "
        f"about what a success looks like for this mode.\n\n{status}"
    )
    assert "Generation failed" not in status, (
        f"mode {mode_id!r} rendered a failure for a successful run:\n\n{status}"
    )
    assert status == render.success(wired.result, modes.get_mode(mode_id)), (
        f"mode {mode_id!r} did not render the success message for its own result"
    )


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_the_handler_hands_back_a_player_for_every_track_the_mode_exports(mode_id, wired):
    """A declared track must arrive at the UI, not merely exist on disk."""
    spec = modes.get_mode(mode_id)
    video, audio, _status = handlers.on_generate(mode_id, *values_for(mode_id))

    for track, value in (("video", video), ("audio", audio)):
        if track in spec.exports:
            assert value is not None, (
                f"mode {mode_id!r} declares exports={list(spec.exports)} but the handler "
                f"returned nothing for the {track} player."
            )
        else:
            assert value is None, (
                f"mode {mode_id!r} does not export {track!r}, but the handler populated its "
                f"player anyway — most likely with the previous run's output."
            )


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_the_handler_forwards_the_paths_generate_actually_produced(mode_id, wired):
    """No re-derivation at the UI layer: the artifact shown is the one written."""
    spec = modes.get_mode(mode_id)
    video, audio, _status = handlers.on_generate(mode_id, *values_for(mode_id))
    result = wired.result

    assert video == (result.video_path if "video" in spec.exports else None)
    assert audio == (result.audio_path if "audio" in spec.exports else None)


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_the_players_the_ui_shows_match_the_tracks_the_run_populates(mode_id, wired):
    """``export_flags`` builds the UI; ``exports`` drives the run. One registry, one answer.

    If these ever disagree, the user gets a visible player that can never fill —
    or a filled track with nowhere to show it.
    """
    shows_video, shows_audio = handlers.export_flags(mode_id)
    video, audio, _status = handlers.on_generate(mode_id, *values_for(mode_id))

    assert (video is not None) == shows_video, (
        f"mode {mode_id!r}: export_flags says video player visible={shows_video}, but the run "
        f"{'produced' if video is not None else 'produced no'} video."
    )
    assert (audio is not None) == shows_audio, (
        f"mode {mode_id!r}: export_flags says audio player visible={shows_audio}, but the run "
        f"{'produced' if audio is not None else 'produced no'} audio."
    )


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_the_handler_marshals_every_declared_field_into_the_request(mode_id, wired):
    """The ``strict=True`` zip is only a guarantee if something drives it."""
    handlers.on_generate(mode_id, *values_for(mode_id))
    params = wired.params[-1]

    assert params["mode"] == mode_id
    assert set(params) == {"mode", *FIELD_KEYS}, (
        f"the handler forwarded {sorted(set(params) - {'mode'})} but FIELD_KEYS declares "
        f"{sorted(FIELD_KEYS)}"
    )


# --------------------------------------------------------------------------
# direction 2: the guard must fire when a declared track really is missing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode_id", "dropped"), MODE_TRACK_PAIRS, ids=_pair_ids(MODE_TRACK_PAIRS, "lost")
)
def test_a_declared_track_missing_from_a_success_renders_the_loud_error(
    mode_id, dropped, wired, monkeypatch
):
    """PLAN §5.3: never render the empty value and let it look like nothing happened.

    The result handed to the handler is a *real* ``GenerationResult`` from a real
    run, with exactly one declared track knocked out — the falsy-artifact shape
    that hid for a week in the predecessor app.
    """
    spec = modes.get_mode(mode_id)
    handlers.on_generate(mode_id, *values_for(mode_id))
    broken = replace(wired.result, **{f"{dropped}_path": None})
    _returns(monkeypatch, broken)

    video, audio, status = handlers.on_generate(mode_id, *values_for(mode_id))

    assert (video, audio) == (None, None), (
        f"mode {mode_id!r} lost its {dropped} track and the handler still populated a player. "
        f"Nothing from a broken run may reach the screen."
    )
    assert status == render.empty_output_bug(spec, [dropped]), (
        f"mode {mode_id!r} lost its {dropped} track but the handler did not report it:\n\n{status}"
    )


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_losing_every_declared_track_names_every_one_of_them(mode_id, wired, monkeypatch):
    """The message must say what is missing, not merely that something is."""
    spec = modes.get_mode(mode_id)
    handlers.on_generate(mode_id, *values_for(mode_id))
    broken = replace(wired.result, **{f"{track}_path": None for track in spec.exports})
    _returns(monkeypatch, broken)

    _video, _audio, status = handlers.on_generate(mode_id, *values_for(mode_id))

    assert EMPTY_OUTPUT_HEADLINE in status
    for track in spec.exports:
        assert track in status, (
            f"mode {mode_id!r} produced no {track} file and the report does not mention it:"
            f"\n\n{status}"
        )


@pytest.mark.parametrize(
    ("mode_id", "dropped"), MODE_TRACK_PAIRS, ids=_pair_ids(MODE_TRACK_PAIRS, "blank")
)
def test_an_empty_string_path_is_as_loud_as_a_missing_one(mode_id, dropped, wired, monkeypatch):
    """``""`` is the exact value the 2.3 outage pushed through four layers."""
    spec = modes.get_mode(mode_id)
    handlers.on_generate(mode_id, *values_for(mode_id))
    _returns(monkeypatch, replace(wired.result, **{f"{dropped}_path": ""}))

    video, audio, status = handlers.on_generate(mode_id, *values_for(mode_id))

    assert (video, audio) == (None, None)
    assert status == render.empty_output_bug(spec, [dropped])


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_a_track_outside_exports_being_absent_is_not_an_error(mode_id, wired, monkeypatch):
    """The guard is keyed on ``exports``, not on "both attributes are set".

    An audio-only mode returning no video is the mode working as designed. A
    guard that fired on it would make eight modes permanently unusable.
    """
    spec = modes.get_mode(mode_id)
    handlers.on_generate(mode_id, *values_for(mode_id))
    undeclared = {f"{track}_path": None for track in TRACKS if track not in spec.exports}
    if not undeclared:
        pytest.skip(f"{mode_id!r} exports both tracks; nothing to leave unset")
    _returns(monkeypatch, replace(wired.result, **undeclared))

    _video, _audio, status = handlers.on_generate(mode_id, *values_for(mode_id))

    assert EMPTY_OUTPUT_HEADLINE not in status, (
        f"mode {mode_id!r} does not export {sorted(k[:-5] for k in undeclared)}, and their "
        f"absence was reported as a bug:\n\n{status}"
    )


# --------------------------------------------------------------------------
# no path through the seam is silent
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_a_typed_failure_reaches_the_user_rather_than_an_empty_form(mode_id, monkeypatch):
    """PLAN §5.3: the failure the predecessor rendered as a silent form reset."""
    from ltx25_studio.errors import EmptyOutputError

    def exploding(**params):
        raise EmptyOutputError(mode=mode_id, expected_path="/nowhere")

    monkeypatch.setattr(generate_module, "generate", exploding)

    video, audio, status = handlers.on_generate(mode_id, *values_for(mode_id))

    assert (video, audio) == (None, None)
    assert status.strip(), "a failed generation returned an empty status string"
    assert "Generation failed" in status


# --------------------------------------------------------------------------
# the audio a user uploads has to reach the model
#
# There is no `audio=` argument on any LTX2 pipeline: audio enters as
# pre-encoded `audio_latents`. The UI wrote `request.audio`, `modes._with_audio`
# read `request.audio_latents`, and NOTHING JOINED THEM -- both ends built, the
# encoder exported, its importer count zero. So the upload was discarded, the
# model generated fresh audio, the run succeeded, the artifact was valid, and
# 1926 tests passed. Only the user's own ears would have told them.
# --------------------------------------------------------------------------

AUDIO_INPUT_MODES = [m for m in MODE_IDS if "audio" in modes.get_mode(m).inputs]


@pytest.mark.parametrize("mode_id", AUDIO_INPUT_MODES)
def test_the_uploaded_audio_reaches_the_pipeline_as_latents(
    mode_id, wired, audio_input_ready, tmp_path
):
    """The whole point: the model must be handed what the user supplied."""
    handlers.on_generate(mode_id, *values_for(mode_id))

    pipe = wired.pipes[-1]
    sent = pipe.calls[-1]
    assert sent.get("audio_latents") is not None, (
        f"{mode_id} was given an audio file and the pipeline received "
        f"audio_latents=None, so the upload was discarded and the model "
        f"generated audio the user did not ask for: {sorted(sent)}"
    )


@pytest.mark.parametrize("mode_id", AUDIO_INPUT_MODES)
def test_the_upload_is_encoded_through_the_audio_vae(mode_id, wired, audio_input_ready):
    """Not merely present -- produced by the encoder, from the user's waveform.

    Asserting only that ``audio_latents`` is set would pass if something handed
    the pipeline a constant. This follows the value back to the waveform.
    """
    handlers.on_generate(mode_id, *values_for(mode_id))

    audio_vae = wired.pipes[-1].audio_vae
    assert audio_vae.encoded, f"{mode_id} never called audio_vae.encode"
    assert audio_vae.mels, f"{mode_id} never ran the mel front-end"
    assert audio_vae.mels[-1].shape == STUB_WAVEFORM.shape, (
        f"{mode_id} encoded something other than the loaded waveform: "
        f"{audio_vae.mels[-1].shape} vs {STUB_WAVEFORM.shape}"
    )


@pytest.mark.parametrize(
    "mode_id", [m for m in MODE_IDS if "audio" not in modes.get_mode(m).inputs]
)
def test_a_mode_that_takes_no_audio_encodes_nothing(mode_id, wired, audio_input_ready):
    """The guard must not fire where there is nothing to encode."""
    handlers.on_generate(mode_id, *values_for(mode_id))

    assert not wired.pipes[-1].audio_vae.encoded, (
        f"{mode_id} takes no audio input and still ran an encode"
    )


def test_an_audio_input_with_no_audio_vae_is_refused_loudly(monkeypatch, tmp_path):
    """Silently dropping it is the defect; generating anyway is the worst form."""
    from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM

    from ltx25_studio import audio as audio_module
    from ltx25_studio.errors import GenerationError

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )
    pipe = FakePipe()
    del pipe.audio_vae

    with pytest.raises(GenerationError) as excinfo:
        generate_module.generate(
            mode="audio_to_audio",
            seed=7,
            audio="an-upload.wav",
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path,
            retention_bytes=10**9,
        )
    assert "audio_vae" in str(excinfo.value), excinfo.value
    assert not pipe.calls, "the pipeline ran anyway, so the user got audio they did not ask for"


def test_a_mono_upload_is_reported_to_the_user_not_only_the_log(monkeypatch, tmp_path):
    """A note that reaches the Space log alone has not reached the user.

    Most wav files are mono, so this is the ordinary case rather than an edge:
    the audio is widened so it can be encoded at all, and the person who chose
    mono is told their file was altered.
    """
    from conftest import STUB_SAMPLING_RATE, waveform_of

    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (waveform_of(1), STUB_SAMPLING_RATE)
    )
    pipe = FakePipe()

    result = generate_module.generate(
        mode="audio_to_audio",
        seed=7,
        audio="a-mono-recording.wav",
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    assert pipe.audio_vae.mels[-1].shape[0] == 2, "the VAE was handed mono and would have raised"
    assert any("mono" in note for note in result.warnings), (
        f"the upload was altered and the result says nothing about it: {result.warnings}"
    )
