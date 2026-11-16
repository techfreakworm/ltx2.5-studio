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

import tempfile
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest
from conftest import STUB_WAVEFORM, audio_file, inputs_for
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


@lru_cache(maxsize=1)
def conditioning_video() -> str:
    """One real, decodable mp4 — built once, on first use by a video mode.

    **This used to be the string ``"conditioning-input.video"``, and that is how
    the video-input defect reached the Space.** Nothing on the path had to
    decode it, so a payload that could never work in production satisfied every
    assertion in this file, and both video modes shipped broken past a green
    gate. ``conditioning.decode_video_frames`` now decodes a video condition
    before the pipeline sees it, which makes a real file the only honest
    fixture — and restores this module's own opening rule, that nothing between
    the layers is stubbed. It was true of every layer and one payload.

    The image and audio placeholders stay strings deliberately: nothing decodes
    those offline, so a real file there would assert nothing that this string
    does not. (For audio that is itself a gap — the decode sits behind the real
    pipe, so it is never exercised here — but it is a separate one.)

    ``av`` is imported inside the function rather than skipped at module scope
    on purpose. A module-level ``importorskip`` would silently drop this entire
    seam suite if the media stack went missing, and this is the file that exists
    because a whole seam went untested once already.
    """
    import av
    import numpy as np

    directory = Path(tempfile.mkdtemp(prefix="ltx25-seam-"))
    path = directory / "conditioning-input.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=24)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24")
        frame.pts, frame.time_base = 0, Fraction(1, 24)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def values_for(mode_id: str, **overrides) -> tuple:
    """One positional value per ``FIELD_KEYS`` entry, as Gradio would send them."""
    values = {spec.key: spec.default for spec in FIELDS}
    values["seed"] = 7
    # Hand the mode what it REQUIRES -- a prompt, a PIL image, a real mp4, a real
    # wav -- because generate() refuses a run missing one, and opens the audio,
    # before it reserves a GPU.
    values.update(inputs_for(mode_id))
    if "video" in modes.get_mode(mode_id).inputs:
        values["video"] = conditioning_video()
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


#: What ``on_generate`` sends ``generate()`` besides the request fields.
#: ``progress`` is a SEAM, like ``pipeline_factory`` and ``encoder`` -- a channel
#: the handler is handed and passes on, not something the user typed. Named here
#: so the equality below stays an equality: a genuinely new field still fails.
HANDLER_SEAMS = {"progress"}


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_the_handler_marshals_every_declared_field_into_the_request(mode_id, wired):
    """The ``strict=True`` zip is only a guarantee if something drives it."""
    handlers.on_generate(mode_id, *values_for(mode_id))
    params = wired.params[-1]

    assert params["mode"] == mode_id
    assert set(params) == {"mode", *HANDLER_SEAMS, *FIELD_KEYS}, (
        f"the handler forwarded {sorted(set(params) - {'mode'})} but FIELD_KEYS declares "
        f"{sorted(FIELD_KEYS)} plus the seams {sorted(HANDLER_SEAMS)}"
    )
    assert not HANDLER_SEAMS & set(FIELD_KEYS), (
        f"a seam has become a declared field: {sorted(HANDLER_SEAMS & set(FIELD_KEYS))}. It "
        f"would take a positional slot on the endpoint and shift every value after it."
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
    # Batched by the time it reaches the transform: (batch, channels, samples).
    assert audio_vae.mels[-1].shape == (1, *STUB_WAVEFORM.shape), (
        f"{mode_id} encoded something other than the loaded waveform: "
        f"{audio_vae.mels[-1].shape} vs {(1, *STUB_WAVEFORM.shape)}"
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
            audio=audio_file(),
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
    from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM, waveform_of

    from ltx25_studio import audio as audio_module

    # Same length as the stereo stub: this test is about the CHANNEL note, and
    # a short mono file would now trip the length policy's refusal instead.
    monkeypatch.setattr(
        audio_module,
        "load_waveform",
        lambda source: (waveform_of(1, samples=STUB_WAVEFORM.shape[-1]), STUB_SAMPLING_RATE),
    )
    pipe = FakePipe()

    result = generate_module.generate(
        mode="audio_to_audio",
        seed=7,
        audio=audio_file(),
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    # -2 is the channel axis once the waveform is batched; reading axis 0 here
    # would read the batch and pass whatever the channel count actually was.
    assert pipe.audio_vae.mels[-1].shape[-2] == 2, "the VAE was handed mono and would have raised"
    assert any("mono" in note for note in result.warnings), (
        f"the upload was altered and the result says nothing about it: {result.warnings}"
    )


# --------------------------------------------------------------------------
# audio in, audio out -- and WHICH audio (the mode-11 defect)
#
# `audio_to_video` took an audio upload and returned an mp4 with ZERO audio
# streams. Confirmed twice on the live Space: the operator hit it, and a
# controlled run reproduced it as `audio: False` in the response tuple. The
# generated track was discarded by `outputs._write_video`'s `include_audio`
# branch and reported through `dropped_audio_warning` -- which advised the user
# to "use the matching audio+video mode to keep it" when mode 11's
# `audio_counterpart` is None and no such mode exists.
#
# The registry declares `exports=("audio", "video")` for it, so a file comes
# back with sound. WHICH sound has been answered twice:
#
#   * `4683a57` muxed the model's OWN generated track, on six reasons recorded
#     at the mode-11 spec in `modes.py`;
#   * a controlled discriminator on the live Space then measured that track.
#     Same pipeline, same 512x960x121 geometry, same seed, one variable --
#     whether audio was supplied. Audio in: -88.3 dBFS, digital silence. No
#     audio in: -24.2 dBFS, real content. **Supplying audio suppresses the
#     model's own audio generation**, 64 dB of separation, so `4683a57` was
#     muxing an empty stream -- which is worse than no stream, because it looks
#     like a track until you play it.
#
# ✅ RULED, for the video-exporting case only: mux the user's SUPPLIED audio.
# The audio-only export modes are untouched -- their artifact IS the model's
# track. These tests pin the three halves that can regress silently: that a
# file comes back with sound, that it is the RIGHT sound for the mode, and that
# the user is told which one it is and why.
# --------------------------------------------------------------------------

#: The modes whose container carries the user's own upload back.
MUXES_UPLOAD = [m for m in AUDIO_INPUT_MODES if generate_module._muxes_supplied_audio(
    modes.get_mode(m).exports
)]
#: The audio-input modes that still hand back what the model rendered.
KEEPS_MODEL_TRACK = [m for m in AUDIO_INPUT_MODES if m not in MUXES_UPLOAD]


@pytest.mark.parametrize("mode_id", AUDIO_INPUT_MODES)
def test_a_mode_that_conditions_on_audio_also_exports_audio(mode_id):
    """The registry-level regression, stated in the terms the defect had.

    Derived coverage exists -- ``audio_to_video`` is now in
    ``test_encoder_contract.JOINT_MODES``, so the real PyAV encoder is checked
    against a real container for it. That coverage is silent about WHY, and a
    future edit narrowing ``exports`` back to ``("video",)`` would simply drop
    out of the derived set with every test still green. This one fails.
    """
    exports = modes.get_mode(mode_id).exports
    assert "audio" in exports, (
        f"{mode_id!r} conditions on an audio upload and declares exports={list(exports)}, so "
        f"`outputs._write_video` will drop the generated track and hand the user a file with "
        f"no audio stream -- which is exactly the defect confirmed twice on the Space"
    )


def _uploaded(mode_id, tmp_path, monkeypatch, *, encoder=one_muxed_file_encoder, **overrides):
    """One generation for ``mode_id`` with a real audio upload behind it."""
    from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM

    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )
    return generate_module.generate(
        mode=mode_id,
        seed=7,
        audio=audio_file(),
        pipeline_factory=factory_returning(FakePipe()),
        encoder=encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
        **overrides,
    )


@pytest.mark.parametrize("mode_id", KEEPS_MODEL_TRACK)
def test_an_audio_export_mode_still_says_the_track_is_the_models(mode_id, tmp_path, monkeypatch):
    """Muxing the right track silently is still a surprise.

    The audio-only export modes are deliberately outside the mode-11 ruling:
    their artifact IS the model's track, and handing back the upload would
    answer a different question than the one the mode asks. So the original
    sentence survives here unchanged -- a user who uploads a song still needs
    telling that the file does not contain it.
    """
    result = _uploaded(mode_id, tmp_path, monkeypatch)

    assert any("model's OWN track" in note for note in result.warnings), (
        f"{mode_id!r} returned the model's audio and never said so, so a user who uploaded "
        f"audio has no way to learn the file does not contain it: {result.warnings}"
    )


@pytest.mark.parametrize("mode_id", MUXES_UPLOAD)
def test_a_video_exporting_mode_hands_back_the_users_own_audio(mode_id, tmp_path, monkeypatch):
    """The flip, at the seam a user actually meets.

    ``4683a57`` muxed the model's track here. The live discriminator then
    measured that track at -88.3 dBFS against -24.2 dBFS for the same pipeline
    with no audio supplied: supplying audio suppresses the model's own audio
    generation, so what shipped was a well-formed AAC stream containing silence.
    The upload is what the user has, what he expects back, and the only one of
    the two that is not empty.
    """
    result = _uploaded(mode_id, tmp_path, monkeypatch)

    assert any("audio YOU supplied" in note for note in result.warnings), (
        f"{mode_id!r} muxes the user's own audio and never says so, which is the same "
        f"silence the mode-11 defect was: {result.warnings}"
    )
    assert not any("model's OWN track" in note for note in result.warnings), (
        f"{mode_id!r} claims to be handing back the model's track, which is both wrong and "
        f"(per the measurement) a claim about digital silence: {result.warnings}"
    )


@pytest.mark.parametrize("mode_id", MUXES_UPLOAD)
def test_the_note_says_WHY_the_upload_is_the_one_that_comes_back(
    mode_id, tmp_path, monkeypatch
):
    """Honest and useful, where the previous behaviour was honest and useless.

    "here is your audio back" invites the reading that we simply preferred it.
    The reason is the whole finding -- the model's generation is *suppressed* by
    the audio input -- and it is what stops the next person re-deriving
    ``4683a57``'s six reasons and flipping this back.
    """
    result = _uploaded(mode_id, tmp_path, monkeypatch)
    provenance = [note for note in result.warnings if "audio YOU supplied" in note]

    assert provenance, result.warnings
    note = provenance[0]
    assert "SUPPRESSES" in note, f"the note states no reason: {note!r}"
    assert "-88.3 dBFS" in note and "-24.2 dBFS" in note, (
        f"the measurement that overturned the previous ruling is not quoted: {note!r}"
    )


@pytest.mark.parametrize("mode_id", MUXES_UPLOAD)
def test_the_encoder_is_handed_the_supplied_waveform_and_its_own_rate(
    mode_id, tmp_path, monkeypatch
):
    """The note is not the behaviour. This is the behaviour.

    A note claiming the file carries the user's audio, over a writer that muxed
    the model's, is the ninth inert lever with a sentence attached. So the seam
    is asserted directly: the encoder receives an ``outputs.SuppliedAudio``, and
    the rate on it is the UPLOAD's, never ``vocoder.config.output_sampling_rate``
    -- 16 kHz against 48 kHz on this stack, and muxing one at the other
    pitch-shifts the track with no error anywhere.
    """
    from conftest import STUB_SAMPLING_RATE

    from ltx25_studio import outputs

    seen: dict = {}

    def spy_encoder(output, **kwargs):
        seen.update(kwargs)
        return one_muxed_file_encoder(output, **kwargs)

    _uploaded(mode_id, tmp_path, monkeypatch, encoder=spy_encoder)

    supplied = seen.get("supplied_audio")
    assert isinstance(supplied, outputs.SuppliedAudio), (
        f"{mode_id!r} told the user it muxed their audio and handed the writer "
        f"supplied_audio={supplied!r}. The writer would have muxed the model's silent track."
    )
    assert supplied.sampling_rate == STUB_SAMPLING_RATE, (
        f"the muxed track was handed the wrong rate: {supplied.sampling_rate} rather than the "
        f"upload's {STUB_SAMPLING_RATE}. The vocoder's rate here would pitch-shift it silently."
    )


@pytest.mark.parametrize("mode_id", KEEPS_MODEL_TRACK)
def test_an_audio_export_mode_is_not_handed_a_substitute_track(mode_id, tmp_path, monkeypatch):
    """The ruling is scoped, and the scope is enforced where it acts.

    ``outputs.write_output`` REFUSES a ``supplied_audio`` for a mode that writes
    no video container, so a generalisation here would not merely widen the
    ruling -- it would break every audio-only mode at the last step of an
    already-paid-for run.
    """
    seen: dict = {}

    def spy_encoder(output, **kwargs):
        seen.update(kwargs)
        return one_muxed_file_encoder(output, **kwargs)

    _uploaded(mode_id, tmp_path, monkeypatch, encoder=spy_encoder)

    assert "supplied_audio" not in seen, (
        f"{mode_id!r} is an audio-export mode and was handed a substitute track: "
        f"{seen.get('supplied_audio')!r}"
    )


def test_the_programmatic_latents_route_is_told_which_track_too(tmp_path):
    """The other way audio enters. Silence here would be the same defect.

    ``_prepared_audio`` returns early down the supplied-``audio_latents`` branch
    without ever touching ``request.audio``, so a note appended alongside the
    file-encode path only would miss this caller entirely.

    And this route CANNOT hand the upload back: the waveform those latents came
    from never entered this process. "Could not" and "chose not to" must not
    read the same, so the note names the one thing the caller can change.
    """
    # Shapeless on purpose: that route's own sync check reports "could not
    # check" and returns, which is the shortest path through the branch and the
    # one a note appended further down would never reach.
    seen: dict = {}

    def spy_encoder(output, **kwargs):
        seen.update(kwargs)
        return one_muxed_file_encoder(output, **kwargs)

    result = generate_module.generate(
        mode="audio_to_video",
        seed=7,
        audio_latents=object(),
        pipeline_factory=factory_returning(FakePipe()),
        encoder=spy_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    assert any("model's OWN track" in note for note in result.warnings), (
        f"audio was supplied as latents and the result never says which track came back: "
        f"{result.warnings}"
    )
    assert any("audio FILE" in note for note in result.warnings), (
        f"the latents route cannot mux the upload and must say what to change instead: "
        f"{result.warnings}"
    )
    assert "supplied_audio" not in seen, (
        f"there is no waveform on this route, so nothing may be handed to the writer as one: "
        f"{seen.get('supplied_audio')!r}"
    )


def test_the_track_note_is_gated_on_exports_not_on_the_mode_id():
    """Tested directly, because no mode currently reaches this branch.

    Every audio-input mode exports audio today, so the gate inside
    ``_supplied_audio_note`` is unexercised end-to-end: deleting it leaves the
    whole suite green. It is not decoration. The note asserts a fact about a
    file the run produced -- "the audio in this file is X" -- and a future
    audio-input mode that exports video only would make that sentence describe a
    track the user never receives.

    Deriving it from ``exports`` rather than from the mode id is what keeps the
    note from outliving the behaviour it describes, and this is the only thing
    that holds the derivation in place.
    """
    for muxed in (True, False):
        assert generate_module._supplied_audio_note(
            "audio_to_video", ("audio", "video"), muxed=muxed
        )
        assert generate_module._supplied_audio_note(
            "some_future_mode", ("video",), muxed=muxed
        ) == (), "a mode that exports no audio would be told its file contains audio"
        assert generate_module._supplied_audio_note("some_future_mode", (), muxed=muxed) == ()
    assert generate_module._supplied_audio_note("audio_to_audio", ("audio",), muxed=False)


def test_which_track_a_mode_gets_is_derived_from_exports_not_from_a_registry_flag():
    """The scope of the mode-11 ruling, pinned as a predicate.

    A boolean on ``ModeSpec`` would be forgotten by whoever adds the next
    audio-input mode, and the failure would be silent: the new mode would ship
    an empty AAC stream exactly as mode 11 did. Derived from ``exports``, it
    cannot be forgotten -- a mode that writes a video container declared to
    carry sound gets the substitution because that is the situation the
    measurement covers.
    """
    muxes = generate_module._muxes_supplied_audio

    assert muxes(("audio", "video")) and muxes(("video", "audio"))
    assert not muxes(("audio",)), (
        "an audio-only export has no container to mux into and is outside the ruling"
    )
    assert not muxes(("video",)), "a video-only mode carries no audio track at all"
    assert not muxes(()) and not muxes(None)
    assert [m for m in MODE_IDS if muxes(modes.get_mode(m).exports)] == [
        "text_to_audio_video",
        "image_to_audio_video",
        "image_text_to_audio_video",
        "audio_to_video",
    ], "the joint modes are the predicate's whole reach"
    # And of those, only a mode that TAKES audio can ever be handed a track to
    # mux: generate() drops an audio upload every other mode does not take
    # (scope_request_to_mode), so a stale upload never reaches 05/06/07.
    assert [
        m
        for m in MODE_IDS
        if muxes(modes.get_mode(m).exports) and modes.accepts(modes.get_mode(m), "audio")
    ] == ["audio_to_video"]


def test_the_writer_refuses_a_substitute_track_for_a_mode_with_no_container(tmp_path):
    """The scope, enforced at the layer that would otherwise drop it silently.

    ``write_output`` is handed ``supplied_audio`` by exactly one caller today.
    If a future one hands it for an audio-only export, ignoring it would return
    the model's track under a note promising the user's -- the same
    honest-and-useless shape this whole change is against.
    """
    from ltx25_studio import outputs
    from ltx25_studio.errors import EncodeError

    with pytest.raises(EncodeError) as caught:
        outputs.write_output(
            FakePipe()(),
            run_dir=tmp_path / "run",
            mode="audio_to_audio",
            exports=("audio",),
            sampling_rate=48000,
            supplied_audio=outputs.SuppliedAudio(waveform=STUB_WAVEFORM, sampling_rate=16000),
        )

    assert "supplied audio" in str(caught.value)


@pytest.mark.parametrize(
    "mode_id",
    [
        m
        for m in MODE_IDS
        if "audio" not in modes.get_mode(m).inputs and "audio" in modes.get_mode(m).exports
    ],
)
def test_a_mode_given_no_audio_is_told_nothing_about_whose_audio_it_is(mode_id, tmp_path):
    """The note must not fire where the user supplied nothing to confuse it with.

    ``text_to_audio_video`` and its siblings export audio without taking any, so
    "this is the model's track, not your upload" would be answering a question
    nobody asked -- and warnings that fire when nothing is wrong are how a
    warnings channel stops being read.
    """
    fields = inputs_for(mode_id)
    fields.pop("audio", None)
    result = generate_module.generate(
        mode=mode_id,
        seed=7,
        **fields,
        pipeline_factory=factory_returning(FakePipe()),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    assert not any("model's OWN track" in note for note in result.warnings), (
        f"{mode_id!r} was given no audio input and still explained whose audio it "
        f"returned: {result.warnings}"
    )


def test_the_seed_reaches_the_pipeline_not_only_the_sidecar(monkeypatch, tmp_path):
    """The seed used to be recorded and never applied.

    `_resolve_seed` ran, the number went into the encoder kwargs as metadata,
    and there was no generator anywhere in the package -- so every run was
    unconditionally random and every same-seed comparison was two independent
    runs. H10b, the before/after pixel diffs, H12b and H4b all rested on it.
    """
    import sys
    import types

    seen = {}

    class Generator:
        def __init__(self, device=None):
            seen["device"] = device

        def manual_seed(self, value):
            seen["seed"] = value
            return self

    torch = types.ModuleType("torch")
    torch.Generator = Generator
    monkeypatch.setitem(sys.modules, "torch", torch)

    pipe = FakePipe()
    generate_module.generate(
        mode="text_to_video",
        prompt="a lighthouse in fog",
        seed=1234,
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    assert seen.get("seed") == 1234, (
        f"the pipeline was never seeded: {seen}. The seed reaching only the sidecar is "
        f"why 'same seed' comparisons compared two random runs."
    )
    assert "generator" in pipe.calls[-1], (
        f"no generator reached the pipeline: {sorted(pipe.calls[-1])}"
    )
