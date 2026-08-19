"""Spec for ``ltx25_studio/audio.py`` — PLAN §5.4, §5.7 rows 5 and 11.

§5.4 is the whole rule: **every mel parameter is read from ``audio_vae.config``
at runtime.** Not one of them is written down here. The pipeline itself works
this way — its constructor reads ``self.vae.spatial_compression_ratio``,
``self.audio_vae.mel_compression_ratio`` and friends from the loaded components —
and a hardcoded 16000 or 80 that happens to be right today is a wrong number
waiting for a checkpoint change.

Tested against a fake config object. No torch, no audio file, no weights.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
from conftest import PACKAGE_ROOT, FakeAudioVae, code_string_constants, waveform_of

from ltx25_studio import audio
from ltx25_studio.errors import LTXStudioError

#: The parameter names PLAN §5.4 names explicitly.
CONFIG_FIELDS = (
    "audio_sampling_rate",
    "audio_hop_length",
    "audio_mel_bins",
    "audio_vae_mel_compression_ratio",
    "audio_vae_temporal_compression_ratio",
    "audio_latent_channels",
)


def fake_audio_vae(**overrides):
    """A stand-in shaped like the REAL checkpoint, keyed by our semantic names.

    Overrides use the §5.4 names; the fake puts each value **where the actual
    audio VAE keeps it** -- four on ``.config`` under different names, two as
    instance attributes absent from ``config.json`` entirely.

    Placement is derived from ``audio.PARAM_SOURCES`` so this fake cannot drift
    from the reader. That makes it useless for catching a wrong *mapping*, which
    is precisely the bug that got here -- so
    ``test_the_mapping_matches_the_real_checkpoint`` pins the mapping against
    the recorded config instead. The fake checks semantics; that test checks
    reality.
    """
    values = {
        "audio_sampling_rate": 44100,
        "audio_hop_length": 512,
        "audio_mel_bins": 128,
        "audio_vae_mel_compression_ratio": 4,
        "audio_vae_temporal_compression_ratio": 8,
        "audio_latent_channels": 32,
    }
    values.update(overrides)
    config, attrs = {}, {}
    for name, value in values.items():
        where, field = audio.PARAM_SOURCES[name]
        (config if where == "config" else attrs)[field] = value
    return SimpleNamespace(config=SimpleNamespace(**config), **attrs)


# --------------------------------------------------------------------------
# every parameter comes from the config
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", CONFIG_FIELDS)
def test_every_documented_parameter_is_read_from_the_config(field):
    params = audio.audio_params_from(fake_audio_vae())
    assert hasattr(params, field) or field in params, f"{field} is not surfaced"


def test_the_values_are_the_config_values():
    params = audio.audio_params_from(fake_audio_vae(audio_sampling_rate=48000, audio_mel_bins=80))
    values = params if isinstance(params, dict) else vars(params)
    assert values["audio_sampling_rate"] == 48000
    assert values["audio_mel_bins"] == 80


def test_an_unusual_config_is_honoured_rather_than_overridden():
    """The decisive one: a checkpoint with a 22050 Hz audio VAE must produce
    22050, not the number someone assumed while writing the module."""
    params = audio.audio_params_from(
        fake_audio_vae(audio_sampling_rate=22050, audio_hop_length=256, audio_latent_channels=16)
    )
    values = params if isinstance(params, dict) else vars(params)
    assert values["audio_sampling_rate"] == 22050
    assert values["audio_hop_length"] == 256
    assert values["audio_latent_channels"] == 16


def test_a_missing_config_field_fails_loudly_rather_than_defaulting():
    """A silent fallback to a plausible default is how the audio track ends up
    subtly wrong for a whole release."""
    incomplete = SimpleNamespace(config=SimpleNamespace(audio_sampling_rate=44100))
    with pytest.raises(LTXStudioError):
        audio.audio_params_from(incomplete)


def test_the_error_names_the_missing_field():
    incomplete = SimpleNamespace(config=SimpleNamespace(audio_sampling_rate=44100))
    with pytest.raises(LTXStudioError) as excinfo:
        audio.audio_params_from(incomplete)
    assert "audio_" in str(excinfo.value)


# --------------------------------------------------------------------------
# ...and none of them is written down (PLAN §5.4)
# --------------------------------------------------------------------------

#: Numbers that would mean someone assumed an audio format.
SUSPICIOUS_LITERALS = {8000, 16000, 22050, 24000, 32000, 44100, 48000, 1024}

AUDIO_MODULE = PACKAGE_ROOT / "audio.py"


def test_no_sample_rate_is_hardcoded():
    """Comments and docstrings may cite numbers; code may not use them."""
    tree = ast.parse(AUDIO_MODULE.read_text())
    docstring_ids = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            docstring_ids.add(id(body[0].value))
    found = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and id(node) not in docstring_ids
        and node.value in SUSPICIOUS_LITERALS
    }
    assert not found, (
        f"audio.py hardcodes {sorted(found)}. Every mel parameter comes from "
        "audio_vae.config at runtime (PLAN §5.4)."
    )


def test_audio_module_does_not_name_config_keys_it_should_be_reading():
    """A string literal like ``"audio_sampling_rate"`` is fine — that IS how you
    read it from the config. This guards the opposite: a *value* smuggled in as
    a string."""
    for _, value in code_string_constants(AUDIO_MODULE):
        assert value not in {str(n) for n in SUSPICIOUS_LITERALS}, value


# --------------------------------------------------------------------------
# shape assertions at our boundary (PLAN §5.7 row 5)
# --------------------------------------------------------------------------


def test_four_dimensional_audio_latents_are_accepted():
    """``audio_latents`` is UNPACKED 4D ``[B, C, L, M]``."""
    assert audio.validate_audio_latents_shape((1, 32, 64, 8)) == (1, 32, 64, 8)


@pytest.mark.parametrize("shape", [(32, 64, 8), (1, 32, 64, 8, 2), (64,), ()])
def test_wrongly_shaped_audio_latents_raise(shape):
    """ "shape errors, or worse, **silent misinterpretation**" — so we assert at
    our own boundary rather than hoping upstream notices."""
    with pytest.raises(LTXStudioError):
        audio.validate_audio_latents_shape(shape)


def test_the_shape_error_names_the_shape_it_got():
    with pytest.raises(LTXStudioError) as excinfo:
        audio.validate_audio_latents_shape((1, 32, 64))
    assert "4" in str(excinfo.value)


# --------------------------------------------------------------------------
# the channel count the VAE actually requires
#
# The audio VAE is stereo and does not adapt. Handed one channel it reaches
# `conv_in` and raises `Given groups=1, weight of size [128, 2, 3, 3], expected
# input[1, 1, 4, 480002] to have 2 channels, but got 1` -- four frames inside
# someone else's conv layer, naming a weight shape rather than anything the user
# did. Most people's wav files are mono, so this is the single most likely way
# the audio-input feature meets a real user.
# --------------------------------------------------------------------------


def test_a_mono_upload_is_widened_rather_than_rejected():
    """Duplicating one channel adds no information and loses none."""
    widened, note = audio.match_channel_count(waveform_of(1), channels=2)

    assert widened.shape[0] == 2, f"mono was not widened: {widened.shape}"
    assert note, "the audio was altered and the user was not told"


def test_the_mono_upmix_is_reported_in_words_a_user_can_act_on():
    _widened, note = audio.match_channel_count(waveform_of(1), channels=2)

    assert "mono" in note and "duplicated" in note, note
    assert "not true stereo" in note, f"the note does not say what the user actually got: {note}"


def test_stereo_passes_through_untouched_and_unremarked():
    """A note on every run is a note nobody reads."""
    source = waveform_of(2)

    passed, note = audio.match_channel_count(source, channels=2)

    assert passed is source, "a conforming waveform was rebuilt for no reason"
    assert note is None, f"nothing changed and a warning was raised anyway: {note}"


def test_more_channels_than_the_vae_takes_is_refused_not_downmixed():
    """That direction loses information; PLAN 5.4 forbids doing it silently."""
    with pytest.raises(LTXStudioError) as excinfo:
        audio.match_channel_count(waveform_of(6), channels=2)

    message = str(excinfo.value)
    assert "6 channels" in message, f"the refusal does not name what was supplied: {message}"
    assert "exactly 2" in message, f"the refusal does not name what is required: {message}"


def test_the_required_count_is_read_from_the_vae_not_hardcoded():
    """A checkpoint with a different layout is handled by reading, not editing."""
    widened, _note = audio.match_channel_count(waveform_of(1), channels=4)

    assert widened.shape[0] == 4, (
        f"the channel count was not taken from the caller: {widened.shape}"
    )


def test_a_mono_upload_reaches_the_vae_as_stereo():
    """The whole point: the conv layer must never see one channel.

    `match_channel_count` is called inside `encode_waveform_to_latents` rather
    than at one call site, so every caller reaches the conv layer conforming.
    """
    vae = FakeAudioVae()
    notes: list[str] = []

    audio.encode_waveform_to_latents(
        waveform_of(1), audio_vae=vae, sampling_rate=vae.config.sample_rate, notes=notes
    )

    assert vae.mels[-1].shape[0] == 2, (
        f"the audio VAE was handed {vae.mels[-1].shape[0]} channel(s); it takes 2 and "
        f"raises a raw conv2d shape error rather than anything actionable"
    )
    assert notes, "the audio was altered and nothing was recorded for the user"


# --------------------------------------------------------------------------
# where the six parameters really live
# --------------------------------------------------------------------------

#: The exact key set of ``audio_vae/config.json`` at revision 871165de…,
#: recorded so the mapping can be checked without the weights on disk.
#: *Read off the cached checkpoint, not inferred.*
REAL_AUDIO_VAE_CONFIG_KEYS = frozenset(
    {
        "_class_name",
        "_diffusers_version",
        "attn_resolutions",
        "base_channels",
        "causality_axis",
        "ch_mult",
        "double_z",
        "dropout",
        "in_channels",
        "is_causal",
        "latent_channels",
        "mel_bins",
        "mel_hop_length",
        "mid_block_add_attention",
        "norm_type",
        "num_res_blocks",
        "output_channels",
        "resolution",
        "sample_rate",
    }
)


@pytest.mark.parametrize(
    "name", [n for n, (where, _f) in audio.PARAM_SOURCES.items() if where == "config"]
)
def test_the_mapping_matches_the_real_checkpoint(name):
    """The check the fake cannot make, because the fake is built from the mapping.

    Every config-sourced parameter must name a key the real ``config.json``
    actually has. All six used to be read off ``.config`` by our own semantic
    names, none of which exist there -- so the reader raised on real weights
    while every offline test passed against a fake that shared the mistake.
    """
    _where, field = audio.PARAM_SOURCES[name]

    assert field in REAL_AUDIO_VAE_CONFIG_KEYS, (
        f"{name!r} is read from audio_vae.config[{field!r}], and the real checkpoint has no "
        f"such key. It would raise on real weights and pass every offline test."
    )


def test_our_own_names_are_not_checkpoint_keys():
    """The specific mistake, pinned so it cannot come back as a 'simplification'."""
    overlap = REAL_AUDIO_VAE_CONFIG_KEYS & set(audio.PARAM_SOURCES)

    assert not overlap, (
        f"{sorted(overlap)} are both our semantic names and real config keys; if that is now "
        f"true the mapping should say so explicitly rather than coincide"
    )


def test_the_input_rate_is_not_the_output_rate():
    """The audio VAE takes 16000; the vocoder emits 48000. Not symmetric.

    Assuming symmetry would resample every upload to the wrong rate -- audible
    as a pitch shift, reported as nothing.
    """
    vae = FakeAudioVae()

    assert audio.audio_params_from(vae).audio_sampling_rate == 16000
    assert (
        audio.export_sampling_rate_from(
            SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000))
        )
        == 48000
    )


def test_a_file_at_another_rate_is_resampled_to_the_vaes(monkeypatch):
    """The rate that wins is the VAE's, and the conversion really happens."""
    seen = {}

    def fake_resample(waveform, *, from_rate, to_rate):
        seen.update(from_rate=from_rate, to_rate=to_rate)
        return waveform

    monkeypatch.setattr(audio, "_resampled", fake_resample)
    vae = FakeAudioVae()

    audio.encode_waveform_to_latents(waveform_of(2), audio_vae=vae, sampling_rate=44100)

    assert seen == {"from_rate": 44100, "to_rate": 16000}, (
        f"the upload was not resampled to the VAE's own rate: {seen}"
    )
