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
from conftest import PACKAGE_ROOT, code_string_constants

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
    values = {
        "audio_sampling_rate": 44100,
        "audio_hop_length": 512,
        "audio_mel_bins": 128,
        "audio_vae_mel_compression_ratio": 4,
        "audio_vae_temporal_compression_ratio": 8,
        "audio_latent_channels": 32,
    }
    values.update(overrides)
    return SimpleNamespace(config=SimpleNamespace(**values))


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
    """"shape errors, or worse, **silent misinterpretation**" — so we assert at
    our own boundary rather than hoping upstream notices."""
    with pytest.raises(LTXStudioError):
        audio.validate_audio_latents_shape(shape)


def test_the_shape_error_names_the_shape_it_got():
    with pytest.raises(LTXStudioError) as excinfo:
        audio.validate_audio_latents_shape((1, 32, 64))
    assert "4" in str(excinfo.value)
