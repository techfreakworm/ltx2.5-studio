"""Spec for the runtime configuration levers.

**Every lever here must be attached to something.** This file's own history is
the reason: a precision variable once existed, was wired to nothing, and made
the startup report state a footprint nobody was loading (PLAN §2.6). Two more
were removed the same way.

The quantization lever below is the counter-example — it is real, it changes
output, and it exists because bf16 does not fit ZeroGPU's disk.
"""

from __future__ import annotations

import inspect

import pytest

from ltx25_studio import config

# --------------------------------------------------------------------------
# the quantization lever
#
# A real lever, unlike the ones this file records as removed: it changes output
# quality, which is why it is a switch rather than folded into the recipe. An
# int8-vs-bf16 pair at one seed answers "what does int8 cost?" only if the two
# can be run independently.
# --------------------------------------------------------------------------


def test_the_default_is_bf16_the_baseline_everything_was_measured_against():
    assert config.load_config().quantization == "bf16"


def test_int8_is_selectable(monkeypatch):
    monkeypatch.setenv(config.ENV_QUANTIZATION, "int8")

    assert config.load_config().quantization == "int8"


def test_an_unrecognised_precision_is_refused_rather_than_defaulted(monkeypatch):
    """Silently falling back to bf16 would deploy the configuration that does
    not fit, on a platform where that is an eviction rather than an error."""
    monkeypatch.setenv(config.ENV_QUANTIZATION, "fp8")

    with pytest.raises(config.ConfigError) as excinfo:
        config.load_config()
    assert "fp8" in str(excinfo.value)


def test_only_the_transformer_is_quantised():
    """Widening costs attributability: a quality change could not then be
    ascribed to the denoiser rather than to prompt encoding."""
    from ltx25_studio import pipelines

    source = inspect.getsource(pipelines._quantization_config)
    assert "load_in_8bit=True" in source
    assert "text_encoder" not in source.split('"""')[2], (
        "the text encoder is named in the quantisation call; B was chosen "
        "deliberately over A and widening needs its own decision"
    )
