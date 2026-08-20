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
    ascribed to the denoiser rather than to prompt encoding.

    ⚠️ **This inspects source text, and that is all it can do.** It asserts the
    function is *written* to quantise the transformer alone. It says nothing
    about whether its result reaches ``from_pretrained`` — verified by deleting
    the ``load_kwargs["quantization_config"]`` line, which leaves this test
    **green** while the Space would deploy bf16 under an int8 variable.

    ``test_int8_reaches_from_pretrained`` is what stands behind the lever. This
    keeps a narrower job: that transformer-only is **expressed as an intent**
    rather than merely being what today's call happens to produce, so widening
    it later is a visible decision.
    """
    from ltx25_studio import pipelines

    source = inspect.getsource(pipelines._quantization_config)
    assert "load_in_8bit=True" in source
    assert "text_encoder" not in source.split('"""')[2], (
        "the text encoder is named in the quantisation call; B was chosen "
        "deliberately over A and widening needs its own decision"
    )
