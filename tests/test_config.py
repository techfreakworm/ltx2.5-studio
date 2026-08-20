"""Spec for the runtime configuration levers.

**Every lever here must be attached to something.** This file's own history is
the reason: a precision variable once existed, was wired to nothing, and made
the startup report state a footprint nobody was loading (PLAN §2.6). Two more
were removed the same way.

The quantization lever below is the counter-example — it is real, it changes
output, and it exists because bf16 does not fit ZeroGPU's disk.
"""

from __future__ import annotations

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


# NOTE: `test_only_the_transformer_is_quantised` lived here and is retired.
# It inspected the source of `_quantization_config` to assert transformer-only
# was *expressed*, because at the time it was an intent rather than a value.
# It is now `components_to_quantize=["transformer"]` -- a real argument -- so
# `test_bound_b_is_expressed_as_an_argument` asserts it behaviourally instead.
# A source-text test earns its place only while the thing it checks has no
# observable form; this one stopped earning it the moment the API gave it one.
