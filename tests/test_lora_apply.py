"""Spec for ``ltx25_studio/lora/apply.py`` — PLAN §4.3 stage 7, §9.3.1.

**This is the most important file in the suite.** Stages 1-6 catch broken files.
Stage 7 catches a *valid* file that is simply for a different model — the case
where diffusers loads it happily, attaches it to nothing, and the user gets base
model output for the rest of the session with no error anywhere.

The contract:

* ``apply_lora(pipeline, source, *, adapter_name="default", weight=1.0,
  fuse=False) -> LoRAAttachReport``
* A module counts as attached when it exposes a ``lora_A`` mapping containing
  ``adapter_name``. That mirrors peft's ``LoraLayer.lora_A: nn.ModuleDict``,
  which is what actually gets created — and, critically, is *not* the same thing
  as ``peft_config`` being populated. peft sets the config whether or not
  anything attached; the fake pipeline reproduces that, so an implementation
  that checks ``peft_config`` fails these tests exactly as it would fail a user.
* Zero attached modules raises ``LoRANotAttachedError``. Never a warning, never a
  log line, never a silent success.
* Connectors count. ``_lora_loadable_modules == ["transformer", "connectors"]``
  and a connector-only adapter is legitimate — counting transformer modules
  alone would reject a working LoRA.
"""

from __future__ import annotations

import inspect as py_inspect

import pytest

from conftest import (
    CONNECTOR_MODULE_PATHS,
    NATIVE_CONNECTOR_PREFIX,
    NATIVE_PREFIX,
    OFFICIAL_METADATA,
    TRANSFORMER_MODULE_PATHS,
    lora_tensors,
)
from ltx25_studio.errors import LoRAError, LoRANotAttachedError
from ltx25_studio.lora import apply as lora_apply


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_a_matching_adapter_attaches_to_more_than_zero_modules(fake_pipeline, good_lora):
    report = lora_apply.apply_lora(fake_pipeline, good_lora)
    assert len(report.attached_modules) > 0


def test_the_report_names_the_modules_it_attached_to(fake_pipeline, good_lora):
    report = lora_apply.apply_lora(fake_pipeline, good_lora)
    assert any("attn1.to_q" in name for name in report.attached_modules), report.attached_modules


def test_the_report_counts_per_component(fake_pipeline, good_lora):
    report = lora_apply.apply_lora(fake_pipeline, good_lora)
    assert report.per_component["transformer"] > 0
    assert set(report.per_component) >= {"transformer", "connectors"}


def test_the_adapter_name_is_carried_through(fake_pipeline, good_lora):
    report = lora_apply.apply_lora(fake_pipeline, good_lora, adapter_name="user_lora")
    assert report.adapter_name == "user_lora"
    assert fake_pipeline.calls[0]["adapter_name"] == "user_lora"


def test_the_pipeline_actually_received_the_source(fake_pipeline, good_lora):
    lora_apply.apply_lora(fake_pipeline, good_lora)
    assert fake_pipeline.calls, "apply_lora must call load_lora_weights"
    assert str(fake_pipeline.calls[0]["source"]) == str(good_lora)


# --------------------------------------------------------------------------
# THE anti-silent-garbage test (PLAN §4.3 stage 7, §9.3.1)
# --------------------------------------------------------------------------


def test_a_wrong_architecture_adapter_raises_instead_of_silently_doing_nothing(
    fake_pipeline, wrong_architecture_lora
):
    """The headline test.

    The adapter is well-formed: every key contains ``lora``, the prefix is the
    recognised native one, the pairs are complete, ``model_version`` says 2.5.0.
    It simply names modules this transformer does not have. diffusers loads it
    without complaint and attaches it to nothing.

    If this test fails, the headline feature ships broken and no one finds out
    until a user wonders why their LoRA "does nothing".
    """
    with pytest.raises(LoRANotAttachedError):
        lora_apply.apply_lora(fake_pipeline, wrong_architecture_lora)


def test_the_not_attached_error_is_specific_and_actionable(fake_pipeline, wrong_architecture_lora):
    with pytest.raises(LoRANotAttachedError) as excinfo:
        lora_apply.apply_lora(fake_pipeline, wrong_architecture_lora, adapter_name="user_lora")
    message = str(excinfo.value)
    assert "wrong-arch-lora.safetensors" in message, message
    assert "user_lora" in message, message
    assert "transformer" in message and "connectors" in message, (
        f"the message should name the components that were searched; got {message!r}"
    )


def test_the_not_attached_error_is_not_a_bare_exception(fake_pipeline, wrong_architecture_lora):
    with pytest.raises(LoRAError) as excinfo:
        lora_apply.apply_lora(fake_pipeline, wrong_architecture_lora)
    assert type(excinfo.value) is not Exception
    assert not isinstance(excinfo.value, (ValueError, RuntimeError, KeyError))


def test_apply_lora_never_returns_a_report_when_nothing_attached(
    fake_pipeline, wrong_architecture_lora
):
    """Belt and braces: not "returns a report with attached=0", not "returns None"."""
    result = None
    try:
        result = lora_apply.apply_lora(fake_pipeline, wrong_architecture_lora)
    except LoRANotAttachedError:
        pass
    assert result is None, f"a zero-attach load must raise, not return {result!r}"


def test_the_fake_pipeline_reproduces_the_peft_config_trap(fake_pipeline, wrong_architecture_lora):
    """Documents *why* the check must count modules.

    After a zero-attach load, ``peft_config`` is populated and
    ``get_list_adapters()`` reports the adapter. Both say "loaded". Only a module
    count tells the truth.
    """
    with pytest.raises(LoRANotAttachedError):
        lora_apply.apply_lora(fake_pipeline, wrong_architecture_lora)
    assert fake_pipeline.transformer.peft_config, "the fake must model peft's own optimism"
    assert fake_pipeline.get_list_adapters()["transformer"]


def test_an_adapter_matching_nothing_in_either_component_raises(make_pipeline, good_lora):
    """Same failure from the other direction: the adapter is fine, the pipeline
    is a different architecture."""
    pipeline = make_pipeline(
        transformer_paths=("some.other.module",), connector_paths=("yet.another",)
    )
    with pytest.raises(LoRANotAttachedError):
        lora_apply.apply_lora(pipeline, good_lora)


# --------------------------------------------------------------------------
# connectors count (PLAN §4.1)
# --------------------------------------------------------------------------


def test_a_connector_only_adapter_is_accepted(fake_pipeline, write_safetensors):
    """``_lora_loadable_modules == ["transformer", "connectors"]``.

    A LoRA that touches only the text-conditioning path attaches to zero
    transformer modules. Counting transformer modules alone would reject it —
    a false negative on a perfectly good adapter.
    """
    path = write_safetensors(
        name="connector-only.safetensors",
        tensors=lora_tensors(NATIVE_CONNECTOR_PREFIX, CONNECTOR_MODULE_PATHS),
        metadata=OFFICIAL_METADATA,
    )
    report = lora_apply.apply_lora(fake_pipeline, path)
    assert report.per_component["connectors"] > 0
    assert report.per_component["transformer"] == 0
    assert len(report.attached_modules) > 0


def test_an_adapter_spanning_both_components_counts_both(fake_pipeline, write_safetensors):
    tensors = lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS)
    tensors |= lora_tensors(NATIVE_CONNECTOR_PREFIX, CONNECTOR_MODULE_PATHS)
    path = write_safetensors(name="both.safetensors", tensors=tensors, metadata=OFFICIAL_METADATA)
    report = lora_apply.apply_lora(fake_pipeline, path)
    assert report.per_component["transformer"] > 0
    assert report.per_component["connectors"] > 0


# --------------------------------------------------------------------------
# fusing must not silently skip the connectors
# --------------------------------------------------------------------------


def _fuse_components(call):
    if call["args"]:
        return list(call["args"][0])
    return list(call["kwargs"].get("components", []))


def test_fusing_passes_both_loadable_components(fake_pipeline, good_lora):
    """``fuse_lora``/``unfuse_lora`` default to ``components=["transformer"]``.

    Fusing with the default leaves connector layers unfused — the adapter is
    half-applied and nothing says so.
    """
    lora_apply.apply_lora(fake_pipeline, good_lora, fuse=True)
    assert fake_pipeline.fuse_calls, "fuse=True must call fuse_lora"
    components = _fuse_components(fake_pipeline.fuse_calls[0])
    assert set(components) >= {"transformer", "connectors"}, components


def test_not_fusing_by_default(fake_pipeline, good_lora):
    lora_apply.apply_lora(fake_pipeline, good_lora)
    assert not fake_pipeline.fuse_calls


def test_unfusing_passes_both_loadable_components(fake_pipeline, good_lora):
    lora_apply.apply_lora(fake_pipeline, good_lora, fuse=True)
    lora_apply.unfuse_lora(fake_pipeline)
    assert fake_pipeline.unfuse_calls, "unfuse_lora must reach the pipeline"
    components = _fuse_components(fake_pipeline.unfuse_calls[0])
    assert set(components) >= {"transformer", "connectors"}, components


# --------------------------------------------------------------------------
# upstream errors are translated, not leaked
# --------------------------------------------------------------------------


def test_the_upstream_substring_rejection_becomes_one_of_our_errors(
    fake_pipeline, write_safetensors
):
    """diffusers raises a bare ``ValueError`` for a checkpoint whose keys lack
    ``lora``. §7.2 says errors cross layers as *our* typed exceptions."""
    path = write_safetensors(
        name="no-lora-substring.safetensors",
        tensors={"diffusion_model.transformer_blocks.0.attn1.to_q.weight": ("F32", [4, 4])},
        metadata=OFFICIAL_METADATA,
    )
    with pytest.raises(LoRAError):
        lora_apply.apply_lora(fake_pipeline, path)


# --------------------------------------------------------------------------
# the check cannot be turned off
# --------------------------------------------------------------------------


def test_there_is_no_flag_that_skips_the_attach_assertion():
    """PLAN §4.3: "It must run on every load, including the built-in LoRAs."

    A ``skip_validation=True`` escape hatch is how this check quietly stops
    running six months from now.
    """
    params = py_inspect.signature(lora_apply.apply_lora).parameters
    banned = {
        "skip_check",
        "skip_validation",
        "validate",
        "check",
        "assert_attached",
        "strict",
        "force",
        "trusted",
    }
    found = banned & set(params)
    assert not found, f"apply_lora exposes {sorted(found)} — stage 7 must not be optional"
