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
import types

import pytest
from conftest import (
    CONNECTOR_MODULE_PATHS,
    NATIVE_CONNECTOR_PREFIX,
    NATIVE_PREFIX,
    OFFICIAL_METADATA,
    TRANSFORMER_MODULE_PATHS,
    lora_tensors,
)

from ltx25_studio.errors import (
    LoRADimensionError,
    LoRAError,
    LoRAFormatError,
    LoRANotAttachedError,
)
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
    # Type identity, not isinstance, on purpose: what this test forbids is the
    # raise site throwing a *naked builtin* and losing the typed contract.
    # Since the fork-boundary fix the whole hierarchy inherits ValueError via
    # gr.Error (errors.LTXStudioError explains the audit), so isinstance
    # against ValueError is now true of every properly typed error and no
    # longer distinguishes bare from typed.
    assert type(excinfo.value) not in (ValueError, RuntimeError, KeyError)


def test_apply_lora_never_returns_a_report_when_nothing_attached(
    fake_pipeline, wrong_architecture_lora
):
    """Belt and braces: not "returns a report with attached=0", not "returns None"."""
    result = None
    with pytest.raises(LoRANotAttachedError):
        result = lora_apply.apply_lora(fake_pipeline, wrong_architecture_lora)
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


def test_real_peft_refuses_a_wrong_dimension_adapter_and_says_which_module(tmp_path):
    """**Stage 7 alone would not explain this one.** The 2B-shaped adapter's
    module names are LTX's, so it gets as far as the load; real peft then raises
    torch's size-mismatch, one line per tensor. The user is told the first
    mismatching module, as a dimension error, not "could not be read"."""
    tiny = _tiny_pipeline()
    path = tiny.adapter(tmp_path / "wrong.safetensors", in_features={"to_q": 8, "to_k": 8})
    with pytest.raises(LoRADimensionError) as excinfo:
        lora_apply.apply_lora(tiny.pipe, path, adapter_name="user_lora_0")
    error = excinfo.value
    assert error.module == "transformer_blocks.0.attn1.to_q"
    assert (error.expected, error.actual) == (16, 8)
    assert "attn1.to_k" not in str(error), "only the first mismatch is named"
    assert isinstance(error.__cause__, RuntimeError)
    assert "size mismatch" in str(error.__cause__)


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


# --------------------------------------------------------------------------
# F55 — load failures say what they are, with the original kept as __cause__
# --------------------------------------------------------------------------

#: torch's own wording, as real peft surfaces it through diffusers (measured on
#: diffusers @ac56fa2f + peft 0.20.0 with a tiny LTX2 transformer).
SIZE_MISMATCH = "Error(s) in loading state_dict for LTX2VideoTransformer3DModel:\n" + "".join(
    f"\tsize mismatch for transformer_blocks.{i}.attn1.to_k.lora_A.user_lora_0.weight: "
    "copying a param with shape torch.Size([32, 2048]) from checkpoint, the shape in "
    "current model is torch.Size([32, 4096]).\n"
    for i in range(448)
)

PEFT_NOTHING_TARGETED = (
    "No modules were targeted for adaptation. This might be caused by a combination of "
    "mismatched target modules and excluded modules. Please check your `target_modules` and "
    "`exclude_modules` configuration. You may also have only targeted modules that are "
    "marked to be saved (`modules_to_save`)."
)


class RaisingPipeline:
    _lora_loadable_modules = ["transformer", "connectors"]

    def __init__(self, exc):
        from conftest import FakeComponent

        self.exc = exc
        self.transformer = FakeComponent(("a.b",))
        self.connectors = FakeComponent(("c.d",))

    def load_lora_weights(self, source, adapter_name="default", **kwargs):
        raise self.exc


def test_a_torch_size_mismatch_becomes_a_one_line_dimension_error(good_lora):
    original = RuntimeError(SIZE_MISMATCH)
    with pytest.raises(LoRADimensionError) as excinfo:
        lora_apply.apply_lora(RaisingPipeline(original), good_lora, adapter_name="user_lora_0")
    error = excinfo.value
    assert error.module == "transformer_blocks.0.attn1.to_k"
    assert (error.expected, error.actual) == (4096, 2048)
    # Bounded as a person reads it: the temp root's length is not the message's.
    message = str(error).replace(str(good_lora), good_lora.name)
    assert len(message) < 600, f"{len(message)} characters: {message[:200]}…"
    assert error.__cause__ is original


def test_a_lora_b_size_mismatch_reads_the_out_dimension(good_lora):
    original = RuntimeError(
        "Error(s) in loading state_dict for X:\n\tsize mismatch for "
        "transformer_blocks.3.ff.net.2.lora_B.user_lora_0.weight: copying a param with shape "
        "torch.Size([2048, 16]) from checkpoint, the shape in current model is "
        "torch.Size([4096, 16])."
    )
    with pytest.raises(LoRADimensionError) as excinfo:
        lora_apply.apply_lora(RaisingPipeline(original), good_lora, adapter_name="user_lora_0")
    assert excinfo.value.module == "transformer_blocks.3.ff.net.2"
    assert (excinfo.value.expected, excinfo.value.actual) == (4096, 2048)


@pytest.mark.parametrize(
    "message",
    [
        PEFT_NOTHING_TARGETED,
        "Target modules {'down_blocks.0.conv'} not found in the base model. Please check the "
        "target modules and try again.",
    ],
    ids=["peft-0.20", "older-peft"],
)
def test_peft_finding_nothing_to_target_is_the_not_attached_error(message, good_lora):
    original = ValueError(message)
    with pytest.raises(LoRANotAttachedError) as excinfo:
        lora_apply.apply_lora(RaisingPipeline(original), good_lora, adapter_name="user_lora_0")
    assert excinfo.value.__cause__ is original
    assert excinfo.value.adapter_name == "user_lora_0"


def test_a_duplicate_adapter_name_is_reported_as_one(good_lora):
    original = ValueError(
        "Adapter name user_lora_0 already in use in the model - please select a new adapter name."
    )
    with pytest.raises(lora_apply.LoRADuplicateAdapterError) as excinfo:
        lora_apply.apply_lora(RaisingPipeline(original), good_lora, adapter_name="user_lora_0")
    assert isinstance(excinfo.value, LoRAError)
    assert "user_lora_0" in str(excinfo.value) and "already" in str(excinfo.value)
    assert "could not be read" not in str(excinfo.value)
    assert excinfo.value.__cause__ is original


def test_any_other_load_failure_is_cut_to_its_first_line(good_lora):
    original = OSError("disk read failed\n" + "noise\n" * 500)
    with pytest.raises(LoRAFormatError) as excinfo:
        lora_apply.apply_lora(RaisingPipeline(original), good_lora)
    assert "disk read failed" in str(excinfo.value)
    assert "noise" not in str(excinfo.value)
    assert excinfo.value.__cause__ is original


# --------------------------------------------------------------------------
# F53 — weight is validated and reported; the report reads as a run-report line
# --------------------------------------------------------------------------


def test_the_attach_report_reads_as_one_run_report_line(fake_pipeline, write_safetensors):
    path = write_safetensors(
        name="BTSMovieSet_LTX25_v1.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata=OFFICIAL_METADATA,
    )
    report = lora_apply.apply_lora(fake_pipeline, path, adapter_name="user_lora_0", weight=0.8)
    assert report.weight == 0.8
    assert report.summary == (
        f"BTSMovieSet_LTX25_v1 attached to {report.attached} modules at weight 0.8"
    )


def test_a_label_names_the_adapter_in_the_report(fake_pipeline, good_lora):
    report = lora_apply.apply_lora(fake_pipeline, good_lora, label="org/my-lora")
    assert report.summary.startswith("org/my-lora attached to ")
    assert report.summary.endswith(" at weight 1.0")


@pytest.mark.parametrize("weight", [0, -0.5, 2.01, float("nan"), float("inf")])
def test_an_out_of_range_weight_is_refused_before_loading(weight, fake_pipeline, good_lora):
    with pytest.raises(LoRAError) as excinfo:
        lora_apply.apply_lora(fake_pipeline, good_lora, weight=weight)
    assert "strength" in str(excinfo.value)
    assert fake_pipeline.calls == [], "nothing may load with a weight outside (0, 2]"


@pytest.mark.parametrize("weight", [0.05, 1.0, 2.0])
def test_weights_inside_the_range_are_applied(weight, fake_pipeline, good_lora):
    lora_apply.apply_lora(fake_pipeline, good_lora, weight=weight)
    assert fake_pipeline.calls[-1] == {"set_adapters": (("default", weight), {})}


# --------------------------------------------------------------------------
# the same mappings through real diffusers + peft (FULL env only)
# --------------------------------------------------------------------------


class _Tiny:
    def __init__(self, pipe, linears):
        self.pipe = pipe
        self.linears = linears

    def adapter(self, path, modules=("to_q", "to_k"), in_features=None, block="attn1"):
        import torch
        from safetensors.torch import save_file

        tensors = {}
        for leaf in modules:
            name = f"transformer_blocks.0.{block}.{leaf}"
            in_dim, out_dim = self.linears.get(name, (16, 16))
            in_dim = (in_features or {}).get(leaf, in_dim)
            tensors[f"diffusion_model.{name}.lora_A.weight"] = torch.zeros(4, in_dim)
            tensors[f"diffusion_model.{name}.lora_B.weight"] = torch.zeros(out_dim, 4)
        save_file(tensors, str(path))
        return path


def _tiny_pipeline():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    diffusers = pytest.importorskip("diffusers")
    from diffusers.loaders.lora_pipeline import LTX2LoraLoaderMixin

    transformer = diffusers.LTX2VideoTransformer3DModel(
        in_channels=8,
        out_channels=8,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=1,
        caption_channels=16,
    )

    class TinyPipeline(LTX2LoraLoaderMixin):
        hf_device_map = None

        def __init__(self):
            self.transformer = transformer
            self.connectors = None

        @property
        def components(self):
            return {"transformer": self.transformer}

    linears = {
        name: (module.in_features, module.out_features)
        for name, module in transformer.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    return _Tiny(TinyPipeline(), linears)


def test_real_peft_attaches_a_matching_adapter_and_reports_it(tmp_path):
    tiny = _tiny_pipeline()
    report = lora_apply.apply_lora(
        tiny.pipe,
        tiny.adapter(tmp_path / "tiny.safetensors"),
        adapter_name="user_lora_0",
        weight=0.5,
    )
    assert report.attached == 2
    assert report.summary == "tiny attached to 2 modules at weight 0.5"


def test_real_peft_finding_nothing_is_the_not_attached_error(tmp_path):
    tiny = _tiny_pipeline()
    path = tiny.adapter(tmp_path / "foreign.safetensors", modules=("conv1",), block="resnets")
    with pytest.raises(LoRANotAttachedError) as excinfo:
        lora_apply.apply_lora(tiny.pipe, path, adapter_name="user_lora_0")
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_real_peft_duplicate_name_is_reported_as_one(tmp_path):
    tiny = _tiny_pipeline()
    path = tiny.adapter(tmp_path / "twice.safetensors")
    lora_apply.apply_lora(tiny.pipe, path, adapter_name="user_lora_0")
    with pytest.raises(lora_apply.LoRADuplicateAdapterError):
        lora_apply.apply_lora(tiny.pipe, path, adapter_name="user_lora_0")


# --------------------------------------------------------------------------
# F57: an adapter a failed detach left behind, on the Space's resident runner
# --------------------------------------------------------------------------


def _adapters_on(transformer) -> set[str]:
    """What peft holds on ``transformer``, read off peft's own records."""
    held = set(getattr(transformer, "peft_config", None) or ())
    for _name, module in transformer.named_modules():
        layers = getattr(module, "lora_A", None)
        if layers is not None:
            held.update(layers.keys())
    return held


def _resident_runner(tiny):
    """``pipelines.StageRunner`` over the tiny pipeline; the denoise records its adapters."""
    from ltx25_studio import pipelines

    seen = []

    def denoise(self, **kwargs):
        seen.append(_adapters_on(self.transformer))
        return types.SimpleNamespace(frames=("frame",), audio=None)

    type(tiny.pipe).__call__ = denoise
    return pipelines.StageRunner(tiny.pipe, kind="text2video", device="cpu"), seen


def _generate_with(runner, tmp_path, *adapters):
    from test_exports_contract import one_muxed_file_encoder

    from ltx25_studio import generate as generate_module

    return generate_module.generate(
        mode="text_to_video",
        prompt="a lighthouse in fog",
        seed=7,
        width=64,
        height=64,
        num_frames=9,
        loras=tuple({"source": str(path), "weight": 1.0, "report": object()} for path in adapters),
        pipeline_factory=lambda **_: runner,
        encoder=one_muxed_file_encoder,
        output_root=tmp_path / "out",
        retention_bytes=10**9,
    )


@pytest.mark.parametrize("stuck", ["once", "always"])
def test_real_peft_a_stuck_adapter_never_reaches_the_next_runs_denoise(tmp_path, monkeypatch, stuck):
    from ltx25_studio.errors import GenerationError

    tiny = _tiny_pipeline()
    runner, seen = _resident_runner(tiny)
    first = tiny.adapter(tmp_path / "first.safetensors")
    real_delete = tiny.pipe.delete_adapters
    attempts = []

    def delete_adapters(names):
        attempts.append(names)
        if stuck == "always" or len(attempts) == 1:
            raise RuntimeError("peft refused (stand-in)")
        return real_delete(names)

    monkeypatch.setattr(tiny.pipe, "delete_adapters", delete_adapters)
    with pytest.raises(GenerationError):
        _generate_with(runner, tmp_path, first)
    assert seen == [{"user_lora_0"}]
    assert _adapters_on(tiny.pipe.transformer) == {"user_lora_0"}, "the detach really failed"

    if stuck == "always":
        with pytest.raises(LoRAError) as caught:
            _generate_with(runner, tmp_path)
        assert len(seen) == 1, "the refused run was sampled anyway"
        assert isinstance(caught.value, lora_apply.LoRALeftoverError), caught.value
        return
    result = _generate_with(runner, tmp_path)
    assert seen[-1] == set(), "the second run was sampled through the first run's adapter"
    assert result.record.adapters == ()


def test_real_peft_a_run_after_a_stuck_adapter_gets_exactly_its_own(tmp_path, monkeypatch):
    tiny = _tiny_pipeline()
    import torch
    from safetensors.torch import load_file, save_file

    from ltx25_studio.errors import GenerationError

    runner, seen = _resident_runner(tiny)
    first = tiny.adapter(tmp_path / "first.safetensors")
    second = tmp_path / "second.safetensors"
    save_file(
        {key: torch.full_like(value, 0.25) for key, value in load_file(first).items()}, str(second)
    )
    real_delete = tiny.pipe.delete_adapters
    attempts = []

    def delete_adapters(names):
        attempts.append(names)
        if len(attempts) == 1:
            raise RuntimeError("peft refused (stand-in)")
        return real_delete(names)

    monkeypatch.setattr(tiny.pipe, "delete_adapters", delete_adapters)
    with pytest.raises(GenerationError):
        _generate_with(runner, tmp_path, first)

    weights = []
    real_call = type(tiny.pipe).__call__

    def denoise(self, **kwargs):
        module = self.transformer.transformer_blocks[0].attn1.to_q
        weights.append(float(module.lora_A["user_lora_0"].weight.detach().flatten()[0]))
        return real_call(self, **kwargs)

    monkeypatch.setattr(type(tiny.pipe), "__call__", denoise)
    result = _generate_with(runner, tmp_path, second)

    assert seen[-1] == {"user_lora_0"}
    assert weights == [0.25], "the denoise ran with the first run's weights, not its own"
    ((label, _weight, attached),) = result.record.adapters
    assert label == "second" and attached == 2
