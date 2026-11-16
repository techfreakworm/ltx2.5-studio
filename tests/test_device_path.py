"""Spec for the device path: which device generates, and what each one needs.

``LTX25_DEVICE`` is ``auto`` (default), ``cuda``, ``mps`` or ``cpu``. ``auto``
means CUDA on ZeroGPU or wherever CUDA exists, and **no generation device**
everywhere else: the UI starts and Generate says, in plain words, how to switch
generation on. Before this, a Mac with the weights on disk would build the whole
~70 GB pipeline on the CPU on the first click (recon B N5).

Most of this file runs without torch. The parts that need real tensors
``importorskip`` it.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest
from conftest import inputs_for

from ltx25_studio import config, device, pipelines, zerogpu
from ltx25_studio import generate as generate_module
from ltx25_studio.errors import ConfigError, ConstraintError, LTXStudioError


def _real_spaces():
    """The platform package as app.py sees it: a module, not the local stub."""
    module = types.ModuleType("spaces")
    module.GPU = lambda *a, **k: lambda f: f
    return module


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: False)


@pytest.fixture
def no_probe(monkeypatch):
    """Any accelerator probe is a failure: on ZeroGPU it initialises CUDA in the
    web process, which the platform refuses."""

    def forbidden():
        raise AssertionError("an accelerator was probed")

    monkeypatch.setattr(zerogpu, "host_has_cuda", forbidden)
    monkeypatch.setattr(device, "host_has_mps", forbidden)


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def test_auto_on_a_machine_with_no_cuda_is_ui_only(no_cuda):
    resolved = device.resolve(zerogpu.local_spaces_stub())

    assert resolved.device is None
    assert not resolved.generation_enabled
    assert "LTX25_DEVICE" in resolved.reason, resolved.reason


def test_auto_on_a_cuda_host_is_cuda(monkeypatch):
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: True)

    assert device.resolve(zerogpu.local_spaces_stub()).device == "cuda"


def test_zerogpu_resolves_to_cuda_without_probing_anything(no_probe):
    resolved = device.resolve(_real_spaces())

    assert resolved.device == "cuda"
    assert resolved.on_zerogpu


@pytest.mark.parametrize("requested", ["mps", "cpu"])
def test_a_local_device_on_zerogpu_is_a_startup_error(monkeypatch, no_probe, requested):
    monkeypatch.setenv(config.ENV_DEVICE, requested)

    with pytest.raises(ConfigError) as excinfo:
        device.resolve(_real_spaces())
    assert requested in str(excinfo.value)


def test_mps_is_an_explicit_opt_in(monkeypatch, no_cuda):
    monkeypatch.setattr(device, "host_has_mps", lambda: True)
    assert device.resolve(None).device is None, "auto must never pick MPS by itself"

    monkeypatch.setenv(config.ENV_DEVICE, "mps")
    assert device.resolve(None).device == "mps"


def test_mps_requested_where_there_is_none_is_refused(monkeypatch, no_cuda):
    monkeypatch.setattr(device, "host_has_mps", lambda: False)
    monkeypatch.setenv(config.ENV_DEVICE, "mps")

    with pytest.raises(ConfigError) as excinfo:
        device.resolve(None)
    assert "mps" in str(excinfo.value).lower()


def test_cuda_requested_where_there_is_none_is_refused(monkeypatch, no_cuda):
    monkeypatch.setenv(config.ENV_DEVICE, "cuda")

    with pytest.raises(ConfigError):
        device.resolve(None)


def test_cpu_is_always_available(monkeypatch, no_cuda):
    monkeypatch.setenv(config.ENV_DEVICE, "cpu")

    assert device.resolve(None).device == "cpu"


# --------------------------------------------------------------------------
# quantisation and dtype follow the device
# --------------------------------------------------------------------------


def test_auto_quantisation_is_int8_on_zerogpu(no_probe):
    """bf16 does not fit the platform's disk cap (docs/ZEROGPU_DISK.md)."""
    assert device.resolve(_real_spaces()).quantization == "int8"


def test_auto_quantisation_is_bf16_on_a_plain_cuda_host(monkeypatch):
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: True)

    assert device.resolve(None).quantization == "bf16"


def test_an_explicit_bf16_on_zerogpu_is_honoured(monkeypatch, no_probe):
    monkeypatch.setenv(config.ENV_QUANTIZATION, "bf16")

    assert device.resolve(_real_spaces()).quantization == "bf16"


@pytest.mark.parametrize("requested", ["mps", "cpu"])
def test_int8_off_cuda_is_a_startup_error_that_says_why(monkeypatch, no_cuda, requested):
    monkeypatch.setattr(device, "host_has_mps", lambda: True)
    monkeypatch.setenv(config.ENV_DEVICE, requested)
    monkeypatch.setenv(config.ENV_QUANTIZATION, "int8")

    with pytest.raises(ConfigError) as excinfo:
        device.resolve(None)
    message = str(excinfo.value)
    assert "int8" in message and "bitsandbytes" in message, message


def test_dtypes_per_device(monkeypatch, no_cuda):
    monkeypatch.setattr(device, "host_has_mps", lambda: True)
    monkeypatch.setenv(config.ENV_DEVICE, "mps")
    assert device.resolve(None).weight_dtype == "bfloat16"

    monkeypatch.setenv(config.ENV_DEVICE, "cpu")
    assert device.resolve(None).weight_dtype == "float32", "fp32 is the CPU reference"
    monkeypatch.setenv(config.ENV_CPU_DTYPE, "bf16")
    assert device.resolve(None).weight_dtype == "bfloat16"


def test_the_cuda_dtype_is_unchanged(no_probe):
    assert device.resolve(_real_spaces()).weight_dtype == pipelines.WEIGHT_DTYPE


# --------------------------------------------------------------------------
# F13: no silent 70 GB load on a machine with no generation device
# --------------------------------------------------------------------------


def test_get_pipeline_refuses_on_a_ui_only_host_and_builds_nothing(monkeypatch, no_cuda):
    built = []
    monkeypatch.setattr(pipelines, "_base_pipeline", lambda **k: built.append(k))
    pipelines.clear_cache()

    with pytest.raises(device.GenerationDisabledError) as excinfo:
        pipelines.get_pipeline(pipeline="text2video")

    assert not built, "a pipeline was constructed on a machine with no generation device"
    message = str(excinfo.value)
    assert "LTX25_DEVICE=mps" in message and "LTX25_DEVICE=cpu" in message, message


def _override_snapshot(monkeypatch, tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path))


def test_a_snapshot_override_keeps_the_resident_stage_runner(monkeypatch, no_cuda, tmp_path):
    from ltx25_studio import local_runtime

    monkeypatch.setenv(config.ENV_DEVICE, "cpu")
    _override_snapshot(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(pipelines, "_base_pipeline", lambda **k: seen.append(k) or object())
    monkeypatch.setattr(pipelines, "derive_pipeline", lambda base, cls: base)
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(LTX2Pipeline=object))
    monkeypatch.setattr(
        local_runtime,
        "build_local_runner",
        lambda **k: pytest.fail("dispatched to the local runner"),
    )
    pipelines.clear_cache()
    try:
        runner = pipelines.get_pipeline(pipeline="text2video")
    finally:
        pipelines.clear_cache()

    assert seen and seen[0]["device"] == "cpu", seen
    assert isinstance(runner, pipelines.StageRunner) and runner.device == "cpu"


@pytest.fixture
def fake_local_runner(monkeypatch):
    """``build_local_runner`` replaced by a recorder: no real weights anywhere."""
    from test_exports_contract import FakePipe

    from ltx25_studio import local_runtime

    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return FakePipe()

    monkeypatch.setattr(local_runtime, "build_local_runner", build)
    monkeypatch.setattr(pipelines, "_base_pipeline", lambda **k: pytest.fail("StageRunner built"))
    monkeypatch.delenv(config.ENV_MODEL_SNAPSHOT_DIR, raising=False)
    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", None)
    pipelines.clear_cache()
    return calls


def test_generate_on_a_local_device_with_real_weights_uses_the_local_runner(
    monkeypatch, no_cuda, tmp_path, fake_local_runner
):
    """The seam the UI uses: generate() -> default factory -> get_pipeline -> dispatch."""
    from test_exports_contract import one_muxed_file_encoder

    from ltx25_studio import preflight

    gguf = tmp_path / "transformer-Q8_0.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setenv(config.ENV_DEVICE, "cpu")
    monkeypatch.setenv(config.ENV_TRANSFORMER_GGUF, str(gguf))
    monkeypatch.setenv(config.ENV_RESIDENCY, "staged")

    result = generate_module.generate(
        mode="image_to_video",
        seed=7,
        width=64,
        height=64,
        num_frames=9,
        encoder=one_muxed_file_encoder,
        output_root=tmp_path / "out",
        retention_bytes=10**9,
        **inputs_for("image_to_video"),
    )

    assert result.output_path.is_file()
    assert len(fake_local_runner) == 1, fake_local_runner
    call = dict(fake_local_runner[0])
    assert callable(call.pop("log"))
    assert call == {
        "kind": "condition",
        "device": "cpu",
        "dtype": "float32",
        "snapshot_dir": preflight.snapshot_dir(),
        "transformer_gguf": str(gguf.resolve()),
        "residency": "staged",
    }


def test_mps_dispatch_uses_bf16_and_the_snapshot_transformer(
    monkeypatch, no_cuda, fake_local_runner
):
    monkeypatch.setattr(device, "host_has_mps", lambda: True)
    monkeypatch.setattr(device, "_MPS_SELF_TEST", _self_test())
    monkeypatch.setenv(config.ENV_DEVICE, "mps")
    monkeypatch.setenv(config.ENV_RESIDENCY, "resident")

    pipelines.get_pipeline(pipeline="text2video")

    call = fake_local_runner[0]
    assert (call["kind"], call["device"], call["dtype"]) == ("text2video", "mps", "bfloat16")
    assert call["transformer_gguf"] is None and call["residency"] == "resident"


def test_an_explicit_mps_device_still_checks_numerics_before_the_local_runner(
    monkeypatch, fake_local_runner
):
    monkeypatch.setattr(device, "_MPS_SELF_TEST", _self_test(sdpa=False))

    with pytest.raises(device.MpsNumericsError):
        pipelines.get_pipeline(pipeline="text2video", device="mps")
    assert not fake_local_runner


def test_cuda_never_dispatches_to_the_local_runner(monkeypatch, tmp_path):
    from test_stage_graph_contract import install_fake_stack

    from ltx25_studio import local_runtime

    install_fake_stack(monkeypatch, tmp_path)
    monkeypatch.setattr(
        local_runtime, "build_local_runner", lambda **k: pytest.fail("CUDA reached local_runtime")
    )
    pipelines.clear_cache()
    try:
        runner = pipelines.get_pipeline(pipeline="condition", device="cuda")
    finally:
        pipelines.clear_cache()

    assert isinstance(runner, pipelines.StageRunner) and runner.device == "cuda"


@pytest.fixture
def ui_only_app(monkeypatch, no_cuda):
    """``import app`` on a machine with no generation device, then restore."""
    pytest.importorskip("gradio")
    for name in (config.ENV_DEVICE, config.ENV_QUANTIZATION, config.ENV_MODEL_SNAPSHOT_DIR):
        monkeypatch.delenv(name, raising=False)
    previous = generate_module.gpu_dispatch()
    sys.modules.pop("app", None)
    try:
        yield importlib.import_module("app")
    finally:
        generate_module.set_gpu_dispatch(previous)
        sys.modules.pop("app", None)


def test_generate_on_a_ui_only_host_raises_a_typed_plain_error(monkeypatch, ui_only_app):
    """The seam the UI uses: generate() -> the GPU entrypoint -> refusal.

    Nothing is loaded first -- not the pipeline and not the 2.5 GB enhancer.
    """
    from ltx25_studio import enhance

    def forbidden(*a, **k):
        raise AssertionError("a model was loaded on a machine with generation disabled")

    monkeypatch.setattr(pipelines, "_base_pipeline", forbidden)
    monkeypatch.setattr(enhance, "_load_enhancer", forbidden)
    pipelines.clear_cache()

    with pytest.raises(device.GenerationDisabledError) as excinfo:
        generate_module.generate(
            mode="text_to_video",
            seed=7,
            width=64,
            height=64,
            num_frames=9,
            enhance_prompt=True,
            **inputs_for("text_to_video"),
        )

    assert isinstance(excinfo.value, LTXStudioError), "the UI renders LTXStudioError only"
    message = str(excinfo.value)
    assert "not enabled on this machine" in message, message
    for internal in (".py", "PLAN", "§"):
        assert internal not in message, f"internal reference {internal!r} in: {message}"


def test_the_app_does_not_build_or_self_test_on_a_ui_only_host(monkeypatch, no_cuda):
    pytest.importorskip("gradio")
    for name in (config.ENV_DEVICE, config.ENV_MODEL_SNAPSHOT_DIR):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*a, **k):
        raise AssertionError("called at startup on a UI-only host")

    monkeypatch.setattr(pipelines, "get_pipeline", forbidden)
    monkeypatch.setattr(device, "mps_self_test", forbidden)
    previous = generate_module.gpu_dispatch()
    sys.modules.pop("app", None)
    try:
        module = importlib.import_module("app")
        assert module.DEVICE.device is None
    finally:
        generate_module.set_gpu_dispatch(previous)
        sys.modules.pop("app", None)


def test_an_injected_pipeline_is_not_refused(ui_only_app):
    """The offline gate injects its own pipeline; there is nothing to build."""
    resolved = ui_only_app.DEVICE
    assert resolved.require_generation(injected=True) is None


def test_the_cuda_gate_costs_nothing(monkeypatch, no_probe):
    """On CUDA the per-run check must not self-test, probe or import torch.mps."""
    monkeypatch.setattr(device, "mps_self_test", lambda **k: pytest.fail("self-tested"))

    assert device.resolve(_real_spaces()).require_generation() == "cuda"


# --------------------------------------------------------------------------
# stage 2 off CUDA: the decoder runs on the CPU and only at tiny sizes
# --------------------------------------------------------------------------


class _Recorder:
    def __init__(self, name, output=None):
        self.name = name
        self.calls = []
        self._output = output or types.SimpleNamespace(frames=f"{name}-latents", audio=None)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self._output


def _runner(device_name):
    pipeline = _Recorder("pipeline")
    return (
        pipelines.StageRunner(
            pipeline,
            kind="text2video",
            upsampler=_Recorder("up"),
            decoder=_Recorder("dec"),
            device=device_name,
        ),
        pipeline,
    )


@pytest.mark.parametrize("device_name", ["mps", "cpu"])
def test_a_large_stage_2_request_off_cuda_is_refused_before_any_work(device_name):
    runner, pipeline = _runner(device_name)

    with pytest.raises(device.StageTwoNeedsCudaError) as excinfo:
        runner(prompt="a lighthouse", width=704, height=480, num_frames=121, enable_stage_2=True)

    assert not pipeline.calls, "stage 1 ran before a refusal that needed no work to decide"
    assert isinstance(excinfo.value, ConstraintError)
    assert "CUDA" in str(excinfo.value)


def test_a_tiny_stage_2_request_off_cuda_runs():
    runner, pipeline = _runner("cpu")

    runner(prompt="a lighthouse", width=64, height=64, num_frames=9, enable_stage_2=True)

    assert len(pipeline.calls) == 2, "stage 1 and stage 2 should both have run"


def test_the_token_threshold_is_the_measured_one():
    """64x64x9 -> 128x128x9 decoded is the size the probe measured at +2.1 GB on
    the CPU; 17 frames at that size passed 9 GB and was killed."""
    assert device.stage_2_decoder_tokens(64, 64, 9) == device.OFF_CUDA_DECODER_TOKEN_LIMIT
    assert device.stage_2_decoder_tokens(64, 64, 17) > device.OFF_CUDA_DECODER_TOKEN_LIMIT


def test_cuda_stage_2_is_never_size_gated():
    runner, pipeline = _runner("cuda")

    runner(prompt="a lighthouse", width=704, height=480, num_frames=121, enable_stage_2=True)

    assert len(pipeline.calls) == 2


# --------------------------------------------------------------------------
# the stage-2 parts per device
# --------------------------------------------------------------------------


def test_stage_2_parts_off_cuda_keep_the_decoder_on_the_cpu_with_its_own_attention(
    monkeypatch, tmp_path
):
    from test_stage_graph_contract import FakeModel, install_fake_stack

    recorders = install_fake_stack(monkeypatch, tmp_path)

    upsampler_pipe, decoder_pipe = pipelines._stage_2_parts(FakeModel(), device="mps")

    natten = recorders["LTX2VideoVaeNeighborhoodNattenProcessor"]
    assert not natten.calls, "NATTEN has no macOS build; constructing it raises"
    decoder_calls = recorders["LTX2VideoDiffusionDecoderModel"].calls
    upsampler_calls = recorders["LTX2LatentUpsamplerModel"].calls
    assert decoder_calls and upsampler_calls


def test_stage_2_parts_on_cuda_are_unchanged(monkeypatch, tmp_path):
    from test_stage_graph_contract import FakeModel, install_fake_stack

    recorders = install_fake_stack(monkeypatch, tmp_path)
    decoder = FakeModel()
    upsampler = FakeModel()
    recorders["LTX2VideoDiffusionDecoderModel"]._result = decoder
    recorders["LTX2LatentUpsamplerModel"]._result = upsampler

    pipelines._stage_2_parts(FakeModel(), device="cuda")

    assert recorders["LTX2VideoVaeNeighborhoodNattenProcessor"].calls, "NATTEN must stay on CUDA"
    assert decoder.moved_to == "cuda" and upsampler.moved_to == "cuda"


def test_the_off_cuda_decoder_is_never_moved_to_the_accelerator(monkeypatch, tmp_path):
    from test_stage_graph_contract import FakeModel, install_fake_stack

    recorders = install_fake_stack(monkeypatch, tmp_path)
    decoder = FakeModel()
    upsampler = FakeModel()
    recorders["LTX2VideoDiffusionDecoderModel"]._result = decoder
    recorders["LTX2LatentUpsamplerModel"]._result = upsampler

    pipelines._stage_2_parts(FakeModel(), device="mps")

    assert decoder.moved_to in (None, "cpu"), "FlexAttention refuses MPS tensors"
    assert upsampler.moved_to == "mps"


# --------------------------------------------------------------------------
# the model-source override
# --------------------------------------------------------------------------


def _snapshot(tmp_path, components):
    import json

    index = {"_class_name": "LTX2Pipeline"}
    for name in components:
        index[name] = ["diffusers", "Something"]
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    index["prompt_enhancer"] = [None, None]
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    return tmp_path


def test_the_override_is_the_directory_every_component_loads_from(monkeypatch, tmp_path):
    from ltx25_studio import preflight

    _snapshot(tmp_path, ["transformer", "vae"])
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path))

    assert preflight.snapshot_dir() == tmp_path.resolve()
    assert pipelines._snapshot_path() == tmp_path.resolve()


def test_completeness_follows_the_override_model_index(monkeypatch, tmp_path):
    """Checked against what THAT snapshot declares, not the 37-file allow-list."""
    from ltx25_studio import preflight

    _snapshot(tmp_path, ["transformer", "vae"])
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path))
    assert preflight.missing_cached_files() == ()
    preflight.require_cache_complete()

    import shutil

    shutil.rmtree(tmp_path / "vae")
    assert preflight.missing_cached_files() == ("vae",)
    with pytest.raises(LTXStudioError) as excinfo:
        preflight.require_cache_complete()
    assert "vae" in str(excinfo.value)


def test_the_default_snapshot_is_unchanged_without_an_override(monkeypatch):
    from ltx25_studio import model_files, preflight

    monkeypatch.delenv(config.ENV_MODEL_SNAPSHOT_DIR, raising=False)

    path = preflight.snapshot_dir()
    assert path.name == model_files.REVISION
    assert "models--Lightricks--LTX-2.5-Diffusers" in str(path)


# --------------------------------------------------------------------------
# F21b: the MPS numerics self-test
# --------------------------------------------------------------------------


def _self_test(*, sdpa=True, linear=True):
    return device.MpsSelfTest(
        torch_version="2.11.0",
        checks=(
            device.SelfTestCheck("sdpa_scale", sdpa, "measured", blocking=True),
            device.SelfTestCheck("bf16_linear", linear, "measured", blocking=True),
            device.SelfTestCheck("float64", False, "unavailable", blocking=False),
        ),
    )


def test_a_failed_check_refuses_generation_naming_it_and_the_remedy(monkeypatch):
    monkeypatch.setattr(device, "_MPS_SELF_TEST", _self_test(sdpa=False))

    with pytest.raises(device.MpsNumericsError) as excinfo:
        device.require_mps_numerics()
    message = str(excinfo.value)
    assert "attention" in message.lower() and "2.15 or newer" in message, message
    assert "2.11.0" in message and "LTX25_MPS_SELF_TEST=warn" in message, message


def test_the_failed_linear_check_refuses_too(monkeypatch):
    monkeypatch.setattr(device, "_MPS_SELF_TEST", _self_test(linear=False))

    with pytest.raises(device.MpsNumericsError):
        device.require_mps_numerics()


def test_float64_alone_never_refuses(monkeypatch):
    """float64 is absent on every MPS device; the RoPE handling covers it."""
    monkeypatch.setattr(device, "_MPS_SELF_TEST", _self_test())

    assert device.require_mps_numerics().ok


def test_warn_mode_lets_an_operator_run_anyway(monkeypatch):
    monkeypatch.setattr(device, "_MPS_SELF_TEST", _self_test(sdpa=False))
    monkeypatch.setenv(config.ENV_MPS_SELF_TEST, "warn")

    result = device.require_mps_numerics()

    assert not result.ok
    assert "sdpa_scale" in result.describe()


def _mps(monkeypatch):
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: False)
    monkeypatch.setattr(device, "host_has_mps", lambda: True)
    monkeypatch.setenv(config.ENV_DEVICE, "mps")
    return device.resolve(None)


def _result():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Result:
        warnings: tuple = ("an earlier note",)

    return Result()


def test_warn_mode_marks_every_result_with_the_failed_check(monkeypatch):
    monkeypatch.setattr(device, "_MPS_SELF_TEST", _self_test(sdpa=False))
    monkeypatch.setenv(config.ENV_MPS_SELF_TEST, "warn")

    annotated = _mps(monkeypatch).annotate(_result())

    assert annotated.warnings[0] == "an earlier note"
    assert len(annotated.warnings) == 2 and "sdpa_scale" in annotated.warnings[1]
    assert "Unverified" in annotated.warnings[1] and "2.15" in annotated.warnings[1]


@pytest.mark.parametrize(
    "mode,self_test",
    [("enforce", _self_test(sdpa=False)), ("warn", _self_test()), ("warn", None)],
    ids=["enforce", "passed", "not-run"],
)
def test_no_unverified_note_unless_warn_let_a_failure_through(monkeypatch, mode, self_test):
    monkeypatch.setattr(device, "_MPS_SELF_TEST", self_test)
    monkeypatch.setenv(config.ENV_MPS_SELF_TEST, mode)
    result = _result()

    assert _mps(monkeypatch).annotate(result) is result


def test_the_cuda_result_is_never_annotated(monkeypatch, no_probe):
    monkeypatch.setattr(device, "_MPS_SELF_TEST", _self_test(sdpa=False))
    monkeypatch.setenv(config.ENV_MPS_SELF_TEST, "warn")
    result = _result()

    assert device.resolve(_real_spaces()).annotate(result) is result


def test_the_entrypoint_annotates_what_it_returns():
    """The note rides on the result itself, the one channel from the worker."""
    import ast

    from conftest import REPO_ROOT

    tree = ast.parse((REPO_ROOT / "app.py").read_text())
    entry = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_generation"
    )
    calls = {
        node.func.attr
        for node in ast.walk(entry)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"annotate", "require_generation", "assert_returnable"} <= calls, calls


def test_mps_generation_checks_numerics_before_any_load(monkeypatch):
    monkeypatch.setattr(device, "_MPS_SELF_TEST", _self_test(sdpa=False))
    monkeypatch.setattr(pipelines, "_snapshot_path", lambda: pytest.fail("loaded first"))
    pipelines.clear_cache()

    with pytest.raises(device.MpsNumericsError):
        pipelines._base_pipeline(device="mps")


def test_the_self_test_detects_what_a_fresh_process_measures():
    """The detector agrees with an independent repro of the torch bug."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device on this machine")
    import subprocess

    repro = (
        "import torch, torch.nn.functional as F\n"
        "g = torch.Generator().manual_seed(1)\n"
        "q, k, v = (torch.randn(1, 2, 32, 16, generator=g) for _ in range(3))\n"
        "errs = []\n"
        "for s in (0.25, 1.0):\n"
        "    ref = F.scaled_dot_product_attention(q, k, v, scale=s)\n"
        "    got = F.scaled_dot_product_attention(q.to('mps'), k.to('mps'), v.to('mps'),"
        " scale=s).cpu()\n"
        "    errs.append(float((got - ref).abs().max() / ref.abs().max()))\n"
        "print(max(errs) > 1e-3)\n"
    )
    bug = subprocess.run([sys.executable, "-c", repro], capture_output=True, text=True, check=True)
    bug_present = bug.stdout.strip() == "True"

    result = device.mps_self_test(force=True)
    checks = {check.name: check for check in result.checks}

    assert checks["sdpa_scale"].ok is (not bug_present), result.describe()
    assert checks["float64"].ok is False, "MPS has no float64; the probe must say so"


def test_the_self_test_passes_on_torch_2_15_or_newer(monkeypatch):
    """Recorded expectation: torch 2.15.0 measures 8.8e-7 on the scale-reuse
    repro (5.19 on 2.11.0), so the local venv for real weights must pass under
    the default ``enforce``."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device on this machine")
    major, minor = (int(part) for part in torch.__version__.split(".")[:2])
    if (major, minor) < (2, 15):
        pytest.skip(f"torch {torch.__version__} predates the fix (needs >= 2.15)")
    monkeypatch.setenv(config.ENV_MPS_SELF_TEST, "enforce")

    result = device.mps_self_test(force=True)

    assert result.ok, result.describe()
    assert device.require_mps_numerics().ok


# --------------------------------------------------------------------------
# RoPE on MPS: the float64 frequencies are computed on the CPU
# --------------------------------------------------------------------------


def _rope_modules():
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed
    from diffusers.pipelines.ltx2.connectors import LTX2RotaryPosEmbed1d

    return LTX2AudioVideoRotaryPosEmbed(dim=64, num_attention_heads=4), LTX2RotaryPosEmbed1d(dim=64)


def test_rope_handling_keeps_double_precision_and_matches_the_cpu_exactly():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device on this machine")
    video_rope, text_rope = _rope_modules()
    coords = video_rope.prepare_video_coords(1, 3, 4, 4, device="cpu")
    ref_video = video_rope(coords, device="cpu")
    ref_text = text_rope(1, 12, device="cpu")

    handled = device.compute_rope_on_cpu(("video", video_rope), ("text", text_rope))

    assert set(handled) == {"video", "text"}
    assert video_rope.double_precision and text_rope.double_precision
    out_video = video_rope(coords.to("mps"), device=torch.device("mps"))
    out_text = text_rope(1, 12, device=torch.device("mps"))
    for out, ref in ((out_video, ref_video), (out_text, ref_text)):
        assert all(t.device.type == "mps" for t in out)
        assert all(torch.equal(o.cpu(), r) for o, r in zip(out, ref, strict=True))


def test_the_rope_handling_is_idempotent_and_takes_any_shape_of_target():
    """The local runner reloads the transformer between calls and re-applies
    this after every load; a module already handled must not be hooked twice."""
    torch = pytest.importorskip("torch")
    video_rope, text_rope = _rope_modules()
    wrapper = torch.nn.Module()
    wrapper.rope = video_rope
    pipeline = types.SimpleNamespace(components={"transformer": wrapper, "tokenizer": "tok"})

    first = device.compute_rope_on_cpu(pipeline)
    hooks = (len(video_rope._forward_pre_hooks), len(video_rope._forward_hooks))
    again = device.compute_rope_on_cpu(pipeline)
    bare = device.compute_rope_on_cpu(text_rope)

    assert first == again == ("transformer.rope",)
    assert (len(video_rope._forward_pre_hooks), len(video_rope._forward_hooks)) == hooks == (1, 1)
    assert bare == ("LTX2RotaryPosEmbed1d",)


def test_the_cuda_path_never_gets_the_rope_handling(monkeypatch, tmp_path):
    from test_stage_graph_contract import install_fake_stack

    install_fake_stack(monkeypatch, tmp_path)
    monkeypatch.setattr(
        device, "compute_rope_on_cpu", lambda *a: pytest.fail("rope moved off CUDA's path")
    )
    pipelines.clear_cache()
    try:
        pipelines._base_pipeline(device="cuda")
    finally:
        pipelines.clear_cache()


# --------------------------------------------------------------------------
# start-up on mps/cpu checks what the local runner loads (review local-runtime-1)
#
# The strict start-up check demanded the Space's whole allow-list (safetensors
# transformer shards and the stage-2 parts), so the app could not start on a Mac
# holding the GGUF layout the local runner is built for.
# --------------------------------------------------------------------------

#: What the GGUF layout on a Mac leaves out: the transformer shards and the stage-2 parts.
NOT_IN_THE_GGUF_LAYOUT = (
    "transformer/diffusion_pytorch_model-0",
    "latent_upsampler/",
    "temporal_latent_upsampler/",
    "diffusion_decoder/",
)


def _gguf_layout(monkeypatch, tmp_path, *, drop=()):
    """A pinned HF cache with the GGUF layout, plus the GGUF file; returns the config."""
    from ltx25_studio import model_files, preflight

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv(config.ENV_MODEL_SNAPSHOT_DIR, raising=False)
    import json

    repo = preflight.cache_repo_dir()
    allow = model_files.download_allow_list()
    for path in allow:
        if path.startswith(NOT_IN_THE_GGUF_LAYOUT) or path in drop:
            continue
        target = model_files.local_path(path, repo_cache=repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = b"x"
        if path.endswith(".safetensors.index.json"):
            folder = path.rsplit("/", 1)[0]
            shards = [p.rsplit("/", 1)[1] for p in allow if p.startswith(folder + "/")]
            weight_map = {f"w{i}": name for i, name in enumerate(shards) if name.endswith("s")}
            content = json.dumps({"weight_map": weight_map}).encode()
        elif path.endswith(".json"):
            content = b"{}"
        target.write_bytes(content)
    gguf = tmp_path / "x-Q8_0.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setenv(config.ENV_TRANSFORMER_GGUF, str(gguf))
    return config.load_config()


def _resolution(name):
    resolution = device.resolve(_real_spaces(), config.load_config())
    import dataclasses

    return dataclasses.replace(resolution, device=name)


@pytest.mark.parametrize("name", ["cpu", "mps"])
def test_a_local_device_starts_with_the_gguf_layout(monkeypatch, tmp_path, name):
    pytest.importorskip("diffusers")
    from ltx25_studio import preflight

    runtime = _gguf_layout(monkeypatch, tmp_path)
    monkeypatch.setattr(device, "mps_self_test", lambda *a, **k: None)

    assert preflight.run(runtime, strict=True, resolution=_resolution(name)) == "verified"


def test_a_local_device_still_refuses_a_file_the_local_runner_needs(monkeypatch, tmp_path):
    pytest.importorskip("diffusers")
    from ltx25_studio import local_runtime, preflight

    runtime = _gguf_layout(monkeypatch, tmp_path, drop=("vae/diffusion_pytorch_model.safetensors",))

    with pytest.raises(local_runtime.LocalFilesMissingError) as caught:
        preflight.run(runtime, strict=True, resolution=_resolution("cpu"))
    assert "vae" in str(caught.value)


def test_a_local_device_without_the_gguf_needs_the_safetensors_transformer(monkeypatch, tmp_path):
    pytest.importorskip("diffusers")
    from ltx25_studio import local_runtime, preflight

    runtime = _gguf_layout(monkeypatch, tmp_path)
    monkeypatch.delenv(config.ENV_TRANSFORMER_GGUF)
    runtime = config.load_config()

    with pytest.raises(local_runtime.LocalFilesMissingError) as caught:
        preflight.run(runtime, strict=True, resolution=_resolution("cpu"))
    assert "transformer" in str(caught.value)


def test_cuda_still_demands_the_whole_allow_list(monkeypatch, tmp_path):
    pytest.importorskip("diffusers")
    from ltx25_studio import preflight
    from ltx25_studio.errors import MissingModelFileError

    runtime = _gguf_layout(monkeypatch, tmp_path)

    with pytest.raises(MissingModelFileError):
        preflight.run(runtime, strict=True, resolution=_resolution("cuda"))
