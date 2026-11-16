"""Spec for the startup report: it says what is actually running.

The live log of the deployed build (2026-11-16) showed, on an int8 container:
``dtype : bf16 (fixed; no precision switch)``, ``expected_weights=72.19GB``,
``spaces=None``, a ``packed 0.0`` disk line measured six seconds before the
platform started packing, and no requested GPU size anywhere. Each of those
lines was read to make a decision. These specs render the report the way
``app.py`` does, against a stand-in for the real ``spaces`` package (which has
no ``__version__``), and read what it says.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
import types

import pytest

from ltx25_studio import config, device, pipelines, preflight, zerogpu


def _real_spaces():
    """The platform package: a plain module, no ``__version__``, not a stub."""
    module = types.ModuleType("spaces")

    def GPU(task=None, *, duration=None, size=None, **kwargs):  # noqa: N802
        return (lambda f: f) if task is None else task

    module.GPU = GPU
    return module


@pytest.fixture
def installed_spaces(monkeypatch):
    """``importlib.metadata`` answers for ``spaces`` as on the Space."""
    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.51.3" if name == "spaces" else real(name),
    )


@pytest.fixture
def int8_xlarge_report(monkeypatch, caplog, installed_spaces):
    monkeypatch.setenv(config.ENV_QUANTIZATION, "int8")
    monkeypatch.setenv(config.ENV_ZEROGPU_SIZE, "xlarge")
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: pytest.fail("CUDA probed on ZeroGPU"))
    spaces = _real_spaces()
    runtime_config = config.load_config()
    resolution = device.resolve(spaces, runtime_config)
    with caplog.at_level(logging.INFO, logger="ltx25_studio"):
        preflight.run(runtime_config, strict=False, spaces_module=spaces, resolution=resolution)
    return "\n".join(record.getMessage() for record in caplog.records)


def test_the_report_states_the_precision_that_was_configured(int8_xlarge_report):
    report = int8_xlarge_report
    assert "fixed; no precision switch" not in report, report
    precision = [line for line in report.splitlines() if "precision" in line]
    assert precision, report
    assert "int8" in precision[0] and "transformer, text_encoder" in precision[0], precision
    assert "bf16" in precision[0], "the components that stay bf16 must be named too"


def test_the_report_does_not_present_the_download_as_the_footprint(int8_xlarge_report):
    assert "expected_weights" not in int8_xlarge_report, int8_xlarge_report


def test_the_report_names_the_installed_spaces_version(int8_xlarge_report):
    assert "0.51.3" in int8_xlarge_report, int8_xlarge_report


def test_no_line_claims_a_post_pack_measurement(int8_xlarge_report):
    """Packing happens at ``demo.launch()``, after every line of this report."""
    assert "after packing" not in int8_xlarge_report, int8_xlarge_report
    packed = [
        line for line in int8_xlarge_report.splitlines() if line.strip().startswith("packed copy")
    ]
    assert packed, int8_xlarge_report
    assert "launch" in packed[0] and not any(ch.isdigit() for ch in packed[0]), (
        f"the packed copy does not exist yet; no number can have been measured: {packed[0]}"
    )


def test_the_report_names_the_requested_size_and_the_reservation(int8_xlarge_report):
    size = [line for line in int8_xlarge_report.splitlines() if "zerogpu size" in line]
    assert size, int8_xlarge_report
    assert "xlarge" in size[0] and "180" in size[0], size[0]


def test_the_report_names_the_device(int8_xlarge_report):
    line = [line for line in int8_xlarge_report.splitlines() if line.strip().startswith("device")]
    assert line and "cuda" in line[0] and "ZeroGPU" in line[0], int8_xlarge_report


def test_the_snapshot_size_is_reported_separately_from_the_download(int8_xlarge_report):
    line = [line for line in int8_xlarge_report.splitlines() if "snapshot on disk" in line]
    assert line, int8_xlarge_report
    assert "download set" in line[0], line[0]


def test_a_ui_only_report_says_how_to_enable_generation(monkeypatch, caplog):
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: False)
    runtime_config = config.load_config()
    resolution = device.resolve(zerogpu.local_spaces_stub(), runtime_config)

    report = pipelines.startup_report(runtime_config, missing_files=(), resolution=resolution)

    assert "UI only" in report and "LTX25_DEVICE" in report, report


def test_an_mps_report_carries_the_self_test(monkeypatch):
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: False)
    monkeypatch.setattr(device, "host_has_mps", lambda: True)
    monkeypatch.setenv(config.ENV_DEVICE, "mps")
    check = device.SelfTestCheck("sdpa_scale", False, "relative error 1.6", blocking=True)
    result = device.MpsSelfTest(torch_version="2.11.0", checks=(check,))
    runtime_config = config.load_config()

    report = pipelines.startup_report(
        runtime_config,
        missing_files=(),
        resolution=device.resolve(None, runtime_config),
        mps_self_test=result,
    )

    line = [line for line in report.splitlines() if "self-test" in line]
    assert line and "FAILED" in line[0] and "sdpa_scale" in line[0], report


# --------------------------------------------------------------------------
# the spaces version
# --------------------------------------------------------------------------


def test_spaces_version_falls_back_to_the_installed_distribution(installed_spaces):
    assert zerogpu.spaces_version(_real_spaces()) == "0.51.3"


def test_spaces_version_prefers_the_module_attribute():
    module = _real_spaces()
    module.__version__ = "9.9.9"
    assert zerogpu.spaces_version(module) == "9.9.9"


def test_spaces_version_is_none_when_nothing_answers(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert zerogpu.spaces_version(_real_spaces()) is None
    assert zerogpu.spaces_version(None) is None
    assert zerogpu.spaces_version(zerogpu.local_spaces_stub()) == "local-stub"


# --------------------------------------------------------------------------
# the decorator line logs VALUES, not key names
# --------------------------------------------------------------------------


def test_the_gpu_kwargs_line_names_the_size_value_and_the_seconds():
    kwargs = {"duration": zerogpu.estimate_duration, "size": "xlarge"}

    line = zerogpu.describe_gpu_kwargs(kwargs, config.load_config())

    assert "size=xlarge" in line and "180 s" in line, line


def test_a_dropped_size_is_said_out_loud(monkeypatch):
    line = zerogpu.describe_gpu_kwargs({"duration": 60}, config.load_config())

    assert "size" in line and "NOT passed" in line, line


def test_auto_says_the_platform_default_applies(monkeypatch):
    monkeypatch.setenv(config.ENV_ZEROGPU_SIZE, "auto")

    line = zerogpu.describe_gpu_kwargs({"duration": 60, "size": None}, config.load_config())

    assert "large" in line and "default" in line, line


def test_the_app_logs_the_requested_size_value(monkeypatch, caplog):
    pytest.importorskip("gradio")
    from ltx25_studio import generate as generate_module

    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: False)
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)
    previous = generate_module.gpu_dispatch()
    sys.modules.pop("app", None)
    try:
        with caplog.at_level(logging.INFO, logger="ltx25_studio"):
            importlib.import_module("app")
    finally:
        generate_module.set_gpu_dispatch(previous)
        sys.modules.pop("app", None)

    lines = [r.getMessage() for r in caplog.records if "spaces.GPU" in r.getMessage()]
    assert lines and "size=xlarge" in lines[0] and "180 s" in lines[0], lines


# --------------------------------------------------------------------------
# which runner a local device will use, and where the transformer comes from
# --------------------------------------------------------------------------


def _runner_line(monkeypatch, device_name):
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: device_name == "cuda")
    monkeypatch.setattr(device, "host_has_mps", lambda: True)
    monkeypatch.setenv(config.ENV_DEVICE, device_name)
    runtime_config = config.load_config()
    report = pipelines.startup_report(
        runtime_config, missing_files=(), resolution=device.resolve(None, runtime_config)
    )
    return next(line for line in report.splitlines() if line.strip().startswith("runner"))


def test_a_local_device_with_real_weights_reports_the_local_runner_and_its_gguf(
    monkeypatch, tmp_path
):
    gguf = tmp_path / "transformer-Q8_0.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setenv(config.ENV_TRANSFORMER_GGUF, str(gguf))
    monkeypatch.setenv(config.ENV_RESIDENCY, "staged")

    line = _runner_line(monkeypatch, "mps")

    assert "local runner" in line and "staged" in line and str(gguf.resolve()) in line, line


def test_a_local_device_without_gguf_reads_the_transformer_from_the_snapshot(monkeypatch):
    line = _runner_line(monkeypatch, "cpu")

    assert "local runner" in line and "snapshot" in line, line


def test_a_snapshot_override_reports_the_resident_stage_runner(monkeypatch, tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path))

    line = _runner_line(monkeypatch, "cpu")

    assert "StageRunner" in line and str(tmp_path.resolve()) in line, line


def test_cuda_reports_the_startup_built_stage_runner(monkeypatch):
    line = _runner_line(monkeypatch, "cuda")

    assert "StageRunner" in line and "CUDA" in line, line
