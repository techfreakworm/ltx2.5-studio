"""Spec for the runtime configuration levers.

**Every lever here must be attached to something.** This file's own history is
the reason: a precision variable once existed, was wired to nothing, and made
the startup report state a footprint nobody was loading (PLAN §2.6). Two more
were removed the same way.

The quantization lever below is the counter-example — it is real, it changes
output, and it exists because bf16 does not fit ZeroGPU's disk.

**The defaults must work with no Space variable set.** A duplicated Space, or
this one after its variables are cleared, runs on the defaults alone; until
2026-11 those were a half-size GPU, a 60 s reservation and a bf16 load that
does not fit the platform's disk cap.
"""

from __future__ import annotations

import pytest

from ltx25_studio import config

# --------------------------------------------------------------------------
# defaults that work without a Space variable
# --------------------------------------------------------------------------


def test_the_default_zerogpu_size_is_xlarge():
    """``auto`` sends no ``size`` and the scheduler then grants ``large`` --
    47.4 GiB measured, half of xlarge. ZeroGPU does not promote by itself."""
    assert config.load_config().zerogpu_size == "xlarge"


@pytest.mark.parametrize("value", ["auto", "large", "xlarge"])
def test_every_size_stays_selectable(monkeypatch, value):
    monkeypatch.setenv(config.ENV_ZEROGPU_SIZE, value)

    assert config.load_config().zerogpu_size == value


def test_the_reservation_has_no_fixed_base_any_more(monkeypatch):
    """The Space ran 180 s by variable from August; the reservation now follows each run's
    size and length (``zerogpu.estimate_duration``), so the variable is read only to say at
    startup that it is ignored, and the one lever left is an optional floor, unset."""
    monkeypatch.delenv(config.ENV_GPU_DURATION_FLOOR_SECONDS, raising=False)
    monkeypatch.setenv(config.ENV_GPU_DURATION_SECONDS, "180")

    settings = config.load_config()

    assert not hasattr(settings, "gpu_duration_seconds")
    assert settings.ignored_gpu_duration_seconds == "180"
    assert settings.gpu_duration_floor_seconds is None
    assert "reservation=estimated" in settings.describe()


# --------------------------------------------------------------------------
# the quantization lever
#
# A real lever, unlike the ones this file records as removed: it changes output
# quality, which is why it is a switch rather than folded into the recipe. An
# int8-vs-bf16 pair at one seed answers "what does int8 cost?" only if the two
# can be run independently.
# --------------------------------------------------------------------------


def test_the_default_quantization_is_auto():
    """``auto`` is int8 on ZeroGPU and bf16 everywhere else; the resolution is
    pinned in ``test_device_path.py`` where the device is known."""
    assert config.load_config().quantization == "auto"


@pytest.mark.parametrize("value", ["bf16", "int8"])
def test_an_explicit_precision_is_honoured(monkeypatch, value):
    monkeypatch.setenv(config.ENV_QUANTIZATION, value)

    assert config.load_config().quantization == value


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


# --------------------------------------------------------------------------
# the device levers
# --------------------------------------------------------------------------


def test_the_device_defaults_to_auto():
    assert config.load_config().device == "auto"


@pytest.mark.parametrize("value", ["auto", "cuda", "mps", "cpu"])
def test_every_device_is_selectable(monkeypatch, value):
    monkeypatch.setenv(config.ENV_DEVICE, value)

    assert config.load_config().device == value


def test_an_unknown_device_is_refused(monkeypatch):
    monkeypatch.setenv(config.ENV_DEVICE, "rocm")

    with pytest.raises(config.ConfigError) as excinfo:
        config.load_config()
    assert "rocm" in str(excinfo.value) and config.ENV_DEVICE in str(excinfo.value)


def test_the_cpu_precision_defaults_to_fp32_the_reference(monkeypatch):
    assert config.load_config().cpu_dtype == "fp32"
    monkeypatch.setenv(config.ENV_CPU_DTYPE, "bf16")
    assert config.load_config().cpu_dtype == "bf16"
    monkeypatch.setenv(config.ENV_CPU_DTYPE, "fp16")
    with pytest.raises(config.ConfigError):
        config.load_config()


def test_the_mps_self_test_is_enforced_by_default(monkeypatch):
    assert config.load_config().mps_self_test == "enforce"
    monkeypatch.setenv(config.ENV_MPS_SELF_TEST, "warn")
    assert config.load_config().mps_self_test == "warn"
    monkeypatch.setenv(config.ENV_MPS_SELF_TEST, "off")
    with pytest.raises(config.ConfigError):
        config.load_config()


def test_residency_defaults_to_staged_on_macos_and_resident_elsewhere(monkeypatch):
    import sys

    expected = "staged" if sys.platform == "darwin" else "resident"
    assert config.load_config().residency == expected
    for value in ("staged", "resident"):
        monkeypatch.setenv(config.ENV_RESIDENCY, value)
        assert config.load_config().residency == value
    monkeypatch.setenv(config.ENV_RESIDENCY, "lazy")
    with pytest.raises(config.ConfigError):
        config.load_config()


def test_the_decoder_policy_defaults_to_auto_and_refuses_anything_else(monkeypatch):
    monkeypatch.delenv(config.ENV_DECODER, raising=False)
    assert config.load_config().decoder == "auto"
    for value in ("auto", "diffusion", "conv"):
        monkeypatch.setenv(config.ENV_DECODER, value)
        assert config.load_config().decoder == value
    monkeypatch.setenv(config.ENV_DECODER, "vae")
    with pytest.raises(config.ConfigError):
        config.load_config()


def test_a_gguf_transformer_is_optional_and_must_be_a_gguf_file(monkeypatch, tmp_path):
    assert config.load_config().transformer_gguf is None

    monkeypatch.setenv(config.ENV_TRANSFORMER_GGUF, str(tmp_path / "missing.gguf"))
    with pytest.raises(config.ConfigError):
        config.load_config()
    other = tmp_path / "transformer.safetensors"
    other.write_bytes(b"")
    monkeypatch.setenv(config.ENV_TRANSFORMER_GGUF, str(other))
    with pytest.raises(config.ConfigError):
        config.load_config()
    gguf = tmp_path / "transformer-Q8_0.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setenv(config.ENV_TRANSFORMER_GGUF, str(gguf))
    assert config.load_config().transformer_gguf == str(gguf)


def test_a_gguf_in_the_hf_cache_is_accepted_through_its_snapshot_link(monkeypatch, tmp_path):
    """The HF cache links ``snapshots/<rev>/x-Q8_0.gguf`` to an extensionless blob."""
    blob = tmp_path / "blobs" / "4f2a9c"
    blob.parent.mkdir()
    blob.write_bytes(b"x")
    link = tmp_path / "snapshots" / "ac3c540c" / "distilled" / "x-Q8_0.gguf"
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)
    monkeypatch.setenv(config.ENV_TRANSFORMER_GGUF, str(link))

    assert config.load_config().transformer_gguf == str(link)


# --------------------------------------------------------------------------
# the model-source override: optional, and validated loudly
# --------------------------------------------------------------------------


def test_no_snapshot_override_by_default():
    assert config.load_config().model_snapshot_dir is None


def test_a_snapshot_override_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path / "nowhere"))

    with pytest.raises(config.ConfigError) as excinfo:
        config.load_config()
    assert config.ENV_MODEL_SNAPSHOT_DIR in str(excinfo.value)


def test_a_snapshot_override_must_be_a_diffusers_layout(monkeypatch, tmp_path):
    """A directory with no ``model_index.json`` is not a pipeline snapshot, and
    finding that out at the first Generate is the late failure §5 forbids."""
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path))

    with pytest.raises(config.ConfigError) as excinfo:
        config.load_config()
    assert "model_index.json" in str(excinfo.value)


def test_a_valid_snapshot_override_is_resolved(monkeypatch, tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path))

    assert config.load_config().model_snapshot_dir == str(tmp_path.resolve())


# --------------------------------------------------------------------------
# the one-line summary says what is running
# --------------------------------------------------------------------------


def test_describe_names_the_quantisation_and_the_device(monkeypatch):
    """The live Space printed ``expected_weights=72.19GB`` on an int8 container
    and never the precision: the summary must name both levers it depends on."""
    monkeypatch.setenv(config.ENV_QUANTIZATION, "int8")
    monkeypatch.setenv(config.ENV_DEVICE, "cpu")

    line = config.load_config().describe()

    assert "quantization=int8" in line, line
    assert "device=cpu" in line, line
    assert "expected_weights" not in line, (
        f"72.19 GB is the download, not what an int8 load holds: {line}"
    )
