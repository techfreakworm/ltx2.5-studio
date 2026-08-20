"""Spec for LoRA validator stages 5.5 and 6.5 — PLAN §4.3 (amended), §4.5.

These two stages exist because **stage 7 is necessary but not sufficient.**

Stage 7 asks "did it attach to anything?". Two real, downloadable adapters answer
"yes" and are still wrong:

* ``charlied98/ltxv_2b_lora_soft_draw_v1`` — an LTX-**Video-2B** adapter whose
  module names are the ones LTX-2.5 uses. It attaches. Its ``to_k.lora_A`` is
  ``[32, 2048]`` where this model is 4096. **Stage 6.5 is the only gate that
  catches it.**
* ``DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer`` — an in-context LoRA. On the
  plain pipeline it will not error and will not work. **Stage 5.5.**

And the trap in the other direction, which is why 6.5 must compare *per module*:
the transformer carries two streams, video at 4096 and audio at 2048, each with a
4x feed-forward expansion. "Contains 2048" is not a signal. "2048 at a module
that expects 4096" is. A naive one-hidden-dim rule rejects
``joyfox/LTX-2.3-Transition-LORA``, the most-downloaded LTX LoRA on the Hub.

The contract:

* ``validate_lora(source, ..., transformer=None)``. With a transformer, stage 6.5
  runs against **that transformer's own** ``nn.Linear`` dimensions; without one
  (pre-load UI validation) it is skipped and the report says so.
* A dimension mismatch raises ``LoRADimensionError`` naming **module, expected,
  actual**. A key whose target module does not exist at all is a **distinct**
  failure — that is what a genuinely foreign adapter looks like.
* An IC-LoRA raises ``LoRAInContextError`` explaining why it cannot be applied.
"""

from __future__ import annotations

import pytest
from conftest import (
    NATIVE_PREFIX,
    TRANSFORMER_MODULE_DIMS,
    lora_tensors,
)

from ltx25_studio import errors
from ltx25_studio.lora import validate as lora_validate


def _error(name):
    cls = getattr(errors, name, None)
    assert cls is not None, f"ltx25_studio.errors.{name} does not exist (PLAN §4.3, amended)"
    return cls


# --------------------------------------------------------------------------
# stage 5.5 — IC-LoRAs are refused, not loaded
# --------------------------------------------------------------------------


def test_an_ic_lora_is_refused(ic_lora, fake_transformer):
    """``reference_downscale_factor`` in ``__metadata__`` marks an in-context LoRA.

    Applied to the plain pipeline it neither errors nor works — the exact
    silent-garbage shape, arriving through a door stage 7 does not watch.
    """
    with pytest.raises(_error("LoRAInContextError")):
        lora_validate.validate_lora(ic_lora, transformer=fake_transformer)


def test_the_ic_lora_error_explains_why_rather_than_just_refusing(ic_lora, fake_transformer):
    with pytest.raises(_error("LoRAInContextError")) as excinfo:
        lora_validate.validate_lora(ic_lora, transformer=fake_transformer)
    message = str(excinfo.value)
    assert "reference_downscale_factor" in message, message
    assert len(message) > 40, f"a refusal the user cannot act on is half a bug; got {message!r}"


def test_an_ic_lora_is_refused_even_without_a_transformer(ic_lora):
    """Stage 5.5 is a header check. It must fire on the Hub route too, before any
    download."""
    with pytest.raises(_error("LoRAInContextError")):
        lora_validate.validate_lora(ic_lora)


def test_a_normal_adapter_is_not_mistaken_for_an_ic_lora(good_lora, fake_transformer):
    assert lora_validate.validate_lora(good_lora, transformer=fake_transformer) is not None


# --------------------------------------------------------------------------
# stage 6.5 — per-module dimension compatibility
# --------------------------------------------------------------------------


def test_a_2b_architecture_adapter_is_rejected(wrong_dimension_lora, fake_transformer):
    """THE case stage 6.5 exists for.

    It passes stage 4 (empty metadata — which our own correct absent-is-not-fatal
    rule permits), stage 5 (``diffusion_model.`` prefix), stage 6 (clean pairs),
    diffusers' built-in check, and stage 7 (its module names match ours).
    """
    with pytest.raises(_error("LoRADimensionError")):
        lora_validate.validate_lora(wrong_dimension_lora, transformer=fake_transformer)


def test_the_dimension_error_names_module_expected_and_actual(
    wrong_dimension_lora, fake_transformer
):
    with pytest.raises(_error("LoRADimensionError")) as excinfo:
        lora_validate.validate_lora(wrong_dimension_lora, transformer=fake_transformer)
    message = str(excinfo.value)
    assert "attn1.to_q" in message or "attn1.to_k" in message, message
    assert "4096" in message, f"the expected dim must be named; got {message!r}"
    assert "2048" in message, f"the actual dim must be named; got {message!r}"


def test_a_two_stream_adapter_is_accepted(two_stream_lora, fake_transformer):
    """The false-positive trap.

    This file contains 2048-wide tensors — at the audio-stream modules, where
    2048 is correct. A "reject anything containing 2048" rule rejects the
    most-downloaded LTX LoRA on the Hub.
    """
    assert lora_validate.validate_lora(two_stream_lora, transformer=fake_transformer) is not None


def test_the_feed_forward_expansion_is_not_mistaken_for_a_mismatch(
    write_safetensors, fake_transformer
):
    """``ff.net.2`` is 16384 in and ``audio_ff.net.2`` is 8192 — 4x expansions, not
    errors. A rule built from ``num_attention_heads x attention_head_dim`` alone
    flags both."""
    path = write_safetensors(
        name="ff-only.safetensors",
        tensors=lora_tensors(
            NATIVE_PREFIX,
            ("transformer_blocks.0.ff.net.2", "transformer_blocks.0.audio_ff.net.2"),
        ),
        metadata={},
    )
    assert lora_validate.validate_lora(path, transformer=fake_transformer) is not None


def test_the_expectation_map_comes_from_the_transformer_not_a_hardcoded_table(
    write_safetensors, make_pipeline
):
    """PLAN §4.3: "Build the expectation map by walking the loaded transformer's
    ``nn.Linear`` modules — never from a hardcoded table."

    Against a transformer whose hidden dim is 512, a 512-wide adapter must be
    accepted and a 4096-wide one rejected — the exact opposite of the real
    model's answer. Only a walked map can produce both.
    """
    odd_dims = {path: (512, 512) for path in TRANSFORMER_MODULE_DIMS}
    transformer = make_pipeline(dims=odd_dims).transformer
    targets = ("transformer_blocks.0.attn1.to_q", "transformer_blocks.0.attn1.to_k")

    matching = write_safetensors(
        name="odd-matching.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, targets, dims={p: (512, 512) for p in targets}),
        metadata={},
    )
    assert lora_validate.validate_lora(matching, transformer=transformer) is not None

    mismatched = write_safetensors(
        name="odd-mismatched.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, targets, dims={p: (4096, 4096) for p in targets}),
        metadata={},
    )
    with pytest.raises(_error("LoRADimensionError")):
        lora_validate.validate_lora(mismatched, transformer=transformer)


def test_a_key_targeting_a_module_that_does_not_exist_is_a_distinct_failure(
    wrong_architecture_lora, fake_transformer
):
    """ "A key whose target module does not exist at all is a **distinct** failure
    — that is what a genuinely foreign adapter looks like." A wrong *dimension*
    and a wrong *module* are different diagnoses and get different errors."""
    with pytest.raises(errors.LoRAError) as excinfo:
        lora_validate.validate_lora(wrong_architecture_lora, transformer=fake_transformer)
    assert not isinstance(excinfo.value, _error("LoRADimensionError")), (
        "a missing module is not a dimension mismatch — the user needs to know which"
    )


def test_stage_six_point_five_is_skipped_without_a_transformer(wrong_dimension_lora):
    """Pre-load UI validation has no transformer to compare against. Skipping is
    correct; skipping *silently* is not."""
    report = lora_validate.validate_lora(wrong_dimension_lora)
    assert report.warnings, "a skipped compatibility check must be reported as skipped"
    assert any("dimension" in w.lower() for w in report.warnings), report.warnings


def test_a_dimensionally_correct_adapter_passes_with_a_transformer(good_lora, fake_transformer):
    report = lora_validate.validate_lora(good_lora, transformer=fake_transformer)
    assert report.model_version == "2.5.0"


# --------------------------------------------------------------------------
# stage 4, revised — four version signals, in priority order (PLAN §4.5)
# --------------------------------------------------------------------------


def test_the_sd_scripts_metadata_block_is_read(sd_scripts_lora, fake_transformer):
    """Not one of the three real community LoRAs measured in §4.5 carries
    ``model_version``. Reading only that field makes stage 4 dead code for 100%
    of real uploads."""
    report = lora_validate.validate_lora(sd_scripts_lora, transformer=fake_transformer)
    assert report is not None


@pytest.mark.parametrize(
    "metadata",
    [
        {"model_version": "2.5.0"},
        {"ss_ltx_version": "2.5"},
        {"modelspec.architecture": "LTX2/lora"},
        {"ss_base_model_version": "ltx2_v1"},
    ],
    ids=["model_version", "ss_ltx_version", "modelspec.architecture", "ss_base_model_version"],
)
def test_each_version_signal_counts_as_a_signal(metadata, write_safetensors, fake_transformer):
    path = write_safetensors(
        name="signal.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, ("transformer_blocks.0.attn1.to_q",)),
        metadata=metadata,
    )
    report = lora_validate.validate_lora(path, transformer=fake_transformer)
    assert not any("version" in w.lower() for w in report.warnings), (
        f"{metadata} is a version signal; it should not warn as absent. Warnings: {report.warnings}"
    )


def test_a_foreign_architecture_signal_is_rejected_at_stage_four(
    write_safetensors, fake_transformer
):
    """``modelspec.architecture`` is a better architecture signal than
    ``model_version``: a Flux or Wan LoRA carries a different value, which is the
    wrong-architecture case caught cheaply, before any weights load."""
    path = write_safetensors(
        name="flux-lora.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, ("transformer_blocks.0.attn1.to_q",)),
        metadata={"modelspec.architecture": "flux-1-dev/lora"},
    )
    with pytest.raises(errors.LoRAError) as excinfo:
        lora_validate.validate_lora(path, transformer=fake_transformer)
    assert "flux" in str(excinfo.value).lower(), str(excinfo.value)


@pytest.mark.parametrize("metadata", [None, {}], ids=["metadata_absent", "metadata_empty"])
def test_both_shapes_of_absent_metadata_only_warn(metadata, write_safetensors, fake_transformer):
    """ "metadata **present but empty** is a different shape from metadata
    **absent**, and both mean no version signal." The positive GPU fixture
    (``AhsanHareem/reelbids-ltx25-camera-lora``) is the empty-dict shape."""
    path = write_safetensors(
        name="no-signal.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, ("transformer_blocks.0.attn1.to_q",)),
        metadata=metadata,
    )
    report = lora_validate.validate_lora(path, transformer=fake_transformer)
    assert report.warnings
    assert report.model_version is None


# --------------------------------------------------------------------------
# §4.6 — the validator carries no heavy dependencies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["inspect.py", "validate.py"])
def test_the_header_path_imports_nothing_heavy(module):
    """PLAN §4.6: "A bad upload must be rejectable without loading torch."

    ``struct`` + ``json`` read the whole header. ``apply.py`` is the only module
    in ``lora/`` that may import diffusers or peft, because it is the only one
    that touches a real model.
    """
    import ast

    from conftest import PACKAGE_ROOT

    tree = ast.parse((PACKAGE_ROOT / "lora" / module).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    heavy = imported & {"torch", "diffusers", "transformers", "peft", "safetensors", "numpy"}
    assert not heavy, f"lora/{module} imports {sorted(heavy)} — stages 1-6.5 must stay torch-free"
