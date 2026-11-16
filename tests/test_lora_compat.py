"""Spec for LoRA validator stages 5.5 and 6.5 — PLAN §4.3 (amended), §4.5.

These two stages exist because **stage 7 is necessary but not sufficient.**

* ``charlied98/ltxv_2b_lora_soft_draw_v1`` — an LTX-**Video-2B** adapter whose
  module names are the ones LTX-2.5 uses; its ``to_k.lora_A`` is ``[32, 2048]``
  where this model is 4096. Real peft refuses it at load, inside the billed GPU
  window. **Stage 6.5 refuses it at validate time.**
* ``DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer`` — an in-context LoRA. On the
  plain pipeline it will not error and will not work. **Stage 5.5.**

The contract:

* ``validate_lora(source, ...)`` runs stage 6.5 against the checked-in shape
  manifest of the pinned transformer — no transformer object, no weights. Keys
  are resolved the way diffusers resolves them (native ``adaln_single`` ->
  ``time_embed`` and the rest of its rename) before the lookup.
* A live ``transformer=``, when given, is cross-checked against the manifest; a
  disagreement is ``LoRAManifestDriftError``. ``manifest=None`` drops the
  manifest and checks against the live transformer alone (or skips, saying so,
  when there is none).
* A dimension mismatch raises ``LoRADimensionError`` naming **module, expected,
  actual** for the first mismatch only. Modules that do not exist after the
  rename are a **distinct** failure past the resolved-fraction threshold.
"""

from __future__ import annotations

import pytest
from lora_fixtures import (
    DIFFUSERS_PREFIX,
    NATIVE_PREFIX,
    REAL_MODULES,
    adapter_tensors,
    diffusers_convention_adapter,
    good_adapter,
    manifest_dims,
    manifest_transformer,
    native_convention_adapter,
    overwrite_weights,
    partial_coverage_adapter,
    renamed_modules,
    write_adapter,
    wrong_dimension_adapter,
)

from ltx25_studio import errors
from ltx25_studio.lora import manifest as lora_manifest
from ltx25_studio.lora import validate as lora_validate


def _error(name):
    cls = getattr(errors, name, None)
    assert cls is not None, f"ltx25_studio.errors.{name} does not exist (PLAN §4.3, amended)"
    return cls


def _absent_warnings(report):
    return [w for w in report.warnings if "absent" in w]


# --------------------------------------------------------------------------
# stage 5.5 — IC-LoRAs are refused, not loaded
# --------------------------------------------------------------------------


def test_an_ic_lora_is_refused(tmp_path):
    """``reference_downscale_factor`` in ``__metadata__`` marks an in-context LoRA."""
    path = write_adapter(
        tmp_path / "colorizer.safetensors",
        adapter_tensors(REAL_MODULES),
        {"reference_downscale_factor": "1"},
    )
    with pytest.raises(_error("LoRAInContextError")) as excinfo:
        lora_validate.validate_lora(path)
    message = str(excinfo.value)
    assert "reference_downscale_factor" in message, message


@pytest.mark.parametrize(
    "name",
    [
        "LTX-2.3-22b-IC-LoRA-Colorizer-0.9.safetensors",
        "my_ic_lora_v2.safetensors",
        "ic-lora-depth.safetensors",
        "Ripple IC-LoRA.safetensors",
    ],
)
def test_an_ic_lora_with_empty_metadata_is_refused_by_its_file_name(name, tmp_path):
    """The upload route has no card; the file name is the only extra signal."""
    path = write_adapter(tmp_path / name, adapter_tensors(REAL_MODULES), {})
    with pytest.raises(_error("LoRAInContextError")) as excinfo:
        lora_validate.validate_lora(path, upload=True)
    assert name in str(excinfo.value)


@pytest.mark.parametrize("name", ["music-lora.safetensors", "basic_lora.safetensors"])
def test_names_that_merely_contain_the_letters_are_not_ic(name, tmp_path):
    path = write_adapter(tmp_path / name, adapter_tensors(REAL_MODULES), {})
    assert lora_validate.validate_lora(path, upload=True) is not None


def _hub_fetcher(path, *, repo_id="WepeNerd/LTX-Ripple", tags=(), base_models=()):
    from ltx25_studio.lora.fetch import FetchedLoRA

    def fetcher(source):
        return FetchedLoRA(
            path=path,
            repo_id=repo_id,
            filename=path.name,
            revision="0" * 40,
            tags=tuple(tags),
            base_models=tuple(base_models),
        )

    return fetcher


def test_an_ic_lora_is_refused_by_its_hub_card_tag(tmp_path):
    """``WepeNerd/LTX-Ripple``: empty metadata, tagged ``ic-lora`` on its card."""
    path = write_adapter(
        tmp_path / "LTX25_Ripple_v11.safetensors", adapter_tensors(REAL_MODULES), {}
    )
    fetcher = _hub_fetcher(path, tags=("comfyui", "lora", "ic-lora", "ltx-2.5"))
    with pytest.raises(_error("LoRAInContextError")) as excinfo:
        lora_validate.validate_lora("WepeNerd/LTX-Ripple", upload=False, fetcher=fetcher)
    assert "ic-lora" in str(excinfo.value)


def test_an_ic_lora_is_refused_by_its_repo_name(tmp_path):
    path = write_adapter(tmp_path / "colorizer.safetensors", adapter_tensors(REAL_MODULES), {})
    fetcher = _hub_fetcher(path, repo_id="someone/LTX-2.3-IC-LoRA-Colorizer")
    with pytest.raises(_error("LoRAInContextError")):
        lora_validate.validate_lora(
            "someone/LTX-2.3-IC-LoRA-Colorizer", upload=False, fetcher=fetcher
        )


def test_an_ic_lora_is_refused_by_its_base_model(tmp_path):
    path = write_adapter(tmp_path / "derived.safetensors", adapter_tensors(REAL_MODULES), {})
    fetcher = _hub_fetcher(
        path, repo_id="someone/derived", base_models=("Lightricks/LTX-2.5-22b-IC-LoRA-Depth",)
    )
    with pytest.raises(_error("LoRAInContextError")):
        lora_validate.validate_lora("someone/derived", upload=False, fetcher=fetcher)


def test_a_normal_hub_adapter_is_not_mistaken_for_an_ic_lora(tmp_path):
    path = good_adapter(tmp_path, "BTSMovieSet_LTX25_v1.safetensors")
    fetcher = _hub_fetcher(
        path,
        repo_id="SOLRICKS/LTX-2.5-BTS-Movie-Set",
        tags=("lora", "ltx-2.5"),
        base_models=("Lightricks/LTX-2.5",),
    )
    report = lora_validate.validate_lora(
        "SOLRICKS/LTX-2.5-BTS-Movie-Set", upload=False, fetcher=fetcher
    )
    assert report.hub_repo == "SOLRICKS/LTX-2.5-BTS-Movie-Set"


# --------------------------------------------------------------------------
# stage 6.5 — runs at validate time, against the manifest, rename-aware
# --------------------------------------------------------------------------


def test_a_2b_adapter_is_rejected_without_any_transformer(tmp_path):
    """THE case stage 6.5 exists for, now caught where the UI validates."""
    with pytest.raises(_error("LoRADimensionError")) as excinfo:
        lora_validate.validate_lora(wrong_dimension_adapter(tmp_path))
    message = str(excinfo.value)
    assert "transformer_blocks.0.attn1.to_k" in message, message
    assert "4096" in message and "2048" in message, message


def test_the_dimension_error_names_the_first_mismatch_only(tmp_path):
    with pytest.raises(_error("LoRADimensionError")) as excinfo:
        lora_validate.validate_lora(wrong_dimension_adapter(tmp_path))
    message = str(excinfo.value)
    assert message.count("transformer_blocks.") == 2, message  # the module and its key
    assert "attn1.to_q" not in message and "attn1.to_v" not in message, message


def test_a_full_native_adapter_resolves_every_module_without_false_absences(tmp_path):
    """Before the rename, the 26 native-named modules (adaln_single, patchify_proj …)
    were reported absent: 26 of 1660, "a normal version difference" — false."""
    report = lora_validate.validate_lora(native_convention_adapter(tmp_path))
    assert report.modules_targeted == report.modules_resolved == 1660
    assert _absent_warnings(report) == [], report.warnings


def test_a_full_diffusers_convention_adapter_resolves_every_module(tmp_path):
    report = lora_validate.validate_lora(diffusers_convention_adapter(tmp_path))
    assert report.modules_targeted == report.modules_resolved == 1660
    assert _absent_warnings(report) == [], report.warnings


def test_an_adapter_of_only_renamed_modules_is_not_refused(tmp_path):
    """26 adaLN/projection modules, every one renamed by diffusers: 100 % resolved,
    where the old prefix-strip would have refused it as 0 % resolved."""
    path = write_adapter(
        tmp_path / "adaln-only.safetensors",
        adapter_tensors(renamed_modules(), prefix=NATIVE_PREFIX),
        {},
    )
    report = lora_validate.validate_lora(path)
    assert report.modules_resolved == report.modules_targeted == 26


def test_the_rename_does_not_hide_a_wrong_dimension_at_a_renamed_module(tmp_path):
    path = write_adapter(
        tmp_path / "adaln-wrong.safetensors",
        adapter_tensors(("time_embed.linear",), dims={"time_embed.linear": (256, 4096)}),
        {},
    )
    with pytest.raises(_error("LoRADimensionError")) as excinfo:
        lora_validate.validate_lora(path)
    assert "time_embed.linear" in str(excinfo.value)


def test_a_two_stream_adapter_is_accepted(tmp_path):
    """2048-wide tensors at the audio modules, where 2048 is right."""
    audio = (
        "transformer_blocks.0.audio_attn1.to_q",
        "transformer_blocks.0.audio_ff.net.0.proj",
        "transformer_blocks.0.audio_ff.net.2",
    )
    path = write_adapter(tmp_path / "two-stream.safetensors", adapter_tensors(audio), {})
    assert lora_validate.validate_lora(path).modules_resolved == 3


def test_the_feed_forward_expansion_is_not_mistaken_for_a_mismatch(tmp_path):
    path = write_adapter(
        tmp_path / "ff-only.safetensors",
        adapter_tensors(("transformer_blocks.0.ff.net.2", "transformer_blocks.0.audio_ff.net.2")),
        {},
    )
    assert lora_validate.validate_lora(path) is not None


@pytest.mark.parametrize(
    "present, absent, outcome",
    [(3, 7, "reject"), (7, 3, "warn-suspect"), (9, 1, "warn-normal"), (10, 0, "clean")],
)
def test_partial_coverage_follows_the_resolved_fraction_thresholds(
    present, absent, outcome, tmp_path
):
    path = partial_coverage_adapter(tmp_path, present=present, absent=absent)
    if outcome == "reject":
        with pytest.raises(_error("LoRAModuleNotFoundError")) as excinfo:
            lora_validate.validate_lora(path)
        assert f"{absent} of {present + absent}" in str(excinfo.value)
        return
    report = lora_validate.validate_lora(path)
    assert report.modules_targeted == present + absent
    assert report.modules_resolved == present
    warnings = _absent_warnings(report)
    if outcome == "clean":
        assert warnings == []
    elif outcome == "warn-suspect":
        assert len(warnings) == 1 and "treat unexpected output as suspect" in warnings[0]
    else:
        assert len(warnings) == 1 and "normal version difference" in warnings[0]


def test_absent_modules_are_named_by_their_resolved_name(tmp_path):
    path = partial_coverage_adapter(tmp_path, present=9, absent=1)
    (warning,) = _absent_warnings(lora_validate.validate_lora(path))
    assert "transformer_blocks.48.attn1.to_q" in warning


def test_transformer_keys_diffusers_will_drop_count_as_not_applied(tmp_path):
    tensors = adapter_tensors(REAL_MODULES, prefix=NATIVE_PREFIX)
    tensors |= adapter_tensors(("transformer_blocks.5.attn1.to_q",), prefix=DIFFUSERS_PREFIX)
    report = lora_validate.validate_lora(write_adapter(tmp_path / "mixed.safetensors", tensors, {}))
    assert report.modules_targeted == len(REAL_MODULES) + 1
    assert report.modules_resolved == len(REAL_MODULES)


# --------------------------------------------------------------------------
# a live transformer is cross-checked against the manifest
# --------------------------------------------------------------------------


def test_a_live_transformer_that_agrees_with_the_manifest_passes(tmp_path):
    report = lora_validate.validate_lora(good_adapter(tmp_path), transformer=manifest_transformer())
    assert report.modules_resolved == len(REAL_MODULES)


def test_a_live_transformer_that_disagrees_is_reported_as_drift(tmp_path):
    drifted = manifest_transformer(dims={"transformer_blocks.7.ff.net.2": (16384, 2048)})
    with pytest.raises(lora_manifest.LoRAManifestDriftError) as excinfo:
        lora_validate.validate_lora(good_adapter(tmp_path), transformer=drifted)
    assert "transformer_blocks.7.ff.net.2" in str(excinfo.value)


def test_a_live_transformer_missing_a_targeted_module_is_drift(tmp_path):
    dims = manifest_dims()
    del dims["time_embed.linear"]
    from conftest import FakeComponent

    live = FakeComponent(tuple(dims), dims=dims)
    with pytest.raises(lora_manifest.LoRAManifestDriftError):
        lora_validate.validate_lora(good_adapter(tmp_path), transformer=live)


def test_peft_wrapper_modules_on_a_warm_transformer_are_not_drift(tmp_path):
    """After an earlier run, peft leaves ``base_layer`` / ``lora_A.<name>`` children."""
    live = manifest_transformer()
    to_q = live.get_module("transformer_blocks.0.attn1.to_q")
    from conftest import FakeLinear

    to_q._children["base_layer"] = FakeLinear(4096, 4096)
    to_q._children["lora_A"] = FakeLinear(4096, 16)
    assert lora_validate.validate_lora(good_adapter(tmp_path), transformer=live) is not None


def test_without_the_manifest_the_live_transformer_is_the_reference(tmp_path, make_pipeline):
    """``manifest=None``: the walk of the live transformer is the only reference,
    so a 512-wide model accepts a 512-wide adapter and refuses a 4096-wide one."""
    from conftest import TRANSFORMER_MODULE_DIMS

    transformer = make_pipeline(dims={p: (512, 512) for p in TRANSFORMER_MODULE_DIMS}).transformer
    targets = ("transformer_blocks.0.attn1.to_q", "transformer_blocks.0.attn1.to_k")
    matching = write_adapter(
        tmp_path / "odd-matching.safetensors",
        adapter_tensors(targets, dims={t: (512, 512) for t in targets}),
        {},
    )
    assert lora_validate.validate_lora(matching, transformer=transformer, manifest=None)
    mismatched = write_adapter(
        tmp_path / "odd-mismatched.safetensors", adapter_tensors(targets), {}
    )
    with pytest.raises(_error("LoRADimensionError")):
        lora_validate.validate_lora(mismatched, transformer=transformer, manifest=None)


def test_with_neither_manifest_nor_transformer_the_skip_is_said_out_loud(tmp_path):
    report = lora_validate.validate_lora(wrong_dimension_adapter(tmp_path), manifest=None)
    assert any("skipped" in w and "dimension" in w for w in report.warnings), report.warnings


# --------------------------------------------------------------------------
# F54 — distillation adapters are flagged, not refused
# --------------------------------------------------------------------------


def _distill_warnings(report):
    return [w for w in report.warnings if "distill" in w.lower()]


def test_a_full_coverage_high_rank_adapter_warns_as_distillation(tmp_path):
    report = lora_validate.validate_lora(native_convention_adapter(tmp_path, rank=256))
    assert len(_distill_warnings(report)) == 1, report.warnings


def test_a_full_coverage_low_rank_adapter_does_not_warn(tmp_path):
    report = lora_validate.validate_lora(native_convention_adapter(tmp_path, rank=16))
    assert _distill_warnings(report) == [], report.warnings


@pytest.mark.parametrize(
    "name",
    [
        "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        "ltx25_turbo_distill_8step_r64.safetensors",
    ],
)
def test_a_distill_file_name_warns(name, tmp_path):
    report = lora_validate.validate_lora(good_adapter(tmp_path, name))
    assert len(_distill_warnings(report)) == 1, report.warnings


def test_a_distill_repo_name_warns(tmp_path):
    path = good_adapter(tmp_path, "adapter.safetensors")
    fetcher = _hub_fetcher(path, repo_id="lvladikov/LTX-2.5-Stage1-Distill-2step-LoRA")
    report = lora_validate.validate_lora(
        "lvladikov/LTX-2.5-Stage1-Distill-2step-LoRA", upload=False, fetcher=fetcher
    )
    assert len(_distill_warnings(report)) == 1, report.warnings


# --------------------------------------------------------------------------
# what the report carries
# --------------------------------------------------------------------------


def test_the_report_carries_rank_alpha_tensors_and_a_content_key(tmp_path):
    report = lora_validate.validate_lora(good_adapter(tmp_path))
    assert report.rank == 16
    assert report.alpha == "16"
    assert report.num_tensors == 2 * len(REAL_MODULES)
    assert report.targets_connectors is False
    assert len(report.content_key) == 64


def test_the_content_key_ignores_the_file_name_but_not_the_content(tmp_path):
    first = lora_validate.validate_lora(good_adapter(tmp_path, "a.safetensors")).content_key
    renamed = lora_validate.validate_lora(good_adapter(tmp_path, "b.safetensors")).content_key
    other = lora_validate.validate_lora(
        good_adapter(tmp_path, "c.safetensors", metadata={"model_version": "2.5.1"})
    ).content_key
    assert first == renamed != other


# lora-3: one trainer's default config gives different adapters byte-identical
# headers (the recon's aztro, elix3r and tamm5y5m5 adapters share one), so a key
# over the header alone called them all the same adapter.


def _same_layout_pair(tmp_path):
    tensors = adapter_tensors(REAL_MODULES)
    character = write_adapter(tmp_path / "character.safetensors", tensors, {})
    style = overwrite_weights(write_adapter(tmp_path / "style.safetensors", tensors, {}), b"\x3c")
    return character, style


def test_one_layout_with_different_weights_is_two_adapters(tmp_path):
    from ltx25_studio import lora

    character, style = _same_layout_pair(tmp_path)
    first = lora_validate.validate_lora(character, upload=True)
    second = lora_validate.validate_lora(style, upload=True)

    assert first.inspection.header_sha256 == second.inspection.header_sha256, "same layout"
    assert first.content_key != second.content_key
    staged = [lora.staged_entry("character", first)]
    assert lora.find_duplicate(staged, lora.staged_entry("style", second)) is None


def test_the_content_key_is_the_files_sha256_as_the_hub_lists_it(tmp_path):
    """The Hub's LFS hash of a file is its sha256, so a key that equals it is
    comparable with what a repo listing says."""
    import hashlib

    _, style = _same_layout_pair(tmp_path)
    report = lora_validate.validate_lora(style, upload=True)
    assert report.content_key == hashlib.sha256(style.read_bytes()).hexdigest()


def test_the_lora_bay_stages_both_adapters_of_one_layout(tmp_path, monkeypatch):
    from ltx25_studio import lora
    from ltx25_studio.ui import handlers

    monkeypatch.setattr(lora, "lora_cache_dir", lambda: tmp_path / "lora-cache")
    character, style = _same_layout_pair(tmp_path)

    _verdict, staged, _ = handlers.on_validate_lora("", str(character), [])
    verdict, staged, _ = handlers.on_validate_lora("", str(style), staged)

    assert len(staged) == 2, f"the second adapter was refused: {verdict}"
    _verdict, staged, _ = handlers.on_validate_lora("", str(style), staged)
    assert len(staged) == 2, "the same file checked again is still one adapter"


# --------------------------------------------------------------------------
# stage 4, revised — four version signals, in priority order (PLAN §4.5)
# --------------------------------------------------------------------------


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
def test_each_version_signal_counts_as_a_signal(metadata, tmp_path):
    path = write_adapter(tmp_path / "signal.safetensors", adapter_tensors(REAL_MODULES), metadata)
    report = lora_validate.validate_lora(path)
    assert not any("version" in w.lower() for w in report.warnings), report.warnings


def test_a_foreign_architecture_signal_is_rejected_at_stage_four(tmp_path):
    path = write_adapter(
        tmp_path / "flux-lora.safetensors",
        adapter_tensors(REAL_MODULES),
        {"modelspec.architecture": "flux-1-dev/lora"},
    )
    with pytest.raises(errors.LoRAError) as excinfo:
        lora_validate.validate_lora(path)
    assert "flux" in str(excinfo.value).lower(), str(excinfo.value)


@pytest.mark.parametrize("metadata", [None, {}], ids=["metadata_absent", "metadata_empty"])
def test_both_shapes_of_absent_metadata_only_warn(metadata, tmp_path):
    path = write_adapter(
        tmp_path / "no-signal.safetensors", adapter_tensors(REAL_MODULES), metadata
    )
    report = lora_validate.validate_lora(path)
    assert report.warnings
    assert report.model_version is None


# --------------------------------------------------------------------------
# §4.6 — the validator carries no heavy dependencies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["inspect.py", "validate.py", "manifest.py"])
def test_the_header_path_imports_nothing_heavy(module):
    """A bad upload must be rejectable without loading torch. ``keys.py`` may
    import diffusers' converter where it exists, behind a table fallback."""
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
