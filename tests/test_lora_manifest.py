"""The transformer shape manifest stage 6.5 checks against, and the key resolution.

Offline: the checked-in table has exactly the 1660 Linears of the pinned
revision, its checksum holds, and the generator turns headers into exactly that
document. FULL env: diffusers' own LTX2 key conversion and the table-driven
rename resolve the same modules. Network (``LTX25_NETWORK_TESTS=1``): the table
re-derived from the Hub's shard headers equals the checked-in one, and both
resolvers agree on real adapter headers.
"""

from __future__ import annotations

import json
import os

import pytest
from lora_fixtures import (
    DIFFUSERS_PREFIX,
    NATIVE_CONNECTOR_PREFIX,
    NATIVE_PREFIX,
    adapter_tensors,
    manifest_dims,
    renamed_modules,
    safetensors_header_bytes,
)

from ltx25_studio import model_files
from ltx25_studio.lora import keys as lora_keys
from ltx25_studio.lora import manifest as lora_manifest

network = pytest.mark.skipif(
    not os.environ.get("LTX25_NETWORK_TESTS"),
    reason="set LTX25_NETWORK_TESTS=1 to re-derive the manifest from the Hub",
)


def _document():
    return json.loads(lora_manifest.MANIFEST_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the checked-in table
# --------------------------------------------------------------------------


def test_the_manifest_lists_exactly_the_1660_linears_of_the_pinned_transformer():
    document = _document()
    assert document["count"] == 1660
    assert len(document["linears"]) == 1660
    assert document["repo_id"] == model_files.REPO_ID
    assert document["revision"] == model_files.REVISION
    assert document["shape"] == "[out_features, in_features]"


def test_the_manifest_checksum_holds_and_is_pinned_in_code():
    document = _document()
    recomputed = lora_manifest.canonical_sha256(document)
    assert recomputed == document["sha256"] == lora_manifest.MANIFEST_SHA256


def test_the_file_is_exactly_what_the_generator_renders():
    """Any hand edit shows up here, not only in the checksum."""
    document = _document()
    assert lora_manifest.MANIFEST_PATH.read_text(encoding="utf-8") == (
        lora_manifest.render_document(document)
    )


def test_the_manifest_matches_the_int8_skip_list_count():
    from test_int8_skip_modules import EXPECTED_LINEARS

    assert lora_manifest.MANIFEST_COUNT == EXPECTED_LINEARS == len(manifest_dims())


def test_known_modules_carry_their_measured_dimensions():
    dims = lora_manifest.load_manifest().dims
    assert dims["transformer_blocks.0.attn1.to_q"] == (4096, 4096)
    assert dims["transformer_blocks.0.ff.net.2"] == (16384, 4096)
    assert dims["transformer_blocks.0.audio_ff.net.0.proj"] == (2048, 8192)
    assert dims["time_embed.linear"] == (4096, 36864)
    assert dims["audio_time_embed.linear"] == (2048, 18432)
    assert dims["proj_in"] == (128, 4096)
    assert "adaln_single.linear" not in dims, "native names never appear in the diffusers model"


def test_a_tampered_manifest_is_refused(tmp_path, monkeypatch):
    document = _document()
    document["linears"]["transformer_blocks.0.attn1.to_q"] = [2048, 4096]
    tampered = tmp_path / "transformer_linear_shapes.json"
    tampered.write_text(lora_manifest.render_document(document), encoding="utf-8")
    monkeypatch.setattr(lora_manifest, "MANIFEST_PATH", tampered)
    lora_manifest.load_manifest.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="checksum"):
            lora_manifest.load_manifest()
    finally:
        lora_manifest.load_manifest.cache_clear()


def test_the_generator_turns_shard_headers_into_this_document():
    """Feed the generator headers in the shape the Hub serves and get the file back."""
    document = _document()
    linears = document["linears"]
    names = sorted(linears)
    shard_of = {
        name: f"diffusion_pytorch_model-0000{1 + i % 4}-of-00004.safetensors"
        for i, name in enumerate(names)
    }
    headers: dict[str, dict] = {}
    for name in names:
        out_features, in_features = linears[name]
        shard = headers.setdefault(shard_of[name], {"__metadata__": {"format": "pt"}})
        shard[f"{name}.weight"] = {"dtype": "BF16", "shape": [out_features, in_features]}
        shard[f"{name}.bias"] = {"dtype": "BF16", "shape": [out_features]}
    headers[shard_of[names[0]]]["scale_shift_table"] = {"dtype": "BF16", "shape": [6, 4096]}

    def read_json(repo_id, filename, revision=None):
        assert (repo_id, revision) == (model_files.REPO_ID, model_files.REVISION)
        return {"weight_map": {f"{n}.weight": shard_of[n] for n in names}}

    def read_header(repo_id, filename, revision=None):
        assert revision == model_files.REVISION
        return headers[filename.removeprefix("transformer/")]

    derived = lora_manifest.derive_manifest(read_json=read_json, read_header=read_header)
    assert derived == document


@network
def test_the_manifest_re_derived_from_the_hub_is_identical():
    derived = lora_manifest.derive_manifest()
    assert derived == _document()


# --------------------------------------------------------------------------
# key resolution — the way diffusers will resolve them
# --------------------------------------------------------------------------


def _keys(tensors):
    return tuple(tensors)


def test_the_26_renamed_modules_resolve_to_their_diffusers_names():
    renamed = renamed_modules()
    assert len(renamed) == 26
    resolution = lora_keys.resolve_keys(
        _keys(adapter_tensors(renamed, prefix=NATIVE_PREFIX)), converter="table"
    )
    assert set(resolution.transformer) == set(renamed)
    assert not resolution.dropped


def test_a_full_native_adapter_resolves_every_manifest_module():
    resolution = lora_keys.resolve_keys(
        _keys(adapter_tensors(sorted(manifest_dims()), prefix=NATIVE_PREFIX)), converter="table"
    )
    assert set(resolution.transformer) == set(manifest_dims())


def test_diffusers_convention_keys_are_not_renamed():
    resolution = lora_keys.resolve_keys(
        _keys(adapter_tensors(("proj_in", "time_embed.linear"), prefix=DIFFUSERS_PREFIX)),
        converter="table",
    )
    assert set(resolution.transformer) == {"proj_in", "time_embed.linear"}


def test_transformer_keys_beside_native_ones_are_dropped_the_way_diffusers_drops_them():
    """diffusers converts the whole file once any key is native, keeping only
    native keys; a ``transformer.`` key in the same file never reaches peft."""
    tensors = adapter_tensors(("transformer_blocks.0.attn1.to_q",), prefix=NATIVE_PREFIX)
    tensors |= adapter_tensors(("transformer_blocks.1.attn1.to_q",), prefix=DIFFUSERS_PREFIX)
    resolution = lora_keys.resolve_keys(_keys(tensors), converter="table")
    assert set(resolution.transformer) == {"transformer_blocks.0.attn1.to_q"}
    assert any("transformer_blocks.1.attn1.to_q" in key for key in resolution.dropped)


def test_connector_keys_resolve_to_the_connectors():
    tensors = adapter_tensors(
        ("aggregate_embed",), prefix=NATIVE_CONNECTOR_PREFIX, dims={"aggregate_embed": (4096, 4096)}
    )
    resolution = lora_keys.resolve_keys(_keys(tensors), converter="table")
    assert set(resolution.connectors) == {"text_proj_in"}
    assert not resolution.transformer


def test_kohya_suffixes_resolve_to_the_same_modules():
    tensors = adapter_tensors(("time_embed.linear",), prefix=NATIVE_PREFIX, style="kohya")
    resolution = lora_keys.resolve_keys(_keys(tensors), converter="table")
    assert set(resolution.transformer) == {"time_embed.linear"}


def _equivalence_cases():
    full_native = adapter_tensors(sorted(manifest_dims()), prefix=NATIVE_PREFIX)
    mixed = adapter_tensors(("transformer_blocks.0.attn1.to_q",), prefix=NATIVE_PREFIX)
    mixed |= adapter_tensors(("transformer_blocks.1.attn1.to_q",), prefix=DIFFUSERS_PREFIX)
    mixed |= adapter_tensors(
        ("aggregate_embed",), prefix=NATIVE_CONNECTOR_PREFIX, dims={"aggregate_embed": (4096, 4096)}
    )
    full_diffusers = adapter_tensors(sorted(manifest_dims()), prefix=DIFFUSERS_PREFIX)
    return {"full-native": full_native, "mixed": mixed, "full-diffusers": full_diffusers}


@pytest.mark.parametrize("case", ["full-native", "mixed", "full-diffusers"])
def test_diffusers_and_the_table_resolve_the_same_modules(case):
    """FULL env only: the table is a copy of diffusers' rename, and this keeps it one."""
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    keys = _keys(_equivalence_cases()[case])
    by_diffusers = lora_keys.resolve_keys(keys, converter="diffusers")
    by_table = lora_keys.resolve_keys(keys, converter="table")
    assert by_diffusers.converter == "diffusers"
    assert by_diffusers.transformer == by_table.transformer
    assert by_diffusers.connectors == by_table.connectors
    assert set(by_diffusers.dropped) == set(by_table.dropped)


def test_the_resolver_falls_back_to_the_table_without_diffusers(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_diffusers(name, *args, **kwargs):
        if name.startswith("diffusers"):
            raise ImportError("no diffusers here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_diffusers)
    resolution = lora_keys.resolve_keys(
        _keys(adapter_tensors(("time_embed.linear",), prefix=NATIVE_PREFIX))
    )
    assert resolution.converter == "table"
    assert set(resolution.transformer) == {"time_embed.linear"}


REAL_HEADERS = (
    ("SOLRICKS/LTX-2.5-BTS-Movie-Set", "BTSMovieSet_LTX25_v1.safetensors"),
    ("joyfox/LTX-2.3-Transition-LORA", "ltx2.3-transition.safetensors"),
    ("Lightricks/LTX-2.5", "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"),
)


@network
@pytest.mark.parametrize("repo_id, filename", REAL_HEADERS)
def test_both_resolvers_agree_on_real_adapter_headers(repo_id, filename):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    if repo_id.startswith("Lightricks/"):
        from huggingface_hub import get_token

        if not get_token():
            pytest.skip("the official adapter is gated; no Hub token is configured")
    header = lora_manifest.read_remote_header(repo_id, filename)
    keys = tuple(k for k in header if k != "__metadata__")
    by_diffusers = lora_keys.resolve_keys(keys, converter="diffusers")
    by_table = lora_keys.resolve_keys(keys, converter="table")
    assert by_diffusers.transformer == by_table.transformer
    assert set(by_diffusers.transformer) <= set(manifest_dims()), (
        f"{repo_id} targets modules absent from the manifest after the rename"
    )


def test_the_fixture_header_writer_round_trips():
    head, payload = safetensors_header_bytes({"a.lora_A.weight": ("BF16", [2, 4])}, {"k": "v"})
    assert payload == 16
    assert json.loads(head[8:])["__metadata__"] == {"k": "v"}
