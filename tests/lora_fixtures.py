"""LoRA test adapters built from the real transformer's module names and dimensions.

The conftest adapters name ``adaln_single.*`` modules the diffusers transformer
does not have; these are built from the checked-in shape manifest instead, read
here with plain ``json`` so the fixtures do not depend on the code under test.

Payloads are sparse: the header is written and the file is extended to its full
size with ``truncate``, so a 1660-module adapter costs no disk blocks and every
size check sees an honest file length.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "ltx25_studio" / "lora" / "transformer_linear_shapes.json"

NATIVE_PREFIX = "diffusion_model."
DIFFUSERS_PREFIX = "transformer."
NATIVE_CONNECTOR_PREFIX = "text_embedding_projection."
DIFFUSERS_CONNECTOR_PREFIX = "connectors."

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2}

#: diffusers name -> the name a native (``diffusion_model.``) adapter uses for it.
#: The inverse of diffusers' ``_convert_non_diffusers_ltx2_lora_to_diffusers``.
_TO_NATIVE = (
    ("audio_time_embed.", "audio_adaln_single."),
    ("time_embed.", "adaln_single."),
    ("audio_proj_in", "audio_patchify_proj"),
    ("proj_in", "patchify_proj"),
    ("av_cross_attn_video_scale_shift", "av_ca_video_scale_shift_adaln_single"),
    ("av_cross_attn_video_a2v_gate", "av_ca_a2v_gate_adaln_single"),
    ("av_cross_attn_audio_scale_shift", "av_ca_audio_scale_shift_adaln_single"),
    ("av_cross_attn_audio_v2a_gate", "av_ca_v2a_gate_adaln_single"),
    ("audio_prompt_adaln", "audio_prompt_adaln_single"),
    ("prompt_adaln", "prompt_adaln_single"),
)

#: A small, representative set of real modules: both streams, a feed-forward
#: expansion, a cross-modal attention and an adaLN Linear (the renamed family).
REAL_MODULES = (
    "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.attn1.to_k",
    "transformer_blocks.0.attn1.to_v",
    "transformer_blocks.1.attn1.to_q",
    "transformer_blocks.0.ff.net.2",
    "transformer_blocks.0.audio_ff.net.0.proj",
    "transformer_blocks.0.audio_ff.net.2",
    "transformer_blocks.0.audio_to_video_attn.to_q",
    "time_embed.linear",
)

#: The 2B-adapter shape: the video attention at 2048 where this model is 4096.
VIDEO_ATTENTION = (
    "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.attn1.to_k",
    "transformer_blocks.0.attn1.to_v",
)

OFFICIAL_METADATA = {"lora_rank": "16", "lora_alpha": "16", "model_version": "2.5.0"}


def manifest_document() -> dict:
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def manifest_dims() -> dict[str, tuple[int, int]]:
    """``module -> (in_features, out_features)`` for every Linear in the manifest."""
    return {name: (i, o) for name, (o, i) in manifest_document()["linears"].items()}


def to_native(module: str) -> str:
    """The native-convention name of a diffusers module name."""
    for diffusers_name, native_name in _TO_NATIVE:
        if module.startswith(diffusers_name):
            return native_name + module[len(diffusers_name) :]
    return module


def renamed_modules() -> tuple[str, ...]:
    """Manifest modules whose native name differs from the diffusers one."""
    return tuple(name for name in manifest_dims() if to_native(name) != name)


def adapter_tensors(
    modules,
    *,
    prefix=NATIVE_PREFIX,
    rank=16,
    dims=None,
    style="peft",
    dtype="BF16",
    native_names=None,
):
    """``{key: (dtype, shape)}`` for low-rank pairs on ``modules``.

    Dimensions come from the manifest unless ``dims`` overrides a module.
    ``native_names`` defaults to True for the native prefix, so a native adapter
    spells the renamed modules the way real native adapters do.
    """
    real = manifest_dims()
    overrides = dims or {}
    if native_names is None:
        native_names = prefix == NATIVE_PREFIX
    down, up = (
        ("lora_A.weight", "lora_B.weight")
        if style == "peft"
        else ("lora_down.weight", "lora_up.weight")
    )
    tensors = {}
    for module in modules:
        in_features, out_features = overrides.get(module) or real.get(module, (4096, 4096))
        name = to_native(module) if native_names else module
        tensors[f"{prefix}{name}.{down}"] = (dtype, [rank, in_features])
        tensors[f"{prefix}{name}.{up}"] = (dtype, [out_features, rank])
    return tensors


def safetensors_header_bytes(tensors, metadata=None) -> tuple[bytes, int]:
    """``(8-byte prefix + JSON header, payload length)`` for ``tensors``."""
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        size = count * DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    return struct.pack("<Q", len(blob)) + blob, offset


def write_adapter(path, tensors, metadata=None) -> Path:
    """Write a safetensors file whose payload is a sparse hole of the right length."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head, payload = safetensors_header_bytes(tensors, metadata)
    with open(path, "wb") as handle:
        handle.write(head)
        handle.truncate(len(head) + payload)
    return path


# --------------------------------------------------------------------------
# the five adapters the brief names
# --------------------------------------------------------------------------


def good_adapter(directory, name="good-adapter.safetensors", *, metadata=OFFICIAL_METADATA):
    """Native convention, real names, real dims: passes every stage."""
    return write_adapter(Path(directory) / name, adapter_tensors(REAL_MODULES), metadata)


def wrong_dimension_adapter(directory, name="ltxv-2b-adapter.safetensors"):
    """The 2B shape: real module names, 2048 where this model is 4096."""
    tensors = adapter_tensors(
        VIDEO_ATTENTION, dims={module: (2048, 2048) for module in VIDEO_ATTENTION}
    )
    return write_adapter(Path(directory) / name, tensors, {})


def partial_coverage_adapter(directory, *, present=3, absent=7, name="partial-adapter.safetensors"):
    """``present`` real modules plus ``absent`` modules this transformer lacks."""
    real = [f"transformer_blocks.{i}.attn1.to_q" for i in range(present)]
    foreign = [f"transformer_blocks.{48 + i}.attn1.to_q" for i in range(absent)]
    tensors = adapter_tensors(real) | adapter_tensors(
        foreign, dims={module: (4096, 4096) for module in foreign}
    )
    return write_adapter(Path(directory) / name, tensors, {})


def native_convention_adapter(directory, name="native-full.safetensors", *, rank=16, metadata=None):
    """Every Linear, native names (``adaln_single``, ``patchify_proj`` …)."""
    tensors = adapter_tensors(sorted(manifest_dims()), prefix=NATIVE_PREFIX, rank=rank)
    return write_adapter(Path(directory) / name, tensors, metadata or OFFICIAL_METADATA)


def diffusers_convention_adapter(
    directory, name="diffusers-full.safetensors", *, rank=16, metadata=None
):
    """Every Linear, diffusers names under ``transformer.``."""
    tensors = adapter_tensors(sorted(manifest_dims()), prefix=DIFFUSERS_PREFIX, rank=rank)
    return write_adapter(Path(directory) / name, tensors, metadata or OFFICIAL_METADATA)


def manifest_transformer(dims=None):
    """A fake transformer exposing every manifest Linear with its real dims."""
    from conftest import FakeComponent

    real = manifest_dims()
    real.update(dims or {})
    return FakeComponent(tuple(real), dims=real)
