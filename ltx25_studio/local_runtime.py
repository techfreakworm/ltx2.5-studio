"""The local runner: the diffusers LTX-2.5 pipelines on mps/cpu, with staged residency.

:class:`LocalStageRunner` IS a ``pipelines.StageRunner``: the stage graph (the
decode policy, stage 2 and its refusals, the audio conditioning, the clip
variation, the run facts) is one piece of code, and this module supplies only
residency, through its one seam (``_around_call``). ``generate.py`` drives both
runners identically.

What differs from the Space path, and why:

* **Staged residency** (default). The text encoder (~24 GB real) and the
  transformer (~23 GB GGUF Q8 / 38 GB bf16) are never resident together. Per
  call: release the transformer, load the text encoder, encode exactly what the
  pipeline would have encoded, release it, load the transformer, then call the
  pipeline with the embeddings instead of the prompt. Embeddings are cached by
  prompt, so a repeated prompt skips the text encoder entirely.
* **GGUF transformer**, through diffusers' own single-file loader, with the 12
  ``prompt_adaln_single`` keys renamed first: the pinned converter lacks that
  rename and would leave those tensors on ``meta``.
* **The checkpoint's float64 RoPE on MPS**, which has no float64: the RoPE
  forward runs on the CPU and only its tables move (``device.compute_rope_on_cpu``),
  applied to the connectors once and to the transformer after every load.
* **The image-conditioning CRF is set explicitly**, because upstream derives it
  from the text encoder, which is absent during a staged denoise.
* **Memory preflight** before every heavy load and before each denoise; after a
  release it judges the gauge once it has settled (``memory.preflight``).
* Stage 2 and the diffusion decoder follow the Space runner's rules: off CUDA
  the decoder runs on the CPU, at very small sizes only
  (``device.check_stage_2_size``, ``pipelines.check_cpu_decoder_size``).

Heavy imports happen inside functions, so the package stays importable without
torch.
"""

from __future__ import annotations

import copy
import gc
import inspect
import itertools
import json
import logging
import re
import time
import traceback
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from . import constraints, memory, pipelines
from .errors import ConstraintError, LoRAError, LTXStudioError, ModelLoadError

__all__ = [
    "EMBEDDING_CACHE_SIZE",
    "LOCAL_DEVICES",
    "RESIDENCIES",
    "SMALL_SET",
    "AdapterNeedsTransformerError",
    "LocalFilesMissingError",
    "LocalStageRunner",
    "PlanStep",
    "TransformerLoadError",
    "UnsupportedLocalDeviceError",
    "build_local_runner",
    "component_files",
    "move_to_device",
    "release_cache",
    "run_files",
    "clear_local_cache",
    "download_command",
    "frames_for_estimate",
    "load_text_encoder",
    "load_transformer",
    "missing_files",
    "preflight_plan",
    "rename_prompt_adaln_keys",
    "resolve_dtype",
    "apply_rope_precision",
    "rope_flags",
    "rope_on_cpu",
    "weight_sizes",
]

_logger = logging.getLogger("ltx25_studio")

STAGED = "staged"
RESIDENT = "resident"
RESIDENCIES = (STAGED, RESIDENT)
LOCAL_DEVICES = ("mps", "cpu")
KINDS = ("condition", "text2video")

#: Loaded once and kept: everything except the staged pair.
SMALL_SET = ("connectors", "vae", "audio_vae", "vocoder", "duration_head", "scheduler", "tokenizer")
_SMALL_WEIGHTS = ("connectors", "vae", "audio_vae", "vocoder", "duration_head")

#: Each entry holds a prompt's embeddings (~0.39 GB each on the real model:
#: 1024 tokens x 188,160 features x bf16), so the cache is kept small.
EMBEDDING_CACHE_SIZE = 4

_WEIGHTS_SUFFIX = ".safetensors"
_PROMPT_ADALN_OLD = "prompt_adaln_single"
_PROMPT_ADALN_NEW = "prompt_adaln"

_HF_CACHE_PATH = re.compile(
    r"models--(?P<org>[^/]+?)--(?P<name>[^/]+)/snapshots/(?P<rev>[^/]+)(?:/(?P<rel>.*))?$"
)

#: This process freed device memory since a judgement last read the gauge, so
#: the next judgement waits for the gauge to settle. Process-wide, as the gauge is.
_freed_since_reading = False


# --------------------------------------------------------------------------
# typed errors
# --------------------------------------------------------------------------


class UnsupportedLocalDeviceError(LTXStudioError):
    """The local runner was asked for a device it does not serve."""

    def __init__(self, *, device, reason=""):
        self.device = device
        detail = f" {reason}" if reason else ""
        super().__init__(
            f"The local runner works on 'mps' (Apple GPU) or 'cpu', not {device!r}.{detail} "
            f"On an NVIDIA GPU (cuda) the app uses its regular pipeline, the one the Space runs."
        )


class TransformerLoadError(ModelLoadError):
    """The transformer did not load completely; continuing would run half a model."""

    def __init__(self, *, reason, key=None):
        self.key = key
        super().__init__(component="transformer", reason=reason)


class LocalFilesMissingError(ModelLoadError):
    """A weight file the local runner needs is not on disk. Nothing is downloaded."""

    def __init__(self, *, missing):
        self.missing = tuple((str(path), command) for path, command in missing)
        path, command = self.missing[0]
        super().__init__(
            component="local weights",
            reason=(
                f"{len(self.missing)} file(s) are not on disk, starting with {path}. "
                f"Nothing is downloaded automatically; fetch it with: {command}"
            ),
        )


class AdapterNeedsTransformerError(LoRAError):
    """A LoRA was attached while the staged runner had released its transformer."""

    def __init__(self):
        super().__init__(
            "This LoRA cannot be attached yet: the local runner loads the transformer only for "
            "denoising, and it is not loaded right now. Load it first (runner.ensure_transformer()), "
            "then attach the LoRA."
        )


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _device_type(device) -> str:
    return str(device).split(":", 1)[0].strip().lower()


_DTYPE_NAMES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


def resolve_dtype(dtype):
    """A ``torch.dtype`` from a dtype or a name such as ``"bf16"``."""
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    name = _DTYPE_NAMES.get(str(dtype).lower().removeprefix("torch."))
    if name is None:
        raise ConstraintError(
            f"dtype must be one of bfloat16, float16 or float32 (or bf16/fp16/fp32), not {dtype!r}"
        )
    return getattr(torch, name)


def _dtype_label(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def apply_rope_precision(name: str, module, device, log=None) -> tuple[str, ...]:
    """Keep the checkpoint's float64 RoPE on ``device``; the one place this is decided.

    MPS has no float64, so there the RoPE forward runs on the CPU and only its
    tables move to the device (``device.compute_rope_on_cpu``: per-instance hooks,
    idempotent, so it must run again on every freshly loaded module). Elsewhere
    nothing changes. Returns the RoPE modules now covered.
    """
    if _device_type(device) != "mps":
        return ()
    from . import device as device_module

    covered = device_module.compute_rope_on_cpu((name, module))
    (log or _logger.info)(
        f"RoPE computed on the CPU in float64 for {len(covered)} {name} module(s)"
    )
    return covered


def rope_on_cpu(pipeline) -> dict[str, bool]:
    """``{module path: hooked}`` for every RoPE module: does it compute on the CPU?"""
    found = {}
    for root in ("connectors", "transformer"):
        module = getattr(pipeline, root, None)
        if module is None:
            continue
        for name, child in module.named_modules():
            if hasattr(child, "double_precision"):
                found[f"{root}.{name}" if name else root] = bool(
                    getattr(child, "_ltx25_rope_cpu", False)
                )
    return found


def rope_flags(pipeline) -> dict[str, bool]:
    """``{module path: double_precision}`` for every RoPE module the pipeline holds."""
    flags = {}
    for root in ("connectors", "transformer"):
        module = getattr(pipeline, root, None)
        if module is None:
            continue
        for name, child in module.named_modules():
            if hasattr(child, "double_precision"):
                flags[f"{root}.{name}" if name else root] = bool(child.double_precision)
    return flags


def rename_prompt_adaln_keys(state_dict: dict) -> tuple[dict, int]:
    """``prompt_adaln_single`` -> ``prompt_adaln`` (covers ``audio_prompt_adaln_single``)."""
    renamed = {}
    count = 0
    for key, value in state_dict.items():
        if _PROMPT_ADALN_OLD in key:
            key = key.replace(_PROMPT_ADALN_OLD, _PROMPT_ADALN_NEW)
            count += 1
        renamed[key] = value
    return renamed, count


# --------------------------------------------------------------------------
# files on disk
# --------------------------------------------------------------------------


def _hf_origin(path) -> tuple[str, str, str] | None:
    match = _HF_CACHE_PATH.search(Path(path).as_posix())
    if not match:
        return None
    repo = f"{match.group('org')}/{match.group('name')}"
    return repo, match.group("rev"), match.group("rel") or ""


def download_command(path, *, component: str | None = None) -> str:
    """The exact ``hf download`` for a file in the HF cache layout."""
    origin = _hf_origin(path)
    if origin is None:
        return f"(not in the Hugging Face cache layout; put the file at {path})"
    repo, revision, relative = origin
    if component:
        return f'hf download {repo} --revision {revision} --include "{component}/*"'
    return f"hf download {repo} {relative} --revision {revision}"


def _component_files(snapshot: Path, component: str) -> tuple[list[Path], int | None]:
    """Weight files a component needs, and the index's declared byte total if any.

    Read off disk (the shard index, else the weight files present) rather than
    off ``model_files``' pinned names, so the tiny checkpoint works too. A
    component with neither is reported by its folder.
    """
    folder = snapshot / component
    if not folder.is_dir():
        return [folder], None
    index = next(iter(sorted(folder.glob(f"*{_WEIGHTS_SUFFIX}.index.json"))), None)
    if index is not None:
        data = json.loads(index.read_text())
        names = sorted(set(data.get("weight_map", {}).values()))
        return [folder / name for name in names], data.get("metadata", {}).get("total_size")
    return sorted(folder.glob(f"*{_WEIGHTS_SUFFIX}")) or [folder], None


def _component_bytes(snapshot: Path, component: str) -> tuple[int, str]:
    if component == "tokenizer":
        path = snapshot / "tokenizer" / "tokenizer.json"
        return (path.stat().st_size if path.is_file() else 0), "files"
    files, declared = _component_files(snapshot, component)
    present = [f for f in files if f.is_file()]
    if len(present) < len(files) and declared:
        return int(declared), "index (files not on disk yet)"
    return sum(f.stat().st_size for f in present), "files"


def weight_sizes(snapshot_dir, transformer_gguf=None) -> dict[str, tuple[int, str]]:
    """``{component: (bytes, source)}`` for every component the runner loads."""
    snapshot = Path(snapshot_dir)
    sizes = {name: _component_bytes(snapshot, name) for name in (*_SMALL_WEIGHTS, "tokenizer")}
    sizes["text_encoder"] = _component_bytes(snapshot, "text_encoder")
    if transformer_gguf is None:
        sizes["transformer"] = _component_bytes(snapshot, "transformer")
    else:
        path = Path(transformer_gguf)
        sizes["transformer"] = (path.stat().st_size, "files") if path.is_file() else (0, "missing")
    return sizes


_CONFIG_FILE = {"scheduler": "scheduler_config.json", "tokenizer": "tokenizer_config.json"}


def missing_files(snapshot_dir, *, transformer_gguf=None) -> list[tuple[Path, str]]:
    """Every file the runner would need that is not on disk, with its download command."""
    snapshot = Path(snapshot_dir)
    missing = []
    index = snapshot / "model_index.json"
    if not index.is_file():
        missing.append((index, download_command(index)))
    for component in (*SMALL_SET, "text_encoder", "transformer"):
        config = snapshot / component / _CONFIG_FILE.get(component, "config.json")
        needed = [config]
        if component in (*_SMALL_WEIGHTS, "text_encoder") or (
            component == "transformer" and transformer_gguf is None
        ):
            needed += _component_files(snapshot, component)[0]
        if component == "tokenizer":
            needed.append(snapshot / "tokenizer" / "tokenizer.json")
        command = download_command(snapshot / component, component=component)
        missing += [(path, command) for path in needed if not path.is_file()]
    if transformer_gguf is not None and not Path(transformer_gguf).is_file():
        missing.append((Path(transformer_gguf), download_command(transformer_gguf)))
    return missing


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

#: Release the file pages behind the weights after about this many bytes moved.
MOVE_CHUNK_BYTES = 2_000_000_000


def component_files(snapshot_dir, component: str) -> list[Path]:
    """The weight files of ``component`` that are on disk (tokenizer: its json)."""
    snapshot = Path(snapshot_dir)
    if component == "tokenizer":
        files = [snapshot / "tokenizer" / "tokenizer.json"]
    else:
        files = _component_files(snapshot, component)[0]
    return [path for path in files if path.is_file()]


def run_files(snapshot_dir, transformer_gguf=None) -> list[Path]:
    """Every weight file a run loads."""
    files = [
        path
        for component in (*_SMALL_WEIGHTS, "tokenizer", "text_encoder")
        for path in component_files(snapshot_dir, component)
    ]
    if transformer_gguf is None:
        files += component_files(snapshot_dir, "transformer")
    elif Path(transformer_gguf).is_file():
        files.append(Path(transformer_gguf))
    return files


def release_cache(files, label: str, log=None) -> int:
    """Give the files' page-cache pages back and log one line."""
    released = memory.release_file_cache(files)
    (log or _logger.info)(f"released {released / memory.GB:.1f} GB of file cache ({label})")
    return released


def move_to_device(
    module, device, *, files=(), label="weights", log=None, chunk_bytes=MOVE_CHUNK_BYTES
):
    """``module.to(device)`` with a bounded transient.

    The weights are mmap'd from their files, and on macOS the touched file pages
    stay committed next to the device copy: one ``.to()`` holds both, 2x at the
    peak. Moving tensor by tensor and releasing the files' pages every
    ``chunk_bytes`` keeps the extra to about one chunk; one release at the end.
    """
    import torch

    target = torch.device(device)
    pending = 0
    moved_any = False
    kept_dtypes: Counter = Counter()

    def move(tensor):
        nonlocal pending, moved_any
        if tensor.device.type == target.type:
            if tensor.is_floating_point():
                kept_dtypes[tensor.dtype] += tensor.numel() * tensor.element_size()
            return tensor
        moved_any = True
        moved = tensor.to(target)
        pending += tensor.numel() * tensor.element_size()
        if pending >= chunk_bytes:
            memory.release_file_cache(files)
            pending = 0
        return moved

    module._apply(move)
    if not moved_any and _are_file_pages(kept_dtypes, files):
        # On the CPU at the file's own dtype the weights ARE the file's mapped
        # pages: releasing them evicts the live model, which the denoise then
        # faults back in uncounted (review local-runtime-9).
        if log is not None:
            log(f"kept {label}'s file pages: they are the weights on {target.type}")
        return module
    release_cache(files, label, log)
    return module


#: torch dtype name -> safetensors dtype tag.
_SAFETENSORS_TAG = {"float32": "F32", "bfloat16": "BF16", "float16": "F16"}


def _are_file_pages(dtypes, files) -> bool:
    """True when most of what stayed in place is at the files' dtype: views of their pages.

    ``dtypes`` maps each dtype left in place to its bytes; a few buffers kept
    in float32 next to converted weights do not make them file pages.
    """
    stored = _files_dtype(files)
    total = sum(dtypes.values())
    if stored is None or not total:
        return False
    same = sum(
        size
        for dtype, size in dtypes.items()
        if _SAFETENSORS_TAG.get(str(dtype).removeprefix("torch.")) == stored
    )
    return same * 2 > total


def _files_dtype(files) -> str | None:
    """The dtype tag of the first tensor in the first readable safetensors file."""
    for path in files:
        path = Path(path)
        if not (path.is_file() and path.name.endswith(_WEIGHTS_SUFFIX)):
            continue
        try:
            with open(path, "rb") as handle:
                length = int.from_bytes(handle.read(8), "little")
                header = json.loads(handle.read(length))
        except (OSError, ValueError):
            continue
        for key, entry in header.items():
            if key != "__metadata__" and isinstance(entry, dict):
                return str(entry.get("dtype"))
    return None


def _clear_tracebacks(exc: BaseException) -> None:
    """Drop the locals held by a failure's frames, so what they loaded can be freed."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__


def load_text_encoder(snapshot_dir, *, dtype, device, log=None):
    """The text encoder named in ``model_index.json``, on ``device``."""
    import transformers

    snapshot = Path(snapshot_dir)
    _, class_name = json.loads((snapshot / "model_index.json").read_text())["text_encoder"]
    try:
        model = getattr(transformers, class_name).from_pretrained(
            snapshot / "text_encoder", dtype=dtype, local_files_only=True
        )
        move_to_device(
            model,
            device,
            files=component_files(snapshot, "text_encoder"),
            label="text_encoder",
            log=log,
        )
    except Exception as exc:
        raise ModelLoadError(component="text_encoder", reason=str(exc)) from exc
    return model.eval()


def _quant_name(value) -> str:
    quant = getattr(value, "quant_type", None)
    if quant is not None:
        return getattr(quant, "name", str(quant))
    return _dtype_label(getattr(value, "dtype", type(value).__name__))


def _expected_transformer_keys(snapshot: Path) -> set[str]:
    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel

    config = LTX2VideoTransformer3DModel.load_config(
        snapshot, subfolder="transformer", local_files_only=True
    )
    with init_empty_weights():
        skeleton = LTX2VideoTransformer3DModel.from_config(config)
    return set(skeleton.state_dict())


def _require_complete(missing, unexpected, log, *, total) -> None:
    missing, unexpected = sorted(missing), sorted(unexpected)
    log(f"transformer keys: {total} loaded, {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        key = (missing or unexpected)[0]
        kind = "has no value in the file" if missing else "is not a parameter of the model"
        raise TransformerLoadError(
            key=key,
            reason=(
                f"the weights do not match the model: {len(missing)} parameter(s) missing and "
                f"{len(unexpected)} unexpected; the first, {key!r}, {kind}. Refusing to run a "
                f"half-loaded transformer"
            ),
        )


def _require_no_meta(model) -> None:
    for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        if tensor.device.type == "meta":
            raise TransformerLoadError(
                key=name,
                reason=f"{name!r} was left without a value (on 'meta'). Refusing to run a "
                f"half-loaded transformer",
            )


def load_transformer(snapshot_dir, *, dtype, device, gguf=None, rename=True, log=None):
    """The transformer from the snapshot, or from a GGUF file through diffusers' loader.

    The GGUF route: ``load_gguf_checkpoint`` -> rename the ``prompt_adaln_single``
    keys -> diffusers' own LTX-2 key conversion -> ``from_single_file`` with the
    snapshot's config and ``compute_dtype`` explicit (its default is float32).
    Either route must load every parameter: no ``meta`` left, 0 missing, 0
    unexpected, or :class:`TransformerLoadError` names the first offending key.
    """
    from diffusers import LTX2VideoTransformer3DModel

    log = log or _logger.info
    snapshot = Path(snapshot_dir)
    try:
        if gguf is None:
            model, info = LTX2VideoTransformer3DModel.from_pretrained(
                snapshot,
                subfolder="transformer",
                torch_dtype=dtype,
                local_files_only=True,
                output_loading_info=True,
            )
            _require_complete(
                info.get("missing_keys", ()),
                info.get("unexpected_keys", ()),
                log,
                total=len(model.state_dict()),
            )
        else:
            model = _load_gguf_transformer(snapshot, Path(gguf), dtype, rename, log)
    except LTXStudioError:
        raise
    except Exception as exc:
        raise TransformerLoadError(reason=str(exc)) from exc
    _require_no_meta(model)
    files = [Path(gguf)] if gguf is not None else component_files(snapshot, "transformer")
    try:
        move_to_device(model, device, files=files, label="transformer", log=log)
    except Exception as exc:
        raise TransformerLoadError(reason=f"moving it to {device} failed: {exc}") from exc
    apply_rope_precision("transformer", model, device, log)
    return model.eval()


def _load_gguf_transformer(snapshot: Path, path: Path, dtype, rename: bool, log):
    from diffusers import GGUFQuantizationConfig, LTX2VideoTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers
    from diffusers.models.model_loading_utils import load_gguf_checkpoint

    state = load_gguf_checkpoint(str(path))
    # Every tensor is now a copy in anonymous memory; the file's pages are not needed.
    release_cache([path], "transformer GGUF, after reading", log)
    if rename:
        state, count = rename_prompt_adaln_keys(state)
        log(f"transformer GGUF: renamed {count} prompt_adaln_single key(s) to prompt_adaln")
    state = convert_ltx2_transformer_to_diffusers(state)
    histogram = Counter(_quant_name(value) for value in state.values())
    log(
        "transformer GGUF quant types: "
        + ", ".join(f"{name} x {count}" for name, count in sorted(histogram.items()))
    )
    loaded = set(state)
    expected = _expected_transformer_keys(snapshot)
    _require_complete(expected - loaded, loaded - expected, log, total=len(loaded))
    model = LTX2VideoTransformer3DModel.from_single_file(
        state,
        config=str(snapshot),
        subfolder="transformer",
        torch_dtype=dtype,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        local_files_only=True,
    )
    after = set(model.state_dict())
    if after != loaded:
        _require_complete(after - loaded, loaded - after, log, total=len(loaded))
    return model


# --------------------------------------------------------------------------
# LoRA adapters across a release
# --------------------------------------------------------------------------


def _capture_adapters(transformer):
    """Everything needed to put the attached adapters back on a fresh transformer."""
    config = getattr(transformer, "peft_config", None)
    if not config:
        return None
    from peft.utils import get_peft_model_state_dict

    adapters = []
    for name, adapter_config in config.items():
        state = get_peft_model_state_dict(transformer, adapter_name=name)
        adapters.append(
            (
                name,
                copy.deepcopy(adapter_config),
                {key: value.detach().to("cpu", copy=True) for key, value in state.items()},
            )
        )
    scaling = {
        name: dict(module.scaling)
        for name, module in transformer.named_modules()
        if isinstance(getattr(module, "scaling", None), dict) and module.scaling
    }
    return {"adapters": adapters, "scaling": scaling, "active": list(transformer.active_adapters())}


def _restore_adapters(transformer, snapshot) -> None:
    """Re-attach captured adapters bit-exactly, or refuse."""
    import torch
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    for name, adapter_config, state in snapshot["adapters"]:
        transformer.add_adapter(copy.deepcopy(adapter_config), adapter_name=name)
        dtype = next(iter(state.values())).dtype if state else None
        for module in transformer.modules():
            for attribute in ("lora_A", "lora_B"):
                layers = getattr(module, attribute, None)
                if layers is not None and name in layers and dtype is not None:
                    layers[name].to(dtype=dtype)
        set_peft_model_state_dict(transformer, state, adapter_name=name)
        restored = get_peft_model_state_dict(transformer, adapter_name=name)
        for key, value in state.items():
            if key not in restored or not torch.equal(restored[key].detach().cpu(), value):
                raise LoRAError(
                    f"the LoRA adapter {name!r} could not be put back after the transformer was "
                    f"reloaded ({key} differs). Refusing to generate without it."
                )
    transformer.set_adapters(snapshot["active"])
    for name, module in transformer.named_modules():
        if name in snapshot["scaling"]:
            module.scaling.update(snapshot["scaling"][name])


# --------------------------------------------------------------------------
# the dry-run plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanStep:
    phase: str
    resident_before_gb: float
    need_gb: float


def frames_for_estimate(call: dict, defaults: dict) -> int:
    """The call's frame count, or the longest the duration head may choose."""
    frames = call.get("num_frames")
    if frames:
        return int(frames)
    seconds = float(call.get("max_seconds", defaults.get("max_seconds", 20.0)))
    shortest = float(call.get("min_seconds", defaults.get("min_seconds", 1.0)))
    rate = float(call.get("frame_rate", defaults.get("frame_rate", 24.0)))
    # The duration head ROUNDS seconds x rate (review local-runtime-10): the
    # longest clip it can return, not one latent frame short of it.
    _shortest, longest = constraints.duration_head_frame_range(
        min_seconds=min(shortest, seconds),
        max_seconds=seconds,
        frame_rate=rate,
        temporal_compression_ratio=memory.TEMPORAL_COMPRESSION,
    )
    return max(1, longest)


def preflight_plan(
    snapshot_dir,
    *,
    transformer_gguf=None,
    residency=STAGED,
    width: int,
    height: int,
    num_frames: int,
    dtype_bytes: int = 2,
) -> list[PlanStep]:
    """The loads and the denoise of one first call, with what each adds. Loads nothing."""
    spike = memory.load_spike_factor("safetensors")
    transformer_spike = memory.load_spike_factor("gguf" if transformer_gguf else "safetensors")
    sizes = {
        name: size / memory.GB
        for name, (size, _) in weight_sizes(snapshot_dir, transformer_gguf).items()
    }
    snapshot = Path(snapshot_dir)

    def need(name, factor):
        return _load_need_gb(sizes[name], factor, dtype_bytes, _stored_dtype_bytes(snapshot, name))

    small = sum(sizes[name] for name in (*_SMALL_WEIGHTS, "tokenizer"))
    small_need = sizes["tokenizer"] * spike + sum(need(name, spike) for name in _SMALL_WEIGHTS)
    encoder, transformer = sizes["text_encoder"], sizes["transformer"]
    encoder_need = need("text_encoder", spike)
    transformer_need = (
        transformer * transformer_spike if transformer_gguf else need("transformer", spike)
    )
    activations = memory.activation_estimate_gb(width, height, num_frames, dtype_bytes=dtype_bytes)
    denoise = f"denoise {int(width)}x{int(height)}x{int(num_frames)}"
    steps = [PlanStep("load small set", 0.0, small_need)]
    if residency == STAGED:
        steps += [
            PlanStep("load text encoder", small, encoder_need),
            PlanStep("load transformer", small, transformer_need),
            PlanStep(denoise, small + transformer, activations),
        ]
    else:
        steps += [
            PlanStep("load transformer", small, transformer_need),
            PlanStep("load text encoder", small + transformer, encoder_need),
            PlanStep(denoise, small + transformer + encoder, activations),
        ]
    return steps


#: Bytes per element of a safetensors dtype tag.
_SAFETENSORS_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2}


def _stored_dtype_bytes(snapshot: Path, component: str) -> int:
    """Bytes per element the component is stored at, read from a safetensors header (else 2)."""
    files, _declared = _component_files(Path(snapshot), component)
    return _tag_bytes(_files_dtype(files))


def _tag_bytes(tag: str | None) -> int:
    """Bytes per element of a safetensors dtype tag (2, the checkpoint's bf16, when unknown)."""
    return 2 if tag is None else _SAFETENSORS_BYTES.get(tag, 1)


def _adapter_bytes(captured) -> int:
    """Bytes the captured adapters take once re-created on the device."""
    if not captured:
        return 0
    return sum(
        value.numel() * value.element_size()
        for _name, _config, state in captured["adapters"]
        for value in state.values()
    )


def _load_need_gb(file_gb: float, spike: float, target_bytes: int, stored_bytes: int) -> float:
    """What loading a file of ``file_gb`` commits: converted copies plus the read pages.

    A load at the stored dtype keeps the file's pages (``spike``); converting to
    another dtype copies every tensor at the target size while the pages it read
    stay active until the move releases them (review local-runtime-3).
    """
    if int(target_bytes) == int(stored_bytes):
        return file_gb * spike
    return file_gb * spike * (target_bytes / stored_bytes) + file_gb


# --------------------------------------------------------------------------
# the shared weights behind every runner of one configuration
# --------------------------------------------------------------------------


class _LocalWeights:
    """The small set, the staged pair, the embedding cache: shared by both kinds."""

    def __init__(
        self, *, device, dtype, snapshot_dir, transformer_gguf, residency, memory_limit_gb, log
    ):
        import torch

        self.device = device
        self.dtype = dtype
        self.dtype_bytes = torch.empty((), dtype=dtype).element_size()
        self.snapshot_dir = Path(snapshot_dir)
        self.transformer_gguf = Path(transformer_gguf) if transformer_gguf is not None else None
        self.residency = residency
        self.memory_limit_gb = memory_limit_gb
        self.log = log or _logger.info
        self.pipelines: dict[str, object] = {}
        self.text_encoder = None
        self.transformer = None
        self.crf = None
        self.timings: list[dict] = []
        self._adapters = None
        self._embeddings: OrderedDict = OrderedDict()
        #: The stage-2 parts and the diffusion decoder, built on first use.
        self.parts: dict = {}
        sizes = weight_sizes(self.snapshot_dir, self.transformer_gguf)
        self._gb = {name: size / memory.GB for name, (size, _) in sizes.items()}

    # -- bookkeeping -------------------------------------------------------

    @contextmanager
    def phase(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            stats = memory.mps_stats()
            self.timings.append({"phase": name, "seconds": round(seconds, 3), "mps": stats})
            line = f"phase[{name}]: {seconds:.2f} s"
            if stats is not None:
                line += (
                    f" | mps allocated {stats['allocated_gb']:.2f} GB, "
                    f"driver {stats['driver_gb']:.2f} GB"
                )
            self.log(line)

    def _preflight(self, phase: str, need_gb: float) -> None:
        global _freed_since_reading
        limit = {} if self.memory_limit_gb is None else {"limit_gb": float(self.memory_limit_gb)}
        result = memory.preflight(phase, need_gb=need_gb, released=_freed_since_reading, **limit)
        _freed_since_reading = False
        if result.settled is not None:
            self.log(result.settled.describe())
        self.log(result.describe())
        result.require()

    def _free(self) -> None:
        global _freed_since_reading
        gc.collect()
        if _device_type(self.device) == "mps":
            import torch

            torch.mps.empty_cache()
        # Freed MPS pages leave the gauge over a few hundred ms, not at once.
        _freed_since_reading = True

    @contextmanager
    def freed_on_failure(self):
        """On any failure, drop what the failed step held and empty the device cache.

        Without it a refusal or an out-of-memory error left the freed weights'
        Metal buffers allocated to the idle app (review local-runtime-8).
        """
        try:
            yield
        except BaseException as exc:
            _clear_tracebacks(exc)
            self._free()
            raise

    def _attach(self, name: str, module) -> None:
        for pipeline in self.pipelines.values():
            setattr(pipeline, name, module)
            if name == "transformer" and module is not None:
                pipeline.transformer_spatial_patch_size = module.config.patch_size
                pipeline.transformer_temporal_patch_size = module.config.patch_size_t

    # -- the small set and the per-kind pipelines --------------------------

    def load_small_set(self) -> None:
        from diffusers import LTX2ConditionPipeline
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

        spike = memory.load_spike_factor("safetensors")
        need = self._gb["tokenizer"] * spike + sum(
            _load_need_gb(
                self._gb[name],
                spike,
                self.dtype_bytes,
                _stored_dtype_bytes(self.snapshot_dir, name),
            )
            for name in _SMALL_WEIGHTS
        )
        self._preflight("load small set", need)
        with self.phase("load small set"), self.freed_on_failure():
            try:
                connectors = LTX2TextConnectors.from_pretrained(
                    self.snapshot_dir,
                    subfolder="connectors",
                    torch_dtype=self.dtype,
                    local_files_only=True,
                )
                base = LTX2ConditionPipeline.from_pretrained(
                    self.snapshot_dir,
                    text_encoder=None,
                    transformer=None,
                    connectors=None,
                    prompt_enhancer=None,
                    processor=None,
                    torch_dtype=self.dtype,
                    local_files_only=True,
                )
                # Attached after the load: passed in, diffusers logs the whole module.
                base.register_modules(connectors=connectors)
                for name in _SMALL_WEIGHTS:
                    if getattr(base, name, None) is None:
                        continue
                    move_to_device(
                        getattr(base, name),
                        self.device,
                        files=component_files(self.snapshot_dir, name),
                        label=name,
                        log=self.log,
                    )
                base.to(self.device)  # anything else; the weights are already there
                apply_rope_precision("connectors", connectors, self.device, self.log)
            except Exception as exc:
                raise ModelLoadError(component="small set", reason=str(exc)) from exc
        pipelines.assert_compression_ratios(base.vae)
        pipelines.configure_vae_tiling(base.vae)
        pipelines.assert_frame_rate_default(base)
        self.pipelines[pipelines.BASE_KIND] = base
        self._guard_adapter_loading(base)
        self.crf = self._image_crf()

    def _image_crf(self) -> int:
        """Upstream's CRF rule, applied to the text encoder's config (no weights needed)."""
        from diffusers.pipelines.ltx2.utils import resolve_default_image_crf

        config = json.loads((self.snapshot_dir / "text_encoder" / "config.json").read_text())
        stand_in = SimpleNamespace(config=SimpleNamespace(model_type=config.get("model_type")))
        crf = int(resolve_default_image_crf(stand_in))
        self.log(f"image-conditioning CRF {crf} (text encoder type {config.get('model_type')!r})")
        return crf

    def pipeline_for(self, kind: str):
        if kind in self.pipelines:
            return self.pipelines[kind]
        from diffusers import LTX2Pipeline

        base = self.pipelines[pipelines.BASE_KIND]
        accepted = inspect.signature(LTX2Pipeline.__init__).parameters
        built = LTX2Pipeline(**{k: v for k, v in base.components.items() if k in accepted})
        self.pipelines[kind] = built
        self._attach("text_encoder", self.text_encoder)
        self._attach("transformer", self.transformer)
        self._guard_adapter_loading(built)
        return built

    def _guard_adapter_loading(self, pipeline) -> None:
        original = pipeline.load_lora_weights
        weights = self

        def load_lora_weights(*args, **kwargs):
            if weights.transformer is None:
                raise AdapterNeedsTransformerError()
            with weights.freed_on_failure():
                return original(*args, **kwargs)

        pipeline.load_lora_weights = load_lora_weights
        original_delete = pipeline.delete_adapters

        def delete_adapters(adapter_names):
            # Deleted while the transformer is released: the copy captured at the
            # release must not be restored into a later run (review local-runtime-2).
            names = [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
            weights.forget_adapters(names)
            try:
                return original_delete(adapter_names)
            finally:
                weights._free()  # the adapter's layers go back, like any release

        pipeline.delete_adapters = delete_adapters

    def preflight_adapter(self, source) -> None:
        """An adapter file is checked like any other load before it is attached."""
        try:
            path = Path(source)
            file_gb = path.stat().st_size / memory.GB
        except (TypeError, OSError):
            return  # a state dict in memory, or a hub id resolved later: already counted
        need = _load_need_gb(
            file_gb,
            memory.load_spike_factor("safetensors"),
            self.dtype_bytes,
            _tag_bytes(_files_dtype([path])),
        )
        self._preflight(f"attach LoRA {path.name}", need)

    def forget_adapters(self, names) -> None:
        """Drop ``names`` from the adapters captured at a release, if any."""
        captured = self._adapters
        if captured is None:
            return
        names = set(names)
        captured["adapters"] = [a for a in captured["adapters"] if a[0] not in names]
        captured["active"] = [a for a in captured["active"] if a not in names]
        for scaling in captured["scaling"].values():
            for name in names:
                scaling.pop(name, None)
        if not captured["adapters"]:
            self._adapters = None

    # -- the staged pair ---------------------------------------------------

    def ensure_text_encoder(self):
        if self.text_encoder is not None:
            return self.text_encoder
        if self.residency == STAGED:
            self.release_transformer()
        need = _load_need_gb(
            self._gb["text_encoder"],
            memory.load_spike_factor("safetensors"),
            self.dtype_bytes,
            _stored_dtype_bytes(self.snapshot_dir, "text_encoder"),
        )
        self._preflight("load text encoder", need)
        with self.phase("load text encoder"), self.freed_on_failure():
            model = load_text_encoder(
                self.snapshot_dir, dtype=self.dtype, device=self.device, log=self.log
            )
        self.text_encoder = model
        self._attach("text_encoder", model)
        return model

    def release_text_encoder(self) -> None:
        if self.text_encoder is None:
            return
        self._attach("text_encoder", None)
        self.text_encoder = None
        self._free()
        self.log("released the text encoder")
        release_cache(component_files(self.snapshot_dir, "text_encoder"), "text_encoder", self.log)

    def ensure_transformer(self):
        if self.transformer is not None:
            return self.transformer
        if self.residency == STAGED:
            self.release_text_encoder()
        source = "gguf" if self.transformer_gguf is not None else "safetensors"
        need = self._gb["transformer"] * memory.load_spike_factor(source)
        if self.transformer_gguf is None:
            need = _load_need_gb(
                self._gb["transformer"],
                memory.load_spike_factor(source),
                self.dtype_bytes,
                _stored_dtype_bytes(self.snapshot_dir, "transformer"),
            )
        if self._adapters is not None:
            # Re-created on the device inside this phase (review local-runtime-7).
            need += _adapter_bytes(self._adapters) / memory.GB
        self._preflight("load transformer", need)
        with self.phase("load transformer"), self.freed_on_failure():
            model = load_transformer(
                self.snapshot_dir,
                dtype=self.dtype,
                device=self.device,
                gguf=self.transformer_gguf,
                log=self.log,
            )
            if self._adapters is not None:
                try:
                    _restore_adapters(model, self._adapters)
                except BaseException:
                    model = None  # dropped before the cache is emptied
                    raise
                self.log(f"re-attached LoRA adapter(s) {self._adapters['active']}")
                self._adapters = None
        self.transformer = model
        self._attach("transformer", model)
        return model

    def release_transformer(self) -> None:
        if self.transformer is None:
            return
        self._adapters = _capture_adapters(self.transformer)
        self._attach("transformer", None)
        self.transformer = None
        self._free()
        self.log("released the transformer")
        release_cache(self._transformer_files(), "transformer", self.log)

    def _transformer_files(self) -> list[Path]:
        if self.transformer_gguf is not None:
            return [self.transformer_gguf]
        return component_files(self.snapshot_dir, "transformer")

    def release_all(self) -> None:
        self.release_text_encoder()
        self.release_transformer()
        self._adapters = None
        self._embeddings.clear()
        self.pipelines.clear()
        self.parts.clear()
        self._free()
        small = [
            p
            for name in (*_SMALL_WEIGHTS, "tokenizer")
            for p in component_files(self.snapshot_dir, name)
        ]
        release_cache(small, "small set", self.log)

    # -- per call -----------------------------------------------------------

    def with_explicit_crf(self, kwargs: dict) -> dict:
        """Every condition without a CRF gets the one resolved at load.

        Upstream would derive it from ``self.text_encoder``, which a staged
        denoise does not have: it would silently fall from 18 to 33.
        """
        conditions = kwargs.get("conditions")
        if not conditions or self.crf is None:
            return kwargs
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        listed = conditions if isinstance(conditions, list) else [conditions]
        explicit = [
            replace(c, crf=self.crf) if isinstance(c, LTX2VideoCondition) and c.crf is None else c
            for c in listed
        ]
        return {**kwargs, "conditions": explicit}

    def prompt_embeddings(self, pipeline, call: dict) -> dict:
        """Replace ``prompt``/``negative_prompt`` with what the pipeline would have encoded."""
        import torch

        if call.get("prompt_embeds") is not None or call.get("prompt") is None:
            return call
        if call.get("enable_prompt_enhancement"):
            raise ConstraintError(
                "the model's built-in prompt rewriting needs the text encoder during the run, which "
                "the local runner frees first; use the app's 'Enhance prompt' option instead"
            )
        if int(call.get("num_videos_per_prompt") or 1) != 1:
            raise ConstraintError("the local runner makes one video per prompt")
        defaults = _call_defaults(pipeline)
        prompt = call.pop("prompt")
        negative = call.pop("negative_prompt", None)
        guidance = float(call.get("guidance_scale", defaults.get("guidance_scale", 1.0)))
        audio_guidance = call.get("audio_guidance_scale", defaults.get("audio_guidance_scale"))
        audio_guidance = float(audio_guidance or guidance)
        negatives = guidance > 1.0 or audio_guidance > 1.0
        length = int(call.get("max_sequence_length", defaults.get("max_sequence_length", 1024)))
        key = (
            tuple(prompt) if isinstance(prompt, list) else prompt,
            (tuple(negative) if isinstance(negative, list) else negative or "")
            if negatives
            else None,
            length,
            _dtype_label(self.dtype),
        )
        cached = self._embeddings.get(key)
        if cached is not None:
            self._embeddings.move_to_end(key)
            self.log("prompt embeddings: cache hit, the text encoder is not loaded")
        else:
            self.log("prompt embeddings: cache miss")
            self.ensure_text_encoder()
            with self.phase("encode prompt"), torch.inference_mode():
                cached = pipeline.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative,
                    do_classifier_free_guidance=negatives,
                    num_videos_per_prompt=1,
                    max_sequence_length=length,
                    device=pipeline._execution_device,
                )
            self.release_text_encoder()
            self._embeddings[key] = cached
            while len(self._embeddings) > EMBEDDING_CACHE_SIZE:
                self._embeddings.popitem(last=False)
        call["prompt_embeds"], call["prompt_attention_mask"] = cached[0], cached[1]
        if negatives:
            call["negative_prompt_embeds"] = cached[2]
            call["negative_prompt_attention_mask"] = cached[3]
        return call

    def preflight_denoise(self, pipeline, call: dict) -> None:
        defaults = _call_defaults(pipeline)
        width = int(call.get("width", defaults.get("width", 768)))
        height = int(call.get("height", defaults.get("height", 512)))
        frames = frames_for_estimate(call, defaults)
        need = memory.activation_estimate_gb(width, height, frames, dtype_bytes=self.dtype_bytes)
        self._preflight(f"denoise {width}x{height}x{frames}", need)


def _call_defaults(pipeline) -> dict:
    parameters = inspect.signature(type(pipeline).__call__).parameters
    return {
        name: parameter.default
        for name, parameter in parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------


class LocalStageRunner(pipelines.StageRunner):
    """``pipelines.StageRunner`` on mps/cpu: the same stage graph, with staged residency.

    Everything about the graph -- the decode policy, stage 2 and its size
    refusals, the audio conditioning, the clip variation, the run facts -- is
    the Space runner's own code. This class supplies residency through the one
    seam (:meth:`_around_call`) that wraps every pipeline call, and preflights
    the clip encode and the stage-2 loads.
    """

    def __init__(self, weights: _LocalWeights, kind: str):
        pipeline = weights.pipeline_for(kind)
        self._weights = weights
        super().__init__(
            pipeline,
            kind=kind,
            device=weights.device,
            stage_2_factory=self._loads(
                pipelines._STAGE_2_CACHE_KEY,
                ("latent_upsampler", "diffusion_decoder"),
                lambda: pipelines._stage_2_parts(
                    weights.pipelines[pipelines.BASE_KIND],
                    dtype=weights.dtype,
                    device=weights.device,
                    cache=weights.parts,
                ),
            ),
            decoder_factory=self._loads(
                pipelines._DECODER_CACHE_KEY,
                ("diffusion_decoder",),
                lambda: pipelines._decoder_part(
                    weights.pipelines[pipelines.BASE_KIND],
                    dtype=weights.dtype,
                    device=weights.device,
                    cache=weights.parts,
                ),
            ),
        )

    @property
    def residency(self) -> str:
        return self._weights.residency

    @property
    def timings(self) -> list[dict]:
        return self._weights.timings

    def ensure_transformer(self):
        """Load the transformer now (before attaching a LoRA, for instance)."""
        return self._weights.ensure_transformer()

    def preflight_adapter(self, source) -> None:
        """Check attaching the adapter at ``source`` fits, before it is loaded."""
        self._weights.preflight_adapter(source)

    def held_adapters(self) -> tuple[str, ...]:
        """Adapters kept for a released transformer, to be put back when it loads again.

        Removing one through ``pipeline.delete_adapters`` forgets it.
        """
        captured = self._weights._adapters
        if not captured:
            return ()
        return tuple(name for name, _config, _state in captured["adapters"])

    # -- the seam ----------------------------------------------------------

    @contextmanager
    def _around_call(self, pipeline, call: dict, *, stage: str):
        """Make the call's components resident, swap the prompt for embeddings, preflight.

        A denoise, in this order: the CRF on every condition lacking one;
        staged, the embedding swap (a cache hit for stage 2, so the text encoder
        is not reloaded and ``prompt`` never comes back) and the transformer made
        resident; the memory preflight; then the call, under ``inference_mode``.
        A stage-2 part (the upsampler, the decoder) runs under the same guards.
        """
        import torch

        weights = self._weights
        if stage in ("upsample", "decode"):
            with weights.phase(stage), weights.freed_on_failure(), torch.inference_mode():
                yield call
            return
        call = weights.with_explicit_crf(call)
        if weights.residency == STAGED:
            call = weights.prompt_embeddings(pipeline, call)
            weights.ensure_transformer()
        weights.preflight_denoise(pipeline, call)
        label = "denoise and decode" if stage == "single" else f"denoise ({stage})"
        with weights.phase(label), weights.freed_on_failure(), torch.inference_mode():
            yield call

    def _pipeline_for(self, audio_noise_level):
        pipeline = super()._pipeline_for(audio_noise_level)
        if pipeline is not self.pipeline:
            # Registered so a release detaches the staged pair from it too;
            # otherwise it would keep a released transformer alive.
            self._weights.pipelines[f"{self.kind}+audio"] = pipeline
        return pipeline

    def _variation(self, frames, level, kwargs) -> dict:
        """Encode the clip under a preflight, before the denoise makes anything resident."""
        import torch

        weights = self._weights
        width, height, count = kwargs["width"], kwargs["height"], kwargs["num_frames"]
        weights._preflight(
            f"encode clip {width}x{height}x{count}",
            memory.activation_estimate_gb(width, height, count, dtype_bytes=weights.dtype_bytes),
        )
        with weights.phase("encode clip"), weights.freed_on_failure(), torch.inference_mode():
            return super()._variation(frames, level, kwargs)

    def _loads(self, key, components, build):
        """A stage-2 factory that preflights what it is about to load."""
        weights = self._weights

        def factory():
            if key not in weights.parts:
                need = sum(
                    _part_need_gb(name, weights.dtype_bytes)
                    for name in components
                    if name != "diffusion_decoder"
                    or pipelines._DECODER_CACHE_KEY not in weights.parts
                )
                weights._preflight(f"load {' and '.join(components)}", need)
            with weights.phase("load stage-2 parts"), weights.freed_on_failure():
                return build()

        return factory


def _part_need_gb(component: str, dtype_bytes: int) -> float:
    """What loading a stage-2 component commits, from its pinned files on disk."""
    from . import preflight

    folder = Path(preflight.component_dir(component))
    files = sorted(folder.glob(f"*{_WEIGHTS_SUFFIX}"))
    file_gb = pipelines._weight_bytes(folder) / memory.GB
    return _load_need_gb(
        file_gb,
        memory.load_spike_factor("safetensors"),
        dtype_bytes,
        _tag_bytes(_files_dtype(files)),
    )


# --------------------------------------------------------------------------
# construction (cached, like pipelines.get_pipeline)
# --------------------------------------------------------------------------

_WEIGHTS: dict[tuple, _LocalWeights] = {}
_RUNNERS: dict[tuple, LocalStageRunner] = {}


def build_local_runner(
    *,
    kind,
    device,
    dtype,
    snapshot_dir,
    transformer_gguf=None,
    residency=STAGED,
    memory_limit_gb=None,
    log=None,
) -> LocalStageRunner:
    """The runner for one workflow kind on mps/cpu, sharing one set of weights.

    The same arguments return the same runner, and both kinds of one
    configuration share their weights, so this can be called per generation.
    """
    device_type = _device_type(device)
    if device_type == "cuda":
        raise UnsupportedLocalDeviceError(device=device)
    if device_type not in LOCAL_DEVICES:
        raise UnsupportedLocalDeviceError(device=device)
    if residency not in RESIDENCIES:
        raise ConstraintError(f"residency must be 'staged' or 'resident', not {residency!r}")
    if kind not in KINDS:
        raise ConstraintError(f"pipeline kind must be one of {', '.join(KINDS)}, not {kind!r}")
    if device_type == "mps":
        memory.mps_watermarks().require()
    import torch

    if device_type == "mps" and not torch.backends.mps.is_available():
        raise UnsupportedLocalDeviceError(
            device=device, reason="MPS is not available on this machine."
        )
    torch_dtype = resolve_dtype(dtype)
    snapshot = Path(snapshot_dir).expanduser()
    gguf = Path(transformer_gguf).expanduser() if transformer_gguf is not None else None
    key = (
        device_type,
        _dtype_label(torch_dtype),
        str(snapshot.resolve()),
        str(gguf),
        residency,
        memory_limit_gb,
    )

    weights = _WEIGHTS.get(key)
    if weights is None:
        absent = missing_files(snapshot, transformer_gguf=gguf)
        if absent:
            raise LocalFilesMissingError(missing=absent)
        weights = _LocalWeights(
            device=device_type,
            dtype=torch_dtype,
            snapshot_dir=snapshot,
            transformer_gguf=gguf,
            residency=residency,
            memory_limit_gb=memory_limit_gb,
            log=log,
        )
        weights.log(
            f"local runner: {device_type}/{_dtype_label(torch_dtype)}, {residency} residency, "
            f"transformer from {gguf if gguf else 'the snapshot'}"
        )
        try:
            weights.load_small_set()
            if residency == RESIDENT:
                weights.ensure_transformer()
                weights.ensure_text_encoder()
        except BaseException as exc:
            # A failed build is not cached: release what it loaded now rather than
            # leave it allocated to an idle app (review local-runtime-8).
            _clear_tracebacks(exc)
            weights.release_all()
            raise
        _WEIGHTS[key] = weights
    elif log is not None:
        weights.log = log

    runner = _RUNNERS.get((*key, kind))
    if runner is None:
        runner = LocalStageRunner(weights, kind)
        _RUNNERS[(*key, kind)] = runner
    return runner


def clear_local_cache() -> None:
    """Release every local runner's weights."""
    for weights in _WEIGHTS.values():
        weights.release_all()
    _WEIGHTS.clear()
    _RUNNERS.clear()
    gc.collect()
