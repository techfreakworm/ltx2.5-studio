"""Shared test infrastructure for the LTX-2.5 Studio spec suite.

Every fixture here obeys the hard constraint from PLAN §9.1: **no GPU, no network,
no model weights**. Nothing in this file imports torch, diffusers, safetensors or
gradio.

Two deliberate choices worth knowing about:

1. **safetensors files are synthesised byte-for-byte in pure Python.**
   The format is trivial and documented: an 8-byte little-endian ``u64`` header
   length, that many bytes of UTF-8 JSON, then the tensor payload. Writing it
   ourselves (rather than via ``safetensors.torch.save_file``) means the LoRA
   suite runs on a bare venv, and — more importantly — it lets us craft the
   *malformed* files that PLAN §4.3 stage 3 requires: a truncated header, a
   corrupt header length, a payload that is never read. A torch-based writer
   cannot produce those. ``test_lora_inspect.py`` cross-checks the synthesised
   bytes against the real ``safetensors`` library when it happens to be
   installed, so the fixtures cannot quietly drift from the real format.

2. **Fakes mirror the *failure modes* of the real objects, not just their happy
   path.** ``fake_pipeline`` sets ``peft_config`` on a zero-attach load, exactly
   as peft does, because an implementation that checks ``peft_config`` instead of
   counting attached modules is the silent-garbage bug PLAN §4.3 stage 7 exists
   to catch. The fake must be able to express the bug or the test is theatre.
"""

from __future__ import annotations

import builtins
import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

# --------------------------------------------------------------------------
# safetensors synthesis (pure Python — no torch, no safetensors)
# --------------------------------------------------------------------------

DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "U8": 1,
    "I8": 1,
    "BOOL": 1,
}

#: Key prefixes verified against real LTX-2.5 adapters (PLAN §4.1, §4.2).
NATIVE_PREFIX = "diffusion_model."
DIFFUSERS_PREFIX = "transformer."
NATIVE_CONNECTOR_PREFIX = "text_embedding_projection."
DIFFUSERS_CONNECTOR_PREFIX = "connectors."

ALL_PREFIXES = (
    NATIVE_PREFIX,
    DIFFUSERS_PREFIX,
    NATIVE_CONNECTOR_PREFIX,
    DIFFUSERS_CONNECTOR_PREFIX,
)

#: Module paths the fake transformer exposes; adapter keys that resolve to one of
#: these are what "attached" means. Mirrors the real target list from PLAN §4.2
#: (``to_q`` / ``to_k`` / ``to_v`` / ``adaln_single``).
TRANSFORMER_MODULE_PATHS = (
    "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.attn1.to_k",
    "transformer_blocks.0.attn1.to_v",
    "transformer_blocks.1.attn1.to_q",
    "adaln_single.linear",
)

CONNECTOR_MODULE_PATHS = (
    "layers.0.proj",
    "layers.1.proj",
)

#: The official adapter's metadata, read from its real header (PLAN §4.2).
OFFICIAL_METADATA = {
    "lora_rank": "450",
    "lora_alpha": "450",
    "model_version": "2.5.0",
    "license": "LTX-2.5 Community License",
}


def build_safetensors_bytes(
    tensors,
    metadata=None,
    *,
    header_length_override=None,
    payload_bytes_override=None,
    align_header=True,
):
    """Return the bytes of a safetensors file.

    ``tensors`` maps tensor name -> ``(dtype, shape)``. The payload is zero-filled;
    nothing here ever needs to be a real tensor, which is the whole point of
    PLAN §4.3 stage 2 (header-only parsing).

    ``header_length_override`` writes a deliberately wrong 8-byte length prefix
    (the "corrupt header length" case). ``payload_bytes_override`` writes fewer
    payload bytes than the header declares.
    """
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * DTYPE_SIZES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    blob = json.dumps(header).encode("utf-8")
    if align_header:
        pad = (8 - (len(blob) % 8)) % 8
        blob += b" " * pad

    declared = len(blob) if header_length_override is None else header_length_override
    payload_len = offset if payload_bytes_override is None else payload_bytes_override
    return struct.pack("<Q", declared) + blob + (b"\x00" * payload_len)


def lora_pair(prefix, module_path, *, style="peft", shape=(8, 8)):
    """Return the ``{key: (dtype, shape)}`` entries for one LoRA A/B pair."""
    if style == "peft":
        down, up = "lora_A.weight", "lora_B.weight"
    elif style == "kohya":
        down, up = "lora_down.weight", "lora_up.weight"
    else:  # pragma: no cover - guard against a typo in a test
        raise ValueError(f"unknown style {style!r}")
    base = f"{prefix}{module_path}"
    return {
        f"{base}.{down}": ("F32", list(shape)),
        f"{base}.{up}": ("F32", list(shape)),
    }


def lora_tensors(prefix, module_paths, *, style="peft"):
    tensors: dict[str, tuple[str, list[int]]] = {}
    for path in module_paths:
        tensors.update(lora_pair(prefix, path, style=style))
    return tensors


def read_safetensors_header(path):
    """Independent reference reader — used by the fakes, never by the code under test."""
    data = Path(path).read_bytes()
    (declared,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8 : 8 + declared].decode("utf-8"))


# --------------------------------------------------------------------------
# safetensors fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def write_safetensors(tmp_path):
    """Factory: write a synthetic safetensors file and return its path."""

    def _write(name="adapter.safetensors", tensors=None, metadata=None, **kwargs):
        if tensors is None:
            tensors = lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS)
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(build_safetensors_bytes(tensors, metadata, **kwargs))
        return target

    return _write


@pytest.fixture
def good_lora(write_safetensors):
    """A well-formed native-convention adapter that targets the fake transformer."""
    return write_safetensors(
        name="good-lora.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata=OFFICIAL_METADATA,
    )


@pytest.fixture
def wrong_architecture_lora(write_safetensors):
    """PLAN §9.3.1: passes stages 1-6, attaches to ZERO modules.

    Every key contains ``lora`` (so diffusers' only built-in check passes), the
    prefix is a recognised one, pairs are complete and ``model_version`` says
    2.5.0. The module paths are simply from a different architecture — which is
    exactly the checkpoint that produces silent garbage today.
    """
    return write_safetensors(
        name="wrong-arch-lora.safetensors",
        tensors=lora_tensors(
            NATIVE_PREFIX,
            (
                "down_blocks.0.resnets.0.conv1",
                "mid_block.attentions.0.to_out.0",
                "up_blocks.3.attentions.2.proj_out",
            ),
        ),
        metadata=OFFICIAL_METADATA,
    )


# --------------------------------------------------------------------------
# fake pipeline (duck-types the diffusers surface we touch)
# --------------------------------------------------------------------------


class FakeModule:
    """Minimal ``torch.nn.Module`` stand-in: a named module tree."""

    def __init__(self):
        self._children: dict[str, FakeModule] = {}

    @classmethod
    def from_paths(cls, paths):
        root = cls()
        for path in paths:
            node = root
            for part in path.split("."):
                node = node._children.setdefault(part, cls())
        return root

    def named_modules(self, prefix=""):
        if prefix:
            yield prefix, self
        for name, child in self._children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def get_module(self, path):
        node = self
        for part in path.split("."):
            node = node._children.get(part)
            if node is None:
                return None
        return node


class FakeComponent(FakeModule):
    """A pipeline component (``transformer`` / ``connectors``) that peft can decorate."""

    def __init__(self, paths):
        super().__init__()
        for path in paths:
            node = self
            for part in path.split("."):
                node = node._children.setdefault(part, FakeModule())
        self.peft_config: dict[str, object] = {}


class FakePipeline:
    """Duck-types the bits of an ``LTX2*Pipeline`` that ``lora/apply.py`` touches.

    ``load_lora_weights`` reproduces upstream behaviour *including its hole*:

    * the only validation is the ``'lora'`` substring test (PLAN §4.2);
    * ``peft_config`` is populated whether or not anything attached — so code
      that checks ``peft_config`` instead of counting modules still sees a
      "successful" load. That is the silent-garbage path stage 7 must close.
    """

    _lora_loadable_modules = ["transformer", "connectors"]

    def __init__(self, transformer_paths=None, connector_paths=None):
        self.transformer = FakeComponent(
            TRANSFORMER_MODULE_PATHS if transformer_paths is None else transformer_paths
        )
        self.connectors = FakeComponent(
            CONNECTOR_MODULE_PATHS if connector_paths is None else connector_paths
        )
        self.calls: list[dict] = []
        self.fuse_calls: list[dict] = []
        self.unfuse_calls: list[dict] = []

    # -- upstream conversion table (PLAN §4.1) ----------------------------
    _ROUTES = {
        NATIVE_PREFIX: "transformer",
        DIFFUSERS_PREFIX: "transformer",
        NATIVE_CONNECTOR_PREFIX: "connectors",
        DIFFUSERS_CONNECTOR_PREFIX: "connectors",
    }

    def load_lora_weights(self, source, *args, adapter_name="default", **kwargs):
        self.calls.append({"source": source, "adapter_name": adapter_name, "kwargs": kwargs})
        header = read_safetensors_header(source)
        keys = [k for k in header if k != "__metadata__"]

        # The one and only upstream check, quoted verbatim from diffusers.
        if not all("lora" in k for k in keys):
            raise ValueError(
                "Invalid LoRA checkpoint. Make sure all LoRA param names contain "
                "`'lora'` substring."
            )

        attached = 0
        for key in keys:
            component_name = None
            stripped = key
            for prefix, route in self._ROUTES.items():
                if key.startswith(prefix):
                    component_name = route
                    stripped = key[len(prefix) :]
                    break
            if component_name is None:
                continue
            module_path = stripped
            for suffix in (
                ".lora_A.weight",
                ".lora_B.weight",
                ".lora_down.weight",
                ".lora_up.weight",
            ):
                if module_path.endswith(suffix):
                    module_path = module_path[: -len(suffix)]
                    break
            component = getattr(self, component_name)
            module = component.get_module(module_path)
            if module is None:
                continue
            lora_a = getattr(module, "lora_A", None)
            if lora_a is None:
                lora_a = {}
                module.lora_A = lora_a
                module.lora_B = {}
            if adapter_name not in lora_a:
                lora_a[adapter_name] = object()
                module.lora_B[adapter_name] = object()
                attached += 1

        # THE TRAP: peft records the adapter config even when nothing attached.
        for component_name in self._lora_loadable_modules:
            component = getattr(self, component_name, None)
            if component is not None:
                component.peft_config[adapter_name] = {"r": 450, "lora_alpha": 450}
        return attached

    def get_list_adapters(self):
        """Also optimistic: reports the adapter whether or not anything attached."""
        return {
            name: list(getattr(self, name).peft_config)
            for name in self._lora_loadable_modules
            if getattr(self, name, None) is not None
        }

    def set_adapters(self, *args, **kwargs):
        self.calls.append({"set_adapters": (args, kwargs)})

    def fuse_lora(self, *args, **kwargs):
        self.fuse_calls.append({"args": args, "kwargs": kwargs})

    def unfuse_lora(self, *args, **kwargs):
        self.unfuse_calls.append({"args": args, "kwargs": kwargs})


@pytest.fixture
def fake_pipeline():
    return FakePipeline()


@pytest.fixture
def make_pipeline():
    def _make(transformer_paths=None, connector_paths=None):
        return FakePipeline(transformer_paths, connector_paths)

    return _make


# --------------------------------------------------------------------------
# byte-counting read spy — proof of PLAN §4.3 stage 2
# --------------------------------------------------------------------------


class _CountingFile:
    def __init__(self, wrapped, spy):
        self._wrapped = wrapped
        self._spy = spy

    def read(self, *args, **kwargs):
        chunk = self._wrapped.read(*args, **kwargs)
        self._spy.bytes_read += len(chunk)
        return chunk

    def readinto(self, buf):
        n = self._wrapped.readinto(buf)
        self._spy.bytes_read += n or 0
        return n

    def readline(self, *args, **kwargs):
        chunk = self._wrapped.readline(*args, **kwargs)
        self._spy.bytes_read += len(chunk)
        return chunk

    def __enter__(self):
        self._wrapped.__enter__()
        return self

    def __exit__(self, *exc):
        return self._wrapped.__exit__(*exc)

    def __iter__(self):
        return iter(self._wrapped)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class ReadSpy:
    def __init__(self):
        self.bytes_read = 0
        self.opens = 0
        self.watched: set[str] = set()

    def watch(self, path):
        self.watched.add(str(Path(path).resolve()))


@pytest.fixture
def read_spy(monkeypatch):
    """Counts bytes read from watched files via ``open()`` / ``io.open()``.

    Both names are patched: ``pathlib.Path.open`` and ``Path.read_bytes`` route
    through ``io.open``, not ``builtins.open``, so patching only one leaves a
    hole a header parser could slip through.
    """
    spy = ReadSpy()
    real_open = builtins.open

    def fake_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        try:
            resolved = str(Path(file).resolve())
        except (TypeError, ValueError, OSError):
            return handle
        if resolved in spy.watched and "b" in mode and "w" not in mode and "a" not in mode:
            spy.opens += 1
            return _CountingFile(handle, spy)
        return handle

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(io, "open", fake_open)
    return spy


# --------------------------------------------------------------------------
# generation request factory
# --------------------------------------------------------------------------


def request_fields(**overrides):
    """Every field a mode kwarg-builder could plausibly read, with legal defaults.

    Deliberately a plain namespace: PLAN §7.2 says the mode registry is data and
    the builders are pure, so they must not require a concrete class from another
    module. If a builder only works with the real ``GenerationRequest``, that is
    a coupling the registry should not have.
    """
    fields = {
        "mode": "text_to_video",
        "prompt": "a cat playing a grand piano, cinematic",
        "negative_prompt": "",
        "width": 704,
        "height": 480,
        "num_frames": 121,
        "seed": 1234,
        "image": None,
        "video": None,
        "audio": None,
        "audio_latents": None,
        "noise_scale": 0.5,
        "conditions": (),
        "strength": 0.8,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "loras": (),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def make_request():
    return request_fields


# --------------------------------------------------------------------------
# source-file helpers for the architecture specs
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "ltx25_studio"


def package_python_files(exclude=()):
    excluded = {str(PACKAGE_ROOT / name) for name in exclude}
    return sorted(
        p
        for p in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and str(p) not in excluded
    )
