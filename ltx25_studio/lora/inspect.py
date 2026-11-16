"""Header-only safetensors inspection: metadata, keys, shapes, convention.

PLAN §4.3 stages 2, 3 and 5, and §4.6.

**Stage 2 is a hard rule: the tensor payload is never materialised.** Real
adapters run to ~9 GB and the official one holds 3320 tensors. Every
compatibility answer comes from the header; the payload is only streamed once,
in fixed-size pieces, through a sha256 that names the adapter by its content
(two adapters from one trainer config have byte-identical headers).

**Stdlib only -- no torch, no diffusers, no ``safetensors`` package.** The
container format is trivial::

    [8 bytes little-endian uint64 = header length N][N bytes of JSON][payload]

so ``struct`` + ``json`` read everything we need. Three reasons (PLAN §4.6):
rejecting a bad upload must not require a 2 GB import; the test gate stays fast
and portable; and because the header sits at the very start of the file, the
Hub route can validate a remote adapter with a single HTTP range request,
**before downloading it**.

Detection classifies, it does not gatekeep. Both key conventions load --
diffusers converts non-diffusers keys itself -- so ``convention`` exists to tell
the user what we saw, not to refuse what we understand.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..errors import LoRAFormatError, LoRAHeaderError, LoRATruncatedError

__all__ = [
    "CONNECTOR_CONVENTIONS",
    "DTYPE_BITS",
    "HEADER_LENGTH_BYTES",
    "KEY_PREFIXES",
    "MAX_HEADER_BYTES",
    "VERSION_METADATA_KEYS",
    "LoRAInspection",
    "inspect_lora",
    "parse_safetensors_header",
    "read_safetensors_header",
]

#: The container's fixed-width length prefix.
HEADER_LENGTH_BYTES = 8

#: Bound on the declared header length, the same one the Hub range reader uses.
#: The largest real adapter header measured is ~0.4 MB (the official 2.5
#: adapter, licence included). Past this is a corrupt prefix or a hostile file,
#: and parsing it would cost hundreds of MB in the web process.
MAX_HEADER_BYTES = 16_000_000

#: Bits per element of every dtype the safetensors loader reads (0.8.0,
#: measured). A tensor of any other dtype is refused by the loader.
DTYPE_BITS = MappingProxyType(
    {
        "BOOL": 8,
        "F4": 4,
        "F6_E2M3": 6,
        "F6_E3M2": 6,
        "U8": 8,
        "I8": 8,
        "F8_E5M2": 8,
        "F8_E4M3": 8,
        "F8_E8M0": 8,
        "F8_E4M3FNUZ": 8,
        "F8_E5M2FNUZ": 8,
        "I16": 16,
        "U16": 16,
        "F16": 16,
        "BF16": 16,
        "I32": 32,
        "U32": 32,
        "F32": 32,
        "C64": 64,
        "F64": 64,
        "I64": 64,
        "U64": 64,
    }
)

#: The four verified prefixes, mapped to a convention name (PLAN §4.1).
#: ``diffusion_model.`` dominates in the wild: every real community adapter
#: measured uses it, so the non-diffusers conversion path is the MAIN path.
KEY_PREFIXES = MappingProxyType(
    {
        "diffusion_model.": "native",
        "transformer.": "diffusers",
        "text_embedding_projection.": "native connectors",
        "connectors.": "diffusers connectors",
    }
)

#: Conventions that target the text-conditioning path rather than the transformer.
CONNECTOR_CONVENTIONS = frozenset({"native connectors", "diffusers connectors"})

#: Version signals, in priority order (PLAN §4.3 stage 4, §4.5). The official
#: Lightricks adapter carries ``model_version``; the sd-scripts / musubi trainer
#: writes an entirely different block. Most community adapters carry none at
#: all, which is why absence warns rather than rejects.
VERSION_METADATA_KEYS = (
    "model_version",
    "ss_ltx_version",
    "modelspec.architecture",
    "ss_base_model_version",
)

#: The payload streams through the content hash in pieces this size.
DIGEST_CHUNK_BYTES = 1 << 20

_HEADER_FIRST_BYTE = b"{"


@dataclass(frozen=True)
class LoRAInspection:
    """Everything the header tells us. Deliberately holds no tensor values."""

    path: Path
    keys: tuple[str, ...]
    metadata: MappingProxyType
    prefixes: tuple[str, ...]
    convention: str | None
    model_version: str | None
    header_bytes: int
    file_bytes: int
    #: tensor name -> (dtype, shape). Shapes only -- this is what stage 6.5
    #: compares against the real modules, and it costs nothing to carry.
    shapes: MappingProxyType
    #: sha256 of the length prefix and header: identifies the layout only.
    header_sha256: str = ""
    #: sha256 of the whole file, the Hub's LFS hash: identifies the adapter, so
    #: the same one staged twice is recognised whatever its file name.
    content_sha256: str = ""

    # -- fork transport ---------------------------------------------------
    #
    # On ZeroGPU the GPU dispatch pickles the decorated function's ARGUMENTS
    # onto the worker's arg queue, and a staged adapter rides in as
    # ``GenerationRequest.loras[i]["report"]`` -- so this object must pickle
    # or the run dies at dispatch, before any GPU work, as 'PicklingError:
    # cannot pickle mappingproxy object'. That is the INBOUND half of the
    # fork boundary; the OUTBOUND half (exceptions coming back) lives in
    # ``errors.LTXStudioError.__reduce__``. mappingproxy refuses to pickle by
    # design, so the state hooks unwrap the two proxy fields to plain dicts
    # for transport and RE-WRAP on restore: the read-only invariant the
    # proxies exist for holds identically on both sides of the fork, and no
    # caller sees a different type in either process.

    def __getstate__(self):
        state = dict(self.__dict__)
        state["metadata"] = dict(self.metadata)
        state["shapes"] = dict(self.shapes)
        return state

    def __setstate__(self, state):
        state = dict(state)
        state["metadata"] = MappingProxyType(state["metadata"])
        state["shapes"] = MappingProxyType(state["shapes"])
        for name, value in state.items():
            # frozen dataclass: normal attribute assignment is refused, which
            # is the point of frozen -- restore the way dataclasses do.
            object.__setattr__(self, name, value)

    @property
    def version_signals(self) -> dict[str, str]:
        """Every recognised version signal present, in priority order."""
        return {k: self.metadata[k] for k in VERSION_METADATA_KEYS if k in self.metadata}

    @property
    def targets_connectors(self) -> bool:
        return any(KEY_PREFIXES.get(p) in CONNECTOR_CONVENTIONS for p in self.prefixes)


def _read_header_bytes(path: Path) -> tuple[bytes, int, int]:
    """Read only the length prefix and the header. Never the payload."""
    try:
        file_bytes = path.stat().st_size
    except OSError as exc:
        raise LoRAFormatError(source=str(path), reason=f"cannot be read ({exc.strerror})") from exc

    if file_bytes < HEADER_LENGTH_BYTES + 1:
        raise LoRAFormatError(
            source=str(path),
            reason=(
                f"the file is {file_bytes} bytes, too short to be a safetensors container "
                f"(it needs at least an {HEADER_LENGTH_BYTES}-byte length prefix and a header)"
            ),
        )

    try:
        with open(path, "rb") as handle:
            prefix = handle.read(HEADER_LENGTH_BYTES)
            sentinel = handle.read(1)
            # A safetensors header is a JSON object, so byte 8 is '{'. Checking
            # it BEFORE trusting the length prefix is what separates "this is
            # not a safetensors file" from "this is one with a corrupt header
            # length" -- two different diagnoses for the user.
            if sentinel != _HEADER_FIRST_BYTE:
                raise LoRAFormatError(
                    source=str(path),
                    reason=(
                        "not a safetensors container: the bytes after the length prefix are "
                        f"{sentinel!r}, but a safetensors header always begins with '{{'"
                    ),
                )

            (declared,) = struct.unpack("<Q", prefix)
            if declared == 0 or declared > MAX_HEADER_BYTES:
                raise LoRAHeaderError(source=str(path), declared_header_bytes=declared)

            available = file_bytes - HEADER_LENGTH_BYTES
            if declared > available:
                raise LoRATruncatedError(
                    source=str(path),
                    declared_bytes=HEADER_LENGTH_BYTES + declared,
                    available_bytes=file_bytes,
                )

            handle.seek(HEADER_LENGTH_BYTES)
            blob = handle.read(declared)
    except OSError as exc:
        raise LoRAFormatError(source=str(path), reason=f"cannot be read ({exc})") from exc

    if len(blob) < declared:
        raise LoRATruncatedError(
            source=str(path),
            declared_bytes=HEADER_LENGTH_BYTES + declared,
            available_bytes=HEADER_LENGTH_BYTES + len(blob),
        )
    return blob, declared, file_bytes


def parse_safetensors_header(blob: bytes, *, source: str) -> dict:
    """Decode the header JSON, or raise a specific format error."""
    try:
        header = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LoRAFormatError(
            source=source, reason=f"the safetensors header is not valid JSON ({_shown(exc, 200)})"
        ) from exc
    except RecursionError as exc:
        raise LoRAFormatError(
            source=source, reason="the safetensors header nests too deeply to be one"
        ) from exc
    if not isinstance(header, dict):
        raise LoRAFormatError(
            source=source,
            reason=(
                f"the safetensors header decodes to {type(header).__name__}, not an object; "
                "a safetensors header is always a JSON object"
            ),
        )
    return header


def read_safetensors_header(path: str | Path) -> tuple[dict, int]:
    """Return ``(header, declared_header_length)`` without touching the payload."""
    path = Path(path)
    blob, declared, _ = _read_header_bytes(path)
    return parse_safetensors_header(blob, source=str(path)), declared


def _shown(value, limit: int = 60) -> str:
    """``repr(value)`` cut short, so a hostile header cannot become a huge message."""
    text = str(value) if isinstance(value, Exception) else repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _kind(value) -> str:
    names = {dict: "an object", list: "a list", str: "text", bool: "true or false"}
    if value is None:
        return "null"
    return names.get(type(value), "a number")


def _counts(value) -> bool:
    """A JSON list of whole numbers of 0 or more (``true`` is not a number here)."""
    return isinstance(value, list) and all(type(item) is int and item >= 0 for item in value)


def _checked_entries(header: dict, *, source: str) -> tuple[dict, dict]:
    """``(metadata, {name: (dtype, shape, begin, end)})`` with every entry typed as the
    safetensors loader requires, or a :class:`LoRAFormatError` naming the first fault."""

    def refuse(reason):
        raise LoRAFormatError(source=source, reason=reason)

    raw = header.get("__metadata__")
    if raw is None:
        metadata = {}
    elif not isinstance(raw, dict):
        refuse(f"its __metadata__ is {_kind(raw)}, not a table of text values")
    else:
        for key, value in raw.items():
            if not isinstance(value, str):
                refuse(f"its __metadata__ entry {_shown(key)} is {_kind(value)}, not text")
        metadata = dict(raw)

    entries = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            refuse(f"tensor {_shown(name)} is described by {_kind(entry)}, not an object")
        dtype, shape = entry.get("dtype"), entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_BITS:
            found = "no dtype" if dtype is None else f"dtype {_shown(dtype)}"
            refuse(f"tensor {_shown(name)} has {found}, which is not one safetensors reads")
        if not _counts(shape):
            refuse(
                f"tensor {_shown(name)} has shape {_shown(shape)}; a shape is a list of "
                f"whole numbers of 0 or more"
            )
        if not (_counts(offsets) and len(offsets) == 2):
            refuse(
                f"tensor {_shown(name)} has data_offsets {_shown(offsets)}; they must be two "
                f"whole numbers of 0 or more"
            )
        entries[name] = (dtype, tuple(shape), offsets[0], offsets[1])
    return metadata, entries


def _payload_bytes(entries: dict, *, source: str) -> int:
    """Where the tensor data ends, checked by the safetensors loader's own rules.

    In offset order, each tensor starts where the previous one ended (the
    first at 0), and spans exactly its shape times its dtype size.
    """
    end_of_previous = 0
    for name, (dtype, shape, begin, end) in sorted(
        entries.items(), key=lambda item: (item[1][2], item[1][3])
    ):
        if begin != end_of_previous or end < begin:
            raise LoRAFormatError(
                source=source,
                reason=(
                    f"tensor {_shown(name)} is placed at bytes {begin}-{end} of the tensor data, "
                    f"but the tensor before it ends at {end_of_previous}; tensors must follow "
                    f"one another with no gap or overlap"
                ),
            )
        bits = math.prod(shape) * DTYPE_BITS[dtype]
        if bits % 8 or end - begin != bits // 8:
            raise LoRAFormatError(
                source=source,
                reason=(
                    f"tensor {_shown(name)} is {dtype} {_shown(list(shape))}, which is "
                    f"{bits / 8:g} bytes, but its data offsets span {end - begin}"
                ),
            )
        end_of_previous = end
    return end_of_previous


def _file_sha256(path: Path, head: bytes, *, expected: int) -> str:
    """sha256 of the whole file: ``head`` (its first bytes, already read), then the rest
    streamed in :data:`DIGEST_CHUNK_BYTES` pieces. Never more than one piece in memory."""
    digest = hashlib.sha256(head)
    buffer = bytearray(DIGEST_CHUNK_BYTES)
    view = memoryview(buffer)
    total = len(head)
    try:
        with open(path, "rb") as handle:
            handle.seek(len(head))
            while count := handle.readinto(buffer):
                digest.update(view[:count])
                total += count
    except OSError as exc:
        raise LoRAFormatError(source=str(path), reason=f"cannot be read ({exc})") from exc
    if total != expected:
        # The file changed size between the checks above and this read.
        raise LoRATruncatedError(source=str(path), declared_bytes=expected, available_bytes=total)
    return digest.hexdigest()


def _prefix_of(key: str) -> str:
    """The leading namespace of a tensor key, always with its trailing dot.

    Derived uniformly rather than matched against the known list, so an
    unrecognised adapter reports the prefix it actually has -- which is what the
    stage 5 error message needs to name.
    """
    head, sep, _ = key.partition(".")
    return f"{head}." if sep else key


def _classify(prefixes: tuple[str, ...]) -> str | None:
    """Pick the convention to report, preferring a transformer-targeting one.

    An adapter may legitimately span both the transformer and the connectors
    (the official one does), so this reports the primary convention rather than
    refusing to choose.
    """
    known = [KEY_PREFIXES[p] for p in prefixes if p in KEY_PREFIXES]
    if not known:
        return None
    for convention in known:
        if convention not in CONNECTOR_CONVENTIONS:
            return convention
    return known[0]


def inspect_lora(path: str | Path) -> LoRAInspection:
    """Read a LoRA's header and report what it is.

    The payload is read once, after every header check has passed, and only
    to hash it; no tensor is ever built.
    """
    path = Path(path)
    blob, declared, file_bytes = _read_header_bytes(path)
    header = parse_safetensors_header(blob, source=str(path))
    # No metadata key, `__metadata__: null` and `__metadata__: {}` all mean "none";
    # all three are common in the wild (PLAN §4.5) and none is an error.
    metadata, entries = _checked_entries(header, source=str(path))

    # The loader in the worker refuses a file whose data is not exactly where
    # the header says; refusing it here keeps that failure out of the GPU window.
    expected = HEADER_LENGTH_BYTES + declared + _payload_bytes(entries, source=str(path))
    if file_bytes < expected:
        raise LoRATruncatedError(
            source=str(path), declared_bytes=expected, available_bytes=file_bytes
        )
    if file_bytes > expected:
        raise LoRAFormatError(
            source=str(path),
            reason=(
                f"the file is {file_bytes} bytes but its header accounts for {expected}; the "
                f"{file_bytes - expected} bytes after the last tensor belong to no tensor"
            ),
        )

    keys = tuple(entries)
    shapes = {name: (dtype, shape) for name, (dtype, shape, _begin, _end) in entries.items()}
    prefixes = tuple(dict.fromkeys(_prefix_of(key) for key in keys))
    head = struct.pack("<Q", declared) + blob

    return LoRAInspection(
        path=path,
        keys=keys,
        metadata=MappingProxyType(metadata),
        prefixes=prefixes,
        convention=_classify(prefixes),
        model_version=metadata.get("model_version"),
        header_bytes=declared,
        file_bytes=file_bytes,
        shapes=MappingProxyType(shapes),
        header_sha256=hashlib.sha256(head).hexdigest(),
        content_sha256=_file_sha256(path, head, expected=expected),
    )
