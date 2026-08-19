"""Header-only safetensors inspection: metadata, keys, shapes, convention.

PLAN §4.3 stages 2, 3 and 5, and §4.6.

**Stage 2 is a hard rule: the tensor payload is never read.** Real adapters run
to ~9 GB and the official one holds 3320 tensors. Materialising one just to
discover it targets the wrong architecture is not an acceptable way to answer
that question.

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

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..errors import LoRAFormatError, LoRAHeaderError, LoRATruncatedError

__all__ = [
    "CONNECTOR_CONVENTIONS",
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

#: Sanity bound on the declared header length. The largest real adapter header
#: measured is ~160 KB for 1152 tensors, so a 3320-tensor adapter lands well
#: under a megabyte. Anything past this is a corrupt prefix, not a big model --
#: most often an HTML error page from a gated repo being read as safetensors.
MAX_HEADER_BYTES = 100_000_000

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
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LoRAFormatError(
            source=source, reason=f"the safetensors header is not valid JSON ({exc})"
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
    """Read a LoRA's header and report what it is. The payload is never read."""
    path = Path(path)
    blob, declared, file_bytes = _read_header_bytes(path)
    header = parse_safetensors_header(blob, source=str(path))

    raw_metadata = header.pop("__metadata__", None)
    # Two distinct shapes mean the same thing: no metadata key at all, and
    # `__metadata__: {}`. Both are common in the wild (PLAN §4.5), so neither is
    # an error and both must normalise to the same empty mapping.
    metadata = {str(k): str(v) for k, v in raw_metadata.items()} if raw_metadata else {}

    keys = tuple(header)
    shapes = {
        name: (entry.get("dtype"), tuple(entry.get("shape", ())))
        for name, entry in header.items()
        if isinstance(entry, dict)
    }
    prefixes = tuple(dict.fromkeys(_prefix_of(key) for key in keys))

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
    )
