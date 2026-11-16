"""The transformer's ``nn.Linear`` shapes, checked in, so stage 6.5 needs no weights.

``transformer_linear_shapes.json`` maps every Linear weight of the pinned
``Lightricks/LTX-2.5-Diffusers`` transformer to ``[out_features, in_features]``
(the weight tensor's own shape). It was derived from the four shard headers at
:data:`~ltx25_studio.model_files.REVISION` -- 2-D ``.weight`` tensors, which in
this transformer are exactly its 1660 Linears -- and carries that revision and a
sha256 of its own canonical JSON.

Regenerate or check it against the Hub (header range reads only, ~0.5 MB)::

    python -m ltx25_studio.lora.manifest           # compare with the checked-in file
    python -m ltx25_studio.lora.manifest --write   # rewrite it

Stdlib only at import. The Hub is touched only by :func:`derive_manifest` and
the header range readers it uses (also used by the opt-in network tests).
"""

from __future__ import annotations

import functools
import hashlib
import json
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..errors import LoRAError
from ..model_files import REPO_ID, REVISION
from .inspect import HEADER_LENGTH_BYTES, MAX_HEADER_BYTES

__all__ = [
    "MANIFEST_COUNT",
    "MANIFEST_PATH",
    "MANIFEST_SHA256",
    "LinearShapes",
    "LoRAManifestDriftError",
    "build_document",
    "canonical_sha256",
    "derive_manifest",
    "linear_shapes_from_headers",
    "load_manifest",
    "read_remote_header",
    "read_remote_header_bytes",
    "read_remote_json",
    "render_document",
]

MANIFEST_PATH = Path(__file__).with_name("transformer_linear_shapes.json")
SUBFOLDER = "transformer"
SHARD_INDEX = f"{SUBFOLDER}/diffusion_pytorch_model.safetensors.index.json"

#: Linears in the pinned transformer (also ``tests/test_int8_skip_modules.py``).
MANIFEST_COUNT = 1660

#: sha256 of the checked-in document's canonical JSON (see :func:`canonical_sha256`).
MANIFEST_SHA256 = "192798e9c476935ba44cc3eeafec0607b99666989be4d528cb0ef6865ae2da18"

_FORMAT = 1
_SHAPE_ORDER = "[out_features, in_features]"


class LoRAManifestDriftError(LoRAError):
    """The live transformer and the checked-in shape table disagree.

    Not a problem with the user's adapter: the deployed model no longer matches
    the table stage 6.5 checks against, so every verdict it gives is suspect.
    """

    def __init__(self, *, module, manifest, live):
        self.module = module
        self.manifest = manifest
        self.live = live
        super().__init__(
            f"The app's shape table for the LTX-2.5 transformer does not match the loaded "
            f"model at {module} (table: {manifest}, loaded model: {live}). This is a problem "
            f"with this deployment, not with your adapter; the adapter check cannot be "
            f"trusted until the table is regenerated."
        )


@dataclass(frozen=True)
class LinearShapes:
    """``module name -> (in_features, out_features)`` for the pinned transformer."""

    repo_id: str
    revision: str
    sha256: str
    dims: Mapping[str, tuple[int, int]]

    def __len__(self) -> int:
        return len(self.dims)


def linear_shapes_from_headers(headers: Iterable[Mapping]) -> dict[str, list[int]]:
    """``{module: [out, in]}`` for every 2-D ``.weight`` tensor in the given headers."""
    shapes: dict[str, list[int]] = {}
    for header in headers:
        for name, entry in header.items():
            if name == "__metadata__" or not isinstance(entry, Mapping):
                continue
            shape = entry.get("shape") or ()
            if name.endswith(".weight") and len(shape) == 2:
                shapes[name[: -len(".weight")]] = [int(shape[0]), int(shape[1])]
    return dict(sorted(shapes.items()))


def build_document(linears: Mapping[str, list[int]], *, revision: str = REVISION) -> dict:
    """The manifest document, its checksum included."""
    document = {
        "format": _FORMAT,
        "repo_id": REPO_ID,
        "revision": revision,
        "subfolder": SUBFOLDER,
        "shape": _SHAPE_ORDER,
        "count": len(linears),
        "linears": {name: [int(o), int(i)] for name, (o, i) in sorted(linears.items())},
    }
    document["sha256"] = canonical_sha256(document)
    return document


def canonical_sha256(document: Mapping) -> str:
    """sha256 of the document without its ``sha256`` field, as sorted compact JSON."""
    body = {key: value for key, value in document.items() if key != "sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def render_document(document: Mapping) -> str:
    """The checked-in text: valid JSON, one Linear per line so a diff is readable."""
    keys = sorted(document)
    lines = ["{"]
    for position, key in enumerate(keys):
        tail = "," if position < len(keys) - 1 else ""
        if key == "linears":
            items = sorted(document["linears"].items())
            lines.append(' "linears": {')
            for index, (name, (out_features, in_features)) in enumerate(items):
                comma = "," if index < len(items) - 1 else ""
                lines.append(f"  {json.dumps(name)}: [{out_features}, {in_features}]{comma}")
            lines.append(" }" + tail)
        else:
            lines.append(f" {json.dumps(key)}: {json.dumps(document[key])}{tail}")
    lines.append("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Hub header range reads: 8 bytes, then the JSON header. Never tensor data, and
# a server that ignores ``Range`` is refused before its body is read.
# --------------------------------------------------------------------------

#: Refuse a declared header larger than this: the bound an uploaded file gets.
#: The largest real LTX adapter header measured is ~0.4 MB (the official 2.5
#: adapter, with a 34 KB licence in its metadata); a transformer shard header
#: is ~0.13 MB.
MAX_REMOTE_HEADER_BYTES = MAX_HEADER_BYTES

_TIMEOUT_SECONDS = 30.0


def _range(url: str, start: int, end: int, headers: dict) -> bytes:
    """Bytes ``start..end`` inclusive, or an error. Never more than asked for."""
    from huggingface_hub.utils import get_session, hf_raise_for_status

    wanted = end - start + 1
    request_headers = {**headers, "Range": f"bytes={start}-{end}"}
    with get_session().stream(
        "GET", url, headers=request_headers, follow_redirects=True, timeout=_TIMEOUT_SECONDS
    ) as response:
        hf_raise_for_status(response)
        if response.status_code != 206:
            raise RuntimeError(
                f"{url} answered a range request with HTTP {response.status_code}, not 206; "
                f"refusing to read a whole file to get its header"
            )
        data = bytearray()
        for chunk in response.iter_bytes():
            data += chunk
            if len(data) >= wanted:
                break
    if len(data) < wanted:
        raise RuntimeError(f"{url} returned {len(data)} of the {wanted} bytes requested")
    return bytes(data[:wanted])


def read_remote_header_bytes(
    repo_id: str, filename: str, *, revision: str | None = None, token=None
) -> bytes:
    """The file's first ``8 + N`` bytes: the length prefix and the JSON header."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    headers = build_hf_headers(token=token)
    prefix = _range(url, 0, HEADER_LENGTH_BYTES - 1, headers)
    (declared,) = struct.unpack("<Q", prefix)
    if not 0 < declared <= MAX_REMOTE_HEADER_BYTES:
        raise RuntimeError(
            f"{repo_id}/{filename} declares a {declared}-byte header; that is not a "
            f"safetensors header this reader will fetch"
        )
    return prefix + _range(url, HEADER_LENGTH_BYTES, HEADER_LENGTH_BYTES + declared - 1, headers)


def read_remote_header(
    repo_id: str, filename: str, *, revision: str | None = None, token=None
) -> dict:
    """The parsed JSON header (``__metadata__`` included) of a Hub safetensors file."""
    raw = read_remote_header_bytes(repo_id, filename, revision=revision, token=token)
    return json.loads(raw[HEADER_LENGTH_BYTES:].decode("utf-8"))


def read_remote_json(
    repo_id: str, filename: str, *, revision: str | None = None, token=None, max_bytes=2_000_000
) -> dict:
    """A small JSON file from the Hub, read into memory and size-capped."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    with get_session().stream(
        "GET",
        url,
        headers=build_hf_headers(token=token),
        follow_redirects=True,
        timeout=_TIMEOUT_SECONDS,
    ) as response:
        hf_raise_for_status(response)
        data = bytearray()
        for chunk in response.iter_bytes():
            data += chunk
            if len(data) > max_bytes:
                raise RuntimeError(f"{repo_id}/{filename} is larger than {max_bytes} bytes")
    return json.loads(bytes(data).decode("utf-8"))


def derive_manifest(
    *,
    revision: str = REVISION,
    read_json: Callable | None = None,
    read_header: Callable | None = None,
) -> dict:
    """Re-derive the document from the Hub: the shard index, then each shard's header.

    ``read_json(repo_id, filename, revision=...)`` and
    ``read_header(repo_id, filename, revision=...)`` are injectable; the
    defaults are the range readers above.
    """
    read_json = read_json or read_remote_json
    read_header = read_header or read_remote_header
    index = read_json(REPO_ID, SHARD_INDEX, revision=revision)
    shards = sorted(set(index["weight_map"].values()))
    headers = [read_header(REPO_ID, f"{SUBFOLDER}/{shard}", revision=revision) for shard in shards]
    return build_document(linear_shapes_from_headers(headers), revision=revision)


@functools.lru_cache(maxsize=1)
def load_manifest() -> LinearShapes:
    """The checked-in table, verified before use. Raises if it was edited or truncated."""
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recomputed = canonical_sha256(document)
    problems = []
    if recomputed != document.get("sha256") or recomputed != MANIFEST_SHA256:
        problems.append(f"checksum {recomputed} does not match the recorded one")
    if document.get("revision") != REVISION:
        problems.append(f"revision {document.get('revision')} is not the pinned {REVISION}")
    if len(document.get("linears", {})) != MANIFEST_COUNT:
        problems.append(f"{len(document.get('linears', {}))} entries, expected {MANIFEST_COUNT}")
    if problems:
        raise RuntimeError(
            f"{MANIFEST_PATH.name} is not the table this build expects: " + "; ".join(problems)
        )
    dims = {name: (int(i), int(o)) for name, (o, i) in document["linears"].items()}
    return LinearShapes(
        repo_id=document["repo_id"],
        revision=document["revision"],
        sha256=recomputed,
        dims=MappingProxyType(dims),
    )


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m ltx25_studio.lora.manifest")
    parser.add_argument("--write", action="store_true", help="rewrite the checked-in file")
    args = parser.parse_args(argv)

    derived = derive_manifest()
    text = render_document(derived)
    print(
        f"derived {derived['count']} Linears at {derived['revision']}, sha256 {derived['sha256']}"
    )
    if args.write:
        MANIFEST_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {MANIFEST_PATH}")
        return 0
    current = MANIFEST_PATH.read_text(encoding="utf-8") if MANIFEST_PATH.exists() else ""
    same = current == text
    print(f"checked-in {MANIFEST_PATH.name}: {'identical' if same else 'DIFFERENT'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
