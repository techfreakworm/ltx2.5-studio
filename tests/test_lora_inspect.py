"""Spec for ``ltx25_studio/lora/inspect.py`` — PLAN §4.3 stages 2, 3 and 5.

Stage 2 is a hard rule, not a preference: **the tensor payload is never read.**
Real adapters are ~9 GB and the official one holds 3320 tensors; materialising
one just to discover it targets the wrong architecture is not an acceptable way
to answer that question.

The contract:

* ``read_safetensors_header(path) -> (header: dict, header_bytes: int)``
* ``inspect_lora(path) -> LoRAInspection`` with fields ``path``, ``keys``,
  ``metadata``, ``prefixes``, ``convention``, ``model_version``,
  ``header_bytes``, ``file_bytes``.
* ``KEY_PREFIXES: Mapping[str, str]`` — the four verified prefixes mapped to a
  convention name. Detection classifies; it does not gatekeep (both conventions
  load — diffusers converts non-diffusers keys itself, PLAN §4.1).
* Malformed input raises a **specific** ``LoRAError`` subclass. Never a bare
  ``Exception``, never ``ValueError``, never a ``KeyError`` escaping from json.
"""

from __future__ import annotations

import json
import struct

import pytest
from conftest import (
    ALL_PREFIXES,
    DIFFUSERS_CONNECTOR_PREFIX,
    DIFFUSERS_PREFIX,
    NATIVE_CONNECTOR_PREFIX,
    NATIVE_PREFIX,
    OFFICIAL_METADATA,
    TRANSFORMER_MODULE_PATHS,
    build_safetensors_bytes,
    lora_tensors,
)

from ltx25_studio.errors import (
    LoRAError,
    LoRAFormatError,
    LoRAHeaderError,
    LoRATruncatedError,
)
from ltx25_studio.lora import inspect as lora_inspect

# --------------------------------------------------------------------------
# stage 2 — header only, payload never touched
# --------------------------------------------------------------------------


def test_inspection_does_not_read_the_tensor_payload(write_safetensors, read_spy):
    """PLAN §4.3 stage 2, proved by counting bytes.

    The fixture's payload is 4 MB. A header-only parser reads a few hundred
    bytes. If this reports zero bytes read, the parser is going through a path
    the spy cannot see (mmap, a C extension) and the header-only guarantee is
    unproven — which is also a failure.
    """
    path = write_safetensors(
        name="fat-lora.safetensors",
        tensors={"diffusion_model.blk.lora_A.weight": ("F32", [1024, 1024])},
        metadata=OFFICIAL_METADATA,
    )
    payload_bytes = 1024 * 1024 * 4
    assert path.stat().st_size > payload_bytes

    read_spy.watch(path)
    result = lora_inspect.inspect_lora(path)

    assert result.model_version == "2.5.0"
    assert read_spy.bytes_read > 0, (
        "no bytes observed through open()/io.open(); header parsing must be a bounded "
        "Python-level read so the payload-never-loaded guarantee is checkable"
    )
    assert read_spy.bytes_read < 100_000, (
        f"read {read_spy.bytes_read} bytes from a {path.stat().st_size}-byte file — "
        "PLAN §4.3 stage 2 forbids materialising the payload"
    )


def test_inspection_reads_far_less_than_the_file_size(write_safetensors, read_spy):
    path = write_safetensors(
        name="fat-lora-2.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS)
        | {"diffusion_model.big.lora_A.weight": ("F32", [1024, 512])},
        metadata=OFFICIAL_METADATA,
    )
    read_spy.watch(path)
    lora_inspect.inspect_lora(path)
    assert read_spy.bytes_read < path.stat().st_size / 10


def test_inspection_result_exposes_no_tensor_values(good_lora):
    result = lora_inspect.inspect_lora(good_lora)
    for field in ("tensors", "state_dict", "weights", "data"):
        assert not hasattr(result, field), (
            f"LoRAInspection.{field} implies the payload was loaded (PLAN §4.3 stage 2)"
        )


def test_header_reader_returns_the_header_and_its_length(good_lora):
    header, header_bytes = lora_inspect.read_safetensors_header(good_lora)
    assert isinstance(header, dict)
    assert header_bytes > 0
    assert "__metadata__" in header
    (declared,) = struct.unpack("<Q", good_lora.read_bytes()[:8])
    assert header_bytes == declared


# --------------------------------------------------------------------------
# metadata (PLAN §4.2)
# --------------------------------------------------------------------------


def test_official_metadata_round_trips(good_lora):
    result = lora_inspect.inspect_lora(good_lora)
    assert dict(result.metadata) == OFFICIAL_METADATA
    assert result.model_version == "2.5.0"


def test_absent_metadata_is_not_an_error(write_safetensors):
    """Most community LoRAs carry no ``__metadata__`` at all. Inspecting one must
    succeed and report the absence, not raise."""
    path = write_safetensors(
        name="community.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata=None,
    )
    result = lora_inspect.inspect_lora(path)
    assert dict(result.metadata) == {}
    assert result.model_version is None


def test_metadata_is_not_reported_as_a_tensor_key(good_lora):
    result = lora_inspect.inspect_lora(good_lora)
    assert "__metadata__" not in result.keys


def test_keys_are_the_tensor_names(write_safetensors):
    tensors = lora_tensors(NATIVE_PREFIX, ("transformer_blocks.0.attn1.to_q",))
    path = write_safetensors(name="two-keys.safetensors", tensors=tensors)
    result = lora_inspect.inspect_lora(path)
    assert set(result.keys) == set(tensors)


def test_file_and_header_sizes_are_reported(good_lora):
    result = lora_inspect.inspect_lora(good_lora)
    assert result.file_bytes == good_lora.stat().st_size
    assert 0 < result.header_bytes < result.file_bytes


# --------------------------------------------------------------------------
# stage 5 — key-convention detection (PLAN §4.1)
# --------------------------------------------------------------------------


def test_all_four_verified_prefixes_are_known():
    assert set(ALL_PREFIXES) <= set(lora_inspect.KEY_PREFIXES), (
        "PLAN §4.1: native LTX uses `diffusion_model.`, diffusers uses `transformer.`, "
        "connector LoRAs use `text_embedding_projection.` (native) / `connectors.` (diffusers)"
    )


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_prefix_is_detected(prefix, write_safetensors):
    path = write_safetensors(
        name="conv.safetensors",
        tensors=lora_tensors(prefix, ("layers.0.proj",)),
        metadata=OFFICIAL_METADATA,
    )
    result = lora_inspect.inspect_lora(path)
    assert prefix in result.prefixes


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_convention_is_classified_for_every_known_prefix(prefix, write_safetensors):
    path = write_safetensors(
        name="conv2.safetensors",
        tensors=lora_tensors(prefix, ("layers.0.proj",)),
        metadata=OFFICIAL_METADATA,
    )
    result = lora_inspect.inspect_lora(path)
    assert result.convention == lora_inspect.KEY_PREFIXES[prefix]
    assert result.convention is not None


def test_native_and_diffusers_conventions_are_distinguished():
    assert (
        lora_inspect.KEY_PREFIXES[NATIVE_PREFIX] != lora_inspect.KEY_PREFIXES[DIFFUSERS_PREFIX]
    ), "the two conventions must be reported distinctly so the UI can say which one it saw"


def test_connector_prefixes_are_recognised_as_connector_targets():
    for prefix in (NATIVE_CONNECTOR_PREFIX, DIFFUSERS_CONNECTOR_PREFIX):
        assert "connector" in lora_inspect.KEY_PREFIXES[prefix].lower(), (
            f"{prefix} targets the connectors (PLAN §4.1); the convention name should say so"
        )


def test_a_real_shaped_adapter_touching_both_transformer_and_connectors(write_safetensors):
    """The official adapter spans the transformer and the text-conditioning path.
    Both prefixes must be reported, and the file must not be called unknown."""
    tensors = lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS)
    tensors |= lora_tensors(NATIVE_CONNECTOR_PREFIX, ("layers.0.proj",))
    path = write_safetensors(name="both.safetensors", tensors=tensors, metadata=OFFICIAL_METADATA)
    result = lora_inspect.inspect_lora(path)
    assert set(result.prefixes) >= {NATIVE_PREFIX, NATIVE_CONNECTOR_PREFIX}
    assert result.convention is not None


def test_unknown_convention_is_reported_with_the_prefixes_actually_found(write_safetensors):
    path = write_safetensors(
        name="unknown-conv.safetensors",
        tensors=lora_tensors("unet.", ("down_blocks.0.resnets.0.conv1",)),
        metadata=OFFICIAL_METADATA,
    )
    result = lora_inspect.inspect_lora(path)
    assert result.convention is None
    assert any("unet" in p for p in result.prefixes), result.prefixes


# --------------------------------------------------------------------------
# stage 3 — malformed input, each with a DISTINCT specific error
# --------------------------------------------------------------------------


def test_a_text_file_raises_a_format_error(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("these are my notes about the lora I meant to upload\n")
    with pytest.raises(LoRAFormatError) as excinfo:
        lora_inspect.inspect_lora(path)
    assert "notes.txt" in str(excinfo.value)


def test_an_empty_file_raises_a_format_error(tmp_path):
    path = tmp_path / "empty.safetensors"
    path.write_bytes(b"")
    with pytest.raises(LoRAFormatError):
        lora_inspect.inspect_lora(path)


def test_a_file_shorter_than_the_length_prefix_raises_a_format_error(tmp_path):
    path = tmp_path / "stub.safetensors"
    path.write_bytes(b"\x01\x02\x03")
    with pytest.raises(LoRAFormatError):
        lora_inspect.inspect_lora(path)


def test_a_header_that_is_not_a_json_object_raises_a_format_error(tmp_path):
    blob = json.dumps([1, 2, 3]).encode("utf-8")
    path = tmp_path / "listheader.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    with pytest.raises(LoRAFormatError):
        lora_inspect.inspect_lora(path)


def test_a_header_that_is_not_valid_json_raises_a_format_error(tmp_path):
    blob = b"{not json at all,,,"
    path = tmp_path / "badjson.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    with pytest.raises(LoRAFormatError):
        lora_inspect.inspect_lora(path)


def test_a_truncated_file_raises_a_truncation_error(tmp_path):
    """A valid header length prefix, then a file that stops mid-header."""
    data = build_safetensors_bytes(
        lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS), OFFICIAL_METADATA
    )
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(data[: 8 + 24])
    with pytest.raises(LoRATruncatedError) as excinfo:
        lora_inspect.inspect_lora(path)
    assert "truncated.safetensors" in str(excinfo.value)


@pytest.mark.parametrize("declared", [0, 2**60])
def test_a_corrupt_header_length_raises_a_header_error(declared, tmp_path):
    data = build_safetensors_bytes(
        lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        OFFICIAL_METADATA,
        header_length_override=declared,
    )
    path = tmp_path / "corrupt-length.safetensors"
    path.write_bytes(data)
    with pytest.raises(LoRAHeaderError) as excinfo:
        lora_inspect.inspect_lora(path)
    assert "corrupt-length.safetensors" in str(excinfo.value)


def test_the_three_malformed_cases_raise_three_distinct_types(tmp_path):
    """PLAN §4.3 stage 3: "specific message: not-safetensors vs unparseable-header".
    One catch-all error for every kind of broken file is not a diagnosis."""
    text = tmp_path / "prose.txt"
    text.write_text("not a model")

    good = build_safetensors_bytes(
        lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS), OFFICIAL_METADATA
    )
    truncated = tmp_path / "cut.safetensors"
    truncated.write_bytes(good[: 8 + 24])

    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(
        build_safetensors_bytes(
            lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
            OFFICIAL_METADATA,
            header_length_override=2**60,
        )
    )

    raised = []
    for path in (text, truncated, corrupt):
        with pytest.raises(LoRAError) as excinfo:
            lora_inspect.inspect_lora(path)
        raised.append(type(excinfo.value))

    assert len(set(raised)) == 3, f"expected three distinct error types, got {raised}"


@pytest.mark.parametrize(
    "name,payload",
    [
        ("prose.safetensors", b"this is not a safetensors file at all, it is prose\n"),
        ("zipped.safetensors", b"PK\x03\x04" + b"\x00" * 64),
    ],
)
def test_garbage_named_safetensors_still_fails_loudly(name, payload, tmp_path):
    path = tmp_path / name
    path.write_bytes(payload)
    with pytest.raises(LoRAError):
        lora_inspect.inspect_lora(path)


def test_a_missing_file_raises_our_error_not_a_bare_oserror(tmp_path):
    with pytest.raises(LoRAError):
        lora_inspect.inspect_lora(tmp_path / "does-not-exist.safetensors")


# --------------------------------------------------------------------------
# the official adapter's license is a wall of text
# --------------------------------------------------------------------------

LONG_LICENSE = "LTX-2.5 Community License Agreement. " + ("Section text. " * 800)


def test_a_very_long_license_value_parses(write_safetensors):
    """The official adapter's ``__metadata__`` carries the full licence text.
    Reading it must not be a special case."""
    path = write_safetensors(
        name="licensed.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata=OFFICIAL_METADATA | {"license": LONG_LICENSE},
    )
    result = lora_inspect.inspect_lora(path)
    assert result.metadata["license"] == LONG_LICENSE, "nothing may be silently truncated"
    assert result.model_version == "2.5.0"


def test_an_error_message_does_not_dump_the_license_text(write_safetensors):
    """An error is read by a human. Ten kilobytes of licence text in it is not a
    message, it is a denial of service on the person reading it."""
    from ltx25_studio.errors import LoRAError
    from ltx25_studio.lora import validate as lora_validate

    path = write_safetensors(
        name="licensed-and-wrong.safetensors",
        tensors=lora_tensors("unet.", ("down_blocks.0.resnets.0.conv1",)),
        metadata=OFFICIAL_METADATA | {"license": LONG_LICENSE},
    )
    with pytest.raises(LoRAError) as excinfo:
        lora_validate.validate_lora(path)
    message = str(excinfo.value)
    assert LONG_LICENSE not in message
    assert len(message) < 1000, f"error message is {len(message)} characters long"


# --------------------------------------------------------------------------
# the fixtures are honest
# --------------------------------------------------------------------------


def test_synthetic_fixtures_are_readable_by_the_real_safetensors_library(good_lora):
    """Guards against the test suite validating a format only it believes in.

    Skipped until ``safetensors`` is installed; it is not a test-time dependency.
    """
    safetensors = pytest.importorskip("safetensors")
    with safetensors.safe_open(str(good_lora), framework="numpy") as handle:
        assert handle.metadata() == OFFICIAL_METADATA
        assert set(handle.keys()) == set(lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS))


# --------------------------------------------------------------------------
# F53 — a cheap content key for refusing duplicate staging
# --------------------------------------------------------------------------


def test_the_header_hash_is_the_sha256_of_the_length_prefix_and_header(good_lora):
    import hashlib

    data = good_lora.read_bytes()
    (declared,) = struct.unpack("<Q", data[:8])
    expected = hashlib.sha256(data[: 8 + declared]).hexdigest()
    assert lora_inspect.inspect_lora(good_lora).header_sha256 == expected


def test_the_header_hash_reads_no_payload(good_lora, read_spy):
    read_spy.watch(good_lora)
    lora_inspect.inspect_lora(good_lora)
    data_len = good_lora.stat().st_size
    assert read_spy.bytes_read < data_len, "hashing must not read the tensor payload"


def test_the_header_hash_survives_the_fork_transport(good_lora):
    import pickle

    inspection = lora_inspect.inspect_lora(good_lora)
    assert pickle.loads(pickle.dumps(inspection)).header_sha256 == inspection.header_sha256


# --------------------------------------------------------------------------
# lora-8 — the header bound, and every entry checked the way safetensors checks it
# --------------------------------------------------------------------------


def raw_adapter(path, header, payload=b""):
    """A file holding exactly ``header`` (an object, or raw JSON bytes) and ``payload``."""
    blob = header if isinstance(header, bytes) else json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)
    return path


def f32(begin, end, shape=(2,)):
    return {"dtype": "F32", "shape": list(shape), "data_offsets": [begin, end]}


def test_the_header_bound_is_the_remote_readers_16_mb():
    """The largest real adapter header is ~0.4 MB; the Hub range reader already
    refuses anything over 16 MB, and an upload gets the same bound."""
    from ltx25_studio.lora import manifest

    assert lora_inspect.MAX_HEADER_BYTES == manifest.MAX_REMOTE_HEADER_BYTES == 16_000_000


def test_a_well_formed_20_mb_header_is_refused_before_it_is_read(tmp_path, read_spy):
    """Parsing it costs hundreds of MB of RAM in the web process for nothing."""
    path = raw_adapter(tmp_path / "padded.safetensors", {"__metadata__": {"pad": "x" * 20_000_000}})
    read_spy.watch(path)
    with pytest.raises(LoRAHeaderError) as excinfo:
        lora_inspect.inspect_lora(path)
    assert "padded.safetensors" in str(excinfo.value)
    assert read_spy.bytes_read < 64, f"read {read_spy.bytes_read} bytes of a refused header"


#: Headers the safetensors loader refuses, with an 8-byte payload behind each.
MALFORMED_ENTRIES = {
    "shape is a number": {"t": {"dtype": "F32", "shape": 2, "data_offsets": [0, 8]}},
    "shape holds a float": {"t": {"dtype": "F32", "shape": [2.0], "data_offsets": [0, 8]}},
    "shape holds a negative": {"t": {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]}},
    "shape holds a boolean": {"t": {"dtype": "U8", "shape": [True], "data_offsets": [0, 8]}},
    "dtype is a number": {"t": {"dtype": 5, "shape": [2], "data_offsets": [0, 8]}},
    "dtype is not one safetensors knows": {
        "t": {"dtype": "bf16", "shape": [4], "data_offsets": [0, 8]}
    },
    "no dtype": {"t": {"shape": [2], "data_offsets": [0, 8]}},
    "no shape": {"t": {"dtype": "F32", "data_offsets": [0, 8]}},
    "no data offsets": {"t": {"dtype": "F32", "shape": [2]}},
    "three data offsets": {"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8, 8]}},
    "a float data offset": {"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8.0]}},
    "an entry that is a list": {"t": [1, 2]},
    "an entry that is text": {"t": "F32"},
    "metadata that is a list": {"__metadata__": ["x"], "t": f32(0, 8)},
    "a metadata value that is a number": {"__metadata__": {"k": 1}, "t": f32(0, 8)},
    "a metadata value that is an object": {"__metadata__": {"k": {"x": "y"}}, "t": f32(0, 8)},
}


@pytest.mark.parametrize("header", MALFORMED_ENTRIES.values(), ids=MALFORMED_ENTRIES)
def test_a_malformed_entry_is_a_format_error_not_a_python_error(header, tmp_path):
    path = raw_adapter(tmp_path / "malformed.safetensors", header, b"\x01" * 8)
    with pytest.raises(LoRAFormatError) as excinfo:
        lora_inspect.inspect_lora(path)
    message = str(excinfo.value)
    assert "malformed.safetensors" in message
    assert len(message) < 600, f"a {len(message)}-character message is not one a person reads"


def test_deeply_nested_json_is_a_format_error_not_a_recursion_error(tmp_path):
    blob = b'{"__metadata__": {"k": ' + b"[" * 100_000 + b"]" * 100_000 + b"}}"
    path = raw_adapter(tmp_path / "nested.safetensors", blob)
    with pytest.raises(LoRAFormatError) as excinfo:
        lora_inspect.inspect_lora(path)
    assert "nested.safetensors" in str(excinfo.value)


def test_a_hostile_value_is_not_quoted_in_full(tmp_path):
    header = {"t": {"dtype": "F32", "shape": [1.5] * 200_000, "data_offsets": [0, 8]}}
    path = raw_adapter(tmp_path / "long.safetensors", header, b"\x01" * 8)
    with pytest.raises(LoRAFormatError) as excinfo:
        lora_inspect.inspect_lora(path)
    assert len(str(excinfo.value)) < 600


@pytest.mark.parametrize("header", MALFORMED_ENTRIES.values(), ids=MALFORMED_ENTRIES)
def test_the_safetensors_loader_refuses_every_entry_we_refuse(header, tmp_path):
    """Refusing at Validate is only right if the worker's loader would refuse too."""
    safetensors = pytest.importorskip("safetensors")
    path = raw_adapter(tmp_path / "malformed.safetensors", header, b"\x01" * 8)
    with (
        pytest.raises(Exception),  # noqa: B017 -- SafetensorError's import path varies
        safetensors.safe_open(str(path), framework="numpy") as handle,
    ):
        list(handle.keys())


#: Headers the safetensors loader accepts, each with the payload it needs.
ACCEPTED_HEADERS = {
    "an extra field in an entry": ({"t": {**f32(0, 8), "note": "x"}}, 8),
    "null metadata": ({"__metadata__": None, "t": f32(0, 8)}, 8),
    "keys out of offset order": ({"b": f32(8, 16), "a": f32(0, 8)}, 16),
    "a scalar": ({"t": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}, 4),
    "a zero-size tensor": (
        {"z": {"dtype": "F32", "shape": [0, 4], "data_offsets": [0, 0]}, "t": f32(0, 8)},
        8,
    ),
    "an empty header": ({}, 0),
    "metadata only": ({"__metadata__": {"k": "v"}}, 0),
    "a 4-bit dtype": ({"t": {"dtype": "F4", "shape": [4], "data_offsets": [0, 2]}}, 2),
}


@pytest.mark.parametrize(("header", "size"), ACCEPTED_HEADERS.values(), ids=ACCEPTED_HEADERS)
def test_headers_the_safetensors_loader_accepts_are_accepted(header, size, tmp_path):
    path = raw_adapter(tmp_path / "fine.safetensors", header, b"\x01" * size)
    result = lora_inspect.inspect_lora(path)
    assert set(result.keys) == {k for k in header if k != "__metadata__"}
    safetensors = pytest.importorskip("safetensors")
    with safetensors.safe_open(str(path), framework="numpy") as handle:
        assert set(handle.keys()) == set(result.keys)
