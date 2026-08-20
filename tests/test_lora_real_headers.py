"""Ground truth: the synthetic fixtures against three real adapters on the Hub.

**Skipped by default.** These are the only tests in the suite that touch the
network, and they exist for one reason: if a synthetic fixture drifts from what
real adapters actually look like, every LoRA spec becomes self-confirming — the
suite would prove only that our parser agrees with our own writer.

Run them with::

    LTX25_NETWORK_TESTS=1 .venv/bin/python -m pytest -q -m network tests/test_lora_real_headers.py

They cost about a second and download **no weights**. Each adapter's header is
fetched with two HTTP range requests — 8 bytes for the little-endian length, then
the JSON — which is the same technique ``lora/inspect.py`` uses for the Hub route
(PLAN §4.6). ``urllib`` from the stdlib is all it takes; ``requests`` is not a
test dependency.

The three adapters, all ungated, verified by header inspection (PLAN §4.5, and
TEST_PLAN §5):

* ``AhsanHareem/reelbids-ltx25-camera-lora`` — a real LTX-**2.5** adapter, 4096
  hidden dim, ``__metadata__`` present but **empty**. The realistic user upload:
  it must be ACCEPTED, which is what makes "missing model_version is not fatal" a
  correctness requirement rather than a nicety.
* ``charlied98/ltxv_2b_lora_soft_draw_v1`` — an LTX-**Video-2B** adapter. Passes
  convention, pairing, version and diffusers' own check, and its module names
  match ours so it can pass the attach assertion too. Only its 2048 dimensions
  give it away.
* ``DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer`` — an in-context LoRA, marked by
  ``reference_downscale_factor`` in its metadata.
"""

from __future__ import annotations

import json
import os
import struct
import urllib.error
import urllib.request

import pytest
from conftest import NATIVE_PREFIX

from ltx25_studio import errors
from ltx25_studio.lora import inspect as lora_inspect
from ltx25_studio.lora import validate as lora_validate

pytestmark = pytest.mark.network

HUB = "https://huggingface.co"

POSITIVE = ("AhsanHareem/reelbids-ltx25-camera-lora", "lora_weights_step_01250.safetensors")
WRONG_ARCHITECTURE = (
    "charlied98/ltxv_2b_lora_soft_draw_v1",
    "comfy_lora_weights_step_03000.safetensors",
)
IN_CONTEXT = ("DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer", None)


@pytest.fixture(scope="session", autouse=True)
def _network_opt_in():
    if not os.environ.get("LTX25_NETWORK_TESTS"):
        pytest.skip("set LTX25_NETWORK_TESTS=1 to check the fixtures against the real Hub")


def _range_read(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.URLError as exc:  # pragma: no cover - network shape varies
        pytest.skip(f"cannot reach the Hub: {exc}")


def fetch_header(repo_id, filename):
    """Two range requests. No weights move."""
    url = f"{HUB}/{repo_id}/resolve/main/{filename}"
    (declared,) = struct.unpack("<Q", _range_read(url, 0, 7))
    assert 0 < declared < 100_000_000, f"implausible header length {declared} for {repo_id}"
    return json.loads(_range_read(url, 8, 8 + declared - 1).decode("utf-8")), declared


def local_copy(tmp_path, repo_id, filename):
    """A byte-accurate local file: the real header, then a **sparse** payload.

    ``truncate`` allocates no blocks, so a 9 GB adapter costs nothing on disk —
    and running the validator against it is itself a demonstration that the
    payload is never read.
    """
    url = f"{HUB}/{repo_id}/resolve/main/{filename}"
    (declared,) = struct.unpack("<Q", _range_read(url, 0, 7))
    header_bytes = _range_read(url, 8, 8 + declared - 1)
    target = tmp_path / filename
    payload = max(
        entry["data_offsets"][1]
        for key, entry in json.loads(header_bytes).items()
        if key != "__metadata__"
    )
    with open(target, "wb") as handle:
        handle.write(struct.pack("<Q", declared) + header_bytes)
        handle.truncate(8 + declared + payload)
    return target


# --------------------------------------------------------------------------
# the positive case
# --------------------------------------------------------------------------


def test_the_real_2_5_adapter_looks_like_our_fixture():
    header, _ = fetch_header(*POSITIVE)
    keys = [k for k in header if k != "__metadata__"]
    assert all(k.startswith(NATIVE_PREFIX) for k in keys), "prefix drifted from diffusion_model."
    assert all("lora" in k for k in keys)
    assert header.get("__metadata__", None) == {}, (
        "the positive fixture is the empty-__metadata__ shape; if that changed, "
        "tests/conftest.py needs updating"
    )
    to_q = next(k for k in keys if k.endswith("attn1.to_q.lora_B.weight"))
    assert header[to_q]["shape"][0] == 4096, header[to_q]["shape"]


def test_the_real_2_5_adapter_is_accepted(tmp_path):
    path = local_copy(tmp_path, *POSITIVE)
    report = lora_validate.validate_lora(path)
    assert report.model_version is None
    assert report.warnings, "no version signal must warn, not reject"


# --------------------------------------------------------------------------
# the negative case — the whole reason stage 6.5 exists
# --------------------------------------------------------------------------


def test_the_real_2b_adapter_passes_every_check_except_dimensions():
    header, _ = fetch_header(*WRONG_ARCHITECTURE)
    keys = [k for k in header if k != "__metadata__"]
    assert all(k.startswith(NATIVE_PREFIX) for k in keys), "passes convention detection"
    assert all("lora" in k for k in keys), "passes diffusers' one built-in check"
    assert not header.get("__metadata__"), "passes the version stage under absent-is-not-fatal"
    assert any("attn1.to_k" in k for k in keys), "uses the same module names LTX-2.5 does"
    to_k = next(k for k in keys if k.endswith("attn1.to_k.lora_A.weight"))
    assert header[to_k]["shape"][1] == 2048, (
        f"the negative fixture's defining property changed: {header[to_k]['shape']}"
    )


def test_the_real_2b_adapter_is_rejected(tmp_path, fake_transformer):
    path = local_copy(tmp_path, *WRONG_ARCHITECTURE)
    with pytest.raises(errors.LoRADimensionError):
        lora_validate.validate_lora(path, transformer=fake_transformer)


# --------------------------------------------------------------------------
# the IC-LoRA case
# --------------------------------------------------------------------------


def test_the_real_ic_lora_carries_its_marker():
    repo_id, filename = IN_CONTEXT
    if filename is None:
        pytest.skip("IC-LoRA filename not recorded; add it to run this check")
    header, _ = fetch_header(repo_id, filename)
    assert "reference_downscale_factor" in header.get("__metadata__", {})


# --------------------------------------------------------------------------
# the header-only claim, at real scale
# --------------------------------------------------------------------------


def test_inspecting_a_real_adapter_reads_only_its_header(tmp_path, read_spy):
    path = local_copy(tmp_path, *POSITIVE)
    read_spy.watch(path)
    result = lora_inspect.inspect_lora(path)
    assert result.keys
    assert 0 < read_spy.bytes_read < 1_000_000, (
        f"read {read_spy.bytes_read} bytes of a {path.stat().st_size}-byte adapter"
    )
