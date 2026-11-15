"""Ground truth: the validator against real adapters on the Hub.

**Skipped by default.** These are the only LoRA tests that touch the network,
and they exist so the synthetic fixtures cannot drift into agreeing only with
our own writer.

Run them with::

    LTX25_NETWORK_TESTS=1 .venv/bin/python -m pytest -q tests/test_lora_real_headers.py

They download **no weights**. Each header is fetched with two HTTP range
requests — 8 bytes for the length, then the JSON — and copied into a sparse
local file of the adapter's true size, so the validator runs exactly as it does
on an upload. An HTTP error (a renamed or deleted fixture, a gated repo) FAILS
the test; only a network that cannot be reached at all skips it.

The adapters, all ungated:

* ``SOLRICKS/LTX-2.5-BTS-Movie-Set`` — a real LTX-**2.5** adapter, 1632 modules
  on both the int8 and the bf16 Linears; it must be ACCEPTED with every module
  resolved.
* ``charlied98/ltxv_2b_lora_soft_draw_v1`` — an LTX-**Video-2B** adapter whose
  module names are LTX's; only its 2048 dimensions give it away (stage 6.5).
* ``DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer`` — an in-context LoRA, marked by
  ``reference_downscale_factor`` in its metadata (stage 5.5).
* ``joyfox/LTX-2.3-Transition-LORA`` — an LTX-2.3 adapter that runs on 2.5.
"""

from __future__ import annotations

import json
import os
import struct
import urllib.error
import urllib.request

import pytest

from ltx25_studio import errors
from ltx25_studio.lora import inspect as lora_inspect
from ltx25_studio.lora import validate as lora_validate

pytestmark = pytest.mark.network

HUB = "https://huggingface.co"
NATIVE_PREFIX = "diffusion_model."

POSITIVE = ("SOLRICKS/LTX-2.5-BTS-Movie-Set", "BTSMovieSet_LTX25_v1.safetensors")
WRONG_ARCHITECTURE = (
    "charlied98/ltxv_2b_lora_soft_draw_v1",
    "comfy_lora_weights_step_03000.safetensors",
)
IN_CONTEXT = (
    "DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer",
    "LTX-2.3-22b-IC-LoRA-Colorizer-0.9.safetensors",
)
LTX_2_3 = ("joyfox/LTX-2.3-Transition-LORA", "ltx2.3-transition.safetensors")


@pytest.fixture(scope="session", autouse=True)
def _network_opt_in():
    if not os.environ.get("LTX25_NETWORK_TESTS"):
        pytest.skip("set LTX25_NETWORK_TESTS=1 to check the fixtures against the real Hub")


def _range_read(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.status == 206, f"{url} ignored the range request"
            return response.read(end - start + 1)
    except urllib.error.HTTPError as exc:
        pytest.fail(f"the Hub answered HTTP {exc.code} for {url}: the fixture is gone or gated")
    except urllib.error.URLError as exc:  # pragma: no cover - network shape varies
        pytest.skip(f"cannot reach the Hub: {exc.reason}")


def fetch_header(repo_id, filename):
    """Two range requests. No weights move."""
    url = f"{HUB}/{repo_id}/resolve/main/{filename}"
    (declared,) = struct.unpack("<Q", _range_read(url, 0, 7))
    assert 0 < declared < 100_000_000, f"implausible header length {declared} for {repo_id}"
    return json.loads(_range_read(url, 8, 8 + declared - 1).decode("utf-8")), declared


def local_copy(tmp_path, repo_id, filename):
    """The real header, then a **sparse** payload of the true length."""
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


def test_the_real_2_5_adapter_looks_like_our_fixtures():
    header, _ = fetch_header(*POSITIVE)
    keys = [k for k in header if k != "__metadata__"]
    assert all(k.startswith(NATIVE_PREFIX) for k in keys), "prefix drifted from diffusion_model."
    assert all("lora" in k for k in keys)
    to_q = next(k for k in keys if k.endswith("attn1.to_q.lora_B.weight"))
    assert header[to_q]["shape"][0] == 4096, header[to_q]["shape"]


def test_the_real_2_5_adapter_is_accepted_with_every_module_resolved(tmp_path):
    report = lora_validate.validate_lora(local_copy(tmp_path, *POSITIVE), upload=True)
    assert report.modules_targeted == report.modules_resolved == 1632
    assert not [w for w in report.warnings if "absent" in w], report.warnings


def test_a_real_2_3_adapter_is_accepted_on_2_5(tmp_path):
    report = lora_validate.validate_lora(local_copy(tmp_path, *LTX_2_3), upload=True)
    assert report.modules_targeted == report.modules_resolved


# --------------------------------------------------------------------------
# the negative case — the whole reason stage 6.5 exists
# --------------------------------------------------------------------------


def test_the_real_2b_adapter_passes_every_check_except_dimensions():
    header, _ = fetch_header(*WRONG_ARCHITECTURE)
    keys = [k for k in header if k != "__metadata__"]
    assert all(k.startswith(NATIVE_PREFIX) for k in keys), "passes convention detection"
    assert all("lora" in k for k in keys), "passes diffusers' one built-in check"
    assert not header.get("__metadata__"), "passes the version stage under absent-is-not-fatal"
    to_k = next(k for k in keys if k.endswith("attn1.to_k.lora_A.weight"))
    assert header[to_k]["shape"][1] == 2048, header[to_k]["shape"]


def test_the_real_2b_adapter_is_rejected_at_validate_time(tmp_path):
    """No transformer object: stage 6.5 runs against the checked-in manifest."""
    path = local_copy(tmp_path, *WRONG_ARCHITECTURE)
    with pytest.raises(errors.LoRADimensionError):
        lora_validate.validate_lora(path, upload=True)


# --------------------------------------------------------------------------
# the IC-LoRA case
# --------------------------------------------------------------------------


def test_the_real_ic_lora_carries_its_marker():
    header, _ = fetch_header(*IN_CONTEXT)
    assert "reference_downscale_factor" in header.get("__metadata__", {})


def test_the_real_ic_lora_is_refused(tmp_path):
    with pytest.raises(errors.LoRAInContextError):
        lora_validate.validate_lora(local_copy(tmp_path, *IN_CONTEXT), upload=True)


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
