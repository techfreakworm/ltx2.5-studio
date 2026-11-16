"""The real-pipeline tier: the pinned diffusers pipelines, not fakes.

The offline suite was green at 1102 tests while five blockers were live,
because it only ever exercised fakes. These specs run the real LTX-2.5 classes
on a public ~95 MB random-weight checkpoint through the app's own
``generate()`` -- the seam the UI uses -- via ``scripts/local_smoke.py`` in a
subprocess, so this process's state (the cached pipeline, the GPU dispatch, the
environment) is untouched.

FULL env only. Skips, with the reason, when the tiny snapshot is not in the HF
cache: tests never download (``python scripts/local_smoke.py --fetch`` does).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from conftest import REPO_ROOT

pytest.importorskip("torch")
pytest.importorskip("diffusers")

from ltx25_studio import model_files, modes  # noqa: E402

SMOKE = REPO_ROOT / "scripts" / "local_smoke.py"


def _tiny_snapshot_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            model_files.TINY_REPO_ID, revision=model_files.TINY_REVISION, local_files_only=True
        )
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _tiny_snapshot_cached(),
    reason=(
        f"{model_files.TINY_REPO_ID}@{model_files.TINY_REVISION[:8]} is not in the HF cache; "
        f"run `python scripts/local_smoke.py --fetch` once (about 95 MB)"
    ),
)


def _smoke(tmp_path, *args) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
    finished = subprocess.run(
        [sys.executable, str(SMOKE), *args, "--out", str(tmp_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    names = ("local_smoke_cpu.json", "local_smoke_mps.json", "rope_parity.json")
    reports = [tmp_path / name for name in names if (tmp_path / name).is_file()]
    assert reports, (
        f"local_smoke wrote no report (exit {finished.returncode}):\n"
        f"{finished.stdout[-2000:]}\n{finished.stderr[-3000:]}"
    )
    payload = json.loads(reports[0].read_text())
    payload["_exit"] = finished.returncode
    payload["_stdout"] = finished.stdout[-3000:]
    return payload


@pytest.fixture(scope="module")
def cpu_modes(tmp_path_factory):
    return _smoke(tmp_path_factory.mktemp("cpu"), "--device", "cpu", "--only", "mode:")


def test_every_mode_runs_on_the_cpu_through_generate(cpu_modes):
    rows = {row["mode"]: row for row in cpu_modes["cases"]}

    assert set(rows) == set(modes.MODES), sorted(set(modes.MODES) ^ set(rows))
    failed = {
        m: (r.get("error_type"), r.get("error")) for m, r in rows.items() if r["outcome"] != "PASS"
    }
    assert not failed, failed
    assert cpu_modes["_exit"] == 0, cpu_modes["_stdout"]


@pytest.mark.parametrize("mode_id", list(modes.MODES))
def test_each_artifact_carries_the_streams_its_mode_declares(cpu_modes, mode_id):
    """Re-probed here, independently of the script's own verdict."""
    from verify_streams import expected_streams, probe_streams

    row = next(row for row in cpu_modes["cases"] if row["mode"] == mode_id)
    actual = probe_streams(row["artifact"])
    expected = expected_streams(mode_id)

    assert actual is not None, f"{row['artifact']} could not be opened"
    assert actual["video"] == expected["video"] and actual["audio"] == expected["audio"], (
        mode_id,
        actual,
    )
    if expected["audio"]:
        assert actual["audio_channels"] == 2 and actual["sampling_rate"] == 48000, actual
    if expected["video"]:
        assert (actual["width"], actual["height"]) == (128, 128), actual


def test_the_cpu_run_used_the_resolved_device(cpu_modes):
    assert cpu_modes["meta"]["device"].startswith("cpu"), cpu_modes["meta"]


def test_one_mode_runs_on_mps(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device on this machine")

    payload = _smoke(tmp_path, "--device", "mps", "--only", "mode:text_to_audio_video")

    rows = {row["case"]: row for row in payload["cases"]}
    assert rows["mode:text_to_audio_video"]["outcome"] == "PASS", rows
    assert rows["mode:text_to_audio_video"]["streams"] == "MATCHES", rows
    refusal = rows["mps_self_test:enforce"]
    if "FAILED" in payload["meta"]["mps_self_test"]:
        assert refusal["outcome"] == "PASS" and refusal["error_type"] == "MpsNumericsError", refusal
        notes = rows["mode:text_to_audio_video"]["warnings"]
        assert any("Unverified" in note for note in notes), (
            f"under LTX25_MPS_SELF_TEST=warn every result must carry the warning: {notes}"
        )
    assert payload["_exit"] == 0, payload["_stdout"]


def test_the_rope_handling_matches_the_float64_cpu_reference(tmp_path):
    """One transformer forward: cpu/fp32 with float64 RoPE vs MPS with ours.

    The bound is relative to the output's own magnitude: fp32 MPS kernels
    measured ~6e-5 relative per transformer call in the probe.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device on this machine")

    payload = _smoke(tmp_path, "--rope-parity")

    shipped = next(v for k, v in payload["results"].items() if "shipped" in k)
    for stream, numbers in shipped.items():
        relative = numbers["max_abs"] / numbers["ref_abs_max"]
        assert relative < 1e-3, (stream, numbers)
