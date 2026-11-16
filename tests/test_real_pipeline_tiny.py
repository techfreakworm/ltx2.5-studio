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

import contextlib
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


# --------------------------------------------------------------------------
# in-process: the stage graph on the real tiny classes, through generate()
#
# CPU, fp32, the tiny decoder with its default processor. Stage 2 at 64x64x9
# returns 128x128x9, the tiny decoder's minimum.
# --------------------------------------------------------------------------

STAGE_2_MODES = tuple(m for m, spec in modes.MODES.items() if not spec.skip_stage_2)


def load_smoke():
    """``scripts/local_smoke.py`` as a module (it is a script, not a package)."""
    import importlib.util

    if "local_smoke" in sys.modules:
        return sys.modules["local_smoke"]
    spec = importlib.util.spec_from_file_location("local_smoke", SMOKE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_smoke"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_cpu():
    return load_smoke().tiny_stack(device="cpu")


@pytest.fixture
def pipeline_calls(monkeypatch):
    """Every kwargs dict that reaches a real diffusers LTX-2 pipeline, in order."""
    from diffusers import LTX2ConditionPipeline, LTX2Pipeline

    seen: list[tuple[str, dict]] = []
    for cls in (LTX2ConditionPipeline, LTX2Pipeline):
        original = cls.__call__

        def spy(self, *args, _original=original, **kwargs):
            seen.append((type(self).__name__, dict(kwargs)))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__call__", spy)
    return seen


def run_tiny(tiny, mode_id, out, **overrides):
    from ltx25_studio import generate

    smoke = load_smoke()
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    width = overrides.get("width", 64)
    frames = overrides.get("num_frames", 9)
    request = dict(
        mode=mode_id,
        seed=7,
        width=64,
        height=64,
        num_frames=9,
        frame_rate=24.0,
        **smoke._media_for(mode_id, media, width, frames),
    )
    request.update(overrides)
    with tiny_snapshot_env(tiny):
        return generate.generate(
            **request,
            pipeline_factory=lambda pipeline: tiny[pipeline],
            output_root=out / "outputs",
            retention_bytes=10**9,
        )


@contextlib.contextmanager
def tiny_snapshot_env(tiny):
    """``LTX25_MODEL_SNAPSHOT_DIR`` at the tiny stack's snapshot, for the length of a run.

    The gate reads the decoder's config before a run; with the tiny stack it must
    read the tiny decoder's, not whatever real checkpoint the HF cache holds.
    """
    name = "LTX25_MODEL_SNAPSHOT_DIR"
    before = os.environ.get(name)
    os.environ[name] = str(tiny["snapshot"])
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = before


@pytest.mark.parametrize("mode_id", STAGE_2_MODES)
def test_every_stage_2_mode_runs_on_the_real_tiny_pipelines(
    tiny_cpu, mode_id, tmp_path, pipeline_calls
):
    from verify_streams import probe_streams

    result = run_tiny(tiny_cpu, mode_id, tmp_path, enable_stage_2=True)

    denoises = [kwargs for _cls, kwargs in pipeline_calls]
    assert len(denoises) == 2, [sorted(k) for k in denoises]
    stage_2 = denoises[1]
    assert stage_2["num_frames"] == 9, stage_2.get("num_frames")
    if modes.get_mode(mode_id).pipeline == "condition":
        assert (stage_2["width"], stage_2["height"]) == (128, 128)
    actual = probe_streams(result.output_path)
    assert (actual["width"], actual["height"]) == (128, 128), actual


def test_stage_2_image_to_video_is_sized_from_the_upsampled_latents(tiny_cpu, tmp_path):
    """Was: ``ValueError: Provided `latents` tensor has shape ... expected ...``.

    The condition pipeline sized its mask from its 512x768 defaults.
    """
    result = run_tiny(tiny_cpu, "image_to_video", tmp_path, enable_stage_2=True)

    assert result.output_path.is_file()


def test_stage_2_audio_to_video_does_not_pass_noise_scale_twice(tiny_cpu, tmp_path):
    """Was: ``TypeError: got multiple values for keyword argument 'noise_scale'``."""
    result = run_tiny(tiny_cpu, "audio_to_video", tmp_path, enable_stage_2=True)

    assert result.output_path.is_file()
