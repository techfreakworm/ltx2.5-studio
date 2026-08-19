"""Spec for output retention in ``ltx25_studio/outputs.py`` — PLAN §2.7, §9.1.

The 2.3 Space filled its disk and went unhealthy. Retention is therefore not
housekeeping, it is an availability requirement: ``outputs/`` is capped at 20 GB,
bounded-bytes, oldest-first, swept on startup **and** after each run, logging
what it evicted.

The contract:

* ``enforce_retention(root, *, max_bytes) -> tuple[Path, ...]`` — the run
  directories it evicted, oldest first, so the caller can log them. Sweeping
  silently is how a user discovers their outputs are gone by not finding them.
* Eviction is **whole run directories**, never individual files inside one. Half
  a run left on disk is worse than none of it.
* **The newest run is never evicted**, even when it alone exceeds the cap.
  Deleting the video the user just made is not a retention policy.
"""

from __future__ import annotations

import os

import pytest

from ltx25_studio import config, outputs


def make_run(root, name, size_bytes, age_seconds):
    """A run directory with one artifact, aged by ``age_seconds``."""
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "output.mp4"
    artifact.write_bytes(b"\x00" * size_bytes)
    stamp = 1_700_000_000 - age_seconds
    os.utime(artifact, (stamp, stamp))
    os.utime(run_dir, (stamp, stamp))
    return run_dir


# --------------------------------------------------------------------------
# the configured bound
# --------------------------------------------------------------------------


def test_the_default_retention_bound_is_twenty_gigabytes():
    assert config.DEFAULT_OUTPUT_RETENTION_GB == 20.0


def test_the_retention_bound_is_overridable_by_environment(monkeypatch):
    """PLAN §2.7: "a single constant in ``config.py`` overridable by env" — so a
    live Space can be retuned without a push."""
    monkeypatch.setenv(config.ENV_OUTPUT_RETENTION_GB, "5")
    assert config.load_config().output_retention_gb == 5.0


# --------------------------------------------------------------------------
# eviction
# --------------------------------------------------------------------------


def test_nothing_is_evicted_below_the_cap(tmp_path):
    keep = make_run(tmp_path, "run-a", 1024, age_seconds=300)
    evicted = outputs.enforce_retention(tmp_path, max_bytes=1_000_000)
    assert evicted == ()
    assert (keep / "output.mp4").exists()


def test_the_oldest_run_is_evicted_first(tmp_path):
    oldest = make_run(tmp_path, "run-old", 4096, age_seconds=9000)
    middle = make_run(tmp_path, "run-mid", 4096, age_seconds=600)
    newest = make_run(tmp_path, "run-new", 4096, age_seconds=10)

    evicted = outputs.enforce_retention(tmp_path, max_bytes=9000)

    assert list(evicted) == [oldest]
    assert not oldest.exists()
    assert middle.exists() and newest.exists()


def test_eviction_continues_until_under_the_cap(tmp_path):
    oldest = make_run(tmp_path, "run-old", 4096, age_seconds=9000)
    middle = make_run(tmp_path, "run-mid", 4096, age_seconds=600)
    newest = make_run(tmp_path, "run-new", 4096, age_seconds=10)

    evicted = outputs.enforce_retention(tmp_path, max_bytes=5000)

    assert list(evicted) == [oldest, middle], "oldest-first, and it must keep going"
    assert newest.exists()


def test_the_whole_run_directory_goes_not_just_its_biggest_file(tmp_path):
    run = make_run(tmp_path, "run-old", 4096, age_seconds=9000)
    (run / "audio.wav").write_bytes(b"\x00" * 512)
    (run / "request.json").write_text("{}")
    make_run(tmp_path, "run-new", 1024, age_seconds=10)

    outputs.enforce_retention(tmp_path, max_bytes=2000)

    assert not run.exists(), "half an evicted run is worse than none of it"


def test_the_newest_run_is_never_evicted(tmp_path):
    """Even a single run larger than the whole cap survives. The alternative is
    generating a video and deleting it in the same breath."""
    only = make_run(tmp_path, "run-new", 8192, age_seconds=5)
    evicted = outputs.enforce_retention(tmp_path, max_bytes=1024)
    assert only.exists()
    assert only not in evicted


def test_the_evicted_paths_are_returned_so_they_can_be_logged(tmp_path):
    make_run(tmp_path, "run-old", 4096, age_seconds=9000)
    make_run(tmp_path, "run-new", 4096, age_seconds=10)
    evicted = outputs.enforce_retention(tmp_path, max_bytes=5000)
    assert evicted, "PLAN §2.7 requires logging what was evicted; return it so it can be"
    assert all(hasattr(path, "name") for path in evicted)


def test_sweeping_an_empty_directory_is_a_no_op(tmp_path):
    assert outputs.enforce_retention(tmp_path, max_bytes=1024) == ()


def test_sweeping_a_missing_directory_does_not_raise(tmp_path):
    """Startup sweeps run before the first generation has created anything."""
    assert outputs.enforce_retention(tmp_path / "not-yet", max_bytes=1024) == ()


def test_a_stray_file_in_the_root_does_not_crash_the_sweep(tmp_path):
    (tmp_path / "stray.txt").write_text("left over")
    make_run(tmp_path, "run-new", 1024, age_seconds=10)
    outputs.enforce_retention(tmp_path, max_bytes=1_000_000)


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_a_nonsense_cap_is_rejected_rather_than_deleting_everything(max_bytes, tmp_path):
    from ltx25_studio.errors import LTXStudioError

    make_run(tmp_path, "run-new", 1024, age_seconds=10)
    with pytest.raises(LTXStudioError):
        outputs.enforce_retention(tmp_path, max_bytes=max_bytes)
