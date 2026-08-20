"""Spec for the post-run sweep — PLAN §2.7.

Removing the sweep from ``generate()`` entirely left the suite fully green.

``outputs.enforce_retention`` is well specified on its own; what was untested is
that **anything calls it**. That distinction is the whole file: retention is not
housekeeping, it is an availability requirement. The predecessor Space filled its
disk and went unhealthy, and the failure has no symptom until it has a total one
— generations keep succeeding, right up until the moment the Space stops coming
back at all.

A never-sweeping ``generate()`` is invisible to every test that checks a single
run, because a single run never fills anything. It is only visible across runs,
which is why the specs here create prior runs and then generate.

Three rules from PLAN §2.7 that are each easy to get backwards, and none of which
a caller-side test can assume the policy still honours:

* evict **whole run directories**, never individual files — half an evicted run
  is worse than none of it;
* **never evict the newest run**, even when it alone exceeds the cap;
* a run's age is its **oldest file**, not the directory mtime — writing a second
  artifact into a run touches the directory, which would sort the oldest run to
  the newest and evict in exactly the wrong order.

And one rule about the sweep as seen from ``generate()``: it must be **non-fatal
but never silent**. The user's video exists and is fine, so a retention failure
must not turn a good run into an error — but "discovering your outputs are gone
by not finding them is a worse experience than being told".
"""

from __future__ import annotations

import os

import pytest
from test_exports_contract import (
    MODE_IDS,
    FakePipe,
    factory_returning,
    one_muxed_file_encoder,
)

from ltx25_studio import generate as generate_module
from ltx25_studio import outputs

#: Comfortably larger than the artifacts the fake encoder writes (2 KiB), so the
#: cap is crossed by the *old* runs rather than by rounding.
ARTIFACT_BYTES = 2048
OLD_RUN_BYTES = 64 * 1024


def make_old_run(root, name, *, age_seconds, size=OLD_RUN_BYTES):
    """A previous run on disk, stamped into the past.

    The mtime is set on the **file**, because ``run_age`` reads the oldest file
    rather than the directory — a fixture that only aged the directory would
    quietly stop exercising the ordering rule.
    """
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / outputs.VIDEO_FILENAME
    artifact.write_bytes(b"\x00" * size)
    stamp = os.stat(artifact).st_mtime - age_seconds
    os.utime(artifact, (stamp, stamp))
    os.utime(run_dir, (stamp, stamp))
    return run_dir


def run(tmp_path, *, mode_id="text_to_video", retention_bytes):
    return generate_module.generate(
        mode=mode_id,
        seed=7,
        pipeline_factory=factory_returning(FakePipe()),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=retention_bytes,
    )


# --------------------------------------------------------------------------
# the sweep happens at all
# --------------------------------------------------------------------------


def test_a_successful_run_evicts_the_runs_that_no_longer_fit(tmp_path):
    """The mutation that stayed green: ``generate()`` never sweeping.

    Invisible to any single-run test, because one run never fills a disk. It only
    shows up across runs — and by the time it shows up in production the Space is
    already unhealthy.
    """
    oldest = make_old_run(tmp_path, "run-oldest", age_seconds=3000)
    older = make_old_run(tmp_path, "run-older", age_seconds=2000)

    # Below both old runs, so getting under it requires evicting both.
    result = run(tmp_path, retention_bytes=ARTIFACT_BYTES * 2)

    assert not oldest.exists(), (
        "generate() completed with the output directory over its cap and evicted nothing. "
        "PLAN §2.7 requires a sweep after every run: the predecessor filled its disk and the "
        "Space went unhealthy."
    )
    assert not older.exists(), "the sweep stopped before getting under the cap"
    assert outputs.directory_size(tmp_path) <= ARTIFACT_BYTES * 2, (
        "the sweep ran but left the output directory over its cap"
    )
    assert result.run_dir.exists(), "the run that just finished must survive its own sweep"


def test_the_sweep_stops_as_soon_as_it_is_under_the_cap(tmp_path):
    """Evicting past the point of need would throw away outputs for nothing.

    "Oldest-first until under the cap" is a stopping rule, not a purge.
    """
    oldest = make_old_run(tmp_path, "run-oldest", age_seconds=3000)
    older = make_old_run(tmp_path, "run-older", age_seconds=2000)

    # Room for exactly one old run plus the new artifact.
    run(tmp_path, retention_bytes=OLD_RUN_BYTES + ARTIFACT_BYTES)

    assert not oldest.exists(), "the oldest run should have gone first"
    assert older.exists(), (
        "the sweep kept evicting after the directory was already under its cap; retention is "
        "a bound, not a purge."
    )


def test_the_sweep_evicts_oldest_first(tmp_path):
    """Newest-first eviction would delete the run the user is most likely to want."""
    oldest = make_old_run(tmp_path, "run-oldest", age_seconds=3000)
    newer = make_old_run(tmp_path, "run-newer", age_seconds=10, size=ARTIFACT_BYTES)

    run(tmp_path, retention_bytes=OLD_RUN_BYTES)

    assert not oldest.exists(), "the oldest run survived while something newer was evicted"
    assert newer.exists(), f"{newer.name} was evicted before the older run"


def test_the_sweep_never_evicts_the_run_that_just_finished(tmp_path):
    """ "Generating a video and deleting it in the same breath is not a policy."

    The cap here is far below the artifact the run just wrote, so a naive sweep
    would delete the very file it is about to hand back.
    """
    make_old_run(tmp_path, "run-old", age_seconds=3000)

    result = run(tmp_path, retention_bytes=1)

    assert result.run_dir.exists(), (
        "the sweep evicted the run that just completed. Its artifact is what generate() is "
        "returning; deleting it makes the result a path to nothing."
    )
    assert result.output_path.is_file(), "the returned artifact was swept away"


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_every_mode_sweeps_after_a_successful_run(mode_id, tmp_path):
    """Retention is not per-mode. A mode that skips it fills the disk just as well."""
    old = make_old_run(tmp_path, "run-old", age_seconds=3000)

    run(tmp_path, mode_id=mode_id, retention_bytes=ARTIFACT_BYTES * 2)

    assert not old.exists(), f"mode {mode_id!r} completed without sweeping the output root"


# --------------------------------------------------------------------------
# the sweep is reported, not silent
# --------------------------------------------------------------------------


def test_an_eviction_is_reported_on_the_result(tmp_path):
    """ "Discovering your outputs are gone by not finding them is worse than being told."

    ``enforce_retention`` returns what it evicted precisely so the caller can say
    so. A caller that drops the list re-creates the silent sweep the policy is
    written against.
    """
    make_old_run(tmp_path, "run-old", age_seconds=3000)

    result = run(tmp_path, retention_bytes=ARTIFACT_BYTES * 2)

    assert result.warnings, (
        "a run was evicted and the result carries no warning about it. The eviction happened "
        "silently, which is the half of PLAN §2.7 that is about the user rather than the disk."
    )
    assert any("run-old" in warning for warning in result.warnings), (
        f"the warnings do not name what was evicted: {result.warnings}"
    )


def test_a_run_that_evicts_nothing_says_nothing(tmp_path):
    """The inverse: no spurious warning on an ordinary run, or the signal is noise."""
    make_old_run(tmp_path, "run-old", age_seconds=3000, size=ARTIFACT_BYTES)

    result = run(tmp_path, retention_bytes=10**9)

    assert result.warnings == (), (
        f"nothing was evicted but the run warned anyway: {result.warnings}"
    )


# --------------------------------------------------------------------------
# a retention failure must not fail the generation
# --------------------------------------------------------------------------


def test_a_broken_sweep_does_not_turn_a_good_run_into_an_error(tmp_path, monkeypatch):
    """PLAN §2.7: "the user's video exists and is fine".

    Retention is an availability concern; the artifact is the user's. Raising
    here would throw away a completed, already-paid-for generation because of a
    housekeeping problem.
    """

    def exploding(*args, **kwargs):
        raise OSError("the filesystem is read-only")

    monkeypatch.setattr(outputs, "enforce_retention", exploding)

    result = run(tmp_path, retention_bytes=ARTIFACT_BYTES)

    assert result.output_path.is_file(), "a retention failure destroyed a good generation"


def test_a_broken_sweep_is_surfaced_rather_than_swallowed(tmp_path, monkeypatch):
    """Non-fatal is not the same as invisible. A disk quietly filling is the outage."""

    def exploding(*args, **kwargs):
        raise OSError("the filesystem is read-only")

    monkeypatch.setattr(outputs, "enforce_retention", exploding)

    result = run(tmp_path, retention_bytes=ARTIFACT_BYTES)

    assert result.warnings, (
        "the sweep failed and the run reported nothing. Retention silently not happening is "
        "how the predecessor's disk filled."
    )
    assert any("retention" in warning.lower() for warning in result.warnings), (
        f"the warning does not say what failed: {result.warnings}"
    )


# --------------------------------------------------------------------------
# the policy the caller depends on
# --------------------------------------------------------------------------


def test_a_runs_age_is_its_oldest_file_not_its_directory(tmp_path):
    """Writing a second artifact into a run touches the directory.

    If age came from the directory mtime, the oldest run would sort newest and
    eviction would run exactly backwards — deleting the newest runs first while
    the ancient ones accumulate forever.
    """
    old = make_old_run(tmp_path, "run-old", age_seconds=3000)
    # A second artifact written later, as a joint audio+video run would do.
    (old / outputs.AUDIO_FILENAME).write_bytes(b"\x00" * 16)

    recent = make_old_run(tmp_path, "run-recent", age_seconds=10)

    assert outputs.run_age(old) < outputs.run_age(recent), (
        "a run whose directory was touched by a later write is being treated as newer than "
        "it is; eviction would run backwards."
    )


def test_whole_run_directories_are_evicted_never_loose_files(tmp_path):
    """ "Half an evicted run is worse than none of it."""
    old = make_old_run(tmp_path, "run-old", age_seconds=3000)
    (old / outputs.AUDIO_FILENAME).write_bytes(b"\x00" * OLD_RUN_BYTES)

    run(tmp_path, retention_bytes=ARTIFACT_BYTES * 2)

    assert not old.exists(), (
        f"{old.name} still exists after the sweep; if any of its files were removed it is now "
        f"a half-run, which is worse than an evicted one."
    )
