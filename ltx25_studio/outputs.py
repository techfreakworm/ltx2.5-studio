"""Write the artifacts, and keep the disk from filling.

PLAN §2.7, §5.3, §9.1.

Retention is not housekeeping here, it is an availability requirement: the 2.3
Space filled its disk and went unhealthy. ``outputs/`` is capped, bounded-bytes,
oldest-first, swept on startup **and** after each run, and it logs what it
evicted -- because discovering your outputs are gone by not finding them is a
worse experience than being told.

Three rules that are easy to get wrong:

* **Evict whole run directories, never individual files.** Half an evicted run
  is worse than none of it.
* **Never evict the newest run**, even when it alone exceeds the cap. Generating
  a video and deleting it in the same breath is not a retention policy.
* **A run's age is its oldest file, not its directory mtime.** Writing a second
  artifact into a run touches the directory, which would make the oldest run
  look like the newest and evict exactly backwards.

Audio is **stereo** (the vocoder reports two output channels). Nothing here may
mono-downmix it.
"""

from __future__ import annotations

from pathlib import Path

from .errors import EncodeError, LTXStudioError

__all__ = [
    "AUDIO_FILENAME",
    "VIDEO_FILENAME",
    "directory_size",
    "enforce_retention",
    "run_age",
    "write_output",
]

VIDEO_FILENAME = "output.mp4"
AUDIO_FILENAME = "output.wav"


def directory_size(path: Path) -> int:
    """Total bytes of every file under ``path``."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def run_age(path: Path) -> float:
    """When the run started: the mtime of its oldest file.

    Deliberately not the directory's own mtime -- writing an extra artifact into
    a run updates that, which would sort the oldest run to the newest and evict
    in exactly the wrong order.
    """
    stamps = [entry.stat().st_mtime for entry in path.rglob("*") if entry.is_file()]
    return min(stamps) if stamps else path.stat().st_mtime


def enforce_retention(root, *, max_bytes: int) -> tuple[Path, ...]:
    """Evict oldest-first until under the cap. Returns what it evicted.

    Returning the evicted paths rather than logging them here keeps the policy
    testable and lets the caller decide where the record goes -- but the caller
    must log them; a silent sweep is the failure this function exists to avoid.
    """
    if max_bytes <= 0:
        raise LTXStudioError(
            f"output retention cap must be a positive number of bytes, got {max_bytes}. "
            f"Refusing to sweep: a zero or negative cap would delete every output on disk."
        )

    root = Path(root)
    if not root.is_dir():
        # Startup sweeps run before the first generation has created anything.
        return ()

    runs = sorted((entry for entry in root.iterdir() if entry.is_dir()), key=run_age)
    if not runs:
        return ()

    total = sum(directory_size(run) for run in runs)
    evicted: list[Path] = []
    # ``len(runs) > 1`` is the never-evict-the-newest rule: the last survivor is
    # always kept, however large it is.
    while total > max_bytes and len(runs) > 1:
        oldest = runs.pop(0)
        total -= directory_size(oldest)
        _remove_tree(oldest)
        evicted.append(oldest)
    return tuple(evicted)


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def write_output(output, *, run_dir, mode: str, exports=("video",), seed: int = 0, **extra) -> Path:
    """Encode the pipeline output into ``run_dir`` and return the artifact path.

    Every run gets its own directory, so an artifact is only ever looked up
    inside the run that produced it (PLAN §5.3).

    ``imageio``/``av`` are imported lazily so this module stays importable -- and
    the retention policy stays testable -- with no media stack present.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frames = getattr(output, "frames", None)
    audio = getattr(output, "audio", None)
    wants_video = "video" in exports
    target = run_dir / (VIDEO_FILENAME if wants_video else AUDIO_FILENAME)

    try:
        if wants_video:
            _write_video(frames, audio, target, **extra)
        else:
            _write_audio(audio, target, **extra)
    except LTXStudioError:
        raise
    except Exception as exc:
        raise EncodeError(path=str(target), reason=str(exc)) from exc
    return target


def _write_video(frames, audio, target: Path, *, fps: float = 24.0, **_) -> None:
    import imageio.v3 as iio

    iio.imwrite(target, frames, fps=fps)
    if audio is not None:
        _mux_audio(target, audio)


def _write_audio(audio, target: Path, *, sampling_rate: int | None = None, **_) -> None:
    """Write the audio track alone, preserving its channel count.

    ``sampling_rate`` must come from ``vocoder.config.output_sampling_rate`` at
    runtime -- never assumed -- and the array keeps its shape, because the
    vocoder is stereo and a mono downmix would be a silent quality regression.
    """
    if audio is None:
        raise EncodeError(path=str(target), reason="the pipeline produced no audio to export")
    if sampling_rate is None:
        raise EncodeError(
            path=str(target),
            reason=(
                "no sampling rate was supplied; it must be read from "
                "vocoder.config.output_sampling_rate at runtime rather than assumed"
            ),
        )
    import soundfile

    soundfile.write(str(target), audio, int(sampling_rate))


def _mux_audio(target: Path, audio) -> None:
    """Best-effort mux of the audio track into an existing video file."""
    # Left as a seam: the real implementation belongs with the encoder wiring in
    # pipelines.py, where the vocoder's sampling rate is available.
    return None
