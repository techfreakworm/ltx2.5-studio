"""Spec for `tests/verify_streams.py` — the artifact-vs-registry checker.

Part A's pass criterion is the expected stream layout per mode. Judged by eye
across sixteen runs that is sixteen chances for a wrong-but-plausible pass, in
the layer that has produced every confidently-wrong artefact on this build.

These tests build **real containers** with `av` rather than faking the probe,
because the thing being checked is whether we can read a real file correctly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from verify_streams import EXPECTED_AUDIO_CHANNELS, expected_streams, main, verify

from ltx25_studio import modes

av = pytest.importorskip("av")


def build(path, *, video=True, audio=True, channels=EXPECTED_AUDIO_CHANNELS, rate=24):
    """A real mp4 with exactly the streams asked for."""
    with av.open(str(path), "w") as container:
        video_stream = audio_stream = None
        if video:
            video_stream = container.add_stream("h264", rate=rate)
            video_stream.width, video_stream.height = 64, 64
            video_stream.pix_fmt = "yuv420p"
        if audio:
            layout = "stereo" if channels == 2 else "mono"
            audio_stream = container.add_stream("aac", rate=24000, layout=layout)
            audio_stream.time_base = Fraction(1, 24000)
        if video_stream is not None:
            for index in range(4):
                frame = av.VideoFrame.from_ndarray(
                    np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24"
                )
                frame.pts, frame.time_base = index, Fraction(1, rate)
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)
        if audio_stream is not None:
            layout = "stereo" if channels == 2 else "mono"
            for index in range(2):
                frame = av.AudioFrame.from_ndarray(
                    np.zeros((channels, 1024), dtype="float32"), format="fltp", layout=layout
                )
                frame.rate, frame.pts = 24000, index * 1024
                frame.time_base = Fraction(1, 24000)
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
    return path


# --------------------------------------------------------------------------
# expectations come from the registry, never from a copy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", sorted(modes.MODES))
def test_expectations_track_the_registry(mode_id):
    """A snapshot would be a second place the truth lives, and it would drift
    invisibly — both sides would agree with themselves."""
    spec = modes.get_mode(mode_id)
    assert expected_streams(mode_id)["video"] is ("video" in spec.exports)
    assert expected_streams(mode_id)["audio"] is ("audio" in spec.exports)


# --------------------------------------------------------------------------
# both directions — a missing declared stream AND an undeclared one present
# --------------------------------------------------------------------------


def test_a_video_only_mode_with_a_video_only_file_matches(tmp_path):
    verdict = verify("text_to_video", build(tmp_path / "a.mp4", audio=False), frame_rate=24.0)
    assert verdict.ok and verdict.exit_code == 0


def test_a_joint_mode_with_both_streams_matches(tmp_path):
    verdict = verify("text_to_audio_video", build(tmp_path / "b.mp4"), frame_rate=24.0)
    assert verdict.ok and verdict.exit_code == 0


def test_an_undeclared_audio_stream_is_caught(tmp_path):
    """The half the exports contract has historically been weak on.

    A "video only" mode shipping sound means `exports` has stopped deciding the
    container's contents — and every other check passes, because the file is
    perfectly valid.
    """
    verdict = verify("text_to_video", build(tmp_path / "c.mp4"), frame_rate=24.0)
    assert not verdict.ok and verdict.exit_code == 1
    assert "does NOT declare" in " ".join(verdict.problems)


def test_a_missing_declared_stream_is_caught(tmp_path):
    verdict = verify("text_to_audio_video", build(tmp_path / "d.mp4", audio=False), frame_rate=24.0)
    assert not verdict.ok and verdict.exit_code == 1
    assert "NO audio stream" in " ".join(verdict.problems)


def test_a_mono_downmix_is_caught(tmp_path):
    """The vocoder is stereo; a mono track is a silent quality regression."""
    verdict = verify("text_to_audio_video", build(tmp_path / "e.mp4", channels=1))
    assert not verdict.ok
    assert "channel" in " ".join(verdict.problems)


def test_a_wrong_frame_rate_is_caught(tmp_path):
    """A container written at the wrong rate plays at the wrong speed, silently."""
    verdict = verify(
        "text_to_video", build(tmp_path / "f.mp4", audio=False, rate=30), frame_rate=24.0
    )
    assert not verdict.ok
    assert "frame rate" in " ".join(verdict.problems)


# --------------------------------------------------------------------------
# "could not probe" is never a pass
# --------------------------------------------------------------------------


def test_an_unreadable_file_exits_two_not_zero(tmp_path):
    """Same shape as `startup_report` once defaulting to "nothing is missing"
    and reporting 37/37 having checked nothing — one layer out."""
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a container")
    verdict = verify("text_to_video", junk)

    assert verdict.probed is False
    assert verdict.ok is False
    assert verdict.exit_code == 2, "could-not-probe must be distinct from both pass and fail"
    assert "NOT a pass" in " ".join(verdict.problems)


def test_a_missing_file_exits_two(tmp_path):
    assert verify("text_to_video", tmp_path / "nope.mp4").exit_code == 2


# --------------------------------------------------------------------------
# the command line: a usage error is exit 2, never a traceback or a "mismatch"
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(["--frame-rate"], id="trailing-flag-without-a-value"),
        pytest.param(["--frame-rate", "fast"], id="frame-rate-not-a-number"),
        pytest.param(["--sampling-rate", "48k"], id="sampling-rate-not-an-integer"),
        pytest.param(["--frmae-rate", "24"], id="unknown-flag"),
        pytest.param(["--geometry", "64by64"], id="geometry-not-WxH"),
        pytest.param(["--geometry", "0x64"], id="geometry-not-positive"),
    ],
)
def test_a_malformed_command_line_exits_two(flags, tmp_path, capsys):
    """Exit 1 means "the file does not match"; a typo must not read as that."""
    path = build(tmp_path / "a.mp4", audio=False)

    assert main(["text_to_video", str(path), *flags]) == 2


def test_a_missing_file_argument_exits_two(capsys):
    assert main(["text_to_video"]) == 2


def test_the_script_reports_a_usage_error_without_a_traceback(tmp_path):
    """The real entry point, as the host scripts run it."""
    path = build(tmp_path / "a.mp4", audio=False)
    script = Path(__file__).resolve().parent / "verify_streams.py"

    proc = subprocess.run(
        [sys.executable, str(script), "text_to_video", str(path), "--frame-rate"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr


def test_flags_are_still_read(tmp_path):
    path = build(tmp_path / "a.mp4", audio=False, rate=30)
    assert main(["text_to_video", str(path), "--frame-rate", "30"]) == 0
    assert main(["text_to_video", str(path), "--frame-rate", "24"]) == 1


# --------------------------------------------------------------------------
# --geometry: the size the file really has
# --------------------------------------------------------------------------


def test_the_requested_geometry_matches(tmp_path):
    verdict = verify("text_to_video", build(tmp_path / "a.mp4", audio=False), geometry=(64, 64))
    assert verdict.ok, verdict.problems


def test_a_different_geometry_is_a_mismatch(tmp_path):
    """Stage 2 doubles both sides; a 64x64 file is not the 128x128 asked for."""
    verdict = verify("text_to_video", build(tmp_path / "a.mp4", audio=False), geometry=(128, 128))
    assert not verdict.ok and verdict.exit_code == 1
    assert "128x128" in " ".join(verdict.problems) and "64x64" in " ".join(verdict.problems)


def test_geometry_from_the_command_line(tmp_path):
    path = build(tmp_path / "a.mp4", audio=False)
    assert main(["text_to_video", str(path), "--geometry", "64x64"]) == 0
    assert main(["text_to_video", str(path), "--geometry", "128x128"]) == 1


def test_a_geometry_check_on_a_file_with_no_picture_is_a_mismatch(tmp_path):
    verdict = verify("text_to_audio", build(tmp_path / "a.m4a", video=False), geometry=(64, 64))
    assert not verdict.ok
    assert "no video stream" in " ".join(verdict.problems)
