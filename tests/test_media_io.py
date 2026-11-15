"""Media in and media out, measured on real files written here by ``av``.

* A conditioning video is read at the generation's frame rate, by timestamp,
  and the right way up (F16).

Nothing is committed as a fixture: every clip and every track is generated in
the test, so what is measured is the file the code under test produced.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from ltx25_studio import conditioning

av = pytest.importorskip("av")


# ---------------------------------------------------------------------------
# real clips
# ---------------------------------------------------------------------------


def write_clip(path, images, *, rate, pts=None, rotation=None, rotate_tag=None):
    """Encode ``images`` (H, W, 3 uint8) with timestamps in units of ``1/rate``."""
    pts = list(range(len(images))) if pts is None else list(pts)
    height, width = images[0].shape[:2]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"crf": "12"}
        if rotation is not None:
            stream.set_display_rotation(rotation)
        if rotate_tag is not None:
            stream.metadata["rotate"] = str(rotate_tag)
        for image, stamp in zip(images, pts, strict=True):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            frame.pts, frame.time_base = stamp, Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return Path(path)


#: Marker geometry: an 8-px-wide white bar at x = MARKER_STEP * k on frame k.
MARKER_STEP = 4
MARKER_WIDTH = 8


def marker_frame(index, *, width=256, height=32):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left = MARKER_STEP * index
    canvas[:, left : left + MARKER_WIDTH] = 255
    return canvas


def marker_index(frame) -> int:
    """Which source frame this is, read back from where its bar sits."""
    profile = frame.astype(np.float32).mean(axis=(0, 2))
    columns = np.flatnonzero(profile > 128)
    assert columns.size, "no marker found in a decoded frame"
    centre = columns.mean() - (MARKER_WIDTH - 1) / 2
    return int(round(centre / MARKER_STEP))


def portrait():
    """An upright 32x64 picture: bright block top-left, mid-grey block bottom-right."""
    canvas = np.zeros((64, 32, 3), dtype=np.uint8)
    canvas[:16, :16] = 255
    canvas[-16:, -16:] = 128
    return canvas


def assert_upright(frames):
    assert frames.shape[1:3] == (64, 32), (
        f"the clip came back {frames.shape[2]}x{frames.shape[1]} (WxH); upright it is 32x64 -- "
        f"the rotation in its metadata was not applied"
    )
    first = frames[0].astype(np.float32)
    top_left = first[:16, :16].mean()
    others = [first[:16, -16:].mean(), first[-16:, :16].mean()]
    assert top_left > 200 and max(others) < 60, (
        f"the bright corner is not top-left (top-left {top_left:.0f}, top-right "
        f"{others[0]:.0f}, bottom-left {others[1]:.0f}): the picture is turned or mirrored"
    )


# ---------------------------------------------------------------------------
# F16 -- timing
# ---------------------------------------------------------------------------


def test_a_60fps_clip_conditions_a_24fps_run_in_real_time(tmp_path):
    """One second of a 60 fps phone clip is one second of picture at 24 fps.

    Read frame by frame, frame ``i`` of the condition was source frame ``i``:
    the clip played at 24/60 = 0.4x speed. Selected by timestamp, frame ``i``
    shows the source frame nearest to ``i / 24`` seconds.
    """
    source = write_clip(tmp_path / "phone.mp4", [marker_frame(k) for k in range(60)], rate=60)

    frames = conditioning.decode_video_frames(source, frame_rate=24)

    assert frames.shape[0] == 24, (
        f"one second of video at 24 fps is 24 frames, got {frames.shape[0]}"
    )
    for i, frame in enumerate(frames):
        shown = Fraction(marker_index(frame), 60)
        assert abs(shown - Fraction(i, 24)) <= Fraction(1, 120), (
            f"condition frame {i} (t={float(Fraction(i, 24)):.4f}s) shows the source frame at "
            f"t={float(shown):.4f}s -- more than half a source frame away from real time"
        )


def test_max_frames_still_bounds_a_rate_matched_read(tmp_path):
    source = write_clip(tmp_path / "phone.mp4", [marker_frame(k) for k in range(60)], rate=60)

    capped = conditioning.decode_video_frames(source, frame_rate=24, max_frames=10)
    whole = conditioning.decode_video_frames(source, frame_rate=24)

    assert capped.shape[0] == 10
    assert np.array_equal(capped, whole[:10])


def test_a_slower_clip_is_held_rather_than_sped_up(tmp_path):
    """12 fps into 24 fps: every source frame is shown twice, none skipped."""
    source = write_clip(tmp_path / "slow.mp4", [marker_frame(k) for k in range(12)], rate=12)

    frames = conditioning.decode_video_frames(source, frame_rate=24)

    assert frames.shape[0] == 24
    assert [marker_index(f) for f in frames] == [i // 2 for i in range(24)]


def test_variable_frame_rate_input_is_selected_by_timestamp(tmp_path):
    """Gaps in the source are gaps in time, not frames to be packed together."""
    stamps_ms = [0, 10, 20, 200, 210, 500, 520, 900, 1000]
    images = [marker_frame(k, width=64) for k in range(len(stamps_ms))]
    source = write_clip(tmp_path / "vfr.mp4", images, rate=1000, pts=stamps_ms)

    frames = conditioning.decode_video_frames(source, frame_rate=24)

    times = [Fraction(ms, 1000) for ms in stamps_ms]

    def nearest(t):
        return min(range(len(times)), key=lambda k: (abs(times[k] - t), times[k]))

    wanted = [nearest(Fraction(i, 24)) for i in range(24)]
    got = [marker_index(f) for f in frames[:24]]
    assert frames.shape[0] >= 24, f"a 1 s clip gave only {frames.shape[0]} frames at 24 fps"
    assert got == wanted, f"selected source frames {got}, nearest by timestamp are {wanted}"


def test_a_short_clip_at_the_same_rate_reads_exactly_as_before(tmp_path):
    """A clip shorter than the request yields fewer frames, unchanged."""
    source = write_clip(tmp_path / "short.mp4", [marker_frame(k) for k in range(9)], rate=24)

    matched = conditioning.decode_video_frames(source, frame_rate=24, max_frames=121)
    frame_by_frame = conditioning.decode_video_frames(source, max_frames=121)

    assert matched.shape[0] == 9
    assert np.array_equal(matched, frame_by_frame)


def test_without_a_rate_the_read_is_frame_by_frame(tmp_path):
    """The old behaviour, kept for a caller that does not pass the rate yet."""
    source = write_clip(tmp_path / "phone.mp4", [marker_frame(k) for k in range(60)], rate=60)

    frames = conditioning.decode_video_frames(source)

    assert [marker_index(f) for f in frames] == list(range(60))


def test_build_conditions_passes_the_rate_to_the_decode(tmp_path):
    """The seam ``generate`` calls: a rate given here reaches the decode."""
    source = write_clip(tmp_path / "phone.mp4", [marker_frame(k) for k in range(60)], rate=60)
    spec = conditioning.ConditionSpec(source=source, kind="video", index=0, strength=1.0)

    built = conditioning.build_conditions([spec], num_frames=121, frame_rate=24)[0]

    assert built.source.shape[0] == 24
    assert marker_index(built.source[12]) == 30, "frame 12 at 24 fps is t=0.5 s, source frame 30"


def test_a_nonsense_rate_is_refused(tmp_path):
    from ltx25_studio.errors import ConstraintError

    source = write_clip(tmp_path / "in.mp4", [marker_frame(k) for k in range(4)], rate=24)
    with pytest.raises(ConstraintError):
        conditioning.decode_video_frames(source, frame_rate=0)


# ---------------------------------------------------------------------------
# F16 -- orientation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", [90, 180, 270, -90])
def test_a_clip_with_a_display_rotation_comes_in_upright(rotation, tmp_path):
    """A phone stores a portrait clip sideways and says how to turn it.

    The stored picture is the upright one turned the other way; the display
    matrix (counter-clockwise degrees) turns it back.
    """
    stored = np.rot90(portrait(), k=-(rotation // 90))
    source = write_clip(tmp_path / "phone.mp4", [stored] * 3, rate=24, rotation=rotation)

    assert_upright(conditioning.decode_video_frames(source))
    assert_upright(conditioning.decode_video_frames(source, frame_rate=24))


def test_an_older_file_with_only_a_rotate_tag_comes_in_upright(tmp_path):
    """The fallback: a ``rotate`` stream tag, which counts degrees clockwise."""
    stored = np.rot90(portrait(), k=1)
    source = write_clip(tmp_path / "old.mkv", [stored] * 3, rate=24, rotate_tag=90)

    assert_upright(conditioning.decode_video_frames(source))


def test_an_unrotated_clip_is_left_alone(tmp_path):
    source = write_clip(tmp_path / "plain.mp4", [portrait()] * 3, rate=24)

    assert_upright(conditioning.decode_video_frames(source))
