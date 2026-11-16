"""Spec for the one input kind that never got a decoder.

Three input kinds reach conditioning, by three different routes, and exactly one
of them arrived at the pipeline undecoded:

* ``gr.Image(type="pil")`` hands over a **decoded PIL image** — upstream accepts
  it directly (``components.py:70``).
* ``gr.Audio(type="filepath")`` hands over a **path**, and ``audio.py`` decodes
  it itself with ``av.open`` before it goes anywhere near the model.
* ``gr.Video(...)`` hands over a **path**, and nothing decoded it. It travelled
  through ``modes._conditions`` into ``ConditionSpec(kind="video")`` and reached
  diffusers as a bare ``str``.

The Space failure that produced this file, from the worker traceback:

    File "diffusers/pipelines/ltx2/pipeline_ltx2_condition.py", line 899,
      in preprocess_conditions
        raise TypeError(f"Unsupported `frames` type for condition {i}: ...")
    TypeError: Unsupported `frames` type for condition 0: <class 'str'>

Both video-input modes failed (`video_to_video`, `video_to_audio`); the other
nine passed. That asymmetry is the whole diagnosis, and these tests are written
against it rather than against the symptom: the load-bearing assertion is that
what leaves ``build_conditions`` for a video condition is a payload
``preprocess_conditions`` **dispatches on**, not merely "not a string".

**Real containers, built with ``av``.** Same discipline as
``tests/test_verify_streams.py``: faking the decode would test the mock. The
files here are small but genuine mp4s, and the failure cases are genuine too — a
real wav with no video stream, real garbage bytes.

**Both directions, every time.** A decoder that only ever succeeds and a
decoder that rejects everything are indistinguishable from each other in a suite
that only tests one side, so every rejection here has an acceptance beside it.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from test_exports_contract import factory_returning, one_muxed_file_encoder

from ltx25_studio import conditioning, modes
from ltx25_studio import generate as generate_module
from ltx25_studio.errors import ConstraintError, LTXStudioError

av = pytest.importorskip("av")

FIXTURE_SIZE = 64
FIXTURE_RATE = 24


# ---------------------------------------------------------------------------
# real media, built here
# ---------------------------------------------------------------------------


def _frame(index: int) -> np.ndarray:
    """A bright block whose position depends on the index.

    A block rather than noise: h264 discards noise and keeps structure, so a
    noise fixture can decode back flatter than it went in and fail an assertion
    for reasons that have nothing to do with the code under test.
    """
    canvas = np.zeros((FIXTURE_SIZE, FIXTURE_SIZE, 3), dtype=np.uint8)
    offset = (index * 8) % (FIXTURE_SIZE - 20)
    canvas[10:30, offset : offset + 20] = 255
    return canvas


def write_video(path, *, frames=9):
    """A real mp4 carrying exactly ``frames`` frames of moving content."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=FIXTURE_RATE)
        stream.width = stream.height = FIXTURE_SIZE
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(_frame(index), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, FIXTURE_RATE)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return Path(path)


def write_wav(path, *, rate=48000, blocks=4):
    """A real wav — a file that decodes perfectly and carries no video at all."""
    with av.open(str(path), "w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout="stereo")
        for index in range(blocks):
            samples = 1024
            t = (np.arange(samples) + index * samples) / float(rate)
            tone = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype("float32")
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(np.stack([tone, tone])), format="fltp", layout="stereo"
            )
            frame.rate, frame.pts = rate, index * samples
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return Path(path)


# ---------------------------------------------------------------------------
# what upstream actually accepts
#
# Reproduced from `preprocess_conditions` at the PINNED diffusers SHA
# ac56fa2f25c1c15b4cbaa1e0b3255e9bc38c06dc, lines 887-899 -- the four branches
# and the `else: raise TypeError` that our path was landing in. Asserting
# "not a str" would pass for an int; this asserts the payload is something the
# dispatch has a branch for, which is the property that actually matters.
# ---------------------------------------------------------------------------


def upstream_accepts(frames) -> bool:
    """True if ``preprocess_conditions`` has a branch for this payload type."""
    import PIL.Image

    if isinstance(frames, PIL.Image.Image):
        return True
    if isinstance(frames, list) and all(isinstance(f, PIL.Image.Image) for f in frames):
        return True
    if isinstance(frames, np.ndarray):
        return True
    return type(frames).__module__.startswith("torch") and hasattr(frames, "detach")


# ---------------------------------------------------------------------------
# the decoder itself
# ---------------------------------------------------------------------------


def test_a_real_video_decodes_to_frames_upstream_dispatches_on(tmp_path):
    frames = conditioning.decode_video_frames(write_video(tmp_path / "in.mp4", frames=9))

    assert upstream_accepts(frames), (
        f"{type(frames)} lands in the `else` branch at pipeline_ltx2_condition.py:899, "
        f"which is exactly the TypeError this decoder exists to prevent"
    )
    assert frames.shape == (9, FIXTURE_SIZE, FIXTURE_SIZE, 3)


def test_the_frames_are_uint8_because_a_single_frame_condition_requires_it(tmp_path):
    """Upstream re-compresses a 1-frame condition at the model CRF and refuses a
    non-uint8 array while doing it (`ValueError`, same function). A float payload
    would decode fine here and fail there, on a metered run."""
    frames = conditioning.decode_video_frames(write_video(tmp_path / "one.mp4", frames=1))

    assert frames.dtype == np.uint8
    assert frames.shape[0] == 1


def test_the_decode_is_capped_at_the_frames_upstream_can_use(tmp_path):
    """`trim_conditioning_sequence` takes `min(F, target - start)` and keeps the
    FIRST of them, so frames past the target are decoded and thrown away. On a
    long upload that is the difference between a few MB and tens of GB."""
    source = write_video(tmp_path / "long.mp4", frames=24)

    capped = conditioning.decode_video_frames(source, max_frames=9)
    whole = conditioning.decode_video_frames(source)

    assert capped.shape[0] == 9
    assert whole.shape[0] == 24
    assert np.array_equal(capped, whole[:9]), "the cap must keep the first frames, not any nine"


def test_a_video_shorter_than_the_cap_keeps_every_frame(tmp_path):
    frames = conditioning.decode_video_frames(
        write_video(tmp_path / "short.mp4", frames=3), max_frames=121
    )
    assert frames.shape[0] == 3


def test_a_missing_file_raises_one_of_ours_and_names_it(tmp_path):
    missing = tmp_path / "not-here.mp4"
    with pytest.raises(LTXStudioError) as caught:
        conditioning.decode_video_frames(missing)
    assert str(missing) in str(caught.value)


def test_a_file_with_no_video_stream_is_refused(tmp_path):
    """A wav decodes perfectly and has nothing to condition on. Passing it
    through would put an empty payload in front of the model."""
    with pytest.raises(ConstraintError) as caught:
        conditioning.decode_video_frames(write_wav(tmp_path / "audio.wav"))
    assert "video stream" in str(caught.value)


def test_undecodable_bytes_raise_ours_rather_than_avs(tmp_path):
    """The failure must be named in our vocabulary at the layer that knows which
    file the user picked. An `av` error escaping from here reaches the UI as
    `unexpected` and points at the library instead of at the upload."""
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 16)

    with pytest.raises(LTXStudioError) as caught:
        conditioning.decode_video_frames(junk)
    assert str(junk) in str(caught.value)


# ---------------------------------------------------------------------------
# the seam: build_conditions is where a video spec stops being a path
# ---------------------------------------------------------------------------


def build_one(source, *, kind, num_frames=121, index=0, strength=0.8):
    spec = conditioning.ConditionSpec(source=source, kind=kind, index=index, strength=strength)
    return conditioning.build_conditions([spec], num_frames=num_frames)[0]


def test_a_video_condition_leaves_the_builder_decoded(tmp_path):
    built = build_one(write_video(tmp_path / "in.mp4"), kind="video")

    assert not isinstance(built.source, (str, Path))
    assert upstream_accepts(built.source)


def test_the_index_and_strength_survive_the_decode(tmp_path):
    """Only the payload type was wrong. Everything else about the condition is
    plumbing that was already correct, and a decode that quietly resets an
    anchor or a strength would replace one silent defect with another."""
    built = build_one(write_video(tmp_path / "in.mp4"), kind="video", index=0, strength=0.42)

    assert built.kind == "video"
    assert built.index == 0
    assert built.strength == pytest.approx(0.42)


def test_an_image_condition_is_left_exactly_as_it_arrived():
    """`gr.Image(type="pil")` already hands over a decoded image, and the image
    modes demonstrably work on the Space. Touching that path would be changing
    the half that was never broken."""
    import PIL.Image

    image = PIL.Image.new("RGB", (8, 8))
    built = build_one(image, kind="image")

    assert built.source is image


def test_a_video_condition_that_is_already_frames_is_not_decoded_again(tmp_path):
    """Idempotence. A spec can be rebuilt — `pipelines._with_real_conditions`
    re-normalises anything that is not already a spec — and a second decode of
    an already-decoded payload would fail on a path that is not a path."""
    frames = conditioning.decode_video_frames(write_video(tmp_path / "in.mp4"))
    built = build_one(frames, kind="video")

    assert built.source is frames


def test_the_builders_cap_is_the_requested_clip_length(tmp_path):
    built = build_one(write_video(tmp_path / "long.mp4", frames=24), kind="video", num_frames=9)
    assert built.source.shape[0] == 9


# ---------------------------------------------------------------------------
# end to end: the two modes that failed on the Space
#
# This is the regression test proper. Everything above can pass while the wiring
# between `modes._conditions` and the pipeline still hands over a string --
# which is precisely how this defect survived a full offline gate.
# ---------------------------------------------------------------------------


class CapturingPipe:
    """Records the kwargs it was called with, so the condition list is readable."""

    def __init__(self):
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return type("Output", (), {"frames": ("frame",), "audio": ("sample",)})()


VIDEO_INPUT_MODES = ("video_to_video", "video_to_audio")


@pytest.mark.parametrize("mode_id", VIDEO_INPUT_MODES)
def test_no_video_path_string_reaches_the_pipeline(mode_id, tmp_path):
    """The exact defect, in the form the Space reported it.

    A `str` here is what raised `Unsupported \\`frames\\` type for condition 0:
    <class 'str'>` inside the forked worker -- where, per the fork-boundary
    finding, the `reason=` we attach is stripped and only the class name
    survives. So this must fail here, offline, where the whole error is legible.
    """
    pipe = CapturingPipe()
    generate_module.generate(
        mode=mode_id,
        seed=7,
        video=str(write_video(tmp_path / "input.mp4")),
        num_frames=9,
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    conditions = pipe.kwargs.get("conditions", ())
    if modes.get_mode(mode_id).video_conditioning == "variation":
        # Mode 04 varies the clip: it travels as decoded frames, and is no condition.
        assert not conditions, f"{mode_id} also conditioned on its clip: {conditions!r}"
        clip = pipe.kwargs["video_variation"]
        assert not isinstance(clip.source, (str, Path)), (
            f"{mode_id} handed the pipeline {clip.source!r} -- a path, not frames."
        )
        assert upstream_accepts(clip.source)
        return
    assert conditions, f"{mode_id} built no condition at all from its video input"
    for condition in conditions:
        assert not isinstance(condition.source, (str, Path)), (
            f"{mode_id} handed the pipeline {condition.source!r} -- a path, not frames. "
            f"diffusers dispatches on the payload TYPE at pipeline_ltx2_condition.py:899 "
            f"and raises TypeError for anything it has no branch for."
        )
        assert upstream_accepts(condition.source)


@pytest.mark.parametrize("mode_id", VIDEO_INPUT_MODES)
def test_an_undecodable_upload_fails_before_any_gpu_work(mode_id, tmp_path):
    """Loud, and early. The alternative is discovering it inside the worker,
    where the metered time is already spent and the message arrives stripped."""
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")
    pipe = CapturingPipe()

    with pytest.raises(LTXStudioError) as caught:
        generate_module.generate(
            mode=mode_id,
            seed=7,
            video=str(junk),
            num_frames=9,
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path,
            retention_bytes=10**9,
        )

    assert str(junk) in str(caught.value)
    assert pipe.kwargs == {}, "the pipeline must never have been called"
