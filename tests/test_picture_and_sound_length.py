"""A clip's length is its picture's, an audio upload's is its sound's (F70).

Matroska and WebM give a stream no duration of its own, and the length used to
fall back to ``container.duration``: the LONGEST stream. A 9-frame WebM whose
sound runs 2 s was planned at 41 frames, a shot at frame 17 passed the anchor
check, the worker decoded 9 frames and shortened the run, and diffusers skipped
the shot with only a log line. The same shot beside an MP4 with the same two
tracks was refused before dispatch.

Two rulings, both pinned here:

* before dispatch, a clip is measured by its picture (the video stream's own
  duration, else its packets over its own frame rate) and an audio upload by its
  sound, so the anchor check, the length note and the "trimmed to the maximum"
  note are true for every container;
* in the worker, conditions are built against the DECODED frame count, so a shot
  outside it is a typed refusal, never a silent skip.

Every clip is built here with ``av``; nothing is downloaded. The real-pipeline
case uses the cached tiny checkpoint and skips without it (and without torch).
Maps onto ``evidence/fixes/F70_variation_keyframe/test_f70_variation_keyframe.py``
case by case; see each test's docstring.
"""

from __future__ import annotations

import logging
from fractions import Fraction

import numpy as np
import pytest
from test_exports_contract import factory_returning, one_muxed_file_encoder

from ltx25_studio import conditioning, constraints
from ltx25_studio import generate as generate_module
from ltx25_studio.errors import AudioVideoDesyncError, ConstraintError, LTXStudioError

av = pytest.importorskip("av")

FPS = 24
#: Containers whose streams carry no duration of their own in PyAV.
MATROSKA = {
    "webm": ("webm", "libvpx-vp9", "libopus"),
    "mkv": ("matroska", "h264", "aac"),
}
#: Containers that give each stream its own duration: the control.
PER_STREAM = {
    "mp4": ("mp4", "h264", "aac"),
    "mov": ("mov", "h264", "aac"),
}
CONTAINERS = {**MATROSKA, **PER_STREAM}


def write_clip(path, kind, *, frames=9, fps=FPS, audio_seconds=None, picture=True):
    """A real clip: ``frames`` frames of a moving block at ``fps``, and silent stereo sound."""
    fmt, vcodec, acodec = CONTAINERS[kind]
    with av.open(str(path), "w", format=fmt) as box:
        video = None
        if picture:
            video = box.add_stream(vcodec, rate=fps)
            video.width = video.height = 64
            video.pix_fmt = "yuv420p"
        audio = box.add_stream(acodec, rate=48000, layout="stereo") if audio_seconds else None
        for index in range(frames if picture else 0):
            pixels = np.full((64, 64, 3), 40, dtype=np.uint8)
            pixels[10:30, (index * 6) % 40 : (index * 6) % 40 + 20] = 200
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in video.encode(frame):
                box.mux(packet)
        if video is not None:
            for packet in video.encode():
                box.mux(packet)
        if audio is not None:
            step, total = 960, int(round(audio_seconds * 48000))
            for start in range(0, total, step):
                count = min(step, total - start)
                block = av.AudioFrame.from_ndarray(
                    np.zeros((2, count), dtype=np.float32), format="fltp", layout="stereo"
                )
                block.sample_rate = 48000
                block.pts, block.time_base = start, Fraction(1, 48000)
                for packet in audio.encode(block):
                    box.mux(packet)
            for packet in audio.encode():
                box.mux(packet)
    return str(path)


def clip_with_long_sound(tmp_path, kind, *, frames=9, audio_seconds=2.0):
    """The F70 clip: a short picture whose sound outlasts it."""
    return write_clip(
        tmp_path / f"clip.{kind}", kind, frames=frames, audio_seconds=audio_seconds
    )


def shot(tmp_path, anchor):
    """A one-frame video shot at pixel frame ``anchor``."""
    source = write_clip(tmp_path / "shot.mp4", "mp4", frames=1)
    return conditioning.ConditionSpec(source=source, kind="video", index=anchor, strength=1.0)


class CapturingPipe:
    def __init__(self):
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return type("Output", (), {"frames": ("frame",), "audio": ("sample",)})()


def run_fake(tmp_path, mode, **fields):
    """``generate`` with a fake pipeline: ``(kwargs the pipeline got, result or refusal)``."""
    pipe = CapturingPipe()
    try:
        result = generate_module.generate(
            mode=mode,
            seed=7,
            prompt="a cabin in snow",
            **fields,
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path / "out",
            retention_bytes=10**9,
        )
    except LTXStudioError as exc:
        return pipe.kwargs, exc
    return pipe.kwargs, result


@pytest.fixture
def refused_before_dispatch(monkeypatch, tmp_path):
    """``generate`` with a GPU dispatch registered that must never be reached."""

    def go(expected, **fields):
        reached = []

        def dispatch(**kwargs):
            reached.append(kwargs)
            raise AssertionError("the GPU dispatch was reached by a request that is refused")

        monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
        with pytest.raises(expected) as caught:
            generate_module.generate(
                seed=7,
                pipeline_factory=lambda **_: pytest.fail("the pipeline was built"),
                encoder=one_muxed_file_encoder,
                output_root=tmp_path / "out",
                retention_bytes=10**9,
                **fields,
            )
        assert reached == [], "refused only after the GPU was reserved"
        assert caught.value.gpu_used is False
        return caught.value

    return go


def latent_frames(num_frames: int) -> int:
    ratio = constraints.DEFAULT_TEMPORAL_COMPRESSION_RATIO
    return (int(num_frames) - 1) // ratio + 1


# --------------------------------------------------------------------------
# before dispatch: a clip is its picture
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("frames", "fps"), [(9, 24), (25, 30), (15, 12)])
@pytest.mark.parametrize("kind", list(CONTAINERS))
def test_a_clip_is_planned_at_the_length_of_its_picture(tmp_path, kind, frames, fps):
    """Repro: ``test_root_cause_the_planned_length_is_the_pictures`` (webm, mkv), with the
    MP4/MOV control and two frame rates other than the generation's."""
    clip = write_clip(tmp_path / f"clip.{kind}", kind, frames=frames, fps=fps, audio_seconds=6.0)

    planned, _spec, notes = generate_module.prepare_request(
        generate_module.GenerationRequest(mode="video_to_video", video=clip)
    )

    expected = constraints.largest_num_frames_within(frames / fps, frame_rate=FPS)
    assert planned.num_frames == expected, (kind, frames, fps, planned.num_frames, notes)
    assert notes == (f"length set from your clip: {expected / FPS:.1f} s.",), notes


@pytest.mark.parametrize("kind", list(CONTAINERS))
def test_a_shot_past_the_picture_is_refused_before_dispatch(
    tmp_path, kind, refused_before_dispatch
):
    """Repro: ``test_defect_a_shot_beside_a_matroska_clip_is_placed_or_refused`` (webm,
    mkv) and ``test_control_an_mp4_with_the_same_tracks_refuses_the_shot_before_dispatch``
    (mp4; mov added): one refusal, the same for every container."""
    clip = clip_with_long_sound(tmp_path, kind)

    error = refused_before_dispatch(
        ConstraintError, mode="video_to_video", video=clip, conditions=[shot(tmp_path, 17)]
    )

    assert "outside this clip, which has 9 frames" in str(error), str(error)


@pytest.mark.parametrize("kind", list(MATROSKA))
def test_a_shot_inside_the_picture_is_placed(tmp_path, kind):
    clip = clip_with_long_sound(tmp_path, kind)

    sent, outcome = run_fake(tmp_path, "video_to_video", video=clip, conditions=[shot(tmp_path, 8)])

    assert not isinstance(outcome, LTXStudioError), outcome
    assert sent["num_frames"] == 9
    assert [c.index for c in sent["conditions"]] == [1] and latent_frames(9) == 2
    assert not any("shorter than its file said" in note for note in outcome.warnings)


def test_the_real_pipeline_places_a_shot_and_never_skips_one(tmp_path, caplog):
    """Repro: ``test_defect_the_real_pipeline_does_not_skip_the_shot``. Through ``generate``,
    the runner and the real condition pipeline (tiny, CPU): a shot inside the WebM's 9
    frames is denoised with the clip, and one past them is refused before the
    transformer runs."""
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from test_real_pipeline_tiny import _tiny_snapshot_cached, load_smoke, run_tiny

    if not _tiny_snapshot_cached():
        pytest.skip("the tiny checkpoint is not in the HF cache")
    tiny = load_smoke().tiny_stack(device="cpu")
    clip = clip_with_long_sound(tmp_path, "webm")
    lengths = []

    def length(_module, _args, kwargs):
        lengths.append(kwargs["hidden_states"].shape[1])

    hook = tiny["base"].transformer.register_forward_pre_hook(length, with_kwargs=True)
    caplog.set_level(logging.WARNING)
    try:
        with pytest.raises(ConstraintError, match="outside this clip"):
            run_tiny(tiny, "video_to_video", tmp_path / "far", video=clip,
                     conditions=[shot(tmp_path, 17)])
        assert lengths == [], "refused only after the transformer ran"

        run_tiny(tiny, "video_to_video", tmp_path / "near", video=clip,
                 conditions=[shot(tmp_path, 8)])
    finally:
        hook.remove()

    skipped = [r.getMessage() for r in caplog.records if "will be skipped" in r.getMessage()]
    assert not skipped, f"diffusers dropped the user's shot: {skipped}"
    # 9 frames of 64x64 are 2x2x2 latents (8 tokens); the placed one-frame shot adds 4.
    assert lengths and lengths[0] == 8 + 4, lengths


def test_video_to_audio_checks_the_shot_against_the_length_that_runs(tmp_path):
    """Repro: ``test_scope_video_to_audio_runs_at_the_requested_length_so_its_shot_is_placed``."""
    clip = clip_with_long_sound(tmp_path, "webm")

    sent, outcome = run_fake(
        tmp_path, "video_to_audio", video=clip, num_frames=41, conditions=[shot(tmp_path, 17)]
    )
    assert not isinstance(outcome, LTXStudioError), outcome
    assert [c.index for c in sent["conditions"]] == [0, 3]
    assert latent_frames(sent["num_frames"]) == 6

    sent, outcome = run_fake(
        tmp_path, "video_to_audio", video=clip, num_frames=9, conditions=[shot(tmp_path, 17)]
    )
    assert isinstance(outcome, ConstraintError) and sent == {}, outcome


def test_a_matroska_shot_source_sets_no_length(tmp_path):
    """Repro: ``test_scope_a_matroska_shot_source_does_not_set_any_length``."""
    clip = write_clip(tmp_path / "clip.mp4", "mp4", frames=25)
    source = write_clip(tmp_path / "shot.webm", "webm", frames=1, audio_seconds=2.0)
    keyframe = conditioning.ConditionSpec(source=source, kind="video", index=17, strength=1.0)

    sent, outcome = run_fake(tmp_path, "video_to_video", video=clip, conditions=[keyframe])

    assert not isinstance(outcome, LTXStudioError), outcome
    assert sent["num_frames"] == 25 and [c.index for c in sent["conditions"]] == [3]


def test_a_picture_only_matroska_clip_keeps_its_shot(tmp_path):
    """Repro: ``test_scope_an_honest_matroska_clip_keeps_its_shot``."""
    clip = write_clip(tmp_path / "clip.webm", "webm", frames=25)

    sent, outcome = run_fake(
        tmp_path, "video_to_video", video=clip, conditions=[shot(tmp_path, 17)]
    )

    assert not isinstance(outcome, LTXStudioError), outcome
    assert sent["num_frames"] == 25 and [c.index for c in sent["conditions"]] == [3]


# --------------------------------------------------------------------------
# the notes are true
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(MATROSKA))
def test_a_short_picture_with_long_sound_is_not_called_trimmed(tmp_path, kind):
    """Repro: ``item3_long_audio_note.py`` -- a 9-frame clip with 25 s of sound was told
    "Your clip is longer than the 20 s maximum, so it is trimmed to fit", then "your clip
    held 9 frames, fewer than its length promised"."""
    clip = clip_with_long_sound(tmp_path, kind, audio_seconds=25.0)

    planned, _spec, notes = generate_module.prepare_request(
        generate_module.GenerationRequest(mode="video_to_video", video=clip)
    )
    _sent, result = run_fake(tmp_path, "video_to_video", video=clip)

    assert planned.num_frames == 9
    assert notes == ("length set from your clip: 0.4 s.",), notes
    assert not isinstance(result, LTXStudioError), result
    said = " ".join(result.warnings)
    assert "maximum" not in said and "shorter than its file said" not in said, result.warnings


@pytest.mark.parametrize("kind", list(MATROSKA))
def test_a_picture_longer_than_the_maximum_is_said_to_be_trimmed(tmp_path, kind):
    """22 frames at 1 fps is a 22 s picture: planned at the maximum, and said so."""
    clip = write_clip(tmp_path / f"long.{kind}", kind, frames=22, fps=1)

    planned, _spec, notes = generate_module.prepare_request(
        generate_module.GenerationRequest(mode="video_to_video", video=clip)
    )

    assert planned.num_frames == constraints.MAX_NUM_FRAMES
    assert notes == (
        "length set from your clip: 20.0 s. Your clip is longer than the 20 s maximum, so it "
        "is trimmed to fit.",
    ), notes


# --------------------------------------------------------------------------
# the audio twin: an audio upload is its sound
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(CONTAINERS))
def test_an_audio_upload_is_measured_by_its_sound_not_its_picture(tmp_path, kind):
    """A 2 s sound under a 5.04 s picture, given as the audio input: 2 s, in every container."""
    upload = write_clip(tmp_path / f"talk.{kind}", kind, frames=121, audio_seconds=2.0)

    assert generate_module._audio_seconds(upload) == pytest.approx(2.0, abs=0.01)
    planned, _spec, notes = generate_module.prepare_request(
        generate_module.GenerationRequest(mode="audio_to_video", audio=upload, auto_duration=True)
    )
    assert planned.num_frames == constraints.largest_num_frames_within(2.0, frame_rate=FPS) == 41
    assert notes == ("length set from your audio: 1.7 s.",), notes


@pytest.mark.parametrize("kind", list(MATROSKA))
def test_a_set_length_past_the_sound_is_refused_before_dispatch(
    tmp_path, kind, refused_before_dispatch
):
    """Measured by the picture (5.04 s), this passed the gate and was refused in the worker."""
    upload = write_clip(tmp_path / f"talk.{kind}", kind, frames=121, audio_seconds=2.0)

    error = refused_before_dispatch(
        AudioVideoDesyncError, mode="audio_to_video", audio=upload, num_frames=121
    )

    assert error.audio_seconds == pytest.approx(2.0, abs=0.01)
    assert str(error).startswith("Your audio is 2.0 s and the length is set to 5.0 s."), error


@pytest.mark.parametrize("kind", list(MATROSKA))
def test_sound_alone_in_matroska_is_measured(tmp_path, kind):
    upload = write_clip(tmp_path / f"sound.{kind}", kind, audio_seconds=2.0, picture=False)

    assert generate_module._audio_seconds(upload) == pytest.approx(2.0, abs=0.01)


# --------------------------------------------------------------------------
# in the worker: the decoded frames are the authority
# --------------------------------------------------------------------------


def _over_promising(monkeypatch, seconds: float) -> None:
    """A file whose header promises ``seconds`` of picture, more than it holds."""
    monkeypatch.setattr(generate_module, "_video_seconds", lambda source: seconds)


def test_a_shot_past_the_decoded_frames_is_refused_in_the_worker(tmp_path, monkeypatch):
    """Planned at 41 frames from the header, the clip decodes to 9: a shot at 17 passed the
    gate, and is refused (typed) where the conditions are built, never skipped."""
    clip = write_clip(tmp_path / "clip.mp4", "mp4", frames=9)
    _over_promising(monkeypatch, 2.0)

    sent, outcome = run_fake(
        tmp_path, "video_to_video", video=clip, conditions=[shot(tmp_path, 17)]
    )

    assert isinstance(outcome, ConstraintError), outcome
    assert "outside this clip, which has 9 frames" in str(outcome), outcome
    assert sent == {}, "the pipeline was called"


def test_a_shot_inside_the_decoded_frames_is_placed_when_the_clip_shortens(
    tmp_path, monkeypatch
):
    clip = write_clip(tmp_path / "clip.mp4", "mp4", frames=9)
    _over_promising(monkeypatch, 2.0)

    sent, outcome = run_fake(tmp_path, "video_to_video", video=clip, conditions=[shot(tmp_path, 8)])

    assert not isinstance(outcome, LTXStudioError), outcome
    assert sent["num_frames"] == 9 and [c.index for c in sent["conditions"]] == [1]
    assert "your clip is shorter than its file said, so the video is 0.4 s long." in (
        outcome.warnings
    )
