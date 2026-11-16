"""Media in and media out, measured on real files written here by ``av``.

* A conditioning video is read at the generation's frame rate, by timestamp,
  and the right way up (F16).
* The user's own audio muxed into an ``audio_to_video`` mp4 is 48 kHz stereo
  AAC, at its own pitch and length (F17).
* 8-bit PCM is centred on zero (F18).
* Conditioning audio is encoded to the distribution's mode, not a random draw
  (F19).
* The mp4 is encoded at CRF 17 with bt709 tags and the index at the front (F20).
* A video+audio run also hands back its soundtrack as a wav (F30).

Nothing is committed as a fixture: every clip and every track is generated in
the test, so what is measured is the file the code under test produced.
"""

from __future__ import annotations

import struct
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ltx25_studio import audio, conditioning, outputs

av = pytest.importorskip("av")

#: The vocoder's rate on this stack, which the writer is handed.
VOCODER_RATE = 48000


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


# Review r1, media-io-1: H.264 with B-frames in an AVI (DVR, CCTV and dashcam
# exports). AVI has no presentation stamps, so the decoder hands back frames in
# display order carrying decode-order pts (1, 4, 3, 5, 2, 8, ...).


def decoded_pts(path) -> list:
    with av.open(str(path)) as container:
        return [frame.pts for frame in container.decode(video=0)]


def nearest_source_frames(count, *, source_rate, rate) -> list[int]:
    """The reference: for each output time, the nearest source frame (ties earlier)."""
    times = [Fraction(k, source_rate) for k in range(count)]
    wanted = []
    for i in range(int(Fraction(count, source_rate) * rate)):
        t = Fraction(i, rate)
        wanted.append(min(range(count), key=lambda k: (abs(times[k] - t), times[k])))
    return wanted


@pytest.mark.parametrize(("source_rate", "rate"), [(24, 24), (30, 24), (60, 24)])
def test_an_avi_with_b_frames_is_read_by_display_time(source_rate, rate, tmp_path):
    count = 60
    source = write_clip(
        tmp_path / "dvr.avi", [marker_frame(k) for k in range(count)], rate=source_rate
    )
    stamps = decoded_pts(source)
    assert stamps != sorted(stamps), "the fixture lost its decode-order stamps"

    frames = conditioning.decode_video_frames(source, frame_rate=rate)

    got = [marker_index(frame) for frame in frames]
    assert got == nearest_source_frames(count, source_rate=source_rate, rate=rate), got


def test_an_avi_read_at_its_own_rate_matches_the_frame_by_frame_read(tmp_path):
    source = write_clip(tmp_path / "dvr.avi", [marker_frame(k) for k in range(48)], rate=24)

    timed = conditioning.decode_video_frames(source, frame_rate=24)
    in_order = conditioning.decode_video_frames(source)

    assert [marker_index(f) for f in timed] == [marker_index(f) for f in in_order]
    assert timed.shape[0] == in_order.shape[0] == 48


def test_a_stream_with_no_timestamps_is_timed_by_its_frame_duration(tmp_path):
    """A raw Annex B .h264 has no stamps, and FFmpeg guesses 25 fps for its
    average rate; each frame still carries its real duration (1/60 s here)."""
    source = write_clip(tmp_path / "raw.h264", [marker_frame(k) for k in range(60)], rate=60)
    assert set(decoded_pts(source)) == {None}, "the fixture carries timestamps after all"

    frames = conditioning.decode_video_frames(source, frame_rate=24)

    got = [marker_index(frame) for frame in frames]
    assert got == nearest_source_frames(60, source_rate=60, rate=24), got


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


# ---------------------------------------------------------------------------
# F17 -- the user's audio in the mp4
# ---------------------------------------------------------------------------


def tone(channels: int, rate: int, *, seconds: float = 2.0):
    """A sine per channel: 440 Hz, except a stereo right channel at 660 Hz."""
    t = np.arange(int(round(seconds * rate))) / rate
    hz = (440.0, 660.0) if channels == 2 else (440.0,) * channels
    return np.stack([0.5 * np.sin(2 * np.pi * f * t) for f in hz]).astype(np.float32)


class JointOutput:
    """Two seconds of picture at 24 fps, and a silent 48 kHz stereo model track."""

    def __init__(self, model_audio=None):
        self.frames = np.zeros((48, 64, 64, 3), dtype=np.uint8)
        self.frames[:, 16:48, 16:48] = 200
        self.audio = (
            np.zeros((2, 2 * VOCODER_RATE), dtype=np.float32)
            if model_audio is None
            else model_audio
        )


def write_joint(tmp_path, *, mode="audio_to_video", supplied=None, model_audio=None, notes=None):
    return outputs.write_output(
        JointOutput(model_audio),
        run_dir=tmp_path / "run",
        mode=mode,
        exports=("audio", "video"),
        seed=1,
        frame_rate=24,
        sampling_rate=VOCODER_RATE,
        warnings=notes,
        supplied_audio=supplied,
    )


def decode_audio(path):
    """``(codec, rate, channels, samples[C, N])`` of the first audio stream."""
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        context = stream.codec_context
        chunks = []
        for frame in container.decode(stream):
            data = frame.to_ndarray()
            if not frame.format.is_planar:
                data = data.reshape(-1, context.channels).T
            chunks.append(data)
        samples = np.concatenate(chunks, axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / 32768.0
    return context.name, context.sample_rate, context.channels, samples


def peak_hz(signal, rate) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    return float(np.fft.rfftfreq(signal.size, 1.0 / rate)[np.argmax(spectrum)])


UPLOADS = [
    pytest.param(1, 24000, id="mono-24k"),
    pytest.param(2, 44100, id="stereo-44.1k"),
    pytest.param(1, 8000, id="mono-8k"),
    pytest.param(6, 48000, id="5.1-48k"),
    pytest.param(1, 37000, id="mono-37k-nonstandard"),
]


@pytest.mark.parametrize(("channels", "rate"), UPLOADS)
def test_a_supplied_upload_is_muxed_as_48k_stereo_aac_at_its_own_pitch(channels, rate, tmp_path):
    notes: list[str] = []
    written = write_joint(
        tmp_path, supplied=outputs.SuppliedAudio(tone(channels, rate), rate), notes=notes
    )

    codec, got_rate, got_channels, samples = decode_audio(written["video"])

    assert (codec, got_rate, got_channels) == ("aac", VOCODER_RATE, 2), (
        f"a {channels}-channel {rate} Hz upload was muxed as {codec} {got_rate} Hz "
        f"{got_channels} ch; the soundtrack is 48 kHz stereo AAC"
    )
    # AAC pads its last frame, so the track is exact only to one 1024-sample frame.
    assert abs(samples.shape[1] - 2 * VOCODER_RATE) < 1024, samples.shape
    expected_hz = (440.0, 660.0) if channels == 2 else (440.0, 440.0)
    measured = tuple(peak_hz(samples[c], VOCODER_RATE) for c in range(2))
    assert all(abs(m - e) <= 1.0 for m, e in zip(measured, expected_hz, strict=True)), (
        f"the tones came back at {measured} Hz, expected {expected_hz}: the track was "
        f"pitch-shifted or its channels swapped"
    )
    assert float(np.abs(samples).max()) <= 1.0, "the downmix clipped"
    assert any("48 kHz" in note and "stereo" in note for note in notes), (
        f"the upload was converted and the run does not say so: {notes}"
    )


@pytest.mark.parametrize(("channels", "rate"), UPLOADS)
def test_the_conversion_keeps_the_uploads_length_to_one_sample(channels, rate, tmp_path):
    """Measured on the wav, which carries the muxed samples without AAC's padding."""
    written = write_joint(tmp_path, supplied=outputs.SuppliedAudio(tone(channels, rate), rate))

    _, got_rate, _, samples = decode_audio(written["audio"])

    seconds_in = tone(channels, rate).shape[1] / rate
    seconds_out = samples.shape[1] / got_rate
    assert abs(seconds_out - seconds_in) <= 1.0 / VOCODER_RATE, (
        f"{seconds_in:.6f} s of upload became {seconds_out:.6f} s"
    )


def test_a_mono_upload_is_duplicated_not_attenuated(tmp_path):
    """Mono to stereo is a copy into both channels, at the same level."""
    mono = tone(1, VOCODER_RATE)
    written = write_joint(tmp_path, supplied=outputs.SuppliedAudio(mono, VOCODER_RATE))

    _, _, _, samples = decode_audio(written["audio"])

    assert np.allclose(samples[0], samples[1], atol=1 / 16384)
    assert np.allclose(samples[0], mono[0], atol=1 / 16384), (
        f"peak {np.abs(samples[0]).max():.3f} against the upload's {np.abs(mono[0]).max():.3f}"
    )


def test_an_upload_already_in_the_soundtrack_format_is_not_reported(tmp_path):
    notes: list[str] = []
    write_joint(
        tmp_path, supplied=outputs.SuppliedAudio(tone(2, VOCODER_RATE), VOCODER_RATE), notes=notes
    )
    assert not any("converted" in note for note in notes), notes


def test_the_models_own_track_is_not_reported_as_converted(tmp_path):
    notes: list[str] = []
    write_joint(
        tmp_path, mode="text_to_audio_video", model_audio=tone(2, VOCODER_RATE), notes=notes
    )
    assert not any("converted" in note for note in notes), notes


def test_a_torch_waveform_from_the_loader_is_conformed_too(tmp_path):
    """``audio.load_waveform`` hands back a torch tensor; that is the real input."""
    torch = pytest.importorskip("torch")
    supplied = outputs.SuppliedAudio(torch.from_numpy(tone(1, 24000)), 24000)

    written = write_joint(tmp_path, supplied=supplied)

    _, got_rate, got_channels, samples = decode_audio(written["video"])
    assert (got_rate, got_channels) == (VOCODER_RATE, 2)
    assert abs(peak_hz(samples[0], VOCODER_RATE) - 440.0) <= 1.0


# ---------------------------------------------------------------------------
# F18 -- 8-bit PCM
# ---------------------------------------------------------------------------


def test_unsigned_8_bit_samples_are_centred_on_zero():
    """8-bit WAV is offset binary: 128 is silence."""
    scaled = audio._as_float_samples(np.array([[0, 128, 255]], dtype=np.uint8))

    assert scaled.dtype == np.float32
    assert scaled.tolist() == [[-1.0, 0.0, 127 / 128]]


def test_signed_16_bit_scaling_is_unchanged():
    scaled = audio._as_float_samples(np.array([[-32768, 0, 16384]], dtype=np.int16))
    assert scaled.tolist() == [[-1.0, 0.0, 0.5]]


def test_a_real_8_bit_wav_loads_without_a_dc_offset(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "u8.wav"
    square = bytes([128 + 100, 128 - 100] * 4000)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8000)
        handle.writeframes(square)

    waveform, rate = audio.load_waveform(path)
    data = waveform.numpy()

    assert rate == 8000
    assert abs(float(data.mean())) < 1e-3, (
        f"8-bit audio loaded with a DC offset of {data.mean():.3f}"
    )
    assert np.allclose(np.unique(data), [-100 / 128, 100 / 128]), np.unique(data)


# ---------------------------------------------------------------------------
# F19 -- the conditioning encode uses the mode
# ---------------------------------------------------------------------------


class RecordingDistribution:
    def __init__(self, *, has_mode=True):
        self.calls: list = []
        if has_mode:
            self.mode = self._mode

    def _mode(self):
        self.calls.append("mode")
        return "the mode"

    def sample(self, generator=None):
        self.calls.append(("sample", generator))
        return "a draw"


def test_the_conditioning_encode_takes_the_mode_not_a_draw():
    distribution = RecordingDistribution()
    generator = object()

    latents = audio._sampled(SimpleNamespace(latent_dist=distribution), generator)

    assert latents == "the mode"
    assert distribution.calls == ["mode"], distribution.calls


def test_an_encoder_without_a_mode_still_draws_with_the_runs_generator():
    distribution = RecordingDistribution(has_mode=False)
    generator = object()

    latents = audio._sampled(SimpleNamespace(latent_dist=distribution), generator)

    assert latents == "a draw"
    assert distribution.calls == [("sample", generator)]


def test_the_real_vae_distribution_encodes_the_same_latents_whatever_the_seed():
    torch = pytest.importorskip("torch")
    vae = pytest.importorskip("diffusers.models.autoencoders.vae")
    parameters = torch.randn(1, 16, 6, 4, generator=torch.Generator().manual_seed(0))
    distribution = vae.DiagonalGaussianDistribution(parameters)
    encoded = SimpleNamespace(latent_dist=distribution)

    first = audio._sampled(encoded, torch.Generator().manual_seed(1))
    second = audio._sampled(encoded, torch.Generator().manual_seed(2))

    assert torch.equal(first, second)
    assert torch.equal(first, distribution.mean)


def test_an_audio_run_hands_the_pipeline_the_mode_of_the_real_distribution(
    tmp_path, audio_input_ready
):
    """Through ``generate``: two seeds, one upload, identical conditioning latents."""
    torch = pytest.importorskip("torch")
    vae = pytest.importorskip("diffusers.models.autoencoders.vae")
    from conftest import FakeAudioVae, inputs_for
    from test_exports_contract import factory_returning, one_muxed_file_encoder

    from ltx25_studio import generate as generate_module

    class RealDistributionVae(FakeAudioVae):
        def encode(self, mel):
            shape = super().encode(mel).latent_dist.sample().shape
            batch, channels, frames, bins = shape
            count = batch * 2 * channels * frames * bins
            parameters = torch.linspace(-1.0, 1.0, count).reshape(batch, 2 * channels, frames, bins)
            return SimpleNamespace(latent_dist=vae.DiagonalGaussianDistribution(parameters))

    class Pipe:
        def __init__(self):
            self.audio_vae = RealDistributionVae()
            self.calls: list[dict] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(frames=("frame",), audio=("sample",))

    handed = []
    for seed in (1, 2):
        pipe = Pipe()
        generate_module.generate(
            mode="audio_to_audio",
            seed=seed,
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path / str(seed),
            retention_bytes=10**9,
            **inputs_for("audio_to_audio"),
        )
        handed.append(pipe.calls[-1]["audio_latents"])

    assert torch.equal(handed[0], handed[1]), "the same upload encoded differently under two seeds"


# ---------------------------------------------------------------------------
# F20 -- the encode
# ---------------------------------------------------------------------------


def top_level_boxes(path) -> list[str]:
    data = Path(path).read_bytes()
    boxes, offset = [], 0
    while offset + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[offset : offset + 8])
        if size == 1:
            size = struct.unpack(">Q", data[offset + 8 : offset + 16])[0]
        boxes.append(kind.decode("latin-1"))
        offset += size if size else len(data)
    return boxes


def x264_settings(path) -> str:
    """The settings string x264 embeds in its first SEI message."""
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
        for side_data in frame.side_data:
            text = bytes(side_data).decode("latin-1", errors="replace")
            if "x264" in text:
                return text
    return ""


@pytest.fixture
def written_video(tmp_path):
    return write_joint(tmp_path, mode="text_to_audio_video", model_audio=tone(2, VOCODER_RATE))[
        "video"
    ]


def test_the_mp4_is_h264_yuv420p_tagged_bt709_limited_range(written_video):
    with av.open(str(written_video)) as container:
        context = container.streams.video[0].codec_context
        tags = (context.name, context.pix_fmt, context.color_primaries, context.color_trc)
        tags += (context.colorspace, context.color_range)

    reformatter = av.video.reformatter
    assert tags == (
        "h264",
        "yuv420p",
        reformatter.ColorPrimaries.BT709,
        reformatter.ColorTrc.BT709,
        reformatter.Colorspace.ITU709,
        reformatter.ColorRange.MPEG,
    ), f"codec, pix_fmt, primaries, transfer, matrix, range = {tags}"


def test_the_index_precedes_the_media_so_playback_can_start_early(written_video):
    boxes = top_level_boxes(written_video)
    assert "moov" in boxes and "mdat" in boxes, boxes
    assert boxes.index("moov") < boxes.index("mdat"), f"moov after mdat: {boxes}"


def test_the_encode_runs_at_crf_17(written_video):
    settings = x264_settings(written_video)
    assert "crf=17.0" in settings, f"x264 settings: {settings[-400:]!r}"
    assert outputs.VIDEO_CRF == 17 and outputs.VIDEO_PRESET == "medium"


def test_colours_survive_the_encode(tmp_path):
    """The tag must match the matrix the pixels were converted with."""
    output = JointOutput(tone(2, VOCODER_RATE))
    output.frames[:] = (200, 120, 40)
    written = outputs.write_output(
        output,
        run_dir=tmp_path / "run",
        mode="text_to_video",
        exports=("video",),
        frame_rate=24,
        sampling_rate=VOCODER_RATE,
    )
    with av.open(str(written["video"])) as container:
        decoded = next(container.decode(video=0)).to_ndarray(format="rgb24")

    centre = decoded[32, 32].astype(int)
    assert np.abs(centre - (200, 120, 40)).max() <= 3, f"(200, 120, 40) came back as {centre}"


# ---------------------------------------------------------------------------
# F30 -- the soundtrack on its own
# ---------------------------------------------------------------------------


def test_a_video_and_audio_write_also_returns_the_soundtrack_as_a_wav(tmp_path):
    model = tone(2, VOCODER_RATE)
    written = write_joint(tmp_path, mode="text_to_audio_video", model_audio=model)

    video, track = written["video"], written["audio"]
    assert video.name == outputs.VIDEO_FILENAME and track.name == outputs.AUDIO_FILENAME
    assert video.parent == track.parent

    codec, rate, channels, samples = decode_audio(track)
    assert (codec, rate, channels) == ("pcm_s16le", VOCODER_RATE, 2)
    assert samples.shape == model.shape
    assert np.abs(samples - model).max() <= 1.0 / 32767, "the wav is not the muxed samples"

    with av.open(str(video)) as container:
        kinds = sorted(stream.type for stream in container.streams)
    assert kinds == ["audio", "video"], f"the mp4 must still carry the sound: {kinds}"


def test_the_wav_of_a_supplied_upload_is_the_converted_track(tmp_path):
    written = write_joint(tmp_path, supplied=outputs.SuppliedAudio(tone(1, 24000), 24000))

    _, rate, channels, samples = decode_audio(written["audio"])

    assert (rate, channels) == (VOCODER_RATE, 2)
    assert abs(peak_hz(samples[0], VOCODER_RATE) - 440.0) <= 1.0


def test_a_video_only_write_returns_no_wav(tmp_path):
    written = outputs.write_output(
        JointOutput(),
        run_dir=tmp_path / "run",
        mode="text_to_video",
        exports=("video",),
        frame_rate=24,
        sampling_rate=VOCODER_RATE,
    )
    assert set(written) == {"video"}
    assert not (tmp_path / "run" / outputs.AUDIO_FILENAME).exists()


def test_a_joint_generation_hands_the_audio_player_a_wav(tmp_path, audio_input_ready):
    """Through ``generate`` with the production writer."""
    from test_encoder_contract import JOINT_MODES, run_production

    for mode_id in JOINT_MODES:
        result = run_production(mode_id, tmp_path / mode_id)
        assert result.audio_path.suffix == ".wav", f"{mode_id}: audio_path={result.audio_path}"
        assert result.video_path.suffix == ".mp4", f"{mode_id}: video_path={result.video_path}"
        assert result.output_path == result.video_path
