"""Spec for what ``generate()`` owes the encoder — PLAN §5.3, §5.4, §5.7.

**The injected-encoder seam hides a whole class of defect.** Every existing
generate() spec passes ``encoder=`` a fake, and a fake accepts whatever it is
handed. So ``generate()`` could forward an argument set that ``outputs.write_output``
cannot actually write with, and every one of those tests would still pass while
the production path failed deterministically for a mode nobody exercised.

That is not hypothetical. ``outputs`` refuses to invent two numbers, both on
purpose (PLAN §5.4): the **sampling rate**, which comes from
``vocoder.config.output_sampling_rate``, and the **frame rate**, which is the
rate the pipeline generated at. A container written at a wrong or absent rate is
pitch-shifted or plays at the wrong speed — and reports no error at all. So the
writer raises instead, and whatever ``generate()`` fails to thread through
becomes a run that dies at the last step.

This file therefore tests the **real** encoder, not an injected one:

* ``encoder=None`` — the production path — for every mode in the registry, with
  real arrays, through the real ``av`` writer, and a real file at the end of it;
* the muxing itself, demuxed back and inspected, because "wrote an mp4" and
  "wrote an mp4 with the audio in it" are the same result to everything except
  the user's ears (PLAN: a silent video is the §5.3 failure one layer out);
* the refusal, when the pipeline genuinely cannot supply what audio needs.

Still no GPU, no weights, no network: the pipeline is faked, the media stack is
``av``, which is already a pinned runtime dependency.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeAudioVae, audio_file, inputs_for
from test_exports_contract import MODE_IDS, factory_returning

from ltx25_studio import generate as generate_module
from ltx25_studio import modes, outputs
from ltx25_studio.errors import EncodeError

av = pytest.importorskip("av", reason="the media stack is a pinned runtime dependency")
np = pytest.importorskip("numpy")

#: What the fake vocoder reports. Distinctive so a hardcoded 44100 anywhere in
#: the path shows up as a mismatch rather than blending in.
VOCODER_SAMPLING_RATE = 48000

#: Small enough to keep the suite fast, large enough to be a real encode.
FRAME_COUNT = 2
FRAME_SIZE = 64
AUDIO_CHANNELS = 2
AUDIO_SAMPLES = VOCODER_SAMPLING_RATE // 10  # 0.1s


class RealOutput:
    """A pipeline output the real writer can actually encode.

    Stereo, because the vocoder is stereo and PLAN forbids a mono downmix
    anywhere in the path — a fake that handed back mono could not detect one.
    """

    def __init__(self):
        self.frames = np.zeros((FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        self.audio = np.zeros((AUDIO_CHANNELS, AUDIO_SAMPLES), dtype=np.float32)


class VocoderPipe:
    """A pipeline that exposes the vocoder ``generate()`` must read the rate off.

    It has an audio VAE too, because the audio-input modes encode their upload.
    """

    def __init__(self, sampling_rate=VOCODER_SAMPLING_RATE):
        self.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=sampling_rate))
        self.audio_vae = FakeAudioVae()
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return RealOutput()


class VocoderlessPipe:
    """A pipeline with nothing to read the sampling rate off (it can still encode an upload)."""

    def __init__(self):
        self.audio_vae = FakeAudioVae()

    def __call__(self, **kwargs):
        return RealOutput()


@pytest.fixture(autouse=True)
def _uploads_load_without_torch(audio_input_ready):
    """The audio modes get a real upload; its loader is stubbed so no torch is needed."""


def run_production(mode_id, tmp_path, *, pipe=None, **overrides):
    """``encoder=None`` — the real ``outputs.write_output``, no fake in the way.

    Real inputs for every kind the mode requires. A caller in another module
    must also request the ``audio_input_ready`` fixture for the audio modes.
    """
    return generate_module.generate(
        mode=mode_id,
        seed=7,
        width=FRAME_SIZE,
        height=FRAME_SIZE,
        pipeline_factory=factory_returning(pipe if pipe is not None else VocoderPipe()),
        encoder=None,
        output_root=tmp_path,
        retention_bytes=10**9,
        **{**inputs_for(mode_id), **overrides},
    )


# --------------------------------------------------------------------------
# the production path has to reach a file, for every mode
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_the_production_encoder_can_actually_write_this_modes_artifact(mode_id, tmp_path):
    """The seam every injected-encoder test steps over.

    A fake encoder accepts any argument set. The real one raises on the ones it
    was not given, and PLAN §5.4 makes that deliberate: it will not guess a rate
    it was supposed to be handed. So this drives the writer for real — if
    ``generate()`` stops forwarding something ``write_output`` requires, this is
    the test that says so, whichever argument it happens to be.
    """
    spec = modes.get_mode(mode_id)
    result = run_production(mode_id, tmp_path)

    for track in spec.exports:
        path = Path(getattr(result, f"{track}_path"))
        assert path.is_file() and path.stat().st_size > 0, (
            f"mode {mode_id!r} declares exports={list(spec.exports)} but the production "
            f"writer produced no usable {track} file at {path}."
        )


@pytest.mark.parametrize("mode_id", [m for m in MODE_IDS if "audio" in modes.MODES[m].exports])
def test_an_audio_exporting_mode_is_handed_the_vocoders_sampling_rate(mode_id, tmp_path):
    """PLAN §5.4: the rate comes off the loaded vocoder, never from a default here.

    ``outputs._write_audio`` raises on a missing rate rather than assuming one,
    so a mode that exports audio and is not handed the rate fails at the very
    last step of an otherwise complete — and already paid for — generation.
    """
    seen: dict = {}
    real_write = outputs.write_output

    def spy(output, **kwargs):
        seen.update(kwargs)
        return real_write(output, **kwargs)

    original = outputs.write_output
    outputs.write_output = spy
    try:
        run_production(mode_id, tmp_path)
    finally:
        outputs.write_output = original

    assert seen.get("sampling_rate") == VOCODER_SAMPLING_RATE, (
        f"mode {mode_id!r} exports audio, but the encoder was handed "
        f"sampling_rate={seen.get('sampling_rate')!r} instead of the vocoder's "
        f"{VOCODER_SAMPLING_RATE}. outputs._write_audio refuses to guess a rate, so this "
        f"mode dies on the production path."
    )


@pytest.mark.parametrize("mode_id", [m for m in MODE_IDS if "video" in modes.MODES[m].exports])
def test_a_video_exporting_mode_is_handed_a_frame_rate(mode_id, tmp_path):
    """The same trap, the other number.

    ``outputs._write_video`` raises on a missing frame rate for exactly the
    reason §5.4 gives: a container written at the wrong rate plays at the wrong
    speed and reports no error. It is the caller's to supply.
    """
    seen: dict = {}
    real_write = outputs.write_output

    def spy(output, **kwargs):
        seen.update(kwargs)
        return real_write(output, **kwargs)

    original = outputs.write_output
    outputs.write_output = spy
    try:
        run_production(mode_id, tmp_path)
    finally:
        outputs.write_output = original

    frame_rate = seen.get("frame_rate")
    assert frame_rate is not None, (
        f"mode {mode_id!r} exports video, but the encoder was handed no frame_rate. "
        f"outputs._write_video refuses to assume one, so this mode dies at the last step "
        f"of an already-paid-for generation."
    )
    assert float(frame_rate) > 0, f"frame_rate must be positive, got {frame_rate!r}"


# --------------------------------------------------------------------------
# muxing: an mp4 with no audio in it is not a success
# --------------------------------------------------------------------------

JOINT_MODES = tuple(m for m in MODE_IDS if set(modes.MODES[m].exports) == {"audio", "video"})


def _streams(path):
    with av.open(str(path)) as container:
        return (
            [s for s in container.streams if s.type == "video"],
            [s for s in container.streams if s.type == "audio"],
        )


@pytest.mark.parametrize("mode_id", JOINT_MODES)
def test_a_joint_mode_writes_a_container_that_really_carries_the_audio(mode_id, tmp_path):
    """Demuxed and inspected, because the file plays either way.

    Writing the frames and dropping the audio produces a video that opens, plays,
    and is wrong: no error, no warning, a plausible-looking artifact. That is the
    §5.3 silent-failure class one layer further out, and the only way to tell is
    to open the file back up.
    """
    result = run_production(mode_id, tmp_path)
    video_streams, audio_streams = _streams(result.video_path)

    assert len(video_streams) == 1, f"{mode_id!r} wrote {len(video_streams)} video streams"
    assert len(audio_streams) == 1, (
        f"mode {mode_id!r} exports BOTH audio and video, and the container it wrote at "
        f"{result.video_path} has {len(audio_streams)} audio streams. This is a silent video "
        f"reported as a success — LTX-2.5's headline capability, dropped without an error."
    )


@pytest.mark.parametrize("mode_id", [m for m in MODE_IDS if "video" in modes.MODES[m].exports])
def test_the_drop_warning_fires_exactly_where_a_track_is_actually_dropped(mode_id, tmp_path):
    """``dropped_audio_warning`` had no test at all. Both directions, now pinned.

    The model is jointly audio-visual with no video-off switch, so every
    video-exporting run renders audio. A mode that does not export it discards
    it and MUST say so -- the user paid GPU-seconds for that track. A mode that
    does export it keeps it and must NOT say so.

    The false direction is the one that just bit: ``audio_to_video`` conditioned
    on an audio upload, dropped the generated track, and told the user to "use
    the matching audio+video mode to keep it" -- when its ``audio_counterpart``
    is None and no such mode has ever existed. A stale warning naming a remedy
    that does not exist sends the user hunting for a mode nobody built, which is
    worse than silence, not better.
    """
    result = run_production(mode_id, tmp_path)
    dropped = [note for note in result.warnings if "dropped rather than muxed" in note]

    if "audio" in modes.MODES[mode_id].exports:
        assert not dropped, (
            f"{mode_id!r} exports audio and keeps its track, yet reports it as dropped: {dropped}"
        )
    else:
        assert dropped, (
            f"{mode_id!r} exports video only, so this run rendered an audio track and threw it "
            f"away, and the result says nothing about it: {list(result.warnings)}"
        )


@pytest.mark.parametrize("mode_id", JOINT_MODES)
def test_the_muxed_audio_track_contains_actual_samples(mode_id, tmp_path):
    """An audio stream declared and never written is the same lie with a header."""
    result = run_production(mode_id, tmp_path)

    decoded = 0
    with av.open(str(result.video_path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        for frame in container.decode(stream):
            decoded += frame.samples

    assert decoded > 0, (
        f"mode {mode_id!r} wrote an audio stream into {result.video_path} that decodes to "
        f"zero samples — a silent track with a header on it."
    )


@pytest.mark.parametrize("mode_id", JOINT_MODES)
def test_the_muxed_audio_keeps_every_channel(mode_id, tmp_path):
    """PLAN: the vocoder is stereo and a mono downmix is a silent quality regression."""
    result = run_production(mode_id, tmp_path)

    with av.open(str(result.video_path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        channels = stream.codec_context.layout.nb_channels

    assert channels == AUDIO_CHANNELS, (
        f"mode {mode_id!r} was handed {AUDIO_CHANNELS}-channel audio and wrote {channels}. "
        f"Nothing in this path may downmix."
    )


def test_a_supplied_upload_reaches_the_container_as_48k_stereo_at_its_own_pitch(
    tmp_path, monkeypatch
):
    """The mode-11 substitution, end to end through the real ``av`` writer.

    Every other test of this substitution stops at the injected-encoder seam,
    and a fake accepts whatever it is handed -- which is the exact blind spot
    this file exists for. So this one drives ``outputs.write_output`` for real
    and opens the file back up.

    The upload is a 16 kHz tone and the vocoder reports 48 kHz. The track in the
    container is converted to the soundtrack's own format, 48 kHz stereo AAC,
    and the tone must come back at the SAME frequency: muxing 16 kHz samples
    as if they were 48 kHz would triple the pitch with no error anywhere.
    """
    from conftest import STUB_SAMPLING_RATE

    from ltx25_studio import audio as audio_module

    tone_hz = 440.0
    samples = round(121 / 24.0 * STUB_SAMPLING_RATE)
    t = np.arange(samples) / STUB_SAMPLING_RATE
    upload = np.stack([0.5 * np.sin(2 * np.pi * tone_hz * t)] * 2).astype(np.float32)
    monkeypatch.setattr(audio_module, "load_waveform", lambda source: (upload, STUB_SAMPLING_RATE))
    result = run_production("audio_to_video", tmp_path, audio=audio_file())

    with av.open(str(result.video_path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        rate = stream.codec_context.sample_rate
        channels = stream.codec_context.channels
        decoded = np.concatenate([frame.to_ndarray() for frame in container.decode(stream)], 1)

    assert (rate, channels) == (VOCODER_SAMPLING_RATE, AUDIO_CHANNELS), (
        f"the upload was muxed at {rate} Hz with {channels} channel(s); the soundtrack is "
        f"{VOCODER_SAMPLING_RATE} Hz stereo, which is also what verify_streams checks"
    )
    spectrum = np.abs(np.fft.rfft(decoded[0] * np.hanning(decoded.shape[1])))
    measured = float(np.fft.rfftfreq(decoded.shape[1], 1.0 / rate)[np.argmax(spectrum)])
    assert abs(measured - tone_hz) <= 1.0, (
        f"a {tone_hz:.0f} Hz upload came back at {measured:.1f} Hz: the track was resampled "
        f"wrongly or written at the wrong rate, a pitch shift that reports no error"
    )
    assert any("16 kHz" in note and "48 kHz" in note for note in result.warnings), (
        f"the upload's rate was changed and the result does not say so: {result.warnings}"
    )


@pytest.mark.parametrize("mode_id", [m for m in MODE_IDS if modes.MODES[m].exports == ("audio",)])
def test_an_audio_only_mode_writes_a_real_track_at_the_vocoders_rate(mode_id, tmp_path):
    """The audio-only export, demuxed: right rate, right channel count, real samples."""
    result = run_production(mode_id, tmp_path)

    with av.open(str(result.audio_path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        assert stream.codec_context.sample_rate == VOCODER_SAMPLING_RATE, (
            f"mode {mode_id!r} wrote its audio at {stream.codec_context.sample_rate} Hz "
            f"instead of the vocoder's {VOCODER_SAMPLING_RATE} Hz — a pitch-shifted file "
            f"that reports no error."
        )
        assert stream.codec_context.layout.nb_channels == AUDIO_CHANNELS
        assert sum(frame.samples for frame in container.decode(stream)) > 0


# --------------------------------------------------------------------------
# the refusal: better a typed error than a guessed number
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", [m for m in MODE_IDS if "audio" in modes.MODES[m].exports])
def test_an_audio_mode_with_no_vocoder_to_read_refuses_rather_than_assuming(mode_id, tmp_path):
    """PLAN §5.4: an assumed rate resamples every export, silently and forever.

    The failure must be typed and must say what was missing, so the next person
    reads "no vocoder to read output_sampling_rate from" rather than a ``None``
    crashing several layers down inside ``av``.
    """
    with pytest.raises(EncodeError) as caught:
        run_production(mode_id, tmp_path, pipe=VocoderlessPipe())

    message = str(caught.value)
    assert "sampling_rate" in message or "sampling rate" in message, (
        f"the refusal must name the missing rate; got {message!r}"
    )


def test_the_writer_still_refuses_a_frame_rate_it_was_not_given(tmp_path):
    """Pins the refusal itself, so the guard cannot be softened into a default.

    If ``write_output`` ever grows a fallback frame rate, every wrong-speed video
    it then writes becomes undetectable — which is precisely why the number is
    the caller's to supply.
    """
    with pytest.raises(EncodeError) as caught:
        outputs.write_output(
            RealOutput(),
            run_dir=tmp_path / "run-nofps",
            mode="text_to_video",
            exports=("video",),
            seed=1,
            sampling_rate=VOCODER_SAMPLING_RATE,
        )

    assert "frame rate" in str(caught.value)


def test_the_writer_still_refuses_a_sampling_rate_it_was_not_given(tmp_path):
    """The same guard on the audio side."""
    with pytest.raises(EncodeError) as caught:
        outputs.write_output(
            RealOutput(),
            run_dir=tmp_path / "run-norate",
            mode="text_to_audio",
            exports=("audio",),
            seed=1,
        )

    assert "sampling rate" in str(caught.value)


# --------------------------------------------------------------------------
# the dtype the model actually produces
#
# Every fixture above hands the writer `np.float32`. The pipeline hands it
# `torch.bfloat16` -- WEIGHT_DTYPE, documented as not configurable -- and numpy
# has no bfloat16, so `.numpy()` raised and SIX OF ELEVEN MODES generated
# successfully and died at the last step. 1926 tests passed over it, because the
# fixture that made the tests possible encoded the assumption that hid the bug.
# --------------------------------------------------------------------------


class BFloat16Tensor:
    """A tensor whose dtype numpy cannot represent, behaving as torch does.

    Reproduces the exact failure rather than approximating it: `.numpy()` raises
    ``TypeError`` the way torch does for bfloat16, and `.float()` is the only way
    through. `test_the_bfloat16_fake_is_faithful_to_torch` is what stops this
    class quietly drifting into a shape that could not fail.
    """

    def __init__(self, array):
        self._array = array
        self.dtype = "torch.bfloat16"

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        raise TypeError("Got unsupported ScalarType BFloat16")

    def float(self):
        # torch returns a TENSOR, not an ndarray -- `.float().numpy()` is two
        # steps. Returning the array here made the fake unfaithful in exactly
        # the direction that would have let a broken downcast pass.
        return _Float32Tensor(self._array)


class _Float32Tensor:
    """What ``.float()`` gives back: still a tensor, now numpy-representable."""

    def __init__(self, array):
        self._array = array
        self.dtype = "torch.float32"

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def bfloat16_output():
    out = RealOutput()
    out.audio = BFloat16Tensor(out.audio)
    return out


class BFloat16Pipe(VocoderPipe):
    """The pipeline as it really behaves: audio comes back as bfloat16."""

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return bfloat16_output()


@pytest.mark.parametrize("mode_id", [m for m in MODE_IDS if "audio" in m])
def test_an_audio_mode_can_be_written_when_the_audio_is_bfloat16(mode_id, tmp_path):
    """The default precision must reach the encoder, for every audio mode."""
    result = run_production(mode_id, tmp_path, pipe=BFloat16Pipe())

    written = result.audio_path or result.video_path
    assert written is not None and written.exists(), (
        f"{mode_id} produced no artifact from bfloat16 audio"
    )
    assert written.stat().st_size > 0, f"{mode_id} wrote an empty file from bfloat16 audio"


def test_the_downcast_lands_on_float32_not_float16():
    """A silent narrowing in the module whose job is 'never silently wrong'."""
    source = np.zeros((AUDIO_CHANNELS, AUDIO_SAMPLES), dtype=np.float32)

    converted = outputs._as_array(BFloat16Tensor(source))

    assert converted.dtype == np.float32, (
        f"bfloat16 audio was converted to {converted.dtype}, not float32"
    )


def test_a_dtype_numpy_cannot_take_is_the_helpers_problem_not_the_callers():
    """`_as_array` promises 'whatever we were given -> a numpy array'.

    Fixing this at one caller would leave the other latent, which is how a
    shared helper with two callers hid it in the first place.
    """
    converted = outputs._as_array(BFloat16Tensor(np.zeros((2, 8), dtype=np.float32)))

    assert isinstance(converted, np.ndarray), (
        f"_as_array returned {type(converted).__name__}, so the promise it makes is not kept"
    )


def test_the_bfloat16_fake_is_faithful_to_torch():
    """The fixture is itself untested by the tests it makes possible.

    That is the gap this whole section exists to close, so it is closed rather
    than described: against real torch, assert the two properties
    :class:`BFloat16Tensor` claims -- that `.numpy()` raises on bfloat16, and
    that `.float()` yields float32. If either stops holding, the fake above is
    testing a failure mode that no longer exists.
    """
    torch = pytest.importorskip("torch", reason="fidelity check needs real torch")

    real = torch.zeros((AUDIO_CHANNELS, 16), dtype=torch.bfloat16)

    with pytest.raises(TypeError):
        real.numpy()
    assert real.float().numpy().dtype == np.float32

    assert outputs._as_array(real).dtype == np.float32, (
        "a real bfloat16 tensor did not survive _as_array"
    )
