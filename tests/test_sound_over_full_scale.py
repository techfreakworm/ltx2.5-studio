"""Spec: sound over full scale is turned down, never clipped (issue #14, S2).

Three Space runs wrote samples pinned at 0 dBFS: a short shouted line took the
model's float track over full scale, and the 16-bit wav clipped it while the
AAC track in the mp4 kept the over-range floats, so the two disagreed.

Where the model's float sound becomes the written track, a peak above full
scale now turns the WHOLE track down so the peak sits at -0.1 dBFS, and the
run's diagnostics say by how many dB. The mp4 and the wav carry those same
samples. A track within full scale is written exactly as before, and a user's
own recording carried into the file (audio to video) is never touched.
"""

from __future__ import annotations

import math
import wave

import pytest
from conftest import STUB_SAMPLING_RATE
from test_encoder_contract import FRAME_COUNT, RealOutput, VocoderPipe, run_production
from test_media_io import VOCODER_RATE, JointOutput, decode_audio, write_joint

from ltx25_studio import audio as audio_module
from ltx25_studio import outputs

av = pytest.importorskip("av", reason="the media stack is a pinned runtime dependency")
np = pytest.importorskip("numpy")

#: -0.1 dBFS, where a turned-down track's peak sits.
CEILING = 10 ** (-0.1 / 20)
#: The harness counts a sample at or above this fraction of full scale as clipped.
CLIP_LEVEL = 0.999
#: One 16-bit step: what writing the wav's samples may move a value by.
ONE_STEP = 1 / 32768


def loud(rate=VOCODER_RATE, seconds=1.0, left=1.6, right=1.2):
    """A stereo line that swells and fades, over full scale: peaks near 1.6 (+4.1 dBFS) and 1.2.

    Faded in and out because AAC rings on a tone that starts at full level, at
    any level (+3.5 dB on a -12 dBFS one), which would say nothing about this.
    """
    t = np.arange(int(round(seconds * rate))) / rate
    swell = np.hanning(t.size)
    return np.stack(
        [left * swell * np.sin(2 * np.pi * 220 * t), right * swell * np.sin(2 * np.pi * 330 * t)]
    ).astype(np.float32)


def within_full_scale(samples=4800, seed=3):
    """A track inside full scale whose values are exact 16-bit steps, both extremes included."""
    ints = np.random.default_rng(seed).integers(-32768, 32768, size=(2, samples), dtype=np.int32)
    ints[0, 10], ints[1, 20] = 32767, -32768
    return ints.astype(np.int16), (ints / 32768.0).astype(np.float32)


def wav_pcm(path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def peak_db(samples) -> float:
    return 20 * math.log10(float(np.abs(samples).max()))


@pytest.fixture
def encoded(monkeypatch):
    """Every float track handed to an audio encoder, in order: the mp4's, then the wav's."""
    seen = []
    real = outputs._encode_audio

    def spy(container, stream, planes, *, sampling_rate):
        seen.append(np.array(planes, copy=True))
        return real(container, stream, planes, sampling_rate=sampling_rate)

    monkeypatch.setattr(outputs, "_encode_audio", spy)
    return seen


# --------------------------------------------------------------------------
# the writer
# --------------------------------------------------------------------------


def test_a_model_track_over_full_scale_is_turned_down_not_clipped(tmp_path):
    track = loud()
    written = write_joint(tmp_path, mode="text_to_audio_video", model_audio=track)

    _, _, _, samples = decode_audio(written["audio"])

    clipped = int(np.count_nonzero(np.abs(samples) >= CLIP_LEVEL))
    assert clipped == 0, f"{clipped} samples of the wav sit at full scale"
    expected = track * (CEILING / float(np.abs(track).max()))
    worst = float(np.abs(samples - expected).max())
    assert worst <= ONE_STEP + 1e-6, (
        f"the wav is not the track scaled by one factor: off by up to {worst:.6f}"
    )
    assert abs(peak_db(samples) - (-0.1)) < 0.01, f"peak {peak_db(samples):.3f} dBFS"


def test_the_mp4_carries_the_same_turned_down_sound_as_the_wav(tmp_path):
    written = write_joint(tmp_path, mode="text_to_audio_video", model_audio=loud())

    _, _, _, in_mp4 = decode_audio(written["video"])
    _, _, _, in_wav = decode_audio(written["audio"])

    assert abs(peak_db(in_mp4) - peak_db(in_wav)) < 0.3, (
        f"the mp4's sound peaks at {peak_db(in_mp4):+.2f} dBFS and the wav's at "
        f"{peak_db(in_wav):+.2f} dBFS: the two files do not carry the same sound"
    )


def test_both_files_are_written_from_one_turned_down_track(tmp_path, encoded):
    track = loud()
    write_joint(tmp_path, mode="text_to_audio_video", model_audio=track)

    assert len(encoded) == 2, "expected one encode for the mp4 and one for the wav"
    muxed, wav = encoded
    assert np.array_equal(muxed, wav), "the wav is not the samples muxed into the mp4"
    assert float(np.abs(muxed).max()) <= CEILING + 1e-6
    ratio = muxed[track != 0] / track[track != 0]
    assert np.allclose(ratio, CEILING / float(np.abs(track).max()), rtol=1e-5), "not one factor"


def test_an_audio_only_track_over_full_scale_is_turned_down(tmp_path):
    track = loud()
    written = outputs.write_output(
        JointOutput(track),
        run_dir=tmp_path / "run",
        mode="text_to_audio",
        exports=("audio",),
        frame_rate=24,
        sampling_rate=VOCODER_RATE,
    )

    _, _, _, samples = decode_audio(written["audio"])

    assert int(np.count_nonzero(np.abs(samples) >= CLIP_LEVEL)) == 0
    expected = track * (CEILING / float(np.abs(track).max()))
    assert np.abs(samples - expected).max() <= ONE_STEP + 1e-6


def test_the_turn_down_is_a_diagnostics_line_saying_by_how_many_db(tmp_path):
    notes: list[str] = []
    facts: list[str] = []
    outputs.write_output(
        JointOutput(loud()),
        run_dir=tmp_path / "run",
        mode="text_to_audio_video",
        exports=("audio", "video"),
        frame_rate=24,
        sampling_rate=VOCODER_RATE,
        warnings=notes,
        diagnostics=facts,
    )

    # 1.6 is +4.08 dBFS; down to -0.1 dBFS is 4.18 dB.
    assert len(facts) == 1 and "4.2 dB" in facts[0] and "4.1 dB" in facts[0], facts
    assert not any("dB" in note for note in notes), f"a diagnostics fact in the notes: {notes}"


def test_a_track_within_full_scale_is_written_bit_exactly(tmp_path, encoded):
    ints, track = within_full_scale()
    facts: list[str] = []
    written = outputs.write_output(
        JointOutput(track),
        run_dir=tmp_path / "run",
        mode="text_to_audio_video",
        exports=("audio", "video"),
        frame_rate=24,
        sampling_rate=VOCODER_RATE,
        diagnostics=facts,
    )

    assert all(np.array_equal(planes, track) for planes in encoded), (
        "a track within full scale reached the encoders changed"
    )
    assert wav_pcm(written["audio"]) == ints.T.astype("<i2").tobytes(), (
        "the wav's samples are not the track's own 16-bit values"
    )
    assert facts == []


def test_a_carried_recording_is_never_touched(tmp_path, encoded):
    """Audio to video carries the user's own recording, over full scale or not."""
    recording = loud(left=1.3, right=0.4)
    facts: list[str] = []
    outputs.write_output(
        JointOutput(),
        run_dir=tmp_path / "run",
        mode="audio_to_video",
        exports=("audio", "video"),
        frame_rate=24,
        sampling_rate=VOCODER_RATE,
        supplied_audio=outputs.SuppliedAudio(recording, VOCODER_RATE),
        diagnostics=facts,
    )

    assert len(encoded) == 2
    assert all(np.array_equal(planes, recording) for planes in encoded), (
        "the user's recording was changed on its way into the files"
    )
    assert facts == []


# --------------------------------------------------------------------------
# through generate(), with the production writer
# --------------------------------------------------------------------------


class LoudPipe(VocoderPipe):
    """The fake pipeline, with a model track that goes over full scale."""

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        out = RealOutput()
        out.audio = loud(seconds=FRAME_COUNT / 24)
        return out


def _turn_down_lines(result):
    return [line for line in result.record.diagnostics if "full scale" in line]


@pytest.mark.parametrize("mode_id", ["text_to_audio_video", "text_to_audio"])
def test_a_run_whose_sound_went_over_full_scale_says_so_in_its_diagnostics(mode_id, tmp_path):
    result = run_production(mode_id, tmp_path, pipe=LoudPipe())

    _, _, _, samples = decode_audio(result.audio_path)
    assert int(np.count_nonzero(np.abs(samples) >= CLIP_LEVEL)) == 0
    lines = _turn_down_lines(result)
    assert len(lines) == 1 and "4.2 dB" in lines[0], result.record.diagnostics
    assert not any("full scale" in note for note in result.warnings), result.warnings


def test_a_run_within_full_scale_has_no_turn_down_line(tmp_path):
    result = run_production("text_to_audio_video", tmp_path)

    assert _turn_down_lines(result) == [], result.record.diagnostics


def test_a_run_carrying_a_loud_recording_leaves_it_as_it_is(
    tmp_path, audio_input_ready, monkeypatch, encoded
):
    """Audio to video through generate(): the upload, over full scale, is not turned down."""
    recording = loud(rate=STUB_SAMPLING_RATE, seconds=121 / 24.0, left=1.3, right=1.1)
    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (recording, STUB_SAMPLING_RATE)
    )

    result = run_production("audio_to_video", tmp_path)

    conformed, _ = outputs._conformed_supplied_audio(
        outputs.SuppliedAudio(recording, STUB_SAMPLING_RATE),
        sampling_rate=VOCODER_RATE,
        target=tmp_path / "probe",
    )
    assert encoded and all(np.array_equal(planes, conformed) for planes in encoded), (
        "the user's recording was changed on its way into the files"
    )
    assert _turn_down_lines(result) == [], result.record.diagnostics
