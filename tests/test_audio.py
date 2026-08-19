"""Spec for ``ltx25_studio/audio.py`` — PLAN §5.4, §5.7 rows 5 and 11.

§5.4 is the whole rule: **every mel parameter is read from ``audio_vae.config``
at runtime.** Not one of them is written down here. The pipeline itself works
this way — its constructor reads ``self.vae.spatial_compression_ratio``,
``self.audio_vae.mel_compression_ratio`` and friends from the loaded components —
and a hardcoded 16000 or 80 that happens to be right today is a wrong number
waiting for a checkpoint change.

Tested against a fake config object. No torch, no audio file, no weights.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
from conftest import (
    PACKAGE_ROOT,
    FakeAudioVae,
    FakeMel,
    FakeVocoder,
    code_string_constants,
    waveform_of,
)

from ltx25_studio import audio
from ltx25_studio.errors import LTXStudioError

#: The parameter names PLAN §5.4 names explicitly.
CONFIG_FIELDS = (
    "audio_sampling_rate",
    "audio_hop_length",
    "audio_mel_bins",
    "audio_vae_mel_compression_ratio",
    "audio_vae_temporal_compression_ratio",
    "audio_latent_channels",
)


def fake_audio_vae(**overrides):
    """A stand-in shaped like the REAL checkpoint, keyed by our semantic names.

    Overrides use the §5.4 names; the fake puts each value **where the actual
    audio VAE keeps it** -- four on ``.config`` under different names, two as
    instance attributes absent from ``config.json`` entirely.

    Placement is derived from ``audio.PARAM_SOURCES`` so this fake cannot drift
    from the reader. That makes it useless for catching a wrong *mapping*, which
    is precisely the bug that got here -- so
    ``test_the_mapping_matches_the_real_checkpoint`` pins the mapping against
    the recorded config instead. The fake checks semantics; that test checks
    reality.
    """
    values = {
        "audio_sampling_rate": 44100,
        "audio_hop_length": 512,
        "audio_mel_bins": 128,
        "audio_vae_mel_compression_ratio": 4,
        "audio_vae_temporal_compression_ratio": 8,
        "audio_latent_channels": 32,
    }
    values.update(overrides)
    config, attrs = {}, {}
    for name, value in values.items():
        where, field = audio.PARAM_SOURCES[name]
        (config if where == "config" else attrs)[field] = value
    return SimpleNamespace(config=SimpleNamespace(**config), **attrs)


# --------------------------------------------------------------------------
# every parameter comes from the config
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", CONFIG_FIELDS)
def test_every_documented_parameter_is_read_from_the_config(field):
    params = audio.audio_params_from(fake_audio_vae())
    assert hasattr(params, field) or field in params, f"{field} is not surfaced"


def test_the_values_are_the_config_values():
    params = audio.audio_params_from(fake_audio_vae(audio_sampling_rate=48000, audio_mel_bins=80))
    values = params if isinstance(params, dict) else vars(params)
    assert values["audio_sampling_rate"] == 48000
    assert values["audio_mel_bins"] == 80


def test_an_unusual_config_is_honoured_rather_than_overridden():
    """The decisive one: a checkpoint with a 22050 Hz audio VAE must produce
    22050, not the number someone assumed while writing the module."""
    params = audio.audio_params_from(
        fake_audio_vae(audio_sampling_rate=22050, audio_hop_length=256, audio_latent_channels=16)
    )
    values = params if isinstance(params, dict) else vars(params)
    assert values["audio_sampling_rate"] == 22050
    assert values["audio_hop_length"] == 256
    assert values["audio_latent_channels"] == 16


def test_a_missing_config_field_fails_loudly_rather_than_defaulting():
    """A silent fallback to a plausible default is how the audio track ends up
    subtly wrong for a whole release."""
    incomplete = SimpleNamespace(config=SimpleNamespace(audio_sampling_rate=44100))
    with pytest.raises(LTXStudioError):
        audio.audio_params_from(incomplete)


def test_the_error_names_the_missing_field():
    incomplete = SimpleNamespace(config=SimpleNamespace(audio_sampling_rate=44100))
    with pytest.raises(LTXStudioError) as excinfo:
        audio.audio_params_from(incomplete)
    assert "audio_" in str(excinfo.value)


# --------------------------------------------------------------------------
# ...and none of them is written down (PLAN §5.4)
# --------------------------------------------------------------------------

#: Numbers that would mean someone assumed an audio format.
SUSPICIOUS_LITERALS = {8000, 16000, 22050, 24000, 32000, 44100, 48000, 1024}

AUDIO_MODULE = PACKAGE_ROOT / "audio.py"


def test_no_sample_rate_is_hardcoded():
    """Comments and docstrings may cite numbers; code may not use them."""
    tree = ast.parse(AUDIO_MODULE.read_text())
    docstring_ids = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            docstring_ids.add(id(body[0].value))
    found = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and id(node) not in docstring_ids
        and node.value in SUSPICIOUS_LITERALS
    }
    assert not found, (
        f"audio.py hardcodes {sorted(found)}. Every mel parameter comes from "
        "audio_vae.config at runtime (PLAN §5.4)."
    )


def test_audio_module_does_not_name_config_keys_it_should_be_reading():
    """A string literal like ``"audio_sampling_rate"`` is fine — that IS how you
    read it from the config. This guards the opposite: a *value* smuggled in as
    a string."""
    for _, value in code_string_constants(AUDIO_MODULE):
        assert value not in {str(n) for n in SUSPICIOUS_LITERALS}, value


# --------------------------------------------------------------------------
# shape assertions at our boundary (PLAN §5.7 row 5)
# --------------------------------------------------------------------------


def test_four_dimensional_audio_latents_are_accepted():
    """``audio_latents`` is UNPACKED 4D ``[B, C, L, M]``."""
    assert audio.validate_audio_latents_shape((1, 32, 64, 8)) == (1, 32, 64, 8)


@pytest.mark.parametrize("shape", [(32, 64, 8), (1, 32, 64, 8, 2), (64,), ()])
def test_wrongly_shaped_audio_latents_raise(shape):
    """ "shape errors, or worse, **silent misinterpretation**" — so we assert at
    our own boundary rather than hoping upstream notices."""
    with pytest.raises(LTXStudioError):
        audio.validate_audio_latents_shape(shape)


def test_the_shape_error_names_the_shape_it_got():
    with pytest.raises(LTXStudioError) as excinfo:
        audio.validate_audio_latents_shape((1, 32, 64))
    assert "4" in str(excinfo.value)


# --------------------------------------------------------------------------
# the channel count the VAE actually requires
#
# The audio VAE is stereo and does not adapt. Handed one channel it reaches
# `conv_in` and raises `Given groups=1, weight of size [128, 2, 3, 3], expected
# input[1, 1, 4, 480002] to have 2 channels, but got 1` -- four frames inside
# someone else's conv layer, naming a weight shape rather than anything the user
# did. Most people's wav files are mono, so this is the single most likely way
# the audio-input feature meets a real user.
# --------------------------------------------------------------------------


def test_a_mono_upload_is_widened_rather_than_rejected():
    """Duplicating one channel adds no information and loses none."""
    widened, note = audio.match_channel_count(waveform_of(1), channels=2)

    assert widened.shape[0] == 2, f"mono was not widened: {widened.shape}"
    assert note, "the audio was altered and the user was not told"


def test_the_mono_upmix_is_reported_in_words_a_user_can_act_on():
    _widened, note = audio.match_channel_count(waveform_of(1), channels=2)

    assert "mono" in note and "duplicated" in note, note
    assert "not true stereo" in note, f"the note does not say what the user actually got: {note}"


def test_stereo_passes_through_untouched_and_unremarked():
    """A note on every run is a note nobody reads."""
    source = waveform_of(2)

    passed, note = audio.match_channel_count(source, channels=2)

    assert passed is source, "a conforming waveform was rebuilt for no reason"
    assert note is None, f"nothing changed and a warning was raised anyway: {note}"


def test_more_channels_than_the_vae_takes_is_refused_not_downmixed():
    """That direction loses information; PLAN 5.4 forbids doing it silently."""
    with pytest.raises(LTXStudioError) as excinfo:
        audio.match_channel_count(waveform_of(6), channels=2)

    message = str(excinfo.value)
    assert "6 channels" in message, f"the refusal does not name what was supplied: {message}"
    assert "exactly 2" in message, f"the refusal does not name what is required: {message}"


def test_the_required_count_is_read_from_the_vae_not_hardcoded():
    """A checkpoint with a different layout is handled by reading, not editing."""
    widened, _note = audio.match_channel_count(waveform_of(1), channels=4)

    assert widened.shape[0] == 4, (
        f"the channel count was not taken from the caller: {widened.shape}"
    )


def test_a_mono_upload_reaches_the_vae_as_stereo():
    """The whole point: the conv layer must never see one channel.

    `match_channel_count` is called inside `encode_waveform_to_latents` rather
    than at one call site, so every caller reaches the conv layer conforming.
    """
    vae = FakeAudioVae()
    notes: list[str] = []

    audio.encode_waveform_to_latents(
        waveform_of(1), audio_vae=vae, sampling_rate=vae.config.sample_rate, notes=notes
    )

    assert vae.mels[-1].shape[0] == 2, (
        f"the audio VAE was handed {vae.mels[-1].shape[0]} channel(s); it takes 2 and "
        f"raises a raw conv2d shape error rather than anything actionable"
    )
    assert notes, "the audio was altered and nothing was recorded for the user"


# --------------------------------------------------------------------------
# where the six parameters really live
# --------------------------------------------------------------------------

#: The exact key set of ``audio_vae/config.json`` at revision 871165de…,
#: recorded so the mapping can be checked without the weights on disk.
#: *Read off the cached checkpoint, not inferred.*
REAL_AUDIO_VAE_CONFIG_KEYS = frozenset(
    {
        "_class_name",
        "_diffusers_version",
        "attn_resolutions",
        "base_channels",
        "causality_axis",
        "ch_mult",
        "double_z",
        "dropout",
        "in_channels",
        "is_causal",
        "latent_channels",
        "mel_bins",
        "mel_hop_length",
        "mid_block_add_attention",
        "norm_type",
        "num_res_blocks",
        "output_channels",
        "resolution",
        "sample_rate",
    }
)


@pytest.mark.parametrize(
    "name", [n for n, (where, _f) in audio.PARAM_SOURCES.items() if where == "config"]
)
def test_the_mapping_matches_the_real_checkpoint(name):
    """The check the fake cannot make, because the fake is built from the mapping.

    Every config-sourced parameter must name a key the real ``config.json``
    actually has. All six used to be read off ``.config`` by our own semantic
    names, none of which exist there -- so the reader raised on real weights
    while every offline test passed against a fake that shared the mistake.
    """
    _where, field = audio.PARAM_SOURCES[name]

    assert field in REAL_AUDIO_VAE_CONFIG_KEYS, (
        f"{name!r} is read from audio_vae.config[{field!r}], and the real checkpoint has no "
        f"such key. It would raise on real weights and pass every offline test."
    )


def test_our_own_names_are_not_checkpoint_keys():
    """The specific mistake, pinned so it cannot come back as a 'simplification'."""
    overlap = REAL_AUDIO_VAE_CONFIG_KEYS & set(audio.PARAM_SOURCES)

    assert not overlap, (
        f"{sorted(overlap)} are both our semantic names and real config keys; if that is now "
        f"true the mapping should say so explicitly rather than coincide"
    )


def test_the_input_rate_is_not_the_output_rate():
    """The audio VAE takes 16000; the vocoder emits 48000. Not symmetric.

    Assuming symmetry would resample every upload to the wrong rate -- audible
    as a pitch shift, reported as nothing.
    """
    vae = FakeAudioVae()

    assert audio.audio_params_from(vae).audio_sampling_rate == 16000
    assert (
        audio.export_sampling_rate_from(
            SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000))
        )
        == 48000
    )


def test_a_file_at_another_rate_is_resampled_to_the_vaes(monkeypatch):
    """The rate that wins is the VAE's, and the conversion really happens."""
    seen = {}

    def fake_resample(waveform, *, from_rate, to_rate):
        seen.update(from_rate=from_rate, to_rate=to_rate)
        return waveform

    monkeypatch.setattr(audio, "_resampled", fake_resample)
    vae = FakeAudioVae()

    audio.encode_waveform_to_latents(waveform_of(2), audio_vae=vae, sampling_rate=44100)

    assert seen == {"from_rate": 44100, "to_rate": 16000}, (
        f"the upload was not resampled to the VAE's own rate: {seen}"
    )


# --------------------------------------------------------------------------
# decoding the file the user uploaded
#
# `torchaudio.load` is unreachable at our pinned torchaudio 2.11.0: it
# dispatches to TorchCodec, which needs SYSTEM FFmpeg shared libraries that the
# host does not have and cannot be given. `av` is already non-optional and
# bundles its own FFmpeg, which is why it works where torchcodec cannot.
# --------------------------------------------------------------------------

av = pytest.importorskip("av", reason="the media stack is a pinned runtime dependency")
np = pytest.importorskip("numpy")


def write_stereo_wav(path, *, left, right, rate=16000):
    """A real wav whose two channels are DIFFERENT.

    Identical channels would satisfy a shape assertion while proving nothing
    about de-interleaving -- the same fixture trap as a float32 fixture over a
    bfloat16 bug. Here the left channel ramps and the right is constant, so any
    mix-up between them is visible in the values.
    """
    planes = np.stack([left, right])
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=rate)
        stream.layout = av.AudioLayout("2c")
        stream.codec_context.open()
        frame = av.AudioFrame.from_ndarray(planes, format="fltp", layout="2c")
        frame.sample_rate = rate
        resampler = av.AudioResampler(
            format=stream.format.name, layout=stream.layout.name, rate=stream.rate
        )
        fifo = av.audio.fifo.AudioFifo()
        for resampled in resampler.resample(frame):
            fifo.write(resampled)
        while True:
            chunk = fifo.read()
            if chunk is None:
                break
            for packet in stream.encode(chunk):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def decoded_frames(path):
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        channels = int(getattr(stream.layout, "nb_channels", 0) or stream.channels)
        planar = bool(stream.format.is_planar)
        return [f.to_ndarray() for f in container.decode(stream)], channels, planar


def test_the_fixture_really_is_a_packed_format(tmp_path):
    """Otherwise the test below exercises the branch that was never the problem."""
    path = write_stereo_wav(
        tmp_path / "probe.wav",
        left=np.linspace(0.1, 0.9, 256, dtype=np.float32),
        right=np.full(256, -0.5, dtype=np.float32),
    )
    chunks, channels, planar = decoded_frames(path)

    assert not planar, "pcm_s16le is packed; a planar fixture would not test de-interleaving"
    assert channels == 2
    assert chunks[0].shape[0] == 1, (
        f"a packed frame should arrive as one interleaved row, got {chunks[0].shape}"
    )


def test_the_channels_come_back_separated_not_split_down_the_middle(tmp_path):
    """``reshape(channels, -1)`` is the bug: it splits the file in half.

    With a ramping left channel and a constant right one, that mutation gives
    two channels that both vary -- so the values catch it where a shape check
    could not.
    """
    samples = 512
    left = np.linspace(0.1, 0.9, samples, dtype=np.float32)
    right = np.full(samples, -0.5, dtype=np.float32)
    path = write_stereo_wav(tmp_path / "stereo.wav", left=left, right=right)

    chunks, channels, planar = decoded_frames(path)
    data = audio.deinterleaved(chunks, channels=channels, planar=planar)

    assert data.shape == (2, samples), f"not (channels, samples): {data.shape}"
    recovered_left = data[0].astype(np.float32) / 32768.0
    recovered_right = data[1].astype(np.float32) / 32768.0
    assert recovered_right.std() < 1e-3, (
        f"the right channel should be constant and varies by {recovered_right.std():.4f} -- "
        f"the channels were split down the middle rather than de-interleaved"
    )
    assert recovered_left[-1] > recovered_left[0] + 0.5, (
        f"the left channel should ramp across the whole file: "
        f"{recovered_left[0]:.3f} -> {recovered_left[-1]:.3f}"
    )


def test_integer_pcm_is_scaled_into_the_range_the_mel_expects():
    """int16 handed over unscaled is a signal four orders of magnitude too loud."""
    loudest = np.array([[32767, -32768]], dtype=np.int16)

    scaled = audio._as_float_samples(loudest)

    assert scaled.dtype == np.float32
    assert abs(scaled).max() <= 1.0, f"not normalised: {scaled}"


def test_a_planar_file_is_concatenated_rather_than_reshaped():
    """The other branch, which must not be de-interleaved."""
    chunks = [np.array([[1.0, 2.0], [9.0, 8.0]]), np.array([[3.0], [7.0]])]

    data = audio.deinterleaved(chunks, channels=2, planar=True)

    assert data.tolist() == [[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]], data


# --------------------------------------------------------------------------
# the mel front-end the checkpoint does not ship
#
# No forward transform exists anywhere in diffusers.pipelines.ltx2 -- the
# vocoder's only `hann` window belongs to its anti-aliasing filter. So the
# forward direction has to be built, from parameters that live across TWO
# components. `decode_latents_to_waveform` needs none of this: the VAE decoder
# emits a mel and the vocoder inverts it. That asymmetry is why H5-H9 produce
# real audio while the audio-INPUT modes cannot start.
# --------------------------------------------------------------------------

#: The exact key set of ``vocoder/config.json`` at revision 871165de….
#: *Read off the cached checkpoint, not inferred.*
REAL_VOCODER_CONFIG_KEYS = frozenset(
    {
        "_class_name",
        "_diffusers_version",
        "act_fn",
        "antialias",
        "antialias_kernel_size",
        "antialias_ratio",
        "bwe_act_fn",
        "bwe_antialias",
        "bwe_antialias_kernel_size",
        "bwe_antialias_ratio",
        "bwe_final_act_fn",
        "bwe_final_bias",
        "bwe_hidden_channels",
        "bwe_in_channels",
        "bwe_leaky_relu_negative_slope",
        "bwe_out_channels",
        "bwe_resnet_dilations",
        "bwe_resnet_kernel_sizes",
        "bwe_upsample_factors",
        "bwe_upsample_kernel_sizes",
        "filter_length",
        "final_act_fn",
        "final_bias",
        "hidden_channels",
        "hop_length",
        "in_channels",
        "input_sampling_rate",
        "leaky_relu_negative_slope",
        "num_mel_channels",
        "out_channels",
        "output_sampling_rate",
        "resnet_dilations",
        "resnet_kernel_sizes",
        "upsample_factors",
        "upsample_kernel_sizes",
        "window_length",
    }
)

REAL_KEYS_BY_OWNER = {
    "audio_vae": REAL_AUDIO_VAE_CONFIG_KEYS,
    "vocoder": REAL_VOCODER_CONFIG_KEYS,
}


@pytest.mark.parametrize("name", sorted(audio.MEL_PARAM_SOURCES))
def test_every_mel_parameter_names_a_key_the_checkpoint_really_has(name):
    """The check a fake cannot make, because fakes are built from the mapping."""
    owner, field = audio.MEL_PARAM_SOURCES[name]

    assert field in REAL_KEYS_BY_OWNER[owner], (
        f"{name!r} is read from {owner}.config[{field!r}] and the real checkpoint has no "
        f"such key -- it would raise on real weights and pass every offline test"
    )


def test_the_window_is_read_across_and_the_hop_is_not():
    """The asymmetry IS the judgement, and it is the thing someone will tidy.

    ``audio_vae.mel_hop_length`` is 160 and ``vocoder.hop_length`` is 80. The
    vocoder's own ``upsample_factors`` multiply to 160, matching the VAE — so
    its 80 means something else inside the vocoder. Reading the hop across for
    consistency would halve it.
    """
    assert audio.MEL_PARAM_SOURCES["win_length"][0] == "vocoder"
    assert audio.MEL_PARAM_SOURCES["n_fft"][0] == "vocoder"
    assert audio.MEL_PARAM_SOURCES["hop_length"] == ("audio_vae", "mel_hop_length"), (
        f"the hop is {audio.MEL_PARAM_SOURCES['hop_length']}. Taking it from the vocoder "
        f"gives 80 against the VAE's 160, and the vocoder's own upsample factors say 160."
    )


def test_the_vocoders_upsample_factors_corroborate_the_vaes_hop():
    """Why 160 wins: the vocoder's own arithmetic agrees with the VAE, not itself."""
    vocoder = FakeVocoder()
    product = 1
    for factor in vocoder.config.upsample_factors:
        product *= factor

    assert product == 160
    assert product != vocoder.config.hop_length, (
        "the vocoder's declared hop now equals its upsample product; if that is really "
        "true the comment in MEL_PARAM_SOURCES needs revisiting"
    )


def test_the_two_components_agree_on_everything_except_the_hop():
    """Four of five parameters are corroborated, which is what makes this a lookup."""
    vae, vocoder = FakeAudioVae(), FakeVocoder()

    assert vae.config.mel_bins == vocoder.config.num_mel_channels
    assert vae.config.sample_rate == vocoder.config.input_sampling_rate
    assert vae.config.mel_hop_length != vocoder.config.hop_length


def test_the_declared_parameters_are_read_not_defaulted():
    params = audio.mel_parameters(FakeAudioVae(), FakeVocoder())

    assert params == {
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 160,
        "n_mels": 64,
        "sample_rate": 16000,
    }, params


def test_building_without_a_vocoder_refuses_rather_than_guessing():
    """A guessed FFT size produces a mel the VAE was never trained on."""
    with pytest.raises(LTXStudioError) as excinfo:
        audio.mel_parameters(FakeAudioVae(), None)

    assert "vocoder" in str(excinfo.value), excinfo.value


def test_the_mel_is_cast_to_the_dtype_the_vae_actually_uses():
    """A dtype boundary has two ends, and only one of them had crashed.

    ``outputs._as_array`` fixes the way out -- bf16 into numpy. This is the way
    in: a float32 mel reaching a bf16 ``conv_in`` raises *"Input type (float)
    and bias type (c10::BFloat16) should be the same"* from four frames inside
    diffusers.
    """
    vae = FakeAudioVae()

    audio.encode_waveform_to_latents(
        waveform_of(2), audio_vae=vae, sampling_rate=vae.config.sample_rate
    )

    assert vae.encoded[-1].dtype == vae.dtype, (
        f"the VAE was handed a {vae.encoded[-1].dtype} mel and its weights are {vae.dtype}; "
        f"conv_in raises rather than casting"
    )


def test_the_dtype_is_read_from_the_component_not_hardcoded():
    """WEIGHT_DTYPE is right today and wrong the moment a quantised load lands."""

    class Int8Vae(FakeAudioVae):
        dtype = "int8"

    vae = Int8Vae()

    audio.encode_waveform_to_latents(
        waveform_of(2), audio_vae=vae, sampling_rate=vae.config.sample_rate
    )

    assert vae.encoded[-1].dtype == "int8", (
        f"the cast targeted {vae.encoded[-1].dtype!r} rather than the VAE's own dtype, so it "
        f"is reading a constant somewhere instead of the loaded component"
    )


def test_an_explicit_to_mel_is_protected_too():
    """A caller supplying `to_mel=` does not know the VAE's dtype."""
    vae = FakeAudioVae()

    audio.encode_waveform_to_latents(
        waveform_of(2),
        audio_vae=vae,
        sampling_rate=vae.config.sample_rate,
        to_mel=lambda waveform: FakeMel("float32"),
    )

    assert vae.encoded[-1].dtype == vae.dtype, (
        "an explicitly-passed to_mel reached the VAE uncast, so the guard lives at the "
        "wrong end of the hand-off"
    )


# --------------------------------------------------------------------------
# the layout audio_vae.encode requires
#
# Upstream, autoencoder_kl_ltx2_audio.py:445, in its own words:
#   # hidden_states expected shape: (batch_size, channels, time, num_mel_bins)
# torchaudio returns [channels, mel, time] -- unbatched AND with the last two
# axes reversed. Both must be corrected, and the transpose is the one that
# hides: the batch fix alone yields a 4D tensor that passes every check.
# --------------------------------------------------------------------------


class ShapedTensor:
    """Enough tensor to follow a rearrangement, with no torch installed.

    The offline gate has no torch, so a test of the real transform can only
    skip -- and a skipped test of the axis order is how both half-fixes went
    unnoticed. This models exactly the two operations the layout step uses.
    """

    def __init__(self, shape):
        self.shape = tuple(shape)

    def transpose(self, a, b):
        axes = list(self.shape)
        axes[a], axes[b] = axes[b], axes[a]
        return ShapedTensor(axes)

    def unsqueeze(self, axis):
        axes = list(self.shape)
        axes.insert(axis, 1)
        return ShapedTensor(axes)


def test_the_mel_reaches_the_encoder_batched_and_time_major():
    """torchaudio gives [channels, mel, time]; the encoder wants [B, C, time, mel]."""
    laid_out = audio.encoder_layout(ShapedTensor((2, 64, 401)))

    assert laid_out.shape == (1, 2, 401, 64), (
        f"got {laid_out.shape}, expected (1, 2, 401, 64) = {audio.MEL_INPUT_DIMS}. "
        f"(1, 2, 64, 401) is the batch-only half-fix: 4D, passes every rank check, and is "
        f"a transposed spectrogram the encoder reads as 64 time steps of 401 bins."
    )


def test_the_real_transform_agrees_with_the_layout_model():
    """The fixture above is only worth its weight if it matches real torch."""
    torch = pytest.importorskip("torch", reason="fidelity check needs real torch")

    to_mel = audio.build_mel_front_end(FakeAudioVae(), FakeVocoder())
    mel = to_mel(torch.zeros((2, 16000), dtype=torch.float32))

    batch, channels, time, mel_bins = mel.shape
    assert (batch, channels, mel_bins) == (1, 2, 64), tuple(mel.shape)
    assert time > mel_bins, (
        f"time ({time}) should exceed mel bins ({mel_bins}) at hop 160; if not, the axes "
        f"are reversed and ShapedTensor is modelling the wrong rearrangement"
    )


def test_our_latent_layout_agrees_with_the_encoder_input_layout():
    """Both are (…, time/frames, mel). The validator never needed loosening.

    Pinned because the instinct on a shape mismatch is to relax the check that
    reported it, and the check was right -- upstream reads the frame count from
    the same axis: ``_, _, audio_num_frames, _ = audio_latents.shape``.
    """
    assert audio.MEL_INPUT_DIMS[-2:] == ("time", "mel")
    assert audio.AUDIO_LATENT_DIMS[-2:] == ("frames", "mel")


def test_a_three_dimensional_mel_is_refused_where_the_mistake_is():
    """conv2d reads a 3D input as unbatched, so the failure surfaces four steps late."""

    class Flat:
        dtype = "bfloat16"
        shape = (2, 64, 401)

        def to(self, dtype):
            return self

    with pytest.raises(LTXStudioError) as excinfo:
        audio.encode_waveform_to_latents(
            waveform_of(2),
            audio_vae=FakeAudioVae(),
            sampling_rate=16000,
            to_mel=lambda waveform: Flat(),
        )

    message = str(excinfo.value)
    assert "3D" in message and "time" in message, message
