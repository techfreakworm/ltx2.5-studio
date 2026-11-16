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
    ShapedTensor,
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
        # `isinstance(body, list)` because an IfExp's `.body` is a single
        # expression, not a block -- a ternary anywhere in the module used to
        # crash this scan with "'Call' object is not subscriptable".
        if not isinstance(body, list):
            continue
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

    assert vae.mels[-1].shape[-2] == 2, (
        f"the audio VAE was handed {vae.mels[-1].shape[-2]} channel(s); it takes 2 and "
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


def test_the_mel_reaches_the_encoder_time_major():
    """A batched waveform gives [B, C, mel, time]; the encoder wants [B, C, time, mel].

    `(1, 2, 64, 401)` is the shape that hides: it is 4D, passes every rank check
    downstream, and is a transposed spectrogram the encoder reads as 64 time
    steps of 401 mel bins.
    """
    laid_out = audio.encoder_layout(ShapedTensor((1, 2, 64, 401)))

    assert laid_out.shape == (1, 2, 401, 64), (
        f"got {laid_out.shape}, expected (1, 2, 401, 64) = {audio.MEL_INPUT_DIMS}"
    )


def test_the_waveform_is_batched_only_after_the_channel_count_is_settled():
    """The two guards contradict each other in any other order.

    `match_channel_count` reads channels off axis 0 and the encoder needs 4D.
    Batching first makes a stereo clip look mono; not batching leaves the
    latents one rank short.
    """
    batched = audio.batched(ShapedTensor((2, 4096)))

    assert batched.shape == (1, 2, 4096), batched.shape
    assert audio.batched(batched).shape == (1, 2, 4096), (
        "an already-batched waveform was batched again"
    )


def test_a_batched_waveform_is_refused_by_the_channel_check():
    """It used to read the batch axis as the channel count and 'widen' stereo."""
    with pytest.raises(LTXStudioError) as excinfo:
        audio.match_channel_count(ShapedTensor((1, 2, 64000)), channels=2)

    message = str(excinfo.value)
    assert "3D" in message and "UNBATCHED" in message, message


def test_the_latents_come_back_frames_major_from_one_rearrangement():
    """The mel's layout decides the latents' layout, so fixing the mel is the fix.

    ``audio_vae.encode`` preserves axis order. Fed a time-major mel it returns
    ``[B, C, L, M]`` directly -- no second transpose, and a second transpose put
    the frame axis back where it started.
    """
    vae = FakeAudioVae()
    rate = vae.config.sample_rate

    # A real clip, not the short default fixture: at 4096 samples there are
    # genuinely fewer frames than mel bins, so the invariant would not hold and
    # the test would be measuring the fixture rather than the code.
    latents = audio.encode_waveform_to_latents(
        waveform_of(2, rate * 4), audio_vae=vae, sampling_rate=rate
    )

    frames = audio.audio_frames_from_latents(latents.shape)
    mel_axis = latents.shape[-1]
    assert frames > mel_axis, (
        f"the frame count came back as {frames} against {mel_axis} on the mel axis. "
        f"For any real clip there are far more frames than mel bins, so this is the "
        f"signature of a transposed latent -- most likely two rearrangements where "
        f"one was needed."
    )


@pytest.mark.parametrize("seconds,expected", [(1.0, 25), (2.0, 50), (4.0, 100)])
def test_the_frame_count_tracks_the_input_duration(seconds, expected):
    """The invariant that catches an axis-order error in EITHER direction.

    Stronger than a shape check, which passes on a transposed tensor of the
    right rank. A doubled transpose pinned every reconstruction to 0.61 s
    regardless of input, and **the fixed length was the tell** -- the mel axis
    does not move when the clip gets longer.
    """
    vae = FakeAudioVae()
    rate = vae.config.sample_rate

    latents = audio.encode_waveform_to_latents(
        waveform_of(2, int(rate * seconds)), audio_vae=vae, sampling_rate=rate
    )

    frames = audio.audio_frames_from_latents(latents.shape)
    assert frames == pytest.approx(expected, abs=2), (
        f"a {seconds}s clip gave {frames} frames, expected about {expected}. A frame "
        f"count that does not scale with duration means the axis being read is the mel "
        f"axis, which is fixed at mel_bins / mel_compression_ratio."
    )


def test_the_real_transform_agrees_with_the_layout_model():
    """The fixture above is only worth its weight if it matches real torch.

    Fed **the way production feeds it**: ``encode_waveform_to_latents`` settles
    the channel count, ``batched`` adds the batch axis, and only then does the
    front-end run. This test used to hand the transform a bare ``(2, 16000)``
    and unpack four axes from the 3D result, so it could never pass -- and it
    never ran, because every recorded gate had no torch. With torch 2.11 the
    batched input gives ``(1, 2, 101, 64)``: time-major, and 16000 // 160 + 1
    frames, which is the count ``FakeAudioVae.mel_spectrogram`` derives.
    """
    torch = pytest.importorskip("torch", reason="fidelity check needs real torch")

    vae = FakeAudioVae()
    to_mel = audio.build_mel_front_end(vae, FakeVocoder())
    waveform, _note = audio.match_channel_count(
        torch.zeros((2, 16000), dtype=torch.float32), channels=audio.audio_channels_from(vae)
    )
    prepared = audio.batched(waveform)
    mel = to_mel(prepared)

    assert mel.ndim == len(audio.MEL_INPUT_DIMS), tuple(mel.shape)
    batch, channels, time, mel_bins = mel.shape
    assert (batch, channels, mel_bins) == (1, 2, 64), tuple(mel.shape)
    assert time > mel_bins, (
        f"time ({time}) should exceed mel bins ({mel_bins}) at hop 160; if not, the axes "
        f"are reversed and ShapedTensor is modelling the wrong rearrangement"
    )
    assert tuple(mel.shape) == vae.mel_spectrogram(prepared).shape, (
        f"the real transform gave {tuple(mel.shape)} and the fake models "
        f"{vae.mel_spectrogram(prepared).shape}; every offline encode test measures the fake"
    )

    # What `batched` exists to prevent: the transform keeps its input's rank, so
    # an unbatched waveform yields a 3D mel -- which conv2d would read as
    # unbatched, and which the rank guard refuses before it gets there.
    unbatched = to_mel(waveform)
    assert tuple(unbatched.shape) == (channels, time, mel_bins), tuple(unbatched.shape)
    with pytest.raises(LTXStudioError):
        audio._validate_mel_rank(unbatched)


def test_the_fake_distribution_takes_what_the_real_one_takes():
    """``FakeLatentDist.sample`` must accept exactly what upstream's does.

    ``sample=lambda: latents`` took nothing while the real
    ``DiagonalGaussianDistribution.sample`` takes a generator, and the
    mismatch hid until torch was installed: then every audio-input test
    raised ``TypeError`` inside the fixture. Pinned against the installed
    diffusers so the fake cannot drift from it again.
    """
    import inspect

    from conftest import FakeLatentDist

    vae_module = pytest.importorskip(
        "diffusers.models.autoencoders.vae", reason="needs the installed diffusers"
    )

    def shape_of(method):
        return [(p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()]

    real = shape_of(vae_module.DiagonalGaussianDistribution.sample)
    fake = shape_of(FakeLatentDist.sample)
    assert fake == real, f"the fake samples as {fake}; diffusers samples as {real}"


def test_a_sample_only_encoder_draws_with_the_generator_the_pipeline_gets(
    monkeypatch, tmp_path, audio_input_ready
):
    """The fallback draw is seeded with the pipeline's own generator.

    A real ``DiagonalGaussianDistribution`` has a ``mode()`` and the encode
    takes it, so no draw happens at all (``test_media_io``). ``FakeLatentDist``
    has only ``sample``, which is the fallback for an encoder without a mode:
    ``generate._prepared_audio`` hands it the SAME generator the pipeline
    receives, so a same-seed rerun still draws the same audio latents.

    A stand-in ``torch`` supplies the generator, so this asserts the same thing
    with or without torch installed: without one, ``_seeded_generator`` returns
    None and the identity would hold trivially.
    """
    import sys
    import types
    from pathlib import Path

    from conftest import audio_file

    from ltx25_studio import generate as generate_module
    from ltx25_studio import outputs

    class Generator:
        def __init__(self, device=None):
            self.device = device

        def manual_seed(self, value):
            self.seed = value
            return self

    fake_torch = types.ModuleType("torch")
    fake_torch.Generator = Generator
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Pipe:
        def __init__(self):
            self.audio_vae = FakeAudioVae()
            self.calls: list[dict] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(frames=("frame",), audio=("sample",))

    def encoder(output, *, run_dir, exports, **kwargs):
        target = Path(run_dir) / outputs.AUDIO_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00" * 2048)
        return target

    pipe = Pipe()
    generate_module.generate(
        mode="audio_to_audio",
        seed=1234,
        audio=audio_file(),
        pipeline_factory=lambda *args, **kwargs: pipe,
        encoder=encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    handed = pipe.calls[-1].get("generator")
    assert isinstance(handed, Generator) and handed.seed == 1234, (
        f"the pipeline was not handed the run's seeded generator: {handed!r}"
    )
    drawn = pipe.audio_vae.sample_generators
    assert len(drawn) == 1 and drawn[0] is handed, (
        f"the audio latents were drawn with {drawn!r}, not the generator the pipeline got "
        f"({handed!r}), so two same-seed runs encode the same upload differently"
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
        device = "cuda"
        shape = (2, 64, 401)

        def to(self, **kwargs):
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


def test_the_mel_is_moved_to_the_device_the_weights_are_on():
    """The second half of the same hand-off, and the fourth attribute of it.

    ``load_waveform`` returns a CPU tensor and the mel is computed on the
    waveform's device, while the VAE's weights are on the accelerator:

        Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type)
        should be the same
    """
    vae = FakeAudioVae()

    audio.encode_waveform_to_latents(
        waveform_of(2), audio_vae=vae, sampling_rate=vae.config.sample_rate
    )

    handed = vae.encoded[-1]
    assert handed.device == vae.device, (
        f"the VAE was handed a mel on {handed.device!r} and its weights are on "
        f"{vae.device!r}; conv_in raises rather than moving it"
    )
    assert handed.dtype == vae.dtype, "the dtype half regressed while the device half landed"


def test_the_device_is_read_from_the_component_not_hardcoded():
    """'cuda' is right until offload or a second device, exactly as bfloat16 is."""

    class SecondDevice(FakeAudioVae):
        device = "cuda:1"

    vae = SecondDevice()

    audio.encode_waveform_to_latents(
        waveform_of(2), audio_vae=vae, sampling_rate=vae.config.sample_rate
    )

    assert vae.encoded[-1].device == "cuda:1", (
        f"the mel went to {vae.encoded[-1].device!r} rather than the component's own "
        f"device, so a constant is being read somewhere"
    )


def test_a_tensor_already_matching_is_not_moved():
    """No-op rather than a needless copy of a mel that is already correct."""
    from ltx25_studio.audio import _matched_to_module

    already = FakeMel("bfloat16", device="cuda")

    assert _matched_to_module(already, FakeAudioVae(), name="audio_vae") is already


# --------------------------------------------------------------------------
# upload-length reconciliation -- TRIM long, REFUSE short, never pad
# --------------------------------------------------------------------------
#
# Upstream reconciles nothing: supplied audio_latents wholesale REPLACE its
# duration-derived length (audio_num_frames is re-read from the tensor), and
# its fresh-noise path uses round(duration_s * rate / hop / temporal_ratio)
# -- pipeline_ltx2_condition.py at the pinned ac56fa2f. So the trim below is
# the only reconciliation anywhere, and it must land on exactly the latent
# count the model would have generated fresh.
#
# The asymmetry is the policy: trimming discards excess the user can see and
# undo; padding fabricates input they never gave; silent alteration is worse
# than refusal.

_RECON_PARAMS = audio.AudioParams(
    audio_sampling_rate=16000,
    audio_hop_length=160,
    audio_mel_bins=64,
    audio_vae_mel_compression_ratio=4,
    audio_vae_temporal_compression_ratio=4,
    audio_latent_channels=8,
)
# 160 * 4 / 16000 = 0.04 s per latent frame; 121 frames @ 25 fps = 4.84 s of
# video = exactly 121 latent frames of audio.


def _recon(waveform, *, sampling_rate=16000, num_frames=121, frame_rate=25.0):
    return audio.reconcile_waveform_with_video(
        waveform,
        sampling_rate=sampling_rate,
        params=_RECON_PARAMS,
        num_frames=num_frames,
        frame_rate=frame_rate,
    )


def test_a_matching_upload_passes_through_untouched():
    wave = waveform_of(2, samples=77440)  # 4.84 s at 16 kHz, exactly the video
    out, note = _recon(wave)
    assert out is wave, "a matching upload must not be copied, resampled or resliced"
    assert note is None


def test_a_mismatch_inside_the_tolerance_is_left_alone():
    wave = waveform_of(2, samples=77440 + 640)  # 0.04 s long -- under the 0.05 s band
    out, note = _recon(wave)
    assert out is wave
    assert note is None


def test_a_long_upload_is_trimmed_to_upstreams_own_latent_grid():
    wave = waveform_of(2, samples=96000)  # 6.0 s against a 4.84 s video
    out, note = _recon(wave)
    assert out.shape[-1] == 77440, "round(4.84 s * 16 kHz) -- whole latent frames, no partial hop"
    assert out.shape[-1] % _RECON_PARAMS.samples_per_latent_frame == 0
    assert note == "audio input trimmed from 6.00s to 4.84s to match 121 frames"


def test_the_trim_converts_to_the_incoming_files_own_rate():
    """The grid is defined at the VAE rate; the file arrives at its own. The
    kept SECONDS are grid-aligned and the sample count follows the file."""
    wave = waveform_of(2, samples=288000)  # 6.0 s at 48 kHz
    out, note = _recon(wave, sampling_rate=48000)
    assert out.shape[-1] == round(4.84 * 48000)
    assert note == "audio input trimmed from 6.00s to 4.84s to match 121 frames"


def test_a_short_upload_is_refused_naming_the_change_that_would_fit():
    from ltx25_studio.errors import AudioVideoDesyncError

    wave = waveform_of(2, samples=48000)  # 3.0 s against a 4.84 s video
    with pytest.raises(AudioVideoDesyncError) as excinfo:
        _recon(wave)
    # floor(3.0 * 25) = 75 raw frames; the largest LEGAL count at or below is
    # 73 = 9*8 + 1. Suggesting 75 would bounce the user off the k*8+1 rule;
    # suggesting 81 would bounce them straight back off this refusal.
    assert excinfo.value.largest_fitting_num_frames == 73
    assert "num_frames=73" in str(excinfo.value)
    assert "4.84" in str(excinfo.value) and "3.0" in str(excinfo.value)
    from ltx25_studio import constraints

    assert constraints.is_valid_num_frames(excinfo.value.largest_fitting_num_frames)


def test_a_short_upload_is_never_padded():
    """The refusal IS the no-padding guarantee, asserted from the other side:
    nothing returns a waveform longer than it was given."""
    from ltx25_studio.errors import AudioVideoDesyncError

    wave = waveform_of(2, samples=16000)
    with pytest.raises(AudioVideoDesyncError):
        _recon(wave)


def test_an_absurdly_short_upload_still_gets_a_legal_suggestion():
    from ltx25_studio.errors import AudioVideoDesyncError

    wave = waveform_of(2, samples=160)  # 0.01 s
    with pytest.raises(AudioVideoDesyncError) as excinfo:
        _recon(wave)
    from ltx25_studio import constraints

    assert constraints.is_valid_num_frames(excinfo.value.largest_fitting_num_frames)


# --------------------------------------------------------------------------
# review r1 media-io-5 -- a long upload is decoded only as far as the run can
# use it, and the trimmed waveform does not keep the whole decode alive
# --------------------------------------------------------------------------


def test_a_trimmed_upload_does_not_keep_the_whole_decode_alive():
    import numpy as np

    wave = waveform_of(2, samples=16000 * 120)  # two minutes against a 4.84 s video
    out, _note = _recon(wave)
    assert out.shape[-1] == 77440
    assert not np.shares_memory(out, wave), "the 4.84 s kept pins all 120 s decoded"


def test_a_trimmed_torch_waveform_does_not_keep_the_whole_decode_alive():
    torch = pytest.importorskip("torch")
    wave = torch.zeros(2, 16000 * 120)
    out, _note = _recon(wave)
    assert out.untyped_storage().nbytes() == out.numel() * out.element_size()


def test_the_trim_note_can_name_the_uploads_real_length():
    _out, note = audio.reconcile_waveform_with_video(
        waveform_of(2, samples=96000),
        sampling_rate=16000,
        params=_RECON_PARAMS,
        num_frames=121,
        frame_rate=25.0,
        source_seconds=3600.0,
    )
    assert note == "audio input trimmed from 3600.00s to 4.84s to match 121 frames"


def _long_wav(path, *, seconds, rate=16000):
    import wave

    import numpy as np

    t = np.arange(int(seconds * rate)) / rate
    left, right = np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)
    samples = (0.4 * 32767 * np.stack([left, right], axis=1)).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


def test_a_bounded_load_decodes_no_further_than_asked(tmp_path):
    pytest.importorskip("torch")
    path = _long_wav(tmp_path / "podcast.wav", seconds=60)

    waveform, rate = audio.load_waveform(path, max_seconds=6.0)

    assert rate == 16000
    assert waveform.shape == (2, 6 * 16000)
    assert waveform.untyped_storage().nbytes() == waveform.numel() * 4, "it holds only itself"


def test_a_bounded_load_gives_the_run_exactly_the_same_audio(tmp_path):
    """Bounded by ``upload_seconds_needed``, the trimmed waveform is the one a
    full decode would have given."""
    torch = pytest.importorskip("torch")
    path = _long_wav(tmp_path / "podcast.wav", seconds=30)
    needed = audio.upload_seconds_needed(num_frames=121, frame_rate=25.0)

    full, rate = audio.load_waveform(path)
    bounded, _ = audio.load_waveform(path, max_seconds=needed)

    assert bounded.shape[-1] < full.shape[-1]
    from_full, _ = _recon(full, sampling_rate=rate)
    from_bounded, _ = _recon(bounded, sampling_rate=rate)
    assert torch.equal(from_full, from_bounded)


def test_a_short_upload_is_still_refused_after_a_bounded_load(tmp_path):
    pytest.importorskip("torch")
    from ltx25_studio.errors import AudioVideoDesyncError

    path = _long_wav(tmp_path / "line.wav", seconds=3)
    needed = audio.upload_seconds_needed(num_frames=121, frame_rate=25.0)

    bounded, rate = audio.load_waveform(path, max_seconds=needed)

    with pytest.raises(AudioVideoDesyncError):
        _recon(bounded, sampling_rate=rate)
