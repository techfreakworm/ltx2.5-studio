"""waveform <-> mel <-> ``audio_vae`` latents.

PLAN §3.1(c), §5.4, §5.7 rows 5 and 11, §10 OQ-5.

**Audio input is not a first-class kwarg.** It enters as pre-encoded
``audio_latents`` combined with ``noise_scale``, where upstream computes
``noised = noise_scale * noise + (1 - noise_scale) * latents`` -- img2img-style
strength, so 1.0 discards the supplied audio entirely. This module owns the
conversion between the waveform a human uploads and the latents the pipeline
wants.

**Every parameter is read from ``audio_vae.config`` at runtime. None of them is
written down here.** That is §5.4 in full, and it mirrors the pipeline itself,
whose constructor reads ``self.vae.spatial_compression_ratio``,
``self.audio_vae.mel_compression_ratio`` and friends from the loaded components.
A hardcoded sample rate or hop length that happens to be right for today's
checkpoint is a wrong number waiting for a checkpoint change -- and unlike a
crash, a wrong mel front-end produces audio that is merely *subtly incorrect*,
for a whole release, with nothing to read afterwards. A missing config field
therefore raises here and names itself; it never falls back to a plausible
default.

**The length trap (§5.7 row 11, OQ-5).** Upstream derives the audio length from
the requested duration, then, when ``audio_latents`` are supplied, logs
``"Got audio_latents of shape [...], audio_num_frames will be inferred"`` and
does ``_, _, audio_num_frames, _ = audio_latents.shape`` -- the latent shape
**overrides** the duration. Meanwhile ``num_frames`` still governs the video. So
supplying audio silently redefines the audio length only, and the two tracks
desync. :func:`validate_audio_latents_against_video` converts that into
:class:`~ltx25_studio.errors.AudioVideoDesyncError` before the call.

**Shapes are asserted at our boundary (§5.7 row 5).** ``audio_latents`` is
**unpacked 4D** ``[B, C, L, M]``. A wrong rank is not reliably a crash upstream;
it can be a silent misinterpretation, which is why we check rather than hope.

The mel front-end and the codec are **injected seams**. ``torch`` and
``torchaudio`` are imported lazily inside the functions that need them, so this
module -- and its spec -- run with no torch, no weights and no audio file
present (PLAN §9.1).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .constraints import DEFAULT_LENGTH_TOLERANCE_SECONDS, validate_audio_video_agreement
from .errors import LTXStudioError

logger = logging.getLogger(__name__)

__all__ = [
    "AUDIO_LATENT_DIMS",
    "AUDIO_LATENT_NDIM",
    "CHANNELS_FIELD",
    "CONFIG_FIELDS",
    "PARAM_SOURCES",
    "EXPORT_SAMPLING_RATE_FIELD",
    "AudioParams",
    "audio_channels_from",
    "audio_frames_from_latents",
    "audio_params_from",
    "decode_latents_to_waveform",
    "encode_waveform_to_latents",
    "load_waveform",
    "deinterleaved",
    "match_channel_count",
    "export_sampling_rate_from",
    "mel_front_end_for",
    "build_mel_front_end",
    "encoder_layout",
    "batched",
    "mel_parameters",
    "MEL_PARAM_SOURCES",
    "MEL_CONVENTIONS",
    "MEL_INPUT_DIMS",
    "validate_audio_latents_against_video",
    "validate_audio_latents_shape",
]

#: The axes of an unpacked ``audio_latents`` tensor, in order (PLAN §5.7 row 5).
#: Named rather than numbered so the rank and the frames axis are *derived* from
#: one declaration instead of appearing as bare integers in three places.
AUDIO_LATENT_DIMS = ("batch", "channels", "frames", "mel")

#: The rank ``audio_latents`` must have. Falls out of :data:`AUDIO_LATENT_DIMS`.
AUDIO_LATENT_NDIM = len(AUDIO_LATENT_DIMS)

#: Which axis upstream reads the audio length from:
#: ``_, _, audio_num_frames, _ = audio_latents.shape``.
_FRAMES_AXIS = AUDIO_LATENT_DIMS.index("frames")

#: The six §5.4 parameters, and **where each one really lives on the loaded
#: audio VAE**. Our names are the semantic layer; the checkpoint uses its own.
#:
#: ⚠️ These are NOT six keys on ``audio_vae.config``. That is what this module
#: assumed, and every one of the six was wrong: four are config keys under
#: *different* names, and two are **instance attributes on the model**, set in
#: ``__init__`` and absent from ``config.json`` entirely. Reading all six off
#: ``.config`` by our own names raised on the real checkpoint, which made every
#: runtime path through this module -- the audio-input encode included --
#: unreachable on real weights while passing offline against a fake that used
#: our names.
#:
#: *Read off ``audio_vae/config.json`` at revision 871165de… and
#: ``autoencoder_kl_ltx2_audio.py`` at diffusers ac56fa2f…, and cross-checked
#: against how ``LTX2Pipeline.__init__`` derives the same six.*
PARAM_SOURCES = MappingProxyType(
    {
        "audio_sampling_rate": ("config", "sample_rate"),
        "audio_hop_length": ("config", "mel_hop_length"),
        "audio_mel_bins": ("config", "mel_bins"),
        "audio_latent_channels": ("config", "latent_channels"),
        # Instance attributes, both LATENT_DOWNSAMPLE_FACTOR. Not in config.json.
        "audio_vae_mel_compression_ratio": ("model", "mel_compression_ratio"),
        "audio_vae_temporal_compression_ratio": ("model", "temporal_compression_ratio"),
    }
)

#: The parameter names PLAN §5.4 names explicitly. Derived from
#: :data:`PARAM_SOURCES` so the two cannot disagree.
CONFIG_FIELDS = tuple(PARAM_SOURCES)

#: Where the export sample rate comes from -- ``vocoder.config``, not the VAE's.
#: The vocoder resamples, so the rate it emits is the only correct one to write
#: into the file, and it is read at runtime rather than assumed.
EXPORT_SAMPLING_RATE_FIELD = "output_sampling_rate"

#: The audio VAE's channel count. The stack is **stereo**; a mono downmix
#: anywhere in the path is a silent quality regression, so the count is read and
#: checked rather than presumed.
CHANNELS_FIELD = "in_channels"


@dataclass(frozen=True)
class AudioParams:
    """The six §5.4 parameters, as read from one loaded ``audio_vae``.

    A frozen record rather than loose kwargs: it is passed between the encode,
    decode and length-derivation paths, and a value that can be edited in flight
    is a value that can disagree with the VAE that produced it.
    """

    audio_sampling_rate: int
    audio_hop_length: int
    audio_mel_bins: int
    audio_vae_mel_compression_ratio: int
    audio_vae_temporal_compression_ratio: int
    audio_latent_channels: int

    @property
    def samples_per_latent_frame(self) -> int:
        """Waveform samples collapsed into one latent frame.

        One hop makes one mel frame; ``audio_vae_temporal_compression_ratio``
        mel frames make one latent frame. Both numbers come from the config, so
        this ratio does too.
        """
        return self.audio_hop_length * self.audio_vae_temporal_compression_ratio

    def seconds_for_latent_frames(self, latent_frames: int) -> float:
        """The wall-clock length of a run of latent frames."""
        return (latent_frames * self.samples_per_latent_frame) / self.audio_sampling_rate

    def latent_frames_for_seconds(self, seconds: float) -> int:
        """How many latent frames a given duration occupies."""
        return int(round(seconds * self.audio_sampling_rate)) // self.samples_per_latent_frame


# ---------------------------------------------------------------------------
# reading the config -- loudly
# ---------------------------------------------------------------------------


def _config_of(component, *, name: str):
    config = getattr(component, "config", None)
    if config is None:
        raise LTXStudioError(
            f"{name} ({type(component).__name__}) has no .config, so its audio parameters "
            f"cannot be read at runtime. Every mel parameter comes from the loaded component's "
            f"config (PLAN §5.4) and this package deliberately holds no defaults to fall back "
            f"on -- a plausible-looking guess would make the audio track subtly wrong rather "
            f"than making it fail."
        )
    return config


def _read(config, field: str, *, owner: str):
    """Return ``config[field]``, or raise naming the field that is absent."""
    if isinstance(config, Mapping):
        if field in config:
            return config[field]
    elif hasattr(config, field):
        return getattr(config, field)
    raise LTXStudioError(
        f"{owner} does not declare {field!r}, so it cannot be read at runtime. PLAN §5.4 "
        f"requires every audio parameter to come from the loaded component's config, and "
        f"substituting a default here is how an audio track ends up subtly wrong for a whole "
        f"release. Either the checkpoint is not the one this app expects, or the component "
        f"passed in is not the audio VAE."
    )


def _read_param(audio_vae, config, name: str):
    """One §5.4 parameter, from wherever the checkpoint actually keeps it."""
    where, field = PARAM_SOURCES[name]
    if where == "config":
        return _read(config, field, owner="audio_vae.config")
    found = getattr(audio_vae, field, None)
    if found is None:
        raise LTXStudioError(
            f"audio_vae does not expose {field!r}, which is where {name!r} comes from. "
            f"Upstream sets it in the model's __init__ rather than declaring it in "
            f"config.json, so a checkpoint without it is not the one this app expects."
        )
    return found


def audio_params_from(audio_vae) -> AudioParams:
    """Read all six §5.4 parameters off a loaded ``audio_vae``.

    Nothing is defaulted and nothing is overridden: a checkpoint whose audio VAE
    runs at an unusual sample rate gets that rate, because the alternative is
    honouring a number someone assumed while writing this module.
    """
    config = _config_of(audio_vae, name="audio_vae")
    return AudioParams(**{name: _read_param(audio_vae, config, name) for name in CONFIG_FIELDS})


def audio_channels_from(audio_vae) -> int:
    """The audio VAE's channel count -- stereo on this stack.

    Read rather than assumed for the same reason as everything else here, and
    surfaced separately from :func:`audio_params_from` so that function stays
    exactly the six parameters §5.4 enumerates.
    """
    config = _config_of(audio_vae, name="audio_vae")
    return int(_read(config, CHANNELS_FIELD, owner="audio_vae.config"))


def export_sampling_rate_from(vocoder) -> int:
    """The rate the finished waveform must be written at.

    Comes from ``vocoder.config`` and not from the VAE: the vocoder is what
    actually emits the samples, and writing a file with the wrong declared rate
    is a pitch-shifted track that nothing reports as an error.
    """
    config = _config_of(vocoder, name="vocoder")
    return int(_read(config, EXPORT_SAMPLING_RATE_FIELD, owner="vocoder.config"))


# ---------------------------------------------------------------------------
# shapes at our boundary -- PLAN §5.7 row 5
# ---------------------------------------------------------------------------


def validate_audio_latents_shape(shape) -> tuple[int, ...]:
    """Return the shape unchanged, or raise naming the rank we got.

    ✅ **Confirmed against real weights**, and it was right. On its first
    contact with a real encode it rejected a 3-D tensor the VAE itself had
    accepted without complaint -- the batch axis was missing, so ``conv2d``
    silently treated the input as unbatched. The instinct was to loosen this
    check; the check was correct and the input was wrong.

    ⚠️ **But do not read acceptance here as more than it means.** ✅ Measured on
    the card, one process, the transpose as the only difference::

        latents (1, 8, 101, 16)  ACCEPTED   reconstruction 4.010 s   4/4 tones
        latents (1, 8,  16, 101)  ACCEPTED   reconstruction 0.610 s   0/4 tones

    **Rank is preserved by transposition, so this cannot distinguish a correct
    tensor from its transpose.** It accepted the wrong one just as readily, and
    that variant lost every tone and pinned the output to a fixed length.

    What does catch it is **the reconstruction length scaling with the input
    length** -- an axis-order error in either direction makes the output
    duration-invariant, and no knowledge of which axis is which is required.
    ``test_the_frame_count_tracks_the_input_duration`` is the offline half;
    TEST_PLAN A.5.1 (H17) is the end-to-end one.

    ``audio_latents`` is **unpacked** ``[B, C, L, M]``. Upstream does not
    reliably raise on a wrong rank -- the documented failure is "shape errors,
    **or worse, silent misinterpretation**" -- so the assertion lives at our
    boundary where it can still say something useful.
    """
    dims = tuple(shape)
    if len(dims) == AUDIO_LATENT_NDIM:
        return dims
    raise LTXStudioError(
        f"audio_latents must be unpacked {AUDIO_LATENT_NDIM}D "
        f"[{', '.join(AUDIO_LATENT_DIMS)}], but got {len(dims)}D {dims}. A packed or "
        f"wrongly-ranked tensor is not reliably an error upstream -- it can be silently "
        f"misinterpreted -- so it is refused here instead."
    )


def audio_frames_from_latents(shape) -> int:
    """The audio length upstream will infer from a supplied latent tensor.

    Quoted from upstream: ``_, _, audio_num_frames, _ = audio_latents.shape``.
    This value **overrides** the duration-derived length while ``num_frames``
    still governs the video, which is the whole of the §5.7 row 11 desync.
    """
    return validate_audio_latents_shape(shape)[_FRAMES_AXIS]


def validate_audio_latents_against_video(
    shape,
    *,
    params: AudioParams,
    video_seconds: float,
    tolerance_seconds: float = DEFAULT_LENGTH_TOLERANCE_SECONDS,
) -> float:
    """Raise unless supplied ``audio_latents`` match the video length.

    Returns the audio length in seconds so a caller can log what it compared.
    This is the check that stops a desynced file being emitted rather than
    refused (PLAN §5.5, §5.7 row 11, OQ-5).
    """
    audio_seconds = params.seconds_for_latent_frames(audio_frames_from_latents(shape))
    validate_audio_video_agreement(
        video_seconds=video_seconds,
        audio_seconds=audio_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    return audio_seconds


# ---------------------------------------------------------------------------
# waveform <-> mel <-> latents
# ---------------------------------------------------------------------------

#: Attribute names a loaded ``audio_vae`` may carry its own mel front-end under.
#: Searched in order; we never construct one from guessed FFT parameters,
#: because the window and FFT size are not among the fields §5.4 declares and
#: inventing them would produce a mel spectrogram the VAE was not trained on --
#: silently, and only audible as degraded audio.
MEL_FRONT_END_ATTRS = ("mel_spectrogram", "mel_transform", "_mel_spectrogram")


#: Where each mel parameter comes from. **Four of the five are declared by BOTH
#: components and agree**, which is what makes this a lookup rather than a
#: reconstruction: the vocoder's ``num_mel_channels`` is 64 and the VAE's
#: ``mel_bins`` is 64; the vocoder's ``input_sampling_rate`` is 16000 and the
#: VAE's ``sample_rate`` is 16000.
#:
#: ⚠️ **The hop is the one disagreement, and it is read from the VAE on purpose.**
#: ``audio_vae.mel_hop_length`` is 160; ``vocoder.hop_length`` is 80. The
#: vocoder's own ``upsample_factors`` -- ``[5, 2, 2, 2, 2, 2]`` -- multiply to
#: **160**, matching the VAE rather than the vocoder's own declared 80, so that
#: 80 refers to something else inside the vocoder. Read ``window_length`` across
#: from the vocoder; **do NOT read a hop across.** The asymmetry is deliberate
#: and is exactly the kind of thing a later reader will "tidy" into consistency.
#:
#: *Read off vocoder/config.json and audio_vae/config.json at revision
#: 871165de…, cross-checked against the upsample factors.*
MEL_PARAM_SOURCES = MappingProxyType(
    {
        "n_fft": ("vocoder", "filter_length"),
        "win_length": ("vocoder", "window_length"),
        "hop_length": ("audio_vae", "mel_hop_length"),
        "n_mels": ("audio_vae", "mel_bins"),
        "sample_rate": ("audio_vae", "sample_rate"),
    }
)

#: ⚠️ **These are CONVENTIONS, not readings.** Neither config declares them and
#: no forward transform ships anywhere in ``diffusers.pipelines.ltx2`` -- the
#: vocoder's only ``hann`` window belongs to its anti-aliasing filter, not to an
#: STFT. So unlike everything in :data:`MEL_PARAM_SOURCES`, each of these is a
#: choice, and a wrong one is *"audible as degraded audio, reported as nothing"*.
#:
#: **The round-trip test is what validates them**, and it is not optional
#: decoration: a wrong window or scaling smears the spectrum and the
#: reconstruction degrades measurably. If that test fails, these are the
#: candidates -- in this order.
MEL_CONVENTIONS = MappingProxyType(
    {
        "power": 1.0,  # magnitude, not power; log is applied after
        "f_min": 0.0,
        "f_max": None,  # -> sample_rate / 2
        "mel_scale": "slaney",
        "norm": "slaney",
        "center": True,
    }
)

#: The layout ``audio_vae.encode`` requires, **quoted from upstream**:
#: ``# hidden_states expected shape: (batch_size, channels, time, num_mel_bins)``
#: -- ``autoencoder_kl_ltx2_audio.py:445`` at diffusers ``ac56fa2f…``.
#:
#: ⚠️ ``torchaudio`` returns ``[channels, mel, time]``: **unbatched, and with
#: the last two axes the other way round.** Both have to be corrected, and the
#: transpose is the one that hides. Supplying only the batch axis gives
#: ``[1, 2, 64, 401]``, which the encoder reads as 64 time steps of 401 mel bins
#: -- **a 4D result that passes the rank check, satisfies the validator, and is
#: a transposed spectrogram.** It would have produced audio, and the audio would
#: have been wrong with nothing reporting it.
#:
#: Note this is ``[B, C, time, mel]`` while the *latents* that come back are
#: ``[B, C, frames, mel]`` -- the same ordering, which is why
#: :data:`AUDIO_LATENT_DIMS` was right and needed no loosening.
MEL_INPUT_DIMS = ("batch", "channels", "time", "mel")

#: Floor before the log, so silence does not become -inf.
MEL_LOG_FLOOR = 1e-5


def mel_parameters(audio_vae, vocoder):
    """The five declared mel parameters, each from the component that owns it."""
    owners = {"audio_vae": audio_vae, "vocoder": vocoder}
    values = {}
    for name, (owner, field) in MEL_PARAM_SOURCES.items():
        component = owners[owner]
        if component is None:
            raise LTXStudioError(
                f"the mel front-end needs {name!r}, which is declared by the {owner}, and no "
                f"{owner} was supplied. Every value comes from a loaded component; this module "
                f"holds no defaults, because a guessed FFT size produces a mel spectrogram the "
                f"VAE was never trained on."
            )
        values[name] = int(_read(_config_of(component, name=owner), field, owner=f"{owner}.config"))
    return values


def encoder_layout(mel):
    """``[..., mel, time]`` -> ``[..., time, mel]``.

    Separated from :func:`build_mel_front_end` so it can be exercised without
    torch installed. The transform itself needs real tensors; **this rearrangement
    is where the mistake lives**, and a test that skips in the offline gate is
    not a test of it -- both half-fixes below pass every other check.

    ``torchaudio`` returns the last two axes the other way round from what the
    encoder wants, and unbatched. Correcting only the batch axis yields a 4D
    tensor that satisfies the rank check and is a **transposed spectrogram** --
    audio out, silently wrong. See :data:`MEL_INPUT_DIMS`.

    **This is the ONLY rearrangement in the path, and that is the point.**
    ``audio_vae.encode`` does not reorder axes -- it downsamples both by the
    compression ratio and preserves their order -- so **the mel's layout decides
    the latents' layout.** Fed a time-major mel it returns ``[B, C, L, M]``
    directly, which is what :data:`AUDIO_LATENT_DIMS` declares and what upstream
    packs.

    ⚠️ A second transpose was once applied to the *encode output*, on a
    measurement taken **before this function existed** -- when the mel was still
    mel-major, the latents were too, and the transposition was the mel's showing
    up one stage later. Applying both corrections put the frame axis back where
    it started and pinned every reconstruction to 0.61 s regardless of input.
    **The symptom's location is not its origin**; fixing the mel is the whole
    fix.
    """
    return mel.transpose(-2, -1)


def build_mel_front_end(audio_vae, vocoder):
    """Construct the forward transform, because the checkpoint ships none.

    **Only the forward direction needs this.** ``decode_latents_to_waveform``
    works without any STFT -- the VAE decoder emits a mel and the vocoder
    inverts it -- which is why H5-H9 produce real audio while the audio-INPUT
    modes cannot start. The gap is asymmetric by nature, not by oversight.
    """
    import torch
    import torchaudio

    params = mel_parameters(audio_vae, vocoder)
    conventions = dict(MEL_CONVENTIONS)
    conventions["f_max"] = conventions["f_max"] or params["sample_rate"] / 2
    transform = torchaudio.transforms.MelSpectrogram(**params, **conventions)

    def to_mel(waveform):
        mel = transform.to(device=waveform.device, dtype=torch.float32)(waveform.float())
        mel = torch.log(torch.clamp(mel, min=MEL_LOG_FLOOR))
        return encoder_layout(mel)

    return to_mel


def mel_front_end_for(audio_vae, vocoder=None):
    """The callable that turns a waveform into the mel the VAE expects.

    Deliberately *found*, never built: see :data:`MEL_FRONT_END_ATTRS`. If the
    loaded VAE does not carry one, the caller must pass ``to_mel=`` explicitly
    rather than have this module guess an FFT size.
    """
    for attribute in MEL_FRONT_END_ATTRS:
        found = getattr(audio_vae, attribute, None)
        if callable(found):
            return found
    if vocoder is not None:
        return build_mel_front_end(audio_vae, vocoder)
    raise LTXStudioError(
        f"the loaded audio_vae carries no mel front-end (looked for "
        f"{', '.join(MEL_FRONT_END_ATTRS)}), and this module will not construct one: the FFT "
        f"and window sizes are not among the parameters audio_vae.config declares, so any "
        f"value chosen here would be a guess that produces a mel spectrogram the VAE was never "
        f"trained on -- audible as degraded audio, reported as nothing. Pass to_mel= explicitly."
    )


def _resampled(waveform, *, from_rate: int, to_rate: int):
    """Match the waveform to the VAE's rate, using the VAE's number, not ours."""
    if int(from_rate) == int(to_rate):
        return waveform
    import torchaudio  # lazy: this module must import with no torch present

    return torchaudio.functional.resample(waveform, int(from_rate), int(to_rate))


def _sampled(encoded, generator=None):
    """Unwrap whatever ``audio_vae.encode`` returned into a latent tensor.

    ``.sample()`` **draws**, so it takes a generator like any other sampler.
    Upstream's ``retrieve_latents`` threads one; we called it bare, which made
    two runs draw different latents regardless of the lever being tested. The
    sampling *mode* was always right -- upstream defaults to ``"sample"`` too.
    """
    distribution = getattr(encoded, "latent_dist", None)
    if distribution is not None:
        return distribution.sample(generator) if generator is not None else distribution.sample()
    for attribute in ("latents", "sample"):
        found = getattr(encoded, attribute, None)
        if found is not None:
            return found
    return encoded


def deinterleaved(chunks, *, channels: int, planar: bool):
    """Decoded ``av`` frames -> one ``(channels, samples)`` array.

    **Packed formats interleave, and most wav files are packed.** ``pcm_s16le``
    reports ``is_planar = False``, so each decoded frame arrives as
    ``(1, samples * channels)`` in L,R,L,R order rather than as
    ``(channels, samples)``.

    ``reshape(-1, channels).T`` is the de-interleave. ``reshape(channels, -1)``
    is the bug that is easier to write: it puts the first half of the recording
    in the left channel and the second half in the right. The array has the
    right shape, every check downstream passes, and the audio is destroyed.
    """
    import numpy as np

    if planar:
        return np.concatenate(chunks, axis=1)
    flat = np.concatenate([chunk.reshape(-1) for chunk in chunks])
    return flat.reshape(-1, channels).T


def _as_float_samples(data):
    """Integer PCM -> float in [-1, 1], the range the mel front-end expects.

    ``pcm_s16le`` decodes to ``int16``. Handing those values to the VAE unscaled
    would be a signal four orders of magnitude too loud.
    """
    import numpy as np

    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32, copy=False)
    info = np.iinfo(data.dtype)
    return data.astype(np.float32) / float(max(abs(info.min), info.max))


def load_waveform(source):
    """A user-supplied audio file -> ``(waveform, sampling_rate)``.

    **Decoded with ``av``, not ``torchaudio.load``.** At torchaudio 2.11.0 --
    our pin -- ``torchaudio.load`` dispatches to TorchCodec, which raises
    ``ImportError: TorchCodec is required``. Installing it is a dead end: it
    needs *system* FFmpeg shared libraries (``libavutil.so.56``), which are
    absent on the host with no sudo to add them and cannot be assumed of the
    Space image either.

    ``av`` is already a non-optional dependency -- image conditioning needs it --
    and **bundles its own FFmpeg**, which is exactly why it works where
    torchcodec does not. No new dependency, for a job a required one already
    does.

    Returns a torch tensor because that is what the rest of the path needs:
    ``torchaudio.functional.resample`` and the VAE's own mel front-end.
    torchaudio stays a dependency for exactly that reason -- only its **I/O** is
    behind TorchCodec. *Verified at the pinned tag: ``functional.py`` at
    torchaudio v2.11.0 references torchcodec zero times and imports only
    ``math``, ``warnings``, ``torch``, ``torchaudio`` and ``.filtering``.*
    """
    import av
    import numpy as np
    import torch

    with av.open(str(source)) as container:
        if not container.streams.audio:
            raise LTXStudioError(
                f"{source} carries no audio stream, so there is nothing to encode. "
                f"Supply an audio file rather than a silent or video-only one."
            )
        stream = container.streams.audio[0]
        rate = int(stream.rate)
        channels = int(getattr(stream.layout, "nb_channels", 0) or stream.channels)
        planar = bool(stream.format.is_planar)
        chunks = [frame.to_ndarray() for frame in container.decode(stream)]

    if not chunks:
        raise LTXStudioError(
            f"{source} decoded to no audio at all. The file may be truncated or empty."
        )

    data = _as_float_samples(deinterleaved(chunks, channels=channels, planar=planar))
    return torch.from_numpy(np.ascontiguousarray(data)), rate


def match_channel_count(waveform, *, channels: int):
    """Give the waveform the channel count the audio VAE requires.

    **The VAE is stereo and does not adapt.** Handed one channel it reaches
    ``conv_in`` and raises ``Given groups=1, weight of size [128, 2, 3, 3],
    expected input[1, 1, 4, 480002] to have 2 channels, but got 1`` -- four
    frames inside someone else's conv layer, naming a weight shape rather than
    anything the user did. Most people's wav files are mono, so that is the
    single most likely way this feature meets a real user.

    **Mono is upmixed by duplication, and the caller is told.** Duplicating one
    channel into two adds no information and loses none, so it is a safe default
    where refusing would reject the common case. It is still reported, because a
    user who chose mono deliberately deserves to know their audio was altered.

    **More channels than the VAE takes is refused, not downmixed.** That
    direction *does* lose information, and PLAN 5.4 is explicit that a silent
    downmix is a quality regression nothing reports.

    ``channels`` is read off the VAE by the caller rather than assumed, so a
    checkpoint with a different layout is handled by reading, not editing.

    Returns ``(waveform, note)``; ``note`` is ``None`` when nothing changed.
    """
    shape = getattr(waveform, "shape", None)
    if not shape:
        return waveform, None
    if len(shape) != 2:
        raise LTXStudioError(
            f"the waveform is {len(shape)}D {tuple(shape)} and this expects an UNBATCHED "
            f"(channels, samples). Given a batched one it would read the batch axis as the "
            f"channel count -- a (1, 2, N) stereo clip looks like mono, is 'widened' by "
            f"duplication, and reaches the VAE with a batch of 2 and the user told their "
            f"audio was mono. Batch after this runs, not before."
        )
    found = int(shape[0])
    if found == channels:
        return waveform, None
    if found > channels:
        raise LTXStudioError(
            f"the audio has {found} channels and this model's audio VAE takes exactly "
            f"{channels}. Mixing {found} down to {channels} would discard part of the "
            f"recording, and doing that silently is the quality regression PLAN 5.4 forbids. "
            f"Supply a {channels}-channel file, or downmix it yourself so the choice is yours."
        )
    if found != 1:
        raise LTXStudioError(
            f"the audio has {found} channels and this model's audio VAE takes exactly "
            f"{channels}. Only mono can be widened without inventing a channel."
        )
    # Fancy indexing rather than `.repeat`, whose meaning differs between torch
    # and numpy; this selects row 0 `channels` times and means the same in both.
    return waveform[[0] * channels], (
        f"the audio was mono and this model's audio VAE takes {channels} channels, so the "
        f"single channel was duplicated. No audio was lost, but it is not true stereo."
    )


def batched(waveform):
    """Add the batch axis the encoder requires, after the channel count is settled.

    **Order matters and the two guards disagree unless it is respected.**
    :func:`match_channel_count` reads the channel count off axis 0, so it must
    see an unbatched ``(channels, samples)``; the encoder needs 4D, which needs
    a batch. Batching *after* the channel check is the only order that satisfies
    both -- batching first makes a stereo clip look mono, and not batching at
    all makes the latents come back one rank short.
    """
    shape = getattr(waveform, "shape", None)
    if shape is None or len(shape) != 2:
        return waveform
    # `[None]` rather than `.unsqueeze(0)`: it means the same thing in torch and
    # in numpy, and the offline gate has no torch. Same reason as the fancy
    # indexing in `match_channel_count`.
    return waveform[None]


def _validate_mel_rank(mel) -> None:
    """Refuse a mel the encoder would silently misread.

    A 3D mel is treated by ``conv2d`` as *unbatched*, so the batch axis never
    exists and the latents come back one rank short -- which
    :func:`validate_audio_latents_shape` then rejects, four steps after the
    actual mistake. Failing here names the real problem.

    Rank is all this can check. **A transposed 4D mel is indistinguishable from
    a correct one** at this boundary, which is why the layout is fixed in
    :func:`build_mel_front_end` and documented in :data:`MEL_INPUT_DIMS` rather
    than guarded here.
    """
    shape = getattr(mel, "shape", None)
    if shape is None or len(shape) == len(MEL_INPUT_DIMS):
        return
    raise LTXStudioError(
        f"the mel handed to audio_vae.encode is {len(shape)}D {tuple(shape)}, and the encoder "
        f"expects {len(MEL_INPUT_DIMS)}D {MEL_INPUT_DIMS}. A `to_mel=` callable must return a "
        f"batched, time-major mel; torchaudio's own output is neither."
    )


def _matched_to_module(tensor, component, *, name: str):
    """Make a tensor match the module it is about to enter -- **every attribute**.

    Named for the module, not for one attribute. It was ``_matched_dtype`` and
    it matched exactly one of the two things that have to agree, so a CPU mel
    reached CUDA weights:

        Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type)
        should be the same

    A helper called after a single attribute keeps that attribute and loses the
    others to whoever tidies it to match its own name.

    **Both are read off the component**, never hardcoded. ``WEIGHT_DTYPE`` is
    right until a quantised load lands; ``"cuda"`` is right until offload or a
    second device. The module knows what it is, the same way the audio VAE knows
    its config keys and the vocoder knows its sample rate.

    ⚠️ **This hand-off has four attributes that can disagree** -- dtype, device,
    rank and axis order -- and all four have now been wrong here in turn. Rank
    and axis order are handled by :func:`encoder_layout` and
    :func:`_validate_mel_rank` because they need reshaping rather than a cast;
    the two that ``.to()`` can fix live here. **The list is written down so the
    fifth is looked for rather than waited for.**
    """
    wanted = {}
    for attribute in ("dtype", "device"):
        target = getattr(component, attribute, None)
        if target is not None and getattr(tensor, attribute, None) != target:
            wanted[attribute] = target
    if not wanted:
        return tensor
    cast = getattr(tensor, "to", None)
    if not callable(cast):
        raise LTXStudioError(
            f"the mel handed to {name} is a {type(tensor).__name__} and cannot be moved to "
            f"{wanted}. A `to_mel=` callable must return a tensor."
        )
    return cast(**wanted)


def encode_waveform_to_latents(
    waveform,
    *,
    audio_vae,
    sampling_rate: int,
    to_mel=None,
    notes=None,
    vocoder=None,
    generator=None,
):
    """waveform -> mel -> ``audio_vae`` latents, ready to pass as ``audio_latents``.

    The img2img route of PLAN §3.1(c): the result is combined upstream as
    ``noise_scale * noise + (1 - noise_scale) * latents``, so the caller must
    always pass ``noise_scale`` explicitly -- its default differs by pipeline
    class (§5.7 row 3).

    ``sampling_rate`` is the rate of the *incoming* waveform; the VAE's own rate
    comes from its config and wins. The returned shape is asserted before it can
    reach the pipeline.

    **Returns unpacked, UN-NORMALISED, un-noised 4D ``[B, C, L, M]``.** That is
    the whole job and stopping there is deliberate:
    ``prepare_audio_latents`` upstream calls ``_pack_audio_latents``,
    ``_normalize_audio_latents`` and ``_create_noised_state`` itself, so
    anything applied here would be applied twice. *"Should this normalise?" is
    the obvious next question and the answer is no.*
    """
    params = audio_params_from(audio_vae)
    prepared = _resampled(waveform, from_rate=sampling_rate, to_rate=params.audio_sampling_rate)
    # In the shared helper, not at one call site: every caller reaches the same
    # conv layer, so a check at one of them leaves the others raising a tensor
    # shape error from four frames down.
    prepared, note = match_channel_count(prepared, channels=audio_channels_from(audio_vae))
    if note:
        logger.warning(note)
        if notes is not None:
            notes.append(note)
    # AFTER the channel check, never before -- see `batched`.
    prepared = batched(prepared)
    mel = (to_mel or mel_front_end_for(audio_vae, vocoder))(prepared)
    # ⚠️ Cast HERE, at the hand-off, not inside the transform. The mel is
    # computed in float32 on purpose -- an STFT in bf16 would lose precision
    # exactly where the log floor matters -- but the VAE's weights are bf16 and
    # `conv_in` will not take a float32 input.
    #
    # In this shared helper rather than at a call site, for the reason two lines
    # above: every caller reaches the same `conv_in`, and a cast at one of them
    # leaves the others raising from inside diffusers. It also protects an
    # explicitly-passed `to_mel=` from a caller who does not know the VAE dtype.
    mel = _matched_to_module(mel, audio_vae, name="audio_vae")
    _validate_mel_rank(mel)
    latents = _sampled(audio_vae.encode(mel), generator)
    validate_audio_latents_shape(getattr(latents, "shape", ()))
    return latents


def decode_latents_to_waveform(latents, *, audio_vae, vocoder):
    """``audio_vae`` latents -> mel -> vocoder -> ``(waveform, sampling_rate)``.

    The audio half of the §7.4 decode graph. The channel layout is whatever the
    vocoder emits and is passed through untouched: the stack is stereo, and a
    mono downmix here would be a quality regression nothing reports.
    """
    validate_audio_latents_shape(getattr(latents, "shape", ()))
    decoded = audio_vae.decode(latents)
    mel = getattr(decoded, "sample", decoded)
    return vocoder(mel), export_sampling_rate_from(vocoder)
