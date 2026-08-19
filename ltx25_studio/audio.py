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

from collections.abc import Mapping
from dataclasses import dataclass

from .constraints import DEFAULT_LENGTH_TOLERANCE_SECONDS, validate_audio_video_agreement
from .errors import LTXStudioError

__all__ = [
    "AUDIO_LATENT_DIMS",
    "AUDIO_LATENT_NDIM",
    "CHANNELS_FIELD",
    "CONFIG_FIELDS",
    "EXPORT_SAMPLING_RATE_FIELD",
    "AudioParams",
    "audio_channels_from",
    "audio_frames_from_latents",
    "audio_params_from",
    "decode_latents_to_waveform",
    "encode_waveform_to_latents",
    "load_waveform",
    "export_sampling_rate_from",
    "mel_front_end_for",
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

#: The parameter names PLAN §5.4 names explicitly. All six are read from
#: ``audio_vae.config`` at runtime; none has a default anywhere in this package.
CONFIG_FIELDS = (
    "audio_sampling_rate",
    "audio_hop_length",
    "audio_mel_bins",
    "audio_vae_mel_compression_ratio",
    "audio_vae_temporal_compression_ratio",
    "audio_latent_channels",
)

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


def audio_params_from(audio_vae) -> AudioParams:
    """Read all six §5.4 parameters off a loaded ``audio_vae``.

    Nothing is defaulted and nothing is overridden: a checkpoint whose audio VAE
    runs at an unusual sample rate gets that rate, because the alternative is
    honouring a number someone assumed while writing this module.
    """
    config = _config_of(audio_vae, name="audio_vae")
    return AudioParams(
        **{name: _read(config, name, owner="audio_vae.config") for name in CONFIG_FIELDS}
    )


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


def mel_front_end_for(audio_vae):
    """The callable that turns a waveform into the mel the VAE expects.

    Deliberately *found*, never built: see :data:`MEL_FRONT_END_ATTRS`. If the
    loaded VAE does not carry one, the caller must pass ``to_mel=`` explicitly
    rather than have this module guess an FFT size.
    """
    for attribute in MEL_FRONT_END_ATTRS:
        found = getattr(audio_vae, attribute, None)
        if callable(found):
            return found
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


def _sampled(encoded):
    """Unwrap whatever ``audio_vae.encode`` returned into a latent tensor."""
    distribution = getattr(encoded, "latent_dist", None)
    if distribution is not None:
        return distribution.sample()
    for attribute in ("latents", "sample"):
        found = getattr(encoded, attribute, None)
        if found is not None:
            return found
    return encoded


def load_waveform(source):
    """A user-supplied audio file -> ``(waveform, sampling_rate)``.

    ``gr.Audio(type="filepath")`` hands us a path, so the file has to be read
    before anything can encode it. ``torchaudio`` is already a pinned dependency
    and returns exactly the tensor :func:`encode_waveform_to_latents` expects,
    at the file's own rate -- which is then resampled to the VAE's rate there,
    using the VAE's number rather than ours.

    Lazy import for the same reason as ``_resampled``: this module must import
    with no torch present, because the offline gate has none.
    """
    import torchaudio

    waveform, rate = torchaudio.load(str(source))
    return waveform, int(rate)


def encode_waveform_to_latents(waveform, *, audio_vae, sampling_rate: int, to_mel=None):
    """waveform -> mel -> ``audio_vae`` latents, ready to pass as ``audio_latents``.

    The img2img route of PLAN §3.1(c): the result is combined upstream as
    ``noise_scale * noise + (1 - noise_scale) * latents``, so the caller must
    always pass ``noise_scale`` explicitly -- its default differs by pipeline
    class (§5.7 row 3).

    ``sampling_rate`` is the rate of the *incoming* waveform; the VAE's own rate
    comes from its config and wins. The returned shape is asserted before it can
    reach the pipeline.
    """
    params = audio_params_from(audio_vae)
    prepared = _resampled(waveform, from_rate=sampling_rate, to_rate=params.audio_sampling_rate)
    mel = (to_mel or mel_front_end_for(audio_vae))(prepared)
    latents = _sampled(audio_vae.encode(mel))
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
