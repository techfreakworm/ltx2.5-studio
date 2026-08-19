"""Audio parameters, read from the model rather than assumed.

PLAN §5.4, §5.7 rows 5 and 11.

**The whole rule: every mel parameter comes from ``audio_vae.config`` at
runtime.** Not one of them is written down in this file. The pipeline itself
works this way -- its constructor reads ``self.vae.spatial_compression_ratio``,
``self.audio_vae.mel_compression_ratio`` and friends off the loaded components,
using literals only as fallbacks when a component is absent -- and a hardcoded
sample rate that happens to be right today is a wrong number waiting for a
checkpoint change. A wrong one would not crash; it would just make the audio
subtly wrong for a whole release.

The same applies to the export rate, which comes from
``vocoder.config.output_sampling_rate``. The model card notes what it should be;
that is a thing to verify at runtime, not to copy into the source.

**Audio input is not a first-class kwarg** (§3.1c). There is no ``audio=``
argument on any LTX2 pipeline. Audio enters as pre-encoded ``audio_latents``,
UNPACKED 4D ``[B, C, L, M]``, combined with ``noise_scale`` -- which behaves like
img2img strength, so 1.0 discards the input entirely. Because a supplied latent
shape *overrides* the duration-derived audio length while ``num_frames`` still
governs the video, the two can desync silently; the shape assertion here is the
first line of that defence and ``constraints.validate_audio_video_agreement`` is
the second.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConstraintError, LTXStudioError

__all__ = [
    "AUDIO_LATENT_NDIM",
    "REQUIRED_AUDIO_CONFIG_FIELDS",
    "AudioParams",
    "audio_params_from",
    "export_sampling_rate_from",
    "validate_audio_latents_shape",
]

#: Read off ``audio_vae.config``. Names only -- never values.
REQUIRED_AUDIO_CONFIG_FIELDS = (
    "audio_sampling_rate",
    "audio_hop_length",
    "audio_mel_bins",
    "audio_vae_mel_compression_ratio",
    "audio_vae_temporal_compression_ratio",
    "audio_latent_channels",
)

#: ``audio_latents`` is UNPACKED ``[B, C, L, M]``.
AUDIO_LATENT_NDIM = len(("batch", "channels", "length", "mel"))


@dataclass(frozen=True)
class AudioParams:
    """Mel/audio parameters, as read from a loaded audio VAE."""

    audio_sampling_rate: int
    audio_hop_length: int
    audio_mel_bins: int
    audio_vae_mel_compression_ratio: int
    audio_vae_temporal_compression_ratio: int
    audio_latent_channels: int


def audio_params_from(audio_vae) -> AudioParams:
    """Read every mel parameter off the loaded audio VAE.

    A missing field raises rather than defaulting. A silent fallback to a
    plausible number is how the audio track ends up subtly wrong for a whole
    release, with nothing anywhere to point at.
    """
    config = getattr(audio_vae, "config", audio_vae)
    missing = [name for name in REQUIRED_AUDIO_CONFIG_FIELDS if not hasattr(config, name)]
    if missing:
        raise LTXStudioError(
            f"the audio VAE config is missing {', '.join(missing)}. Every mel parameter is read "
            f"from the loaded model at runtime; refusing to substitute a default, because a "
            f"plausible wrong value would produce audio that is quietly incorrect rather than "
            f"an error anyone can see."
        )
    return AudioParams(**{name: getattr(config, name) for name in REQUIRED_AUDIO_CONFIG_FIELDS})


def export_sampling_rate_from(vocoder) -> int:
    """The rate to write files at, from the vocoder rather than from memory."""
    config = getattr(vocoder, "config", vocoder)
    rate = getattr(config, "output_sampling_rate", None)
    if rate is None:
        raise LTXStudioError(
            "the vocoder config has no output_sampling_rate. It is the only authority for the "
            "export rate; assuming one would silently resample every output."
        )
    return int(rate)


def validate_audio_latents_shape(shape):
    """Assert ``audio_latents`` is unpacked 4D at our own boundary.

    Upstream will accept a wrongly shaped tensor and misinterpret it rather than
    complain, which turns a shape bug into a quietly wrong audio track.
    """
    shape = tuple(shape)
    if len(shape) != AUDIO_LATENT_NDIM:
        raise ConstraintError(
            f"audio_latents must be unpacked {AUDIO_LATENT_NDIM}-dimensional [B, C, L, M], but "
            f"the supplied tensor has {len(shape)} dimensions ({shape}). Pass unpacked latents; "
            f"a packed tensor is silently misread rather than rejected."
        )
    return shape
