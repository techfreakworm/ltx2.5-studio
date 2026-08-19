"""waveform <-> mel <-> ``audio_vae`` latents.

PLAN §3.1(c): audio input is not a first-class kwarg. It enters as
``audio_latents`` combined with ``noise_scale``, where
``noised = noise_scale * noise + (1 - noise_scale) * latents`` — img2img-style
strength, so 1.0 discards the input entirely.

PLAN §5.4: every mel parameter is read from ``audio_vae.config`` at runtime
(``audio_sampling_rate``, ``audio_hop_length``, ``audio_mel_bins``,
``audio_vae_mel_compression_ratio``, ``audio_vae_temporal_compression_ratio``,
``audio_latent_channels``). Nothing here is hardcoded.

See §10 OQ-5 before writing the length-derivation code.
"""
