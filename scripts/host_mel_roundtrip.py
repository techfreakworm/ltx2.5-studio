"""TEST_PLAN A.5.1 / H17 — validate the constructed mel by reconstruction.

    ./scripts/hinode_run.sh host_mel_roundtrip.py

**Required before H10/H11 are believed.** The forward mel transform does not exist
in the checkpoint, so `build_mel_front_end` constructs it. Five parameters are
*read* — `n_fft`/`win_length` from the vocoder, `hop_length`/`n_mels`/`sample_rate`
from the audio VAE — and the rest (`power`, `f_min`/`f_max`, `mel_scale`/`norm`,
`center`, the log and its floor) are **conventions nobody declared**. Nothing
offline can validate those, and a wrong one is *"audible as degraded audio,
reported as nothing"*. H10/H11 will produce well-formed audio with a wrong window,
which is exactly what that phrase means.

So it is validated by reconstruction::

    known waveform -> to_mel -> audio_vae.encode -> audio_vae.decode
                   -> vocoder -> waveform            -> compare

**The fixture must have structure a wrong window would damage.** Two tones an
octave apart, per channel, plus an amplitude envelope — never noise and never
silence, because both survive a wrong analysis window looking much the same and
would make this pass while proving nothing. Same trap as a float32 fixture over a
bfloat16 defect.

Reported per stage, because "the round trip is bad" is not actionable: a failure in
the mel, in the VAE, or in the vocoder all look identical at the output.
"""

import os

import numpy as np
import torch

from ltx25_studio import audio as audio_module
from ltx25_studio import pipelines

GiB = 1024**3
SECONDS = float(os.environ.get("MEL_SECONDS", "4.0"))


def log(*a):
    print(*a, flush=True)


def structured_waveform(rate, channels=2, seconds=SECONDS):
    """Two tones an octave apart plus an envelope. NOT noise, NOT silence.

    A wrong analysis window smears narrow spectral peaks, so tones are what make
    the damage measurable. Broadband noise has no peaks to smear and silence has
    no content at all: both reconstruct to something that looks fine.
    """
    n = int(seconds * rate)
    t = np.arange(n, dtype="float32") / rate
    envelope = 0.45 * (1.0 - np.exp(-4.0 * t)) * (0.65 + 0.35 * np.sin(2 * np.pi * 0.8 * t))
    left = envelope * (0.7 * np.sin(2 * np.pi * 220.0 * t) + 0.3 * np.sin(2 * np.pi * 1320.0 * t))
    right = envelope * (0.7 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 2640.0 * t))
    return torch.from_numpy(np.stack([left, right])[:channels])


def spectral_profile(x, rate, n_fft=1024):
    """Magnitude spectrum of the whole signal, for comparing shape not phase.

    The vocoder does not preserve phase and the round trip is lossy by design, so
    a sample-wise difference would fail on a correct implementation. Spectral
    shape is what a wrong window actually damages.
    """
    win = np.hanning(n_fft).astype("float32")
    hop = n_fft // 4
    cols = []
    for i in range(max((len(x) - n_fft) // hop, 1)):
        seg = x[i * hop : i * hop + n_fft]
        if len(seg) < n_fft:
            break
        cols.append(np.abs(np.fft.rfft(seg * win)))
    if not cols:
        return None
    return np.stack(cols, axis=1).mean(axis=1)


def peak_bins(profile, rate, n_fft=1024, count=4):
    """Where the energy actually is, in Hz. A smeared window moves these."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
    idx = np.argsort(profile)[::-1][:count]
    return sorted(round(float(freqs[i]), 1) for i in idx)


def main() -> int:
    pipe = pipelines._base_pipeline(device="cuda")
    vae, vocoder = pipe.audio_vae, pipe.vocoder
    log(f"loaded; cuda {torch.cuda.memory_allocated() / GiB:.3f} GiB")

    log("\n=== the mel specification, read vs chosen ===")
    params = audio_module.mel_parameters(vae, vocoder)
    for k, v in params.items():
        log(f"  READ       {k:<14} = {v}")
    for k, v in audio_module.MEL_CONVENTIONS.items():
        log(f"  CONVENTION {k:<14} = {v}")
    log(f"  CONVENTION {'log floor':<14} = {audio_module.MEL_LOG_FLOOR}")

    rate = int(params["sample_rate"])
    channels = audio_module.audio_channels_from(vae)
    wave = structured_waveform(rate, channels)
    log(f"\nfixture: {tuple(wave.shape)} @ {rate} Hz, {channels} ch, "
        f"two tones + envelope per channel")
    src = wave.numpy()
    for c in range(src.shape[0]):
        prof = spectral_profile(src[c], rate)
        log(f"  ch{c}: rms {float(np.sqrt((src[c] ** 2).mean())):.6f}  "
            f"peaks at {peak_bins(prof, rate)} Hz")

    dev_wave = wave.to(device=vae.device, dtype=vae.dtype)

    log("\n=== stage 1: forward mel ===")
    # The app's own transform, unwrapped. `encode_waveform_to_latents` casts to the
    # VAE's dtype itself as of 5aa5d7b, so a wrapper here would test the wrapper.
    to_mel = audio_module.build_mel_front_end(vae, vocoder)
    mel = to_mel(dev_wave)
    log(f"  mel {tuple(mel.shape)} dtype={mel.dtype} "
        f"min {float(mel.min()):.4f} max {float(mel.max()):.4f} "
        f"mean {float(mel.float().mean()):.4f}")
    if float(mel.max()) - float(mel.min()) < 1e-3:
        log("  *** mel is nearly constant -- the transform is not producing structure ***")

    log("\n=== stage 2: audio_vae.encode ===")
    # UNBATCHED (channels, samples) -- what `load_waveform` actually returns and
    # what `match_channel_count` expects. This is the shape a user's upload takes,
    # so it is the one worth testing; batching it instead tests a shape no caller
    # produces.
    log(f"  waveform as load_waveform returns it: {tuple(dev_wave.shape)}")
    latents = audio_module.encode_waveform_to_latents(
        dev_wave, audio_vae=vae, sampling_rate=rate, to_mel=to_mel
    )
    log(f"  latents {tuple(latents.shape)} dtype={latents.dtype} "
        f"absmax {float(latents.float().abs().max()):.4f} "
        f"rms {float(latents.float().pow(2).mean().sqrt()):.4f}")
    log(f"  shape validated against AUDIO_LATENT_DIMS {audio_module.AUDIO_LATENT_DIMS}")

    log("\n=== stage 3+4: audio_vae.decode -> vocoder ===")
    out, out_rate = audio_module.decode_latents_to_waveform(
        latents, audio_vae=vae, vocoder=vocoder
    )
    rec = out.detach().float().cpu().numpy()
    while rec.ndim > 2:
        rec = rec[0]
    log(f"  reconstruction {rec.shape} @ {out_rate} Hz")

    log("\n=== compare ===")
    log(f"  input  {src.shape} @ {rate} Hz    output {rec.shape} @ {out_rate} Hz")
    # The vocoder emits 48000 from a 16000 analysis path, so compare each in its
    # own rate rather than resampling one to the other and blaming the resampler.
    ok = True
    for c in range(min(src.shape[0], rec.shape[0])):
        in_rms = float(np.sqrt((src[c] ** 2).mean()))
        out_rms = float(np.sqrt((rec[c] ** 2).mean()))
        in_peaks = peak_bins(spectral_profile(src[c], rate), rate)
        out_peaks = peak_bins(spectral_profile(rec[c], out_rate), out_rate)
        log(f"  ch{c}: rms {in_rms:.6f} -> {out_rms:.6f}  "
            f"({20 * np.log10((out_rms + 1e-12) / (in_rms + 1e-12)):+.1f} dB)")
        log(f"       input peaks  {in_peaks} Hz")
        log(f"       output peaks {out_peaks} Hz")
        # The tones are at 220/1320 (L) and 440/2640 (R). A correct round trip
        # should put energy near those; a smeared window spreads it elsewhere.
        expect = [220.0, 1320.0] if c == 0 else [440.0, 2640.0]
        near = [f for f in expect if any(abs(p - f) < max(60.0, 0.15 * f) for p in out_peaks)]
        log(f"       tones recovered: {near} of {expect}")
        if not near:
            ok = False
        if out_rms < in_rms * 0.02:
            log("       *** output is >34 dB below input -- reconstruction failed ***")
            ok = False
    log(f"\nH17 VERDICT: {'PASS -- the constructed mel reconstructs' if ok else 'FAIL -- see TEST_PLAN A.5.1 candidate order: log/floor -> mel_scale/norm -> power -> center'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
