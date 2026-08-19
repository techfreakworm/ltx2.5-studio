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
from pathlib import Path

import numpy as np
import torch

from ltx25_studio import audio as audio_module
from ltx25_studio import pipelines

GiB = 1024**3
SECONDS = float(os.environ.get("MEL_SECONDS", "4.0"))
SWAP = os.environ.get("MEL_SWAP") == "1"
FORCE_DEVICE = os.environ.get("MEL_FORCE_DEVICE") == "1"
#: (fundamental, harmonic) per channel. The harmonic is the one at risk.
TONES_L = (220.0, 1320.0)
TONES_R = (440.0, 2640.0)
#: Every expected tone must be recovered. The previous criterion was ">=1 tone per
#: channel", under which 3 of 4 tones passed BY DESIGN -- a threshold that reports
#: PASS while a quarter of the signal is missing is the kind of green result this
#: build keeps producing.
REQUIRE_ALL_TONES = os.environ.get("MEL_REQUIRE_ALL", "1") == "1"


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
    lo, hi = TONES_L, TONES_R
    if SWAP:
        # The discriminator: put the HIGH tone on the other channel. If the high
        # tone is lost whichever channel carries it, the loss is frequency-dependent
        # and benign. If one channel always loses its high tone, it is
        # channel-dependent and worth chasing. One run separates the two.
        lo, hi = (TONES_L[0], TONES_R[1]), (TONES_R[0], TONES_L[1])
    left = envelope * (0.7 * np.sin(2 * np.pi * lo[0] * t) + 0.3 * np.sin(2 * np.pi * lo[1] * t))
    right = envelope * (0.7 * np.sin(2 * np.pi * hi[0] * t) + 0.3 * np.sin(2 * np.pi * hi[1] * t))
    log(f"  tones: ch0 {lo} Hz   ch1 {hi} Hz" + ("   (SWAPPED)" if SWAP else ""))
    return torch.from_numpy(np.stack([left, right])[:channels]), (lo, hi)


def write_wav(path, wave, rate):
    """Persist the fixture so `load_waveform` can be the thing under test.

    Reading the fixture back through the app's own loader means the round trip
    starts where a user's upload starts -- including its device and dtype, which is
    where this harness previously diverged from the application and hid a defect.
    """
    import av
    import numpy as np

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    planes = wave.detach().cpu().numpy()
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("pcm_s16le", rate=rate)
        stream.layout = "stereo" if planes.shape[0] == 2 else "mono"
        # s16 is PACKED: one row of interleaved samples, not (channels, n) planar.
        interleaved = np.clip(planes.T.reshape(1, -1), -1.0, 1.0)
        frame = av.AudioFrame.from_ndarray(
            (interleaved * 32767).astype("int16"), format="s16", layout=stream.layout.name
        )
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


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


def band_energy(x, rate, centre, n_fft=4096, width_hz=80.0):
    """Energy in a narrow band around `centre`, as a fraction of total.

    This replaces top-N peak picking, which was the wrong instrument: it reported
    only the four largest peaks, so a real tone sitting fifth was indistinguishable
    from a tone that was not there. Measuring the band directly answers "is there
    energy at this frequency" without depending on what else is louder.

    A larger n_fft than the display spectrogram, because resolving 80 Hz around
    2640 Hz needs it.
    """
    prof = spectral_profile(x, rate, n_fft=n_fft)
    if prof is None:
        return 0.0
    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
    mask = np.abs(freqs - centre) <= width_hz
    total = float(prof.sum()) or 1.0
    return float(prof[mask].sum()) / total


def peak_bins(profile, rate, n_fft=1024, count=4):
    """Where the energy actually is, in Hz. A smeared window moves these."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
    idx = np.argsort(profile)[::-1][:count]
    return sorted(round(float(freqs[i]), 1) for i in idx)


def ab_transpose(vae, vocoder, wave, rate, tones, src):
    """Round-trip WITH and WITHOUT `unpacked_latents`' transpose, in one process.

    Comparing across commits proves a difference exists somewhere between them.
    Running both variants side by side in one process, over one input, with one
    line as the only difference, proves which line. That is the difference between
    evidence of a regression and evidence of its cause.

    Every step is the app's own helper except the transpose itself, so this is not
    a reimplementation of the path -- it is the path, forked at one point.
    """
    import torch

    log("\n=== A/B: the ONLY difference is unpacked_latents' transpose ===")
    prepared, note = audio_module.match_channel_count(
        wave, channels=audio_module.audio_channels_from(vae)
    )
    if note:
        log(f"  {note}")
    mel = audio_module.build_mel_front_end(vae, vocoder)(prepared)
    mel = mel.unsqueeze(0) if mel.dim() == 3 else mel
    mel = audio_module._matched_to_module(mel, vae, name="audio_vae")
    raw = audio_module._sampled(vae.encode(mel))
    log(f"  encode output (before any transpose): {tuple(raw.shape)}")

    results = {}
    for label, latents in (
        ("WITHOUT transpose", raw),
        ("WITH transpose (current tree)", audio_module.unpacked_latents(raw)),
    ):
        log(f"\n  --- {label} --- latents {tuple(latents.shape)}")
        try:
            audio_module.validate_audio_latents_shape(tuple(latents.shape))
            log("      validate_audio_latents_shape: accepted")
        except Exception as exc:
            log(f"      validate_audio_latents_shape REJECTED: {exc}")
            continue
        out, out_rate = audio_module.decode_latents_to_waveform(
            latents, audio_vae=vae, vocoder=vocoder
        )
        rec = out.detach().float().cpu().numpy()
        while rec.ndim > 2:
            rec = rec[0]
        seconds = rec.shape[1] / out_rate
        log(f"      reconstruction {rec.shape} @ {out_rate} = {seconds:.3f} s "
            f"(input {src.shape[1] / rate:.3f} s)")
        retained = []
        for c in range(min(src.shape[0], rec.shape[0])):
            for f in (tones[0] if c == 0 else tones[1]):
                e_in = band_energy(src[c], rate, f)
                e_out = band_energy(rec[c], out_rate, f)
                pct = 100 * (e_out / e_in if e_in else 0)
                retained.append((c, f, pct))
                log(f"      ch{c} {f:7.1f} Hz: {pct:5.1f}% retained"
                    f"  {'RECOVERED' if pct > 10 else 'LOST'}")
        results[label] = (seconds, retained)

    log("\n  VERDICT:")
    for label, (seconds, retained) in results.items():
        good = sum(1 for _, _, pct in retained if pct > 10)
        log(f"    {label:<30} {seconds:.3f}s, {good}/{len(retained)} tones recovered")
    return results


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
    wave, tones = structured_waveform(rate, channels)
    log(f"\nfixture: {tuple(wave.shape)} @ {rate} Hz, {channels} ch, "
        f"two tones + envelope per channel")
    src = wave.numpy()
    for c in range(src.shape[0]):
        prof = spectral_profile(src[c], rate)
        log(f"  ch{c}: rms {float(np.sqrt((src[c] ** 2).mean())):.6f}  "
            f"peaks at {peak_bins(prof, rate)} Hz")

    # Write it out and read it back through `load_waveform`, so the input is
    # LITERALLY what the app receives from a user upload -- same decoder, same
    # dtype, same device (CPU).
    #
    # The previous version did `wave.to(device=vae.device, dtype=vae.dtype)` here,
    # and that is how this harness concealed a real defect: the app does not move
    # the waveform, so `encode_waveform_to_latents` hands a CPU mel to CUDA weights
    # and raises -- while this test, having moved it, passed. A harness that
    # performs a step the application omits is the same failure as a float32
    # fixture over a bfloat16 bug, and it was mine.
    fixture_path = os.environ.get(
        "MEL_FIXTURE", "/home/wakeuser/ltx25-studio/mel-fixture.wav"
    )
    write_wav(fixture_path, wave, rate)
    loaded, loaded_rate = audio_module.load_waveform(fixture_path)
    log(f"\nread back via load_waveform: {tuple(loaded.shape)} @ {loaded_rate} Hz "
        f"dtype={loaded.dtype} device={loaded.device}")
    dev_wave = loaded
    if FORCE_DEVICE:
        # Moves the waveform to the VAE's device, which the APP DOES NOT DO -- see
        # the CPU/CUDA defect at `encode_waveform_to_latents`. Off by default for
        # exactly that reason.
        #
        # It is offered because device placement does not change arithmetic: the
        # mel, the VAE and the vocoder compute the same values wherever the tensor
        # lives. So this variant can answer a question about FREQUENCY CONTENT --
        # which tones survive the round trip -- while the app path is blocked. It
        # answers nothing about whether a user's upload works, and a run using it
        # must not be reported as an app-path pass.
        dev_wave = loaded.to(device=vae.device)
        log(f"  MEL_FORCE_DEVICE=1: moved to {dev_wave.device} "
            f"(answers frequency content only, NOT the app path)")

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
        # Band energy at each expected tone, in and out, rather than top-N peaks.
        expect = list(tones[0] if c == 0 else tones[1])
        recovered = []
        for f in expect:
            e_in = band_energy(src[c], rate, f)
            e_out = band_energy(rec[c], out_rate, f)
            # Relative to the input's own concentration at that tone: a harmonic
            # carried at 0.3 amplitude should not be held to the fundamental's bar.
            ratio = e_out / e_in if e_in > 0 else 0.0
            got = ratio > 0.10
            recovered.append(f) if got else None
            log(f"       {f:7.1f} Hz: band energy in {e_in:.5f} -> out {e_out:.5f} "
                f"({ratio * 100:5.1f}% retained)  {'RECOVERED' if got else 'LOST'}")
        log(f"       tones recovered: {recovered} of {expect}")
        if REQUIRE_ALL_TONES and len(recovered) < len(expect):
            ok = False
        elif not recovered:
            ok = False
        if out_rms < in_rms * 0.02:
            log("       *** output is >34 dB below input -- reconstruction failed ***")
            ok = False
    log(f"  criterion: {'ALL tones must be recovered' if REQUIRE_ALL_TONES else '>=1 tone per channel'}")
    if os.environ.get("MEL_AB") == "1":
        ab_transpose(vae, vocoder, dev_wave, rate, tones, src)

    log(f"\nH17 VERDICT: {'PASS -- the constructed mel reconstructs' if ok else 'FAIL -- see TEST_PLAN A.5.1 candidate order: log/floor -> mel_scale/norm -> power -> center'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
