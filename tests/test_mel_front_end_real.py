"""The forward mel transform against the real audio VAE and vocoder (issue F69).

Loads only the pinned ``audio_vae`` and ``vocoder`` (about 350 MB on disk) from
the local Hugging Face cache, on the CPU in float32, at the revision
``ltx25_studio.model_files`` pins. Never downloads and never loads anything
else. Skips, with the reason, when torch, torchaudio or diffusers is missing,
when either component is not cached, or when
``LTX25_SKIP_REAL_COMPONENT_TESTS=1``.

Two checks, both through the app's own ``encode_waveform_to_latents`` and
``decode_latents_to_waveform``:

- the round trip returns the signal at the level it was given;
- the transform of the vocoder's output matches the mel the vocoder was given.

A transform built with a 512 window fails both, which is what shipped: every
encoded sound reached the model about 9 dB too quiet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from ltx25_studio import audio

SKIP_ENV = "LTX25_SKIP_REAL_COMPONENT_TESTS"

#: The upload: stereo at a common upload rate, so the encode resamples as it does for a user.
RATE = 48000
SECONDS = 2.0
THREADS = 4

LEVEL_TOLERANCE_DB = 1.5
OFFSET_TOLERANCE = 0.3  # natural-log mel units


@dataclass
class Pair:
    torch: object
    torchaudio: object
    audio_vae: object
    vocoder: object
    waveform: object


@dataclass(frozen=True)
class Measured:
    level_db: float
    offset: float


def _skip_unless_available():
    if os.environ.get(SKIP_ENV) == "1":
        pytest.skip(f"{SKIP_ENV}=1")
    torch = pytest.importorskip("torch", reason="needs torch")
    torchaudio = pytest.importorskip("torchaudio", reason="needs torchaudio")
    pytest.importorskip("diffusers", reason="needs diffusers")
    return torch, torchaudio


def _cached_dir(name):
    """The pinned component's folder in the local HF cache, or skip. Never downloads."""
    from ltx25_studio import model_files, preflight

    spec = model_files.get_component(name)
    folder = model_files.component_dir(name, repo_cache=preflight.cache_repo_dir())
    missing = [filename for filename in spec.files if not (folder / filename).is_file()]
    if missing:
        pytest.skip(
            f"the pinned {name} ({spec.repo_id}@{spec.revision[:8]}) is not in the local "
            f"Hugging Face cache: {folder} lacks {', '.join(missing)}. This test never downloads."
        )
    return folder


def _signal(torch):
    """Two tones per channel under a tremolo, plus a pink noise burst inside the VAE's band.

    The noise carries most of the energy and covers 50-7500 Hz, below the audio
    VAE's 8 kHz Nyquist, so the resample to its rate removes nothing. Isolated
    pure tones would not do: even with the 1024 window this vocoder renders them
    11-16 dB low (measured 2026-11-16), which it does not do to speech or noise.
    """
    samples = int(RATE * SECONDS)
    t = torch.arange(samples, dtype=torch.float64) / RATE
    fades = torch.clamp(torch.minimum(t, SECONDS - t) / 0.05, max=1.0)
    tremolo = fades * (0.65 + 0.35 * torch.sin(2 * torch.pi * 1.5 * t))
    left = 0.05 * torch.sin(2 * torch.pi * 440 * t) + 0.03 * torch.sin(2 * torch.pi * 1320 * t)
    right = 0.05 * torch.sin(2 * torch.pi * 660 * t) + 0.03 * torch.sin(2 * torch.pi * 1760 * t)
    tones = torch.stack([left, right]) * tremolo

    generator = torch.Generator().manual_seed(69)
    spectrum = torch.fft.rfft(torch.randn((2, samples), generator=generator, dtype=torch.float64))
    freqs = torch.fft.rfftfreq(samples, 1.0 / RATE)
    spectrum = spectrum / torch.clamp(freqs, min=1.0).sqrt()  # pink: power falls as 1/f
    spectrum[:, (freqs < 50) | (freqs > 7500)] = 0
    noise = torch.fft.irfft(spectrum, n=samples)
    noise = noise / noise.pow(2).mean().sqrt()
    start, stop = int(0.15 * samples), int(0.85 * samples)
    burst = torch.zeros(samples, dtype=torch.float64)
    burst[start:stop] = torch.hann_window(stop - start, periodic=False, dtype=torch.float64)

    return (tones + 0.2 * noise * burst).float()


@pytest.fixture(scope="module")
def pair():
    torch, torchaudio = _skip_unless_available()
    from diffusers import AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    folders = {name: _cached_dir(name) for name in ("audio_vae", "vocoder")}
    threads = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    try:
        loaded = Pair(
            torch=torch,
            torchaudio=torchaudio,
            audio_vae=AutoencoderKLLTX2Audio.from_pretrained(
                folders["audio_vae"], dtype=torch.float32, local_files_only=True
            ).eval(),
            vocoder=LTX2VocoderWithBWE.from_pretrained(
                folders["vocoder"], dtype=torch.float32, local_files_only=True
            ).eval(),
            waveform=_signal(torch),
        )
        yield loaded
    finally:
        torch.set_num_threads(threads)


class _Recording:
    """The real vocoder, keeping the mel it was given."""

    def __init__(self, vocoder):
        self._vocoder = vocoder
        self.config = vocoder.config
        self.given = []

    def __call__(self, mel):
        self.given.append(mel)
        return self._vocoder(mel)


def _rms_db(torch, x):
    return 20 * torch.log10(x.double().pow(2).mean().sqrt()).item()


def _mean_offset(mel, given):
    """Mean of ``mel - given`` in log units, over the frames that carry energy.

    Per mel band over the frames whose mean exceeds the clip's mean minus one,
    then averaged across bands. 0 means the vocoder's output, put through the
    transform, is the mel it was given.
    """
    frames = min(mel.shape[-2], given.shape[-2])
    mel, given = mel[..., :frames, :], given[..., :frames, :]
    loud = given.mean(dim=-1, keepdim=True) > given.mean() - 1.0
    per_band = ((mel - given) * loud).sum(dim=(0, 1, 2)) / loud.sum().clamp(min=1)
    return per_band.mean().item()


def _round_trip(pair) -> Measured:
    """Encode the signal the way ``generate`` does, decode it, and measure both checks."""
    torch = pair.torch
    with torch.inference_mode():
        latents = audio.encode_waveform_to_latents(
            pair.waveform, audio_vae=pair.audio_vae, sampling_rate=RATE, vocoder=pair.vocoder
        )
        vocoder = _Recording(pair.vocoder)
        decoded, rate = audio.decode_latents_to_waveform(
            latents, audio_vae=pair.audio_vae, vocoder=vocoder
        )
        to_mel = audio.mel_front_end_for(pair.audio_vae, pair.vocoder)
        vae_rate = audio.audio_params_from(pair.audio_vae).audio_sampling_rate
        resampled = pair.torchaudio.functional.resample(decoded[0], int(rate), int(vae_rate))
        mel = to_mel(audio.batched(resampled))
    (given,) = vocoder.given
    return Measured(
        level_db=_rms_db(torch, decoded) - _rms_db(torch, pair.waveform),
        offset=_mean_offset(mel, given),
    )


@pytest.fixture(scope="module")
def as_built(pair):
    """Both checks on the front end the app builds."""
    return _round_trip(pair)


@pytest.fixture(scope="module")
def with_512_window(pair):
    """The same round trip with the front end built with n_fft = win_length = 512."""
    real = pair.torchaudio.transforms.MelSpectrogram
    built = []

    def narrowed(**kwargs):
        built.append(real(**{**kwargs, "n_fft": 512, "win_length": 512}))
        return built[-1]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pair.torchaudio.transforms, "MelSpectrogram", narrowed)
        measured = _round_trip(pair)
    assert built, "the 512 arm never built a transform, so it measured the app's own"
    return measured


def test_the_round_trip_returns_the_level_it_was_given(as_built):
    assert abs(as_built.level_db) <= LEVEL_TOLERANCE_DB, (
        f"encode -> decode changed the level by {as_built.level_db:+.1f} dB (allowed "
        f"+/-{LEVEL_TOLERANCE_DB}). A 512 window loses about 9 dB: the mel the audio VAE is "
        f"given is too small, so everything encoded reaches the model too quiet."
    )


def test_the_vocoders_output_maps_back_to_the_mel_it_was_given(as_built):
    assert abs(as_built.offset) <= OFFSET_TOLERANCE, (
        f"the transform of the vocoder's output is {as_built.offset:+.2f} log units away "
        f"from the mel the vocoder was given (allowed +/-{OFFSET_TOLERANCE}). The forward "
        f"transform is not the one the audio VAE and vocoder agree on."
    )


def test_a_512_window_fails_both_checks(with_512_window):
    """The regression guard: the window the app used to take from the vocoder's config."""
    assert abs(with_512_window.level_db) > LEVEL_TOLERANCE_DB, (
        f"with a 512 window the round trip changed the level by only "
        f"{with_512_window.level_db:+.1f} dB, so the level check no longer tells the windows apart"
    )
    assert abs(with_512_window.offset) > OFFSET_TOLERANCE, (
        f"with a 512 window the offset is only {with_512_window.offset:+.2f} log units, so the "
        f"consistency check no longer tells the windows apart"
    )
