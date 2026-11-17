"""Spec: the model's own sound reaches the writer unclamped (issue #14, follow-up).

LTX-2.5's vocoder is diffusers' ``LTX2VocoderWithBWE``, whose forward ends
``waveform = torch.clamp(residual + skip, -1, 1)``. The model clipped its own
sound before the writer could turn it down: on real weights (mode 05,
704x480x49, seed 7) the wav had two samples at -32768 and no turn-down line,
and deploy 1's 78 pinned samples were this clamp.

Where each runner builds the vocoder (the Space pipeline and the local runner),
its forward becomes the pinned one minus that clamp, every other step and the
trim kept, so the writer's whole-track turn-down does the job. Only when the
installed class's forward is the pinned one (its source hash); otherwise one
line is logged and the vendor forward is left alone.
"""

from __future__ import annotations

import logging
import math

import pytest
from test_media_io import decode_audio
from test_sound_over_full_scale import CLIP_LEVEL

from ltx25_studio import outputs

torch = pytest.importorskip("torch")
vocoder_module = pytest.importorskip("diffusers.pipelines.ltx2.vocoder")
LTX2VocoderWithBWE = vocoder_module.LTX2VocoderWithBWE


def small_vocoder(seed: int = 0):
    """The real class at the tiny checkpoint's widths, randomly initialised."""
    torch.manual_seed(seed)
    return LTX2VocoderWithBWE(hidden_channels=64, bwe_hidden_channels=32).eval()


def a_mel(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 2, 20, 64, generator=generator)


def unclamp(vocoder):
    from ltx25_studio import audio

    return audio.unclamp_vocoder(vocoder)


@torch.no_grad()
def test_in_range_output_is_bit_identical_to_the_vendor_forward():
    vocoder, mel = small_vocoder(), a_mel()
    vendor = vocoder(mel)
    assert 0 < float(vendor.abs().max()) < 1.0, "the case must be in range"

    assert unclamp(vocoder) is True
    assert torch.equal(vocoder(mel), vendor)


@torch.no_grad()
def test_an_over_range_output_comes_out_unclamped():
    vocoder, mel = small_vocoder(), a_mel()
    vocoder.bwe_generator.conv_out.weight.mul_(10.0)
    vendor = vocoder(mel)
    assert float(vendor.abs().max()) == 1.0, "the vendor forward clamps"

    unclamp(vocoder)
    unclamped = vocoder(mel)

    assert float(unclamped.abs().max()) > 1.0
    assert unclamped.shape == vendor.shape
    # The pinned forward minus its last clamp: clamped again, it is the vendor's output.
    assert torch.equal(unclamped.clamp(-1, 1), vendor)


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


class Rewritten(LTX2VocoderWithBWE):
    """A diffusers release whose forward is not the pinned one."""

    def forward(self, mel_spec):
        return super().forward(mel_spec)


def _lines(caplog):
    return [r for r in caplog.records if r.name == "ltx25_studio.audio" and "clamp" in r.getMessage()]


@torch.no_grad()
def test_a_forward_that_is_not_the_pinned_one_is_left_alone_with_one_line(caplog):
    torch.manual_seed(0)
    vocoder = Rewritten(hidden_channels=64, bwe_hidden_channels=32).eval()
    vocoder.bwe_generator.conv_out.weight.mul_(10.0)
    caplog.set_level(logging.INFO, logger="ltx25_studio.audio")

    assert unclamp(vocoder) is False

    assert "forward" not in vocoder.__dict__, "the vendor forward was replaced"
    assert float(vocoder(a_mel()).abs().max()) == 1.0, "the vendor clamp is still there"
    lines = _lines(caplog)
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING, [r.getMessage() for r in lines]


def test_a_forward_someone_else_replaced_is_left_alone_with_one_line(caplog):
    vocoder = small_vocoder()
    wrapped = vocoder.forward
    vocoder.forward = lambda mel_spec: wrapped(mel_spec)
    caplog.set_level(logging.INFO, logger="ltx25_studio.audio")

    assert unclamp(vocoder) is False
    assert len(_lines(caplog)) == 1


def test_lifting_the_clamp_twice_changes_nothing_more(caplog):
    vocoder = small_vocoder()
    caplog.set_level(logging.INFO, logger="ltx25_studio.audio")
    assert unclamp(vocoder) is True
    forward = vocoder.forward

    assert unclamp(vocoder) is True
    assert vocoder.forward == forward
    assert len(_lines(caplog)) == 1, [r.getMessage() for r in _lines(caplog)]


def test_anything_but_this_vocoder_is_not_touched(caplog):
    from types import SimpleNamespace

    caplog.set_level(logging.INFO, logger="ltx25_studio.audio")
    fake = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000))

    assert unclamp(None) is False and unclamp(fake) is False
    assert not hasattr(fake, "forward") and _lines(caplog) == []


# --------------------------------------------------------------------------
# both runners build it that way, and a loud run is turned down, not clipped
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_composite_dir(tmp_path_factory):
    from test_decoder_at_startup import _tiny_composite

    return _tiny_composite(tmp_path_factory)


@pytest.fixture
def space_runner(tiny_composite_dir, monkeypatch):
    """``pipelines.get_pipeline`` as the Space builds it, on the tiny checkpoint, on the CPU."""
    from ltx25_studio import config, pipelines

    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tiny_composite_dir))
    monkeypatch.setenv(config.ENV_DEVICE, "cpu")
    pipelines.clear_cache()
    try:
        yield lambda **_: pipelines.get_pipeline(pipeline="text2video", device="cpu")
    finally:
        pipelines.clear_cache()


@pytest.fixture
def local_runner():
    """The local runner on the tiny checkpoint, on the CPU (kept per process, as in the app)."""
    from test_local_runtime import _factory

    factory = _factory("resident")
    return lambda **_: factory(pipeline="text2video")


@pytest.fixture(params=["space", "local"])
def runner_factory(request):
    return request.getfixturevalue(f"{request.param}_runner")


def test_each_runner_builds_its_vocoder_without_the_clamp(runner_factory):
    from ltx25_studio import audio

    vocoder = runner_factory().pipeline.vocoder

    assert isinstance(vocoder, LTX2VocoderWithBWE)
    assert getattr(vocoder.forward, "__func__", None) is audio.forward_without_final_clamp


@pytest.fixture
def loud_vocoder(runner_factory):
    """The runner's vocoder, its BWE residual carrying a 220 Hz line peaking at 1.6."""

    def louder(_module, _inputs, residual):
        t = torch.arange(residual.shape[-1], dtype=residual.dtype) / 48000.0
        return residual + 1.6 * torch.sin(2 * math.pi * 220.0 * t)

    vocoder = runner_factory().pipeline.vocoder
    handle = vocoder.bwe_generator.register_forward_hook(louder)
    try:
        yield runner_factory
    finally:
        handle.remove()


@pytest.fixture
def encoded(monkeypatch):
    """Every float track handed to an audio encoder: the mp4's, then the wav's."""
    import numpy as np

    seen = []
    real = outputs._encode_audio

    def spy(container, stream, planes, *, sampling_rate):
        seen.append(np.array(planes, copy=True))
        return real(container, stream, planes, sampling_rate=sampling_rate)

    monkeypatch.setattr(outputs, "_encode_audio", spy)
    return seen


def test_a_loud_run_is_turned_down_in_both_files_and_says_by_how_much(
    loud_vocoder, encoded, tmp_path
):
    import numpy as np

    from ltx25_studio import generate as generate_module

    token = generate_module._IN_GPU_WORKER.set(True)
    try:
        result = generate_module.generate(
            mode="text_to_audio_video",
            prompt="a man shouts across a canyon",
            seed=7,
            width=64,
            height=64,
            num_frames=9,
            decoder="conv",
            pipeline_factory=loud_vocoder,
            output_root=tmp_path,
            retention_bytes=10**9,
        )
    finally:
        generate_module._IN_GPU_WORKER.reset(token)

    _, _, _, samples = decode_audio(result.audio_path)
    clipped = int(np.count_nonzero(np.abs(samples) >= CLIP_LEVEL))
    assert clipped == 0, f"{clipped} samples of the wav sit at full scale"
    muxed, wav = encoded
    assert np.array_equal(muxed, wav), "the mp4 does not carry the wav's samples"
    lines = [line for line in result.record.diagnostics if "over full scale" in line]
    assert len(lines) == 1 and " dB " in lines[0], result.record.diagnostics
