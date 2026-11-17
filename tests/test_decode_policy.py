"""Spec for the decode policy — which decoder finishes the frames (F07).

``auto | diffusion | conv``, from the request, else ``LTX25_DECODER``, default
``auto``. ``auto`` is the diffusion decoder (the model's native one) where it can
run -- CUDA, its kernel built, the video at least its minimum -- and the conv
VAE otherwise, with the reason in the run's notes. An explicit ``diffusion`` that
cannot run raises; it is never quietly downgraded. Stage 2 always finishes with
the diffusion decoder; the modes that discard their video always use conv.
"""

from __future__ import annotations

import pytest
from conftest import inputs_for
from test_stage_graph_contract import (
    REAL_DECODER_CONFIG,
    AudioPipeline,
    FakeTensor,
    Recorder,
    Staged,
    _ConfiguredDecoder,
)

from ltx25_studio import config, modes, pipelines
from ltx25_studio import generate as generate_module
from ltx25_studio.errors import (
    ConstraintError,
    DecoderTooSmallError,
    DecoderUnavailableError,
    ModelLoadError,
)

#: Large enough for the pinned decoder (224x224x17), small enough to name.
OK = {"width": 256, "height": 256, "num_frames": 17}
SMALL = {"width": 128, "height": 128, "num_frames": 9}


class _Clip(FakeTensor):
    """Frames with a length, as ``generate._decoded`` asks of a real output."""

    def __len__(self):
        return 1


#: The tiny decoder's kernels (scripts/local_smoke.TINY_STAGE_2): minimum 128x128x9.
TINY_DECODER_CONFIG = {
    **REAL_DECODER_CONFIG,
    "decoder_stage_kernels": [[3, 3, 3], [3, 7, 7], [3, 5, 5], [3, 5, 5]],
    "decoder_stage5_kernel": [9, 11, 11],
}


def runner_on(device, *, decoder="ok", decoder_config=REAL_DECODER_CONFIG):
    """A StageRunner over a fake pipeline; ``decoder`` is "ok", "broken" or "none"."""
    pipeline = AudioPipeline()
    pipeline._output = Staged(_Clip("video-latents"), audio=FakeTensor("audio-latents"))
    built = []

    def decoder_factory():
        built.append("decoder")
        if decoder == "broken":
            raise ModelLoadError(component="diffusion_decoder", reason="no NATTEN build")
        made = _ConfiguredDecoder(decoder_config)
        made._output = Staged(_Clip("decoded"))
        return made

    def stage_2_factory():
        built.append("upsampler")
        return Recorder("upsampler", output=Staged("up")), decoder_factory()

    runner = pipelines.StageRunner(
        pipeline,
        kind="text2video",
        device=device,
        decoder_factory=None if decoder == "none" else decoder_factory,
        stage_2_factory=stage_2_factory,
    )
    return runner, pipeline, built


# --------------------------------------------------------------------------
# the resolution table
# --------------------------------------------------------------------------

TABLE = [
    # policy, device, decoder, size, decoder that ran, note fragment
    ("conv", "cuda", "ok", OK, "conv", None),
    ("auto", "cuda", "ok", OK, "diffusion", None),
    ("auto", "cuda", "broken", OK, "conv", "could not be loaded"),
    ("auto", "cuda", "ok", SMALL, "conv", "smaller than the diffusion decoder's minimum"),
    ("auto", "mps", "ok", OK, "conv", "NVIDIA"),
    ("auto", "cpu", "ok", OK, "conv", "NVIDIA"),
    ("diffusion", "cuda", "ok", OK, "diffusion", None),
]


@pytest.mark.parametrize("policy,device,decoder,size,ran,note", TABLE)
def test_the_policy_resolves_per_run(policy, device, decoder, size, ran, note):
    runner, pipeline, built = runner_on(device, decoder=decoder)

    runner(prompt="a lighthouse", video_decoder=policy, **size)

    facts = runner.last_run
    assert facts["decoder"] == ran and facts["stage_2"] is False, facts
    assert len(pipeline.calls) == 1
    if ran == "diffusion":
        assert pipeline.kwargs["output_type"] == "latent"
        assert len(runner.decoder.calls) == 1
        assert runner.decoder.kwargs["denormalize"] is False
    else:
        assert pipeline.kwargs.get("output_type") != "latent"
    if note is None:
        assert facts["why_conv"] is None
    else:
        assert note in facts["why_conv"], facts
    assert "upsampler" not in built, "a single-stage run built the latent upsampler"


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_auto_and_conv_off_cuda_never_build_the_decoder(device):
    for policy in ("auto", "conv"):
        if device == "cuda" and policy == "auto":
            continue
        runner, _pipeline, built = runner_on(device)
        runner(prompt="a", video_decoder=policy, **OK)
        assert built == [], (policy, device, built)


def test_an_explicit_diffusion_that_cannot_load_raises():
    runner, pipeline, _built = runner_on("cuda", decoder="broken")

    with pytest.raises(DecoderUnavailableError) as caught:
        runner(prompt="a", video_decoder="diffusion", **OK)
    assert "could not be loaded" in str(caught.value)
    assert not pipeline.calls


def test_an_explicit_diffusion_below_the_minimum_raises_naming_the_smallest_request():
    runner, pipeline, _built = runner_on("cuda")

    with pytest.raises(DecoderTooSmallError) as caught:
        runner(prompt="a", video_decoder="diffusion", **SMALL)
    assert caught.value.smallest == (224, 224, 17) and caught.value.stage_2 is False
    assert "choose Sharper (standard decoder)" in str(caught.value)
    assert not pipeline.calls


def test_an_explicit_diffusion_off_cuda_runs_only_where_the_cpu_decoder_fits():
    runner, _pipeline, _built = runner_on("cpu", decoder_config=TINY_DECODER_CONFIG)
    small_ok = {"width": 128, "height": 128, "num_frames": 9}  # 9216 positions: the limit
    runner(prompt="a", video_decoder="diffusion", **small_ok)
    assert runner.last_run["decoder"] == "diffusion"

    runner, pipeline, _built = runner_on("cpu")
    with pytest.raises(DecoderUnavailableError) as caught:
        runner(prompt="a", video_decoder="diffusion", **OK)
    assert "CUDA" in str(caught.value)
    assert not pipeline.calls


def test_an_unknown_policy_is_refused():
    runner, _pipeline, _built = runner_on("cuda")
    with pytest.raises(ConstraintError):
        runner(prompt="a", video_decoder="vae", **OK)


def test_stage_2_always_finishes_with_the_diffusion_decoder():
    runner, _pipeline, _built = runner_on("cuda")

    runner(prompt="a", video_decoder="conv", enable_stage_2=True, **OK)

    assert runner.last_run["decoder"] == "diffusion" and runner.last_run["stage_2"] is True


def test_a_diffusion_decoded_audio_run_keeps_its_audio():
    runner, pipeline, _built = runner_on("cuda")

    result = runner(prompt="a", **OK)

    assert result.audio == FakeTensor("waveform<-mel<-audio-latents@bfloat16")
    assert result.frames == FakeTensor("decoded")


# --------------------------------------------------------------------------
# through generate(): where the policy comes from, and the note
# --------------------------------------------------------------------------


def _generate(mode_id, tmp_path, runner, **overrides):
    from test_exports_contract import one_muxed_file_encoder

    return generate_module.generate(
        mode=mode_id,
        seed=7,
        **{**inputs_for(mode_id), **overrides},
        pipeline_factory=lambda **_: runner,
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )


class _Recording(Recorder):
    def __init__(self):
        from conftest import FakeAudioVae

        super().__init__("pipeline", output=Staged(_Clip("frames")))
        self.audio_vae = FakeAudioVae()


def test_the_request_wins_over_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_DECODER, "diffusion")
    pipe = _Recording()
    _generate("text_to_video", tmp_path, pipe, decoder="conv")
    assert pipe.kwargs["video_decoder"] == "conv"


def test_the_environment_sets_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_DECODER, "conv")
    pipe = _Recording()
    _generate("text_to_video", tmp_path, pipe)
    assert pipe.kwargs["video_decoder"] == "conv"


def test_auto_is_sent_as_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(config.ENV_DECODER, raising=False)
    pipe = _Recording()
    _generate("text_to_video", tmp_path, pipe)
    assert "video_decoder" not in pipe.kwargs


@pytest.mark.usefixtures("audio_input_ready")
@pytest.mark.parametrize("mode_id", [m for m, s in modes.MODES.items() if "video" not in s.exports])
def test_a_mode_that_discards_its_video_always_decodes_with_conv(mode_id, tmp_path):
    pipe = _Recording()
    _generate(mode_id, tmp_path, pipe, decoder="diffusion")
    assert pipe.kwargs["video_decoder"] == "conv"


def test_an_unknown_request_decoder_is_refused_before_dispatch():
    with pytest.raises(ConstraintError):
        generate_module.prepare_request(
            generate_module.GenerationRequest(mode="text_to_video", prompt="p", decoder="vae")
        )


@pytest.mark.parametrize(
    "device,ran",
    [("cuda", "model's diffusion decoder"), ("cpu", "standard (convolutional) decoder")],
)
def test_the_diagnostics_name_the_decoder_that_ran(device, ran, tmp_path):
    """Decoder policy is a diagnostic (UI review 2, B-1); the record says which ran."""
    runner, _pipeline, _built = runner_on(device)

    result = _generate("text_to_video", tmp_path, runner, **OK)

    diagnostics = result.record.diagnostics
    assert any(f"decoded by the {ran}" in line for line in diagnostics), diagnostics
    assert not any("decoded by the" in note for note in result.warnings), result.warnings


# --------------------------------------------------------------------------
# the real tiny pipelines (CPU): a single-stage diffusion decode
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from test_real_pipeline_tiny import _tiny_snapshot_cached, load_smoke

    if not _tiny_snapshot_cached():
        pytest.skip("the tiny checkpoint is not in the HF cache")
    return load_smoke().tiny_stack(device="cpu")


@pytest.mark.parametrize("mode_id", ["text_to_audio_video", "image_to_video", "audio_to_video"])
def test_a_single_stage_run_decodes_with_the_tiny_diffusion_decoder(tiny, mode_id, tmp_path):
    """128x128x9 is the tiny decoder's minimum and the CPU decoder's limit."""
    from test_real_pipeline_tiny import run_tiny
    from verify_streams import probe_streams

    decoder = tiny["condition"].decoder
    calls = []
    original = type(decoder).__call__

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    type(decoder).__call__ = spy
    try:
        result = run_tiny(
            tiny, mode_id, tmp_path, width=128, height=128, num_frames=9, decoder="diffusion"
        )
    finally:
        type(decoder).__call__ = original

    assert len(calls) == 1 and calls[0]["denormalize"] is False
    runner = tiny[modes.get_mode(mode_id).pipeline]
    assert runner.last_run["decoder"] == "diffusion"
    actual = probe_streams(result.output_path)
    assert (actual["width"], actual["height"]) == (128, 128), actual
    assert actual["audio"] == ("audio" in modes.get_mode(mode_id).exports), actual
    assert result.record.decoder == "diffusion", result.record
    assert any("diffusion decoder" in line for line in result.record.diagnostics), (
        result.record.diagnostics
    )


def test_a_single_stage_diffusion_decode_is_labelled_and_the_bar_never_retreats():
    labels, fractions = [], []

    def handle(position, desc=None):
        labels.append(desc)
        if isinstance(position, (int, float)):
            fractions.append(position)

    runner, _pipeline, _built = runner_on("cuda")
    reporter = generate_module.ProgressReporter(handle)

    runner(prompt="a", progress=reporter, **OK)

    assert labels.index("sampling") < labels.index("decoding the frames"), labels
    assert fractions == sorted(fractions), fractions


# --------------------------------------------------------------------------
# refused before dispatch: an explicit diffusion decode, or stage 2, the decoder
# cannot take (the decoder's minimum; off CUDA, the CPU path's size)
# --------------------------------------------------------------------------


@pytest.fixture
def pinned_decoder(monkeypatch, tmp_path):
    """The pinned decoder's config.json where the check looks for it."""
    import json

    from ltx25_studio import preflight

    monkeypatch.delenv(config.ENV_MODEL_SNAPSHOT_DIR, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    target = preflight.component_dir("diffusion_decoder") / "config.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(REAL_DECODER_CONFIG))


def _on(monkeypatch, name):
    import dataclasses

    from ltx25_studio import device

    resolution = dataclasses.replace(device.resolve(None), device=name)
    monkeypatch.setattr(device, "current", lambda: resolution)


def _dispatched(monkeypatch):
    calls = []
    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", lambda **kw: calls.append(kw))
    return calls


@pytest.mark.usefixtures("pinned_decoder")
def test_an_explicit_diffusion_below_the_minimum_is_refused_before_dispatch(monkeypatch):
    from ltx25_studio.errors import categorise

    _on(monkeypatch, "cuda")
    calls = _dispatched(monkeypatch)

    with pytest.raises(DecoderTooSmallError) as caught:
        generate_module.generate(
            mode="text_to_video", prompt="p", decoder="diffusion", **SMALL, seed=7
        )
    assert not calls
    assert categorise(caught.value).gpu_used is False


@pytest.mark.usefixtures("pinned_decoder")
def test_stage_2_below_the_decoders_minimum_is_refused_before_dispatch(monkeypatch):
    _on(monkeypatch, "cuda")
    calls = _dispatched(monkeypatch)

    with pytest.raises(DecoderTooSmallError) as caught:
        generate_module.generate(
            mode="text_to_video",
            prompt="p",
            enable_stage_2=True,
            width=96,
            height=64,
            num_frames=17,
            seed=7,
        )
    assert not calls and caught.value.stage_2 is True


@pytest.mark.usefixtures("pinned_decoder")
def test_an_explicit_diffusion_off_cuda_beyond_the_cpu_path_is_refused_before_dispatch(
    monkeypatch,
):
    _on(monkeypatch, "mps")
    calls = _dispatched(monkeypatch)

    with pytest.raises(DecoderUnavailableError):
        generate_module.prepare_request(
            generate_module.GenerationRequest(
                mode="text_to_video", prompt="p", decoder="diffusion", **OK
            )
        )
    assert not calls


@pytest.mark.usefixtures("pinned_decoder")
@pytest.mark.parametrize("device_name", ["cuda", "mps"])
def test_auto_is_never_refused_before_dispatch(monkeypatch, device_name):
    _on(monkeypatch, device_name)

    generate_module.prepare_request(
        generate_module.GenerationRequest(mode="text_to_video", prompt="p", **SMALL)
    )


@pytest.mark.usefixtures("pinned_decoder")
def test_an_explicit_diffusion_that_fits_passes_the_gate(monkeypatch):
    _on(monkeypatch, "cuda")
    generate_module.prepare_request(
        generate_module.GenerationRequest(
            mode="text_to_video", prompt="p", decoder="diffusion", **OK
        )
    )
