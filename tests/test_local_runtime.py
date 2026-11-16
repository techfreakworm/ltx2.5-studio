"""Spec: the local (mps/cpu) runner — staged residency, GGUF transformer, RoPE, CRF.

Issue F21. Every behavioural test here runs the REAL diffusers classes on the
tiny random-weight LTX-2.5 checkpoint already in the HF cache, through the same
seam the app uses: ``generate()`` with the runner injected as its pipeline
factory. Nothing is downloaded and nothing larger than the tiny checkpoint is
ever loaded.

The central claim: a staged run (text encoder loaded, used and freed before the
transformer is loaded) produces exactly what a resident run (everything loaded,
the pipeline encoding its own prompt) produces.
"""

from __future__ import annotations

import gc
import itertools
import json
from pathlib import Path

import pytest
from local_fakes import (
    RecordingEncoder,
    as_array,
    install_tiny_audio_encode_shim,
    tiny_snapshot,
    transformer_inputs,
    write_faithful_gguf,
)

from ltx25_studio import local_runtime, memory
from ltx25_studio.errors import ConstraintError, LTXStudioError, ModelLoadError

SIZE = 64
FRAMES = 9
SEED = 1234
PROMPT = "a red ball bouncing on a wooden floor"


@pytest.fixture(autouse=True)
def _fresh_runtime():
    local_runtime.clear_local_cache()
    yield
    local_runtime.clear_local_cache()
    gc.collect()


@pytest.fixture(autouse=True)
def _no_limit_surprises(monkeypatch):
    """The tiny runs need well under a GB; the gauge itself stays live."""
    monkeypatch.delenv("LTX25_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("LTX25_LOAD_SPIKE_FACTOR", raising=False)
    monkeypatch.delenv("LTX25_ACTIVATION_CALIBRATION", raising=False)


def _torch():
    return pytest.importorskip("torch")


def _runner(kind, residency, *, device="cpu", dtype="float32", log=None, **extra):
    _torch()
    pytest.importorskip("diffusers")
    return local_runtime.build_local_runner(
        kind=kind,
        device=device,
        dtype=dtype,
        snapshot_dir=tiny_snapshot(),
        residency=residency,
        log=log,
        **extra,
    )


def _factory(residency, *, device="cpu", dtype="float32", log=None, shim_audio=False):
    def factory(*, pipeline, **_):
        runner = _runner(pipeline, residency, device=device, dtype=dtype, log=log)
        if shim_audio:
            install_tiny_audio_encode_shim(runner.pipeline.audio_vae)
        return runner

    return factory


def _inputs(tmp_path):
    """A real image, a real 9-frame mp4 and a real WAV, written to tmp_path."""
    from fractions import Fraction

    import av
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8))
    video = tmp_path / "in.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("h264", rate=24)
        stream.width = stream.height = SIZE
        stream.pix_fmt = "yuv420p"
        for index in range(FRAMES):
            frame = av.VideoFrame.from_ndarray(
                rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    wav = tmp_path / "in.wav"
    rate = 16000
    samples = int(rate * FRAMES / 24.0) + 1
    tone = (np.sin(np.linspace(0, 200 * np.pi, samples)) * 8000).astype(np.int16)
    with av.open(str(wav), "w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout="stereo")
        frame = av.AudioFrame.from_ndarray(
            np.stack([tone, tone], axis=1).reshape(1, -1), format="s16", layout="stereo"
        )
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return {"image": image, "video": str(video), "audio": str(wav)}


def _generate(residency, mode, tmp_path, *, progress=None, shim_audio=False, **fields):
    from ltx25_studio import generate as generate_module

    encoder = RecordingEncoder()
    out = tmp_path / f"out-{residency}-{mode}"
    result = generate_module.generate(
        mode=mode,
        seed=SEED,
        width=SIZE,
        height=SIZE,
        num_frames=FRAMES,
        pipeline_factory=_factory(residency, shim_audio=shim_audio),
        encoder=encoder,
        output_root=out,
        retention_bytes=10**12,
        progress=progress,
        **fields,
    )
    assert result.output_path.is_file()
    (output,) = encoder.outputs
    return output


def _assert_same(staged, resident):
    import numpy as np

    for field in ("frames", "audio"):
        a = as_array(getattr(staged, field, None))
        b = as_array(getattr(resident, field, None))
        assert (a is None) == (b is None), field
        if a is None:
            continue
        assert a.shape == b.shape, field
        assert np.max(np.abs(a - b)) <= 1e-6, f"{field} differs by {np.max(np.abs(a - b))}"


# --------------------------------------------------------------------------
# refusals that need no weights
# --------------------------------------------------------------------------


def test_cuda_is_refused_the_space_keeps_its_own_runner(tmp_path):
    with pytest.raises(local_runtime.UnsupportedLocalDeviceError) as caught:
        local_runtime.build_local_runner(
            kind="condition", device="cuda", dtype="bfloat16", snapshot_dir=tmp_path
        )
    assert isinstance(caught.value, LTXStudioError)
    assert "cuda" in str(caught.value).lower()


def test_an_mps_runner_refuses_a_watermark_setting_that_breaks_freeing_memory(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    with pytest.raises(memory.MpsWatermarkError):
        local_runtime.build_local_runner(
            kind="condition", device="mps", dtype="bfloat16", snapshot_dir=tmp_path
        )


def test_an_unknown_residency_is_refused(tmp_path):
    with pytest.raises(ConstraintError, match="residency"):
        local_runtime.build_local_runner(
            kind="condition",
            device="cpu",
            dtype="float32",
            snapshot_dir=tmp_path,
            residency="sometimes",
        )


def test_the_rename_covers_both_prompt_adaln_families():
    state = {
        "prompt_adaln_single.linear.weight": 1,
        "prompt_adaln_single.emb.timestep_embedder.linear_1.bias": 2,
        "audio_prompt_adaln_single.linear.weight": 3,
        "adaln_single.linear.weight": 4,
        "transformer_blocks.0.attn1.to_q.weight": 5,
    }
    renamed, count = local_runtime.rename_prompt_adaln_keys(state)
    assert count == 3
    assert set(renamed) == {
        "prompt_adaln.linear.weight",
        "prompt_adaln.emb.timestep_embedder.linear_1.bias",
        "audio_prompt_adaln.linear.weight",
        "adaln_single.linear.weight",
        "transformer_blocks.0.attn1.to_q.weight",
    }
    assert renamed["audio_prompt_adaln.linear.weight"] == 3


def test_a_missing_file_names_the_exact_download_command(tmp_path):
    snapshot = tmp_path / "hub" / "models--Lightricks--LTX-2.5-Diffusers" / "snapshots" / "871165de"
    (snapshot / "text_encoder").mkdir(parents=True)
    (snapshot / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 10}, "weight_map": {"a": "model-1.safetensors"}})
    )
    gguf = (
        tmp_path
        / "hub"
        / "models--unsloth--LTX-2.5-GGUF"
        / "snapshots"
        / "ac3c540c"
        / "distilled"
        / "x-Q8_0.gguf"
    )
    missing = dict(local_runtime.missing_files(snapshot, transformer_gguf=gguf))
    assert missing[gguf] == (
        "hf download unsloth/LTX-2.5-GGUF distilled/x-Q8_0.gguf --revision ac3c540c"
    )
    shard = snapshot / "text_encoder" / "model-1.safetensors"
    assert missing[shard] == (
        'hf download Lightricks/LTX-2.5-Diffusers --revision 871165de --include "text_encoder/*"'
    )


def test_weight_files_are_found_on_disk_whatever_they_are_called(tmp_path):
    """No checkpoint file name is assumed: the shard index, else any weight file."""
    snapshot = tmp_path / "snapshot"
    configs = {"scheduler": "scheduler_config.json", "tokenizer": "tokenizer_config.json"}
    for component in (*local_runtime.SMALL_SET, "text_encoder", "transformer"):
        (snapshot / component).mkdir(parents=True)
        (snapshot / component / configs.get(component, "config.json")).write_text("{}")
    (snapshot / "model_index.json").write_text("{}")
    (snapshot / "tokenizer" / "tokenizer.json").write_text("{}")
    oddly_named = {
        "vae": "vae-weights.safetensors",
        "audio_vae": "a.safetensors",
        "vocoder": "model.safetensors",
        "duration_head": "head.safetensors",
        "text_encoder": "encoder.safetensors",
        "transformer": "dit.safetensors",
    }
    for component, name in oddly_named.items():
        (snapshot / component / name).write_bytes(b"x" * 100)
    (snapshot / "connectors" / "x.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}})
    )
    for part in ("part-1", "part-2"):
        (snapshot / "connectors" / f"{part}.safetensors").write_bytes(b"y" * 50)

    assert local_runtime.missing_files(snapshot) == []
    sizes = local_runtime.weight_sizes(snapshot)
    assert sizes["vae"] == (100, "files") and sizes["connectors"] == (100, "files")
    assert local_runtime.component_files(snapshot, "vae") == [
        snapshot / "vae" / "vae-weights.safetensors"
    ]
    assert snapshot / "transformer" / "dit.safetensors" in local_runtime.run_files(snapshot)

    (snapshot / "vocoder" / "model.safetensors").unlink()
    (snapshot / "connectors" / "part-2.safetensors").unlink()
    missing = [path for path, _ in local_runtime.missing_files(snapshot)]
    assert missing == [snapshot / "connectors" / "part-2.safetensors", snapshot / "vocoder"]
    assert not any("diffusion_pytorch_model" in str(path) for path in missing)


# --------------------------------------------------------------------------
# staged == resident, through generate()
# --------------------------------------------------------------------------

EQUIVALENCE_CASES = {
    "text": ("text_to_video", lambda i: {"prompt": PROMPT}),
    "image": ("image_text_to_video", lambda i: {"prompt": PROMPT, "image": i["image"]}),
    "video": ("video_to_video", lambda i: {"prompt": PROMPT, "video": i["video"]}),
    "audio_latents": ("audio_to_video", lambda i: {"prompt": PROMPT, "audio": i["audio"]}),
    "text_audio": ("text_to_audio_video", lambda i: {"prompt": PROMPT}),
}


@pytest.mark.parametrize("case", sorted(EQUIVALENCE_CASES))
def test_staged_and_resident_runs_are_identical(case, tmp_path):
    mode, fields_for = EQUIVALENCE_CASES[case]
    fields = fields_for(_inputs(tmp_path))
    shim = case == "audio_latents"
    staged = _generate("staged", mode, tmp_path, shim_audio=shim, **fields)
    local_runtime.clear_local_cache()
    resident = _generate("resident", mode, tmp_path, shim_audio=shim, **fields)
    _assert_same(staged, resident)


def test_the_audio_pipeline_does_not_keep_a_released_transformer_alive(tmp_path):
    """The audio-conditioned derivative shares the staged pair; a release must reach it."""
    fields = {"prompt": PROMPT, "audio": _inputs(tmp_path)["audio"]}
    local_runtime.clear_local_cache()
    _generate("staged", "audio_to_video", tmp_path, shim_audio=True, **fields)
    runner = _runner("condition", "staged")
    derived = runner._audio_pipeline
    assert derived is not None and derived.transformer is not None

    runner._weights.release_transformer()

    assert derived.transformer is None, "the audio pipeline still holds the released transformer"


def test_the_duration_head_chooses_the_same_length_staged_and_resident(tmp_path):
    fields = {"prompt": PROMPT, "auto_duration": True, "min_seconds": 0.3, "max_seconds": 0.7}
    staged = _generate("staged", "text_to_audio_video", tmp_path, **fields)
    local_runtime.clear_local_cache()
    resident = _generate("resident", "text_to_audio_video", tmp_path, **fields)
    _assert_same(staged, resident)


def test_progress_sees_the_same_steps_staged_and_resident(tmp_path):
    counts = {}
    for residency in ("staged", "resident"):
        calls = []
        local_runtime.clear_local_cache()
        _generate(
            residency,
            "text_to_video",
            tmp_path,
            prompt=PROMPT,
            progress=lambda *a, sink=calls, **k: sink.append((a, tuple(sorted(k.items())))),
        )
        counts[residency] = [c for c in calls if "sampling" in repr(c).lower()]
    assert counts["staged"], "no sampling progress reached the handle"
    assert len(counts["staged"]) == len(counts["resident"])


# --------------------------------------------------------------------------
# the residency rule itself
# --------------------------------------------------------------------------


class _Watch:
    """A progress object (generate.ProgressReporter's shape) that inspects the pipeline."""

    def __init__(self):
        self.text_encoder_present_during_steps = []
        self.phases = []

    def phase(self, label):
        self.phases.append(label)

    def sampling(self, *, index=1, passes=1):
        def callback(pipeline, step, timestep, callback_kwargs):
            self.text_encoder_present_during_steps.append(pipeline.text_encoder is not None)
            return callback_kwargs

        return callback


def _watched_encode(runner):
    seen = []
    pipeline = runner.pipeline
    original = pipeline.encode_prompt

    def encode_prompt(*args, **kwargs):
        if kwargs.get("prompt_embeds") is not None:
            # The pipeline's own pass-through of precomputed embeddings: no encoding.
            return original(*args, **kwargs)
        seen.append(
            {
                "transformer_present": pipeline.transformer is not None,
                "text_encoder_present": pipeline.text_encoder is not None,
            }
        )
        return original(*args, **kwargs)

    pipeline.encode_prompt = encode_prompt
    return seen


def test_the_text_encoder_and_the_transformer_are_never_resident_together():
    torch = _torch()
    runner = _runner("text2video", "staged")
    encodes = _watched_encode(runner)
    watch = _Watch()
    for prompt in (PROMPT, "a second, different prompt"):
        runner(
            progress=watch,
            prompt=prompt,
            width=SIZE,
            height=SIZE,
            num_frames=FRAMES,
            generator=torch.Generator().manual_seed(SEED),
            output_type="np",
        )
    assert encodes == [
        {"transformer_present": False, "text_encoder_present": True},
        {"transformer_present": False, "text_encoder_present": True},
    ]
    assert watch.text_encoder_present_during_steps
    assert not any(watch.text_encoder_present_during_steps)
    assert runner.pipeline.text_encoder is None and runner.pipeline.transformer is not None


def test_a_repeated_prompt_skips_the_text_encoder_and_says_so():
    torch = _torch()
    lines = []
    runner = _runner("text2video", "staged", log=lines.append)
    encodes = _watched_encode(runner)
    for _ in range(2):
        runner(
            prompt=PROMPT,
            width=SIZE,
            height=SIZE,
            num_frames=FRAMES,
            generator=torch.Generator().manual_seed(SEED),
            output_type="np",
        )
    assert len(encodes) == 1
    assert any("cache hit" in line for line in lines)
    assert sum("load transformer" in line and "preflight" in line for line in lines) == 1


def test_a_preflight_runs_before_every_heavy_load_and_before_the_denoise():
    torch = _torch()
    lines = []
    runner = _runner("text2video", "staged", log=lines.append)
    runner(
        prompt=PROMPT,
        width=SIZE,
        height=SIZE,
        num_frames=FRAMES,
        generator=torch.Generator().manual_seed(SEED),
        output_type="np",
    )
    preflights = [line for line in lines if line.startswith("preflight[")]
    order = [line.split("]")[0].removeprefix("preflight[") for line in preflights]
    assert order[0] == "load small set"
    assert order[1:] == ["load text encoder", "load transformer", f"denoise {SIZE}x{SIZE}x{FRAMES}"]


def test_a_failing_preflight_refuses_before_loading_anything(monkeypatch):
    _torch()
    runner = _runner("text2video", "staged")
    monkeypatch.setenv("LTX25_MEMORY_LIMIT_GB", "1")
    with pytest.raises(memory.MemoryPreflightError):
        runner(prompt=PROMPT, width=SIZE, height=SIZE, num_frames=FRAMES, output_type="np")
    assert runner.pipeline.text_encoder is None
    assert runner.pipeline.transformer is None


def test_both_kinds_share_one_set_of_weights():
    a = _runner("text2video", "staged")
    b = _runner("condition", "staged")
    assert a is _runner("text2video", "staged")
    assert a.pipeline.vae is b.pipeline.vae
    assert a.pipeline.connectors is b.pipeline.connectors


# --------------------------------------------------------------------------
# file-cache pages: bounded moves and releases
# --------------------------------------------------------------------------


def _local_device():
    return "mps" if _mps_available() else "cpu"


@pytest.mark.parametrize("component", ["transformer", "text_encoder"])
def test_the_bounded_move_is_value_identical_to_a_plain_move(component, monkeypatch):
    torch = _torch()
    pytest.importorskip("diffusers")
    snapshot = tiny_snapshot()
    device = _local_device()
    releases = []
    monkeypatch.setattr(
        memory, "release_file_cache", lambda paths, **_: releases.append(list(paths)) or 0
    )

    def fresh():
        if component == "transformer":
            from diffusers import LTX2VideoTransformer3DModel

            return LTX2VideoTransformer3DModel.from_pretrained(
                snapshot, subfolder="transformer", torch_dtype=torch.float32
            )
        import transformers

        cls = getattr(
            transformers, json.loads((snapshot / "model_index.json").read_text())["text_encoder"][1]
        )
        return cls.from_pretrained(snapshot / "text_encoder", dtype=torch.float32)

    plain = fresh().to(device)
    lines = []
    files = sorted((snapshot / component).glob("*.safetensors"))
    bounded = local_runtime.move_to_device(
        fresh(), device, files=files, label=component, log=lines.append, chunk_bytes=4096
    )
    expected = dict(itertools.chain(plain.named_parameters(), plain.named_buffers()))
    got = dict(itertools.chain(bounded.named_parameters(), bounded.named_buffers()))
    assert set(got) == set(expected)
    for name, tensor in got.items():
        assert tensor.device.type == device, name
        assert tensor.dtype == expected[name].dtype, name
        assert torch.equal(tensor.cpu(), expected[name].cpu()), name
    if device != "cpu":
        assert len(releases) > 1, "a multi-chunk move releases file pages as it goes"
    assert releases[-1] == files
    assert lines[-1] == f"released 0.0 GB of file cache ({component})"


def test_every_staged_component_gives_its_file_pages_back(monkeypatch):
    torch = _torch()
    released = []

    def spy(paths, **_):
        released.append({Path(p).parent.name for p in paths})
        return 0

    monkeypatch.setattr(memory, "release_file_cache", spy)
    lines = []
    runner = _runner("text2video", "staged", log=lines.append)
    for prompt in (PROMPT, "a second, different prompt"):
        runner(
            prompt=prompt,
            width=SIZE,
            height=SIZE,
            num_frames=FRAMES,
            generator=torch.Generator().manual_seed(SEED),
            output_type="np",
        )
    touched = set().union(*released)
    assert {"connectors", "vae", "audio_vae", "vocoder", "duration_head"} <= touched
    assert {"text_encoder", "transformer"} <= touched
    # each text-encoder load is followed by a release, and so is each unload
    assert sum(s == {"text_encoder"} for s in released) >= 4
    assert sum(s == {"transformer"} for s in released) >= 2
    assert "released 0.0 GB of file cache (text_encoder)" in lines


# --------------------------------------------------------------------------
# image-conditioning CRF (recon K H7)
# --------------------------------------------------------------------------


def test_staged_and_resident_runs_recompress_the_image_at_the_same_crf(tmp_path, monkeypatch):
    _torch()
    pytest.importorskip("diffusers")
    from diffusers.pipelines.ltx2 import pipeline_ltx2_condition as condition_module
    from diffusers.pipelines.ltx2.utils import resolve_default_image_crf
    from transformers import AutoConfig

    used = []
    original = condition_module.apply_image_conditioning_crf

    def recording(image, crf):
        used.append(crf)
        return original(image, crf)

    monkeypatch.setattr(condition_module, "apply_image_conditioning_crf", recording)
    image = _inputs(tmp_path)["image"]
    by_residency = {}
    for residency in ("staged", "resident"):
        used.clear()
        local_runtime.clear_local_cache()
        _generate(residency, "image_text_to_video", tmp_path, prompt=PROMPT, image=image)
        by_residency[residency] = list(used)

    text_encoder_config = AutoConfig.from_pretrained(tiny_snapshot() / "text_encoder")
    expected = resolve_default_image_crf(type("TE", (), {"config": text_encoder_config})())
    assert expected == 18, "the tiny checkpoint carries the LTX-2.5 Gemma-4 encoder"
    assert resolve_default_image_crf(None) == 33, "what a staged run would silently fall to"
    assert by_residency["staged"] == by_residency["resident"] == [expected]


# --------------------------------------------------------------------------
# stage 2 and LoRA
# --------------------------------------------------------------------------


def test_stage_2_is_refused_in_plain_language():
    runner = _runner("text2video", "staged")
    with pytest.raises(local_runtime.StageTwoUnavailableError) as caught:
        runner(enable_stage_2=True, prompt=PROMPT, width=SIZE, height=SIZE, num_frames=FRAMES)
    message = str(caught.value)
    assert "CUDA" in message and "Space" in message
    assert runner.pipeline.text_encoder is None and runner.pipeline.transformer is None


def _adapter_config():
    from peft import LoraConfig

    return LoraConfig(r=4, lora_alpha=8, target_modules=["to_q", "to_v"], init_lora_weights=False)


def test_an_attached_adapter_survives_the_transformer_being_released():
    torch = _torch()
    pytest.importorskip("peft")

    def run(runner, prompt):
        return runner(
            prompt=prompt,
            width=SIZE,
            height=SIZE,
            num_frames=FRAMES,
            generator=torch.Generator().manual_seed(SEED),
            output_type="np",
        )

    outputs = {}
    for residency in ("staged", "resident"):
        local_runtime.clear_local_cache()
        runner = _runner("text2video", residency)
        run(runner, "warm up")  # leaves the transformer resident
        torch.manual_seed(0)
        runner.pipeline.transformer.add_adapter(_adapter_config(), adapter_name="user_lora_0")
        runner.pipeline.set_adapters("user_lora_0", 0.7)
        outputs[residency] = run(runner, PROMPT)  # staged: release, encode, reload
        assert "user_lora_0" in runner.pipeline.transformer.peft_config
        runner.pipeline.delete_adapters("user_lora_0")
    _assert_same(outputs["staged"], outputs["resident"])


def test_attaching_an_adapter_needs_the_transformer_loaded_first():
    """A staged runner holds no transformer between loads. Attaching then is refused
    in plain language (a LoRAError, so ``lora/apply.py`` passes it through untouched)
    instead of failing deep in peft; after ``ensure_transformer()`` it attaches."""
    pytest.importorskip("peft")
    torch = _torch()
    from peft.utils import get_peft_model_state_dict

    from ltx25_studio.errors import LoRAError

    donor = _runner("text2video", "resident").pipeline.transformer
    torch.manual_seed(0)
    donor.add_adapter(_adapter_config(), adapter_name="donor")
    weights = {
        f"transformer.{key}": value
        for key, value in get_peft_model_state_dict(donor, adapter_name="donor").items()
    }
    local_runtime.clear_local_cache()
    runner = _runner("text2video", "staged")
    assert runner.pipeline.transformer is None
    with pytest.raises(local_runtime.AdapterNeedsTransformerError) as caught:
        runner.pipeline.load_lora_weights(dict(weights), adapter_name="user_lora_0")
    assert isinstance(caught.value, LoRAError)
    runner.ensure_transformer()
    runner.pipeline.load_lora_weights(dict(weights), adapter_name="user_lora_0")
    assert "user_lora_0" in runner.pipeline.transformer.peft_config


# --------------------------------------------------------------------------
# RoPE precision
# --------------------------------------------------------------------------

SIX_ROPE_MODULES = {
    "connectors.video_connector.rope",
    "connectors.audio_connector.rope",
    "transformer.rope",
    "transformer.audio_rope",
    "transformer.cross_attn_rope",
    "transformer.cross_attn_audio_rope",
}


def test_cpu_keeps_the_checkpoints_double_precision_rope_untouched():
    runner = _runner("text2video", "resident")
    flags = local_runtime.rope_flags(runner.pipeline)
    assert set(flags) == SIX_ROPE_MODULES
    assert all(flags.values())
    assert not any(local_runtime.rope_on_cpu(runner.pipeline).values())


def _mps_available():
    torch = _torch()
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


@pytest.mark.skipif("not _mps_available()", reason="MPS is not available on this host")
def test_mps_keeps_float64_rope_by_computing_all_six_modules_on_the_cpu():
    runner = _runner("text2video", "resident", device="mps", dtype="bfloat16")
    flags = local_runtime.rope_flags(runner.pipeline)
    assert set(flags) == SIX_ROPE_MODULES
    assert all(flags.values()), "the checkpoint's double precision is kept, not switched off"
    assert local_runtime.rope_on_cpu(runner.pipeline) == dict.fromkeys(SIX_ROPE_MODULES, True)


@pytest.mark.skipif("not _mps_available()", reason="MPS is not available on this host")
def test_after_a_staged_reload_the_transformer_still_computes_rope_on_the_cpu():
    """The staged runner reloads the transformer between prompts; the CPU-RoPE
    hooks live on module instances, so the fresh instance must get them again —
    and a forward on MPS must then match the CPU."""
    torch = _torch()
    runner = _runner("text2video", "staged", device="mps", dtype="float32")
    loaded = []
    for prompt in (PROMPT, "a second, different prompt"):
        runner(
            prompt=prompt,
            width=SIZE,
            height=SIZE,
            num_frames=FRAMES,
            generator=torch.Generator().manual_seed(SEED),
            output_type="np",
        )
        loaded.append(runner.pipeline.transformer)
    assert loaded[0] is not loaded[1], "the second prompt released and reloaded the transformer"
    assert local_runtime.rope_on_cpu(runner.pipeline) == dict.fromkeys(SIX_ROPE_MODULES, True)

    reference = local_runtime.load_transformer(tiny_snapshot(), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        on_cpu = reference(**transformer_inputs(reference.config, "cpu", torch.float32))[0]
        on_mps = loaded[1](**transformer_inputs(loaded[1].config, "mps", torch.float32))[0]
    error = float((on_mps.cpu() - on_cpu).abs().max() / on_cpu.abs().max())
    assert error <= 1e-3, f"MPS forward differs from the CPU by {error:.2e} (relative)"


@pytest.mark.skipif("not _mps_available()", reason="MPS is not available on this host")
def test_a_staged_tiny_text_to_video_completes_on_mps_in_bf16(tmp_path):
    from ltx25_studio import generate as generate_module

    result = generate_module.generate(
        mode="text_to_video",
        prompt=PROMPT,
        seed=SEED,
        width=SIZE,
        height=SIZE,
        num_frames=FRAMES,
        pipeline_factory=_factory("staged", device="mps", dtype="bfloat16"),
        output_root=tmp_path / "out",
        retention_bytes=10**12,
    )
    assert result.output_path.is_file()
    runner = local_runtime.build_local_runner(
        kind="text2video", device="mps", dtype="bfloat16", snapshot_dir=tiny_snapshot()
    )
    assert runner.pipeline.device.type == "mps"
    flags = local_runtime.rope_flags(runner.pipeline)
    assert set(flags) == SIX_ROPE_MODULES and all(flags.values())
    assert all(local_runtime.rope_on_cpu(runner.pipeline).values())


# --------------------------------------------------------------------------
# the GGUF transformer (needs the `gguf` package)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def faithful_gguf(tmp_path_factory):
    pytest.importorskip("gguf", reason="the GGUF route needs the `gguf` package")
    target = tmp_path_factory.mktemp("gguf") / "tiny-real-names-Q8_0.gguf"
    reference = write_faithful_gguf(tiny_snapshot(), target)
    return target, reference


def test_the_tiny_gguf_really_uses_the_original_key_names(faithful_gguf):
    from diffusers.models.model_loading_utils import load_gguf_checkpoint

    path, _ = faithful_gguf
    keys = set(load_gguf_checkpoint(str(path)))
    assert any(k.startswith("prompt_adaln_single.") for k in keys)
    assert any(k.startswith("audio_prompt_adaln_single.") for k in keys)
    assert any(k.startswith("adaln_single.") for k in keys)
    assert not any(k.startswith("time_embed.") for k in keys)


def test_without_the_rename_the_gguf_load_is_refused_naming_the_key(faithful_gguf):
    torch = _torch()
    path, _ = faithful_gguf
    with pytest.raises(local_runtime.TransformerLoadError) as caught:
        local_runtime.load_transformer(
            tiny_snapshot(), dtype=torch.bfloat16, device="cpu", gguf=path, rename=False
        )
    assert isinstance(caught.value, ModelLoadError)
    assert "prompt_adaln" in str(caught.value)


def test_with_the_rename_the_gguf_loads_completely_and_runs(faithful_gguf, monkeypatch):
    torch = _torch()
    path, _ = faithful_gguf
    released = []
    monkeypatch.setattr(
        memory, "release_file_cache", lambda paths, **_: released.append(list(paths)) or 0
    )
    lines = []
    model = local_runtime.load_transformer(
        tiny_snapshot(), dtype=torch.bfloat16, device="cpu", gguf=path, log=lines.append
    )
    assert not [n for n, p in model.named_parameters() if p.device.type == "meta"]
    assert any("0 missing, 0 unexpected" in line for line in lines)
    assert any("Q8_0" in line for line in lines), lines
    # once after the read (the loader has copied everything), once after the move
    assert sum(paths == [path] for paths in released) >= 2


def test_a_staged_run_with_the_gguf_transformer_completes(faithful_gguf, tmp_path):
    from ltx25_studio import generate as generate_module

    path, _ = faithful_gguf

    def factory(*, pipeline, **_):
        return local_runtime.build_local_runner(
            kind=pipeline,
            device="cpu",
            dtype="bfloat16",
            snapshot_dir=tiny_snapshot(),
            transformer_gguf=path,
        )

    result = generate_module.generate(
        mode="text_to_video",
        prompt=PROMPT,
        seed=SEED,
        width=SIZE,
        height=SIZE,
        num_frames=FRAMES,
        pipeline_factory=factory,
        output_root=tmp_path / "out",
        retention_bytes=10**12,
    )
    assert result.output_path.is_file()


@pytest.mark.skipif("not _mps_available()", reason="MPS is not available on this host")
def test_the_gguf_transformer_on_mps_computes_its_float64_rope_on_the_cpu(faithful_gguf):
    torch = _torch()
    path, _ = faithful_gguf
    model = local_runtime.load_transformer(
        tiny_snapshot(), dtype=torch.bfloat16, device="mps", gguf=path
    )
    rope = [m for m in model.modules() if hasattr(m, "double_precision")]
    assert len(rope) == 4
    assert all(m.double_precision for m in rope)
    assert all(getattr(m, "_ltx25_rope_cpu", False) for m in rope)
    with torch.no_grad():
        out = model(**transformer_inputs(model.config, "mps", torch.bfloat16))[0]
    assert bool(torch.isfinite(out).all())


# --------------------------------------------------------------------------
# scripts/local_generate.py: exit codes and the dry run
# --------------------------------------------------------------------------


def _script(*args, tmp_path):
    import os
    import subprocess
    import sys

    from conftest import REPO_ROOT

    report = tmp_path / "report.json"
    done = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "local_generate.py"),
            *args,
            "--json",
            str(report),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    data = json.loads(report.read_text()) if report.is_file() else {}
    return done, data


TINY_ARGS = (
    "--mode",
    "text_to_video",
    "--prompt",
    PROMPT,
    "--width",
    "64",
    "--height",
    "64",
    "--num-frames",
    "9",
    "--device",
    "cpu",
    "--dtype",
    "float32",
)


def test_the_dry_run_prints_the_plan_and_the_arithmetic_and_loads_nothing(tmp_path):
    done, data = _script(
        *TINY_ARGS, "--snapshot-dir", str(tiny_snapshot()), "--dry-run", tmp_path=tmp_path
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "nothing was loaded" in done.stdout
    phases = [line.split("]")[0].split("[")[1] for line in data["projected_preflight"]]
    assert phases == ["load small set", "load text encoder", "load transformer", "denoise 64x64x9"]
    assert "phases" not in data and "artifacts" not in data
    assert "file_cache" not in data, "a dry run touches nothing, not even the page cache"
    assert "left alone (dry run)" in done.stdout


def test_a_missing_gguf_exits_2_with_the_download_command(tmp_path):
    gguf = (
        tmp_path
        / "hub"
        / "models--unsloth--LTX-2.5-GGUF"
        / "snapshots"
        / "ac3c540c"
        / "distilled"
        / "x-Q8_0.gguf"
    )
    done, data = _script(
        *TINY_ARGS,
        "--snapshot-dir",
        str(tiny_snapshot()),
        "--transformer-gguf",
        str(gguf),
        tmp_path=tmp_path,
    )
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    command = "hf download unsloth/LTX-2.5-GGUF distilled/x-Q8_0.gguf --revision ac3c540c"
    assert command in done.stdout
    assert data["missing"] == [{"path": str(gguf), "command": command}]


def test_the_plan_shows_the_mps_watermarks_and_a_bad_pair_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    monkeypatch.setenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    done, data = _script(
        *TINY_ARGS, "--snapshot-dir", str(tiny_snapshot()), "--dry-run", tmp_path=tmp_path
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert data["plan"]["mps_watermarks"] == {"high": 0.5, "low": 0.4}
    assert "recommended_max_gb" in data["plan"]

    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO")
    for extra in (("--dry-run",), ()):
        done, data = _script(
            *TINY_ARGS, "--snapshot-dir", str(tiny_snapshot()), *extra, tmp_path=tmp_path
        )
        assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
        assert data["error"]["type"] == "MpsWatermarkError"
        assert "PYTORCH_MPS_LOW_WATERMARK_RATIO" in data["error"]["message"]
        assert not data.get("phases") and "file_cache" not in data


def test_a_memory_refusal_exits_2_before_anything_loads(tmp_path):
    _torch()
    pytest.importorskip("diffusers")
    done, data = _script(
        *TINY_ARGS,
        "--snapshot-dir",
        str(tiny_snapshot()),
        "--memory-limit-gb",
        "1",
        tmp_path=tmp_path,
    )
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert data["error"]["type"] in ("MemoryPreflightError", "ModelLoadError")
    assert "Not enough memory for 'load small set'" in data["error"]["message"]
    assert not data.get("phases")
    assert [entry["when"] for entry in data["file_cache"]] == ["before the run", "at exit"]
    assert all(entry["files"] > 0 for entry in data["file_cache"])
