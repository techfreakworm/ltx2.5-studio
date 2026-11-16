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
    else:
        # On the CPU at the file's own dtype the weights are its pages (local-runtime-9).
        assert releases == [], "the live weights' pages were released"
        assert lines[-1] == f"kept {component}'s file pages: they are the weights on cpu"


def test_a_converting_cpu_load_still_gives_the_read_pages_back(monkeypatch):
    """Converted copies are not the file: its pages are released as before."""
    torch = _torch()
    pytest.importorskip("diffusers")
    from diffusers import LTX2VideoTransformer3DModel

    releases = []
    monkeypatch.setattr(
        memory, "release_file_cache", lambda paths, **_: releases.append(list(paths)) or 0
    )
    model = LTX2VideoTransformer3DModel.from_pretrained(
        tiny_snapshot(), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    files = sorted((tiny_snapshot() / "transformer").glob("*.safetensors"))
    local_runtime.move_to_device(model, "cpu", files=files, label="transformer")

    assert releases == [files]


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_every_staged_component_gives_its_file_pages_back(monkeypatch, dtype):
    """Each converting load and each unload releases its file's pages.

    At the checkpoint's own dtype (the tiny one is fp32) a CPU load keeps them:
    they are the weights (local-runtime-9). The unloads still release.
    """
    torch = _torch()
    released = []

    def spy(paths, **_):
        released.append({Path(p).parent.name for p in paths})
        return 0

    monkeypatch.setattr(memory, "release_file_cache", spy)
    lines = []
    runner = _runner("text2video", "staged", dtype=dtype, log=lines.append)
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
    assert {"text_encoder", "transformer"} <= touched
    assert "released 0.0 GB of file cache (text_encoder)" in lines
    if dtype == "bfloat16":
        assert {"connectors", "vae", "audio_vae", "vocoder", "duration_head"} <= touched
        # each text-encoder load is followed by a release, and so is each unload
        assert sum(s == {"text_encoder"} for s in released) >= 4
        assert sum(s == {"transformer"} for s in released) >= 2
    else:
        assert sum(s == {"text_encoder"} for s in released) >= 2
        assert "kept text_encoder's file pages: they are the weights on cpu" in lines


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


def test_a_real_size_stage_2_is_refused_before_anything_loads():
    """Off CUDA the stage-2 decoder runs on the CPU, at very small sizes only."""
    from ltx25_studio.device import StageTwoNeedsCudaError

    runner = _runner("text2video", "staged")
    with pytest.raises(StageTwoNeedsCudaError) as caught:
        runner(enable_stage_2=True, prompt=PROMPT, width=704, height=480, num_frames=121)
    assert "CUDA" in str(caught.value)
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


def _quantized_cases():
    from ltx25_studio import audio_conditioning, modes

    cases = [(mode_id, {}) for mode_id in modes.MODES if mode_id != "audio_to_audio"]
    cases += [
        ("audio_to_audio", {"noise_scale": level}) for level in audio_conditioning.VARIATION_LEVELS
    ]
    return cases


@pytest.mark.parametrize("mode_id,extra", _quantized_cases())
def test_every_mode_runs_on_the_quantized_runner(
    faithful_gguf, tmp_path, monkeypatch, mode_id, extra
):
    """The real-weights failure: deriving the audio pipeline cast the GGUF transformer.

    ``from_pipe(base, dtype=...)`` calls ``.to(dtype)`` on every module, and a
    GGUF-quantized model refuses it ("Casting a quantized model to a new dtype is
    unsupported"). Every mode once on the quantized runner, so the next
    quantization-only failure cannot hide.
    """
    from ltx25_studio import generate as generate_module
    from ltx25_studio import modes

    monkeypatch.setattr(modes, "MIN_RESOLUTION", SIZE)
    path, _ = faithful_gguf
    media = _inputs(tmp_path)
    fields = {"prompt": PROMPT}
    for kind in modes.get_mode(mode_id).inputs:
        if kind in media:
            fields[kind] = media[kind]

    def factory(*, pipeline, **_):
        runner = local_runtime.build_local_runner(
            kind=pipeline,
            device="cpu",
            dtype="bfloat16",
            snapshot_dir=tiny_snapshot(),
            transformer_gguf=path,
        )
        install_tiny_audio_encode_shim(runner.pipeline.audio_vae)
        return runner

    result = generate_module.generate(
        mode=mode_id,
        seed=SEED,
        width=SIZE,
        height=SIZE,
        num_frames=FRAMES,
        **{**fields, **extra},
        pipeline_factory=factory,
        output_root=tmp_path / "out",
        retention_bytes=10**12,
    )
    assert result.output_path.is_file()
    if "noise_scale" in extra:
        assert result.record.audio_level == extra["noise_scale"]


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


# --------------------------------------------------------------------------
# video_to_video's clip variation on the local runner (F66)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("residency", ["staged", "resident"])
@pytest.mark.parametrize("level", [0.421875, 0.725, 0.909375])
def test_a_clip_variation_runs_on_the_local_runner(residency, level, tmp_path, monkeypatch):
    torch = _torch()
    from diffusers import LTX2ConditionPipeline, LTX2VideoTransformer3DModel

    from ltx25_studio import audio_conditioning
    from ltx25_studio import generate as generate_module

    sigmas, sent = [], []
    forward = LTX2VideoTransformer3DModel.forward

    def spy_forward(self, *args, **kwargs):
        sigmas.append(float(kwargs["sigma"].flatten()[0]) / 1000.0)
        return forward(self, *args, **kwargs)

    call = LTX2ConditionPipeline.__call__

    def spy_call(self, *args, **kwargs):
        sent.append(kwargs)
        return call(self, *args, **kwargs)

    monkeypatch.setattr(LTX2VideoTransformer3DModel, "forward", spy_forward)
    monkeypatch.setattr(LTX2ConditionPipeline, "__call__", spy_call)
    lines = []

    def factory(*, pipeline, **_):
        return local_runtime.build_local_runner(
            kind=pipeline,
            device="cpu",
            dtype="float32",
            snapshot_dir=tiny_snapshot(),
            residency=residency,
            log=lines.append,
        )

    result = generate_module.generate(
        mode="video_to_video",
        prompt=PROMPT,
        seed=SEED,
        width=SIZE,
        height=SIZE,
        video=_inputs(tmp_path)["video"],
        noise_scale=level,
        pipeline_factory=factory,
        output_root=tmp_path / "out",
        retention_bytes=10**12,
    )

    ((kwargs),) = sent
    tail = audio_conditioning.sigma_tail(level)
    assert not kwargs.get("conditions")
    assert kwargs["sigmas"] == tail and kwargs["noise_scale"] == level
    assert (
        tuple(kwargs["latents"].shape)[:3] == (1, 8, 2) and kwargs["latents"].dtype == torch.float32
    )
    assert sigmas == pytest.approx(tail)
    assert result.record.video_level == level and result.record.frames == FRAMES
    if residency == "staged":
        order = [
            line.split("]")[0].removeprefix("preflight[")
            for line in lines
            if line.startswith("preflight[")
        ]
        assert order.index(f"encode clip {SIZE}x{SIZE}x{FRAMES}") < order.index("load text encoder")


# --------------------------------------------------------------------------
# review findings: adapters across a staged release, the first LoRA run, and
# the load preflight's dtype (local-runtime-2 = lora-1, -6 = lora-2, -3)
# --------------------------------------------------------------------------


def _tiny_adapter(tmp_path):
    torch = _torch()
    from test_real_pipeline_tiny import load_smoke

    adapter = tmp_path / "look.safetensors"
    donor = local_runtime.load_transformer(tiny_snapshot(), dtype=torch.float32, device="cpu")
    load_smoke().make_lora(adapter, donor)
    return {"source": str(adapter), "path": str(adapter), "weight": 1.0, "report": object()}


def _run(runner, tmp_path, name, **fields):
    from ltx25_studio import generate as generate_module

    return generate_module.generate(
        mode="text_to_video",
        seed=SEED,
        width=SIZE,
        height=SIZE,
        num_frames=FRAMES,
        pipeline_factory=lambda **_: runner,
        output_root=tmp_path / name,
        retention_bytes=10**12,
        **fields,
    )


def test_an_adapter_from_a_failed_staged_run_is_not_restored_into_the_next(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    lora = _tiny_adapter(tmp_path)
    runner = _runner("text2video", "staged")
    real_preflight = runner._weights._preflight

    def text_encoder_refuses(phase, need_gb):
        if phase == "load text encoder":  # after the adapter attached
            raise memory.MemoryPreflightError(reason="the text encoder does not fit (stand-in)")
        return real_preflight(phase, need_gb)

    monkeypatch.setattr(runner._weights, "_preflight", text_encoder_refuses)
    with pytest.raises(memory.MemoryPreflightError):
        _run(runner, tmp_path, "a", prompt="a first prompt", loras=(lora,))
    assert runner.pipeline.transformer is None, "the failed run released the transformer"

    monkeypatch.setattr(runner._weights, "_preflight", real_preflight)
    result = _run(runner, tmp_path, "b", prompt="a second prompt")

    assert not getattr(runner.pipeline.transformer, "peft_config", None), (
        "the failed run's adapter was re-attached to a run that asked for none"
    )
    assert result.record.adapters == ()


def test_the_first_lora_run_on_a_fresh_staged_runner_attaches(tmp_path):
    pytest.importorskip("peft")
    lora = _tiny_adapter(tmp_path)
    runner = _runner("text2video", "staged")
    assert runner.pipeline.transformer is None

    result = _run(runner, tmp_path, "a", prompt=PROMPT, loras=(lora,))

    ((_label, weight, attached),) = result.record.adapters
    assert weight == 1.0 and attached > 0


def test_a_converting_load_promises_the_converted_copy_and_the_read_pages():
    from local_fakes import tiny_snapshot as tiny

    assert local_runtime._load_need_gb(10.0, 1.15, 2, 2) == pytest.approx(11.5)
    assert local_runtime._load_need_gb(10.0, 1.15, 4, 2) == pytest.approx(10 * 1.15 * 2 + 10)
    stored = local_runtime._stored_dtype_bytes(tiny(), "vae")
    assert stored in (2, 4)
    other = 4 if stored == 2 else 2
    native = local_runtime.preflight_plan(
        tiny(), width=SIZE, height=SIZE, num_frames=FRAMES, dtype_bytes=stored
    )
    converting = local_runtime.preflight_plan(
        tiny(), width=SIZE, height=SIZE, num_frames=FRAMES, dtype_bytes=other
    )
    by_phase = {step.phase: step for step in native}
    assert converting[0].need_gb > by_phase["load small set"].need_gb


# --------------------------------------------------------------------------
# review minors: local-runtime-7 (adapter bytes), -8 (freed on failure),
# -10 (the duration head's longest clip)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seconds, frames", [(3.7, 89), (2.7, 65), (4.36, 105), (5.0, 113)])
def test_the_auto_duration_estimate_is_the_longest_clip_the_head_can_return(seconds, frames):
    from ltx25_studio import constraints

    got = local_runtime.frames_for_estimate({"max_seconds": seconds}, {"frame_rate": 24.0})

    assert got == frames
    assert (
        got
        == constraints.duration_head_frame_range(
            min_seconds=1.0, max_seconds=seconds, frame_rate=24.0
        )[1]
    )


def _phases(monkeypatch):
    seen = []
    real = local_runtime._LocalWeights._preflight

    def spy(self, phase, need_gb):
        seen.append((phase, need_gb))
        return real(self, phase, need_gb)

    monkeypatch.setattr(local_runtime._LocalWeights, "_preflight", spy)
    return seen


@pytest.mark.parametrize("residency", ["staged", "resident"])
def test_attaching_an_adapter_is_preflighted_like_any_load(tmp_path, monkeypatch, residency):
    pytest.importorskip("peft")
    lora = _tiny_adapter(tmp_path)
    runner = _runner("text2video", residency)
    seen = _phases(monkeypatch)

    _run(runner, tmp_path, "a", prompt=PROMPT, loras=(lora,))

    phases = [phase for phase, _ in seen]
    ((phase, need),) = [entry for entry in seen if entry[0].startswith("attach LoRA")]
    assert phase == "attach LoRA look.safetensors"
    assert need >= Path(lora["path"]).stat().st_size / memory.GB
    assert phases.index(phase) < next(i for i, p in enumerate(phases) if p.startswith("denoise"))


def test_an_adapter_that_does_not_fit_is_refused_as_memory_not_as_a_bad_file(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    lora = _tiny_adapter(tmp_path)
    runner = _runner("text2video", "staged")
    runner.ensure_transformer()
    monkeypatch.setenv("LTX25_MEMORY_LIMIT_GB", "1")

    with pytest.raises(memory.MemoryPreflightError) as caught:
        _run(runner, tmp_path, "a", prompt=PROMPT, loras=(lora,))
    assert "attach LoRA look.safetensors" in str(caught.value)


def test_reloading_the_transformer_promises_the_adapter_it_puts_back(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    lora = _tiny_adapter(tmp_path)
    runner = _runner("text2video", "staged")
    runner.ensure_transformer()
    runner.pipeline.load_lora_weights(lora["path"], adapter_name="look")
    weights = runner._weights
    weights.ensure_text_encoder()  # releases the transformer, capturing the adapter
    adapter_gb = local_runtime._adapter_bytes(weights._adapters) / memory.GB
    assert adapter_gb > 0
    seen = _phases(monkeypatch)

    weights.ensure_transformer()

    ((_, need),) = [entry for entry in seen if entry[0] == "load transformer"]
    bare = local_runtime._load_need_gb(
        weights._gb["transformer"],
        memory.load_spike_factor("safetensors"),
        weights.dtype_bytes,
        local_runtime._stored_dtype_bytes(weights.snapshot_dir, "transformer"),
    )
    assert need == pytest.approx(bare + adapter_gb)


def _frees(monkeypatch):
    calls = []
    real = local_runtime._LocalWeights._free

    def spy(self):
        calls.append(self)
        return real(self)

    monkeypatch.setattr(local_runtime._LocalWeights, "_free", spy)
    return calls


def test_a_failed_load_empties_the_device_cache(monkeypatch):
    runner = _runner("text2video", "staged")
    frees = _frees(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("MPS backend out of memory (stand-in)")

    monkeypatch.setattr(local_runtime, "load_transformer", fail)
    with pytest.raises(RuntimeError):
        runner.ensure_transformer()

    assert frees, "the failed load left the device cache as it was"


def test_a_failed_denoise_empties_the_device_cache(monkeypatch):
    torch = _torch()
    runner = _runner("text2video", "staged")
    frees = _frees(monkeypatch)
    reached = []
    real = runner._weights.preflight_denoise

    def preflight_then_watch(pipeline, call):
        real(pipeline, call)
        reached.append(True)
        frees.clear()  # only what happens from the denoise on counts

    monkeypatch.setattr(runner._weights, "preflight_denoise", preflight_then_watch)
    with pytest.raises(ValueError):
        runner(
            prompt=PROMPT,
            width=SIZE + 1,
            height=SIZE,
            num_frames=FRAMES,
            generator=torch.Generator().manual_seed(SEED),
            output_type="np",
        )

    assert reached, "the run failed before the denoise"
    assert frees, "the failed denoise left the device cache as it was"


def test_a_failed_resident_build_releases_what_it_loaded(monkeypatch):
    """The designed refusal: the transformer fits, the text encoder does not."""
    import weakref

    _torch()
    pytest.importorskip("diffusers")
    local_runtime.clear_local_cache()
    built = []
    real = local_runtime.load_transformer

    def load(*args, **kwargs):
        model = real(*args, **kwargs)
        built.append(weakref.ref(model))
        return model

    def refuse(*args, **kwargs):
        raise RuntimeError("the text encoder does not fit (stand-in)")

    monkeypatch.setattr(local_runtime, "load_transformer", load)
    monkeypatch.setattr(local_runtime, "load_text_encoder", refuse)
    frees = _frees(monkeypatch)
    try:
        with pytest.raises(RuntimeError, match="does not fit"):
            _runner("text2video", "resident")
    finally:
        local_runtime.clear_local_cache()
    gc.collect()

    assert frees
    assert built and built[0]() is None, "the failed build kept the transformer alive"


# --------------------------------------------------------------------------
# one stage graph: the SAME tests against the Space runner and the local runner
#
# The local runner is a pipelines.StageRunner that supplies residency through
# its one seam, so the decode policy (F07), stage 2 and its refusal (F08), the
# audio conditioning (F09), the clip variation (mode 04) and the run record (R)
# are one piece of code. Each behaviour runs through generate() on the tiny
# checkpoint, on the CPU, for the Space runner and the local runner staged and
# resident.
# --------------------------------------------------------------------------

RUNNERS = ("space", "staged", "resident")


@pytest.fixture(scope="module")
def composite(tmp_path_factory):
    """The tiny checkpoint plus tiny stage-2 models, as one snapshot directory."""
    _torch()
    pytest.importorskip("diffusers")
    from test_real_pipeline_tiny import load_smoke

    path, missing = load_smoke().composite_snapshot(
        tiny_snapshot(), tmp_path_factory.mktemp("composite") / "snapshot", fetch=False
    )
    if missing:
        pytest.skip(missing)
    return path


def _factory_for(kind, composite, monkeypatch):
    if kind == "space":
        from test_real_pipeline_tiny import load_smoke

        stack = load_smoke().tiny_stack(device="cpu")
        # The pre-dispatch decoder checks read the decoder's config from here.
        monkeypatch.setenv("LTX25_MODEL_SNAPSHOT_DIR", str(stack["snapshot"]))
        return lambda *, pipeline, **_: stack[pipeline]
    # The stage-2 parts resolve through preflight.component_dir.
    monkeypatch.setenv("LTX25_MODEL_SNAPSHOT_DIR", str(composite))

    def factory(*, pipeline, **_):
        runner = local_runtime.build_local_runner(
            kind=pipeline, device="cpu", dtype="float32", snapshot_dir=composite, residency=kind
        )
        install_tiny_audio_encode_shim(runner.pipeline.audio_vae)
        return runner

    return factory


def _through_generate(kind, composite, monkeypatch, tmp_path, mode, **fields):
    from test_real_pipeline_tiny import load_smoke

    from ltx25_studio import generate as generate_module

    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    request = dict(mode=mode, seed=SEED, width=SIZE, height=SIZE, num_frames=FRAMES)
    request.update(
        load_smoke()._media_for(
            mode, media, fields.get("width", SIZE), fields.get("num_frames", FRAMES)
        )
    )
    request.update(fields)
    return generate_module.generate(
        **request,
        pipeline_factory=_factory_for(kind, composite, monkeypatch),
        output_root=tmp_path / "out",
        retention_bytes=10**12,
    )


@pytest.fixture
def sigmas_seen(monkeypatch):
    """Per transformer call, on every runner: the video and the audio sigma."""
    _torch()
    pytest.importorskip("diffusers")
    from diffusers import LTX2VideoTransformer3DModel

    seen = []
    forward = LTX2VideoTransformer3DModel.forward

    def spy(self, *args, **kwargs):
        seen.append(
            (
                float(kwargs["sigma"].flatten()[0]) / 1000.0,
                float(kwargs["audio_sigma"].flatten()[0]) / 1000.0,
            )
        )
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(LTX2VideoTransformer3DModel, "forward", spy)
    return seen


@pytest.mark.parametrize("kind", RUNNERS)
def test_auto_decodes_with_conv_off_cuda_on_every_runner(kind, composite, monkeypatch, tmp_path):
    result = _through_generate(
        kind, composite, monkeypatch, tmp_path, "text_to_video", prompt=PROMPT
    )

    record = result.record
    assert (record.decoder, record.stage_2) == ("conv", False)
    assert (record.width, record.height, record.frames) == (SIZE, SIZE, FRAMES)
    assert any("NVIDIA" in line for line in result.record.diagnostics), result.record.diagnostics
    assert record.device == "cpu", "the run's time is labelled by the device that ran it"


@pytest.mark.parametrize("kind", RUNNERS)
def test_an_explicit_diffusion_decode_runs_at_the_cpu_decoders_size_on_every_runner(
    kind, composite, monkeypatch, tmp_path
):
    result = _through_generate(
        kind,
        composite,
        monkeypatch,
        tmp_path,
        "text_to_audio_video",
        prompt=PROMPT,
        width=128,
        height=128,
        decoder="diffusion",
    )

    assert result.record.decoder == "diffusion"
    assert (result.record.width, result.record.height) == (128, 128)


@pytest.mark.parametrize("kind", RUNNERS)
def test_stage_2_runs_at_tiny_size_and_doubles_it_on_every_runner(
    kind, composite, monkeypatch, tmp_path, sigmas_seen
):
    from ltx25_studio import pipelines

    result = _through_generate(
        kind, composite, monkeypatch, tmp_path, "image_to_video", enable_stage_2=True
    )

    record = result.record
    assert (record.stage_2, record.decoder) == (True, "diffusion")
    assert (record.width, record.height) == (2 * SIZE, 2 * SIZE)
    video = [video for video, _audio in sigmas_seen]
    assert video == pytest.approx(
        list(pipelines.DISTILLED_SIGMA_VALUES) + list(pipelines.STAGE_2_DISTILLED_SIGMA_VALUES)
    )


@pytest.mark.parametrize("kind", RUNNERS)
def test_stage_2_at_a_real_size_is_refused_on_every_runner(kind, composite, monkeypatch):
    from ltx25_studio.device import StageTwoNeedsCudaError

    runner = _factory_for(kind, composite, monkeypatch)(pipeline="text2video")
    with pytest.raises(StageTwoNeedsCudaError):
        runner(enable_stage_2=True, prompt=PROMPT, width=704, height=480, num_frames=121)


@pytest.mark.parametrize("kind", RUNNERS)
def test_audio_to_video_holds_the_audio_frozen_on_every_runner(
    kind, composite, monkeypatch, tmp_path, sigmas_seen
):
    from ltx25_studio import pipelines

    result = _through_generate(kind, composite, monkeypatch, tmp_path, "audio_to_video")

    assert [audio for _video, audio in sigmas_seen] == [0.0] * len(sigmas_seen)
    assert [video for video, _audio in sigmas_seen] == pytest.approx(
        list(pipelines.DISTILLED_SIGMA_VALUES)
    )
    assert result.record.audio_level == 0.0


@pytest.mark.parametrize("kind", RUNNERS)
def test_audio_to_audio_varies_the_audio_at_its_level_on_every_runner(
    kind, composite, monkeypatch, tmp_path, sigmas_seen
):
    from ltx25_studio import audio_conditioning, pipelines

    result = _through_generate(
        kind, composite, monkeypatch, tmp_path, "audio_to_audio", noise_scale=0.8
    )

    level = 0.725
    assert [audio for _video, audio in sigmas_seen] == pytest.approx(
        audio_conditioning.audio_sigmas(pipelines.DISTILLED_SIGMA_VALUES, level)
    )
    record = result.record
    assert (record.audio_level, record.decoder, record.stage_2) == (level, "conv", False)
    assert "your audio was changed moderately." in result.warnings, result.warnings


@pytest.mark.parametrize("kind", RUNNERS)
def test_a_clip_variation_denoises_the_tail_from_its_level_on_every_runner(
    kind, composite, monkeypatch, tmp_path, sigmas_seen
):
    from ltx25_studio import audio_conditioning

    level = 0.725
    result = _through_generate(
        kind, composite, monkeypatch, tmp_path, "video_to_video", prompt=PROMPT, noise_scale=level
    )

    assert [video for video, _audio in sigmas_seen] == pytest.approx(
        audio_conditioning.sigma_tail(level)
    )
    assert result.record.video_level == level
    assert result.record.frames == FRAMES


def test_a_staged_stage_2_run_encodes_the_prompt_once_and_keeps_the_transformer(
    composite, monkeypatch, tmp_path
):
    """The seam swaps the prompt for embeddings per call; stage 2 is a cache hit."""
    factory = _factory_for("staged", composite, monkeypatch)
    loads = []
    for name in ("load_text_encoder", "load_transformer"):
        real = getattr(local_runtime, name)
        monkeypatch.setattr(
            local_runtime,
            name,
            lambda *a, _real=real, _name=name, **k: loads.append(_name) or _real(*a, **k),
        )
    from ltx25_studio import generate as generate_module

    runner = factory(pipeline="text2video")
    result = generate_module.generate(
        mode="text_to_video",
        prompt="a prompt only this test encodes",
        seed=SEED,
        width=SIZE,
        height=SIZE,
        num_frames=FRAMES,
        enable_stage_2=True,
        pipeline_factory=lambda **_: runner,
        output_root=tmp_path / "out",
        retention_bytes=10**12,
    )

    assert result.record.stage_2
    assert loads.count("load_text_encoder") == 1, loads
    assert loads.count("load_transformer") <= 1, loads
    phases = [entry["phase"] for entry in runner.timings]
    assert "denoise (stage 1)" in phases and "denoise (stage 2)" in phases
