#!/usr/bin/env python
"""The real-pipeline tier ("L1"): every mode through the app's own generate().

Runs the pinned diffusers LTX-2.5 pipelines -- real classes, real code paths --
on a public ~95 MB random-weight checkpoint, on the CPU or on Apple's MPS:

    python scripts/local_smoke.py --device cpu [--fetch] [--out DIR]
    python scripts/local_smoke.py --device mps --out DIR
    python scripts/local_smoke.py --rope-parity --out DIR

It imports ``app`` the way the Space does, so a run goes UI-seam ->
``generate()`` -> the ``@spaces.GPU`` entrypoint -> ``pipelines.get_pipeline``
on the device ``LTX25_DEVICE`` resolves to, with the weights read from
``LTX25_MODEL_SNAPSHOT_DIR``. Output quality means nothing here; which code
paths, ops and dtypes run is the point. Every case reports PASS/FAIL, the
exception class and first line, the artifact and ``tests/verify_streams.py``'s
verdict; a table is printed, JSON is written, and the exit code is non-zero on
any unexpected result -- including an entry of :data:`EXPECTED_FAILURES` that
now passes, so a fix cannot land without its entry being removed.

Safety: tiny sizes only (<= 128 px, <= 17 frames; the CPU stage-2 decoder at 9
frames), a snapshot larger than 1 GB is refused, and the process kills itself
past 8 GB of resident memory. ``--fetch`` is the only thing that downloads: the
tiny checkpoint at its pinned revision and two stage-2 ``config.json`` files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _path in (REPO, REPO / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ltx25_studio import model_files  # noqa: E402 - stdlib-only module

MAX_SNAPSHOT_BYTES = 1_000_000_000
RSS_LIMIT_BYTES = 8 * 1024**3
SIZE, FRAMES = 128, 17
STAGE_2_SIZE, STAGE_2_FRAMES = 64, 9
FRAME_RATE = 24.0
SEED = 1234
PROMPT = "a red ball bouncing on a wooden floor, with a hollow knock on each bounce"

#: Tiny stage-2 models: the real configs with reduced widths and depths. The
#: stage-1 neighbourhood kernel shrinks (3,7,7)->(3,3,3) because a 128 px output
#: is a 4x4 latent grid, and stage 5's temporal kernel 11->9 because 9 frames is
#: what the dense CPU decoder fits.
TINY_STAGE_2 = {
    "latent_upsampler": {"in_channels": 8, "mid_channels": 32, "num_blocks_per_stage": 1},
    "diffusion_decoder": {
        "latent_channels": 8,
        "decoder_head_dim": 8,
        "decoder_stage_channels": [64, 32, 16, 16, 8],
        "decoder_stage_depths": [1, 1, 1, 1, 1],
        "decoder_t_emb_dim": 32,
        "decoder_stage_kernels": [[3, 3, 3], [3, 7, 7], [3, 5, 5], [3, 5, 5]],
        "decoder_stage5_kernel": [9, 11, 11],
    },
}

#: Known failures that belong to another owner. ``case -> (exception class,
#: message substring, reason)``. A listed case that passes is an UNEXPECTED
#: PASS: remove its entry in the same change as the fix.
_STAGE_2_SIZE_BUG = (
    "DecodeError",
    "expected shape",
    "StageRunner strips height/width for stage 2 and LTX2ConditionPipeline.prepare_latents "
    "sizes its mask from the 512x768 defaults (recon B N1; pipelines.py StageRunner, next brief)",
)
EXPECTED_FAILURES: dict[str, tuple[str, str, str]] = {
    "stage2:image_to_video": _STAGE_2_SIZE_BUG,
    "stage2:image_text_to_video": _STAGE_2_SIZE_BUG,
    "stage2:video_to_video": _STAGE_2_SIZE_BUG,
    "stage2:image_to_audio_video": _STAGE_2_SIZE_BUG,
    "stage2:image_text_to_audio_video": _STAGE_2_SIZE_BUG,
    "stage2:audio_to_video": (
        "DecodeError",
        "multiple values for keyword argument",
        "StageRunner forwards the request's noise_scale/audio_latents and stage_2_kwargs adds "
        "them again (probe bug B; pipelines.py StageRunner, next brief)",
    ),
}

#: Cases not run on a device, with the measured reason. Unlike an expected
#: failure these never execute, so they cannot turn red by themselves.
SKIP_BY_DEVICE: dict[str, dict[str, str]] = {
    "mps": {
        "enhance:image_text_to_video": (
            "not run on MPS: measured 8.60 GB RSS (watchdog kill) on 2026-11-16 -- the random "
            "tiny enhancer decodes 600 tokens of noise through the image route; the text-only "
            "route passes on MPS and both pass on the CPU"
        ),
    },
}

#: Device-specific known failures, merged over the table above for that device.
EXPECTED_FAILURES_BY_DEVICE: dict[str, dict[str, tuple[str, str, str]]] = {}


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------


def start_watchdog() -> dict:
    """Kill this process past 8 GB resident (or 8 GB MPS driver memory)."""
    import psutil

    peak = {"rss": 0, "mps": 0}
    process = psutil.Process(os.getpid())

    def watch():
        while True:
            rss = process.memory_info().rss
            mps = 0
            torch = sys.modules.get("torch")
            if torch is not None:
                try:
                    if torch.backends.mps.is_available():
                        mps = int(torch.mps.driver_allocated_memory())
                except Exception:
                    mps = 0
            peak["rss"], peak["mps"] = max(peak["rss"], rss), max(peak["mps"], mps)
            if rss > RSS_LIMIT_BYTES or mps > RSS_LIMIT_BYTES:
                sys.stderr.write(
                    f"\nlocal_smoke: RSS {rss / 1e9:.2f} GB / MPS {mps / 1e9:.2f} GB is past the "
                    f"8 GB limit; killing this process\n"
                )
                sys.stderr.flush()
                os._exit(99)
            time.sleep(0.2)

    threading.Thread(target=watch, daemon=True).start()
    return peak


def _tree_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=True):
        for name in files:
            total += os.stat(os.path.join(root, name)).st_size
    return total


def tiny_snapshot(*, fetch: bool, explicit: str | None = None) -> Path:
    """The tiny checkpoint's snapshot directory, refusing anything over 1 GB."""
    if explicit:
        path = Path(explicit).resolve()
    else:
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                model_files.TINY_REPO_ID,
                revision=model_files.TINY_REVISION,
                local_files_only=not fetch,
            )
        )
    size = _tree_bytes(path)
    if size > MAX_SNAPSHOT_BYTES:
        raise SystemExit(
            f"local_smoke: {path} holds {size / 1e9:.2f} GB; this tier runs tiny checkpoints "
            f"only (limit {MAX_SNAPSHOT_BYTES / 1e9:.0f} GB). Refusing."
        )
    return path


def _real_config(subfolder: str, *, fetch: bool) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        model_files.REPO_ID,
        f"{subfolder}/config.json",
        revision=model_files.REVISION,
        local_files_only=not fetch,
    )
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}


def composite_snapshot(tiny: Path, into: Path, *, fetch: bool) -> tuple[Path, str | None]:
    """Tiny base (symlinked) plus tiny random stage-2 models built from real configs.

    Returns ``(path, why stage 2 is unavailable or None)``.
    """
    import torch
    from diffusers import LTX2VideoDiffusionDecoderModel
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    into.mkdir(parents=True, exist_ok=True)
    for child in tiny.iterdir():
        (into / child.name).symlink_to(child.resolve() if child.is_file() else child)
    try:
        configs = {sub: _real_config(sub, fetch=fetch) for sub in TINY_STAGE_2}
    except Exception as exc:  # noqa: BLE001 - offline without the configs
        return (
            into,
            f"the real stage-2 configs are not cached ({exc.__class__.__name__}); rerun with --fetch",
        )
    torch.manual_seed(0)
    for sub, cls in (
        ("latent_upsampler", LTX2LatentUpsamplerModel),
        ("diffusion_decoder", LTX2VideoDiffusionDecoderModel),
    ):
        model = cls.from_config({**configs[sub], **TINY_STAGE_2[sub]}).to(torch.bfloat16)
        model.save_pretrained(into / sub, safe_serialization=True)
    return into, None


# ---------------------------------------------------------------------------
# media, generated on the fly
# ---------------------------------------------------------------------------


def make_image(size: int):
    """A PIL image, as the UI's ``gr.Image(type="pil")`` hands one over."""
    import numpy as np
    import PIL.Image

    rng = np.random.default_rng(0)
    pixels = (rng.random((size, size, 3)) * 255).astype("uint8")
    pixels[size // 4 : size // 2, size // 4 : size // 2] = [220, 30, 30]
    return PIL.Image.fromarray(pixels)


def make_video(path: Path, size: int, frames: int) -> str:
    import av
    import numpy as np

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=int(FRAME_RATE))
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            pixels = np.full((size, size, 3), 40, dtype="uint8")
            x = (8 + index * 6) % (size - 24)
            pixels[size // 3 : size // 3 + 24, x : x + 24] = [30, 200, 60]
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return str(path)


def make_audio(path: Path, frames: int, rate: int = 48000) -> str:
    import numpy as np

    count = round(frames / FRAME_RATE * rate)
    t = np.arange(count) / rate
    stereo = np.stack([0.4 * np.sin(2 * np.pi * 440 * t), 0.4 * np.sin(2 * np.pi * 660 * t)], 1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((stereo * 32767).astype("<i2").tobytes())
    return str(path)


def make_lora(path: Path, transformer, rank: int = 4) -> int:
    """A rank-4 adapter on the tiny transformer's real attention Linears."""
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(3)
    tensors = {}
    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in ("attn1.to_q", "attn1.to_k", "attn1.to_v")):
            continue
        down = torch.randn(rank, module.in_features, generator=generator) * 0.01
        up = torch.randn(module.out_features, rank, generator=generator) * 0.01
        tensors[f"diffusion_model.{name}.lora_A.weight"] = down.contiguous()
        tensors[f"diffusion_model.{name}.lora_B.weight"] = up.contiguous()
    metadata = {"model_version": "2.5.0", "lora_rank": str(rank), "lora_alpha": str(rank)}
    save_file(tensors, str(path), metadata=metadata)
    return len(tensors) // 2


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def _media_for(mode_id: str, media: Path, size: int, frames: int) -> dict:
    from ltx25_studio import modes

    fields: dict = {}
    for kind in modes.get_mode(mode_id).inputs:
        if kind == "text":
            fields["prompt"] = PROMPT
        elif kind == "image":
            fields["image"] = make_image(size)
        elif kind == "video":
            fields["video"] = make_video(media / f"video_{size}_{frames}.mp4", size, frames)
        elif kind == "audio":
            fields["audio"] = make_audio(media / f"audio_{frames}.wav", frames)
    return fields


def build_cases(device: str, media: Path, stage_2_ready: str | None) -> list[dict]:
    """``{"id", "mode", "request", "expect_error"?, "env"?, "skip"?}`` per case."""
    from ltx25_studio import modes

    cases = []
    for mode_id in modes.MODES:
        request = dict(
            mode=mode_id,
            seed=SEED,
            width=SIZE,
            height=SIZE,
            num_frames=FRAMES,
            frame_rate=FRAME_RATE,
            **_media_for(mode_id, media, SIZE, FRAMES),
        )
        cases.append({"id": f"mode:{mode_id}", "mode": mode_id, "request": request})
    cases.append(
        {
            "id": "duration_head:text_to_audio_video",
            "mode": "text_to_audio_video",
            "request": dict(
                mode="text_to_audio_video",
                seed=SEED,
                width=SIZE,
                height=SIZE,
                frame_rate=FRAME_RATE,
                prompt=PROMPT,
                auto_duration=True,
                min_seconds=0.3,
                max_seconds=0.7,
            ),
        }
    )
    cases.append(
        {
            "id": "lora:text_to_video",
            "mode": "text_to_video",
            "lora": True,
            "request": dict(
                mode="text_to_video",
                seed=SEED,
                width=SIZE,
                height=SIZE,
                num_frames=FRAMES,
                frame_rate=FRAME_RATE,
                prompt=PROMPT,
            ),
        }
    )
    for mode_id in modes.MODES:
        if modes.get_mode(mode_id).skip_stage_2:
            continue
        case = {
            "id": f"stage2:{mode_id}",
            "mode": mode_id,
            "stage_2": True,
            "request": dict(
                mode=mode_id,
                seed=SEED,
                width=STAGE_2_SIZE,
                height=STAGE_2_SIZE,
                num_frames=STAGE_2_FRAMES,
                frame_rate=FRAME_RATE,
                enable_stage_2=True,
                **_media_for(mode_id, media, STAGE_2_SIZE, STAGE_2_FRAMES),
            ),
        }
        if stage_2_ready is not None:
            case["skip"] = stage_2_ready
        cases.append(case)
    # Off CUDA a stage-2 request above the CPU decoder's limit is refused, typed.
    cases.append(
        {
            "id": "stage2_too_large:text_to_video",
            "mode": "text_to_video",
            "expect_error": "StageTwoNeedsCudaError",
            "request": dict(
                mode="text_to_video",
                seed=SEED,
                width=SIZE,
                height=SIZE,
                num_frames=FRAMES,
                frame_rate=FRAME_RATE,
                prompt=PROMPT,
                enable_stage_2=True,
            ),
        }
    )
    # Last: on MPS the random tiny enhancer's 600-token decode leaves resident
    # memory behind (0.89 -> 6.06 GB RSS measured, not returned by clearing the
    # enhancer or the MPS cache), which stacked under stage 2 crossed 8 GB.
    for mode_id in ("text_to_video", "image_text_to_video"):
        cases.append(
            {
                "id": f"enhance:{mode_id}",
                "mode": mode_id,
                "enhance": True,
                "request": dict(
                    mode=mode_id,
                    seed=SEED,
                    width=SIZE,
                    height=SIZE,
                    num_frames=FRAMES,
                    frame_rate=FRAME_RATE,
                    enhance_prompt=True,
                    **_media_for(mode_id, media, SIZE, FRAMES),
                ),
            }
        )
    for case in cases:
        reason = SKIP_BY_DEVICE.get(device, {}).get(case["id"])
        if reason:
            case["skip"] = reason
    return cases


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0][:300] if text else repr(exc)


def classify(case_id: str, error: BaseException | None, streams_ok: bool | None, expected) -> str:
    entry = expected.get(case_id)
    if entry is None:
        return "PASS" if error is None and streams_ok is not False else "FAIL"
    cls, needle, _reason = entry
    if error is None:
        return "XPASS"
    if type(error).__name__ == cls and needle in str(error):
        return "XFAIL"
    return "FAIL"


def _flush(out: Path, device: str, rows) -> None:
    """Rows so far, so a watchdog kill still leaves a readable record."""
    (out / f"local_smoke_{device}.partial.json").write_text(json.dumps(rows, indent=2, default=str))


def run_cases(cases, *, device: str, out: Path, expected) -> list[dict]:
    from verify_streams import verify

    from ltx25_studio import enhance, generate, pipelines
    from ltx25_studio.lora import apply as lora_apply
    from ltx25_studio.lora import validate_lora

    base = pipelines._CACHE["__base__"]
    rate = int(base.vocoder.config.output_sampling_rate)
    attached: list[int] = []
    real_apply = lora_apply.apply_lora

    def counting_apply(*args, **kwargs):
        report = real_apply(*args, **kwargs)
        attached.append(report.attached)
        return report

    def tiny_enhancer(prompt, image=None):
        return enhance.enhance(
            prompt,
            image=image,
            repo_id="tiny-local",
            loader=lambda repo_id: (base.prompt_enhancer, base.processor),
        )

    rows = []
    for case in cases:
        row = {"case": case["id"], "mode": case["mode"], "device": device}
        if case.get("skip"):
            row.update(outcome="SKIP", error=case["skip"])
            rows.append(row)
            continue
        request = dict(case["request"])
        seams: dict = {"output_root": str(out / "outputs")}
        if case.get("enhance"):
            enhance.clear_enhancer_cache()
            seams["enhancer"] = tiny_enhancer
        if case.get("lora"):
            adapter = out / "media" / "tiny_rank4.safetensors"
            row["lora_pairs"] = make_lora(adapter, base.transformer)
            report = validate_lora(str(adapter), transformer=base.transformer, manifest=None)
            request["loras"] = ({"source": str(adapter), "weight": 1.0, "report": report},)
            attached.clear()
            lora_apply.apply_lora = counting_apply
        print(f"... {case['id']} (RSS {_rss() / 1e9:.2f} GB)", flush=True)
        started = time.perf_counter()
        error = None
        result = None
        try:
            result = generate.generate(**request, **seams)
        except Exception as exc:  # noqa: BLE001 - every failure is a row, not a crash
            error = exc
            row["traceback"] = traceback.format_exc(limit=6)[-1500:]
        finally:
            lora_apply.apply_lora = real_apply
        row["seconds"] = round(time.perf_counter() - started, 2)
        row["rss_gb_after"] = round(_rss() / 1e9, 2)
        if case.get("lora"):
            row["lora_attached_modules"] = sum(attached)
        streams_ok = None
        if result is not None:
            row["artifact"] = str(result.output_path)
            scale = 2 if case.get("stage_2") else 1
            geometry = (
                (request["width"] * scale, request["height"] * scale)
                if "video" in _exports(case["mode"])
                else None
            )
            verdict = verify(
                case["mode"],
                result.output_path,
                frame_rate=FRAME_RATE,
                sampling_rate=rate,
                geometry=geometry,
            )
            streams_ok = verdict.ok
            row["streams"] = "MATCHES" if verdict.ok else "; ".join(verdict.problems)
            row["streams_actual"] = verdict.actual
            row["warnings"] = list(result.warnings)
            if case.get("lora") and not sum(attached):
                streams_ok = False
                row["streams"] = "the adapter attached to 0 modules"
        if case.get("expect_error"):
            ok = error is not None and type(error).__name__ == case["expect_error"]
            row["outcome"] = "PASS" if ok else "FAIL"
            row["expected_error"] = case["expect_error"]
        else:
            row["outcome"] = classify(case["id"], error, streams_ok, expected)
        if error is not None:
            row["error_type"] = type(error).__name__
            row["error"] = _first_line(error)
        if case["id"] in expected:
            row["expected_failure_reason"] = expected[case["id"]][2]
        rows.append(row)
        _flush(out, device, rows)
        print(
            f"[{row['outcome']:5}] {row['case']:40} {row['seconds']:6.2f}s "
            f"{row.get('error_type', '')} {row.get('error', '')[:100]}",
            flush=True,
        )
    return rows


def _rss() -> int:
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss


def _exports(mode_id: str):
    from ltx25_studio import modes

    return modes.get_mode(mode_id).exports


def print_table(rows) -> None:
    print()
    print(f"{'outcome':7} {'case':40} {'streams':10} exception / artifact")
    for row in rows:
        detail = (
            f"{row.get('error_type')}: {row.get('error', '')[:90]}"
            if row.get("error_type")
            else row.get("artifact", row.get("error", ""))
        )
        streams = (
            "-" if "streams" not in row else ("ok" if row["streams"] == "MATCHES" else "MISMATCH")
        )
        print(f"{row['outcome']:7} {row['case']:40} {streams:10} {detail}")


# ---------------------------------------------------------------------------
# the RoPE parity measurement (one transformer forward)
# ---------------------------------------------------------------------------


def rope_parity(snapshot: Path) -> dict:
    """cpu/fp32 with float64 RoPE (reference) vs MPS with the app's handling."""
    import torch
    from diffusers import LTX2VideoTransformer3DModel

    from ltx25_studio import device as device_module

    def load():
        return LTX2VideoTransformer3DModel.from_pretrained(
            str(snapshot), subfolder="transformer", dtype=torch.float32
        ).eval()

    config = load().config
    generator = torch.Generator().manual_seed(0)
    frames, height, width, audio_frames, text = 3, 4, 4, 18, 16
    inputs = {
        "hidden_states": torch.randn(
            1, frames * height * width, config.in_channels, generator=generator
        ),
        "audio_hidden_states": torch.randn(
            1, audio_frames, config.audio_in_channels, generator=generator
        ),
        "encoder_hidden_states": torch.randn(1, text, config.caption_channels, generator=generator),
        "audio_encoder_hidden_states": torch.randn(
            1, text, config.caption_channels, generator=generator
        ),
        "timestep": torch.tensor([990.0]),
        "sigma": torch.tensor([0.99]),
        "encoder_attention_mask": torch.ones(1, text),
        "audio_encoder_attention_mask": torch.ones(1, text),
    }
    fixed = {
        "num_frames": frames,
        "height": height,
        "width": width,
        "fps": FRAME_RATE,
        "audio_num_frames": audio_frames,
        "use_cross_timestep": True,
        "return_dict": False,
    }

    def forward(model, where):
        with torch.no_grad():
            out = model.to(where)(**{k: v.to(where) for k, v in inputs.items()}, **fixed)
        return [tensor.float().cpu() for tensor in out]

    def flip(model):
        for module in model.modules():
            if hasattr(module, "double_precision"):
                module.double_precision = False
        return model

    reference = forward(load(), "cpu")
    candidates = {
        "mps fp32, RoPE on the CPU in float64 (shipped handling)": None,
        "cpu fp32, RoPE in float32 (cost of the flag alone)": lambda: forward(flip(load()), "cpu"),
        "mps fp32, RoPE in float32 on MPS (the double_precision=False alternative)": (
            lambda: forward(flip(load()), "mps")
        ),
    }

    def handled():
        model = load()
        device_module.compute_rope_on_cpu(("transformer", model))
        return forward(model, "mps")

    candidates["mps fp32, RoPE on the CPU in float64 (shipped handling)"] = handled
    rows = {}
    for label, run in candidates.items():
        outputs = run()
        rows[label] = {
            name: {
                "max_abs": float((got - ref).abs().max()),
                "mean_abs": float((got - ref).abs().mean()),
                "ref_abs_max": float(ref.abs().max()),
            }
            for name, got, ref in zip(("video", "audio"), outputs, reference, strict=True)
        }
    return {
        "torch": torch.__version__,
        "inputs": "one transformer forward, seed 0, 3x4x4 latents",
        "results": rows,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _environment(args, composite: Path, out: Path) -> None:
    os.environ["LTX25_DEVICE"] = args.device
    os.environ["LTX25_MODEL_SNAPSHOT_DIR"] = str(composite)
    os.environ["LTX25_CPU_DTYPE"] = args.cpu_dtype
    os.environ["LTX25_OUTPUT_DIR"] = str(out / "outputs")
    os.environ["LTX25_PREFETCH_ENHANCER"] = "0"
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    if not args.fetch:
        os.environ["HF_HUB_OFFLINE"] = "1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--cpu-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--fetch", action="store_true", help="download the tiny checkpoint (~95 MB)"
    )
    parser.add_argument("--snapshot", help="a local tiny snapshot instead of the HF cache")
    parser.add_argument("--out", default=None, help="directory for JSON, media and outputs")
    parser.add_argument("--only", default="", help="comma-separated case-id prefixes")
    parser.add_argument("--rope-parity", action="store_true", help="measure RoPE parity only")
    args = parser.parse_args(argv)

    out = Path(args.out or tempfile.mkdtemp(prefix="ltx25_smoke_")).resolve()
    (out / "media").mkdir(parents=True, exist_ok=True)
    peak = start_watchdog()
    tiny = tiny_snapshot(fetch=args.fetch, explicit=args.snapshot)

    if args.rope_parity:
        parity = rope_parity(tiny)
        (out / "rope_parity.json").write_text(json.dumps(parity, indent=2))
        print(json.dumps(parity, indent=2))
        return 0

    work = Path(tempfile.mkdtemp(prefix="ltx25_composite_"))
    try:
        composite, stage_2_missing = composite_snapshot(tiny, work / "snapshot", fetch=args.fetch)
        _environment(args, composite, out)
        return _run(args, out, composite, stage_2_missing, peak)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run(args, out: Path, composite: Path, stage_2_missing, peak) -> int:
    import importlib

    from ltx25_studio import modes

    # Tiny tier: the 256 px render of the audio-only modes is held to <= 128 px.
    modes.MIN_RESOLUTION = SIZE
    expected = {**EXPECTED_FAILURES, **EXPECTED_FAILURES_BY_DEVICE.get(args.device, {})}
    rows: list[dict] = []

    from ltx25_studio import device as device_module
    from ltx25_studio import generate, pipelines

    if args.device == "mps":
        # Under the default (enforce), a failed self-test must refuse, typed.
        os.environ["LTX25_MPS_SELF_TEST"] = "enforce"
        self_test = device_module.mps_self_test()
    app = importlib.import_module("app")
    meta = {
        "device": app.DEVICE.describe(),
        "precision": pipelines.precision_summary(app.DEVICE),
        "snapshot": str(composite),
        "stage_2_models": stage_2_missing or "tiny random, built from the real configs",
    }
    if args.device == "mps":
        meta["mps_self_test"] = self_test.describe()
        row = {"case": "mps_self_test:enforce", "mode": "text_to_video", "device": "mps"}
        if self_test.ok:
            row.update(outcome="SKIP", error="the self-test passed; there is nothing to refuse")
        else:
            try:
                generate.generate(
                    mode="text_to_video",
                    seed=SEED,
                    width=SIZE,
                    height=SIZE,
                    num_frames=FRAMES,
                    prompt=PROMPT,
                )
                row.update(outcome="FAIL", error="a failed self-test did not refuse generation")
            except device_module.MpsNumericsError as exc:
                row.update(
                    outcome="PASS",
                    error_type=type(exc).__name__,
                    error=_first_line(exc),
                    expected_error="MpsNumericsError",
                )
        rows.append(row)
        print(f"[{row['outcome']:5}] {row['case']}: {row.get('error', '')[:120]}")
        # The rest exercises the MPS path itself, with the failure acknowledged.
        os.environ["LTX25_MPS_SELF_TEST"] = "warn"

    for kind in ("condition", "text2video"):
        pipelines.get_pipeline(pipeline=kind)
    base = pipelines._CACHE["__base__"]
    meta["audio_vae_shim"] = _install_tiny_audio_vae_shim(base.audio_vae)

    cases = build_cases(args.device, out / "media", stage_2_missing)
    if args.only:
        prefixes = tuple(p.strip() for p in args.only.split(",") if p.strip())
        cases = [case for case in cases if case["id"].startswith(prefixes)]
    rows += run_cases(cases, device=args.device, out=out, expected=expected)

    meta.update(
        _versions(),
        peak_rss_gb=round(peak["rss"] / 1e9, 2),
        peak_mps_driver_gb=round(peak["mps"] / 1e9, 2),
    )
    unexpected = [row["case"] for row in rows if row["outcome"] in ("FAIL", "XPASS")]
    payload = {
        "meta": meta,
        "expected_failures": {k: v[2] for k, v in expected.items()},
        "unexpected": unexpected,
        "cases": rows,
    }
    target = out / f"local_smoke_{args.device}.json"
    target.write_text(json.dumps(payload, indent=2, default=str))
    (out / f"local_smoke_{args.device}.partial.json").unlink(missing_ok=True)
    print_table(rows)
    counts = {
        o: sum(r["outcome"] == o for r in rows) for o in ("PASS", "XFAIL", "SKIP", "FAIL", "XPASS")
    }
    print(f"\n{counts} -> {target}")
    if unexpected:
        print(f"UNEXPECTED: {unexpected}")
    return 1 if unexpected else 0


def _install_tiny_audio_vae_shim(audio_vae) -> str:
    """Tiny-checkpoint artifact, not an app fix: its audio VAE has ``ch_mult=[1]``.

    So its encoder does not downsample while the class assumes 4x4. The posterior
    is average-pooled 4x4 to the shape the real VAE produces; everything after it
    is real code.
    """
    import torch.nn.functional as functional
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_outputs import AutoencoderKLOutput

    if getattr(audio_vae, "_tiny_shim", False):
        return "already installed"
    original = audio_vae.encode

    def encode(x, return_dict=True):
        params = original(x, return_dict=True).latent_dist.parameters
        pooled = functional.avg_pool2d(params.float(), kernel_size=4, ceil_mode=True)
        dist = DiagonalGaussianDistribution(pooled.to(params.dtype))
        return AutoencoderKLOutput(latent_dist=dist) if return_dict else (dist,)

    audio_vae.encode = encode
    audio_vae._tiny_shim = True
    return "audio_vae.encode posterior avg-pooled 4x4 (tiny ch_mult=[1] artifact)"


def _versions() -> dict:
    import diffusers
    import torch

    return {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "python": sys.version.split()[0],
    }


if __name__ == "__main__":
    sys.exit(main())
