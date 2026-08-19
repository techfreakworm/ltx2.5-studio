"""Measure the audio tensor BEFORE it is encoded, to localise two findings.

    ./scripts/hinode_run.sh host_raw_audio.py

Two questions the finished files cannot answer, because the encode step sits
between the model and the evidence:

**Is `video_to_audio`'s near-silence the model's or ours?** The artifact is ~50 dB
below `text_to_audio`. If the raw tensor is already that quiet, the model produced
near-silence and the export is faithful. If the tensor is healthy, something in
our path attenuates it. Those are entirely different defects and the file cannot
distinguish them.

**Does the vocoder emit beyond ±1.0?** The PCM wavs peak at exactly 1.0000 — the
signature of clipping at the int16 boundary rather than of a signal that merely
reaches full scale — and the aac containers decode as high as 1.78. Either the
model emits hot and we clip it, or we apply gain. Again, only the tensor says
which.

Uses `generate(encoder=...)`, a documented injection seam, so the measurement sits
exactly where the real encoder would and nothing about the generation changes. The
injected encoder measures and then delegates to the real writer, so the run still
produces a genuine artifact and still satisfies the never-empty contract.
"""

import os

import numpy as np
import torch

from ltx25_studio import generate as generate_module
from ltx25_studio import outputs, pipelines

GiB = 1024**3
OUTPUT_ROOT = os.environ.get("RAW_OUTPUT_ROOT", "/home/wakeuser/ltx25-studio/outputs")
VIDEO_FIXTURE = os.environ.get("RAW_VIDEO_FIXTURE", "")
PROMPT = (
    "A red sports car driving fast along a winding coastal road at sunset, "
    "waves breaking on rocks below, cinematic camera following alongside"
)


def log(*a):
    print(*a, flush=True)


def describe(tag, value, depth=0):
    """Report every tensor-ish thing on the pipeline output, with its range."""
    pad = "    " * (depth + 1)
    if isinstance(value, torch.Tensor):
        v = value.detach().float().cpu()
        log(f"{pad}{tag}: Tensor {tuple(v.shape)} dtype={value.dtype} "
            f"min {v.min():.6f} max {v.max():.6f} "
            f"absmax {v.abs().max():.6f} rms {v.pow(2).mean().sqrt():.6f}")
        over = int((v.abs() > 1.0).sum())
        if over:
            log(f"{pad}  ** {over} of {v.numel()} samples exceed |1.0| "
                f"({100 * over / v.numel():.3f}%) -> clipped by any int16 write **")
        return
    if isinstance(value, np.ndarray):
        log(f"{pad}{tag}: ndarray {value.shape} dtype={value.dtype} "
            f"min {value.min():.6f} max {value.max():.6f}")
        return
    if isinstance(value, (list, tuple)) and value and depth < 2:
        log(f"{pad}{tag}: {type(value).__name__} len={len(value)}")
        describe(f"{tag}[0]", value[0], depth + 1)
        return
    log(f"{pad}{tag}: {type(value).__name__}")


def measure(mode, **extra):
    captured = {}

    def probe_encoder(output, **kwargs):
        # Sits exactly where the real encoder does, measures, then delegates —
        # so the run still writes a real file and the never-empty contract holds.
        log(f"  --- pipeline output for {mode!r}, BEFORE encode ---")
        for name in ("audio", "audios", "frames", "waveform"):
            if hasattr(output, name):
                describe(name, getattr(output, name))
        audio = getattr(output, "audio", None)
        if isinstance(audio, torch.Tensor):
            captured["absmax"] = float(audio.detach().float().abs().max())
        return outputs.write_output(output, **kwargs)

    log(f"\n=== {mode} ===")
    result = generate_module.generate(
        mode=mode,
        prompt=PROMPT,
        width=704,
        height=480,
        num_frames=121,
        seed=1234,
        output_root=OUTPUT_ROOT,
        encoder=probe_encoder,
        **extra,
    )
    log(f"  wrote {result.output_path} ({result.output_path.stat().st_size} bytes)")
    return captured


def main() -> int:
    pipelines.get_pipeline(pipeline="condition", device="cuda")
    log(f"loaded; cuda {torch.cuda.memory_allocated() / GiB:.3f} GiB")

    # text_to_audio is the healthy reference at -5.2 dBFS.
    measure("text_to_audio")

    # video_to_audio is the one producing -55 dBFS artifacts.
    if VIDEO_FIXTURE and os.path.exists(VIDEO_FIXTURE):
        import av

        frames = []
        with av.open(VIDEO_FIXTURE) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_image())
                if len(frames) >= 121:
                    break
        log(f"\nvideo fixture: {len(frames)} PIL frames from {VIDEO_FIXTURE}")
        measure("video_to_audio", video=frames)
    else:
        log("\nRAW_VIDEO_FIXTURE unset or missing; skipping video_to_audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
