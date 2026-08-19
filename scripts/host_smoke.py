"""One real generation on a real card, verified down to the container streams.

Run on hinode; see docs/HINODE_HOST_SETUP.md. Configured by environment so the
same script covers the whole mode matrix::

    SMOKE_MODE=text_to_video SMOKE_W=512 SMOKE_H=320 SMOKE_FRAMES=25

Two things here are deliberate rather than convenient.

**It seeds the CUDA cache the way app.py does.** ``generate()``'s own pipeline
factory calls ``get_pipeline`` with **no device**, so it relies on a cache entry
that a module-scope ``get_pipeline(..., device="cuda")`` already put there. Calling
that first is not a shortcut around the app -- it is the same contract app.py
depends on, and asserting every component landed on cuda is what catches the case
where it silently did not.

**It decodes every frame rather than trusting the header.** A container whose
header parses and whose packets are garbage opens perfectly cleanly. PLAN §5.3 is
about never reporting success for an artifact nobody verified, and "the file is
non-empty" is not verification.
"""

import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

import torch  # noqa: E402

from ltx25_studio import generate as generate_module  # noqa: E402
from ltx25_studio import modes, pipelines  # noqa: E402

GiB = 1024**3

MODE = os.environ.get("SMOKE_MODE", "text_to_video")
WIDTH = int(os.environ.get("SMOKE_W", "512"))
HEIGHT = int(os.environ.get("SMOKE_H", "320"))
FRAMES = int(os.environ.get("SMOKE_FRAMES", "25"))
STAGE_2 = os.environ.get("SMOKE_STAGE_2", "0") == "1"
OUTPUT_ROOT = os.environ.get("SMOKE_OUTPUT_ROOT", "/home/wakeuser/ltx25-studio/outputs")
PROMPT = os.environ.get(
    "SMOKE_PROMPT",
    "A red sports car driving fast along a coastal road at sunset, cinematic",
)


def describe_streams(path) -> None:
    import av

    print("\n=== REAL STREAM LAYOUT (PyAV, reading the file back off disk) ===")
    with av.open(str(path)) as container:
        print(f"container   : {container.format.name} ({container.format.long_name})")
        dur = container.duration
        print(f"duration    : {dur} ({(dur or 0) / 1e6:.3f} s)")
        print(f"streams     : {len(container.streams)}")
        for stream in container.streams:
            ctx = stream.codec_context
            print(f"  [{stream.index}] type={stream.type} codec={ctx.name}")
            if stream.type == "video":
                print(
                    f"       {ctx.width}x{ctx.height} pix_fmt={ctx.pix_fmt} "
                    f"avg_rate={stream.average_rate} time_base={stream.time_base} "
                    f"declared_frames={stream.frames}"
                )
            elif stream.type == "audio":
                print(
                    f"       sample_rate={ctx.sample_rate} channels={ctx.channels} "
                    f"layout={stream.layout.name} format={stream.format.name}"
                )
    # Reopen to decode: a clean header proves nothing about the packets.
    counts: dict[str, int] = {}
    with av.open(str(path)) as container:
        for frame in container.decode():
            kind = "video" if type(frame).__name__ == "VideoFrame" else "audio"
            counts[kind] = counts.get(kind, 0) + 1
    print(f"DECODED     : {counts}")
    return counts


def main() -> int:
    spec = modes.get_mode(MODE)
    print(f"mode {MODE!r} -> pipeline {spec.pipeline!r}  exports {spec.exports}")
    print(f"geometry {WIDTH}x{HEIGHT}x{FRAMES}  stage_2={STAGE_2}", flush=True)

    t0 = time.time()
    runner = pipelines.get_pipeline(pipeline=spec.pipeline, device="cuda")
    print(f"pipeline loaded and moved to cuda in {time.time() - t0:.1f}s", flush=True)

    inner = runner.pipeline
    devs: dict[str, list[str]] = {}
    for name, mod in (getattr(inner, "components", None) or {}).items():
        if isinstance(mod, torch.nn.Module):
            dev = next((str(p.device) for p in mod.parameters()), "no-params")
            devs.setdefault(dev, []).append(name)
    print("component devices:", {k: sorted(v) for k, v in devs.items()})
    print(
        f"cuda allocated {torch.cuda.memory_allocated() / GiB:.3f} GiB  "
        f"reserved {torch.cuda.memory_reserved() / GiB:.3f} GiB",
        flush=True,
    )
    off = [d for d in devs if not d.startswith("cuda")]
    if off:
        # Not a warning. A component left on CPU means either a silent
        # host-to-device copy per step or a device-mismatch crash mid-run.
        raise SystemExit(f"components are NOT on cuda: { {d: devs[d] for d in off} }")
    print("ASSERT every weight-bearing component is on cuda: PASS", flush=True)

    t0 = time.time()
    result = generate_module.generate(
        mode=MODE,
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=FRAMES,
        seed=int(os.environ.get("SMOKE_SEED", "1234")),
        enable_stage_2=STAGE_2,
        output_root=OUTPUT_ROOT,
    )
    elapsed = time.time() - t0

    print(f"\n=== GENERATION COMPLETED in {elapsed:.1f}s ===")
    print(f"output_path : {result.output_path}")
    print(f"run_dir     : {result.run_dir}")
    print(f"run_id      : {result.run_id}")
    print(f"mode / seed : {result.mode} / {result.seed}")
    print(f"video_path  : {result.video_path}")
    print(f"audio_path  : {result.audio_path}")
    print(f"warnings    : {result.warnings}")
    print(f"size        : {result.output_path.stat().st_size} bytes")
    print(f"peak cuda   : {torch.cuda.max_memory_allocated() / GiB:.3f} GiB")

    describe_streams(result.output_path)

    print("\nrun dir contents:")
    for path in sorted(result.run_dir.iterdir()):
        print(f"  {path.name:<24} {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
