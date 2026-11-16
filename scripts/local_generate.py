#!/usr/bin/env python
"""One local generation (mps/cpu) through the app's own ``generate()``.

    python scripts/local_generate.py --mode text_to_video --prompt "a red tram ..." \\
        --width 704 --height 480 --num-frames 121 --device mps \\
        --snapshot-dir <LTX-2.5-Diffusers snapshot> \\
        [--transformer-gguf <...-Q8_0.gguf>] [--dry-run] [--json run.json]

The local runner (``ltx25_studio.local_runtime``) is handed to ``generate()``
through its ``pipeline_factory`` seam, so this works whether or not
``pipelines.get_pipeline`` knows about local devices yet.

Prints the resolved plan, every memory-preflight line, the wall time of each
phase, the peak committed memory (``vm_stat``, sampled every 0.5 s) and MPS
driver memory, the files written and the stream verdict; ``--json`` writes the
same. ``--dry-run`` prints the plan and the preflight arithmetic and loads
nothing. Nothing is ever downloaded: a missing file is reported with the exact
``hf download`` command.

Exit codes: 0 ok, 2 refused (memory preflight, validation, missing files),
3 generation error.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import threading
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ltx25_studio import local_runtime, memory, modes  # noqa: E402
from ltx25_studio.errors import ConfigError, ConstraintError, LoRAError  # noqa: E402

EXIT_OK, EXIT_REFUSED, EXIT_ERROR = 0, 2, 3

#: Refusals: the run was stopped before or instead of generating, for a reason the
#: operator can act on. Everything else is a generation error.
REFUSALS = (
    memory.MemoryPreflightError,
    memory.MpsWatermarkError,
    ConstraintError,
    ConfigError,
    LoRAError,
    local_runtime.UnsupportedLocalDeviceError,
    local_runtime.LocalFilesMissingError,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", required=True, choices=sorted(modes.MODES))
    p.add_argument("--prompt", default="")
    p.add_argument("--image", type=Path)
    p.add_argument("--video", type=Path)
    p.add_argument("--audio", type=Path)
    p.add_argument("--width", type=int, default=704)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--seed", type=int, default=-1, help="-1 picks one at random")
    p.add_argument("--auto-duration", action="store_true")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=5.0)
    p.add_argument("--noise-scale", type=float, default=0.5)
    p.add_argument("--strength", type=float, default=0.8)
    p.add_argument("--enhance", action="store_true", help="rewrite the prompt with the enhancer")
    p.add_argument("--lora", action="append", default=[], metavar="PATH[:WEIGHT]")
    p.add_argument("--device", choices=local_runtime.LOCAL_DEVICES, default="mps")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--snapshot-dir", type=Path, required=True)
    p.add_argument("--transformer-gguf", type=Path)
    p.add_argument("--residency", choices=local_runtime.RESIDENCIES, default="staged")
    p.add_argument("--memory-limit-gb", type=float)
    p.add_argument("--out", type=Path, help="output root (default: the app's own)")
    p.add_argument("--json", type=Path, help="write the report here as JSON")
    p.add_argument("--dry-run", action="store_true", help="print the plan; load nothing")
    return p


def _lora(spec: str) -> dict:
    path, sep, weight = spec.rpartition(":")
    if sep and path:
        try:
            return {"source": path, "weight": float(weight)}
        except ValueError:
            pass
    return {"source": spec, "weight": 1.0}


class _Report:
    def __init__(self):
        self.data: dict = {"preflight": [], "log": []}

    def line(self, text: str = "") -> None:
        print(text, flush=True)

    def log(self, text: str) -> None:
        self.data["log"].append(text)
        if text.startswith("preflight["):
            self.data["preflight"].append(text)
        self.line(f"  | {text}")


class _Sampler(threading.Thread):
    """Peak committed memory and MPS driver memory, sampled every 0.5 s."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_committed_gb = None
        self.peak_mps_driver_gb = None
        self._halt = threading.Event()

    def _sample(self) -> None:
        committed = memory.committed_gb()
        if committed is not None:
            self.peak_committed_gb = max(self.peak_committed_gb or 0.0, committed)
        stats = memory.mps_stats()
        if stats is not None:
            self.peak_mps_driver_gb = max(self.peak_mps_driver_gb or 0.0, stats["driver_gb"])

    def run(self) -> None:
        while not self._halt.is_set():
            with contextlib.suppress(Exception):  # a gauge must never kill the run
                self._sample()
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)
        with contextlib.suppress(Exception):
            self._sample()


def _verify_streams():
    spec = importlib.util.spec_from_file_location(
        "_ltx25_verify_streams", REPO_ROOT / "tests" / "verify_streams.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _release_file_cache(args, report: _Report, when: str) -> None:
    """Give back the page-cache pages of every file this run loads; print the gauge."""
    files = local_runtime.run_files(args.snapshot_dir, args.transformer_gguf)
    before = memory.committed_gb()
    released = memory.release_file_cache(files)
    after = memory.committed_gb()

    def gb(value):
        return f"{value:.1f} GB" if value is not None else "n/a"

    line = (
        f"file cache {when}: released {released / memory.GB:.1f} GB across {len(files)} files; "
        f"committed {gb(before)} -> {gb(after)}"
    )
    report.data.setdefault("file_cache", []).append(
        {
            "when": when,
            "released_gb": released / memory.GB,
            "committed_before_gb": before,
            "committed_after_gb": after,
            "files": len(files),
        }
    )
    report.line(f"== {line}")


def _classify(exc: BaseException) -> int:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, REFUSALS):
            return EXIT_REFUSED
        exc = exc.__cause__ or exc.__context__
    return EXIT_ERROR


def _plan(args, report: _Report) -> tuple[list[tuple[str, str]], bool]:
    """Print the plan and the projected preflight; return (missing, every step fits)."""
    spec = modes.get_mode(args.mode)
    dtype_bytes = 4 if args.dtype.lower() in ("float32", "fp32", "float") else 2
    if args.auto_duration:
        frames = local_runtime.frames_for_estimate(
            {"max_seconds": args.max_seconds}, {"frame_rate": 24.0}
        )
    else:
        frames = args.num_frames
    width, height = (
        (modes.MIN_RESOLUTION, modes.MIN_RESOLUTION)
        if spec.min_resolution
        else (args.width, args.height)
    )
    sizes = local_runtime.weight_sizes(args.snapshot_dir, args.transformer_gguf)
    missing = local_runtime.missing_files(args.snapshot_dir, transformer_gguf=args.transformer_gguf)
    limit = args.memory_limit_gb if args.memory_limit_gb is not None else memory.default_limit_gb()
    watermarks = memory.mps_watermarks()

    report.line("== plan")
    plan = {
        "mode": args.mode,
        "pipeline_kind": spec.pipeline,
        "device": args.device,
        "dtype": args.dtype,
        "residency": args.residency,
        "size": f"{width}x{height}x{frames}"
        + (" (upper bound, auto duration)" if args.auto_duration else ""),
        "snapshot_dir": str(args.snapshot_dir),
        "transformer": str(args.transformer_gguf)
        if args.transformer_gguf
        else "snapshot transformer/",
        "memory_limit_gb": limit,
        "load_spike_factor": {
            source: memory.load_spike_factor(source) for source in memory.LOAD_SPIKE_FACTORS
        },
        "activation_calibration": memory.activation_calibration(),
        "stage_2": "not available locally (CUDA-only decoder)",
        "mps_watermarks": {"high": watermarks.high, "low": watermarks.low},
        "recommended_max_gb": (memory.mps_stats() or {}).get("recommended_max_gb"),
        "loras": [_lora(item) for item in args.lora],
        "weights_gb": {name: round(size / memory.GB, 3) for name, (size, _) in sizes.items()},
        "weights_source": {name: source for name, (_, source) in sizes.items()},
    }
    for key, value in plan.items():
        report.line(f"  {key}: {value}")
    report.data["plan"] = plan
    terms = memory.activation_terms(width, height, frames, dtype_bytes=dtype_bytes)
    report.line(
        "  activation terms (GB): "
        + ", ".join(f"{name} {value:.3f}" for name, value in terms.items())
        + f", fixed {memory.FIXED_ACTIVATION_GB:.1f}"
    )
    report.data["activation_terms_gb"] = terms

    if missing:
        report.line(f"== {len(missing)} file(s) missing (nothing is downloaded)")
        for path, command in missing:
            report.line(f"  {path}\n      {command}")
    report.data["missing"] = [{"path": str(p), "command": c} for p, c in missing]

    committed_now = memory.committed_gb()
    report.line(
        "== preflight arithmetic (projected; committed now "
        + (f"{committed_now:.1f} GB" if committed_now is not None else "n/a")
        + " + what earlier steps leave resident)"
    )
    steps = local_runtime.preflight_plan(
        args.snapshot_dir,
        transformer_gguf=args.transformer_gguf,
        residency=args.residency,
        width=width,
        height=height,
        num_frames=frames,
        dtype_bytes=dtype_bytes,
    )
    fits = True
    projected = []
    for step in steps:
        result = memory.preflight(
            step.phase,
            need_gb=step.need_gb,
            limit_gb=limit,
            committed_gb=None if committed_now is None else committed_now + step.resident_before_gb,
        )
        fits = fits and result.verdict != "FAIL"
        line = f"{result.describe()}  [resident before: {step.resident_before_gb:.1f} GB]"
        projected.append(line)
        report.line(f"  {line}")
    report.data["projected_preflight"] = projected
    return missing, fits


def _run(args, report: _Report) -> int:
    spec = modes.get_mode(args.mode)
    from ltx25_studio import generate as generate_module

    loras = tuple(_lora(item) for item in args.lora)

    def factory(*, pipeline, **_):
        runner = local_runtime.build_local_runner(
            kind=pipeline,
            device=args.device,
            dtype=args.dtype,
            snapshot_dir=args.snapshot_dir,
            transformer_gguf=args.transformer_gguf,
            residency=args.residency,
            memory_limit_gb=args.memory_limit_gb,
            log=report.log,
        )
        if loras:
            runner.ensure_transformer()  # adapters attach to a loaded transformer
        report.data["runner"] = runner
        return runner

    fields = {
        "mode": args.mode,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "auto_duration": args.auto_duration,
        "min_seconds": args.min_seconds,
        "max_seconds": args.max_seconds,
        "noise_scale": args.noise_scale,
        "strength": args.strength,
        "enhance_prompt": args.enhance,
        "loras": loras,
    }
    if args.image is not None:
        from PIL import Image

        with Image.open(args.image) as opened:
            fields["image"] = opened.convert("RGB")
    if args.video is not None:
        fields["video"] = str(args.video)
    if args.audio is not None:
        fields["audio"] = str(args.audio)

    report.line("== generating")
    sampler = _Sampler()
    sampler.start()
    started = time.perf_counter()
    try:
        result = generate_module.generate(
            pipeline_factory=factory,
            output_root=args.out,
            **fields,
        )
    finally:
        sampler.stop()
        report.data["wall_seconds_generate"] = round(time.perf_counter() - started, 3)
        report.data["peak_committed_gb"] = sampler.peak_committed_gb
        report.data["peak_mps_driver_gb"] = sampler.peak_mps_driver_gb
        runner = report.data.pop("runner", None)
        report.data["phases"] = list(runner.timings) if runner is not None else []
        vocoder = getattr(getattr(runner, "pipeline", None), "vocoder", None)

    report.line("== phases")
    for entry in report.data["phases"]:
        report.line(f"  {entry['phase']}: {entry['seconds']:.2f} s")
    report.line(f"  generate() total: {report.data['wall_seconds_generate']:.2f} s")
    report.line(
        "  peak committed: "
        + (
            f"{sampler.peak_committed_gb:.1f} GB"
            if sampler.peak_committed_gb is not None
            else "n/a"
        )
        + " | peak MPS driver: "
        + (
            f"{sampler.peak_mps_driver_gb:.2f} GB"
            if sampler.peak_mps_driver_gb is not None
            else "n/a"
        )
    )

    artifacts = {
        "output": str(result.output_path),
        "video": str(result.video_path) if result.video_path else None,
        "audio": str(result.audio_path) if result.audio_path else None,
        "seed": result.seed,
        "warnings": list(result.warnings),
    }
    report.data["artifacts"] = artifacts
    report.line("== artifacts")
    for key, value in artifacts.items():
        report.line(f"  {key}: {value}")

    verify_streams = _verify_streams()
    rate = None
    if "audio" in spec.exports:
        rate = getattr(getattr(vocoder, "config", None), "output_sampling_rate", None)
    geometry = None
    if "video" in spec.exports and not spec.min_resolution:
        geometry = (args.width, args.height)
    verdict = verify_streams.verify(
        args.mode, result.output_path, frame_rate=24.0, sampling_rate=rate, geometry=geometry
    )
    report.data["verify_streams"] = {
        "exit_code": verdict.exit_code,
        "ok": verdict.ok,
        "expected": verdict.expected,
        "actual": verdict.actual,
        "problems": verdict.problems,
    }
    report.line("== verify_streams")
    report.line("  " + verdict.render().replace("\n", "\n  "))
    return EXIT_OK


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    report = _Report()
    report.data["argv"] = list(sys.argv if argv is None else argv)
    code = EXIT_OK
    live = False
    try:
        for flag in ("image", "video", "audio"):
            path = getattr(args, flag)
            if path is not None and not path.is_file():
                raise ConstraintError(f"--{flag} {path} does not exist")
        for item in args.lora:
            if not Path(_lora(item)["source"]).is_file():
                raise ConstraintError(f"--lora {item}: no such file")
        watermarks = memory.mps_watermarks()
        report.line(
            f"== MPS allocator: {memory.HIGH_WATERMARK_ENV}="
            f"{watermarks.high if watermarks.high is not None else 'unset'}, "
            f"{memory.LOW_WATERMARK_ENV}={watermarks.low if watermarks.low is not None else 'unset'}"
        )
        if not args.dry_run:
            watermarks.require()  # before anything is touched
        if args.dry_run:
            report.line(
                "== file cache: left alone (dry run); a live run first gives back the page-cache "
                "pages of every file it will load, so 'committed now' is not inflated by an "
                "earlier run, and gives them back again at exit"
            )
        else:
            live = True
            _release_file_cache(args, report, "before the run")
        missing, fits = _plan(args, report)
        if args.dry_run:
            report.line("== dry run: nothing was loaded")
            watermarks.require()
            code = EXIT_REFUSED if (missing or not fits) else EXIT_OK
        elif missing:
            raise local_runtime.LocalFilesMissingError(missing=missing)
        else:
            code = _run(args, report)
    except BaseException as exc:  # noqa: BLE001 - every failure becomes an exit code and a record
        if isinstance(exc, KeyboardInterrupt):
            raise
        code = _classify(exc)
        report.data["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        report.line(
            f"== {'refused' if code == EXIT_REFUSED else 'error'}: {type(exc).__name__}: {exc}"
        )
    if live:
        with contextlib.suppress(Exception):
            _release_file_cache(args, report, "at exit")
    report.data["exit_code"] = code
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.data, indent=2, default=str))
        report.line(f"== report written to {args.json}")
    return code


if __name__ == "__main__":
    sys.exit(main())
