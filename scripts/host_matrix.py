"""Run Part A of docs/TEST_PLAN.md on the GPU host, one process, one model load.

    MATRIX_RUNS=H1,H2,H3 ./scripts/hinode_run.sh host_matrix.py

Why one process for many runs: the first load reads ~68 GiB off disk. That is I/O,
not compute, and it is a one-off — `pipelines._CACHE` holds the pipeline for the
rest of the process. Running each mode in its own process would pay that cost per
mode and attribute disk time to the model. Load and generate are timed separately
for the same reason.

Each run prints a `>>> RUN <id>` block and flushes, so progress is readable from
the log while the process is still going rather than only at the end.

Streams are checked by `tests/verify_streams.py`, which resolves the expected
layout from the mode registry on every call. Reading `ffprobe` by eye across
sixteen runs is sixteen chances for a wrong-but-plausible pass. **Exit 2 is "could
not probe" and is not a pass.**
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ltx25_studio import generate as generate_module
from ltx25_studio import modes, pipelines

GiB = 1024**3
REPO = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = os.environ.get("MATRIX_OUTPUT_ROOT", "/home/wakeuser/ltx25-studio/outputs")

#: PLAN/TEST_PLAN A.1. 704x480 are multiples of 32; 121 is k*8+1.
BASE_W, BASE_H, BASE_FRAMES = 704, 480, 121
#: Pinned. TEST_PLAN A.4.1: seed=-1 draws a fresh value per call, so any
#: same-seed comparison run at the default differs for that reason alone and
#: passes vacuously.
SEED = 1234

PROMPT = (
    "A red sports car driving fast along a winding coastal road at sunset, "
    "waves breaking on rocks below, cinematic camera following alongside"
)


def log(*a):
    print(*a, flush=True)


def make_image(width: int, height: int):
    """A structured test image, not a flat fill.

    A solid colour is a degenerate conditioning signal: the model can produce
    something plausible from it while the anchor conversion is wrong, which is the
    failure this fixture exists to expose.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):  # sky gradient
        t = y / max(height - 1, 1)
        draw.line([(0, y), (width, y)], fill=(int(30 + 200 * t), int(60 + 120 * t), int(160 - 60 * t)))
    draw.rectangle([0, int(height * 0.68), width, height], fill=(38, 34, 30))  # road
    for x in range(0, width, 80):  # lane markings
        draw.rectangle([x, int(height * 0.82), x + 40, int(height * 0.84)], fill=(230, 225, 210))
    draw.ellipse(  # sun
        [int(width * 0.66), int(height * 0.16), int(width * 0.80), int(height * 0.30)],
        fill=(255, 214, 120),
    )
    draw.rectangle(  # the car
        [int(width * 0.40), int(height * 0.60), int(width * 0.58), int(height * 0.74)],
        fill=(180, 30, 30),
    )
    return img


def make_wav(path, *, seconds=5.0, rate=48000, channels=2):
    """A synthetic stereo wav to drive the audio-INPUT modes.

    Two tones an octave apart, one per channel, plus an amplitude envelope. Not
    silence and not identical channels: silence encodes to a degenerate latent
    that tells us nothing, and duplicated channels would hide a mono downmix in
    exactly the place the stereo path is supposed to be checked.
    """
    import av
    import numpy as np

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * rate)
    t = np.arange(n, dtype="float32") / rate
    envelope = 0.4 * (1.0 - np.exp(-3.0 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t))
    left = envelope * np.sin(2 * np.pi * 220.0 * t)
    right = envelope * np.sin(2 * np.pi * 440.0 * t)
    planes = np.stack([left, right])[:channels]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("pcm_s16le", rate=rate)
        stream.layout = "stereo" if channels == 2 else "mono"
        # `s16` is a PACKED format, so from_ndarray wants one row of interleaved
        # samples -- (1, n*channels) -- not the (channels, n) planar layout.
        # Handing it planar data raises "Expected packed array.shape[0] to equal 1".
        interleaved = (planes.T.reshape(1, -1) * 32767).astype("int16")
        frame = av.AudioFrame.from_ndarray(
            interleaved, format="s16", layout=stream.layout.name
        )
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def vocoder_sampling_rate(pipe):
    from ltx25_studio import audio as audio_module

    vocoder = getattr(pipe, "vocoder", None)
    if vocoder is None:
        return None
    try:
        return audio_module.export_sampling_rate_from(vocoder)
    except Exception:
        return None


def audio_level(path):
    """Report level for any artifact carrying audio.

    After H9 -- correct codec, correct rate, 2 channels, verify_streams exit 0,
    and 50 dB below its siblings -- no audio artifact gets reported here without
    its level. A structural check cannot see silence.
    """
    import av

    try:
        with av.open(str(path)) as container:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                return None
            planes = [f.to_ndarray() for f in container.decode(audio=0)]
    except Exception as exc:
        log(f"  audio level: could not probe ({type(exc).__name__}: {exc})")
        return None
    if not planes:
        return None
    raw = np.concatenate(planes, axis=-1)
    if np.issubdtype(raw.dtype, np.integer):
        raw = raw.astype("float32") / 32768.0
    else:
        raw = raw.astype("float32")
    peak = float(np.abs(raw).max())
    rms = float(np.sqrt((raw**2).mean()))
    dbfs = 20 * np.log10(rms + 1e-12)
    flag = "  *** NEAR-SILENT ***" if dbfs < -40 else ""
    log(f"  audio level: peak {peak:.4f} rms {rms:.6f} = {dbfs:.1f} dBFS{flag}")
    return dbfs


def verify(mode_id: str, path: Path, *, sampling_rate=None) -> int:
    cmd = [sys.executable, str(REPO / "tests" / "verify_streams.py"), mode_id, str(path)]
    cmd += ["--frame-rate", "24"]
    if sampling_rate:
        cmd += ["--sampling-rate", str(int(sampling_rate))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log(proc.stdout.strip() or proc.stderr.strip())
    log(f"  verify_streams exit = {proc.returncode} "
        f"({ {0: 'MATCHES', 1: 'MISMATCH', 2: 'COULD NOT PROBE'}.get(proc.returncode, '?') })")
    return proc.returncode


# --- the run table, mirroring TEST_PLAN A.2 ---------------------------------
# `needs` names a fixture the run requires; `extra` is merged into the request.
RUNS = {
    "H1":  dict(mode="text_to_video",             needs=(),        extra={}),
    "H2":  dict(mode="image_to_video",            needs=("image",), extra={}),
    "H3":  dict(mode="image_text_to_video",       needs=("image",), extra={}),
    "H4":  dict(mode="video_to_video",            needs=("video",), extra={"strength": 0.7}),
    "H5":  dict(mode="text_to_audio_video",       needs=(),        extra={}),
    "H6":  dict(mode="image_to_audio_video",      needs=("image",), extra={}),
    "H7":  dict(mode="image_text_to_audio_video", needs=("image",), extra={}),
    "H8":  dict(mode="text_to_audio",             needs=(),        extra={}),
    "H9":  dict(mode="video_to_audio",            needs=("video",), extra={}),
    "H10": dict(mode="audio_to_audio",            needs=("audio",), extra={"noise_scale": 0.5}),
    "H11": dict(mode="audio_to_video",            needs=("audio",), extra={"noise_scale": 0.5}),
    # A.4 -- the two features that need their own run.
    # H15: duration_head picks the length, so num_frames must not be sent at all;
    # generate() pops it when auto_duration is set and passes the bounds instead.
    "H15": dict(mode="text_to_video", needs=(), extra={
        "auto_duration": True, "min_seconds": 2.0, "max_seconds": 6.0}),
    # A.4 H14 -- stage 2. Same prompt and seed as H1, so the comparison is the
    # refinement chain and nothing else; the artifact must DIFFER from H1's.
    "H14": dict(mode="text_to_video", needs=(), extra={"enable_stage_2": True}),
    # H9 strength sweep. video_to_audio lands ~50 dB below every other audio
    # mode, and modes._conditions applies a video condition at
    # MAX_VIDEO_CONDITION_STRENGTH = 0.99. If a condition that strong leaves the
    # shared denoise schedule with almost nothing to do, the audio stream would
    # come out near-silent exactly as measured. Sweeping the one lever tests the
    # mechanism directly rather than probing an input property; the level should
    # climb as strength falls if that is the cause.
    "H9s_099": dict(mode="video_to_audio", needs=("video",), extra={"strength": 0.99}),
    "H9s_050": dict(mode="video_to_audio", needs=("video",), extra={"strength": 0.50}),
    "H9s_020": dict(mode="video_to_audio", needs=("video",), extra={"strength": 0.20}),
    "H9s_005": dict(mode="video_to_audio", needs=("video",), extra={"strength": 0.05}),
    # A second stage-2 run in the SAME process, to settle whether NATTEN's
    # version->ref lookup fires once per process or once per generation. One
    # stage-2 run cannot distinguish the two; two can. Different seed so it is a
    # genuinely separate generation rather than a cache hit.
    "H14b": dict(mode="text_to_video", needs=(), extra={"enable_stage_2": True},
                 seed=99999),
    # A.4.1 H4b -- strength 0.3 vs 0.9 on the same seed. Two runs that must differ
    # from each other; identical output means `strength` reaches nothing.
    "H4b_lo": dict(mode="video_to_video", needs=("video",), extra={"strength": 0.3}),
    "H4b_hi": dict(mode="video_to_video", needs=("video",), extra={"strength": 0.9}),
    # A.4.1 H10b -- the acceptance test for the audio-input path. noise_scale 1.0
    # discards the input BY DESIGN, so a bug that drops it entirely is
    # indistinguishable from correct operation: the mode produces audio either
    # way and every stream check passes. Only these two differing proves the
    # upload arrived.
    "H10b_lo": dict(mode="audio_to_audio", needs=("audio",), extra={"noise_scale": 0.1}),
    "H10b_hi": dict(mode="audio_to_audio", needs=("audio",), extra={"noise_scale": 0.9}),
    # A.5 -- the resolution a user would actually choose. 1216 and 704 are both
    # multiples of 32; everything above this is a correctness check, this is the
    # one that says whether it is usable.
    "H16": dict(mode="text_to_audio_video", needs=(), extra={},
                width=1216, height=704),
}


def load_video_frames(path, limit=BASE_FRAMES):
    """Decode an mp4 into PIL frames. `frames` accepts list[PIL.Image]."""
    import av

    out = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            out.append(frame.to_image())
            if len(out) >= limit:
                break
    return out


def encode_audio_latents(pipe, wav_path):
    """Encode a wav into audio latents through the APP's own audio path.

    There is no ``audio=`` argument on any LTX2 pipeline (PLAN §3.1c) — audio
    enters as pre-encoded ``audio_latents``. Rather than hand-roll that, this
    calls ``audio.load_waveform`` and ``audio.encode_waveform_to_latents``, which
    is what the UI path uses: waveform -> resample to the VAE's own rate -> the
    VAE's own mel front-end -> ``audio_vae.encode``, with the latent shape
    asserted before it can reach the pipeline.

    Using the real helpers is the point. My first attempt built the tensor here
    and handed ``audio_vae.encode`` a raw waveform, which failed with
    ``expected input[1, 1, 4, 480002] to have 2 channels`` — the VAE consumes a
    **mel spectrogram**, not samples, and the mel front-end is deliberately
    *found* on the VAE rather than constructed, because the FFT and window sizes
    are not in its config and any guess yields a mel it was never trained on.
    So a fixture that skips these helpers is not a shortcut, it is a different
    and wrong computation.
    """
    from ltx25_studio import audio as audio_module

    waveform, rate = audio_module.load_waveform(wav_path)
    log(f"  load_waveform: {tuple(waveform.shape)} @ {rate} Hz")
    params = audio_module.audio_params_from(pipe.audio_vae)
    log(f"  audio_vae params: rate={params.audio_sampling_rate} "
        f"mel_bins={params.audio_mel_bins} hop={params.audio_hop_length}")
    waveform = waveform.to(device=pipe.audio_vae.device, dtype=pipe.audio_vae.dtype)
    latents = audio_module.encode_waveform_to_latents(
        waveform, audio_vae=pipe.audio_vae, sampling_rate=rate
    )
    log(f"  audio latents {tuple(latents.shape)} dtype={latents.dtype}  "
        f"(dims {audio_module.AUDIO_LATENT_DIMS})")
    return latents


def main() -> int:
    wanted = [r.strip() for r in os.environ.get("MATRIX_RUNS", "H1").split(",") if r.strip()]
    unknown = [r for r in wanted if r not in RUNS]
    if unknown:
        log(f"unknown run ids: {unknown}; known: {sorted(RUNS)}")
        return 2

    log(f"torch {torch.__version__}  {torch.cuda.get_device_name()}  "
        f"cap {torch.cuda.get_device_capability()}")
    log(f"runs requested: {wanted}")
    log(f"baseline {BASE_W}x{BASE_H}x{BASE_FRAMES}  seed {SEED} (pinned)\n")

    # ---------------- A.0 gates ----------------
    log("=== A.0 GATES ===")
    from ltx25_studio import preflight, zerogpu
    from ltx25_studio.config import load_config

    cfg = load_config()
    stub = zerogpu.local_spaces_stub()
    log(f"  HF_TOKEN in env      : {bool(os.environ.get('HF_TOKEN'))}")
    log(f"  is_zerogpu           : {zerogpu.is_zerogpu(stub)}")
    log(f"  will_use_gpu         : {zerogpu.will_use_gpu(stub)}")
    log(f"  missing cached files : {len(preflight.missing_cached_files())}")
    log(pipelines.startup_report(cfg, missing_files=preflight.missing_cached_files()))

    bypass = os.environ.get("MATRIX_BYPASS_STAGE2") == "1"
    t0 = time.time()
    if bypass:
        # Harness accommodation, NOT a fix. get_pipeline builds the stage-2 parts
        # unconditionally, and the diffusion decoder's NATTEN processor resolves
        # its kernel version through the Hub -- which fails under
        # HF_HUB_OFFLINE=1, the mode the model load itself requires to avoid
        # refetching 44.33 GB of dead shards. The two are mutually exclusive
        # today, so nothing loads.
        #
        # This routes around it using the app's OWN seam rather than editing it:
        # `generate(pipeline_factory=...)` is a documented injection point, and
        # the factory below builds the same StageRunner over the same
        # derive_pipeline the real path uses. Only the stage-2 parts are absent,
        # and no mode run under this flag uses them.
        base = pipelines._base_pipeline(device="cuda")
        load_seconds = time.time() - t0
        log("  (MATRIX_BYPASS_STAGE2=1: base loaded directly; stage-2 parts absent)")
    else:
        cond = pipelines.get_pipeline(pipeline="condition", device="cuda")
        load_seconds = time.time() - t0
        base = cond.pipeline
    a1 = torch.cuda.memory_allocated()
    log(f"\n  LOAD condition       : {load_seconds:.1f}s")
    log(f"  cuda allocated       : {a1} bytes ({a1 / GiB:.3f} GiB)")
    log(f"  cuda reserved        : {torch.cuda.memory_reserved() / GiB:.3f} GiB")
    log(f"  base.dtype           : {base.dtype}")

    if bypass:
        from diffusers import LTX2Pipeline

        t2v = pipelines.derive_pipeline(base, LTX2Pipeline)
    else:
        t2v = pipelines.get_pipeline(pipeline="text2video", device="cuda").pipeline
    a2 = torch.cuda.memory_allocated()
    log(f"  after text2video     : {a2} bytes ({a2 / GiB:.3f} GiB)  DELTA {a2 - a1} bytes")

    # Identity, not equality. Equality passes with two full copies loaded, which
    # is the entire defect this gate exists to catch.
    shared_ok = True
    for name in ("transformer", "vae", "audio_vae", "text_encoder", "connectors", "vocoder"):
        a, b = getattr(base, name, None), getattr(t2v, name, None)
        same = a is b and a is not None
        shared_ok &= same
        log(f"  base.{name:<13} is t2v.{name:<13} -> {same}")
    # And the property the identity check alone does NOT establish: sharing and
    # preserving are two things, and from_pipe recast the shared modules in place.
    dtype_ok = base.dtype == torch.bfloat16
    log(f"  base.dtype still bfloat16 after derive : {dtype_ok}")

    sched = getattr(base, "audio_scheduler", "ABSENT")
    log(f"  condition.audio_scheduler : {type(sched).__name__ if sched != 'ABSENT' else 'ABSENT'}")
    log(f"  text2video has audio_scheduler attr : {hasattr(t2v, 'audio_scheduler')}")
    log(f"  A.0 VERDICT: identity={shared_ok} dtype_preserved={dtype_ok}\n")

    rate = vocoder_sampling_rate(base)
    log(f"  vocoder output_sampling_rate : {rate}\n")

    # ---------------- fixtures ----------------
    fixtures: dict[str, object] = {}
    if any("image" in RUNS[r]["needs"] for r in wanted):
        fixtures["image"] = make_image(BASE_W, BASE_H)
        log(f"fixture image  : PIL {fixtures['image'].size}")
    for key, env in (("video", "MATRIX_VIDEO_FIXTURE"), ("audio", "MATRIX_AUDIO_FIXTURE")):
        if any(key in RUNS[r]["needs"] for r in wanted):
            # Per-key and non-fatal: a fixture this run cannot build must not
            # cancel the modes that never asked for it. Runs needing a missing
            # fixture are skipped and named in the summary instead.
            src = os.environ.get(env)
            if key == "audio" and (not src or not Path(src).exists()):
                # Synthesised rather than taken from an earlier run, deliberately.
                # Every audio-EXPORTING mode is currently broken by the bf16
                # encode defect, so there is no generated wav to borrow — and an
                # audio *input* fixture is a different path from the broken
                # encoder anyway, so making one here keeps H10/H11 runnable.
                src = str(Path(OUTPUT_ROOT).parent / "fixture-audio.wav")
                make_wav(src, seconds=5.0)
                log(f"synthesised audio fixture -> {src}")
            if not src or not Path(src).exists():
                log(f"MISSING {key} fixture: set {env} to a real file "
                    f"(produce one from an earlier run)")
                continue
            # try/except, not just an existence check: a fixture that RAISES has
            # to be as survivable as one that is absent, or an unrelated bug in
            # fixture construction cancels every run in the batch.
            try:
                if key == "video":
                    fixtures["video"] = load_video_frames(src)
                    log(f"fixture video  : {len(fixtures['video'])} PIL frames from {src}")
                else:
                    # The PATH, not pre-encoded latents. cbfbe87 made
                    # generate() encode request.audio itself, and that join is
                    # exactly what was missing — the UI wrote request.audio,
                    # _with_audio read request.audio_latents, and nothing
                    # connected them, so these modes silently discarded the
                    # upload and generated fresh audio. Passing latents here
                    # would bypass the fix and re-prove nothing.
                    fixtures["audio"] = src
                    log(f"fixture audio  : file {src} (generate() encodes it)")
            except Exception as exc:
                log(f"fixture {key} FAILED to build: {type(exc).__name__}: {exc}")
                log(f"  runs needing {key!r} will be SKIPPED; the rest proceed")
    log("")

    # ---------------- the runs ----------------
    results = []
    for run_id in wanted:
        spec_row = RUNS[run_id]
        mode_id = spec_row["mode"]
        spec = modes.get_mode(mode_id)
        log(f">>> RUN {run_id}  mode={mode_id}  pipeline={spec.pipeline}  "
            f"exports={spec.exports}")
        req = dict(
            mode=mode_id,
            prompt=PROMPT,
            width=spec_row.get("width", BASE_W),
            height=spec_row.get("height", BASE_H),
            num_frames=spec_row.get("frames", BASE_FRAMES),
            seed=spec_row.get("seed", SEED),
            output_root=OUTPUT_ROOT,
            **spec_row["extra"],
        )
        log(f"  request geometry {req['width']}x{req['height']}x{req['num_frames']}"
            + ("  auto_duration ON (num_frames will be dropped)"
               if spec_row["extra"].get("auto_duration") else ""))
        absent = [n for n in spec_row["needs"] if n not in fixtures]
        if absent:
            log(f"  SKIPPED: fixture(s) {absent} unavailable")
            results.append((run_id, mode_id, "SKIPPED", None, None))
            log("")
            continue
        for need in spec_row["needs"]:
            req[need] = fixtures[need]
        if spec.min_resolution:
            log(f"  (registry forces min resolution {modes.MIN_RESOLUTION} for this mode)")

        if bypass:
            def _factory(*, pipeline, **_kw):
                inner = base if pipeline == pipelines.BASE_KIND else t2v
                return pipelines.StageRunner(inner, kind=pipeline)

            req["pipeline_factory"] = _factory

        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            result = generate_module.generate(**req)
        except Exception as exc:
            log(f"  *** FAILED after {time.time() - started:.1f}s: "
                f"{type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            results.append((run_id, mode_id, "FAILED", None, None))
            log("")
            continue
        gen_seconds = time.time() - started

        size = result.output_path.stat().st_size
        log(f"  GENERATE  {gen_seconds:.1f}s   peak cuda "
            f"{torch.cuda.max_memory_allocated() / GiB:.3f} GiB")
        log(f"  output    {result.output_path}  ({size} bytes)")
        log(f"  video={result.video_path}  audio={result.audio_path}  seed={result.seed}")
        for w in result.warnings:
            log(f"  warning   {w}")
        audio_level(result.output_path)
        code = verify(mode_id, result.output_path, sampling_rate=rate)
        results.append((run_id, mode_id, gen_seconds, result.output_path, code))
        log("")

    log("=== SUMMARY ===")
    log(f"load {load_seconds:.1f}s (one-off, ~68 GiB off disk)")
    for run_id, mode_id, secs, path, code in results:
        if secs in ("FAILED", "SKIPPED"):
            log(f"  {run_id:<4} {mode_id:<26} {secs}")
        else:
            log(f"  {run_id:<4} {mode_id:<26} {secs:6.1f}s  verify={code}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
