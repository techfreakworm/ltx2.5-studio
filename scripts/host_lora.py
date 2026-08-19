"""TEST_PLAN H12 / H13a / H13b — the LoRA validator, both directions, real adapters.

    ./scripts/hinode_run.sh host_lora.py

**Reports which stage rejected and with what message, never just the verdict.**
The two `charlied98` files are near-identical in size and differ in key
convention (`comfy_` vs plain), so they could be refused by *different* stages —
and two mechanisms reaching the same verdict reads exactly like a pass. That is
the "red for the wrong reason" trap, and only naming the stage avoids it.

Stage 6.5 needs a real transformer to compare per-module dimensions against, so
the pipeline is loaded once and reused across all cases. Stage 7 (`apply_lora`)
needs it too — it asserts the adapter attached to more than zero modules, which
is the check that turns "it validated" into "it is actually in the graph".
"""

import os
import time
import traceback
from pathlib import Path

import torch

from ltx25_studio import pipelines
from ltx25_studio.lora import apply as lora_apply
from ltx25_studio.lora import inspect_lora, validate_lora

GiB = 1024**3
HUB = Path("/home/wakeuser/.cache/huggingface/hub")

CASES = [
    dict(
        run="H12",
        repo="models--joyfox--LTX-2.3-Transition-LORA",
        filename="ltx2.3-transition.safetensors",
        expect="ACCEPT",
        note="mixed dims (4096 attention / 2048 audio_ff / 16384 ff.net.2) are NOT a mismatch",
    ),
    dict(
        run="H13a",
        repo="models--charlied98--ltxv_2b_lora_soft_draw_v1",
        filename="lora_weights_step_03000.safetensors",
        expect="REJECT",
        note="diffusers key convention; 2B adapter against a 22B transformer",
    ),
    dict(
        run="H13b",
        repo="models--charlied98--ltxv_2b_lora_soft_draw_v1",
        filename="comfy_lora_weights_step_03000.safetensors",
        expect="REJECT",
        note="comfy key convention, same weights — may be refused by a DIFFERENT stage",
    ),
    dict(
        run="H12c",
        repo="models--AhsanHareem--reelbids-ltx25-camera-lora",
        filename="lora_weights_step_01250.safetensors",
        expect="UNKNOWN",
        note="an actual LTX-2.5 adapter; verdict not specified by the plan",
    ),
]

#: Which validator stage each error type belongs to, per PLAN §4.3. Reported so a
#: rejection names its mechanism rather than only its outcome.
STAGE_OF = {
    "LoRATooLargeError": "1 (size bound, before any parse)",
    "LoRAFormatError": "2 (not a readable safetensors file)",
    "LoRAHeaderError": "3 (header parse)",
    "LoRATruncatedError": "3 (header/file length disagree)",
    "LoRAVersionError": "4 (model series)",
    "LoRAConventionError": "5 (key convention)",
    "LoRAInContextError": "5.5 (IC-LoRA refused rather than loaded)",
    "LoRAPairingError": "6 (lora_A/lora_B pairing)",
    "LoRADimensionError": "6.5 (PER-MODULE dimension check)",
    "LoRAModuleNotFoundError": "6.5 (named module absent from the transformer)",
    "LoRANotAttachedError": "7 (attached to zero modules)",
}


def log(*a):
    print(*a, flush=True)


def resolve(repo: str, filename: str) -> Path | None:
    matches = sorted((HUB / repo).glob(f"snapshots/*/{filename}"))
    return matches[0] if matches else None


def main() -> int:
    log(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    t0 = time.time()
    # `_base_pipeline`, not `get_pipeline`, and deliberately. get_pipeline now
    # builds the stage-2 parts unconditionally, and the diffusion decoder's NATTEN
    # processor resolves its kernel version through the Hub -- which fails under
    # HF_HUB_OFFLINE=1, the very mode the model load requires to avoid refetching
    # 44.33 GB of dead shards. Until that conflict is resolved in pipelines.py,
    # loading the base directly is the only way to exercise the validator at all.
    # The LoRA path needs the transformer and the loadable components, both of
    # which the base carries; stage 2 is irrelevant to it.
    pipe = pipelines._base_pipeline(device="cuda")
    log(f"pipeline loaded in {time.time() - t0:.1f}s  "
        f"cuda {torch.cuda.memory_allocated() / GiB:.3f} GiB")
    transformer = getattr(pipe, "transformer", None)
    log(f"transformer for stage 6.5: {type(transformer).__name__}")
    log(f"loadable components: {lora_apply._loadable_components(pipe)}\n")

    results = []
    for case in CASES:
        run, expect = case["run"], case["expect"]
        path = resolve(case["repo"], case["filename"])
        log(f">>> {run}  {case['filename']}  expect={expect}")
        log(f"    {case['note']}")
        if path is None:
            log(f"    *** fixture not in cache under {case['repo']}")
            results.append((run, expect, "MISSING-FIXTURE", ""))
            log("")
            continue
        log(f"    path {path}  ({path.stat().st_size} bytes)")

        # Header inspection first: it has zero heavy dependencies (PLAN §4.6) and
        # tells us the convention even when validation goes on to refuse.
        try:
            insp = inspect_lora(path)
            log(f"    inspect: convention={insp.convention!r} keys={len(insp.keys)} "
                f"shapes={len(insp.shapes)}")
            log(f"      prefixes      = {insp.prefixes}")
            log(f"      model_version = {insp.model_version!r}")
            log(f"      metadata keys = {sorted(insp.metadata)}")
            log(f"      first 3 keys  = {insp.keys[:3]}")
        except Exception as exc:
            log(f"    inspect raised {type(exc).__name__}: {exc}")

        # ---- stages 1-6.5 ----
        verdict, stage, message = None, "", ""
        try:
            report = validate_lora(path, transformer=transformer)
            verdict = "ACCEPTED"
            log(f"    VALIDATED: convention={report.convention!r} "
                f"model_version={report.model_version!r}")
            for w in report.warnings:
                log(f"      warning: {w}")
        except Exception as exc:
            verdict = "REJECTED"
            stage = STAGE_OF.get(type(exc).__name__, f"unmapped ({type(exc).__name__})")
            message = str(exc)
            log(f"    REJECTED at stage {stage}")
            log(f"      {type(exc).__name__}: {message}")

        # ---- stage 7, only for something that passed 1-6.5 ----
        if verdict == "ACCEPTED":
            try:
                attach = lora_apply.apply_lora(
                    pipe, path, adapter_name=f"host_{run.lower()}", weight=1.0
                )
                log(f"    STAGE 7 ATTACHED: {attach.attached} modules "
                    f"across {attach.components}")
                log(f"      per_component = {attach.per_component}")
                if attach.attached == 0:
                    log("    *** attached to ZERO modules")
                # Detach so the next case starts from the base model rather than
                # inheriting this adapter — otherwise every later verdict is
                # measured against a silently modified transformer.
                try:
                    pipe.delete_adapters(f"host_{run.lower()}")
                    log("      detached (delete_adapters)")
                except Exception as exc:
                    log(f"      could not detach: {type(exc).__name__}: {exc}")
            except Exception as exc:
                verdict = "STAGE7-FAILED"
                stage = STAGE_OF.get(type(exc).__name__, f"unmapped ({type(exc).__name__})")
                message = str(exc)
                log(f"    STAGE 7 FAILED at stage {stage}")
                log(f"      {type(exc).__name__}: {message}")
                traceback.print_exc()

        ok = (expect == "ACCEPT" and verdict == "ACCEPTED") or (
            expect == "REJECT" and verdict == "REJECTED"
        )
        log(f"    => {verdict}  (expected {expect})  {'OK' if ok else 'UNEXPECTED' if expect != 'UNKNOWN' else 'recorded'}")
        results.append((run, expect, verdict, stage))
        log("")

    log("=== SUMMARY ===")
    for run, expect, verdict, stage in results:
        log(f"  {run:<6} expect={expect:<8} got={verdict:<14} stage={stage or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
