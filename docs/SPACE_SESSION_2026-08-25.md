# The 2026-08-25 session: byte identity settled, a shippable resolution proven, and the stage-2 ceiling found

> ## ✅ Four headline results, each a measurement
>
> 1. **Byte-level reproducibility is now VERIFIED by sha256** — a fresh
>    `text_to_video` 256×256×9 seed-1234 run hashed **equal to the surviving
>    2026-08-22 artifact** (`b78803cc…212d`), on a third container and a
>    different code sha — and then verified **again** through the redesigned UI
>    on the tiled-decode build (§3). The previously-unverifiable claim in
>    `SPACE_MODE_MATRIX.md` §5 resolves in the happy direction.
> 2. **A shippable resolution generates on ZeroGPU**: 1216×704×121 — after one
>    diagnosed failure at the VAE decode and the tiled/framewise fix
>    (`cdb3211`) it produced 6,258,833 B of real, 121-distinct-frame video (§4).
> 3. **Stage 2 ran on the Space for the first time — and found its ceiling**:
>    PASS at 704×480×25 input (output geometry verified 1408×960), FAIL at
>    ×49 in the diffusion decoder. The ceiling sits **between 25 and 49 input
>    frames at 704×480** (§5).
> 4. **Conditioning demonstrably drives the output**: same seed, same prompt,
>    two different conditioning clips → mean |Δ| 17.277 vs **0.864**, and the
>    output's motion statistics mirror the input's (§7). The null hypothesis —
>    prompt-only generation ignoring the clip — is dead.
>
> Also proven live: both halves of the fork-boundary error fix (§2), both LoRA
> directions with a passing inertness check and route equivalence (§6), and the
> adapter-detach lifecycle (§3, R8).

**Every load-bearing number is marked:** ✅ *we measured it* · 📄 *read off a
pinned source, the platform, or the campaign record* · ⚠️ *inferred* ·
⏸️ *not established*.

> ### ⚠️ The two rules this document is organised around
>
> **A hash is the claim.** Every byte-identity statement here is a sha256
> comparison against a banked artifact, re-runnable by anyone: the bank's
> directory names *are* the hashes, and the banked copies were re-hashed at
> write-up time and match ✅. Nothing below says "identical" on size or
> statistics alone — that mistake was made once and corrected in
> `SPACE_MODE_MATRIX.md` §5.
>
> **A wall-clock figure is not a billing figure.** No readable quota endpoint
> exists 📄, so every cost is `wall × 2` (xlarge) and bounds spend from above;
> it is written `<=` everywhere it appears.

⚠️ **Run ids in this document are the session's own (R1a…R9). They are NOT
`TEST_PLAN.md` Part B's R1–R7** — the numbering collides and the meanings
differ (Part B's R3 is LoRA; this session's R3 is 1216×704).

---

## 1. What ran

| | |
|---|---|
| Space | `techfreakworm/LTX2.5-Studio`, private, ZeroGPU, stage RUNNING 📄 |
| Deploy 1 | Space sha `8e9f7a1` 📄 — the morning hardening tip: fork-boundary pair `6bb211a` through `f0daa75` |
| Deploy 2 | Space sha `ff142d9` 📄 — adds `cdb3211` (tiled + framewise VAE decode), `0be0014` (inbound report picklability), `b9ee967` (Hub repo-id LoRA fetch), `2eaeb2c` (audio-length reconcile), `3a54499` + `56983b8` (UI redesign) |
| Branch tip at write-up | `56983b8`; offline gate **2344 passed / 13 skipped** ✅ |
| Account | PRO 📄, 2400 GPU-s/day |

Deploy-1 runs: R1a, R1b, R-repro, R2, R3 (first attempt), R7a, R7b, R5, R6
(first attempt). Deploy-2 runs: R3 retry, R4a, R4b, R6 retry, R8, R9. 📄

Artifacts are banked under `runs_0825/`, one directory per run, **directory
name = the artifact's sha256** ✅. The three failed runs (R1a, R3 first
attempt, R4b) banked no artifact — their directories are empty, which is
correct: no file was produced.

---

## 2. The error paths — the fork-boundary fix proven live, in both classes

`6bb211a`'s pair (`SPACE_MODE_MATRIX.md` §4) had been verified offline only.
Deploy 1 exercised it against the real platform, deliberately, before anything
expensive:

| run | provocation | what came back to the client | wall ✅ | cost |
|---|---|---|---|---|
| **R1a** | a deliberately **truncated mp4** as the conditioning upload | **`ConstraintError`** — our class, our message, the file path, and the wrapped `av` errno, all intact | 7.44 s | `<=` 14.9 |
| **R1b** | `num_frames=10` sent by direct client call, past the UI | **`FrameCountError`** — with its full remedy sentence | 5.17 s | `<=` 10.3 ⚠️ manual — bypassed the runner's ledger |

✅ **Both halves of the design are now proven on the platform, not just against
the read source**: R1a is a message-carrying class surviving the fork OUTBOUND
(the exact loss §4 documented — `'DecodeError'` arriving as eleven characters —
does not recur), and R1b is one of the 22 keyword-only classes that could not
previously pickle at all. The failing-run legibility paid again the same day:
R3's decode failure (§4) and R6's dispatch failure (§6) both arrived
client-side as readable, attributable errors.

---

## 3. Reproducibility — sha256, four comparisons, all identical

### R-repro — the planned close of `SPACE_MODE_MATRIX.md` §5, run as planned

`text_to_video` 256×256×9, seed 1234, the banked configuration. 16.79 s wall ✅,
`<=` 33.6 GPU-s.

```
sha256(fresh artifact)  = b78803cc998645dde06007c70cfa13d30af0ed37dd87e31dcc1d068da4aa212d
sha256(2026-08-22 bank) = b78803cc998645dde06007c70cfa13d30af0ed37dd87e31dcc1d068da4aa212d
IDENTICAL — 39,146 bytes
```

✅ **Byte identity is now a verified claim** — third container, different code
sha (`8e9f7a1` against the bank's `8fb292d6`-era build). What §5's correction
said could no longer be checked against the *dropped day-1 file* is settled the
only way it could be: by direct comparison against the surviving artifact.

### R7a — the condition pipeline is deterministic too

`video_to_video` at the 2026-08-22 configuration: same conditioning clip, seed
1234. 18.12 s ✅, `<=` 36.2.

```
sha256 = ef8571e629aac853f748102e976ff0ca6e15f691c7bc151fb1af2ae729ae4c8f
IDENTICAL to the banked 2026-08-22 video_to_video artifact — 33,812 bytes
```

✅ This doubles as the **control** for §7's discriminator: under a fixed seed
and fixed inputs the noise floor is **exactly zero**, so any difference in R7b
is attributable to the one thing that changed.

### R8 — the same hash through everything that changed today

`text_to_video`, banked configuration, fresh session, deploy 2: **through the
redesigned UI** (`3a54499`/`56983b8`), **on the tiled-decode build**
(`cdb3211`), **on a warm worker that had just carried a LoRA adapter** (R6).
15.78 s ✅, `<=` 31.6.

```
sha256 = b78803cc…212d — IDENTICAL again
```

Three claims in one comparison ✅: the new UI serves real generations;
`cdb3211`'s dispatch-gated tiling leaves the ≤432² path untouched **by
construction, now confirmed on CUDA**; and `de78923`'s detach-after-every-run
lifecycle holds live — no adapter leakage into the next run.

### R9 = R6 — the LoRA path is deterministic across routes

Both LoRA generations (upload route and Hub repo-id route, §6) produced
**byte-identical** output: `sha256 20a3a84b…763f`, 39,543 B, `cmp` clean ✅.

⏸️ **Scope**: all identity claims are at the configurations named. Nothing here
extends byte-identity to other modes, geometries, or stage 2.

---

## 4. The resolution ladder — 704×480×49 clean, 1216×704×121 via the tiled decode

| run | config | verdict | wall ✅ | cost | artifact ✅ |
|---|---|---|---|---|---|
| **R2** | t2v 704×480×49 | ✅ PASS | 20.89 s | `<=` 41.8 | 1,007,581 B · 49/49 distinct · mean \|Δ\| 19.408 |
| **R3** | t2v 1216×704×121, deploy 1 | ⛔ FAIL at VAE decode | — | `<=` 103.6 | none |
| **R3 retry** | same, deploy 2 (tiled) | ✅ **PASS** | 66.47 s | `<=` 132.9 | 6,258,833 B · 121/121 distinct · mean \|Δ\| 19.482 |

**The R3 failure is fully characterised, and it was legible from the client** —
the fork fix earning its keep on the first real failure after it shipped:
denoise **succeeded** (~52 s ✅), then the VAE decoder raised
`RuntimeError` — an NVML_SUCCESS assert from the CUDA caching allocator
(`CUDACachingAllocator.cpp:1165`) 📄 — inside `conv3d` in
`autoencoder_kl_ltx2`, where full-resolution decode intermediates run
⚠️ ~26 GB. The client received our `DecodeError` with the wrapped reason
intact.

✅ **`cdb3211`'s tiled + framewise decode is the difference between the two R3
rows** — same request, same geometry, PASS through the tiled path at a decode
cost of roughly **+14 s** over the failing run's denoise-only wall ⚠️ (two
different runs, not a controlled pair). **A resolution a user would actually
choose now generates on ZeroGPU.**

---

## 5. Stage 2 on the Space — first ever, and its ceiling

| run | config | verdict | wall ✅ | cost | evidence |
|---|---|---|---|---|---|
| **R4a** | stage 2, 704×480×25 input | ✅ **PASS** | 41.49 s | `<=` 83.0 | **output geometry verified 1408×960** ✅ — exactly 2× — 322,272 B, sha `2345efaa…2581` |
| **R4b** | stage 2, 704×480×49 input | ⛔ **FAIL** | — | `<=` 55.4 | same NVML allocator assert, **this time in the diffusion decoder** |

R4a is the first stage-2 execution on this platform: the latent upsampler, the
extra denoise, **and the unmitigated diffusion decoder** all survived at the
25-frame input. R4b failed one rung up, in
`LTX2VideoDiffusionDecoderModel` — which **exposes no tiling API at the pinned
diffusers SHA** 📄, exactly as `cdb3211`'s report predicted when it could tile
only the VAE.

> ⛔ **The stage-2 ceiling on ZeroGPU sits between 25 and 49 input frames at
> 704×480.** One attempt per failure, by the operator's ruling — documented,
> not chased. The follow-up is a **diffusion-decoder chunking investigation**
> (§9), which is its own piece of work, not a knob.

⏸️ Where in [25, 49] the ceiling falls, and how it scales with input area, are
unmeasured.

---

## 6. LoRA — both directions, an inertness check, and one inbound regression found and fixed

| run | what | verdict | wall | cost |
|---|---|---|---|---|
| **R5** | upload-direction validate (stages 1–6) | ✅ ACCEPTED | — | **0 GPU** ✅ (web process) |
| **R6** first | upload validate + generate with staged adapter, deploy 1 | ⛔ **FAIL INBOUND** at dispatch | 2.05 s | `<=` 4.1 ⚠️ manual |
| **R6** retry | same, deploy 2 | ✅ **PASS** | 17.10 s | `<=` 34.2 ⚠️ manual |
| **R9** | **Hub repo-id direction**: validate + generate, `AhsanHareem/reelbids-ltx25-camera-lora` | ✅ ACCEPTED + PASS | 22.33 s | `<=` 44.7 ⚠️ manual |

**The R6 failure was the fork boundary again — the other direction.**
`PicklingError: cannot pickle 'mappingproxy'` at dispatch: the staged-LoRA
report's `inspection.metadata`/`.shapes` carried mappingproxy views INBOUND
into the worker. Diagnosed offline, fixed at `0be0014` (proxy invariant
restored). ✅ The retry passed on deploy 2. `6bb211a` fixed the outbound
crossing; this was the inbound payload's first real exercise, and it found the
one unpicklable thing left in it.

✅ **The inertness check passed** — the claim `TEST_PLAN.md` H12b exists for,
delivered on the platform: R6's output (39,543 B, sha `20a3a84b…763f`) differs
from the same-configuration no-adapter baseline (39,146 B, `b78803cc…212d`).
**The adapter demonstrably changed the output.** Injection overhead ⚠️ ≈
negligible — wall within 0.3 s of a base run.

✅ **R9 proves the repo-id direction end to end**: the bounded web-side Hub
fetch (`b9ee967`) resolved and validated in 3.7 s at **zero GPU cost**, and its
generation came back **byte-identical to R6's** — both LoRA routes work, they
are **route-equivalent**, and the LoRA path is deterministic (§3).

---

## 7. R7b — conditioning drives the output

Same seed, same prompt, same geometry as the R7a control; **the only change is
the conditioning clip** — the near-static bank clip (mean |Δ| 0.905) in place
of the active one (18.894):

| | conditioning clip mean \|Δ\| | output mean \|Δ\| ✅ | output size ✅ |
|---|---:|---:|---:|
| R7a control (active clip) | 18.894 | **17.277** | 33,812 B |
| R7b discriminator (near-static clip) | 0.905 | **0.864** | 6,111 B |

24.60 s ✅, `<=` 49.2. With R7a's zero noise floor (§3), the difference is
attributable to the clip alone.

✅ **The output's motion statistics mirror the conditioning clip's.** The null
hypothesis — that `video_to_video` ignores its clip and generates from the
prompt — predicted ~17 from both runs. It is dead. This also **reclassifies
`SPACE_MODE_MATRIX.md` §6's near-static observation**: a near-static output
from a near-static input is the conditioning *working*, not a defect.

> ⏸️ **The limit, stated as prominently as the result: motion mirroring is not
> full semantic fidelity.** This measures that the clip's *motion energy*
> reaches the output. Whether the output depicts the *content* of the clip —
> its scene, its objects, its camera path — is a different claim, and no test
> in this session reads content.

---

## 8. Cost

```
deploy 1   R1a 14.9 · R1b 10.3 · R-repro 33.6 · R2 41.8 · R3 103.6
           R7a 36.2 · R7b 49.2 · R5 0 · R6 4.1            <= ~293.7
deploy 2   R3retry 132.9 · R4a 83.0 · R4b 55.4
           R6retry 34.2 · R8 31.6 · R9 44.7               <= ~381.8
                                                          -----------
today (UTC 2026-08-25), all of it                         <= ~675.5  GPU-s
budget   2400 GPU-s/day (PRO) 📄                          ->  <= ~28.1% consumed
ceiling  1920 GPU-s (80%)                                 ->  never approached
```

Per-run bounds are `wall × 2` rounded to 0.1; the sum of unrounded values is
675.46, so the day's bound is written **≤ ~675.5**. ⚠️ Loose in the safe
direction — wall clock includes queueing, upload, download and client overhead.

✅ **The ledger's carried-spend accounting worked in production.** The
cross-invocation ledger (`62d612f`) carried spend forward across runner
invocations all day — the mid-day blind spot recorded in
`SPACE_MODE_MATRIX.md` §7/§8 follow-up #10 did not recur for runner-driven
runs. ⚠️ Its remaining blind spot is **manual runs**: R1b, R6 (both attempts)
and R9 were direct client calls that bypassed the ledger, and they are named
and bounded here by hand — `<=` ~93.3 GPU-s of the total.

⏸️ Exact billed seconds remain unmeasurable — the quota endpoints still 404.

---

## 9. What this session does not establish, and the follow-ups it creates

**Does not establish:**

* Where the stage-2 ceiling falls inside [25, 49] input frames, or how it
  scales with input area (§5).
* Semantic fidelity to a conditioning clip — §7 measures motion statistics,
  not content.
* Anything about `auto_duration` on the platform, or the audio-input
  discriminator (**H10b**) — both remain unexercised on ZeroGPU. With R6/R9
  having delivered H12b's claim and R7b having answered H4b's question via the
  clip-swap route, **H10b is the last discriminator standing**.
* Why the 2026-08-20 workers were SIGKILLed. **Documented-unexplained, by
  operator ruling** — route C (instrument the worker) stands designed and
  un-run, and no run today touched that question.

**Follow-ups created:**

| | item | status |
|---|---|---|
| 1 | **Diffusion-decoder chunking** — the stage-2 ceiling (§5). `LTX2VideoDiffusionDecoderModel` has no tiling API at the pinned SHA 📄, so this is an investigation (chunk the decode ourselves, or a different mitigation), not a configuration change. The one-attempt ruling spent R4b establishing the failure point; the next spend should be designed against it. | ⏸️ open |
| 2 | **Slim-projection dispatch payload** — trim what crosses the fork inbound to the fields the worker actually reads (the no-undo optimization proposed off the back of the `0be0014` diagnosis). To be taken with the UI owner, since the payload shape is theirs. | ⏸️ open |
| 3 | **`video_to_audio` at −19.5 dBFS** (`SPACE_MODE_MATRIX.md` §6.1/follow-up #9) — now **consistent-with** §7's finding: conditioning mirrors its input, and a gentle scene plausibly yields quiet audio. ⚠️ Consistent-with is not established-by; nothing today probed audio conditioning directly. Stays open as an observation. | ⏸️ open, observation |
| 4 | **H10b** — the audio-input same-seed `noise_scale` pair, the last discriminator (above). Cheap: two short runs at 256×256×9-class cost. | ⏸️ open |
