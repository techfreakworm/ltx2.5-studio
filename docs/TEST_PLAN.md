# LTX-2.5 Studio — Test Plan

Two matrices, for two different questions. **Neither answers the other's.**

| | **Part A — hinode** | **Part B — ZeroGPU** |
|---|---|---|
| answers | **does the app work?** | does it fit the platform? |
| hardware | RTX PRO 6000 Blackwell, **unpartitioned**, 97887 MiB | same silicon, MIG-partitioned |
| RAM / disk | 124 GB / 136 GB persistent | ~68 GB / 150 GB, wiped on cold start |
| budget | **wall-clock only** | 2400 GPU-s/day |
| status | **ACTIVE — the whole objective** | ⏸️ **PARKED** |

**Label every claim.** A mode that works here is **HOST-PROVEN**. Whether it fits
ZeroGPU is **ZEROGPU-PENDING**, and nothing in Part A can speak to it.

**`host-runner` executes this matrix. Nobody else drives the box.**

---

# Part A — hinode

**The objective, in one sentence: every one of the 11 modes produces a real file
you can open, and the LoRA validator works in both directions against the real
adapters.**

There is no quota here, so the economies that shaped Part B are gone. Part B
grouped modes by shared code path and skipped the ones already covered. **Do not
do that here.** Each mode gets its own run and its own artifact, because the
operator wants to see each one work — and because the failures this build has
actually produced were things that *ran and were wrong*, which only looking at
the output catches.

## A.0 Preconditions — check these before the first run

| check | why it matters |
|---|---|
| `HF_TOKEN` **exported in the launching environment** | not merely present in an interactive shell — it must survive `nohup`/`tmux`. Without it, startup fails at `model_index.json (1 of 37; GatedRepoError)` |
| startup log: `will_use_gpu=True` | if `False` the app does **no GPU work at all** and the UI still comes up |
| startup log: `weights cached : 37/37 files` | anything less means an incomplete download; the failure would surface later and more confusingly |
| startup log: `dtype : bf16 (fixed; no precision switch)` | there is no precision lever — if you are looking for one, see PLAN §2.6 |
| `pipe_a.transformer is pipe_b.transformer` | **identity, not equality.** Two copies is 134 GiB against a 96 GiB card — an OOM on the second build |
| `audio_scheduler` present on the condition pipeline | the one component the two classes do not share. If sharing dropped it, the joint audio degrades in a way **the video never reveals** |

## A.0.1 ⛔ GATE — smoke test before anything else

**Run `smoke_t2v.py` and get a file out of it before starting H1.** Do not skip
this because the quota constraint is gone.

Part B's *smallest thing that can fail first* principle still holds — what changed
is the currency, not the logic. A broken stack should announce itself in **two
minutes**, not twenty, and H1 at 704×480×121 is a twenty-minute way to discover
that the model did not load. Losing cheap-first-failure was a side effect of
dropping the quota, not a decision.

**If the smoke test fails, nothing below it means anything.** Stop and diagnose.

## A.1 Resolution and length

**Baseline: 704 × 480, 121 frames** (16 latents). Part B's 256×256×9 existed only
to ration quota and should not be reused here — **a 9-frame clip passes several
things by accident**: the `num_frames % 8 == 1` arithmetic, the pixel→latent
anchor conversion, and the stereo mux all have more room to be wrong at 121
frames than at 9.

All values verified legal against `constraints.py`: 704×480 and 1216×704 are both
multiples of 32; 121 and 241 frames are both `k·8+1`.

## A.2 The matrix — 11 modes, each producing a file

**"Expected streams" is the pass criterion**, not "it produced a file". A mode
that declares `exports=("audio","video")` and hands back a silent video is the
exact defect that shipped here once already.

| # | Mode | Inputs | Config | **Expected streams in the artifact** |
|---|---|---|---|---|
| H1 | `text_to_video` | prompt | baseline | **video only** — h264. No audio stream |
| H2 | `image_to_video` | image | baseline | **video only** |
| H3 | `image_text_to_video` | image + prompt | baseline | **video only** |
| H4 | `video_to_video` | clip, strength 0.7 | baseline | **video only** |
| H5 | `text_to_audio_video` | prompt | baseline | **video + audio** — h264 + aac, **2 ch**, rate from the vocoder |
| H6 | `image_to_audio_video` | image | baseline | **video + audio**, 2 ch |
| H7 | `image_text_to_audio_video` | image + prompt | baseline | **video + audio**, 2 ch |
| H8 | `text_to_audio` | prompt | min res, stage 2 skipped | **audio only** — no video stream in the file |
| H9 | `video_to_audio` | clip | min res, stage 2 skipped | **audio only** |
| H10 | `audio_to_audio` | audio clip | min res, stage 2 skipped | **audio only** |
| H11 | `audio_to_video` | audio clip | baseline | **video only, no audio stream.** The audio is an *input* here; encoding it is a cost, not an output |

**H8–H10 still render video internally** and discard it — that is the model, not a
bug. Expect them to take as long as a video mode. `GenerationResult.warnings`
should carry the dropped-audio note naming the counterpart mode.

**H1–H4 and H11 must have NO audio stream.** If `ffprobe` shows one, `exports` is
not deciding the container's contents and the contract is one-directional again.

## A.3 The LoRA validator — both directions, real adapters

Stage 7 needs a real load, so it belongs here rather than in the offline suite.

| # | Adapter | Expected |
|---|---|---|
| H12 | `joyfox/LTX-2.3-Transition-LORA` (0.39 GB, ungated) | **ACCEPTED.** Mixed dims — 4096 at attention, 2048 at `audio_ff`, 16384 at `ff.net.2` — and none of that is a mismatch. Stage 7 reports **> 0 modules attached** |
| H13a | `charlied98/ltxv_2b_lora_soft_draw_v1` → **`lora_weights_step_03000.safetensors`** | **REJECTED at stage 6.5**, `LoRADimensionError`: `2048 at transformer_blocks.0.attn1.to_k, expected 4096` |
| H13b | same repo → **`comfy_lora_weights_step_03000.safetensors`** | **REJECTED at stage 6.5**, same error |

**H12 is the false-positive guard.** It contains 2048-wide tensors *at the audio
modules, where 2048 is correct*. A validator that rejects it has learned "contains
2048" instead of "2048 where 4096 belongs", and would reject the most-downloaded
LTX LoRA on the Hub.

**H13 is the whole point of the validator.** It passes stages 4, 5 and 6, passes
diffusers' only built-in check, and **would pass stage 7 too** — its module names
match. Only stage 6.5 catches it.

### Why H13 is two runs, and why the stage matters more than the verdict

The repo ships **two files of identical size and different key conventions**:

| file | prefix | convention | metadata |
|---|---|---|---|
| `lora_weights_step_03000.safetensors` | `transformer.` | **diffusers** | `{'format': 'pt'}` |
| `comfy_lora_weights_step_03000.safetensors` | `diffusion_model.` | **native / ComfyUI** | *absent* |

Both carry 448 tensors, clean `lora_A`/`lora_B` pairing, and **2048 in-dims**.

**"Rejected" is not a result when two mechanisms could produce it.** A file
rejected at stage 5 for an unrecognised convention would look identical in a pass
column to one rejected at 6.5 for a dimension mismatch — and the first would mean
the convention detector had *failed*, not succeeded.

**Predicted and then verified** by running the real validator against both
headers with a 48-layer 4096-dim transformer:

```
lora_weights_step_03000.safetensors        REJECTED at stage 6.5  LoRADimensionError
comfy_lora_weights_step_03000.safetensors  REJECTED at stage 6.5  LoRADimensionError
```

**Both converge on 6.5, which is the right answer**: the convention detector
accepted *both real conventions* and handed each to the dimension check. Until
now it had only ever been exercised by synthetic fixtures — this is the first
evidence it works on files somebody else wrote.

**Record the stage, not just the verdict.** A run of H13 that reports "rejected"
without naming `LoRADimensionError` has not distinguished success from the
convention detector breaking.

## A.4 The two features that need their own run

| # | Feature | Config | Expected |
|---|---|---|---|
| H14 | **stage 2** | `text_to_video`, **`enable_stage_2=True`** | upsample → 3-step schedule → diffusion decode. Visibly more detail than H1 at the same prompt/seed |
| H15 | **duration_head** | `text_to_video`, **omit `num_frames`**, `min_seconds`/`max_seconds` | length chosen by the model, within the bounds |

> ⚠️ **H14 must pass `enable_stage_2=True` explicitly.** It defaults **off**, so a
> run that omits it quietly exercises the stage-1 path **and reports a pass** —
> spending the run to prove the thing it was designed to test never happened.
> Compare against H1 at the same seed; if they are identical, stage 2 did not run.

## A.4.1 The undeclared-export direction — covered offline, no run needed

The exports contract is enforced **both ways**: declared-but-missing raises
`EmptyOutputError`; written-but-undeclared is dropped and noted on
`GenerationResult.warnings`.

**Every run in A.2 proves the first direction. None proves the second**, and it is
the direction that silently discards a real artifact.

**It does not need a GPU run** — it needs an encoder that returns more than the
mode declares, which is a fake, not a model. Already covered:

```
tests/test_exports_contract.py::test_a_track_the_mode_never_declared_does_not_reach_the_result
    11 passed — parametrised across every mode
tests/test_ui_generate_seam.py — the same contract from the handler's side
```

**Recorded here so the gap is visibly closed rather than merely absent.** If those
tests are ever removed, this line becomes false and nothing else in Part A would
notice.

## A.5 One high-resolution run

| # | Mode | Config | Purpose |
|---|---|---|---|
| H16 | `text_to_audio_video` | **1216 × 704**, 121 frames | the resolution a user would actually choose. Everything above is a correctness check; this is the one that says whether it is usable |

## A.6 Run log

| Run | Mode | Result | Wall clock | **Streams observed** | **Output looks right?** | Notes |
|---|---|---|---|---|---|---|
| **S0** | **smoke_t2v.py — GATE** | — | — | — | — | **run this first** |
| H1 | text_to_video | — | — | — | — | not run |
| H2 | image_to_video | — | — | — | — | not run |
| H3 | image_text_to_video | — | — | — | — | not run |
| H4 | video_to_video | — | — | — | — | not run |
| H5 | text_to_audio_video | — | — | — | — | not run |
| H6 | image_to_audio_video | — | — | — | — | not run |
| H7 | image_text_to_audio_video | — | — | — | — | not run |
| H8 | text_to_audio | — | — | — | — | not run |
| H9 | video_to_audio | — | — | — | — | not run |
| H10 | audio_to_audio | — | — | — | — | not run |
| H11 | audio_to_video | — | — | — | — | not run |
| H12 | LoRA accepted | — | — | n/a | — | not run |
| H13a | LoRA rejected (diffusers conv.) | — | — | n/a | n/a | not run — must name the stage |
| H13b | LoRA rejected (native conv.) | — | — | n/a | n/a | not run — must name the stage |
| H14 | stage 2 | — | — | — | — | not run |
| H15 | duration_head | — | — | — | — | not run |
| H16 | high resolution | — | — | — | — | not run |

**Both bold columns are required.** *Streams observed* is checkable —
`ffprobe -show_streams` against the expected layout in A.2. *Output looks right?*
is the judgement, and it is the one no test can make: every failure this build
produced was something that ran, exited zero, and was wrong.

## A.7 What hinode cannot tell us

So a green Part A is not mistaken for a shippable Space: **no tensor packing, no
150 GB ceiling, 124 GB RAM against ~68, and the `spaces` integration is
first-contact.** All of Part A is **HOST-PROVEN / ZEROGPU-PENDING**.

---

# Part B — ZeroGPU: does it fit the platform?

> ⏸️ **PARKED.** Retained unchanged because it is the matrix we return to, and
> because its economics — a 2400 GPU-s/day budget — **do not apply to hinode and
> must not leak into Part A**. Findings to date are in
> [`docs/ZEROGPU_DISK.md`](ZEROGPU_DISK.md).

## B.1 The economics

| | |
|---|---|
| PRO quota | **2400 GPU-seconds/day** — **verified**: `pro_quota_minutes = 40` at `spaces/zero/client.py:190`. Fixed 24 h window from first use |
| Our tier | **`xlarge`** — forced, because packed weights are 67.23 GiB and `xlarge_threshold` is **42.00 GiB exactly** (`45097156608` bytes, **verified in the platform's own `spaces/zero/configs.json`**) |
| Multiplier | **assumed 2×** — ⚠️ **UNVERIFIED.** A 75 s run is budgeted at 150 GPU-seconds |
| Practical ceiling | **~16 runs/day** |
| **HARD STOP** | **1920 GPU-seconds (80%)** |

**Never run the tank dry.** The stop line exists so that a failure discovered on run 5 can still
be fixed and retested on the same day. A test budget with no reserve is a test budget that
guarantees the last bug found is the one you cannot verify fixed.

> ⚠️ **Reservation math is known. Billing math is NOT — and the unknown one sized this matrix.**
>
> **Known**, from `spaces` 0.51.1: `duration_factor = 1.5` at `client.py:112`, applied
> `if gpu_size != 'xlarge'`. So **`large` reservations are inflated 1.5× and `xlarge` are not.**
> In *reservation* terms that makes xlarge **cheaper** than large — the inverse of the
> assumption this matrix was built on.
>
> **Not known:** a 2× xlarge **billing** multiplier appears **nowhere** in `client.py`, `api.py`,
> `wrappers.py` or `decorator.py`. It may exist server-side. Recorded as unverified rather than
> deleted, because absence from the client is not absence from the platform.
>
> **The budget stays pessimistic at 2×.** Being wrong that way costs unused quota; being wrong
> the other way costs a reservation failure mid-matrix. Do not re-plan around the cheaper figure
> until a real run's **billed** seconds confirm it — **the first run is the measurement**, and it
> costs nothing extra to take.

### B.1.1 Two metering paths — they are not the same number

This distinction decides how we set `duration=`:

| path | compares | consequence |
|---|---|---|
| **Reservation** (pre-check) | the **requested** `duration=` against remaining quota | a generous `duration=` can be **refused up front** even when the run would have been cheap |
| **Billing** (after) | the **effective** duration × the size multiplier | you are charged for what you used, doubled |

So: **keep `duration=` tight.** An over-generous request costs nothing in billing and everything
in schedulability — late in the day, a `duration=300` reservation fails against 200 s remaining
while the actual 40 s run would have fit fine.

One trap from PLAN §5.8, now **confirmed at source**: `large` applies a hidden
**1.5× `duration_factor`** (so `duration=60` schedules as 90 s). `spaces/zero/client.py:112`
gates it on `gpu_size != 'xlarge'`, so it does **not** apply to us. The claim that `xlarge`
"doubles the request instead" is the unverified half — see the warning above. And the maximum duration is a **server-side, tier-dependent, unpublished cap** —
discover it, never hardcode it.

---

## B.2 What does NOT need a GPU

This is the largest cost saving in the document, so it comes before the matrix.

| capability | why no GPU |
|---|---|
| LoRA validator **stages 1–6** | header-only parsing. Real fixtures already captured (PLAN §4.5) — three real LoRA headers, read over HTTP range requests |
| Constraint math (`% 32`, `% 8 == 1`, CRF selection, sigma schedules) | pure functions over numbers |
| The mode registry | data; kwarg builders assert on dicts |
| `model_files.py` resolver + allow-list emission | data; the drift test compares README to declared paths |
| The never-empty-output invariant | stub pipeline returns nothing → assert a specific exception |
| `zerogpu.py` return boundary and `duration=` signature match | no CUDA needed to prove a CUDA tensor is rejected |
| Preflight RAM/disk checks | inject fake numbers |
| **Audio-only export (modes 8, 9, 10, 11)** | see §2.1 |

### B.2.1 Modes 8–11 need NO dedicated GPU run

This is the clearest illustration of "code paths, not tabs".

Modes 8 (text→audio), 9 (video→audio), 10 (audio→audio) and 11 (audio→video) share their entire
*pipeline* path with R1, R4 and R5. The **only** thing unique to them is which track gets
exported at the end — and muxing/demuxing an already-produced tensor is offline-testable with a
fixture.

Spending four GPU runs on them would buy nothing but four tabs turning green.

### B.2.2 LoRA stage 7 needs a load, not a generation

Stage 7 — "assert the adapter attached to > 0 modules" — needs **real weights in memory**, but it
does **not** need a completed denoise. It is folded into **R3** rather than given its own run, so
the same 120 GPU-seconds proves both the attach assertion and that generation still works with an
adapter applied.

---

## B.3 The minimal legal configuration

Verified against `check_inputs`: **the only dimension rule is `height % 32 == 0` and
`width % 32 == 0`.** There is **no minimum resolution**, and `num_frames % 8 == 1` is not checked
upstream at all (PLAN §5.5 — which is exactly why *we* check it).

**Therefore the cheapest legal proving run is `256 × 256`, `num_frames = 9`.**

Quality will be bad. That is correct and intended: these runs prove **code paths**, not output
quality. A showcase run happens at the end, only if quota remains (§B.6).

---

## B.4 The matrix — 7 runs, 11 modes, ordered by risk

Ordered so that **the most likely failure surfaces while the most quota remains**. R1 is first
because it is the riskiest thing we have: if the packer, the worker, or the diffusion decode is
broken, nothing else matters and we want to know at 120 GPU-seconds spent, not 700.

| Run | Config | `duration=` | Budget (GPU-s) | Proves — the path **nothing else proves** | Modes | **Actual** |
|---|---|---:|---:|---|---|---|
| **R1** | t2v · 256×256 · 9 frames · distilled 8-step sigmas · no stage 2 | 60 | 120 | Pipeline loads → **packs** → forked worker runs → stage-1 denoise → **diffusion decode** → muxed mp4 **with audio**. The riskiest single path in the app. | 1, 5, 8 | _(pending)_ |
| **R2** | i2v · same tiny config · one condition at index 0 | 60 | 120 | `LTX2VideoCondition` construction, VAE encode of the condition, the **CRF-18 recompression** branch, and the **pixel→latent index** math | 2, 3, 6, 7 | _(pending)_ |
| **R3** | R2 config **+ two adapters** (§5) | 60 | 120 | **Both directions in one call.** (a) the 2.5 adapter loads, attaches to **> 0 modules**, and generates — the headline feature works; (b) the 2B adapter is then attempted and **rejected loudly and specifically** — the anti-silent-garbage guard fires. | LoRA | _(pending)_ |
| **R4** | same tiny config + a short input audio clip | 60 | 120 | audio-in: waveform → mel → `audio_vae` → **4D `audio_latents`** + `noise_scale` | 10, 11 | _(pending)_ |
| **R5** | same tiny config + a short input video | 60 | 120 | **video** conditioning: a multi-frame condition — note multi-frame is **never** CRF-recompressed, unlike single-frame, so this is a genuinely different branch from R2 | 4, 9 | _(pending)_ |
| **R6** | R1 config **+ `enable_stage_2=True`** | 90 | 180 | The stage-2 chain: `LTX2LatentUpsamplePipeline` → `STAGE_2_DISTILLED_SIGMA_VALUES` → `noise_scale` → **`denormalize=False`** on the diffusion decode | quality path | _(pending)_ |
| **R7** | R1 config, **no `num_frames`**, tight `min_seconds`/`max_seconds` | 75 | 150 | `duration_head` auto-duration | auto-duration | _(pending)_ |
| | | | **930** | | | |

**Budget: 930 GPU-seconds of a 1920 stop line — 990 s (~8 runs) held in reserve for retests.**

**Deviation from the uniform-60 s estimate, stated plainly:** the lead's baseline was 7 × 120 =
840 GPU-s. I raised two runs, costing 90 GPU-s more:

**Setting up R6:** stage 2 is opt-in and defaults **off** (PLAN §7.4), so the run must pass
`enable_stage_2=True`. Note the registry overrides the request — a mode with `skip_stage_2` set
forces it off, so R6 must use a video-exporting mode. R1's `text_to_video` qualifies.

- **R6 → 90 s.** The stage-2 chain does strictly more work than R1 (an upsample pass plus three
  further denoise steps on a larger latent). Requesting 60 s risks a timeout — and a timeout
  costs the full reservation *and* proves nothing.
- **R7 → 75 s.** With `num_frames` omitted, the duration head chooses the length. Even with tight
  `min_seconds`/`max_seconds` it may pick more than 9 frames, so the run is not bounded by our
  config the way the others are. **Pass the tightest legal `min_seconds`/`max_seconds`** to keep
  this from being the expensive one.

Both are still tight enough to reserve comfortably (§B.1.1).

---

## B.5 The R3 fixtures — one positive, one negative

R3 proves **both directions in a single GPU call**: that a good adapter works, and that a bad one
is refused. All candidates were assessed by HTTP **range-reading their safetensors headers** — no
full downloads.

### B.5.1 Rejected: the official distilled LoRA

`ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (9.699 GB) is documented for **DEV-transformer**
workflows. **We ship the distilled transformer**, so this is the wrong adapter for our
configuration and a passing result would be misleading. Already excluded from the disk budget
(PLAN §2.4) for the same reason.

### B.5.2 Rejected: the official IC-LoRAs

`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` (0.327 GB) and the 2.3 IC-LoRA family are
**separately gated** and the operator's token does not have them — verified, they 403. Not worth
a second gate request, because §5.3 is a better fixture anyway.

### B.5.3 **Positive fixture: `AhsanHareem/reelbids-ltx25-camera-lora`**

File `lora_weights_step_01250.safetensors`.

| | |
|---|---|
| Size | **201 MB** — downloads inside a run |
| Access | **ungated** |
| Generation | an actual **LTX-2.5** camera LoRA |
| Shape | 768 tensors, prefix `diffusion_model.`, rank 32, `to_q.lora_B` = `[4096, 32]` → **hidden dim 4096, matching our 22B transformer** |
| Metadata | **`__metadata__` present but EMPTY (`{}`)** — no `model_version` |

**Why this and not the official adapter.** `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` is
8.9 GB and documented for **DEV-transformer** workflows; we ship the **distilled** transformer,
so it is the wrong adapter for our configuration and a passing result would be misleading. It is
already excluded from the disk budget (PLAN §2.4) for the same reason.

**Why this is a *better* fixture, not merely an available one.** Its empty `__metadata__` makes it
a realistic stand-in for what users actually upload, and it exercises stage 4's
absent-is-not-fatal branch against a real file. Note the distinction the code must honour:
metadata **present but empty** is a different shape from metadata **absent**, and both mean "no
version signal" (PLAN §4.5).

### B.5.4 **Negative fixture: `charlied98/ltxv_2b_lora_soft_draw_v1`**

An LTX-**Video-2B** adapter — flatly the wrong architecture, and the reason PLAN §4.3 has a stage
6.5 at all. It passes stages 4, 5 and 6, passes diffusers' only built-in check, and may even pass
**stage 7** because its module names match ours. The single thing wrong with it is
`to_k.lora_A = [32, 2048]` where this model is 4096.

R3 attempts it **after** the positive case and asserts a loud, specific rejection naming module,
expected and actual dims. One run, both directions.

### B.5.5 Rejected candidates

| candidate | why not |
|---|---|
| `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` | **separately gated**, token does not have it (403) |
| `Lightricks/LTX-2.3-22b-LoRA-Cinemagraph` | **403** — another separately gated repo. Do not depend on it |
| `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8.9 GB, DEV-transformer workflow — wrong adapter for our distilled config |

**Offline-only fixtures** (no GPU run, used for stages 1–6.5):
`joyfox/LTX-2.3-Transition-LORA` (0.390 GB — the mixed-dimension case that proves 6.5 must be
per-module) and `vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa` (0.705 GB — the
metadata-present case, `ss_ltx_version: "2.3"`, `modelspec.architecture: "LTX2/lora"`).

**Real fixtures are better where they apply -- but synthetic ones are not optional.** An earlier
draft of this section said "no synthetic adapter is needed". That was wrong, and the distinction
matters:

| what it proves | fixture |
|---|---|
| the validator accepts a real adapter, rejects a real wrong-architecture one | **real** files (above) |
| a **header-truncated** file, a **corrupt length prefix**, a header that is not JSON | **synthetic only** -- no real adapter is corrupt on demand |
| the payload is **provably never read** | **synthetic only** -- it needs a known payload size to count bytes against |

You cannot download a broken file that is broken in exactly the way you want to test. The
stdlib-built fixtures (`struct` + `json`, PLAN §4.6) are the only way to construct those cases,
and they are what makes stages 1-3 testable at all.

---

## B.6 Showcase runs

**Only if quota remains after the matrix is green**, and only with the stop line respected. A
showcase run at full resolution is a *demo*, not a test — it must never displace a retest.

---

## B.7 Rules of engagement

1. **Log actual cost after every run** — requested `duration=`, `size`, and **real** GPU-seconds
   — into §8. A budget nobody reconciles is a budget nobody has.
2. **Stop at 1920 GPU-seconds.** No exceptions, no "just one more".
3. **Never auto-retry a timeout.** PLAN §5.8 #5: the captured Gradio request's `X-IP-Token` ages
   past its TTL during attempt 1, so attempt 2 always 401s with *"Expired ZeroGPU proxy token"* —
   **masquerading as the root cause** and burning a second reservation to tell you a lie.
4. **A failed run still costs quota.** Before re-running, fix something. Re-running unchanged
   code to see if it fails differently is how a 1920 s budget becomes a 0 s budget.
5. **Run in matrix order.** R1 first. It is ordered by risk deliberately.
6. **One variable per run.** Each row changes exactly one thing from a run that already passed —
   which is what makes a failure diagnostic rather than merely bad news.

---

## B.8 Run log

Filled in after each run. Running total maintained here.

| Run | Date | `duration=` | size | **Actual GPU-s** | Cumulative | Result | Notes |
|---|---|---:|---|---:|---:|---|---|
| R1 | — | — | — | — | — | not run | |
| R2 | — | — | — | — | — | not run | |
| R3 | — | — | — | — | — | not run | |
| R4 | — | — | — | — | — | not run | |
| R5 | — | — | — | — | — | not run | |
| R6 | — | — | — | — | — | not run | |
| R7 | — | — | — | — | — | not run | |

**Budgeted: 930 · Spent: 0 · Stop line: 1920 · Remaining to stop line: 1920**
