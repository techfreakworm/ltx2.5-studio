# LTX-2.5 Studio — GPU Test Plan

**This document must exist and be reported before the first GPU call.**

Companion to [`docs/PLAN.md`](PLAN.md). PLAN says what we build; this says **what we are allowed
to spend proving it works**, and it is a budget document as much as a test document.

**Governing principle: cover code paths, not tabs.** Eleven UI modes do not need eleven GPU
runs. Seven runs cover every distinct code path we have. Everything else is a unit test.

---

## 1. The economics

| | |
|---|---|
| PRO quota | **2400 GPU-seconds/day**, fixed 24 h window starting at first use |
| Our tier | **`xlarge`** — forced, because packed weights are 67.23 GiB and `xlarge_threshold` is 42.00 GiB (PLAN §8.6) |
| Multiplier | **2×**. A 75 s run costs **150 GPU-seconds** |
| Practical ceiling | **~16 runs/day** |
| **HARD STOP** | **1920 GPU-seconds (80%)** |

**Never run the tank dry.** The stop line exists so that a failure discovered on run 5 can still
be fixed and retested on the same day. A test budget with no reserve is a test budget that
guarantees the last bug found is the one you cannot verify fixed.

### 1.1 Two metering paths — they are not the same number

This distinction decides how we set `duration=`:

| path | compares | consequence |
|---|---|---|
| **Reservation** (pre-check) | the **requested** `duration=` against remaining quota | a generous `duration=` can be **refused up front** even when the run would have been cheap |
| **Billing** (after) | the **effective** duration × the size multiplier | you are charged for what you used, doubled |

So: **keep `duration=` tight.** An over-generous request costs nothing in billing and everything
in schedulability — late in the day, a `duration=300` reservation fails against 200 s remaining
while the actual 40 s run would have fit fine.

Two related traps from PLAN §5.8: `large` applies a hidden **1.5× `duration_factor`** (so
`duration=60` schedules as 90 s), while `xlarge` skips that factor but **doubles the request**
instead. And the maximum duration is a **server-side, tier-dependent, unpublished cap** —
discover it, never hardcode it.

---

## 2. What does NOT need a GPU

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

### 2.1 Modes 8–11 need NO dedicated GPU run

This is the clearest illustration of "code paths, not tabs".

Modes 8 (text→audio), 9 (video→audio), 10 (audio→audio) and 11 (audio→video) share their entire
*pipeline* path with R1, R4 and R5. The **only** thing unique to them is which track gets
exported at the end — and muxing/demuxing an already-produced tensor is offline-testable with a
fixture.

Spending four GPU runs on them would buy nothing but four tabs turning green.

### 2.2 LoRA stage 7 needs a load, not a generation

Stage 7 — "assert the adapter attached to > 0 modules" — needs **real weights in memory**, but it
does **not** need a completed denoise. It is folded into **R3** rather than given its own run, so
the same 120 GPU-seconds proves both the attach assertion and that generation still works with an
adapter applied.

---

## 3. The minimal legal configuration

Verified against `check_inputs`: **the only dimension rule is `height % 32 == 0` and
`width % 32 == 0`.** There is **no minimum resolution**, and `num_frames % 8 == 1` is not checked
upstream at all (PLAN §5.5 — which is exactly why *we* check it).

**Therefore the cheapest legal proving run is `256 × 256`, `num_frames = 9`.**

Quality will be bad. That is correct and intended: these runs prove **code paths**, not output
quality. A showcase run happens at the end, only if quota remains (§6).

---

## 4. The matrix — 7 runs, 11 modes, ordered by risk

Ordered so that **the most likely failure surfaces while the most quota remains**. R1 is first
because it is the riskiest thing we have: if the packer, the worker, or the diffusion decode is
broken, nothing else matters and we want to know at 120 GPU-seconds spent, not 700.

| Run | Config | `duration=` | Budget (GPU-s) | Proves — the path **nothing else proves** | Modes | **Actual** |
|---|---|---:|---:|---|---|---|
| **R1** | t2v · 256×256 · 9 frames · distilled 8-step sigmas · no stage 2 | 60 | 120 | Pipeline loads → **packs** → forked worker runs → stage-1 denoise → **diffusion decode** → muxed mp4 **with audio**. The riskiest single path in the app. | 1, 5, 8 | _(pending)_ |
| **R2** | i2v · same tiny config · one condition at index 0 | 60 | 120 | `LTX2VideoCondition` construction, VAE encode of the condition, the **CRF-18 recompression** branch, and the **pixel→latent index** math | 2, 3, 6, 7 | _(pending)_ |
| **R3** | R2 config **+ a real adapter** (§5) | 60 | 120 | **LoRA stage 7 end-to-end** — adapter loads and attaches to **> 0 modules** on the real transformer. The headline feature. | LoRA | _(pending)_ |
| **R4** | same tiny config + a short input audio clip | 60 | 120 | audio-in: waveform → mel → `audio_vae` → **4D `audio_latents`** + `noise_scale` | 10, 11 | _(pending)_ |
| **R5** | same tiny config + a short input video | 60 | 120 | **video** conditioning: a multi-frame condition — note multi-frame is **never** CRF-recompressed, unlike single-frame, so this is a genuinely different branch from R2 | 4, 9 | _(pending)_ |
| **R6** | R1 config **+ stage 2** | 90 | 180 | The stage-2 chain: `LTX2LatentUpsamplePipeline` → `STAGE_2_DISTILLED_SIGMA_VALUES` → `noise_scale` → **`denormalize=False`** on the diffusion decode | quality path | _(pending)_ |
| **R7** | R1 config, **no `num_frames`**, tight `min_seconds`/`max_seconds` | 75 | 150 | `duration_head` auto-duration | auto-duration | _(pending)_ |
| | | | **930** | | | |

**Budget: 930 GPU-seconds of a 1920 stop line — 990 s (~8 runs) held in reserve for retests.**

**Deviation from the uniform-60 s estimate, stated plainly:** the lead's baseline was 7 × 120 =
840 GPU-s. I raised two runs, costing 90 GPU-s more:

- **R6 → 90 s.** The stage-2 chain does strictly more work than R1 (an upsample pass plus three
  further denoise steps on a larger latent). Requesting 60 s risks a timeout — and a timeout
  costs the full reservation *and* proves nothing.
- **R7 → 75 s.** With `num_frames` omitted, the duration head chooses the length. Even with tight
  `min_seconds`/`max_seconds` it may pick more than 9 frames, so the run is not bounded by our
  config the way the others are. **Pass the tightest legal `min_seconds`/`max_seconds`** to keep
  this from being the expensive one.

Both are still tight enough to reserve comfortably (§1.1).

---

## 5. The R3 fixture — **`joyfox/LTX-2.3-Transition-LORA`**

R3 needs a real adapter. I searched the Hub and read candidate safetensors **headers** over HTTP
range requests — no full downloads.

### 5.1 Rejected: the official distilled LoRA

`ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (9.699 GB) is documented for **DEV-transformer**
workflows. **We ship the distilled transformer**, so this is the wrong adapter for our
configuration and a passing result would be misleading. Already excluded from the disk budget
(PLAN §2.4) for the same reason.

### 5.2 Rejected: the official IC-LoRAs

`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` (0.327 GB) and the 2.3 IC-LoRA family are
**separately gated** and the operator's token does not have them — verified, they 403. Not worth
a second gate request, because §5.3 is a better fixture anyway.

### 5.3 **Chosen: `joyfox/LTX-2.3-Transition-LORA`**

| | |
|---|---|
| Size | **0.390 GB** — small enough to download inside a run |
| Popularity | **19,445 downloads, 162 likes** — the most-downloaded LTX LoRA on the Hub |
| Access | **ungated** |
| Convention | `diffusion_model.` prefix (1152 tensors) — exercises `_convert_non_diffusers_ltx2_lora_to_diffusers` |
| Pairing | 576 × `lora_A.weight` + 576 × `lora_B.weight` — clean peft-style pairing |
| Metadata | **`__metadata__` entirely ABSENT** |

**Why this one is better than an official adapter, not merely acceptable:**

1. **It is what users actually upload.** The most-downloaded LTX LoRA on the Hub is a far more
   representative stage-7 subject than a first-party adapter shipped alongside the model.
2. **It exercises the dominant real-world key convention.** All three real LoRAs I measured use
   `diffusion_model.`, so the *conversion* path is the main path (PLAN §4.5).
3. **Its missing `__metadata__` exercises stage 4's warn-not-reject branch** — the branch that
   would otherwise be untested, and the branch whose absence would reject the Hub's most popular
   LTX LoRA.
4. **It directly tests risk #6** (PLAN §10): a **2.3** LoRA against the **2.5 distilled**
   transformer. The model card claims most 2.3 LoRAs run on 2.5 unchanged. R3 is where that claim
   either holds or does not, and either outcome is information we need.

**Backup fixture:** `vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa` (0.705 GB, ungated) — it
carries a full sd-scripts metadata block (`ss_ltx_version: "2.3"`,
`modelspec.architecture: "LTX2/lora"`). Better used **offline** as the metadata-present fixture
for stages 1–6; a second GPU run for it would prove nothing R3 does not.

**No synthetic adapter is needed.** Real fixtures were available and are strictly more
informative.

---

## 6. Showcase runs

**Only if quota remains after the matrix is green**, and only with the stop line respected. A
showcase run at full resolution is a *demo*, not a test — it must never displace a retest.

---

## 7. Rules of engagement

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

## 8. Run log

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
