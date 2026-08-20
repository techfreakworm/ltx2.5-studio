# The Space starts and cannot generate

> **First generation ever attempted on the live private Space. Startup succeeded;
> the generation failed.** The forked ZeroGPU worker was **killed twice, with no
> Python traceback**, and `spaces` returned `'GPU task aborted'`.
>
> This file is the only place these measurements exist.

**Every load-bearing number is marked:** ✅ *we measured it* · 📄 *read off a
pinned source or the platform* · ⚠️ *inferred* · ⏸️ *not established*.

> ### ⚠️ The rule this document is organised around
>
> **What we measured and what we inherited are kept apart.** The RAM ceiling has
> two numbers attached to it and only one is ours. Blurring them is how a
> borrowed figure becomes a fact, and this build has spent a day undoing exactly
> that.

---

## 1. What happened

| | |
|---|---|
| Space | `techfreakworm/LTX2.5-Studio`, **private**, stage RUNNING 📄 |
| Space SHA | `a1791c9e856f2b96ab10d89a05a84a0899c8dca5` 📄 |
| `spaces` | 0.51.1 📄 |
| Hardware id | `zero-a10g` 📄 — **a legacy id string, not an actual A10G** |
| Account | PRO ✅ (`whoami-v2` → `isPro: true`), 2400 GPU-s/day |
| Client | `gradio_client` 2.6.0 → `/on_generate` |

**Startup: healthy.** **Generation: failed.** Those are two different claims and
the rest of this file exists to keep them apart.

---

## 2. The request

Chosen to be the cheapest thing that exercises the path — 256 × 256 × 9 frames,
no conditioning, no stage 2.

```
mode_id           = text_to_video
prompt            = 'a red balloon rising over a calm lake at dawn'
negative_prompt   = ''
image/video/audio = None
conditions        = (empty)
width, height     = 256, 256
num_frames        = 9
auto_duration     = False        min_seconds = 2.0   max_seconds = 5.0
enable_stage_2    = False
seed              = 1234
strength          = 0.8          noise_scale = 0.5
```

✅ Pre-validated offline **before spending any quota**:
`validate_dimensions(256, 256) -> (256, 256)`, `validate_num_frames(9) -> 9`.

Space variables at the time of the run 📄:

```
LTX25_TRANSFORMER_PRECISION  = bf16      LTX25_ZEROGPU_SIZE          = auto
LTX25_TEXT_ENCODER_PRECISION = bf16      LTX25_GPU_DURATION_SECONDS  = 60
LTX25_QUANTIZATION           = int8      LTX25_OUTPUT_RETENTION_GB   = 20
HF_HUB_DISABLE_XET           = 1
```

---

## 3. The failure

✅ Returned value:

```
[null, null, "### ❌ Generation failed — unexpected `Error`

> 'GPU task aborted'

This error is not one of ours, which usually means it came from the pipeline
or the platform. The full traceback is in the Space logs."]
```

`'GPU task aborted'` is raised by the **`spaces` runtime**, not by our code 📄.

✅ **No artifact was produced** — the download directory was empty afterwards.
✅ `CONNECT_SECONDS 3.34` · `PREDICT_WALL_CLOCK_SECONDS 100.35`

### ⛔ There is no traceback. That is the finding, not a gap in the report.

**There is no Python traceback anywhere in the Space run log, and no OOM or
SIGKILL text either** ✅.

> **The forked GPU worker was *killed*. It did not raise.**

An exception unwinds and leaves a stack. A `SIGKILL` leaves nothing. **The
absence is evidence**, and it is the single detail that most constrains the
hypotheses in §8 — an application-level fault of any kind would have left a
trace, and none did.

### ⚠️ And our error text points the user at something that does not exist

The message says *"The full traceback is in the Space logs."* **There isn't
one.** Our own copy sends whoever reads it to look for a thing that is not
there, in the one situation where the absence is the most important fact
available.

That is a defect in the error text, found by accident, and it is listed in §11.

---

## 4. The device-API trace — the container's entire history

✅ **This run is the only GPU request this container has ever served.** Seven
lines, in full:

```
startup-report                                   200
schedule  taskId=140074752841728                 200
allow     tok=2bca839a  pid=262                  200
release   tok=2bca839a  fail=true                404
schedule  taskId=140074752841728                 200   <-- SAME taskId
allow     tok=1c3ef0c9  pid=286                  200
release   tok=1c3ef0c9  fail=true                404
```

**Identical `taskId`, two different allow-tokens, two different worker pids.**
So: **one logical task, two forked workers, both dead, one retry by `spaces`.**
Both released with `fail=true` — the platform booked them as failures.

The retry is the platform's, not ours. **It died the same way twice**, which
rules out a transient.

---

## 5. Startup — healthy, and genuinely new evidence

```
fetched 37/37 files
LTX-2.5 Studio starting
  config          : zerogpu_size=auto duration=60s expected_weights=72.19GB outputs<=20GB
  dtype           : bf16 (fixed; no precision switch)
  weights cached  : 37/37 files
  disk            : 82.2 GB of 150 GB cap [OK] = snapshot 72.2 + packed 0.0 + image ~10
  zerogpu size    : not yet known (reported after packing)
  model revision  : 871165de037793df7d75c23a02dd7f45fd364c97
loaded transformer as int8 | peak RSS 0.84 -> 55.89 GiB
The module 'LTX2VideoTransformer3DModel' has been loaded in `bitsandbytes` 8bit
spaces.GPU kwargs: ['duration', 'size']
Cleaned 65.25GB of tensor files from ~/.cache/huggingface/hub/models--*/blobs/* after packing
POST http://device-api.zero/startup-report "HTTP/1.1 200 OK"
* Running on local URL: http://0.0.0.0:7860, with SSR
```

**Everything the deploy was built to do, it did** ✅ — the 37-file runtime
download at the pinned revision, the int8 transformer load, tensor packing, and
the post-pack prune. **On a Space that had evicted twice and never run.**

---

## 6. The RAM ceiling — one measured bound of ours, one inherited number

> ### ✅ Ours: the ceiling is **above 60.0 GB**
>
> Startup reached **55.89 GiB = 60.0 GB** peak RSS **and was not killed.**
> Therefore the cgroup ceiling is measurably higher than that. **This is our own
> datum, from this platform, from this run** — the first real information anyone
> on this build has had about that number.

> ### ⚠️ Inherited: the **~68 GB** figure
>
> A community cgroup observation. **We have never measured it.** It is
> *consistent* with our lower bound and it is not confirmed by it. Every
> headroom figure computed against ~68 GB inherits that status, and they are all
> marked ⚠️ below for that reason.

**These are different kinds of claim.** One is a bound we established; the other
is a value someone else reported. The distance between them — `>60.0` versus
`≈68` — is exactly the room the hypotheses in §8 live in, so collapsing them
would erase the uncertainty that matters.

---

## 7. `ZEROGPU_DISK.md` §4.2 predicted this, and it lands

Written **before** the deploy, when both statements were ⏸️:

> *"int8's RSS should be expected to approach the **materialised** weight size —
> order 38 GB for A, 53 GB for B … **Not 20 GiB.** That turns the ~68 GB cgroup
> figure from a number someone mentioned into **the binding constraint, and the
> first thing to measure.**"*

| prediction | measured | |
|---|---|---|
| RSS ≈ 53 GB on route B, **not** ~20 GiB | **60.0 GB** ✅ | lands, with ~1.13× for the transient bf16 source held during conversion |
| the prune **may free nothing** | **fired: 65.25 GB freed** ✅ | the favourable branch, which §4.2 declined to assume |
| RAM, not disk, is the binding constraint | ⚠️ leading hypothesis, §8 | the disk line read 82.2 / 150 GB **[OK]** and the workload still died |

**Both ⏸️ items are now ✅.** The 20.05 GiB bf16 figure was correctly rejected as
a floor for a quantising load — the mechanism given at the time (safetensors
mmaps; quantisation materialises anonymous memory) is the same mechanism that
governs the prune, and both halves behaved as that predicted.

⏸️ **`ZEROGPU_DISK.md` §4.2 should be upgraded ⏸️ → ✅ from this run.** Not done
here — that file is gated behind the operator's docs-last order. See §11.

---

## 8. Hypotheses

### 1. LEADING — host RAM: the forked worker is cgroup-OOM-killed

Steady state **60.0 GB ✅**, against a ceiling **> 60.0 GB ✅ (ours)** and
**≈ 68 GB ⚠️ (inherited)**. A fork that touches shared pages copy-on-write adds
to that, and there is little room to add into.

**The signature fits:** instant death, **twice**, **no traceback**, no
exception, both `fail=true`. That is `SIGKILL`, and `SIGKILL` is what a cgroup
OOM does.

**This now rests on two of our own measurements plus one inherited number**,
rather than on the inherited number alone.

### 2. WEAKER — the 60-second duration reservation

Both attempts died at **~50 s** ✅ (100.35 s wall-clock across two) — **under**
the reservation. **There was no timeout to hit.**

Not excluded: a slow fork-and-restore could be culled near the boundary. But it
is a poor fit for a death that leaves no trace, and it does not explain why the
retry failed identically.

### ⏸️ Not established

Whether the kill happens **at fork** or **at load inside the worker**. That
distinction decides whether §9's recommendation helps or hurts, and nothing in
this run separates them.

---

## 9. Recommendation — route B, and its first signal is free

**Flip `LTX25_TEXT_ENCODER_PRECISION` to int8, restart, and read the peak-RSS
line.**

### ✅ The first signal costs zero GPU quota

Packing and loading happen **at module scope, before any `@spaces.GPU` call** —
and the device-api trace confirms it: `startup-report` precedes the first
`schedule` ✅. So:

- **RSS drops to ~46 GB** → retry the same 256 × 256 × 9 request, and that is a
  real test of the RAM hypothesis.
- **Startup dies instead** → revert, and **we have learned the kill is at load
  rather than at fork.** Also free, and it answers the ⏸️ in §8.

Either outcome is informative and neither spends quota to get there.

### The corrected saving

⚠️ **A previously-circulated figure of 15.13 GB was wrong.** It came from
`ZEROGPU_DISK.md`'s bound **A**, which quantises the text encoder **and**
connectors. The lever actually available — `LTX25_TEXT_ENCODER_PRECISION` —
covers the **text encoder alone**: 23.92 GB bf16 📄, so **≈ 12 GB saved.**

| | packed | expected RSS | headroom if ceiling ≈ 68 GB ⚠️ *(inherited, §6)* |
|---|---:|---:|---:|
| **now** — transformer int8 only | 53.2 GB | **60.0 GB ✅ measured** | ~8 GB ⚠️ |
| **+ text encoder int8** | ~41.2 GB ⚠️ | ~46.5 GB ⚠️ | ~21.5 GB ⚠️ |

⚠️ The derived column scales by the **1.13 transient factor** the measured run
implies — **one data point**, so it is an extrapolation from a single
observation, not a rate.

### ⚠️ Two costs, and the first is serious

**Route B puts at risk the one thing on this Space that demonstrably works.**

**Quantising more raises *transient* load-time RSS even as it lowers steady
state**, because each module briefly holds both its bf16 source and its int8
output. **If the kill is at load rather than at fork, B makes things worse, not
better.** The peak-RSS line in `_base_pipeline` is what tells those apart on the
next run — which is why the recommendation is *"flip it and read the line"*
rather than *"flip it and retry the generation"*.

**Attributability.** §4.2 wanted int8 sequenced as a same-seed pair against a
signed-off bf16 baseline, so *"what does int8 cost in quality?"* could be
answered in one look. **Running it as a fix spends that**, and it cannot be
recovered later without re-running both halves.

### The routes not recommended

- ⚠️ **Route A — duration 60 → 180.** No free signal, and it targets the branch
  the evidence already argues against: both workers died **under** the existing
  reservation.
- ⏸️ **Route C — instrument the worker.** The only route that returns
  **evidence** rather than a better inference. More expensive; worth it if B's
  free signal is ambiguous.

---

## 10. ✅ One thing worked, and it was the acceptance criterion

On failure the app returned **`[null, null, error]`** and **did not invent a
file path** ✅.

**The no-silent-empty-output invariant held under a real platform failure.**
That is the 2.3 outage class — a Space reporting success while producing
nothing — and it was the acceptance criterion in the original brief. It was
tested by accident, on the first real failure, and it passed.

The error was surfaced as an error, named as **not ours**, and no artifact was
fabricated to accompany it.

---

## 11. Follow-ups this run created

| | item | status |
|---|---|---|
| 1 | **Our error text tells users to read a traceback that does not exist** (§3). In a kill, there is nothing to read, and that is the most informative fact — the copy should say so instead. | ⏸️ open |
| 2 | **`ZEROGPU_DISK.md` §4.2**: upgrade the RSS prediction and the prune branch ⏸️ → ✅ from this run. | ⏸️ gated, docs-last |
| 3 | **`OPERATOR_HANDOFF.md` §3 and §5 are stale in *both* directions** — §3 says the Space *"has never run"*, §5 says a deploy is *"not currently possible"*. **Startup demonstrably works and generation demonstrably does not**, and the document currently cannot express that distinction. | ⏸️ gated, docs-last |
| 4 | ⏸️ **No readable ZeroGPU quota endpoint** — all three candidate URLs 404, so billed seconds cannot be measured, only bounded. Worst case ~200 GPU-s of 2400/day for this run (`xlarge` bills at 2×; packed 51.1 GB is over the 42.00 GiB threshold 📄). | ⏸️ open |
| 5 | **Bus defect**: `lead-gate.mjs:1709` computes `isLead = (session_id === SOMA_LEAD_SESSION_UUID)`. A `/clear` rotates the session id while the env var stays baked in from launch, so every soma bus tool is denied and the lead cannot file a report. **Side effect: it also disarms the lead-only `git push` / `git merge` guard.** | ⏸️ open, infrastructure |

---

## 12. Repo state at the time of writing

- `b26c2d0` is **unpushed**; `origin/soma/ltx25-studio` is at `cbaa3ad`. Pushing
  is the orchestrator's — this team does not push by design.
- PR **#1** is OPEN and draft:
  <https://github.com/techfreakworm/ltx2.5-studio/pull/1>
- Offline gate at that tip: **2051 passed, 13 skipped, 0 failed.**
