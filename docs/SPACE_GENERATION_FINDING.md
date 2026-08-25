# The Space: a killed worker, a falsified ceiling, and a working generation

> ## ✅ RESOLVED — the Space generates. **Read §9.3 before citing why.**
>
> A real **39,146-byte** video, **content-verified**, at sha `8fb292d6`.
>
> ⚠️ **The fix is NOT attributed.** Two things changed between the failing and
> succeeding runs, not one. **"The Space now serves" is established; "the
> widening is why" is not.** §9.3.
>
> The investigation below is kept exactly as written. Its measurements stand, and
> **two of its central claims were falsified** by the run that resolved it —
> including one this build had been planning against since before any deploy.

> **Original framing, first attempt — accurate as history:**
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

> ✅ **Superseded on the retry at sha `8fb292d6`: generation now succeeds.** The
> separation above is what made that legible — startup never broke, so the
> failure was always in the second claim. §9.3.

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

⚠️ **The two `*_PRECISION` variables were inert** — not wired to anything, at
this run or ever. They are recorded here because they were genuinely set on the
Space at the time, not because they did anything. **Both have since been deleted
from the Space.** See §9.1.

> ⛔ **[On "Both have since been deleted"] — half-false when read back from the
> platform (2026-08-25).** The live Space variables show
> `LTX25_TRANSFORMER_PRECISION=bf16` still **PRESENT**; only
> `LTX25_TEXT_ENCODER_PRECISION` is gone. The deletion was asserted, not read
> back — this document's own organising rule, violated by this document. (§9.1's
> "both precision variables were **removed**" remains true: that sentence is
> about the code-side removal recorded in `PLAN.md` §2.6, which did happen.)
> The survivor is doubly wrong-shaped: inert as ever, **and** it now advertises
> bf16 on a Space whose transformer actually loads in bitsandbytes int8 — a live
> instance of exactly the advertised-lever class §9.1 warns about. Its deletion
> is being requested with the next deploy.

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

> **[On `loaded transformer as int8`]** — accurate as printed. At the time of this
> run `components_to_quantize=["transformer"]`: **one** component was quantised,
> and the single `bitsandbytes` line above corroborates it. The build changed twice
> afterwards — `02fe943` added the text encoder, `c7617b1` made the log name every
> component it quantises — so the same load today prints `quantised transformer,
> text_encoder as int8`. Expect doc and code to differ **here and only here**; the
> quote is left byte-identical because it is the record of what ran. Consequence
> for §6: **the 55.89 GiB was measured with one component in int8, not two.**

**Everything the deploy was built to do, it did** ✅ — the 37-file runtime
download at the pinned revision, the int8 transformer load, tensor packing, and
the post-pack prune. **On a Space that had evicted twice and never run.**

---

## 6. The RAM ceiling — two measured bounds of ours, one inherited number now falsified

> ### ✅ Ours: the ceiling is **above 60.0 GB**
>
> Startup reached **55.89 GiB = 60.0 GB** peak RSS **and was not killed.**
> Therefore the cgroup ceiling is measurably higher than that. **This is our own
> datum, from this platform, from this run** — the first real information anyone
> on this build has had about that number.

> ### ✅ Ours, superseding: the ceiling is **above 93.56 GB**
>
> The route-B run at sha `8fb292d6` — transformer **and** text encoder in int8 —
> reached **87.13 GiB = 93.56 GB** peak RSS and **was not killed either.**
> Startup completed, packing ran, the Space came up RUNNING. Same platform, same
> kind of datum, and it replaces the bound above as our best lower bound.
>
> The 60.0 GB figure stays exactly as recorded: it is a true measurement of a
> **different configuration** (§5), and the note at that log line already says so.

> ### ⛔ Inherited: the **~68 GB** figure is **FALSIFIED**
>
> A community cgroup observation. **We never measured it, and now we have
> measured past it.** A process on this platform survived **93.56 GB — 25.56 GB
> above it.** It is no longer "unconfirmed but consistent"; for this Space it is
> **wrong**, and every headroom figure computed against it was wrong by at least
> that margin.
>
> **What it cost while it stood:** it made the 60.0 GB run look like ~8 GB from
> the edge. The true headroom at that moment was **≥ 33.56 GB**. The whole
> cgroup-OOM hypothesis in §8 was built inside a gap that did not exist.

**The axis was right, and the inherited half is the half that broke.** Keeping
"ours" apart from "inherited" is what made this correctable in one step: nothing
we measured had to be revisited, and the falsification landed on exactly the
claims already marked ⚠️. Had the two been collapsed into a single "≈68 GB
ceiling", surviving 93.56 GB would have read as our own data contradicting
itself, rather than as a refutation of someone else's number.

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

⛔ **One correction inside that quote.** It called the ~68 GB figure *"the
binding constraint, and the first thing to measure."* **The instruction was
right and the number was wrong** — we measured it, and §6 now records it as
falsified. The quote is left byte-identical because it is the record of what was
predicted; what it got right was *what to measure*, not *what the value was*.

⏸️ **`ZEROGPU_DISK.md` §4.2 should be upgraded ⏸️ → ✅ from this run.** Not done
here — that file is gated behind the operator's docs-last order. See §11.

---

### ⛔ The same mechanism, third application — and this time it ran against us

§4.2 gave **one** mechanism: a quantising load materialises int8 tensors as
anonymous memory, losing the laziness `safetensors` mmap provides. It used that
one mechanism to settle two questions — why int8 RSS approaches the materialised
weight size, and why the post-pack prune might free nothing. **It has now
settled a third, in the direction opposite to the one we expected:**

> **Quantising the text encoder materialised what `safetensors` had been keeping
> lazy. Peak RSS went UP by 31.24 GiB (33.54 GB)** — from 55.89 GiB to
> **87.13 GiB** — in a change made to bring it *down*.

That is a stronger result than a new explanation would have been. It is not a
mechanism we lacked; it is **the mechanism we already had, predicting an outcome
we still got wrong.**

✅ **And §9 said so, in prose, two paragraphs below the table that got it wrong:**
*"Quantising more raises **transient** load-time RSS even as it lowers steady
state ... If the kill is at load rather than at fork, B makes things worse, not
better."* The transient half of that landed exactly.

---

### The instrument lesson — why the arithmetic missed what the prose had

Projected **~46.5 GB**; measured **93.56 GB**. Wrong by 47.6 GB, in the opposite
direction. The cause is visible in §9's table, inside a single column:

| row | `expected RSS` | what that number actually is |
|---|---:|---|
| now — transformer int8 only | **60.0 GB ✅ measured** | a **peak**, from `peak_rss_gib()` |
| + text encoder int8 | ~46.5 GB ⚠️ | a projected **steady state** |

**Those are different quantities, and the table subtracted one from the other.**
A weight-footprint saving — ~12 GB of bf16 text encoder — was deducted from a
measured *peak*, yielding a number that is neither one nor the other. The
transient cost was named in the prose underneath and never entered the
arithmetic.

`peak_rss_gib()` reports the maximum across the load. **That is the right number
for "will loading survive" and the wrong number for "how much room does a forked
worker have."** It answered the question it was built to answer; the question we
needed — steady state, once the load settles — was invisible to it, and still is.

⏸️ **We have no steady-state measurement, and steady state is what the RAM
hypothesis was always about.** This is §9.1.6's ladder one rung further out: the
argument reached the function, with the right arguments, of the right type — and
measured **a different quantity than the one the decision needed.**

---

## 8. Hypotheses

### 1. ⛔ WAS LEADING — host RAM: the forked worker is cgroup-OOM-killed

> ⛔ **The arithmetic this hypothesis stood on is falsified.** It needed a
> ceiling near ≈68 GB with the load sitting ~8 GB beneath it. The ceiling is
> **above 93.56 GB** (§6) and the load at that moment sat **≥ 33.56 GB** beneath
> it. **There was no narrow margin for a copy-on-write fork to exhaust.**

⚠️ **Two errors are corrected here, not one.** This paragraph previously opened
*"Steady state **60.0 GB ✅**"*. **60.0 GB was a peak, not a steady state** —
`peak_rss_gib()` reports the maximum across the load (§7). Marked ✅, a
mislabelled quantity was reading as a confirmed one, and it was the quantity the
hypothesis turned on.

**The signature still fits, and still has nothing to attach to:** instant death,
**twice**, **no traceback**, no exception, both `fail=true`. That is `SIGKILL` —
which is what a cgroup OOM does, and also what several other things do.

**Precisely what died and what did not:**

| claim | status |
|---|---|
| the ceiling is ≈68 GB and we sat ~8 GB below it | ⛔ **falsified** (§6) |
| the kill was a cgroup OOM of the forked worker | ⚠️ **severely weakened, not disproven** |
| RAM is involved at all | ⏸️ **open** |

RAM-at-fork is **not strictly excluded**: we know neither the steady state (§7)
nor what the forked worker materialises on top of it. What is gone is the
*reason to believe it* — and that reason was arithmetic, not evidence.

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

**Widen `components_to_quantize` to include the text encoder, redeploy, and read
the peak-RSS line.**

> ⛔ **An earlier version of this section said "flip
> `LTX25_TEXT_ENCODER_PRECISION` to int8". That instruction was false** — the
> variable is wired to nothing. See **§9.1**, which is the more important half
> of this section.

### ✅ The first signal costs zero GPU quota

Packing and loading happen **at module scope, before any `@spaces.GPU` call** —
and the device-api trace confirms it: `startup-report` precedes the first
`schedule` ✅. So:

- **RSS drops to ~46 GB** → retry the same 256 × 256 × 9 request, and that is a
  real test of the RAM hypothesis.
- **Startup dies instead** → revert, and **we have learned the kill is at load
  rather than at fork.** Also free, and it answers the ⏸️ in §8.

Either outcome is informative and neither spends quota to get there.

> ⛔ **Neither branch fired. The signal was free, and it was a third outcome the
> section did not enumerate:** RSS **rose** to 93.56 GB and startup **survived**
> anyway. Both branches assumed the ceiling was close; the run falsified the
> ceiling instead of choosing between them. ✅ The claim that the signal costs no
> quota held — packing and loading precede the first `@spaces.GPU` call, exactly
> as the device-api trace predicted.

### The corrected saving

⚠️ **A previously-circulated figure of 15.13 GB was wrong.** It came from
`ZEROGPU_DISK.md`'s bound **A**, which quantises the text encoder **and**
connectors. The change actually available — adding `"text_encoder"` to
`components_to_quantize` in `_quantization_config` — covers the **text encoder
alone**: 23.92 GB bf16 📄, so **≈ 12 GB saved.**

| | packed | expected RSS | headroom if ceiling ≈ 68 GB ⛔ *(falsified, §6)* |
|---|---:|---:|---:|
| **now** — transformer int8 only | 53.2 GB | **60.0 GB ✅ measured** | ~8 GB ⚠️ |
| **+ text encoder int8** | ~41.2 GB ⚠️ | ~46.5 GB ⚠️ | ~21.5 GB ⚠️ |

⚠️ The derived column scales by the **1.13 transient factor** the measured run
implies — **one data point**, so it is an extrapolation from a single
observation, not a rate.

> ⛔ **Superseded by measurement — kept because it is the reasoning the change
> was made on.** Two independent faults: the ceiling is not ≈68 GB (§6), and the
> `expected RSS` column mixes a measured **peak** with a projected **steady
> state** (§7). Measured outcome of the `+ text encoder int8` row:
> **87.13 GiB = 93.56 GB peak**, against ~46.5 GB projected — **wrong by 47.6 GB
> and in the opposite direction.** The `packed ~41.2 GB` estimate is not
> contradicted by this run; only the RSS column is.

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

---

## 9.1 ⛔ The instruction in §9 was false, and this document nearly proved the RAM hypothesis wrong

**`LTX25_TEXT_ENCODER_PRECISION` is not a lever. It is wired to nothing.**

✅ Verified by mutation rather than by reading, twice and independently:

```
grep -rn TEXT_ENCODER_PRECISION ltx25_studio/ app.py   ->  no match, any file

config, variable UNSET     -> RuntimeConfig(..., quantization='bf16')
config, variable = 'int8'  -> RuntimeConfig(..., quantization='bf16')
IDENTICAL under mutation                                  True
illegal value 'fp4-nonsense' accepted silently            True

CONTRAST — a lever that IS wired:
LTX25_QUANTIZATION='fp4-nonsense'  ->  ConfigError: not a recognised value.
```

It is not half-wired. **It is not a `RuntimeConfig` field at all**, so it is
never resolved, never validated and never logged. 📄 `PLAN.md` records why: both
precision variables were **removed** because *"an advertised lever attached to
nothing is worse than no lever at all"*. The Space Variables were leftovers from
before that removal.

**And the running Space said so, in a line already quoted in §5:**

```
dtype : bf16 (fixed; no precision switch)
```

**Nobody read it.** The evidence was in the log we were all working from.

### What it would have cost

Flipping the variable and restarting would have produced **no change in peak
RSS** — because nothing would have changed. **We would have read an unmoved
number as evidence against the RAM hypothesis** (§8.1), abandoned the leading
explanation, and gone looking elsewhere.

> **A wrong conclusion from a real experiment, indistinguishable from a
> legitimate negative result.**

That is the same failure shape this project has now hit four times, and it is
the one `997b080`'s commit message describes for the transformer config: *if the
config never reaches `from_pretrained`, the Space deploys bf16 while the variable
says int8 … and the conclusion drawn is "int8 does not fit", from a run in which
int8 was never applied.*

**This document was written to warn about that class, and contained a live
instance of it.** Writing the warning does not confer immunity to it.

### How it happened — the sequence, which is the useful part

1. The **orchestrator** set the Space Variable believing it was a live lever. In
   his words: *"it is not wired, so it did nothing except trigger a pointless
   restart … A false instruction sitting in the only record of this failure is
   the worst possible place for one, and it is my instruction that is false."*
2. The **lead's dispatch repeated it** without checking whether it was wired.
3. The **implementer wrote it into this file** — the only record of the failure
   — likewise unchecked.
4. The **operator imposed a wiring check** before any reading from a restart
   would be trusted.
5. The **mutation test settled it** before a restart could be misinterpreted.

**Three people passed the same unverified claim along, and the check that caught
it cost about thirty seconds.** The variable had been sitting in the Space doing
nothing, and the startup log had been announcing that fact the whole time.

📄 `PLAN.md` §9.1.6 lists ladder rung 5 as *"hand-built, one call site at a
time"*. **This is rung 5 applied to an environment variable rather than a call
site — and it caught a live one.** The ladder is not only about function
arguments; anything that looks like an input has a rung 5.

---

### The routes not recommended

- ⚠️ **Route A — duration 60 → 180.** No free signal, and it targets the branch
  the evidence already argues against: both workers died **under** the existing
  reservation.
- ✅ **Route C — instrument the worker. NOW THE LEADING ROUTE**, no longer the
  fallback. B's free signal has been spent: it returned a real measurement, that
  measurement **falsified the ceiling** rather than confirming the margin, and
  the kill is unexplained again. Route C is the only remaining path that returns
  **evidence** instead of another inference — and every inference we have made
  about this kill has now been wrong at least once.

---

## 9.2 What route B bought, what it cost, and the risk it introduced

**Bought:** ✅ the ceiling, as a real number of ours — *above 93.56 GB* — which
falsified an inherited figure that had been load-bearing in planning since before
any deploy. ✅ Confirmation that the widening reaches the checkpoint: **two**
`bitsandbytes` module lines where the measured run had one —

```
The module 'Gemma4UnifiedForConditionalGeneration' has been loaded in `bitsandbytes` 8bit
The module 'LTX2VideoTransformer3DModel'          has been loaded in `bitsandbytes` 8bit
```

**Cost:** ⚠️ the attributability §9 warned about is spent — int8 was run as a fix
rather than as a same-seed pair against a signed-off bf16 baseline, and it cannot
be recovered without re-running both halves.

### ⚠️ The new risk, stated plainly

**The load now peaks 33.54 GB higher than the configuration that was working,
and we know the ceiling only as "above 93.56 GB" — not by how much.**

We may be loading far closer to the real limit than before, and we are paying
that for a **steady-state saving we have never measured** (§7). The motivating
hypothesis for the change — a tight RAM margin at fork — is itself the thing the
change disproved. That is an uncomfortable position: the justification did not
survive the experiment it justified.

> ⛔ **Operator's standing decision: if the retry fails, `02fe943` is reverted.**
> Keeping a change whose motivating hypothesis is dead, at **31.24 GiB** of
> additional peak RSS, for an unmeasured benefit, is not defensible.

### ✅ Retry outcome — it succeeded

**It did not fail, so `02fe943` STAYS.** The standing revert decision above is
discharged, not overridden. Detail and the attribution limit: §9.3.

---

## 9.3 ✅ The Space generated — and what that does **not** establish

```
PREDICT_WALL_CLOCK_SECONDS 21.44
RESULT: ["output.mp4", null, "✅ Text to video complete. seed 1234 · run directory outputs/run-fc6d3a762eaa"]
```

### The artifact is real — structure **and** content

Structure alone was not accepted, because **H9 taught this build that a green
stream table and a silent file are indistinguishable.** So the video got the
analogue of the RMS check that caught H9:

```
39,146 bytes · h264 · 256x256 · 24 fps · 9 frames · duration 0.375 s
verify_streams text_to_video -> MATCHES, exit 0   (video: True, audio: False -- correct for this mode)

global pixel std        72.657      (~0 would be a flat/blank frame)
mean |frame-to-frame|   18.894      per-pair 57.17 .. 8.35, none near zero
distinct frames         9 of 9
dynamic range           0.0 .. 255.0 on every frame
```

✅ **Varied, moving, all frames distinct.** Not a blank file that passes a
schema.

### The device-API contrast — the cleanest evidence in this file

```
FAILED run:  schedule -> allow tok=2bca839a pid=262 -> release fail=true  404
             schedule -> allow tok=1c3ef0c9 pid=286 -> release fail=true  404   (same taskId, spaces retried)
SUCCESS run: schedule -> allow tok=34ddf8b8 pid=258 -> release fail=FALSE 200   (one worker, no retry)
```

**`fail=true` twice against `fail=false` once**, from the platform's own
accounting rather than from ours.

---

### ⛔ The caveat, stated as prominently as the success: **the fix is not attributed**

**The failing and succeeding runs differ by TWO things:**

1. `02fe943` — the text encoder moved to int8.
2. **A full container rebuild** — new image, new download, plausibly a different
   worker node.

**This is not the clean A/B we had for the RSS number, and the difference is
worth being exact about.** For RSS, an AST diff of the shipped package showed the
quantised set was the *sole* runtime variable between the two measurements — so
the 31.24 GiB rise is attributable. Here a rebuild sits in the middle, and a
rebuild changes things we do not observe.

> **n=1 success against n=1 failure, with two variables between them, supports
> "the Space now serves" and does not support "the widening is why."**

**And the reason to resist the causal reading is stronger than the usual one:**
the hypothesis that motivated `02fe943` — cgroup OOM against a ~68 GB ceiling —
is **dead**, falsified by our own 87.13 GiB survival (§6). So we changed
something for a reason that turned out to be wrong, and the thing then worked.
**Those are two facts, not one causal chain.** ⏸️ Readings still open:

- RAM at fork was real, with different arithmetic than ours, and the lower steady
  state genuinely helped.
- The first failure was a platform-side transient; **the rebuild alone fixed it.**
- Something else about the new container.

✅ **Route C — instrument the worker — remains the leading path.** This success
does not retire it; it makes it *cheaper to justify*, because there is now a
**working baseline to instrument against** instead of a dead Space.

### Costs

```
failed run    100.35 s wall-clock, TWO worker attempts
success run    21.44 s wall-clock, ONE worker attempt
session total 121.79 s wall-clock; at xlarge 2x that is <= ~243.58 GPU-seconds
budget        2400 GPU-seconds/day (PRO) -> ~10.2% consumed
```

⏸️ Exact billed seconds remain unmeasurable — no readable ZeroGPU quota endpoint
(404 on all three probed). The figure above is a **bound**, not a measurement.

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

> ✅ **And now the other half.** On the successful run the app returned a path
> **and the file behind it was real** — 39,146 bytes, 9 distinct frames, pixel
> std 72.657 (§9.3). The invariant has now been exercised in both directions:
> **no file claimed on failure, and a genuine file behind the claim on success.**
> A build that only ever failed could not have shown the second half.

---

## 11. Follow-ups this run created

| | item | status |
|---|---|---|
| 1 | **Our error text tells users to read a traceback that does not exist** (§3). In a kill, there is nothing to read, and that is the most informative fact — the copy should say so instead. | ⏸️ open |
| 2 | **`ZEROGPU_DISK.md` §4.2**: upgrade the RSS prediction and the prune branch ⏸️ → ✅ from this run. **→ Applied in the 2026-08-25 docs pass** — §4.2 now carries the dated upgrade annotation. | ✅ resolved (2026-08-25) |
| 3 | **`OPERATOR_HANDOFF.md` §3 and §5 are stale in *both* directions** — §3 says the Space *"has never run"*, §5 says a deploy is *"not currently possible"*. **Startup demonstrably works and generation demonstrably does not**, and the document currently cannot express that distinction. **→ Doubly stale itself by 2026-08-25: generation now demonstrably works (item 10 — 11 of 11), and the handoff was reworked to "one mode, once", which the mode matrix then superseded in turn. The handoff now carries dated annotations for both; this row is kept as the record of the gap as it stood after the first run.** | ✅ resolved (2026-08-25) |
| 4 | ⏸️ **No readable ZeroGPU quota endpoint** — all three candidate URLs 404, so billed seconds cannot be measured, only bounded. Worst case ~200 GPU-s of 2400/day for this run (`xlarge` bills at 2×; packed 51.1 GB is over the 42.00 GiB threshold 📄). | ⏸️ open |
| 5 | **Bus defect**: `lead-gate.mjs:1709` computes `isLead = (session_id === SOMA_LEAD_SESSION_UUID)`. A `/clear` rotates the session id while the env var stays baked in from launch, so every soma bus tool is denied and the lead cannot file a report. **Side effect: it also disarms the lead-only `git push` / `git merge` guard.** | ⏸️ open, infrastructure |
| 6 | ⏸️ **No steady-state RSS measurement exists.** `peak_rss_gib()` reports the load peak; the RAM question is about what remains resident when the load settles. Every RSS number in this document is a peak. **This is the measurement that would have prevented the wrong prediction in §7** and it is still not taken. | ⏸️ open |
| 7 | ⛔ **The ~68 GB ceiling was inherited, never measured, and is now falsified** (§6). Audit any other planning number with the same provenance — carried in from outside, never verified here, and load-bearing. It survived this long because it was *consistent* with our data, which is not the same as supported by it. | ⏸️ open |
| 8 | ⚠️ **`02fe943` STAYS** — the retry succeeded, so the revert condition never triggered (§9.2). But it stands on a hypothesis this run killed and costs 31.24 GiB of peak RSS for a steady-state saving still unmeasured. **Kept because it did not fail, not because it was shown to help.** | ✅ resolved, ⚠️ unjustified |
| 9 | ⛔ **The fix is unattributed** — a container rebuild changed alongside `02fe943`, so n=1 vs n=1 with two variables (§9.3). Distinguishing them needs either a revert-and-retry on the same image, or route C. **Until then no document should say the widening fixed generation.** | ⏸️ open |
| 10 | ✅ **SUPERSEDED — all 11 modes have now run and generate**, at the same 256x256x9. The mel front-end and the 16 kHz/48 kHz asymmetry have both been exercised on this platform and both passed. The matrix found 9 of 11 and one real defect — both video-INPUT modes, from a filepath `str` reaching `LTX2VideoCondition(frames=…)` — fixed at `d2eb6f2` and ✅ **re-verified on the Space at sha `287ba9c`**. Full evidence: **`docs/SPACE_MODE_MATRIX.md`**. ⏸️ Still one configuration only: 256x256x9, stage 2 off, no LoRA. | ✅ resolved |

---

## 12. Repo state at the time of writing

- **`origin/soma/ltx25-studio` is at `1802bfd`** — the six earlier commits were
  pushed by the orchestrator; the findings commits after it are unpushed. Pushing
  is the orchestrator's — this team does not push by design. Verified with
  `git rev-list --count origin/soma/ltx25-studio..HEAD`, which is the only
  authority on this; an earlier count taken from *"what I authored this round"*
  said four and silently omitted `51e1985`, this document.
- PR **#1** is OPEN and draft:
  <https://github.com/techfreakworm/ltx2.5-studio/pull/1>
- Offline gate at that tip: **2053 passed, 13 skipped, 0 failed.**

> **Superseded as repo state (annotated 2026-08-25).** `origin/soma/ltx25-studio`
> advanced to `17b589c` — everything through the 11-of-11 findings is pushed
> (`git rev-list --count` = 0, verified in the 2026-08-25 audit). Commits since
> are local and queued for the next push: the fork-boundary pair `6bb211a`
> (SPACE_MODE_MATRIX §4), five hardening fixes through `f0daa75` (LoRA
> lifecycle, duration estimator, upload bound, desync guard `59cfa6e`, mmap
> startup report), and this docs pass. PR #1 remains OPEN and draft. The lines
> above stand as the record of the state when this file was written.
