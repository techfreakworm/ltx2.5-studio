# The NVML allocator asserts: one root cause, and the scoreboard that is honest about how much of it is proved

> ## Status on 2026-11-16
>
> This is the August 2026 record; its body is kept as written. Superseded since:
>
> - **§1 "`auto` is the default (`config.DEFAULT_ZEROGPU_SIZE`)"**: the default is now `xlarge`
>   in code, so the size is passed explicitly even where the Space variable is missing. The Space
>   itself still runs the 2026-08-26 build with `LTX25_ZEROGPU_SIZE=xlarge` set as a variable.
> - **§1 "`spaces` 0.51.1"**: the Space reports `spaces` 0.51.3. Its source, read on 2026-11-16,
>   takes the same four hops (`client.py:110` and `:112-113`, one line lower than in 0.51.1;
>   `api.py:112-113`), so `auto` still sends no size there.
> - **§2/§3 "sighting 3 … pending the residency change in this commit"**: the residency change is
>   `52a0f56`, the commit before this file's last edit, and it was deployed on 2026-08-26 (Space
>   `2495a6b`). No enhanced run has been recorded since, so sighting 3 is still open.
>
> Still standing: the R4b retirement (704×480×49 passes on `xlarge`, 55.906 GiB reserved) and
> the R3 verdict. Where the current truth is: [`SPACE_MODE_MATRIX.md`](SPACE_MODE_MATRIX.md) §10
> and [`../README.md`](../README.md). The Space has not been redeployed since 2026-08-26, so
> nothing about the Space after the 2026-11-16 fixes is known yet.

> ## ✅ THE ROOT CAUSE IS FOUND. THE ACCOUNT IS **SUPPORTED, NOT PROVEN.**
>
> This file began as a register of three `RuntimeError`s that looked alike and
> had no explanation. It now has one: **the app spent its entire life on a GPU
> partition half the size it was entitled to.** `LTX25_ZEROGPU_SIZE=auto` cannot
> promote a Space to `xlarge`, so every run this build has ever made — every
> allocator assert, every headroom decision, every "the enhancer must be freed
> for VRAM safety" — happened inside **47.375 GiB** instead of **94.971 GiB**.
>
> **The single-cause account is SUPPORTED, NOT PROVEN.** One measured
> retirement, one consistent-but-untested, one pending. §2 is the scoreboard and
> it is the part of this file to read if you read nothing else.
>
> ⚠️ **Do not read this file as "all three are resolved."** It is one ✅, one ⏸️
> and one ⏸️. The ✅ marks elsewhere in this document mean *"we measured this
> number"*, per the legend below — they are not retirements.

**Every load-bearing number is marked:** ✅ *we measured it* · 📄 *read off a
pinned source or the platform* · ⚠️ *inferred* · ⏸️ *not established*.

---

## 1. The root cause: `auto` is an inert lever, in four hops

📄 Traced through the platform's own `spaces` package (0.51.1 on this machine;
the same code in every copy checked). Our side sets `size=None` whenever
`LTX25_ZEROGPU_SIZE` is `auto` — `app.py`, and `auto` is the default
(`config.DEFAULT_ZEROGPU_SIZE`). That `None` then travels:

| hop | where | what happens |
|---|---|---|
| **1** | `spaces/zero/__init__.py:29-35` | `startup()` packs the weights and, if the total exceeds `xlarge_threshold`, sets the **module global** `_default_gpu_size = 'xlarge'`. This is the auto-promotion the whole build believed in. Ours packs well past the threshold, so it **does** fire. |
| **2** | `spaces/zero/client.py:109` | `get_duration_seconds()` does `gpu_size = gpu_size if gpu_size is not None else default_gpu_size()`. ⛔ **This is the only consumer of `default_gpu_size()` anywhere in the package, and it rebinds a LOCAL.** |
| **3** | `spaces/zero/client.py:111-112` | That local feeds exactly one branch: `if gpu_size != 'xlarge': duration_seconds *= duration_factor`. So the promotion's entire observable effect is **skipping a 1.5× multiplier on the requested duration**. |
| **4** | `spaces/zero/client.py:144` → `spaces/zero/api.py:112-113` | `schedule()` forwards its **own** `gpu_size` — still `None`, never the promoted value — to `api_client().schedule(gpu_size=gpu_size)`, where `if gpu_size is not None: params['gpuSize'] = gpu_size`. With `None`, **`gpuSize` is never sent to the scheduler at all**, and the platform allocates its default: `large`. |

⛔ **So the promotion was real, and it reached the billing meter instead of the
GPU.** It changed how long we were allowed to ask for. It never changed what we
were given. A lever that moves a number in the invoice and nothing in the
hardware is the inert-lever pathology in its purest form, and this one was
*inside the vendor's package*, which is why four separate documents in this repo
assert the opposite in good faith.

### 📄 The arithmetic, from the platform's own config file

`spaces/zero/configs.json`, the entry for `NVIDIA RTX PRO 6000 Blackwell Server
Edition`:

| field | bytes | GiB |
|---|---|---|
| `memory` | `101973950464` | **94.971** |
| `xlarge_threshold` | `45097156608` | **42.00** |
| `duration_factor` | `1.5` | — |

✅ Measured from inside the worker, before and after setting
`LTX25_ZEROGPU_SIZE=xlarge`:

```
before   47.375 GiB total   headroom 1.19 GiB   -> residency impossible
after    94.971 GiB total   headroom ~53.7 GiB  -> residency trivially affordable
```

The "after" figure is `memory` to three decimals. ⚠️ The "before" figure is half
the card (47.485 GiB) less ~0.11 GiB of partition overhead — i.e. **`large` is
half of `xlarge`**, which is what `HINODE_HOST_SETUP.md`'s "large=48 GB,
xlarge=96 GB" was already saying in round numbers.

### ⛔ Inert-lever instance #9, and the most expensive one this build has found

The previous eight cost a wrong log line, a guard that checked nothing, a
comparison that was never a comparison. This one cost **the hardware**. Charged
against it:

* ✅ **R4b's stage-2 failure**, which was never a size ceiling to be found by
  bisection — it needed 55.906 GiB on a 47.375 GiB partition and always would
  have (§2).
* ✅ **+30.78 s on every enhanced generation**, because residency was refused on
  headroom grounds that were an artefact of the wrong partition. Fixed in the
  commit before this one.
* ⏸️ **A day of allocator-pathology hypothesis** in this file's own §3, written
  because the boring explanation seemed not to fit.

The class register is `SPACE_MODE_MATRIX.md` §8 — *"an advertised guard wired to
nothing… strictly worse than its absence, because an inert lever merely does
nothing while an inert report actively tells you that you are safe."* This
instance is both: `auto` did nothing, and every doc that reasoned from it
reported a 94 GiB partition we did not have.

---

## 2. ⛔ THE SCOREBOARD

| instance | status |
|---|---|
| R4b, stage 2 704x480x49 | ✅ **RETIRED — measured.** Re-ran on xlarge: PASS, 53.08 s, correct 1408x960 geometry, peak **55.906 GiB reserved**. That is **8.5 GiB more than the entire old 47.375 GiB partition** — not "nearly full", arithmetically impossible, and always would have been. |
| R3, untiled VAE decode 1216x704x121 | ⏸️ **CONSISTENT, UNTESTED — permanently.** Tiling has been unconditional since `cdb3211` with no toggle, so a re-run exercises the tiled path and cannot test the untiled failure. A tiling-off switch was considered and REFUSED by the operator: it would exist only to reproduce a failure already fixed by other means, and it would put a live lever in shipping code that lets a user OOM themselves — the inert-lever pathology inverted. |
| sighting 3, enhancer allocation | ⏸️ pending the residency change in this commit |

**The single-cause account is SUPPORTED, NOT PROVEN.** One measured retirement,
one consistent-but-untested, one pending. Read the column, not the story: a file
that had three unexplained failures and now has one measured explanation has
*one* measured explanation.

### ⏸️ The optional derivation was NOT done, and here is why

R3's allocation size is still the ⚠️ **~26 GB estimate** it always was. Upgrading
it to a derivation off `AutoencoderKLLTX2`'s own shapes and compression ratios
would have needed two things this machine does not have: the **pinned diffusers
revision** `ac56fa2f` (local copies are 0.37/0.39 — different enough that even
the class name differs, `AutoencoderKLLTX2Video`), and the checkpoint's real
`vae/config.json`, which lives with 72 GB of weights that are not here.

Deriving it from a neighbouring revision would replace an estimate *labelled* an
estimate with a derivation whose provenance is wrong, which is worse than
leaving it alone. ⏸️ It stays an estimate, and it remains **not a test** either
way — the same label the row above carries.

---

## 3. The sightings, side by side

| | 1 — VAE decode | 2 — diffusion decoder | 3 — prompt enhancer |
|---|---|---|---|
| **date** | 2026-08-25 ✅ | 2026-08-25 ✅ | 2026-08-27 ✅ |
| **run** | `text_to_video` 1216×704×121, stage 1 (R3, deploy 1) | stage 2, 704×480×49 input (R4b, deploy 2) | the **first** enhanced generation in a fresh worker |
| **what had already succeeded** | denoise, ~52 s ✅ | the latent upsample and the extra denoise steps ✅ | the worker was up with the transformer VRAM-resident ✅ |
| **where it raised** | `conv3d` inside `autoencoder_kl_ltx2.py:108` 📄 | `LTX2VideoDiffusionDecoderModel` 📄 | loading / running the 2.49 GB 📄 enhancer onto CUDA |
| **the error** | `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":1165` 📄 | recorded as *"same NVML allocator assert"* ✅ — ⏸️ **its traceback was not quoted at the time** | the same assert ✅, wrapped as `PromptEnhancementError` |
| **what the user saw** | our `DecodeError`, wrapped reason intact, legible client-side ✅ | ⛔ FAIL, no artifact ✅ | `PromptEnhancementError` naming the repo and the remedy 📄 |
| **allocation size** | ⚠️ ~26 GB of full-resolution decode intermediates — still an estimate, see §2 | ✅ **55.906 GiB reserved**, measured on the corrected partition | 2.49 GB of weights 📄, plus whatever the 600-token decode holds |
| **partition it ran on** | 47.375 GiB ✅ | 47.375 GiB ✅ | 47.375 GiB ✅ |
| **the FAILURE fixed?** | ✅ yes — tiled + framewise decode at `cdb3211`; the retry passed at 66.47 s / 6,258,833 B ✅ | ✅ yes — passes on `xlarge`, 53.08 s | ⏸️ pending the residency change |
| **the CAUSE established?** | ⏸️ **no** — consistent with the partition, permanently untestable (§2) | ✅ **RETIRED — measured** (§2) | ⏸️ **no** — pending |

⛔ **The last two rows are split on purpose, because collapsing them is how this
file gets misread.** "The failure is fixed" and "we know why it happened" are
different claims, and sighting 1 has the first without the second — `cdb3211`
made the allocation smaller, which works whether the cause was the partition or
something else. Only the middle column has both, and only the middle column is
retired. Counting ✅ down the *fix* row gives two; counting down the row that
matters gives **one**.

Sources, unchanged and not rewritten:
[`SPACE_SESSION_2026-08-25.md`](SPACE_SESSION_2026-08-25.md) §4 (sighting 1) and
§5 (sighting 2).

> ### ⚠️ Sighting 2 was already in the record when this file said "two"
>
> The brief that created this file named two sightings. The 2026-08-25 session
> doc carried a third in §5 — R4b, *"same NVML allocator assert, **this time in
> the diffusion decoder**"* — logged the same day as sighting 1 and never set
> beside it. It was included because leaving it out would have made the register
> say "two" while its own source said "three".
>
> ✅ **It is now the one that got retired**, which is worth noting as a general
> matter: the instance that was hardest to fit the hypothesis (§4) is the
> instance that turned out to have the boring explanation, and it was only
> available to be retired because somebody refused to drop it for not fitting.

---

## 4. ⏸️ What became of the allocator-pathology hypothesis

The hypothesis was: *this partition may mishandle a large allocation that
follows another large allocation.* It was labelled **not established**
throughout, and the ruling was to measure the headroom before looking for a
mechanism. That was the right order and it is what happened.

**The headroom number arrived and it says there was no room.** R4b needed
55.906 GiB on a 47.375 GiB partition. That is ordinary exhaustion — the "boring
explanation" §3 of the old file said it preferred *until a number said
otherwise*. The number said so.

⏸️ **What the hypothesis is NOT — and this is the part a later reader will get
wrong.** It is not refuted. It is *unneeded for one of three sightings*:

* **Sighting 2**: explained by size. No pathology required. ✅
* **Sighting 1**: ⚠️ its ~26 GB estimate against a 47.375 GiB partition already
  holding an int8 transformer is *consistent* with plain exhaustion, and always
  was. Consistent is not measured, and §2 explains why it cannot now be.
* **Sighting 3**: ⏸️ still the awkward one, for the same reason it always was.
  2.49 GB is not a size ceiling on a partition that had just held a 22B
  transformer plus 121 frames of decode, and ✅ **the second call in the same
  worker succeeded** — a hard ceiling would have failed that one too. On the old
  47.375 GiB partition, with ✅ 37.924 GiB resident on entry and 1.19 GiB of
  headroom, plain exhaustion is *newly plausible* for it in a way it was not
  when nobody knew the partition size. ⏸️ Plausible is not measured either, and
  what would measure it is the residency change now shipping: an enhancer that
  loads once per worker either stops producing this assert or does not.

**So: one sighting explained by arithmetic, two consistent with it, none of the
three requiring a pathology — and no experiment that has excluded one.** That is
what "supported, not proven" means, stated at the level of the individual
sighting rather than the headline.

---

## 5. The transferable lesson, which now has THREE instances

> ### ⚠️ "Optimise one axis, never measure the other."
>
> The operator's own framing, and the reason this file records it next to the
> allocator sightings rather than in a commit message that scrolls away.

| | the axis that was optimised | the axis that was never measured | what it cost |
|---|---|---|---|
| **prompt enhancement, placement** | **disk** — CPU-in-the-web-process keeps the enhancer out of ZeroGPU's packing window. Researched, checked against the packer's source, and **correct** 📄 | **latency** — nobody put a stopwatch on a 600-token greedy decode on a CPU | ✅ **+538 s warm** (553.45 s against a 15.38 s baseline at 256×256×9). A nine-minute wall, priced at "1-2 s" — a GPU number applied to a CPU placement. Corrected at `f967e30` |
| **prompt enhancement, lifetime** | **VRAM safety** — load to CUDA, use, free, every call, so the enhancer is never resident at the denoise peak. Also a real property, also correctly argued | **the reload** — nobody measured what freeing it costs on the next call | ✅ **+30.78 s per enhanced generation** (46.53 s against a 15.75 s baseline at 256×256×9) ⚠️ almost exactly a 2.49 GB reload each time. **Resolved**: the enhancer is now resident |
| **the GPU size itself** | **quota** — `auto` was left in place partly because passing `size=` to an older `spaces` raises `TypeError` at import, which on a Space is an unexplained startup failure. A real hazard, correctly identified 📄 | **whether `auto` promotes at all** — the promotion was read in `startup()`, believed, and never checked against the request that actually reaches the scheduler | ✅ **Half the GPU, for the life of the project.** 47.375 GiB instead of 94.971 GiB. Cost R4b, cost the enhancer's residency, and cost the day of hypothesis in §4 |

**None of the three was a wrong instinct, and that is the point.** All three
identified a real constraint and answered it well. All three then treated the
axis they had answered as if it were the only axis.

> ### ⚠️ The third instance is worse than the first two, and it is worth saying why
>
> The first two were *our* unmeasured axes. The third was a claim about
> **somebody else's code** — that `startup()`'s promotion reaches the scheduler
> — held by four documents in this repo, each citing the others, none of which
> had followed the value past the function it was set in. Reading a source and
> reading a source *to the end of the value's life* are different acts, and only
> one of them is evidence.
>
> ⏸️ **The generalisation nobody has run:** every remaining claim in this repo of
> the form *"the platform does X for us automatically"*, traced to the call that
> would actually observe X.

---

## 6. What this file does not say

* It does **not** say the three sightings are resolved. §2 is one ✅ and two ⏸️,
  and §4 is explicit about which reasoning is measured and which is merely
  consistent.
* It does **not** say the allocator hypothesis is refuted. It says it is
  unneeded for the one sighting that got measured, and untested for the others.
* It does **not** reopen `cdb3211`. The tiled + framewise VAE decode is measured,
  deployed and load-bearing ✅. It is now known to have been fixing a real
  shortfall on a real partition, which strengthens it rather than the reverse —
  and §2 records why the operator refused a tiling-off toggle even for testing.
* It does **not** claim the partition change is free. `xlarge` bills at 2×, and
  the reservation arithmetic that assumed otherwise is corrected in
  `zerogpu.py`, `TEST_PLAN.md` and `OPERATOR_HANDOFF.md` alongside this file.
