# The NVML allocator assert: three sightings, one shape, and a hypothesis nobody is chasing yet

> ## ⏸️ This file is a REGISTER, not an investigation.
>
> It exists because the same `RuntimeError` has now killed a run on this Space
> on two separate days, in three different modules, and each time it was
> recorded where that day's work happened — a resolution ladder in one session
> doc, an enhancer commit message in another. **Scattered, they are incidents.
> Side by side, they are a shape.** That is the whole reason to collect them.
>
> **Nobody should chase this yet, and that is a ruling rather than an
> oversight.** Sighting 2's premise may be about to disappear: the enhancer is
> loaded and freed on every call, and if it is made resident the allocation that
> failed stops happening. If the shape survives that change we will have two
> independent data points instead of one, which is worth more than an
> investigation launched now off a hypothesis with nothing under it.
>
> **What is being done instead is measurement.** `0a4ddbb` makes every
> generation report peak VRAM — allocated, reserved and total — split into an
> enhancement window and a generation window. Every question below is blocked on
> a headroom number this build has never had.

**Every load-bearing number is marked:** ✅ *we measured it* · 📄 *read off a
pinned source or the platform* · ⚠️ *inferred* · ⏸️ *not established*.

---

## 1. The sightings, side by side

| | 1 — VAE decode | 2 — diffusion decoder | 3 — prompt enhancer |
|---|---|---|---|
| **date** | 2026-08-25 ✅ | 2026-08-25 ✅ | 2026-08-27 ✅ |
| **run** | `text_to_video` 1216×704×121, stage 1 (R3, deploy 1) | stage 2, 704×480×49 input (R4b, deploy 2) | the **first** enhanced generation in a fresh worker |
| **what had already succeeded** | denoise, ~52 s ✅ | the latent upsample and the extra denoise steps ✅ | the worker was up with the transformer VRAM-resident ✅ |
| **where it raised** | `conv3d` inside `autoencoder_kl_ltx2.py:108` 📄 | `LTX2VideoDiffusionDecoderModel` 📄 | loading / running the 2.49 GB 📄 enhancer onto CUDA |
| **the error** | `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":1165` 📄 | recorded as *"same NVML allocator assert"* ✅ — ⏸️ **its traceback was not quoted at the time** | the same assert ✅, wrapped as `PromptEnhancementError` |
| **what the user saw** | our `DecodeError`, wrapped reason intact, legible client-side ✅ | ⛔ FAIL, no artifact ✅ | `PromptEnhancementError` naming the repo and the remedy — *"Turn the enhance-prompt toggle off to run with the prompt you typed"* 📄 |
| **allocation size** | ⚠️ ~26 GB of full-resolution decode intermediates | ⏸️ never measured | 2.49 GB of weights 📄, plus whatever the 600-token decode holds |
| **resolved?** | ✅ **yes** — tiled + framewise decode at `cdb3211`; the retry passed at 66.47 s / 6,258,833 B ✅ | ⏸️ **no** — `LTX2VideoDiffusionDecoderModel` exposes no tiling API at the pinned diffusers SHA 📄. The stage-2 ceiling sits between 25 and 49 input frames at 704×480 ✅ | ⏸️ **not resolved, and it went away**: the **second** call in the same worker succeeded ✅ |

Sources, unchanged and not rewritten:
[`SPACE_SESSION_2026-08-25.md`](SPACE_SESSION_2026-08-25.md) §4 (sighting 1) and
§5 (sighting 2); sighting 3 is recorded here for the first time.

> ### ⚠️ Sighting 2 is a THIRD instance, and it was already in the record
>
> The brief for this file named two. The 2026-08-25 session doc carries a third
> in §5 — R4b, *"same NVML allocator assert, **this time in the diffusion
> decoder**"* — logged the same day as sighting 1 and never set beside it. It is
> included because leaving it out would have made this register say "two" while
> the source says "three", and because it is the instance that most resists the
> hypothesis below (see §3). ⏸️ Its traceback was not quoted at the time, so
> "same assert" is that day's characterisation rather than a line-for-line match.

---

## 2. ⏸️ THE SHARED SHAPE — A HYPOTHESIS, EXPLICITLY NOT ESTABLISHED

> **This partition may mishandle a large allocation that follows another large
> allocation.**
>
> ⏸️ **Nothing establishes it.** It is a shape three incidents have in common,
> which is not the same as a mechanism, and this build has already spent a day
> undoing a confident claim that had a pattern behind it and nothing else.

What the three do share, stated as facts rather than as evidence for the
hypothesis:

* ⚠️ **All three were first-of-kind allocations.** 1216×704×121 had never been
  decoded on this Space; stage 2 had never run at 49 input frames; it was the
  **first** enhanced generation in that worker. None was a steady-state failure.
* ⚠️ **All three were multi-GB tensors landing on top of a resident int8
  transformer.** The transformer never leaves; every one of these allocations
  had to fit in what was left.
* 📄 **All three raised from inside the CUDA caching allocator's own
  bookkeeping**, not from a kernel and not as a plain `torch.cuda.OutOfMemoryError`.
  An NVML assert is the allocator failing to *ask* the driver a question, which
  is a different failure from being told "no".

⏸️ **The single most load-bearing detail, and it cuts both ways: the second
enhanced call in the same worker succeeded.** ✅ Measured. A hard headroom
ceiling would have failed the second call too. Something about the *first*
large-allocation-after-large-allocation in a worker is different — a
fragmentation state, a driver handshake, a cold NVML path. ⏸️ Which of those, or
whether it is any of them, is exactly what is not established.

### ⛔ What would refute it, cheaply

* Sighting 1 was fixed by making the allocation *smaller* (tiling), which is
  equally consistent with the boring explanation — **there simply was not enough
  memory** — and needs no allocator pathology at all. ⚠️ The ~26 GB estimate is
  inferred, and against an unknown partition size it settles nothing.
* Sighting 2 has no second-call counterpart: R4b was one attempt under the
  one-attempt ruling ✅, so nobody knows whether a retry in the same worker
  would have passed. If it would have, sighting 2 joins sighting 3. If it would
  not, the hypothesis has to explain why one is retriable and the other is not.
* A single flagship run with `0a4ddbb`'s reporting on could kill the hypothesis
  outright: if peak **reserved** is already within a couple of GiB of
  `total_memory`, all three sightings are ordinary exhaustion and there is
  nothing here.

**That is the order of work when this is unparked: read the headroom number
first, and only look for a mechanism if the headroom says there was room.**

---

## 3. ⚠️ Where the hypothesis is weakest

Sighting 2 fits worst. Its allocation was **not** obviously new-on-top-of-large
in the way the others were — the latent upsample and three denoise steps had
already run and released, and the diffusion decoder is the same code path R4a
survived one rung down at 25 input frames ✅. Read strictly, R4b looks like a
size ceiling: 49 frames needs more than 25 frames and there was not enough. That
is the plain reading, it requires no pathology, and **it is the reading this
file prefers until a number says otherwise.**

The reason to keep sighting 2 in the register anyway is that "plain size
ceiling" is also available for sighting 1, and *sighting 3 refuses it* — 2.49 GB
is not a size ceiling on a partition that had just held a 22B transformer plus
121 frames of decode, and it succeeded on the very next call.

⏸️ So the honest summary is: **one instance (3) resists the boring explanation,
one (1) is consistent with either, and one (2) leans boring.** That is a
register worth keeping and not a finding worth acting on.

---

## 4. The transferable lesson, which now has TWO instances

> ### ⚠️ "Optimise one axis, never measure the other."
>
> The operator's own framing, and the reason this file records it next to the
> allocator sightings rather than in a commit message that scrolls away.

| | the axis that was optimised | the axis that was never measured | what it cost |
|---|---|---|---|
| **prompt enhancement, placement** | **disk** — CPU-in-the-web-process keeps the enhancer out of ZeroGPU's packing window. Researched, checked against the packer's source, and **correct** 📄 | **latency** — nobody put a stopwatch on a 600-token greedy decode on a CPU | ✅ **+538 s warm** (553.45 s against a 15.38 s baseline at 256×256×9). A nine-minute wall, priced at "1-2 s" — a GPU number applied to a CPU placement. Corrected at `f967e30` |
| **prompt enhancement, lifetime** | **VRAM safety** — load to CUDA, use, free, every call, so the enhancer is never resident at the denoise peak. Also a real property, also correctly argued | **the reload** — nobody measured what freeing it costs on the next call | ✅ **+30.78 s per enhanced generation** (46.53 s against a 15.75 s baseline at 256×256×9) ⚠️ almost exactly a 2.49 GB reload each time. Unresolved — see the ruling at the top |

**Neither was a wrong instinct, and that is the point.** Both decisions
identified a real constraint and answered it well. Both then treated the axis
they had answered as if it were the only axis. A placement decision that answers
one axis and is silent on the other is not a decision, it is half of one — and
the missing half is the half that reaches the user.

> ### ⚠️ The failure mode is *silence*, not error
>
> In both cases the unmeasured axis produced no error, no warning and no log
> line. It produced a number nobody looked at. That is why the answer is
> `0a4ddbb` — **peak VRAM on every generation, in the warnings channel, whether
> anyone asked or not** — and not a runbook step someone is supposed to remember
> before the next residency decision.
>
> ⏸️ Note what that instrumentation still does not measure: **wall-clock per
> phase.** The +538 s and the +30.78 s figures above both came from an operator
> with a stopwatch on the live Space, not from the app. This file records that
> gap rather than closing it.

---

## 5. What this file does not say

* It does **not** say the three sightings share a cause. §2 is a hypothesis and
  is labelled as one throughout.
* It does **not** say the enhancer should be made resident. That is exactly the
  decision `0a4ddbb`'s numbers exist to inform, and making it now — off the same
  evidence base that produced the two rows in §4 — would be the third instance
  of the pattern §4 is about.
* It does **not** reopen `cdb3211`. The tiled + framewise VAE decode is measured,
  deployed and load-bearing ✅; nothing here suggests changing it, whatever the
  headroom number turns out to be.
* It does **not** know the size of this MIG partition. ⏸️ No number in this
  build has ever read `total_memory` on the Space. That is the first thing the
  next flagship run will print.
