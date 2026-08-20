# Operator handoff — LTX-2.5 Studio

> **What you have, what you do not have, and what to do next.** Written for
> someone who was not here. [`PLAN.md`](PLAN.md) records how things were
> decided; [`TEST_PLAN.md`](TEST_PLAN.md) records what was run; this tells you
> where the project actually stands.

**Every load-bearing number below is marked with where it came from:**
✅ *measured on hinode* · 📄 *read off a pinned source or config* ·
⚠️ *inferred* · ⏸️ *not yet established*.
If a number carries no mark, treat that as a defect in this document.

---

## 1. What this is

A Gradio studio for **LTX-2.5** (Lightricks' 22B video+audio model) covering
**all eleven generation modes**, with **user-uploadable LoRA validated at
inference time**. One model load serves every mode. Python 3.12, `diffusers`
pinned to an exact SHA, weights pinned to an exact revision.

Destination is a private HuggingFace ZeroGPU Space. ✅ **It runs and generates
there** — startup, one clean GPU worker, and a content-verified video, at Space
sha `8fb292d6`.

⏸️ **That is one mode, once.** Ten of the eleven have never executed on that
platform; §2's eleven-of-eleven is a **hinode** result and stays one. §3 is the
section that keeps those two evidence bases apart — read it before treating
either as the other.

---

## 2. HOST-PROVEN

Everything in this section ran on **hinode**: an **unpartitioned RTX PRO 6000
with 124 GB RAM** ✅ — through `get_pipeline` exactly as `app.py` calls it, not
through a test harness.

### 2.1 Startup gates

| Gate | Result | |
|---|---|---|
| Weights present | **37 / 37** files, 72.185 GB | ✅ |
| Full repo, for contrast | 56 files, **164,192,096,189 B (164.19 GB)** | 📄 manifest |
| Components shared between the two pipelines | **9 / 9 identical by `is`** | ✅ |
| `base.dtype` after deriving | **`torch.bfloat16`**, unchanged | ✅ |
| CUDA cost of the second pipeline | **−1024 bytes** | ✅ |
| Total for both pipelines | **65.370 GiB** | ✅ |

The **−1024 bytes** is the number to re-check after any change to pipeline
construction: it is what proves the second workflow shares the first's weights
rather than loading its own copy. A positive figure in the tens of GB means the
sharing has broken.

### 2.2 Modes

Baseline is **704 × 480 × 121 frames**, seed pinned to **1234**. "Streams" is
what `tests/verify_streams.py` observed in the container, not a human reading.

| # | Mode | Result | Streams observed | Size | Time |
|---|---|---|---|---|---|
| H1 | `text_to_video` | ✅ | h264 704×480@24, f=121 | 791,782 B | 8.4 s |
| H2 | `image_to_video` | ✅ | h264 704×480@24, f=121 | 307,583 B | 7.8 s |
| H3 | `image_text_to_video` | ✅ | h264 704×480@24, f=121 | 775,213 B | 8.0 s |
| H4 | `video_to_video` | ✅ | h264 704×480@24, f=121 | 708,885 B | 8.6 s |
| H5 | `text_to_audio_video` | ✅ | h264 + **aac 48000 Hz 2 ch** | 792,803 B | 7.8 s |
| H6 | `image_to_audio_video` | ✅ | h264 + aac 48000 Hz 2 ch | 374,184 B | 7.9 s |
| H7 | `image_text_to_audio_video` | ✅ | h264 + aac 48000 Hz 2 ch | 995,843 B | 8.2 s |
| H8 | `text_to_audio` | ✅ | **pcm_s16le 48000 Hz 2 ch**, no video | 961,998 B | 1.8 s |
| H9 | `video_to_audio` | ✅ | pcm_s16le 48000 Hz 2 ch, no video | 961,998 B | 2.0 s |
| H10 | `audio_to_audio` | ✅ | pcm_s16le 48000 Hz 2 ch, **−10.4 dBFS** | 961,998 B | 2.8 s |
| H11 | `audio_to_video` | ✅ | h264 704×480@24, f=121, no audio | 193,825 B | 8.0 s |

**Eleven of eleven.** All ✅ measured on the real `get_pipeline` path, no bypass
and no harness accommodation; `verify_streams` returned 0 on every one.

| beyond the baseline | | |
|---|---|---|
| **H14** stage 2 | 1408 × 960, f=121 | 1,807,646 B · 35.2 s |
| **H15** `duration_head` | 704 × 480, **f=97** not 121 | 593,527 B · 5.9 s |
| **H16** high resolution | **1216 × 704** + aac stereo | 1,775,743 B · 21.0 s |
| **H17** mel round-trip | 4.000 s in → 4.010 s out, **4/4 tones** | strict gate |
| **H12 / H13a / H13b** | LoRA accepted / rejected / rejected | stage 6.5 |

> **Wall-clock excludes the model load**, which is **14.0 s once** per batch ✅.

> ### ⚠️ Structure is not content
>
> Every run here passed `verify_streams`. **The one defect that mattered most
> was invisible to it**: `video_to_audio` came back 50 dB below its siblings
> with correct codec, channels, rate, duration and a non-empty file. **Silence
> and a healthy signal have identical stream layout.**
>
> It was found because someone measured RMS on an artifact that had already
> been signed off, and it led to a defect affecting **all twelve modes** (gap
> 7). A green table says these are well-formed files. **It is a weaker claim
> than "correct output", and the gap between the two is where this build spent
> its afternoon.**
>
> H8 and H9 are byte-identical in size because uncompressed PCM of equal
> duration always is. That is why size proves nothing about content, and why it
> took an RMS measurement to notice.
>
> **On audio levels generally.** Measured levels sort by *condition type*, not
> by mode ✅ — text-only −5.2 to −6.2 dBFS, image-conditioned −8.2 to −15.2,
> video-conditioned at strength 0.8 −54.5 to −55.0. The text-only group peaks at
> exactly **1.0000 with no samples exceeding**, so the export is faithful and
> **the saturation is upstream, in the model, not in our writer**. Those files
> are consistent with each other and sit at the rail; ⏸️ whether that is correct
> for this model has not been established.

### 2.5 What the recipe fix bought — before / after

Same seed, same geometry on both sides of every pair; **only the distilled
recipe (gap 7) differs.**

| # | mode | before | after | wall-clock |
|---|---:|---:|---:|---|
| H1 | `text_to_video` | 886,265 | 791,782 | 25.7 → **8.4 s** |
| H2 | `image_to_video` | 432,486 | 307,583 | 26.3 → **7.8 s** |
| H3 | `image_text_to_video` | 1,130,178 | 775,213 | 26.6 → **8.0 s** |
| H4 | `video_to_video` | 727,490 | 708,885 | 27.0 → **8.6 s** |
| H5 | `text_to_audio_video` | 1,091,468 | 792,803 | 25.0 → **7.8 s** |
| H6 | `image_to_audio_video` | 620,537 | 374,184 | 26.6 → **7.9 s** |
| H7 | `image_text_to_audio_video` | 784,282 | **995,843** | 26.7 → **8.2 s** |
| H8 | `text_to_audio` | 961,998 | 961,998 | 5.6 → **1.8 s** |
| H9 | `video_to_audio` | 961,998 | 961,998 | 6.0 → **2.0 s** |
| H14 | stage 2 | 2,254,799 | 1,807,646 | 84.3 → **35.2 s** |
| H15 | `duration_head` | 684,374 | 593,527 | 19.4 → **5.9 s** |
| H16 | 1216 × 704 | 2,331,528 | 1,775,743 | 68.7 → **21.0 s** |

**Geometry is identical on both sides of every pair** — H14 stays 1408 × 960,
H15 stays 97 frames, the rest 704 × 480 × 121. *Sizes changed; geometry did
not* is the shape of the result.

Three things a reader needs:

- **H7 is the only artifact that grew.** h264 size tracks content complexity,
  so that is neither good nor bad on its own. Flagged, not interpreted.
- **H8 and H9 are byte-identical** because uncompressed PCM at one duration,
  rate and channel count always is. **Their change is entirely in level**, not
  in bytes — see below — so reading this table alone would call them unchanged
  when they are two of the three that moved most.
- **~3× faster everywhere.** At `guidance_scale=1.0` with STG and modality
  zeroed there are no classifier-free branches, so it is one forward pass per
  step instead of two or three. **This is the one column here that seeding does
  not confound** — wall-clock is not drawn from a distribution — and it follows
  from the mechanism rather than merely accompanying it.

> ### ⚠️ These pairs are not yet controlled — the audio evidence is
>
> **The seed never reached the model** until `4ebb8b9` (gap 8), so both halves
> of every pair above were unconditionally random. **The pixel differences
> cannot evidence that the recipe fix caused them** — two runs of one mode
> differed anyway. ✅ Seeding is fixed, and verified byte-identical **for
> `audio_to_audio`**; a seeded gallery would supersede this table for
> attribution ⏸️ **once the video path is shown reproducible too**. These
> figures remain as the record of what the defect cost.
>
> **The audio levels do stand on their own**, because they are systematic and
> orders of magnitude outside run-to-run variation:
>
> | | before | after |
> |---|---:|---:|
> | H9 `video_to_audio` | −55.0 dBFS | **−17.8 dBFS** |
> | H8 `text_to_audio` | −5.2 dBFS, **peak 1.0000** (railed) | **−15.3 dBFS**, peak 0.6953 |
> | highest peak, any mode | **1.7811** | **1.0050** |
>
> A 50 dB gap and a clipped peak going away are not a seed.
>
> ⏸️ **And the replacement is not yet in hand.** Bitwise reproducibility is
> verified for `audio_to_audio`; the **video** path is still being checked, and
> the seeded gallery can only attribute per-pair change once it is.

> ### ✅ Sizes differed between the two code paths — almost certainly just the seed
>
> Three modes were run through a **bypass** (`_base_pipeline` plus an injected
> `StageRunner`) and through the **real** `get_pipeline` path, **all pre-fix**:
>
> | | bypass | real path |
> |---|---:|---:|
> | H1 | 865,447 B | 886,265 B |
> | H2 | **577,051 B** | **432,486 B** — 25% smaller |
> | H3 | 1,040,007 B | 1,130,178 B |
>
> This was recorded as an open question — *"GPU non-determinism, or a genuine
> difference between the routes?"* — and described as *"at the same seed."*
>
> **It was not at the same seed. The seed reached nothing** (gap 8). These are
> six unconditionally random runs, and two runs of one mode differed by this
> much anyway. **There was never an anomaly to explain**, and the phrase *"at
> the same seed"* was doing all the work: it made an ordinary difference look
> like a finding.
>
> ⏸️ Cheap to settle properly now — run one mode both ways at a fixed seed.
> Nobody has, so it is *probably* the seed rather than certainly.
>
> The advice it produced stands on its own and is worth more than the question:
> **assert stream layout, never an exact size.**

### 2.3 Beyond the baseline

| Run | What it proves | Result |
|---|---|---|
| **H16** high resolution | usability, not just correctness | ✅ `text_to_audio_video` at **1216 × 704** @24, f=121 + aac 48000 Hz 2 ch — 1,775,743 B, **21.0 s** |
| **H15** `duration_head` | the model chose the length, rather than a default leaking through | ✅ **f=97**, not 121 — 704 × 480, 593,527 B, 5.9 s |
| **H14** stage 2 | upsample → 3-step schedule → diffusion decode | ✅ **1408 × 960**, f=121 — 1,807,646 B, 35.2 s |

*Post-fix figures, matching §2.2. §2.5 carries the pre-fix column.*

H15 is worth understanding: `num_frames` is *omitted* and the duration head
picks. Had it returned 121 — the default — the run would have looked identical
to a pass while proving the opposite. **97 is the whole result.**

**H14's proof is its resolution, not a comparison.** ✅ 1408 × 960 against H1's
704 × 480 is **exactly 2× in both dimensions** — a size only the spatial
upsampler can produce, and **the one claim here that no vintage or seed can
affect**, since geometry is identical across every before/after pair. That is stronger evidence than the
same-seed diff the matrix planned, because a same-seed comparison shows *a*
difference while this shows *the specific* difference stage 2 exists to make.

### 2.4 The LoRA validator, both directions

| Run | Adapter | Result |
|---|---|---|
| **H12** | `joyfox/LTX-2.3-Transition-LORA` | ✅ **ACCEPTED**, **576 modules** across transformer + connectors |
| **H13a** | `charlied98/…` `lora_weights_step_03000` | ✅ **REJECTED at stage 6.5**, `LoRADimensionError` |
| **H13b** | same repo, `comfy_lora_weights_step_03000` | ✅ **REJECTED at stage 6.5**, same mechanism |

H12 carries **mixed dimensions** — 4096 at attention, 2048 at `audio_ff`, 16384
at `ff.net.2` ✅ — and none of that is a mismatch. An earlier "expect hidden dim
4096" rule would have **rejected a perfectly good adapter**; the false-positive
guard is what stops that. H13a and H13b differ only in key prefix
(`transformer.` vs `diffusion_model.`) and reach the same verdict by the same
mechanism, which is what makes the rejection trustworthy rather than lucky.

---

## 3. SPACE-PROVEN vs HOST-PROVEN — read this before believing §2

> ### ✅ The Space runs and generates. ⏸️ That is **one mode, once.**
>
> `text_to_video` at 256×256×9 produced a real, content-verified video on the
> live private Space ✅. **Ten of the eleven modes have never executed there.**
>
> A green table on a dedicated host and a working deployment are different
> claims. **Both have now been made — about different things.** That is exactly
> why this section still exists: the two evidence bases have diverged rather
> than merged, and §2's eleven-of-eleven remains a **hinode** claim.

**Detailed record: [`SPACE_GENERATION_FINDING.md`](SPACE_GENERATION_FINDING.md)**
— the first deploy, the killed workers, the falsified ceiling, and the run that
generated. Not restated here on purpose.

hinode is an **unpartitioned** card with **124 GB RAM** and an ordinary
filesystem. ZeroGPU is none of those things. **The first deploy answered several
of these questions and changed the answer to one of them:**

- ✅ **Tensor packing** — exercised. Packing ran and the post-pack prune fired,
  freeing 65.25 GB.
- ✅ **The 150 GB ephemeral-disk ceiling** — never the binding constraint. The
  disk line read **82.2 GB of 150 [OK]** while the workload died of something
  else. Arithmetic in [`ZEROGPU_DISK.md`](ZEROGPU_DISK.md).
- ⛔ **RAM — the ~68 GB figure is FALSIFIED.** See the finding below; this is the
  single most important correction in this document.
- ✅ **The private-Space fetch path** — works. 37/37 files at the pinned revision,
  downloaded at runtime because `preload_from_hub` 401s on a private repo.
- ✅ **The `size` kwarg** and the **fork boundary** — reached; a worker forked,
  ran, and released cleanly (`fail=false`).
- ⏸️ **Quota metering** — still unreadable. All three candidate endpoints 404, so
  billed seconds can only be **bounded**, never measured.
- ⏸️ **Ten of eleven modes** — no conditioning, no stage 2, no LoRA, and **no
  audio path** has run on ZeroGPU. The audio modes exercise the mel front-end,
  the `av` loader and the 16 kHz/48 kHz asymmetry; none of that has executed
  there once.

---

### The three findings from the first deploy

> #### 1. ⛔ The **~68 GB** RAM ceiling is **FALSIFIED**
>
> It was a community cgroup observation — **inherited, never ours**, and
> load-bearing in planning since before any deploy. Startup reached
> **87.13 GiB = 93.56 GB** peak RSS and **was not killed**.
>
> ✅ **Replace it with our own measured lower bound: the ceiling is above
> 93.56 GB.** Every headroom figure computed against ~68 GB was wrong by at
> least 25.56 GB.

> #### 2. ⚠️ Quantising **more** raised peak RSS by **31.24 GiB**
>
> Adding the text encoder to `components_to_quantize` moved RSS **up**, from
> 55.89 to 87.13 GiB, in a change made to bring it down. `safetensors` mmaps
> lazily; **quantisation materialises those weights as anonymous memory**, so
> quantising a component destroys the laziness that was keeping it cheap.
>
> This is the **same single mechanism** `ZEROGPU_DISK.md` §4.2 already used to
> settle two other questions, now deciding a third — in the opposite direction
> from what we predicted.

> #### 3. ⏸️ The `SIGKILL` is **still unexplained**
>
> cgroup-OOM is **dead** as an explanation: with the ceiling above 93.56 GB
> there was never the narrow margin the hypothesis required. ✅ **Route C —
> instrument the worker — is now the LEADING path, not the fallback.** It is the
> only route that returns evidence rather than another inference, and every
> inference made about this kill has been wrong at least once.

> #### ⛔ And the fix is **UNATTRIBUTED**
>
> Two things changed between the failing and succeeding runs: the quantisation
> widening **and a full container rebuild**. n=1 against n=1 with two variables.
> **"The Space now serves" is established; "the widening is why" is not** — and
> the hypothesis that motivated the widening is itself dead. Do not let any
> document claim the widening fixed generation.

**Cost so far: ≤ 243.58 GPU-seconds** of 2400/day ⚠️ *a bound, not a
measurement* — quota endpoint 404s.

**One hinode number with a direct ZeroGPU consequence.** Stage 2 peaked at
**83.897 GiB** ✅, against 70.497 GiB for a single-stage run. So **stage 2 cannot
run on a `large` 48 GB MIG slice at all**, and on `xlarge` (96 GB) it leaves
roughly **11 GiB** before any packing overhead. That is measured on an
unpartitioned card; the packing overhead is ⏸️ unknown and comes off that margin.

One live risk that Part A *has* surfaced: the stage-2 diffusion decoder resolves
a **NATTEN kernel from the Hub at generation time** 📄, which is the only
network dependency that fires *after* the user presses Generate. Whether it
caches per image or per cold worker is ⏸️ unknown and decides its severity.

---

## 4. Known gaps, stated plainly

1. **H10 / H11 — the audio-input modes — have not passed.** ⏸️ The path they
   need was connected to nothing until very recently: the UI collected a file,
   the encoder existed and was exported, and **nothing called it**. It is built
   and tested now, and it has not yet produced an artifact on the card.
   **The acceptance test is H10b**, not H10: two `noise_scale` values at the
   same pinned seed. *Identical audio means the input is still not reaching the
   model*, whatever the stream check says — because `noise_scale=1.0` discards
   the input by design, so a bug that drops it entirely is indistinguishable
   from correct operation.

2. **Stage 2 works; its NATTEN fetch is characterised.** ✅ **Resolved once per
   process, not per generation** — two stage-2 generations in one process, one
   kernel fetch. So a Space needs outbound network at a worker's **first**
   stage-2 use rather than on every press of Generate. Still a real dependency,
   because **each cold worker pays it once**. Retained below for the mechanism.

   ✅ H14 ran and
   produced 1408 × 960. But the diffusion decoder resolves its NATTEN kernel
   through a **version → ref lookup that goes over the network even when the
   kernel files are cached** 📄, and **whether that fires once per process or
   once per generation is ⏸️ unknown**. It cannot be grepped after the fact —
   the run log contains **zero** natten mentions — so answering it needs
   deliberate instrumentation. On a private Space this is the 401 class again,
   except firing *after* the user presses Generate.

3. **`connectors: 0` on every accepted adapter.** ✅ observed. Stage 7 asserts
   *total* modules attached > 0, which passes on the transformer alone — so an
   adapter that *ought* to touch `connectors` and silently does not would still
   be reported as attached. Probably correct for the three adapters tested
   (2.3/2B-era), so **not currently a defect** — but the assertion as written
   cannot distinguish "legitimately unused" from "silently missed". A
   per-component expectation would close it. See PLAN §4.4.

4. **`temporal_latent_upsampler` has no known role.** 📄 It is the same class as
   the spatial upsampler, differing only by axis, and upstream's component table
   and both documented recipes mention only the spatial one. We download it
   (0.2619 GB) and do **not** wire it in. Not a defect; an open question.

5. ### ✅ `video_to_audio` near-silence — **fixed, and the first diagnosis was wrong**

   H9 came back at **−55.0 dBFS** against `text_to_audio`'s −5.2 ✅. **The cause
   was the distilled recipe reaching only the stage-2 path (item 7).** With that
   corrected, H9 runs at **−17.8 dBFS** — 2.5 dB from its siblings — at the
   **unchanged** condition strength of 0.8.

   | | before | after |
   |---|---:|---:|
   | H9 `video_to_audio` | −55.0 dBFS, peak 0.0146 | **−17.8 dBFS**, peak 0.5078 |
   | H8 `text_to_audio` | −5.2 dBFS, **peak 1.0000** (railed) | **−15.3 dBFS**, peak 0.6953 |
   | gap | **~50 dB** | **2.5 dB** |

   **One mode 50 dB down and another driven into the rail were the same
   misconfiguration seen from opposite ends.**

   ⚠️ **The first diagnosis blamed condition strength and was wrong.** A
   single-lever sweep at fixed seed gave 0.8 → −55.0 and 0.5 → −5.1, and that
   was read as the cause. It was a real interaction **between condition strength
   and the broken guidance** — a genuine effect inside a pipeline that no longer
   exists. A per-mode strength cap was written, committed and reverted.

   > **A single-lever same-seed sweep is strong evidence that something moves,
   > and no evidence about why.**

   The source citation said throughout that a video condition *cannot* attenuate
   audio in the denoise loop — the conditioning mask is applied to
   `noise_pred_video` and never touches `noise_pred_audio`. It was correct, and
   a clean correlation was believed over it.

6. **The audio input rate is 16000, the vocoder output is 48000.** 📄 They are
   not symmetric. Anything that assumes one rate for both directions will
   pitch-shift audio without erroring.

7. ### ✅ The distilled recipe reached only the stage-2 path — **fixed**

   Eleven of twelve runs sent **no sigmas at all**, so upstream built its own
   schedule and five guidance parameters took their **SFT** defaults —
   `audio_guidance_scale` at **7.0** against an intended 1.0.

   ```
   sent:      [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
   intended:  [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
   ```

   ✅ All twelve artifacts differed before and after, with magnitudes ordering
   by conditioning strength (text-only 37–59 mean |diff|, image-conditioned
   15–16, video-conditioned **3.71**).

   ⏸️ **Neither of those establishes the fix caused them, and the ordering is
   not the corroboration it looks like.** Both halves were unseeded (gap 8), so
   two runs of one mode differed anyway — and a **more strongly conditioned mode
   is more constrained, so its run-to-run variance is smaller too.** The
   ordering is exactly what random variation would produce. It cannot separate
   the two explanations, and an earlier draft of this section claimed it could.

   **What does stand, unconfounded:**

   - **the timings.** 25.7 → 8.4 s is not run-to-run variation, and it follows
     directly from the mechanism: unguided means one forward pass per step
     instead of two or three. **The defect cost 3× the compute**, and that is
     measured rather than inferred.
   - **the audio levels and peaks** — −55.0 → −17.8 dBFS, highest peak
     1.7811 → 1.0050. Systematic and orders of magnitude outside variation.
   - 📄 **that the schedule and guidance were wrong at all**, which is a source
     fact and needs no run.

   **It cost 3× the compute as well as the quality.** At `guidance_scale=1.0`
   with STG and modality zeroed there are no classifier-free branches, so it is
   one forward pass per step rather than two or three:

   | | before | after |
   |---|---:|---:|
   | H1 `text_to_video` | 25.7 s | **8.4 s** |
   | H8 `text_to_audio` | 5.6 s | **1.8 s** |
   | H14 stage 2 | 84.3 s | **35.2 s** |
   | H16 **1216×704 + stereo** | 68.7 s | **21.0 s** |

   Clipping went with it — highest peak **1.7811 → 1.0050** ✅ — so no limiter is
   needed and `outputs._write_audio`'s *"what comes back is what the vocoder
   produced"* principle stands. **Our path never added gain; the material was
   arriving at ±1.0.** Stage 2's brightness falloff was this too, not stage 2:
   mean `99.48 → 34.44` collapsing before, `82.52 → 110.47` stable after.


8. ### ✅ The seed reached only the sidecar — **fixed, and verified byte-identical**

   `_resolve_seed` ran and the number went into the encoder metadata. **There
   was no generator anywhere in the package**, so every run was unconditionally
   random and every *"same seed"* comparison was two independent runs.

   ✅ **Fixed** in `4ebb8b9`: a generator seeded from the request, built on the
   pipeline's own device, threaded to the pipeline **and** to
   `audio_vae.encode`'s `.sample()`.

   ✅ **Verified on the card, for `audio_to_audio`.** Two runs at identical seed
   and configuration:

   ```
   run-a1d35aaaab06/output.wav   md5 2988f01159a5322afb28abb6e9a73d26
   run-b9dde07ed8bb/output.wav   md5 2988f01159a5322afb28abb6e9a73d26
   ```

   ✅ And the fix was complete when it landed, not repaired afterwards:
   `git log -S 'generator' 4804da1..e4e8df1 -- ltx25_studio/` returns `4ebb8b9`
   alone.

   > **On this path, `seed` determines the output byte-for-byte, on GPU, with no
   > deterministic-algorithm flags set.** Stronger than expected — the weaker
   > *"seed-consistent but not bit-reproducible"* was drafted and never needed.

   ⏸️ **Scope, and it matters for §2.5:** that measurement is **one mode**. The
   **video path is not yet shown to be bitwise reproducible**, and the whole
   twelve-mode pixel-diff gallery depends on it — if a video mode carries its
   own unseeded draw, those per-pair numbers stay unattributable however good
   the audio seeding is. A `text_to_video` pair at one seed is running to
   settle it.

   **The remaining consequence is narrow**: the §2.5 comparison table was
   measured *before* this worked, so its pixel diffs cannot attribute the change
   to the recipe fix. The seeded gallery replaces it. Everything structural was
   never affected — H14's exact 2× resolution, H15's 97 frames, H17's frequency
   recovery, and every audio level.

   > The lever that should have caught it was ours: `decode_kwargs` took a
   > `generator` argument, **with a docstring explaining why the diffusion
   > decoder must be seeded**, and read a key nothing ever set. See PLAN §9
   > rule 12.

9. ### ⏸️ `noise_scale` and `sigmas[0]` disagree — open, deliberately unfixed

   📄 Upstream infers one from the other — *"first (largest) sigma value if
   using custom sigmas"* — and our `stage_2_kwargs` honours it. The
   audio-input modes do not: they send `noise_scale` while `sigmas[0]` stays
   1.0.

   **It was diagnosed as the cause of a silent `audio_to_audio` and it was
   not** (§4.1). A fix was written, committed and **withdrawn**: truncating the
   schedule at `noise_scale` takes an eight-step run to **two** steps at 0.5
   and **one** at 0.1 — values that measured healthy on eight.

   **Agreeing with a contract is not evidence that a change improves
   anything.** A test pins the current behaviour and explains why, so the next
   attempt arrives with a measurement rather than with agreement.

---

## 4.1 ⚠️ Known limitation — unusual audio input produces a silent file

**Not a defect we can properly fix, and the most likely way a user meets a
disappointing result.**

`audio_vae.encode` is **never called by the pipeline**. Our
`encode_waveform_to_latents` is the only producer of those latents, and
**nothing anywhere validates their distribution** — not upstream, not us.

So audio that is far from what the model was trained on encodes to latents the
model cannot use, and the result is **a structurally perfect, inaudible file
with no error and nothing to act on.** ✅ A real instance, and it cost a day of
misdiagnosis: a two-tone synthetic fixture at **−14.8 dBFS in** produced
**−87.9 dBFS out**, with correct codec, channels, rate and duration, and
`verify_streams` exit 0. The same mode fed a real recording produces **−10.4
dBFS**.

**Likely triggers:** pure tones, sine sweeps, test signals, near-silence,
heavily synthesised material. **Ordinary recorded audio is fine** — every
realistic input tried has worked.

⛔ **Do not "fix" this with an out-of-distribution detector.** Reliable OOD
detection at that boundary is not something this project can do, and a
detector that is wrong in either direction is worse than the limitation.

⏸️ **A narrower thing is worth doing and is deliberately not done yet.** We
measure the input level and can measure the output level, so **a file that
comes back essentially silent could say so** in `GenerationResult.warnings`.
The decision that makes it honest is that it must be a **measurement, not a
diagnosis**: *"this output is −87.9 dBFS, effectively silent"* is a fact about
the artifact, checkable and cause-free. *"Your audio was out of distribution"*
is a claim we cannot support.

It is not built because it needs a threshold, a threshold needs evidence, and
the PR opens with **one** named defect. **Adding an unvalidated heuristic to
the export path hours beforehand is the same move as shipping a schedule change
on a contract reading** — which was reverted for that reason. It belongs in the
next pass, with a measured floor.

---

## 5. How to run it

### On hinode

```bash
./scripts/sync_to_hinode.sh          # ships git archive HEAD, not the working tree
```

`rsync` is deliberately not used: it is not installed on the host and there is
no sudo to add it. Needs an HF token with access to the gated model repo. Then
`scripts/hinode_run.sh`, or `scripts/host_matrix.py` for the full matrix.

**Do not set `HF_HUB_OFFLINE=1`.** Construction loads from the local snapshot
directory, so nothing is fetched at load time — and stage 2 needs outbound
network for the NATTEN kernel. Forcing offline breaks the second without helping
the first.

### Locally, with no GPU

The whole test suite runs with **no GPU, no weights and no network**. That is a
deliberate property, not a coincidence, and it is worth preserving: it is what
makes it possible to catch a defect before it costs a run.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

`torch`, `diffusers` and `torchaudio` are **not** needed for the suite. `av` and
`numpy` are, because they are how real containers get written and read.

### On a Space

✅ **This works.** The repo *is* the Space — `README.md`'s YAML front matter is
the Space config, and the `space` remote points at it:

```bash
git remote -v | grep space
# space  https://huggingface.co/spaces/techfreakworm/LTX2.5-Studio
```

Deploying is a push to that remote's `main`. ⛔ **This team does not push** — not
to `origin`, not to `space`. The orchestrator does it; that is a deliberate
boundary, not a missing capability.

**What the deploy depends on, all of it already load-bearing:**

| | |
|---|---|
| `app_file: app.py`, `sdk: gradio 6.24.0`, `python_version: "3.12"` | 📄 front matter |
| `suggested_hardware: zero-a10g` | 📄 **a legacy id string — not an A10G.** Changing it moves the Space onto paid dedicated hardware |
| `startup_duration_timeout: 1h` | 📄 **required.** The default is 30 min and this Space downloads 72.19 GB at build time. When it blows, the failure looks like something else entirely |
| an HF token secret with access to the gated model repo | 📄 weights fetch at **runtime**, because `preload_from_hub` 401s on a private Space |

`tests/test_deploy_drift.py` asserts these rather than trusting them — run the
suite before any deploy and the drift shows up locally instead of as a
40-minute build that dies at the end.

**Reading the startup log is the cheapest diagnostic available**, and it costs
zero GPU quota because packing and loading happen at module scope, before the
first `@spaces.GPU` call. The load line names its own configuration:

```
quantised transformer, text_encoder as int8 | peak RSS 0.91 -> 87.13 GiB
```

⚠️ **That peak is the number to watch.** It is a *peak across the load*, not a
steady state — we have never measured steady state, and the RAM questions are
about steady state. See [`SPACE_GENERATION_FINDING.md`](SPACE_GENERATION_FINDING.md) §7.

---

## 6. The lessons worth carrying

Not a diary. These are the ones that would change how the next person works.

### A fake derived from the mapping can never catch a wrong mapping

The audio VAE's six parameters were read from `audio_vae.config` **by names that
do not exist there** 📄 — four live under different keys, two are instance
attributes absent from `config.json` entirely. Every runtime path through that
module raised on real weights, and every offline test passed, because the fake
used *our* names. It agreed with the code and with nothing else.

The instinctive repair — build the fake from the same mapping the code uses — is
a trap for exactly the same reason. The fix is a **recorded fact about reality**:
the real config's key set, checked into the tests with its provenance. The fake
checks semantics; the recorded keys check the world.

### The fixture that makes a test possible is itself untested by that test

Three instances in one day, all in tests that were otherwise excellent:

1. `test_encoder_contract.py` drove the **real** encoder through **real**
   containers to a real file — and built its audio as `np.float32`. The model
   produces `bfloat16`. **Six of eleven modes were broken at the last step and
   1926 tests passed** ✅ *both counts measured — the modes on hinode, the tests
   in this repo at the commit before the fix*.
2. The replacement fake's `.float()` returned an `ndarray` where torch returns a
   *tensor*. Written the convenient way, it would have passed a broken fix.
3. The audio-VAE fake used our own key names — the case above.

The mitigation that worked: **check the fake against the real thing.** In (2)
that check failed within a minute of existing.

### A search that cannot see the claim reports that the claim is gone

Three defects in this build share one shape: **the instrument returned a
confident negative it was structurally incapable of turning positive.**

- **A wrapped line.** §7 said *"The Space has evicted twice and has never run"*
  as one line. §1 said the same thing **split across a line break**, so a grep
  for the phrase found the occurrences that fit on one line and pronounced the
  audit clean.
- **A shell modifier.** `"$c:ltx25_studio/..."` is parsed by zsh as the `:l`
  lowercase modifier, so `git show` got a mangled path and **every** commit
  answered *"constant absent"* — including the one that demonstrably contains it.
- **A stale marker.** A `⚠️ (inherited)` pointing at a section that had been
  rewritten to say **FALSIFIED** still scanned as *merely unverified*.

**None of the three raised an error.** Each returned a clean, plausible, wrong
answer, and two of them nearly closed an audit.

The defence is not a better search string. It is **running the instrument
against a case whose answer you already know**: if the check cannot go positive
on a known positive, its negatives carry no information. The zsh case was caught
only because *"absent at HEAD too"* was impossible on its face.

### A claim moving known → unknown is nearly always a regression

Documents accrete facts and rarely lose them. When an edit is about to record
something as uncertain, search first for it being *known* — the likeliest
explanation is that someone established it and the current author has not read
it. This has since caught its own inverse: a correction issued against this
document's history that the history did not actually contain.

### Checks over comments

Twice, a constraint written down did not survive the next function written by
its own author — a `dtype` requirement violated two commits after the ⚠️ comment
demanding it, and a 44.33 GB refetch reintroduced *while reading the docstring
that predicts it*. A comment is a message to a future stranger; it has no
purchase on the author five minutes later. **Make the wrong thing unsayable
instead:** derive the value once and share it, so the two callers cannot drift.

### The instruments encode the failure you imagined, not the failures available

`host-runner`'s generalisation, in their words:

> *all three instruments encoded the failure I imagined rather than the failures
> available*

— a `pkill -f` pattern that matched the killer, a wait loop with no failure
exit, and a fixture guard that handled a fixture being *absent* but not one that
*raises*. **The three had different authors, including the lead.** That is the
point of recording it: it is not one person's blind spot, it is what happens
whenever the person who imagines the failure is the person who writes the check
for it.

### We can prove a function ran. We cannot prove it ran with the right arguments.

**The gap this build ends on**, and the one worth closing next.

`DISTILLED_SIGMA_VALUES` was defined, documented as mandatory, quoted in its
module's docstring, and **referenced** — so every check we own passed. It was
referenced from `stage_1_kwargs()`, which `StageRunner` applies **only when
stage 2 is enabled**. Eleven of twelve host runs were generated with upstream's
linear fallback and its **SFT** guidance defaults instead:

```
sent:      [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
intended:  [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
```

A reference check passes. A reachability check at module level passes. Even the
function-level sweep passes, unless someone thought to name that particular
function. **What nothing checked was the arguments the pipeline was actually
called with.**

The pattern that does work, used three times here, is to **record the vendor's
specification independently and assert against that** — `MODEL_CARD_DISTILLED`,
`REAL_AUDIO_VAE_CONFIG_KEYS`, `REAL_VOCODER_CONFIG_KEYS`. Each is a fact copied
from the source of truth rather than derived from our own constants, so changing
ours cannot quietly redefine what correct means. Without that independence the
test agrees with the code and with nothing else.

> **The escalation ladder, in the order this build climbed it:** is the module
> imported → is it *reached by the path that does the work* → did this specific
> *function* run → **were the arguments the ones the vendor specifies.** Only
> the last rung is still hand-built, one call site at a time.

### A module can be reached and still contain a dead function

The reachability sweep that exists to catch importer-count-zero missed one, for
two independent reasons: the tracer recorded which **file** ran and discarded
which **function** — the name was in the same frame object — and the probe drove
**one mode**, so a path reachable from no other mode was invisible by
construction. **Either fix alone would still have missed it.** "Sharpen the
instrument" was the obvious response and would not have worked.

### The one time source-reading and measurement confirmed each other

Every other pairing today went the other way: one corrected the other, and the
lesson was about which to trust when they disagreed. Once, they agreed.

A reproducibility control failed — two runs at one seed, 8.7% apart — and the
obvious reading was an unseeded draw somewhere. Instead of hunting, **every
noise source in the pinned pipeline was enumerated**: `_create_noised_state`,
the conditioning mask, `prepare_*_latents`, the decode timestep, the condition
VAE encode. All threaded the generator; the encode used `sample_mode="argmax"`
and was deterministic regardless. **Source said it should already be
reproducible.**

Rather than write either sentence — *"the seed is attached to nothing"* or
*"we are seed-consistent but not bit-reproducible"* — a scoping run was
requested. Before it ran, the control was re-checked: **it had been measured
against a tree that predated the seeding fix.** At the tip, two runs are
byte-identical.

> **The true claim was stronger than either draft**: on `audio_to_audio`, `seed`
> determines the output byte-for-byte, on GPU, with no deterministic-algorithm
> flags. Both drafted sentences were weaker *and* wrong, and both would have
> shipped if the reluctance to write an unmeasured claim had not held for one
> more exchange.

**And then the corrected sentence was written without its scope, twice more in
this document** — as a property of the build rather than of one mode. ⏸️ The
video path is still unverified. Recording that here because the lesson is not
*"be careful"*: it is that **a claim narrowed in one place does not narrow
itself in the others**, and a document is exactly where that goes wrong. §4
gap 8 carries the scope; §2.5 and this paragraph each had to be found and
fixed separately.

**Refusing to write a sentence you cannot yet support is the same discipline as
refusing to change a threshold to make a test green** — and it is the cheaper
of the two, because nobody has to notice you did it.

### Staffing — two makers and a lead who verifies

**Not five, and not one.** This build ran briefly with five teammates and settled
at two: one implementer owning the code, one runner owning the hardware. The
three stood down were not underused — they were **a coordination cost with no
matching throughput**, and the collision they produced (two agents assigned
`outputs.py`; a cross-boundary change landing under a commit message about other
files) cost more than their output was worth. Two makers with **disjoint, stated
file ownership** is the shape that worked, and the line held all day under
pressure: the runner found four product defects and edited none of them.

**The scarce resource was execution on real hardware, not review capacity.** At
the point the suite reported **1926 passing** ✅, six of eleven modes were broken,
the audio-input path was connected to nothing, stage 2 was unreachable, and every
runtime path through `audio.py` raised on real weights. None of that was found by
reviewing harder — it was found by an afternoon on a real GPU. **Another reviewer
would have added agreement, not information.** Budget for one more agent is
better spent on a second executor with *different* hardware than on a second
reviewer.

**A lead who cannot edit is forced to verify by observation rather than by
reading their own diff**, and that constraint turned out to be a feature. Every
claim the lead checked was checked by running something — `git merge-base
--is-ancestor` rather than trusting a commit message, `av` on the container
rather than trusting a verifier's exit code, `pgrep` on the box rather than
trusting a status report. That caught the box silently trailing the tip **three
times**, a stale bypass premise, and a wait loop that had frozen a teammate. A
lead who could edit would have been tempted to fix those, and would have stopped
looking.

**The default-REJECT verifier role was filled by reality rather than by an
agent.** The plan called for an independent verifier holding a default-REJECT
stance; in practice the hardware held it. It rejected the `from_pipe` cast, the
bf16 encode, the missing audio parameters and the unreachable stage 2 without
being asked, and without any possibility of being talked round. **Where a real
gate exists, prefer it to a reviewer agent — an agent can be persuaded and a
`RuntimeError` cannot.**

**What it cost.** Message-crossing: three times a blocking item was buried inside
a long message and lost a cycle behind work already in flight. The fix is
mechanical and worth adopting from the start — **one blocker, one message,
nothing else in it.** Long messages are for context; short ones are for
priority.

---

## 7. If you do one thing

**Re-run the comparisons seeded.** Everything works; the open item is that some
of the *evidence* was gathered before the seed reached the model.

### The runs that need repeating, and what each would then establish

| run | what it would establish | why it does not yet |
|---|---|---|
| **§2.5 before/after pairs** | that the recipe fix caused the pixel differences | both halves were unconditionally random |
| **H10b** — `noise_scale` 0.1 vs 0.9 | that the user's audio reaches the model | same, plus the audio encode drew unseeded |
| **H12b** — LoRA effect | that attaching the adapter changed the output | same |
| **H4b** — `strength` 0.3 vs 0.9 | that the conditioning clip reaches the model | same |

Add a **same-value control** to each — two runs at *identical* settings. Under
a working seed its spread should be **zero**, and that is what turns "the halves
differ" into evidence rather than a coin toss. An earlier control measured 1.8%
and was read as a bound on the variance; **an unseeded control cannot bound
anything**, which is the mistake worth not repeating.

### What is already earned, and does not need repeating

**Eleven of eleven modes produce artifacts with the expected stream layout on
the real path.** ✅ H14's exact 2× resolution, H15's 97 frames, H17's round-trip
recovering 4/4 tones under a strict gate, both LoRA directions, and every audio
level — all structural or systematic, and none of them depends on seeding.

**And read §3 before treating any of this as a deployment.** ✅ The Space now
runs and generates — ⏸️ **for one of the eleven modes.** Everything in this
section is a **hinode** result; the Space has proven `text_to_video` at
256×256×9 and nothing else. Those are different claims about different machines,
and §3 is where they are kept apart.

