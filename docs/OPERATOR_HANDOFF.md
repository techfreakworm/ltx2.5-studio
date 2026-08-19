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

Intended destination is a private HuggingFace ZeroGPU Space. **It does not run
there today** — see §3.

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

| # | Mode | Result | Streams | Size | Time |
|---|---|---|---|---|---|
| H1 | `text_to_video` | ✅ pass | h264 @24, no audio | 865,447 B | ~26 s |
| H2 | `image_to_video` | ✅ pass | h264 @24, no audio | 577,051 B | ~26 s |
| H3 | `image_text_to_video` | ✅ pass | h264 @24, no audio | 1,040,007 B | ~26 s |
| H4 | `video_to_video` | ✅ pass | video only | ⏸️ not recorded | ⏸️ |
| H5 | `text_to_audio_video` | ✅ pass | h264 + **aac, 2 ch** | ⏸️ not recorded | ⏸️ |
| H6 | `image_to_audio_video` | ✅ pass | h264 + aac, 2 ch | ⏸️ not recorded | ⏸️ |
| H7 | `image_text_to_audio_video` | ✅ pass | h264 + aac, 2 ch | ⏸️ not recorded | ⏸️ |
| H8 | `text_to_audio` | ✅ pass | **pcm_s16le, 2 ch**, no video | ⏸️ not recorded | ⏸️ |
| H9 | `video_to_audio` | ✅ pass | pcm_s16le, 2 ch, no video | ⏸️ not recorded | ⏸️ |
| H10 | `audio_to_audio` | ⏸️ **pending** | — | — | — |
| H11 | `audio_to_video` | ⏸️ **pending** | — | — | — |

**Nine of eleven modes have produced a real artifact with the expected stream
layout.** The two pending ones are the audio-**input** modes; see §4.

### 2.3 Beyond the baseline

| Run | What it proves | Result |
|---|---|---|
| **H16** high resolution | usability, not just correctness | ✅ `text_to_audio_video` at **1216 × 704**, aac stereo at 48 kHz |
| **H15** `duration_head` | the model chose the length, rather than a default leaking through | ✅ returned **97 frames**, not 121 |
| **H14** stage 2 | upsample → 3-step schedule → diffusion decode | ⏸️ **pending** |

H15 is worth understanding: `num_frames` is *omitted* and the duration head
picks. Had it returned 121 — the default — the run would have looked identical
to a pass while proving the opposite. **97 is the whole result.**

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

## 3. ZEROGPU-PENDING — read this before believing §2

> ### ⛔ The Space has evicted twice and has never run.
>
> Nothing in §2 is evidence that it will. A green table on a dedicated host and
> a working deployment are different claims, and only the first has been made.

hinode is an **unpartitioned** card with **124 GB RAM** and an ordinary
filesystem. ZeroGPU is none of those things, so Part A cannot answer:

- **Tensor packing** — ZeroGPU intercepts `.to("cuda")`, rewrites it, and packs
  every tensor to `~/.zerogpu/tensors` via O_DIRECT. Never exercised.
- **The 150 GB ephemeral-disk ceiling** — see
  [`ZEROGPU_DISK.md`](ZEROGPU_DISK.md) for the arithmetic, which is not
  restated here on purpose.
- **RAM** — 124 GB on hinode against a **~68 GB** community cgroup observation
  ⚠️ *inferred, never measured by us*.
- **The `size` kwarg**, the **fork boundary**, and **quota metering** — all
  first-contact.
- **The private-Space fetch path** — weights download at runtime because
  `preload_from_hub` 401s on a private repo.

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

2. **H14 — stage 2 — has not run.** ⏸️ It was unreachable (never wired into
   `StageRunner`), and fixing that surfaced six further defects in the stage
   graph, each of which would have produced a valid-looking video. All are
   fixed; none is exercised. **Treat stage 2 as unproven.**

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

5. **The audio input rate is 16000, the vocoder output is 48000.** 📄 They are
   not symmetric. Anything that assumes one rate for both directions will
   pitch-shift audio without erroring.

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

Not currently possible — §3. The parked analysis in
[`ZEROGPU_DISK.md`](ZEROGPU_DISK.md) lists four routes with what each costs;
choosing between them is an operator decision, not a code change.

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

### A module can be reached and still contain a dead function

The reachability sweep that exists to catch importer-count-zero missed one, for
two independent reasons: the tracer recorded which **file** ran and discarded
which **function** — the name was in the same frame object — and the probe drove
**one mode**, so a path reachable from no other mode was invisible by
construction. **Either fix alone would still have missed it.** "Sharpen the
instrument" was the obvious response and would not have worked.

### ⏸️ The staffing conclusion

**Not recorded here yet, deliberately.** It was reached in discussion and exists
in no document I can find, and writing it from memory would make this the second
place a fact lives and the first place it is wrong. It needs to be stated by the
person who reached it.

---

## 7. If you do one thing

Run **H10b** and **H14**. They are the two claims this project has not earned:
that a user's uploaded audio reaches the model, and that stage 2 does anything
at all. Both are built, both are tested offline, and **neither has produced an
artifact.** Everything else in §2 has.
