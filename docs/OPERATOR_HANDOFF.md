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

| # | Mode | Result | Streams observed | Size | Time |
|---|---|---|---|---|---|
| H1 | `text_to_video` | ✅ | h264 704×480@24, f=121 | 886,265 B | 25.7 s |
| H2 | `image_to_video` | ✅ | h264 704×480@24, f=121 | 432,486 B | 26.3 s |
| H3 | `image_text_to_video` | ✅ | h264 704×480@24, f=121 | 1,130,178 B | 26.6 s |
| H4 | `video_to_video` (strength 0.7) | ✅ | h264 704×480@24, f=121 | 727,490 B | 27.0 s |
| H5 | `text_to_audio_video` | ✅ | h264 + **aac 48000 Hz 2 ch** | 1,091,468 B | 25.0 s |
| H6 | `image_to_audio_video` | ✅ | h264 + aac 48000 Hz 2 ch | 620,537 B | 26.6 s |
| H7 | `image_text_to_audio_video` | ✅ | h264 + aac 48000 Hz 2 ch | 784,282 B | 26.7 s |
| H8 | `text_to_audio` | ✅ | **pcm_s16le 48000 Hz 2 ch**, no video | 961,998 B | 5.6 s |
| H9 | `video_to_audio` | ⚠️ §4.5 | pcm_s16le 48000 Hz 2 ch, no video | 961,998 B | 6.0 s |
| H10 | `audio_to_audio` | ⏸️ **not run** | — | — | — |
| H11 | `audio_to_video` | ⏸️ **not run** | — | — | — |

All ✅ measured on the real `get_pipeline` path, no bypass, `verify_streams` 0.

> **Wall-clock excludes the model load**, which is **14.0 s once** for a whole
> batch ✅. A reader comparing 5.6 s against 25.7 s is comparing generation, not
> startup.
>
> ⚠️ **These sizes predate the schedule-and-guidance fix** (gap **7**). The
> corrected matrix has since run and **all twelve modes changed** — so treat
> these as the *before* half of a pair, not as expected values. Corrected
> timings are in gap 7; **every run is roughly 3× faster**, and the flagship
> 1216×704 with stereo now takes **21.0 s**.

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

> ### ⏸️ Sizes differ between the two code paths, and it is not known why
>
> Three modes were run both through a **bypass** (`_base_pipeline` plus an
> injected `StageRunner`) and through the **real** `get_pipeline` path at the
> same seed:
>
> | | bypass | real path |
> |---|---:|---:|
> | H1 | 865,447 B | 886,265 B |
> | H2 | **577,051 B** | **432,486 B** — 25% smaller |
> | H3 | 1,040,007 B | 1,130,178 B |
>
> **This is recorded as open, not as noise.** It may be GPU non-determinism —
> cuDNN picks convolution algorithms heuristically and small pixel differences
> compound hard through h264 — or it may be a genuine difference between the two
> routes. ⏸️ Nobody has established which, and calling it non-determinism would
> be a guess in the reassuring direction.
>
> Either way: **assert stream layout, never an exact size.** A byte-count check
> here would fail for reasons nobody understands, and be "fixed" by loosening it
> until it proved nothing.

### 2.3 Beyond the baseline

| Run | What it proves | Result |
|---|---|---|
| **H16** high resolution | usability, not just correctness | ✅ `text_to_audio_video` at **1216 × 704** @24, f=121 + aac 48000 Hz 2 ch — **2,331,528 B** |
| **H15** `duration_head` | the model chose the length, rather than a default leaking through | ✅ **f=97**, not 121 — 704 × 480, 684,374 B |
| **H14** stage 2 | upsample → 3-step schedule → diffusion decode | ✅ **1408 × 960**, 121 frames, 2,254,799 B, 84.3 s |

H15 is worth understanding: `num_frames` is *omitted* and the duration head
picks. Had it returned 121 — the default — the run would have looked identical
to a pass while proving the opposite. **97 is the whole result.**

**H14's proof is its resolution, not a comparison.** ✅ 1408 × 960 against H1's
704 × 480 at 886,265 B and 25.7 s is **exactly 2× in both dimensions** — a size
only the spatial upsampler can produce. That is stronger evidence than the
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

   **All twelve modes changed when it was corrected** ✅ — nothing silently
   survived it. The magnitudes order by conditioning strength (text-only 37–59
   mean |diff|, image-conditioned 15–16, video-conditioned **3.71**), which is
   what makes them coherent rather than noise: the strongest condition pins the
   output hardest and leaves the guidance least room to move it.

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

**Two things, and they are the two claims this project has not earned.**

### First: run H17, the mel round-trip

**H10 and H11 passing does not establish that the audio-input path is correct**
— only that it runs. The forward mel transform does not exist in the checkpoint
and had to be built; **five of its parameters are read from config and the rest
are conventions** (`power`, `f_min`/`f_max`, `mel_scale`/`norm`, `center`, the
log and its floor). A wrong choice there produces **well-formed audio that is
quietly degraded** — which is precisely the failure mode this document keeps
returning to. `TEST_PLAN` A.5.1 specifies the round-trip and the
failure-ordered candidate list.

⚠️ Its fixture must have **structure a wrong window would damage** — two tones
and an envelope. **Noise and silence both survive a wrong analysis window
looking much the same**, so either would make the test pass while proving
nothing.

### Second: run H10b

*That a user's uploaded audio reaches the model* is otherwise unproven.

Two `noise_scale` values — 0.1 and 0.9 — at the **same pinned seed**, on
`audio_to_audio`. **Identical audio means the input is not reaching the model**,
however good the file looks and whatever the stream check returns.

It is the only run that can see this, and the reason is worth understanding
before deciding to skip it: `noise_scale=1.0` discards the audio input **by
design**, so a bug that drops the input entirely is indistinguishable from
correct operation. The mode produces valid audio either way. Every other check
in the matrix passes either way.

Everything else in §2 has produced an artifact and had its **structure**
observed. Structure was never the hard part.
