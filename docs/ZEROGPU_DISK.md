# ZeroGPU disk — PARKED. Findings to date.

> ## ⏸️ This is not an active workstream.
>
> **ZeroGPU is parked.** The sequence is: prove the app works on a dedicated GPU
> host **first**, then return to platform logistics. This file exists so the
> findings are not lost, **not** so anyone acts on them.
>
> **No further research has been done and none should be** until the app is
> proven. Several questions below are marked ❓ deliberately — they are open, and
> opening them is future work rather than a gap.
>
> **The two things to re-read on return** are **§1 — the packing-peak arithmetic**
> and **§6 — the mmap precondition and the guard for it**. Both stand alone;
> everything else is context for choosing between options.

The app does not fit ZeroGPU's ephemeral disk. Four routes out are recorded below
with what each costs. **Nothing here is implemented, and the choice is the
operator's.**

**Every claim is labelled.** ✅ *verified against source* · ⚠️ *inferred* ·
❓ *assumed, and the assumption decides the answer*. This document exists because
several earlier confident numbers turned out to have nothing behind them.

---

## 1. The problem, in one table

ZeroGPU gives **150 GB of ephemeral disk**, wiped on every cold start. Startup
holds **two** copies of the weights at once:

| | GB |
|---|---:|
| HF cache snapshot (downloaded) | 72.19 |
| ZeroGPU packed copy in `~/.zerogpu/tensors` | 72.19 |
| **peak** | **144.37** |
| the image itself — torch + CUDA | +8–12 |
| **total vs a 150 GB cap** | **152–156 — OVER** |

✅ ZeroGPU *does* free the snapshot after packing. ✅ It frees it **too late**: the
unlink runs strictly after `_pack()` returns (`patching.py:395`), so the peak
above happens first regardless.

**The margin is negative, not tight.** Steady state afterwards would be fine at
~72 GB. It is the peak that evicts us.

### 1.1 A different mechanism, which fits the timing the peak does not

⚠️ **Candidate cause, not settled — the Space has not been re-run.**

`from_pretrained` was resolving against the Hub and starting a naive snapshot of
**48 files** even with a verified-complete 37/37 cache, pulling the dead 8-shard
`transformer/` set and the dead single-file `connectors/` copy: **44.33 GB** on
top of the 72.19 GB we intend. ✅ Measured on the host, cache 68 → 109 GiB and
still climbing when it was killed. That is **116.52 GB before packing begins at
all**.

**Why this displaces the peak as the explanation of the actual failure.** The
Space died during `Loading pipeline components 6/9` / `Loading checkpoint shards
2/4` — *inside* `from_pretrained`. The packed-copy arithmetic in §1 requires
packing to have started, and packing begins only **after** `from_pretrained`
returns. The peak was right as a constraint and wrong as the cause, and the
contradiction was visible in a log we had already read.

✅ **Fixed** in `preflight.snapshot_dir()` — construction now loads from the local
snapshot directory, which has no manifest and no repo to resolve against, so the
refetch is structurally impossible rather than merely disabled.

⛔ **`local_files_only=True` is not the fix**, and was never recorded here as one.
Tested on the host: it switches on a strict completeness check against that same
manifest, which demands the 14 dead files the allow-list excludes **on purpose**,
turning a working load into a hard refusal.

### 1.2 The allow-list is not an optimisation — it is what makes the Space possible

✅ Read off the cached manifest, not inferred: the full repo is **56 files,
164,192,096,189 bytes (164.19 GB)**.

> **The full repo does not fit ZeroGPU's 150 GB cap even with nothing else on
> disk** — no packed copy, no image, no CUDA.

Our 37-file / 72.19 GB allow-list is therefore load-bearing rather than frugal.
What it excludes:

| excluded | GB |
|---|---:|
| dead `-of-00008` `transformer/` set | 37.99 |
| dead single-file `connectors/` copy | 6.34 |
| `transformer_full/` (full/SFT DiT; we run the distilled one) | 37.98 |
| standalone distilled LoRA | 9.70 |
| **total excluded** | **92.01** |

72.19 + 92.01 = 164.20, which reconciles with the manifest total to rounding.

---

## 2. What is already fixed, and what it bought

Two defects were found and fixed while diagnosing this. **Neither is sufficient**,
and the table above is already post-fix.

| fix | before | after |
|---|---:|---:|
| loading the model **once** instead of per workflow | 216.56 GB peak | 144.37 |
| ✅ the packer dedupes shared tensors at pack time (`patching.py:360`, `AliasId`) | — | confirms the above |

So the eviction was **over-determined**: three independent causes, and the two
smallest absorbed most of the diagnostic effort. What remains is structural.

---

## 3. The four options

### Option A — mount the model repo as a read-only volume · ⚠️ **mechanism real, target filesystem almost certainly incompatible**

Hugging Face supports mounting a model repo into a Space rather than downloading
it:

```python
api.set_space_volumes(repo_id=..., volumes=[
    Volume(type="model", source="Lightricks/LTX-2.5-Diffusers",
           mount_path="/models", read_only=True)])
```

**If the weights are mounted, there is no snapshot on the ephemeral disk at all.**

| | GB |
|---|---:|
| snapshot | **0 — mounted** |
| packed copy | 72.19 |
| image | ~10 |
| **peak** | **~82 vs 150** |

**What it buys beyond the disk:** full bf16 quality, **no 25-minute cold-start
download**, no 1-hour startup timeout, and the gated-repo authentication problem
disappears. It solves four things at once.

**❓ Four assumptions, and the first two decide it:**

1. **Are mounted volumes counted outside the 150 GB ephemeral quota?** The docs
   describe them as distinct storage but never state this. If they count against
   the same cap, Option A buys nothing.
2. **Does the mount support `mmap`?** Packing requires it. A mount that does not
   support `mmap` fails outright.
3. **Revision pinning.** We pin `REVISION = 871165de…`. A mounted repo probably
   tracks `main`, which would silently defeat the pinning that makes *"has
   upstream changed under us?"* answerable at all.
4. `model_files.py` needs a local-path mode — small, and precisely what the
   single indirection was built for.

**Cost:** whatever HF charges for volume storage. **Verdict:** best outcome if
assumptions 1 and 2 hold; both are cheap to test and neither costs GPU quota.

---

### Option B — int8 runtime quantization with bitsandbytes · **most robust**

✅ The packer sizes by **real tensor bytes** (`packing.py:76`,
`size = tensor.numel() * tensor.element_size()`), so an int8 tensor packs at
1 byte/element against bf16's 2. **Runtime quantization shrinks the packed copy
even though the snapshot on disk stays bf16.**

| scheme | packed | peak = **steady if the prune fails** | steady **if the prune fires** |
|---|---:|---:|---:|
| bf16 — current | 72.2 | **154.4 — OVER** | 82.2 |
| **int8** | 38.1 | **120.3 ✓** | 48.1 |
| int8, connectors left bf16 | 41.2 | 123.4 ✓ | 51.2 |
| nf4 / 4-bit | 21.0 | 103.2 ✓ | 31.0 |

All columns include the ~10 GB image. Non-quantised remainder (vae, audio_vae,
vocoder, upsamplers, diffusion_decoder) is 3.95 GB.

> ⚠️ **The middle column used to read 8–12 GB lower**, because a "steady state"
> column had the image subtracted out. **If the prune never fires, nothing is
> freed — steady state *is* the peak**, image included. And **the image is
> precisely the term that pushes bf16 over the cap**, so dropping it understated
> the hard case in the favourable direction, in the one row a reader quotes.
> Corrected; the conclusion is unchanged.

**The middle column is why this is the robust option.** Every other route needs
the prune to work. Int8 fits **in both branches** — if quantised tensors break
the mmap capture and nothing is ever freed, it is still 120.3 GB and still fits.
**It does not need a second thing to go right.**

**✅ bitsandbytes is explicitly supported and needs no special handling.**
`spaces/zero/torch/bitsandbytes.py` is a 43-line version check that **no-ops for
bnb ≥ 0.46.0**, wired into the packing path at `patching.py:328/355/445`. The
legacy shim exists only for older versions.

**✅ torchao is NOT supported — there is no equivalent module.** That asymmetry is
invisible from outside and it selects the library. Use bitsandbytes.

> **Correction on the record:** an earlier version of the plan said the packer's
> handling of quantised tensors was *unknown*, and that reason was used to remove
> the precision lever. **It was false.** The lever's removal stands, because it
> was wired to nothing — but the stated reason was wrong and is corrected here and
> in PLAN §2.6.

**Cost:** output quality. int8 is a real reduction from bf16, unquantified for
this model. **Verdict:** the option to take if Option A's assumptions fail, and
the one to take *first* if quality loss is acceptable, because it is the only one
that survives both branches.

---

### Option C — relocate the packed copy via `ZEROGPU_OFFLOAD_DIR`

⚠️ A clean lever: read once at `config.py:27`, used only in `pack()`. Point the
packed copy at storage outside the ephemeral disk and the peak drops to
snapshot + image ≈ 82 GB. **Full bf16 quality retained.**

**⚠️ The risk is `O_DIRECT`.** Both the write and the read path use it
(`packing.py:108`, `:199`), and `O_DIRECT` commonly returns `EINVAL` on
network-backed volumes. If persistent storage is network-backed, this fails.

**Cost:** persistent storage. **Verdict:** full quality, one sharp unverified
risk, cheap to test.

---

### Option D — make the Space public and preload · **do not take this**

Counterintuitive, so stated plainly: **`preload_from_hub` makes the problem
permanent rather than transient.**

⚠️ *Inference from overlayfs semantics, not verified.* Preloaded weights arrive as
**read-only image layers**. The autoprune frees space by calling `path.unlink()`.
Unlinking a file whose bytes live in a read-only overlay layer creates a
**whiteout**: the name disappears, the bytes do not.

So the prune would log `"Cleaned 72GB of tensor files after packing"` and free
**nothing** — converting a survivable transient peak into a permanent floor, **and
lying in the log while doing it.**

**Verdict:** rejected. It also conflicts with the stated requirement that the
Space be private.

---

## 4. Blocked — not an option, so nobody re-proposes it

**Pre-quantized checkpoints.** The int8-convrot and nvfp4 files exist **only in
`Lightricks/LTX-2.5`, the native repo** — none are in the Diffusers pack. ✅ And
our single-file path cannot load them: `CHECKPOINT_KEY_NAMES["ltx2"]` →
`ltx2-dev` → config repo `Lightricks/LTX-2` (**that is 2.0**), with every
converter named `LTX_2_0_*` and **nothing for 2.5's connectors,
diffusion_decoder, vocoder, duration_head or latent_upsampler**.

**The blocker is FORMAT, not size.** This is a different thing from Option B and
must not be conflated with it — runtime quantization is the live lever;
pre-quantized files are unreachable.

---

## 4.1 ⚠️ Option A is almost certainly blocked at the filesystem

The lever is real — `ZEROGPU_OFFLOAD_DIR` is read at import (`config.py:27`) and
is settable as a Space Variable. **What it points at is the problem.**

The packer uses `O_DIRECT`, **plus `posix_fallocate` of the whole ~72 GB file**,
plus hand-aligned page buffers. That is raw-NVMe idiom. The only writable volume
HF offers is an NFS/FUSE lazily-fetched object store, where `posix_fallocate` is
commonly unsupported — and it is **the first call after open**, so it fails
before a single tensor is written.

✅ *Answered from `spaces` source.* Cheap to confirm if anyone wants certainty:
it fails immediately or not at all.

---

## 4.2 ✅ RULED: int8 via bitsandbytes — **the arithmetic, derived here**

**Both bounds fit, and the conservative one has 12.6 GB of headroom.**

Derived from `model_files` component sizes rather than taken from an earlier
estimate. **The snapshot stays bf16 on disk** — that is what we download — so
only the *packed* copy shrinks, and the peak is `snapshot + packed + image`.

```
snapshot on disk (bf16)                      72.185 GB
  Linear-heavy: transformer/text_encoder/connectors   68.240
  conv-only:    vae/audio_vae/vocoder/decoder/upsamplers  3.909
  tokenizer/scheduler/duration_head                       0.036
```

| | packed | peak (+image 8–12) | vs 150 GB |
|---|---:|---:|---|
| **A** all Linear-heavy quantised | 38.065 | **118.3 – 122.3** | ✅ fits |
| **B** transformer only *(conservative)* | 53.197 | **133.4 – 137.4** | ✅ fits |
| — bf16 today, for contrast | 72.185 | **152.4 – 156.4** | ⛔ over |

**Both bounds are given because the split cannot be determined from here.**
`bitsandbytes` `Linear8bitLt` replaces `nn.Linear` **only** — convolutions,
norms and embedding tables keep their dtype. **A** assumes every Linear-heavy
component halves; **B** assumes only the transformer does, which covers the
common case of leaving the text encoder alone and the possibility that Gemma's
embedding table is a large non-Linear share of its 23.9 GB. ⏸️ The true figure
is between them and needs measuring on the platform.

### Why this is the route: it fits **whether or not the prune fires**

| | prune fires | prune never fires |
|---|---:|---:|
| **A** steady state | ~50 GB | **118.3 – 122.3** ✅ |
| **B** steady state | ~65 GB | **133.4 – 137.4** ✅ |

⚠️ **A quantised load materialises new tensors, so the autoprune may free
nothing** — the right-hand column is that world, and it is already priced in.
**That is not a defect waiting to be discovered; it is the branch this option
was chosen to survive.** Every other option needs the prune to work, and the
prune is the one thing that cannot be exercised off-platform.

### ✅ The derivation retrodicts the evictions

**The strongest thing that can be said for this arithmetic is that it predicts
a failure nobody fed into it.** bf16 comes out at **152.4 – 156.4 GB against a
150 GB cap**, and the Space **evicted twice**. The model was built from
component sizes and a published ceiling, with no reference to what happened;
it explains what happened anyway.

✅ And the input reconciles to a host measurement: the derived snapshot of
**72.185 GB** against the **72,186,355,712 B** actually downloaded — **agreement
to about 1 MB.**

That moves this from *internally consistent* to *having a confirmed
prediction*, which is the only kind of arithmetic worth deploying on.

### ⚠️ What would still sink it: RAM, not disk

⏸️ **Disk is not the risk.** The bounds clear the cap by 12.6 GB even in the
conservative case, and the model has a confirmed retrodiction behind it.

**RAM is, and it is now quantified from one side.** ✅ A bf16 load peaks at
**20.05 GiB** host RSS against **65.370 GiB** of weights — far below, because
safetensors is **mmap-backed** and the pages are never all resident.

⚠️ **A quantising load loses exactly that property.** It materialises int8
tensors, which is anonymous memory by construction — *the same mechanism this
document already relies on to argue the post-pack prune may free nothing.* One
mechanism, and it decides both questions.

So int8's RSS should be expected to approach the **materialised** weight size —
order **38 GB for A, 53 GB for B** — plus whatever of the bf16 source is
resident while converting. **Not 20 GiB.** That turns the ~68 GB cgroup figure
from ⚠️ *a number someone mentioned* into **the binding constraint, and the
first thing to measure.** `_base_pipeline` logs peak RSS across the load for
exactly this reason.

The other open one is whether `spaces` packs quantised tensor subclasses at all — 📄 `bitsandbytes ≥ 0.46.0` is explicitly
no-op'd by `spaces/zero/torch/bitsandbytes.py` (`patching.py:328/355/445`),
which is what makes bnb the supported path and **torchao not one**.



📄 **bitsandbytes ≥ 0.46.0 needs no ZeroGPU handling at all**:
`spaces/zero/torch/bitsandbytes.py` no-ops `_patch`/`_unpatch`/`_move` above that
version, wired at `patching.py:328/355/445`. **There is no torchao equivalent** —
which is what makes bnb the supported path rather than a preference. Pin it with
a floor **and** a ceiling.

> ⏸️ **Prepared and gated, deliberately not landed.** It must not ride along with
> the schedule-and-guidance correction: quantisation changes output quality too,
> and landing both at once means **neither can be attributed**. Sequence is the
> corrected bf16 matrix first, signed off; **then** int8 against that baseline —
> which turns "what does int8 cost us in quality?" from a guess into a
> same-seed pair anyone can look at once and answer.

---

## 5. Recommendation

| rank | option | quality | needs the prune to work? | blocked on |
|---|---|---|---|---|
| **1** | **A — mounted volume** | full bf16 | no | ❓ quota accounting, ❓ mmap support |
| **2** | **B — int8 runtime** | reduced | **no** | quality acceptance |
| **3** | **C — offload dir** | full bf16 | no | ⚠️ `O_DIRECT` on network storage |
| — | D — public + preload | full bf16 | **makes it worse** | rejected |

**A first**, because it costs no quality and solves the cold start and the auth
problem alongside the disk. **B if A's assumptions fail**, because it is the only
route that survives both branches of the prune question. **C as the full-quality
fallback** if int8 quality proves unacceptable.

A and C are both cheap to test and neither costs GPU quota. **Test A's two
assumptions before choosing** — the answer to "are mounted volumes counted
separately?" decides this document.

---

## 6. The mmap guard — BUILT and PARKED

**Status: built, tested, mutation-checked, and deliberately not wired in.** It was
written after ZeroGPU was parked — crossed messages — and kept rather than
reverted because it is provably inert off-Space and re-enabling costs one line.
**Do not extend or maintain it until ZeroGPU is unparked.**

`zerogpu.check_mmap_backed()` and its helpers are ZeroGPU-only and cannot fire
elsewhere.

**What they do:** walk `/proc/self/maps`, resolve each model parameter's
`data_ptr()` back to its backing file, and require a match against the autoprune
glob. Anything that materialises tensors into anonymous memory — a dtype cast, a
`.clone()`, an eager `low_cpu_mem_usage=False`, a non-safetensors path — leaves
nothing for the prune to unlink. It then frees **nothing**, and the Space dies at
**steady state** with 72 GB it cannot account for and no error explaining why.

**To re-enable:** one call in `pipelines._base_pipeline()` after
`from_pretrained` and before any device move, raising `ModelLoadError` when it
reports a violation. The tests already cover the cast case, the wrong-file case,
and weakening the predicate itself.

**Why it is not on now:** it was escalated to a requirement mid-refocus. It is a
ZeroGPU requirement, and ZeroGPU is parked.

## 7. What this does not decide

The app itself is being proven on a dedicated GPU host, which answers *is the
application correct?* and **cannot** answer *does it fit ZeroGPU?* — there is no
tensor packing there, no 150 GB ceiling, and 124 GB of RAM against ZeroGPU's
~68. A green result there is **HOST-PROVEN** and remains **ZEROGPU-PENDING**.
