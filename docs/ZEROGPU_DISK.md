# LTX-2.5 Studio on ZeroGPU — the disk problem, and four ways out

**A decision document.** The app is finished and correct; it does not fit ZeroGPU's
ephemeral disk. There are four routes out, they trade money against quality
against risk differently, and **the choice is the operator's**. Nothing here has
been implemented.

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

### Option A — mount the model repo as a read-only volume · **recommended, if the assumption holds**

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

| scheme | packed | peak (snap + packed + ~10 image) | steady state **if the prune fails** |
|---|---:|---:|---:|
| bf16 — current | 72.2 | **154.4 — OVER** | 144.4 |
| **int8** | 38.1 | **120.3 ✓** | **110.3 ✓** |
| int8, connectors left bf16 | 41.2 | 123.4 ✓ | 113.4 ✓ |
| nf4 / 4-bit | 21.0 | 103.2 ✓ | 93.2 ✓ |

Non-quantised remainder (vae, audio_vae, vocoder, upsamplers, diffusion_decoder)
is 3.95 GB.

**The last column is why this is the robust option.** Every other route needs the
prune to work. Int8 fits **in both branches** — if quantised tensors were to break
the mmap capture and nothing were ever freed, steady state is still 110 GB and
still fits. It does not need a second thing to go right.

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

## 6. What this does not decide

The app itself is being proven on a dedicated GPU host, which answers *is the
application correct?* and **cannot** answer *does it fit ZeroGPU?* — there is no
tensor packing there, no 150 GB ceiling, and 124 GB of RAM against ZeroGPU's
~68. A green result there is **HOST-PROVEN** and remains **ZEROGPU-PENDING**.
