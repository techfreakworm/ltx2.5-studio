# LTX-2.5 Studio — Design Plan

**Status:** implemented; not yet run on a GPU. The matrix in
[`docs/TEST_PLAN.md`](TEST_PLAN.md) is what proves it works.
**Branch:** `soma/ltx25-studio` · **Remote:** `git@github.com:techfreakworm/ltx2.5-studio.git`
**Target:** private Hugging Face Space `techfreakworm/LTX2.5-Studio` (gradio SDK, ZeroGPU).

> ### ⚠️ Read this before you trust the test suite
>
> **This suite was green at 1102 passing tests while five blockers were live in the tree.**
> At that moment the app could not reach a GPU, could not apply a LoRA, and reported every
> successful run in its three flagship modes as an internal error. Nothing was failing.
>
> The ~700 tests added since are almost entirely **seam and reachability** tests, not more unit
> coverage — because more tests of the original shape would have found none of it. §9.1.1
> explains the pattern and the one check that catches it.
>
> If you are about to trust a green gate here the way we did: that is the mistake this document
> exists to warn you out of.

This document is the contract for the build. Every section is either a **decision** (settled,
build to it) or a **risk** (§10, needs measurement not argument). Where a claim was verified
against real source or real repo metadata, the evidence is cited inline so a future reader can
re-check it instead of trusting it.

**Prime directive:** this app must not repeat the failure modes of its predecessor
(`ltx2.3-AIO-generator`). Those were not model failures. They were *packaging, pinning, and
silent-failure* failures. §5 and §8 are the sections the build will be judged on.

**A note on how to read this document.** Several statements here contradict things that are
true of *other* diffusion apps, or true of LTX-2.5 on a normal GPU box but false on ZeroGPU. The
biggest is §8: **`enable_model_cpu_offload()` must not be used.** If a future reader's instinct
says otherwise, §8.4 explains why the instinct is wrong here.

---

## 1. Decision 1 — Integration path: **Diffusers**, pinned to an exact SHA

We build on `diffusers`, loading `Lightricks/LTX-2.5-Diffusers`.

### 1.1 Options considered

| | Path | What it is | Verdict |
|---|---|---|---|
| **A** | `ltx-pipelines` | Lightricks' own repo, installed via `uv sync` of a git workspace | **Rejected** |
| **B** | ComfyUI runtime | Vendor ComfyUI + custom nodes, drive it via API graphs | **Rejected** |
| **C** | `diffusers` | `diffusers.pipelines.ltx2`, weights from `Lightricks/LTX-2.5-Diffusers` | **CHOSEN** |

### 1.2 Why C

**Path A is a heavy, fragile install for a Space.** It needs `uv sync` of a git workspace,
CUDA >= 12.7, torch ~= 2.7. A Space build that resolves a git workspace at image-build time is
a build we cannot reproduce or pin.

**Path B is the documented cause of the fragility we are told to avoid.** The 2.3 project's
`requirements.txt` vendors an entire ComfyUI runtime — ~50 dependencies, nearly all unpinned:
`spandrel`, `torchsde`, `kornia`, `PyOpenGL`, `glfw`, `SQLAlchemy`, `alembic`, `Mako`, `blake3`,
`comfyui-frontend-package`, `comfyui-workflow-templates`, `comfyui-embedded-docs`, plus a
hand-maintained block of *custom-node deps we discovered are required but aren't auto-installed
by their respective nodes* (`gguf`, `imageio_ffmpeg`, `opencv-python`, `matplotlib`,
`scikit-image`, `onnxruntime`, `yt-dlp`). That comment block is the evidence: the dependency
surface was discovered by breakage, not by design. Diffusers replaces all of it with ~16 pinned
direct deps and zero custom-node surface.

**Decisive coupling to §2.** The `*-comfy-int8-convrot` quantised weights are **ComfyUI-only**
— the model card says so. Choosing A or C means those 36.9 GB of int8 weights are unusable to
us. Path choice and disk budget are one decision, not two.

### 1.3 The pinned revision — and why a release tag will not do

```
diffusers @ git+https://github.com/huggingface/diffusers@ac56fa2f25c1c15b4cbaa1e0b3255e9bc38c06dc
```

HEAD at 2026-08-19T12:11:04Z. Contains the LTX-2.5 merge (PR #14447, 2026-08-11) plus #14453
and #14462. PEP 508 direct reference — **not** the legacy `#egg=` form.

**A SHA is more pinned than a version range, not less.** It is byte-for-byte reproducible and
cannot drift. It costs a git clone at build time and manual upgrades; both are acceptable.

**The latest release (0.39.0) does not have 2.5, and it fails LATE.** `LTX2Pipeline` imports
fine on 0.39.0 — so the app starts, the UI renders, weights begin downloading — and only then
does it fail on a missing `temporal_latent_upsampler`. A failure that happens after 70 GB of
download is the worst kind.

**Confirmed by the repo itself.** `model_index.json` on `Lightricks/LTX-2.5-Diffusers`
declares `_diffusers_version: 0.40.0.dev0`. The weights and the code agree on the dev version;
our SHA pin is not a guess.

**Therefore: a startup fail-fast assertion.**

```python
assert diffusers.__version__ == "0.40.0.dev0", (
    f"LTX-2.5 needs the pinned diffusers SHA (reports 0.40.0.dev0); got {diffusers.__version__}. "
    "See docs/PLAN.md §1.3."
)
```

This converts a late, confusing, expensive failure into an immediate, obvious one. It lives in
`preflight.py` (§7.1) and it is a test case (§9.3).

### 1.4 The single-file escape hatch: investigated and REJECTED

`diffusers/loaders/single_file_utils.py` does contain an LTX-2 route:
`CHECKPOINT_KEY_NAMES["ltx2"]` → `model_type = "ltx2-dev"` → config repo
`{"pretrained_model_name_or_path": "Lightricks/LTX-2"}`. Note the repo: **`LTX-2`, i.e. 2.0,
not 2.5**. The converters it ships are only `convert_ltx2_transformer_to_diffusers`,
`convert_ltx2_vae_to_diffusers` and `convert_ltx2_audio_vae_to_diffusers` — all keyed on
`LTX_2_0_*` rename dicts.

There is **no** single-file converter for 2.5's `connectors`, `diffusion_decoder`, `vocoder`,
`duration_head` or `latent_upsampler`. Even a successful load would give us a pipeline missing
the components that make 2.5 what it is. Settled; do not revisit without new upstream
converters.

---

## 2. Decision 2 — Disk, RAM and dtype budget

Three separate resource ceilings, and the tightest one is not the obvious one.

| resource | ceiling | our load | margin |
|---|---|---|---|
| **VRAM** (`xlarge`) | ~95 GiB | 67.2 GiB weights | ~28 GiB for activations — **comfortable** |
| **CPU RAM** | **~68 GB** (community cgroup observation, *not* an official figure — HF publishes ZeroGPU RAM as `"-"` / "Dynamic resources") | ~72 GB | **marginal — risk #1** |
| **Ephemeral disk** | 150 GB, **wiped on every restart/rebuild** | **152–156 GB peak** (§2.5) | **EXCEEDS THE CAP — settled, not a risk** |

**VRAM was never the problem.** The problems are CPU RAM and the startup disk peak — and the
second is now settled rather than uncertain: it does not fit (§2.5).

### 2.1 Trap A — duplicate shard sets

Two folders ship the *same weights twice*, in two different shardings:

| folder | set 1 | set 2 | note |
|---|---|---|---|
| `transformer/` | 4 shards: 9.932 + 9.919 + 9.986 + 8.138 = **37.975 GB** | 8 shards: 4.991 + 4.944 + 4.994 + 4.928 + 4.977 + 4.995 + 4.994 + 3.163 = **37.986 GB** | only **one** `diffusion_pytorch_model.safetensors.index.json` |
| `connectors/` | single file: **6.344 GB** | 2 shards: 5.000 + 1.344 = **6.344 GB** | same |

**Resolved against the actual index files — which set is live is now measured, not inferred:**

| folder | LIVE set | DEAD weight |
|---|---|---:|
| `transformer/` | the **4-shard** set — 37.976 GB | 37.986 GB |
| `connectors/` | the **2-shard** set — 6.345 GB | 6.344 GB |
| `text_encoder/` | all 5 shards — 23.920 GB | none |
| `transformer_full/` | 4 shards (excluded entirely, §2.2) | none |
| | | **total avoidable waste: 44.33 GB** |

Note the direction on `connectors/`: the **2-shard** set is live and the single 6.344 GB file is
the dead one — the opposite of the naive guess that a single file must be canonical.

A naive `snapshot_download` of those folders pays the full 44.33 GB. `from_pretrained` reads the
index JSON and fetches only the referenced shards.

**Binding consequences:** (1) we rely on the index, never hand-rolling a `snapshot_download` of
`transformer/`; (2) any `preload_from_hub:` entry must name **only the live shards**. A wrong
list here does not fail fast — it fails after tens of GB, and on a 150 GB cap it evicts us.

### 2.2 Trap B — `transformer_full/` is the DEV transformer

A further **37.975 GB**, the *dev* (non-distilled) transformer. We ship the distilled 8-step
path, so we do not ship it. Documented as opt-in (§2.5), not deleted from our mental model.

### 2.3 The bf16 subset we ship — **37 files, 72.185 GB (67.23 GiB)**

Measured from the Hub's own byte counts, not estimated. The `GB` column is decimal
(bytes / 1e9), matching how the Hub reports sizes.

| component | GB | loaded via | why |
|---|---:|---|---|
| `transformer/` | 37.9762 | index | the model (4 live shards + config + index) |
| `text_encoder/` | 23.9195 | index | `Gemma4UnifiedForConditionalGeneration` (§10 OQ-4) |
| `connectors/` | 6.3445 | index | text → transformer conditioning; also a LoRA target |
| `vae/` | 1.4522 | index | video VAE (conv-decoder fallback — §7.4) |
| `latent_upsampler/` | 0.9957 | **subfolder** | stage-2 spatial upscale |
| `diffusion_decoder/` | 0.8343 | index | **the native 2.5 decoder** — §7.4 |
| `temporal_latent_upsampler/` | 0.2619 | **subfolder** | stage-2 temporal upscale |
| `vocoder/` | 0.2583 | index | `LTX2VocoderWithBWE` — **not** plain `LTX2Vocoder` |
| `audio_vae/` | 0.1065 | index | audio VAE (mel ↔ latent) |
| `tokenizer/` | 0.0322 | index | `GemmaTokenizer` |
| `duration_head/` | 0.0038 | index | duration prediction |
| `scheduler/` | ~0 | index | `FlowMatchEulerDiscreteScheduler` -- config-only, but every pipeline constructor needs it, and it uses `scheduler_config.json`, not `config.json` |
| `model_index.json` (repo root) | ~0 | — | the component manifest |
| **TOTAL** | **72.185** | | **= 67.23 GiB, 44% of the repo — we skip 92.005 GB** |

**`model_index.json`, verbatim** — this is the authority for what the pipeline expects:

```
_class_name:       LTX2Pipeline
_diffusers_version: 0.40.0.dev0
audio_vae:         ['diffusers',    'AutoencoderKLLTX2Audio']
connectors:        ['ltx2',         'LTX2TextConnectors']
diffusion_decoder: ['diffusers',    'LTX2VideoDiffusionDecoderModel']
duration_head:     ['ltx2',         'LTX2DurationHead']
processor:         [None, None]
prompt_enhancer:   [None, None]
scheduler:         ['diffusers',    'FlowMatchEulerDiscreteScheduler']
text_encoder:      ['transformers', 'Gemma4UnifiedForConditionalGeneration']
tokenizer:         ['transformers', 'GemmaTokenizer']
transformer:       ['diffusers',    'LTX2VideoTransformer3DModel']
vae:               ['diffusers',    'AutoencoderKLLTX2Video']
vocoder:           ['ltx2',         'LTX2VocoderWithBWE']
```

**Two loading mechanisms, and `model_files.py` must model the difference.**
`latent_upsampler` and `temporal_latent_upsampler` are **not in `model_index.json`**. They are
loaded separately by subfolder, exactly as the model card does:

```python
LTX2LatentUpsamplerModel.from_pretrained(MODEL_ID, subfolder="latent_upsampler")
```

Index-components and subfolder-components are fetched by *different calls*. A resolver that
assumes one mechanism silently misses two components — and they are precisely the two the
stage-2 quality path needs.

### 2.4 What we exclude, and why

| excluded | GB | reason |
|---|---:|---|
| `transformer_full/` | 37.977 | the DEV transformer; not needed for the distilled 8-step path |
| duplicate shard sets (transformer 8-shard + connectors single-file) | 44.330 | Trap A — same weights, different sharding |
| `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (repo root) | 9.699 | the distilled LoRA for **DEV** workflows; useless when we ship the distilled transformer |

**Arithmetic check:** 72.185 + 37.977 + 44.330 + 9.699 = **164.191 GB**, closing against the
repo's stated 164.19 GB across 56 files (37 shipped, 19 skipped).

**Precision note, because it has already bitten once.** Summing figures that were each
truncated to three decimals gives 37.975 for the transformer; the exact byte count
(37,976,221,088 B) gives **37.9762**. The difference is 220 KB and irrelevant to the disk budget,
but it is larger than a ±0.001 assertion tolerance. `model_files.py` carries the byte-exact
values, so any test asserting on sizes should use a tolerance of ±0.01, not ±0.001. The accounting is complete; there is no
unexplained remainder.

### 2.5 The startup disk peak — negative margin, not tight margin

> **Authoritative section for this arithmetic.** The mechanism is §8.3; the
> consequence for the deploy is §7.4. If those and this ever disagree, this one
> is wrong — do not reconcile by editing the other two.
>
> *Provenance: verified against `spaces` 0.51.1 source (`zero/torch/patching.py`,
> `zero/configs.json`), not inferred.*

Startup holds **two** copies: the HF cache (~72.19 GB) *and* the packed copy ZeroGPU writes to
`~/.zerogpu/tensors` (~72.19 GB). The 2.3 project was evicted with exactly this:
`"Workload evicted, storage limit exceeded (150G)"`.

**The packer does auto-prune the HF blobs** — `ZEROGPU_MMAP_AUTOPRUNE_PATTERN` =
`~/.cache/huggingface/hub/models--*/blobs/*`, logging `"Cleaned NGB of tensor files ... after
packing"`. **That is verified, and it does not save us**, because the unlink runs *strictly
after* `_pack()` returns:

| | GB |
|---|---:|
| HF cache snapshot | 72.19 |
| ZeroGPU packed copy (**one** pipeline, post-`from_pipe`) | 72.19 |
| **peak, before the prune can run** | **144.37** |
| the image itself — torch + CUDA | **+8–12** |
| **total against a 150 GB cap** | **152–156** |

**The margin is negative, not tight.** An earlier draft of this section read it as
*"6 GB of margin is not much"* — that figure never counted the image, which is not optional and
is present for the whole window. The prune is real; it simply happens **after** the peak it
would have to prevent.

Steady state afterwards is fine at ~72 GB. **It is the peak that evicts us, not the residency**,
and the two have different fixes — which is why they are stated separately here.

§5.6's preflight measures free disk and reports a number, so an eviction arrives as a
diagnosable message rather than anonymously.

Remaining headroom after packing: `150 − 72.19 ≈ 77.8 GB` for the pip environment, user LoRA
uploads and outputs. `transformer_full/` (37.98 GB) fits in that — which is why it is opt-in
rather than impossible — but enabling it leaves ~40 GB and makes output eviction urgent.

### 2.6 dtype and quantization — **default bf16**, quantization is a documented fallback

Packed weight size decides which ZeroGPU tier we land on, and the tier decides quota cost. The
threshold is `xlarge_threshold = 42.00 GiB`: **exceed it and ZeroGPU silently auto-promotes to
`xlarge` and doubles quota cost, with no code change.**

| config | approx total | packed vs 42.00 GiB | tier | quota | verdict |
|---|---:|---|---|---|---|
| **bf16 everything** | 72.19 GB (67.2 GiB) | over | `xlarge` | **2×** | **DEFAULT** — documented by the model card, matches the official Lightricks Space pattern; RAM/disk marginal |
| fp8 transformer (~19) + bf16 TE (~24) + rest (~10) | ~53 GB (~49.3 GiB) | over | `xlarge` | 2× | comfortable RAM/disk, no quota win |
| fp8 transformer + int8 TE | ~41 GB (~38.2 GiB) | **under** | `large` | **1×** | best quota economics, highest accuracy risk |

**⚠️ The quantized checkpoints are not available to us at all — the blocker is FORMAT, not
size.** The int8 / nvfp4 files exist only in `Lightricks/LTX-2.5`, the **native** repo; there are
none in the Diffusers pack. And our single-file path cannot load them:
`CHECKPOINT_KEY_NAMES["ltx2"]` → `ltx2-dev` → config repo `Lightricks/LTX-2` (that is **2.0**),
with every converter named `LTX_2_0_*` and **nothing for 2.5's connectors, diffusion_decoder,
vocoder, duration_head or latent_upsampler** (§1.4). So no plan should be written around the
quantized files being reachable — they are a different repo in a format we have no converter for.

**Why bf16 is the default and not the quantised config.** Shipping quantised by default would
trade a *known* uncertainty for an *unverified* one:

- The RAM ceiling (~68 GB) is a community cgroup observation, not an official number. It is
  uncertain, but it is uncertainty about a configuration that upstream documents and that a
  production Space already runs.
- ~~Quantization carries its own unquantified risk: bitsandbytes and torchao use custom tensor
  subclasses and the packer's handling of them is unverified.~~ **This was wrong.**
  `spaces/zero/torch/bitsandbytes.py` is a 43-line version check that **no-ops for
  bnb ≥ 0.46.0**, wired into the packing path at `patching.py:328/355/445` — so modern
  bitsandbytes needs **no special handling at all**. There is **no equivalent module for
  torchao**, and that asymmetry is what actually selects the library. *Verified against
  `spaces` 0.51.1.* See `docs/ZEROGPU_DISK.md`.

So bf16 ships and `size="xlarge"` is **declared explicitly** rather than being silently promoted
(§8.5).

**⚠️ There is no precision switch, and the reason is a correction rather than a decision.**
`LTX25_TRANSFORMER_PRECISION` and `LTX25_TEXT_ENCODER_PRECISION` existed, resolved cleanly, and
were **wired to nothing** — `pipelines.get_pipeline` never read the config, so the load was
always bf16. Worse, `expected_weights_gb` applied the precision factor anyway, so the startup
report stated **53.20 GB for a load that was still 72.19 GB**. It did not fail silently: it
*spoke*, and the number was wrong in the direction that misdirects the next decision. It was one
approval away from being reached for to fix a disk eviction, twice.

**An advertised lever attached to nothing is worse than no lever at all**, so it was removed
rather than half-wired. Wiring it properly needs `bitsandbytes` or `torchao` — neither pinned,
neither verifiable without a GPU. **The reason originally given here — that the packer's handling
of quantised tensors was unknown — was false; see above.** The lever is gone because it was
**wired to nothing**, which is reason enough on its own. Quantization remains a *documented
option*; it
is not a *configured* one, and the difference is now visible in the code rather than implied.

The routes actually available under disk pressure are persistent storage, a public Space, or
smaller checkpoints — see §6.2.1.

**⚠️ Flipping precision is a REBUILD, not a runtime switch.** Preloading is all-or-nothing
(§6.2): preloaded files are owned by the build user and read-only to the runtime, so a quantised
variant needing *different files* cannot be fetched on demand into that cache. If a future
precision setting changes the file list, the Space must be rebuilt with a new `preload_from_hub`
block. Setting the variable alone changes how the existing weights are loaded -- nothing more. This is **technical risk #1** (§10), and it is to be settled by
measurement on the Space, not by argument.

### 2.7 Output retention — 20 GB, evicted oldest-first

`outputs/` is capped at **20 GB**, bounded-bytes, oldest-first eviction, as a single constant in
`config.py` overridable by env. Swept **on startup** (`app.py`) **and after every successful
generation** (`generate.py`), logging each eviction and surfacing it on the result's warnings.

Both call sites are deliberately **non-fatal**: the user's video exists and is fine, so a
retention failure must not turn a good run into an error, and a bad cap must not stop the app
from starting. A policy that exists but is never called prevents nothing -- which is precisely
what was true here until the call sites were added.

Rationale: ~78 GB headroom minus the pip environment and a user LoRA or two. 20 GB is many
generations and leaves the margin the 2.3 Space did not have when it filled its disk and went
unhealthy.

---

## 3. Decision 3 — Mode coverage: 11 UI modes over 4 real workflows

All 11 requested modes are reachable. **They are not 11 code paths**, and the plan says so
plainly — describing them as independent implementations would be dishonest and would produce a
much worse registry.

### 3.1 The four real workflows

From `_workflow_map` in `diffusers/modular_pipelines/ltx2/modular_blocks_ltx25.py`:

```
text2video   {prompt}
image2video  {image, prompt}
condition    {conditions, prompt}
in_context   {reference_conditions, num_frames, prompt}
```

Our **11 UI modes are a presentation layer over those four**, plus the audio-latent input route
and the audio-only export. That is the honest description, and it is also the design that keeps
the registry small: `modes.py` is data over four builders, not eleven.

### 3.2 Three facts that shape everything

**(a) One conditioning mechanism covers both image and video.**
`LTX2VideoCondition.frames` is typed
`PIL.Image.Image | list[PIL.Image.Image] | np.ndarray | torch.Tensor` — "the image (**or
video**) to condition the video on" — and carries `index` and `strength`. Image- and
video-conditioning are *the same code path* with a different payload. See §5.7 for the
`index` trap, which is severe.

**(b) The model is jointly audio-visual. There is no video-off switch.**
The return type is `LTX2PipelineOutput(frames=..., audio=...)`, both non-optional tensors. Every
call spends video compute and returns a video. This is the most important cost fact in the app.

**(c) Audio input is not a first-class kwarg.**
There is no `audio=` argument. Audio enters as `audio_latents` — upstream documents it as
*"Optional pre-encoded audio latents; random noise is used when not provided"* — combined with
`noise_scale`. `prepare_audio_latents` routes through `_create_noised_state`, whose body is
`noise_scale * noise + (1 - noise_scale) * latents`: img2img-style strength, where `1.0`
discards the input entirely.

### 3.3 The mode registry

| # | Mode | Workflow | How it is reached | Cost note |
|---|---|---|---|---|
| 1 | text → video | text2video | `prompt` | first-class |
| 2 | image → video | condition | one `LTX2VideoCondition(frames=PIL, index=0)`, empty prompt | first-class |
| 3 | image + text → video | condition | as #2 with a prompt | first-class |
| 4 | video → video | condition | `LTX2VideoCondition(frames=<video>, strength<1.0)` | first-class |
| 5 | text → audio+video | text2video | #1 — the audio track is always produced | **the default shape** |
| 6 | image → audio+video | condition | #2 | first-class |
| 7 | image + text → audio+video | condition | #3 | first-class |
| 8 | text → audio | text2video | #1, export only the audio track | ⚠️ **still pays full video compute** |
| 9 | video → audio | condition | #4, export only the audio track | ⚠️ same |
| 10 | audio → audio | condition | `audio_latents` + `noise_scale`, export only audio | ⚠️ same |
| 11 | audio → video | condition | `audio_latents` + `noise_scale`, export only video | ⚠️ pays audio encode on top |

**Mandatory mitigation for modes 8–11:** run at the **minimum legal resolution**, **skip the
stage-2 spatial upscale**, export only the requested track. The UI must say in words that
audio-only modes still render video internally — a user who does not know that will read the
runtime as a bug.

### 3.4 Multishot — **downgraded to honest**

There is **no multishot or scene API**. Multishot is prompt-structure driven only.

Worse: the 2.5 enhancer system prompts **explicitly instruct the model not to describe scene
cuts**, so prompt-level multishot actively fights the enhancer. We present it as *prompt-structure
driven, unverified, no API support* — a documented technique in the UI copy, not a feature with a
button that implies support we cannot back.

(Superseding an earlier draft of this plan that called it a first-class affordance. It is not.)

### 3.5 Prompt enhancement: **unavailable by default; we ship no dead toggle**

**`model_index.json` ships it as explicitly absent:** `prompt_enhancer: [None, None]` (and
`processor: [None, None]`). `LTX25AutoBlocks` declares the component and it appears in
`_optional_components`, but the pack has no `prompt_enhancer/` folder — all 56 files enumerated,
and the index itself declares the slot empty. It would come from `google/gemma-4-E2B-it`, a separate model. So
`enable_prompt_enhancement=True` fails on our subset.

Shipping a toggle that cannot work is exactly the silent lie §5 exists to prevent. No UI
control until the component is fetchable and wired.

Recorded for whoever enables it later: `prompt_max_new_tokens=600`, greedy decoding,
`prompt_enhancement_seed` is **inert under LTX-2.5's greedy decoding**, and the system prompt is
auto-selected between `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` and `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT`.

### 3.6 Duration prediction: ships

`duration_head` is in the pack (0.004 GB) and in `_optional_components`. Omit `num_frames`, pass
`min_seconds` / `max_seconds`. See §5.7 for the `num_frames` default trap and the >1-prompt
restriction.

---

## 4. Decision 4 — LoRA: the headline feature

User-uploadable LoRA at inference time is why this app exists. It must be the most robust thing
in it.

### 4.1 Verified mechanics

- All generation pipelines are declared
  `class LTX2…Pipeline(DiffusionPipeline, FromSingleFileMixin, LTX2LoraLoaderMixin)`.
- `LTX2LoraLoaderMixin._lora_loadable_modules = ["transformer", "connectors"]`. Note
  **`connectors`** — a LoRA may legitimately touch the text-conditioning path.
- **Both key conventions load.** `lora_state_dict()` detects
  `is_non_diffusers_format = any(k.startswith("diffusion_model.") for k in state_dict)` and runs
  `_convert_non_diffusers_ltx2_lora_to_diffusers(state_dict)`; separately it detects
  `has_connector = any(k.startswith("text_embedding_projection.") for k in state_dict)` and
  converts those into the connectors sub-dict.
- `lora_state_dict(..., return_lora_metadata=True)` returns the file's `__metadata__`.

### 4.2 The danger: upstream validation is almost non-existent

The **only** built-in check is a substring test:
`ValueError("Invalid LoRA checkpoint. Make sure all LoRA param names contain `'lora'` substring.")`

A LoRA for a *completely different architecture* passes that check — it has `lora` in its key
names — then attaches to **zero** modules. Generation proceeds. Output looks like the base
model. **No error is raised.** That is silent garbage, and it is precisely the class of failure
the operator has told us to eliminate.

We do have a real compatibility signal. Reading the official LoRA's safetensors header directly:
3320 tensors, all under prefix `diffusion_model.`, peft-style `lora_A.weight`/`lora_B.weight`,
targeting `to_q`/`to_k`/`to_v`/`to_gate_logits`, `adaln_single`, `audio_adaln_single`; and
`__metadata__ = {'lora_rank': '450', 'lora_alpha': '450', 'model_version': '2.5.0', 'license': …}`.
**`model_version` is machine-readable.**

### 4.3 The validator (9 stages — each stage is a test unit)

Accepts either a **Hub repo-id** or a **direct file upload**. Both routes converge on one parser.

| # | Stage | Rule | On failure |
|---|---|---|---|
| 1 | **Size bound** | reject above a configured max | refuse before any download or parse. On the Hub route this comes from the HTTP `Content-Length` — before a single weight byte moves |
| 2 | **Header-only parse** | read the safetensors header, **never** materialise the payload | — |
| 3 | **Format** | safetensors with a parseable header | distinct messages: not-safetensors vs unparseable-header |
| 4 | **Version** | read a *set* of signals in priority order (§4.5): `model_version`, `ss_ltx_version`, `modelspec.architecture`, `ss_base_model_version` | **warn** if all absent, **reject** only on explicit mismatch. Absent is the **common** case, not an edge case |
| 5 | **Convention** | classify: `diffusion_model.` / `transformer.` / `text_embedding_projection.` / unknown | on unknown, the error **names the prefixes actually found** |
| **5.5** | **IC-LoRA detection** | an in-context LoRA is marked by `__metadata__.reference_downscale_factor` | **refuse with a specific explanation.** An IC-LoRA applied to the plain pipeline neither errors nor works — see below |
| 6 | **Pairing** | `lora_A`/`lora_B` (or `lora_down`/`lora_up`) must pair | name the unpaired keys |
| **6.5** | **Per-module dimension compatibility** | resolve each pair to its **target module in the real transformer** and compare against *that module's* `in_features`/`out_features` | reject naming **module, expected, actual**. A key whose target module does not exist at all is a **distinct** failure — that is what a genuinely foreign adapter looks like |
| 7 | **Attach assertion** | **after** `load_lora_weights`, assert the adapter attached to **> 0** modules | specific error if zero — the anti-silent-garbage check |

Stage 2 is a hard rule: a multi-GB upload must never be fully materialised just to discover it
is the wrong architecture.

#### Stage 7 is necessary but NOT sufficient — do not delete 6.5 as redundant

The case that proves it: `charlied98/ltxv_2b_lora_soft_draw_v1`, an LTX-**Video-2B** adapter, i.e.
flatly the wrong architecture. Header measured:

- prefix `diffusion_model.` → **passes stage 5**
- peft `lora_A`/`lora_B` pairs → **passes stage 6**
- every key contains `lora` → **passes diffusers' one and only built-in check**
- `__metadata__` empty → **passes stage 4** under our own (correct) absent-is-not-fatal rule
- module names are `transformer_blocks.N.attn1.to_k` — **the same names LTX-2.5 uses**, so it may
  attach to > 0 modules and **pass stage 7 as well**

The only thing wrong with it is that `to_k.lora_A` is `[32, 2048]` where this model is 4096.
**Stage 6.5 is the only gate that catches it.**

#### Stage 6.5 compares PER MODULE — there is no single "expected hidden dim"

The transformer carries **two streams**, and a naive "everything must be 4096" rule would reject
the most-downloaded LTX LoRA on the Hub. Measured from `joyfox/LTX-2.3-Transition-LORA`:

| module family | in-dim |
|---|---:|
| `attn1.to_q/k/v/out.0`, `attn2.…` | **4096** (video stream) |
| `ff.net.N.proj` | 4096 |
| `ff.net.2` | **16384** (4× feed-forward expansion) |
| `audio_ff.net.N.proj` | **2048** (audio stream) |
| `audio_ff.net.2` | **8192** |

Corroborated by the official 2.5 adapter, which pairs `adaln_single…linear_1.lora_B [4096, 256]`
with `audio_adaln_single…linear_1.lora_B [2048, 256]`.

So **"contains 2048" is not a signal. "2048 at a module that expects 4096" is.** Build the
expectation map by walking the loaded transformer's `nn.Linear` modules — never from a hardcoded
table — so it stays correct across checkpoints. The derived
`num_attention_heads × attention_head_dim = 32 × 128 = 4096` is one entry in that map, not the
rule.

#### 5.7.1 The anchor is a frame, and the conversion is lossy

**Ruling: the user types a pixel frame. `conditioning.py` converts frame → latent.** Users think
in frames — "put my second shot at frame 24". A latent index is an artefact of the VAE's 8:1
temporal compression and means nothing to someone generating a video. This keeps the conversion
in exactly one place, which is what this row assigns to `conditioning.py`, and it *handles* the
trap rather than explaining it: labelling the field LATENT would make the off-by-8 the user's
problem.

**The lossy direction must not be silent.** Frame → latent is 8:1, so a run of frames collapses
onto one anchor. The readout says so rather than snapping quietly:

```
| shot | anchor (frame) | lands on latent | frames sharing it |
| 2    | `25`           | **4**           | 25–32             |
| 3    | `30`           | **4**           | 25–32 ⟵           |

⟵ Temporal compression is 8:1, so these anchors snap. Frame `30` anchors at the
  same place as frame `25` — frames 25–32 all share latent **4**.
```

`conditioning.latent_group_for()` owns that arithmetic; the UI words the answer and computes none
of it. A second copy in the render layer would eventually disagree with the real one, and **a
readout that disagrees with what runs is worse than no readout**.

*Historical note:* the two halves of this once disagreed — the column was labelled
"Anchor (LATENT index)" while `normalise_condition` treated it as a pixel frame — because an
instruction asked for both at once. Wiring them naively would have double-converted every anchor.

#### Stage 5.5 — why IC-LoRAs must be refused rather than loaded

`DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer` carries
`__metadata__ = {'reference_downscale_factor': '1'}`. That maps to the `reference_downscale_factor`
kwarg on `LTX2InContextPipeline`, which upstream documents as needing to match the factor the
IC-LoRA was trained with. Applied to the plain pipeline an IC-LoRA **will not error and will not
work** — the exact silent-garbage shape stage 7 exists to prevent, arriving through a door stage 7
does not watch.

### 4.4 Fusing — the connectors trap

`fuse_lora` / `unfuse_lora` default to `components=["transformer"]` **only**.

**Connector LoRA layers are silently not fused** unless we pass
`components=["transformer", "connectors"]`. A LoRA that trains the text-conditioning path would
partially apply — worse than not applying, because it looks like it worked. We always pass both
components explicitly, and this is a test case.

### 4.5 What real LoRAs actually look like — measured, and it changes stage 4

**Technique note, since this document cites measured headers in several places:** every such
measurement was taken by **HTTP range-reading the safetensors header** — `Range: bytes=0-7` for
the 8-byte little-endian header length, then `Range: bytes=8-(8+N-1)` for the JSON — with no full
download. It reads a 9 GB adapter's complete key list in well under a second. This is the same
technique `lora/inspect.py` uses for the Hub route (§4.6).

I range-read the headers of three real public LTX LoRAs:

| LoRA | size | tensors | prefix | `__metadata__` | `model_version`? | connectors keys |
|---|---:|---:|---|---|---|---:|
| `joyfox/LTX-2.3-Transition-LORA` (19.4k downloads) | 0.390 GB | 1152 | `diffusion_model.` | **ABSENT** | — | 0 |
| `vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa` | 0.705 GB | 1152 | `diffusion_model.` | present, **sd-scripts style** | **no such key** | 0 |
| `elix3r/LTX-2.3-22b-AV-LoRA-talking-head` | 0.428 GB | 2304 | `diffusion_model.` | **ABSENT** | — | 0 |

**Four findings, each of which changes the design:**

1. **Not one of them carries `model_version`.** Stage 4 as originally written — "read
   `__metadata__.model_version`" — would be warn-only for **100% of real community uploads**,
   i.e. effectively dead code. The official Lightricks LoRA does carry it (`'2.5.0'`), so the
   field is real -- but **the official adapter is the exception, not the template.** A validator
   written from it as the reference would have been written from the one file in the population
   that is unrepresentative of the rest.

   **The absent case has two distinct shapes and the code must handle both:** no `__metadata__`
   key at all, *and* `__metadata__: {}` present but empty. Both mean "no version signal". Since
   this is the **common** path rather than an edge case, stage 4's warn-not-reject branch must not
   be "tightened" later by someone who assumes metadata is normally present.

2. **There is a second, richer convention we must read.** The sd-scripts / musubi-tuner trainer
   writes a large metadata block with entirely different keys — from the measured header:
   `ss_ltx_version: "2.3"`, `ss_base_model_version: "ltx2_v1"`,
   `modelspec.architecture: "LTX2/lora"`, `ss_network_module: "networks.lora_ltx2"`,
   `ss_sd_model_name: "ltx-2.3-22b-dev.safetensors"`, `ss_network_dim: "64"`,
   `ss_network_alpha: "32"`, `ss_lora_target_preset: "full"`.

   **`modelspec.architecture == "LTX2/lora"` is a better *architecture* signal than
   `model_version` is** — a Flux or Wan LoRA carries a different value there, which is exactly
   the wrong-architecture case stage 7 exists to catch. Reading it lets us catch that at stage 4
   instead, cheaply, before any weights load.

   **Revised stage 4** reads, in priority order: `model_version` → `ss_ltx_version` →
   `modelspec.architecture` → `ss_base_model_version`. All absent → **warn and proceed**;
   rejecting on absence would reject the most popular LTX LoRA on the Hub.

3. **`diffusion_model.` is the dominant real-world convention, not an edge case.** All three use
   it, so `_convert_non_diffusers_ltx2_lora_to_diffusers` is on the *main* path. The
   diffusers-native `transformer.` convention is the rarer one.

4. **None of them touch the connectors** (`text_embedding_projection.` count is 0 across all
   three). So in practice stage 7 will see transformer-only attachment. §4.4's both-components
   fusing rule stays — it must be right for the LoRA that does touch them — but we should not
   expect connector attachment as the normal case, and a stage-7 assertion demanding it would
   reject every real LoRA.

These three headers become **offline test fixtures** for stages 1–6.5 (§9.1): real-world data,
zero GPU, zero weights, no network at test time.

**The convention detector is now backed by real inputs in both conventions.**
Until the H13 work it had only ever seen fixtures *we* wrote, shaped by our own
reading of the format. `charlied98/ltxv_2b_lora_soft_draw_v1` ships the same
adapter twice — `lora_weights_step_03000.safetensors` under `transformer.`
(diffusers) and `comfy_lora_weights_step_03000.safetensors` under
`diffusion_model.` (native) — and running the real validator against both gives:

```
transformer.       (diffusers)  ->  rejected at stage 6.5, LoRADimensionError
diffusion_model.   (native)     ->  rejected at stage 6.5, LoRADimensionError
```

**Both converge on the correct stage.** The detector accepted two conventions
neither of us invented and routed each to the dimension check — which retires the
worry that the two paths might diverge, or that one might not reject at all.

It also fixes what "rejected" means as a result: **a run reporting "rejected"
without naming `LoRADimensionError` has not distinguished success from the
convention detector breaking.** Two mechanisms could produce the same verdict, and
only one of them is the validator working.

### 4.6 `lora/inspect.py` has ZERO heavy dependencies — and that buys a real feature

The safetensors container is trivial:

```
[8 bytes little-endian uint64 = header length N][N bytes of JSON header][raw tensor bytes]
```

`struct` + `json` from the stdlib read `__metadata__`, every key name, every dtype and every
shape. No torch, no diffusers, no `safetensors` package. Three reasons, in order:

1. **A bad upload must be rejectable without loading torch.** A validator that needs a 2 GB
   import to say "this is not a safetensors file" is the wrong shape.
2. **It keeps the gate fast and portable.** Stages 1–6.5 become offline, torch-free unit tests
   that run in milliseconds, with fixtures hand-constructed from `struct` + `json`. Only stage 7
   needs real weights, and that is `gpu`-marked.
3. **It lets us validate a Hub LoRA *before* downloading it.** The header sits at the very start
   of the file, so one range request validates the whole thing.

**Hub route:** resolve the file → **range-read the header** → run stages 1–6.5 → only then
download. A user who pastes a repo-id for a 2B-architecture adapter gets a specific rejection in
about a second instead of after a 300 MB download. Stage 1 comes free from `Content-Length`.

**Upload route:** the bytes are already local, so the same parser runs against the file. Both
routes converge on one code path, which is what §4.3 asks for.

`validate.py` stays dependency-light too. **`apply.py` is the only module in `lora/` that may
import diffusers or peft**, because it is the only one that touches a real model.

---

## 5. Decision 5 — Resilience doctrine

Binding on all code in the repo.

### 5.1 Pin every dependency, with upper bounds — and know what NOT to pin

Evidence from the predecessor: `spaces>=0.30.0` and `torch>=2.4.0` — floors with **no ceiling**
— alongside exactly one hard pin, `torchaudio==2.10.0`, carrying this comment:

> *"HF Spaces ships torch==2.10.0; bare 'torchaudio' lets pip pick an older wheel whose C++ ABI
> doesn't match → import crashes the whole audio code path… Pin the matching version."*

That comment is a post-mortem, applied to one line while ~50 stayed unbounded.

**But the inverse mistake is also real, and the 2.3 repo made it.** Its own `AGENTS.md:19` says
**do not pin `spaces`** — the platform pins its own version (live Spaces report 0.49.3 / 0.50.0 /
0.50.4 while PyPI is 0.51.1) and a conflicting pin breaks pip resolution at build time. Its
`requirements.txt:2` then pins `spaces>=0.30.0` anyway. We follow the rule the repo wrote and
did not keep: **no `spaces` line in `requirements.txt`**, and the import is guarded in code
(`try: import spaces / except ImportError: spaces = None`) so local dev works.

The full pin set:

| package | pin | why |
|---|---|---|
| `torch` | `==2.11.0` | ZeroGPU supports exactly 2.8.0 / 2.9.1 / 2.10.0 / **2.11.0**; ceiling is 2.11.0. CUDA 12.8, sm_120 needs cu128+ |
| `torchvision` | `==0.26.0` | **trap:** `torchvision 0.28.0` hard-declares `torch==2.13.0`, so a floating torchvision silently upgrades torch past the ceiling |
| `torchaudio` | `==2.11.0` | must match torch exactly (the ABI break above) |
| `gradio` | `>=6.24.0,<7` | **5.x is END OF LIFE** (last stable 5.50.0, 2025-11-21) |
| `diffusers` | `@ git+…@ac56fa2f…` | §1.3 |
| `transformers` | `>=5.15.1,<6` | |
| `accelerate` | `>=1.14.0,<2` | |
| `peft` | `>=0.20.0,<0.21` | LoRA backend |
| `safetensors` | `>=0.8.0,<0.9` | |
| `huggingface_hub` | `>=1.28.0,<2` | |
| `hf_xet` | `>=1.6.0,<2` | |
| `imageio` | `>=2.37.4,<3` | |
| `imageio-ffmpeg` | `>=0.6.0,<0.7` | |
| `av` | `>=18.1.0,<19` | PyAV — required by CRF re-compression (§5.7) |
| `numpy` | `>=2.5.2,<3` | **numpy 2.** Do NOT pin `<2`: 1.26.4 is 2 years EOL and nothing in the stack caps it |
| `pillow` | `>=12.3.0,<13` | |
| `spaces` | **absent, deliberately** | platform-pinned; see above |

**gradio 6 breaking changes that hit us:** `gr.Video` subtitle-tuple return removed;
`show_download_button` / `show_share_button` → `buttons=[...]`; `gr.Audio` `show_controls` →
`show_recording_waveform`; `min_length` / `max_length` → event-listener validators.

**Python 3.12 is required, not preferred:** numpy 2.5.2 needs ≥3.12 and `av` 18 needs ≥3.11. The
default 3.10.13 image silently resolves both *downward* rather than failing.

### 5.2 One central model-file indirection

Every Hub repo-id, revision, subfolder, filename **and dtype/quantization choice** (§2.6) lives
in **one** module — `ltx25_studio/model_files.py` — as data, never inlined at a call site.

A missing file must raise an error naming **both the file and its URL**. Not hypothetical:
Lightricks deleted a 2.3 checkpoint from the Hub with no warning and broke a live Space. When it
happens again, the message should tell the operator which URL to go look at.

Pinning a **revision** per repo is part of this module's job — a repo whose content can change
under us is not pinned just because its name is stable.

**New requirement: `model_files.py` emits the download allow-list as data.** It must be able to
produce the exact list of **37 paths** (§2.3), so that:

- the **runtime download** and the Space README **`preload_from_hub:`** list are *generated from
  the same source*, never hand-maintained in two places; and
- a **drift test** can assert the committed README list equals what `model_files.py` declares.

**And the live shard names must come from reading the index files, not from a hardcoded list.**
That is what makes an upstream re-shard get picked up automatically instead of silently
downloading the dead set.

**But the index alone is not enough, and this bug was made for real.** A first attempt at
generating the list produced **33** files: for the three *sharded* components it took only the
shards named in the weight map plus the index, and so silently dropped `transformer/config.json`,
`text_encoder/config.json`, `connectors/config.json` and `text_encoder/generation_config.json`.
`from_pretrained` cannot construct a component without its config, so a Space preloaded from that
list would have failed at startup **after a 72 GB download**.

The weight-map index tells you which *weights* are live. It says nothing about sibling config
files. So the rule is per component:

- **sharded** (`transformer`, `text_encoder`, `connectors`): the shards named in the index, plus
  the index itself, plus **every non-`.safetensors` file in the folder**
- **whole-file** (everything else): every file in the folder
- plus root `model_index.json`

And the emitter **asserts, as a self-check, that every component contributes a config file** --
`config.json` for most, `scheduler_config.json` for `scheduler`, `tokenizer_config.json` for
`tokenizer`. That assertion is what would have caught the mistake, and it is cheap. Hardcoding today's shard names would reintroduce Trap A (§2.1) the
moment Lightricks re-shards — which, given they have already deleted a checkpoint without
warning, is not a remote possibility.

This is the single-source-of-truth principle (§5.2) applied to deployment, and it is the thing
that stops the shard-set trap from silently reappearing.

### 5.3 Never return an empty output — and the four layers that let it happen

Any generation that completes without producing a file **must** raise a specific, actionable
error that the UI renders visibly.

**The full anatomy of the 2.3 silent reset**, because each layer was individually "reasonable"
and that is the lesson:

1. `models.ensure_models` hit a missing file and did `logger.warning(...) + continue`.
2. ComfyUI's executor caught every node exception and returned a FAILURE **tuple**, without
   raising.
3. An `OutputEvent(video_path="")` was pushed **unconditionally**.
4. The UI did `event.video_path if event.video_path else gr.update()` — rendering an empty path
   as a silent form reset.

The user saw "nothing happened". No error, no log, no signal. It hid for a **week**.

**The `exports` tuple is a contract, checked in BOTH directions** (`generate._artifacts_for`):

* **declared but missing → raises.** A mode promising a track and producing nothing is the
  never-empty bug one level up.
* **written but undeclared → dropped and noted, not raised.** The registry decides a mode's
  exports, so an over-generous encoder is over-generous rather than wrong; failing a run that
  produced a perfectly good video is the harsher answer to the milder problem. The extra track
  never reaches the result, and the drop is surfaced on `GenerationResult.warnings`.

Enforcing only the first direction is what let a mode declare `exports=("audio","video")` and
return `audio_path=None`, so the UI reported every successful joint-mode run as an internal
error. One-directional contracts have now failed here twice; the direction that is *not* checked
is the one that bites.

**Our rules, both halves mandatory:**
- **An empty path becomes an error AT THE SOURCE**, and every layer propagates it. A success
  path that can return a falsy artifact is a bug regardless of current reachability.
- **No "newest file in the output directory" fallback.** The 2.3 one had no run correlation and
  would show the *previous* run's video for a failed run — a worse failure than showing nothing.
  **Each run is stamped into its own output subdirectory**, and a run's artifact is only ever
  looked up inside that run's directory.

### 5.4 Read parameters from config, never hardcode

Mel/audio parameters come from `audio_vae.config` at runtime: `audio_sampling_rate`,
`audio_hop_length`, `audio_mel_bins`, `audio_vae_mel_compression_ratio`,
`audio_vae_temporal_compression_ratio`, `audio_latent_channels`.

This mirrors the pipeline itself, whose constructor reads `self.vae.spatial_compression_ratio`,
`self.vae.temporal_compression_ratio`, `self.audio_vae.mel_compression_ratio`,
`self.audio_vae.temporal_compression_ratio` and `self.transformer.config.patch_size` from the
loaded components, using literals only as fallbacks. Same discipline. §5.7 extends this to CRF.

### 5.5 Enforce and surface the hard constraints

| constraint | source | handling |
|---|---|---|
| `height % 32 == 0`, `width % 32 == 0` | enforced upstream (raises) in all generation pipelines *and* the latent-upsample pipeline | validate before the call, snap in the UI, **show** the snapped value |
| `num_frames % 8 == 1` | **not** hard-raised upstream. Falls out of `latent_num_frames = (num_frames - 1) // vae.temporal_compression_ratio + 1`, which **silently floors** a bad value | we raise. **The `8` is `vae.temporal_compression_ratio`** — see below |
| audio/video length agreement | audio and video lengths are computed independently, and a supplied `audio_latents` **overrides** the duration-derived audio length (§5.7) | validate they agree; raise rather than emit a desynced file |

**The `num_frames` modulus is a parameter, not a literal.** Hardcoding `8` would violate §5.4.
Resolution:

- `constraints.py` takes the ratio as a **parameter with a documented fallback of `8`**, because
  the UI must validate input *before* any VAE exists.
- `pipelines.py` **asserts the loaded `vae.temporal_compression_ratio` equals that fallback** at
  load time, raising loudly if upstream ever changes it.

That gives a working pre-load validator *and* a tripwire. Both are test cases.

**Verified constants** (`diffusers/pipelines/ltx2/utils.py`):

```
DISTILLED_SIGMA_VALUES         = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]   # 8 steps, CFG=1
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875]                                          # 3 steps
```

Re-exported and asserted in `constraints.py` against the installed diffusers, so a schedule
change upstream fails a test instead of silently changing output quality.

### 5.6 Preflight — fail loudly on resources, before the platform kills us

At startup, `preflight.py` measures **free RAM and free disk** and compares them against the
requirement computed from the `model_files.py` table. If short, it raises a specific error
naming the numbers:

> `need ~67.2 GiB RAM for the bf16 load, saw 41.3 GiB` (see §2.6 for the quantised fallbacks)

This is the fail-loudly principle applied to the resource budget. It turns an anonymous OOM or a
`"Workload evicted, storage limit exceeded (150G)"` into a diagnosable message that names the
fix. It also runs the §1.3 diffusers-version assertion. Test case.

**The startup report** (`pipelines.startup_report`) prints, before anything expensive:

```
LTX-2.5 Studio starting
  config          : transformer=bf16 text_encoder=bf16 zerogpu_size=auto duration=60s
                    expected_weights=72.19GB outputs<=20GB
  weights cached  : 36/37 files  MISSING 1
  disk            : 412.5 GB free of 1995.2 GB
  zerogpu size    : not yet known (reported after packing)
  model revision  : 871165de037793df7d75c23a02dd7f45fd364c97
  first missing   : vae/config.json
```

The **cache count is the load-bearing line**: `36/37`, with the first missing file named, makes
"preload worked" and "preload silently did nothing" two different sentences. When a GPU run fails
on a metered Space behind a 72 GB build, these lines are the difference between diagnosing it in
one round trip and burning several -- and there is no opportunity to add a print statement
afterwards.

### 5.7 API gotchas that cause silent wrongness

Every row here is a **test case**, not a note. These are the ways this API is wrong-by-default.

| # | Gotcha | Consequence | Our handling |
|---|---|---|---|
| 1 | **`LTX2VideoCondition.index` is a LATENT index, not a pixel frame index.** `start_idx = max((latent_start_idx - 1) * vae_temporal_compression_ratio + 1, 0)` | **off-by-8 guaranteed** if a pixel frame number is passed | convert at the boundary in `conditioning.py`. **The user types a FRAME** (§5.7.1); the pipeline gets a latent index |
| 2 | **Out-of-range condition indices are silently SKIPPED** with only a `logger.warning` | a whole conditioning image does nothing, silently | validate range ourselves and **raise**; this is a silent-failure path we convert into a loud one |
| 3 | **`noise_scale` default differs by class:** `0.0` on `LTX2Pipeline` / `LTX2ImageToVideoPipeline`, `None` on `LTX2ConditionPipeline` (resolving to `sigmas[0]` or 1.0) | copying a call between classes changes behaviour silently | always pass `noise_scale` explicitly; never rely on the default |
| 4 | **`num_frames` default differs:** `None` (auto-duration) on `LTX2Pipeline`/`ImageToVideo`/`Condition`; hard `121` on `LTX2InContextPipeline` and `LTX2HDRPipeline`, which have **no `duration_head` at all** | auto-duration silently unavailable on two pipelines | registry records per-workflow duration support; auto-duration also **raises** with >1 prompt and `num_frames=None` — validate before the call |
| 5 | `latents` must be **UNPACKED 5D** `[B,C,F,H,W]`; `audio_latents` **UNPACKED 4D** `[B,C,L,M]` | shape errors, or worse, silent misinterpretation | assert shapes at our boundary |
| 6 | **`output_type="latent"` returns ALREADY-DENORMALIZED latents** | re-feeding them re-normalizes → corrupted output | drives the mandatory `denormalize=False` in §7.4 |
| 7 | **Image-conditioning CRF re-compression is applied BY DEFAULT**, needs PyAV + uint8/PIL input; a float tensor **raises** unless `crf=0` | crashes on float input | normalise to uint8/PIL in `conditioning.py`; `av` is a pinned dep for this reason |
| 8 | **CRF is 18 when `text_encoder.config.model_type in ("gemma4_unified", "gemma4")`, else 33** | hardcoding either number is wrong for the other | **derive from `text_encoder.config.model_type` at runtime** (§5.4) |
| 9 | `from diffusers.pipelines.ltx2.export_utils import encode_video` is **DEPRECATED**, removal in 0.40.0 — and our pin *is* 0.40.0.dev0 | breaks on the very revision we pin | use `from diffusers.utils import encode_video` |
| 10 | `spatio_temporal_guidance_blocks` defaults to `[28]` for 2.3/2.5 (`[29]` was 2.0) and is a **mutable shared default** | in-place mutation corrupts every later call in the process | never mutate in place; copy before modifying |
| 11 | A supplied `audio_latents` **overrides** the duration-derived `audio_num_frames`: upstream logs `"Got audio_latents of shape […], audio_num_frames will be inferred"` then does `_, _, audio_num_frames, _ = audio_latents.shape` | supplying audio **silently redefines audio length** while `num_frames` still governs video → desync | §5.5 validation, with a concrete desync test case |
| 12 | `fuse_lora`/`unfuse_lora` default `components=["transformer"]` | connector LoRA layers silently unfused | §4.4 — always pass both |

Reference: `_optional_components = ["processor", "prompt_enhancer", "duration_head"]` (plus
`audio_scheduler` on Condition), and
`model_cpu_offload_seq = "prompt_enhancer->text_encoder->connectors->duration_head->transformer->vae->audio_vae->vocoder"`.
Recorded for completeness — **but see §8.4: we do not use accelerate's offload at all.**

### 5.8 Operational lessons from the 2.3 post-mortem

| # | Lesson | Rule |
|---|---|---|
| 1 | **ZeroGPU quota attribution rides a contextvar** (`gradio.context.LocalContext.request`). It does **not** cross a plain `threading.Thread` — without it every user is bucketed as Space-anonymous and you get *"Space app has reached its GPU limit"* at **0 minutes used** | any thread we spawn must carry `contextvars.copy_context()`; better, do not spawn plain threads across the GPU boundary |
| 2 | The `duration=` callable must accept **the exact same positional signature** as the decorated function, trailing kwargs included | otherwise TypeError at schedule time; test the signature match |
| 3 | **Hidden 1.5× duration multiplier on `large`** (`duration_factor`): `duration=60` is scheduled as 90 s. On `xlarge` the factor is skipped but the request is **doubled** instead | never reason about wall-clock quota from the `duration=` number alone |
| 4 | Default duration is 60 s. The **max is a server-side, tier-dependent, unpublished cap** | **discover it, never hardcode it.** Empirically ~120 s is the safe ceiling; the official Lightricks LTX Space ships `@spaces.GPU(duration=75, size="xlarge")` |
| 5 | **Do NOT auto-retry a ZeroGPU timeout** using the captured Gradio request: the `X-IP-Token` ages past TTL during attempt 1, so attempt 2 always 401s with *"Expired ZeroGPU proxy token"* — **masquerading as the root cause** | surface the original timeout; never retry behind the user's back |
| 6 | `preload_from_hub` is **capped at 10 entries**, and preloaded files are owned by the *build* user and **read-only to runtime uid 1000** — a later lazy `hf_hub_download` fails with `Permission denied (os error 13)` unless the cache is mirrored | §6.2 |
| 7 | Accumulated generations filled the 2.3 Space's disk and it went unhealthy | §2.7 — sweep on startup and after each run |

---

## 6. Decision 6 — Deploy shape

| | |
|---|---|
| **Space** | `techfreakworm/LTX2.5-Studio`, **private** |
| **SDK** | `sdk: gradio`, `sdk_version: 6.24.0` — **a custom Dockerfile forfeits ZeroGPU entirely** |
| **Python** | `python_version: "3.12"` — required, not optional (§5.1) |
| **Hardware** | ZeroGPU. `suggested_hardware: zero-a10g` is the **legacy id string the Spaces API still reports — it is NOT an A10G** (§8.1) |
| **Size** | `size="xlarge"` declared explicitly (§8.5) |
| **Weights** | README frontmatter `preload_from_hub:` |
| **Secret** | `HF_TOKEN` — required; `Lightricks/LTX-2.5-Diffusers` is gated |
| **Git** | GitHub `git@github.com:techfreakworm/ltx2.5-studio.git`; Space and GitHub stay in sync |

Naming precedent: `techfreakworm/LTX2.3-Studio`, `wan-studio` (private), `SAM3-cutout-studio`
(private).

### 6.2 Download strategy — REVISED: runtime download, not `preload_from_hub`

**`preload_from_hub` cannot work for a PRIVATE Space, and this was found the
expensive way — by a `RUNTIME_ERROR` after a successful build.**

```
Initialization step 'preload' failed with exit code: 1
GatedRepoError: 401 Client Error
Cannot access gated repo .../LTX-2.5-Diffusers/.../audio_vae/config.json
```

**401 is unauthenticated, not unauthorised.** The preload runs as a separate,
platform-managed init step that does not receive Space secrets, so the request
arrives with no credentials at all. Verified both directions:

| Space | privacy | repo gate | preload |
|---|---|---|---|
| `techfreakworm/LTX2.3-Studio` | **public** | `gated=manual` (stricter) | works — RUNNING |
| `techfreakworm/LTX2.5-Studio` | **private** | `gated=auto` (looser) | **401** |

Privacy is the differentiating variable — not the gate, not the token. The same
URL at the same revision fetches 200 by hand.

**So the 37 files are downloaded at runtime by `preflight.ensure_weights()`**,
inside the app process, where the token exists. Same
`model_files.download_allow_list()`, so the single-source property is unchanged.

**This is the better design even if the diagnosis above is wrong**, which is why
it was chosen over simply making the Space public: it **removes the ownership bug
class entirely**. Preloaded files belong to the *build* user and are read-only to
runtime uid 1000, so any later lazy fetch fails `Permission denied (os error 13)`
— the 2.3 failure. When the runtime user downloads its own cache, it owns it, and
that class cannot occur. A fix that holds even if the root-cause analysis is
wrong is worth more than one that depends on it.

#### 6.2.1 The cost: a 72 GB refetch on every cold start

Measured on the live Space:

```
storage   : None
gcTimeout : 172800 s = 48 hours
```

There is **no persistent storage**, so the full **72.186 GB** is refetched on
every cold start — restart, rebuild, or 48 hours idle. `startup_duration_timeout:
1h` covers it. **This is accepted for the development phase**, not endorsed.

Two routes back to preloading, either of which restores it:

1. **Make the Space public** — the plan, once every mode is verified.
2. **Attach persistent storage** — the cache survives, so the refetch stops
   mattering.

The `preload_from_hub` block is therefore **kept in `README.md`, commented**,
with its re-enable conditions beside it and a drift test asserting it still
matches `model_files.preload_from_hub_entry()`. A commented block drifts *more*
readily than a live one — nothing exercises it, so nothing ever points at it, and
it is trusted on exactly the day it is re-enabled.

### 6.2.2 Original strategy, superseded

**`preload_from_hub:` with the file list generated from `model_files.py` (§5.2), plus a runtime
fallback that fetches anything missing.**

Preloading bakes the weights into the image so a restart does not re-fetch 72 GB — which matters
because ZeroGPU's ephemeral disk is **wiped on every restart/rebuild** (§2). The runtime fallback
exists because a preload that silently drops a file must not become a mysterious load failure.

Three hazards to code around:

1. **Only live shards** (§2.1 Trap A). A wrong list fails after tens of GB, not fast. Mitigated
   structurally: the list is *generated*, and a drift test asserts README == `model_files.py`.
2. **Capped at 10 entries.** Not a problem — we are **one repo, so one entry**, with
   comma-separated filenames. It fits in ~1.5 KB. (A naive one-file-per-entry list would not: 37 files.)
3. **Ownership.** Preloaded files belong to the *build* user and are read-only to runtime uid
   1000, so a later lazy `hf_hub_download` into the same cache fails with
   `Permission denied (os error 13)`.

   **⚠️ DO NOT move `HF_HOME` to dodge this.** An earlier draft of this plan said to point
   `HF_HOME` / `HF_HUB_CACHE` at a writable location. That advice was wrong and would have caused
   the exact failure it was meant to prevent. The Spaces config reference warns, verbatim, on
   `preload_from_hub`:

   > *"Files are saved in the default `huggingface_hub` disk cache `~/.cache/huggingface/hub`. If
   > your application expects them elsewhere or **you changed your `HF_HOME` variable, this
   > preloading does not follow that at this time**."*

   Moving `HF_HOME` means the 72 GB preload lands in the default cache, the app looks elsewhere,
   finds nothing, and **re-downloads 72 GB at runtime** — on top of the preloaded copy, which on a
   150 GB cap is a near-certain eviction.

   **Correct approach: leave `HF_HOME` alone, read from the default cache, and avoid ever needing
   a write.** See §6.2.1.

   Related and reassuring: ZeroGPU's packer deletes the HF cache blobs after packing anyway
   (§2.5). **Treat the HF cache as transient by design** — it is not a durable store, and any
   code that assumes a file stays there after startup is wrong.

4. **Revision.** The entry format is `repo_id file1,file2,… commit_sha`. **The SHA is not
   optional.** `model_files.py` pins `REVISION`; if the preload fetches `main` instead, those are
   *different cache keys*, the preload is wasted, and the app re-downloads everything. The drift
   test therefore compares the **whole entry string including the revision**, not just the file
   set — and `download_allow_list()` returns sorted paths so that comparison is stable.

5. **Startup timeout.** `startup_duration_timeout` defaults to **30 minutes**. Downloading and
   packing 72 GB will plausibly exceed that; the Space frontmatter sets `1h`. If the first deploy
   still times out, this is the first thing to raise — not the second.

### 6.2.1 The completeness invariant — the real fix for the permission problem

**If every file we ever load is in the preload set, nothing needs to be fetched lazily, and the
read-only-cache problem cannot arise.** That is a stronger position than mirroring, so we make it
an explicit, checked invariant rather than a hope:

> At startup, verify every path in `download_allow_list()` is present in the cache. If any is
> missing, **fail loudly with the specific filename AND its URL** rather than silently triggering
> a runtime download.

This converts the whole 2.3 permission-bug class into a diagnosable startup error. It lives in
`preflight.py` (§5.6) and it is a test case.

Mirroring stays documented as the fallback if a lazy write ever does prove necessary: hardlink
blobs so it is zero-copy, byte-copy anything under `refs/` because the HF library rewrites those,
and preserve snapshot symlinks as relative symlinks. But needing it at all should be treated as a
bug in the allow-list, not as normal operation.

### 6.3 ✅ Main gate RESOLVED · ⚠️ one secondary gate still shut

**`Lightricks/LTX-2.5-Diffusers` — OPEN.** The gate was accepted and access is verified:
`model_index.json` returns HTTP 200 with the operator's token. Weights work is unblocked.

**Still shut: the Lightricks *LoRA* repos.** Verified separately — `LTX-2.5-22b-IC-LoRA-*` and
`LTX-2.3-22b-IC-LoRA-*` are each gated on their own acceptance and the token does not have
them. This is **not** a blocker: §4.5 shows the ungated community LoRAs are a *better* R3
fixture anyway, because they are what users actually upload. Recorded so nobody is surprised
when an official IC-LoRA 403s.

**Local reality check:** the gate being open does not mean we can generate locally. The 22B
model does not run on macOS. Real generations happen on the Space, under the quota discipline
in [`docs/TEST_PLAN.md`](TEST_PLAN.md).

---

## 7. Architecture

### 7.1 Module layout

```
app.py                       # entrypoint. `import spaces` FIRST (§8.2), then build + launch.
ltx25_studio/
  config.py                  # device/dtype selection, paths, retention cap (§2.7), env overrides
  model_files.py             # §5.2 — THE single source of truth: repo ids, revisions, subfolders,
                             #   filenames, sizes, AND the dtype/quantization switch (§2.6)
  errors.py                  # typed exception hierarchy; every error carries an actionable detail
  preflight.py               # §5.6 — RAM/disk headroom vs the model_files requirement;
                             #   §1.3 — the diffusers-version fail-fast assertion
  zerogpu.py                 # §8 — the `spaces` shim, the static @spaces.GPU policy, the
                             #   duration callable, contextvar propagation, and the
                             #   "never return CUDA tensors" boundary
  constraints.py             # §5.5 — dims/frames/CRF/sigmas. Pure functions over numbers.
  pipelines.py               # construction on CPU, module-scope .to("cuda"), component wiring,
                             #   and the vae.temporal_compression_ratio tripwire (§5.5)
  conditioning.py            # UI inputs -> list[LTX2VideoCondition]; pixel->latent index
                             #   conversion (§5.7 #1); uint8/PIL normalisation (§5.7 #7)
  audio.py                   # waveform <-> mel <-> audio_vae latents; params from config (§5.4);
                             #   vocoder decode
  modes.py                   # §3.3 registry: 11 UI modes as data over the 4 real workflows
  generate.py                # §7.4 stage graph. OWNS the never-empty invariant (§5.3) and the
                             #   per-run output directory
  outputs.py                 # mux video+audio, single-track export, 20 GB oldest-first eviction
  lora/
    inspect.py               # header-only safetensors parse; metadata + convention detection
    validate.py              # validator stages 1-6 (§4.3)
    apply.py                 # load_lora_weights, stage-7 attach assertion, both-component fusing
  ui/
    blocks.py                # Gradio Blocks assembly
    components.py            # reusable widgets (condition list, LoRA picker, constraint hints)
    handlers.py              # thin adapters: UI event -> one call -> UI update. No logic.
```

### 7.2 Layering rules

- **Dependencies point inward.** `ui/` may import anything; `generate.py` must not import `ui/`.
  `errors.py` imports nothing from the package at all; `model_files.py` and `constraints.py`
  may import **`errors` and nothing else**. An architecture test asserts it, because the value of
  a bottom layer is entirely in what it refuses to depend on.
- **`modes.py` is data, not branching.** Adding a mode is a registry entry, not an `if` in
  `generate.py`.
- **`ui/handlers.py` contains no business logic.** Marshal inputs → call one function → marshal
  outputs. This is what makes the app testable without a GPU and without Gradio.
- **Errors cross layers as typed exceptions.** The UI is the only place that renders one as a
  string. `generate.py` never returns an error-shaped success (§5.3).

### 7.3 Why this shape

The predecessor put backend, models, modes and workflow in four flat modules with `ui.py`
importing all of them. That works until the failure modes get interesting: the empty-output bug
(§5.3) lived precisely in the seam between "workflow returned something" and "UI rendered it",
and there was no layer boundary at which to assert. Here `generate.py` owns that invariant and
can be tested for it directly.

### 7.4 The generation stage graph — **classic pipelines PLUS an explicit diffusion decode**

This is the most consequential architectural decision in the document, because the obvious
implementation silently discards 2.5's headline quality feature.

**`LTX25AutoBlocks` docstring, verbatim:**

> *"Auto blocks for LTX-2.5 supporting text-to-video, image-to-video, condition-to-video and
> in-context (IC-LoRA) generation (joint video + audio). Identical to `LTX2AutoBlocks` except
> that the video decoder is `LTX2DiffusionVaeDecoderStep`, **since the diffusion decoder is the
> native default from LTX-2.5 on.** To decode with the convolutional VAE instead, swap the
> decode block."*

**But the classic `LTX2ConditionPipeline` / `LTX2ImageToVideoPipeline` decode with
`self.vae.decode(...)` — the convolutional VAE, NOT the diffusion decoder.** So a naive classic
call silently gives up what the model card calls *"a new diffusion video decoder [that] replaces
the VAE reconstruction stage"*. It would not error. It would just be worse, forever, and nobody
would know why.

**Decision: classic pipelines, with the diffusion decode added as an explicit stage.**

```
stage 1   LTX2ConditionPipeline(..., sigmas=DISTILLED_SIGMA_VALUES,
                                     output_type="latent")
          → latents, audio_latents

optional  LTX2LatentUpsamplePipeline(latents)
          → upsampled latents

stage 2   same pipeline, latents=upsampled,
                          sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                          noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                          output_type="latent"
          → refined latents

decode V  LTX2VideoDiffusionDecodePipeline(latents=..., denormalize=False)   ← see below
decode A  audio_vae.decode(audio_latents) → mel → vocoder(...) → waveform
```

**`denormalize=False` is MANDATORY.** `output_type="latent"` returns *already-denormalized*
latents (§5.7 #6). Passing the default `True` **double-denormalizes and silently corrupts the
output**. This is a test case.

#### The model is loaded ONCE and every workflow shares it

`LTX2Pipeline` and `LTX2ConditionPipeline` take **identical weight-bearing
components** at the pinned revision — `scheduler, vae, audio_vae, text_encoder,
tokenizer, connectors, transformer, vocoder, processor, prompt_enhancer,
duration_head`. The only difference is that the condition pipeline additionally
accepts `audio_scheduler`: a scheduler, **zero weights**, defaulting to `None`.

Calling `from_pretrained` per workflow therefore materialises **two complete
independent copies of the model, ~67 GiB each**:

| host | consequence |
|---|---|
| ZeroGPU | ~134 GiB packed to `~/.zerogpu/tensors` against a **150 GB cap** — and packing runs *after* both builds, so it blows the ceiling even when the download fits |
| a 96 GiB card | outright OOM on the second build |

So `pipelines._base_pipeline()` loads **once** and `derive_pipeline()` builds the
other workflow from those same modules. There is no configuration in which two
copies are wanted, which is why this was fixed before the measurement confirming
its magnitude landed — the fix is correct in every branch.

**Settled by inspection, not assumption.** `diffusers/pipelines/pipeline_utils.py`
at the pinned revision has **no module-level cache of any kind** — no `lru_cache`,
no loaded-module map — and `load_sub_model` calls `load_method(...)`
unconditionally, so every call materialises fresh. Our own `_CACHE` was keyed on
the *workflow name*, so it never deduplicated across the two.

##### ⚠️ Necessary, and NOT sufficient for ZeroGPU

This fix removes **one of three** causes of the 150 GB eviction. It does not make
the Space work, and recording it as if it did would be the same confidently-wrong
statement as the `expected_weights=53.20GB` line deleted in §2.6.

The packed copy is **separate from** the HF cache snapshot, and the prune that would reclaim it
runs only *after* packing completes. **The authoritative arithmetic is §2.5**; the short version
is that even the fixed, single-pipeline case peaks at 144.37 GB before the image is counted, and
152–156 GB with it, against a 150 GB cap.

**So one pipeline does not fit either.** Double-loading took it to 216.56 GB; the Xet chunk cache
would have taken it to 288.74 GB. The eviction was **over-determined** — three independent
causes, and the smallest of them absorbed two cycles of tuning.

**Settled, not open.** ZeroGPU *does* free the snapshot — verified in `spaces` 0.51.1 — and it
frees it too late, because the unlink follows `_pack()`. So **bf16 on `storage: None` is
structurally infeasible for this checkpoint set**, and the routes are persistent storage, a
public Space with `preload_from_hub` (§6.2), or smaller checkpoints. All three are operator
decisions; none is a knob we can turn.

**The base is always the superset class**, `LTX2ConditionPipeline`. Deriving
superset → subset can only drop `audio_scheduler`, which the subset does not
take. The reverse would silently drop it from the pipeline that *does* use it,
and a missing audio scheduler degrades the joint audio track in a way the video
output never reveals — this build's recurring failure shape.

The tests assert **module identity** (`is`), not equality. Two copies work
perfectly, right up until the disk or the card runs out, so any check that only
asks whether both pipelines function is blind to it.

**Stage 2 is opt-in, and the registry overrides the request.** `GenerationRequest.enable_stage_2`
defaults to **off**: it is a real quota cost on a metered Space and the user should own that
trade. But a mode whose `skip_stage_2` is set generates video nobody sees (§3.2), so upscaling it
is pure quota — `generate.stage_2_enabled()` forces it off there, because honouring the toggle
would be honouring a mistake. `StageRunner` owns the graph so `generate.py` carries no branch on
it, and requesting stage 2 without the upsampler loaded **raises** rather than quietly rendering
single-stage, which would look like it worked and be visibly worse.

**Why not the modular path.** The model card's own Option C example uses the classic pipelines,
so that is the documented, lower-risk API. The modular path (`LTX25ModularPipeline` /
`LTX25AutoBlocks`, loaded from `modular_model_index.json`) is newer, and its interaction with
**ZeroGPU's tensor packing** (§8.3) and with **`load_lora_weights`** (§4) is unverified. Given
LoRA is the headline feature and ZeroGPU is the deploy target, betting both on an unverified
integration is the wrong trade.

The modular path is recorded here as **the canonical alternative**: if we move to it, the decode
block is swappable exactly as the docstring describes, and `vae/` (1.452 GB) stays in the subset
as the conv-VAE fallback either way.

---

## 8. ZeroGPU execution model

**This section replaces the conventional wisdom about diffusion-model offloading. On ZeroGPU
that wisdom is not merely suboptimal — it is counterproductive.**

### 8.1 The hardware is not what the API says

ZeroGPU is **NVIDIA RTX PRO 6000 Blackwell Server Edition**, partitioned via MIG (sm_120).

| tier | VRAM | quota |
|---|---:|---|
| `large` (default) | **48 GB** | 1× |
| `xlarge` | **96 GB** | 2× |

The Spaces API still reports the legacy id string **`zero-a10g`**. **Do not read that as an
A10G** — it is a stale identifier, and sizing decisions made from it would be wrong by a factor
of several.

**VRAM is not our constraint.** 67.2 GiB of weights against `xlarge`'s ~95 GiB leaves ~28 GiB
for activations. The constraints are CPU RAM and transient disk (§2).

### 8.2 Import order is load-bearing

```python
import spaces          # MUST be first
import torch           # only after
import diffusers       # only after
```

`import spaces` **must precede** torch / diffusers / transformers, or startup fails with
`"CUDA has been initialized before importing the spaces package"`. This is why `app.py` is thin
and why the `spaces` shim lives in `zerogpu.py` with a guarded import for local dev.

### 8.3 The packing model: build on CPU, `.to("cuda")` at module scope

ZeroGPU **replaces** CPU offload with its own disk-packing engine. The correct pattern inverts
normal GPU-box practice:

1. Build the pipeline **on CPU**, at module scope.
2. Call `pipe.to("cuda")` **at module scope**. That call is **intercepted**, rewritten to
   `.to("cpu")`, and every tensor is packed to disk (`~/.zerogpu/tensors`, O_DIRECT). After
   packing, the main process holds no weights.
3. Apply `@spaces.GPU(...)` **statically at module import**. Dynamic or inner decoration fails
   ZeroGPU's startup analyzer with *"No @spaces.GPU function detected during startup"*.
4. Inside the decorated function real CUDA is live, and a forked worker streams
   **disk → pinned host memory → VRAM**.

HF's guidance, quoted: **models must be placed on cuda at the root module level.**

##### ⚠️ The snapshot is freed after packing — but only if the mmap survives

Read from `spaces` 0.51.1 (`zero/torch/patching.py:395`) rather than inferred:

```python
with unmap_capture(mmap_addrs) as unmapped_paths:
    total_size = _pack(Config.zerogpu_offload_dir)
for path in unmapped_paths:
    path.unlink(missing_ok=True)
```

`unmap_capture` reads `/proc/self/maps`, maps each tensor's `data_ptr()` back to
**the blob file it is mmap'd from**, and unlinks those matching
`~/.cache/huggingface/hub/models--*/blobs/*`.

**So the 72 GB snapshot IS reclaimed — conditionally.** Two preconditions, both
easy to break and both silent when broken:

1. **Tensors must still be mmap'd from the blobs when packing runs.** A dtype
   **cast**, a `.clone()`, an eager `low_cpu_mem_usage=False`, or a
   non-safetensors path materialises them into anonymous memory, leaving nothing
   in the map to capture. Nothing is freed, 72 GB stays forever, **no error and
   no log line**. This is why `torch_dtype` must equal the checkpoint's on-disk
   dtype rather than being a free parameter — recorded at the risk site in
   `pipelines.get_pipeline`.
2. **The cache must live under `~/.cache/huggingface/hub`**, or the autoprune
   pattern stops matching. **This is the sharper reason not to move `HF_HOME`** —
   §6.2 gives the preload reason, and this one is worse because it fails
   silently rather than loudly.

**And the unlink runs strictly AFTER `_pack()` completes**, so both copies
coexist for the whole packing window — which is the peak in §2.5, the authoritative arithmetic.
Steady state afterwards is fine at ~72 GB. **It is the peak that evicts us, not
the residency.**

**Corollary: do NOT lazy-load models inside the decorated function.** HF documents that
placements are optimised for startup; lazy loading defeats the packer entirely.

##### Considered and rejected: deferring the second workflow

Building only one pipeline at startup and the other on demand would halve the
packed footprint, and the disk arithmetic in §7.4 makes that look attractive.

**It is a worse fix, and worse in a way visible only on ZeroGPU.** A pipeline
built lazily inside the decorated function is never packed, so it is re-transferred
from RAM on **every cold worker** — trading a one-off disk cost for a per-request
cost, on the platform where cold starts are the normal case. It would show up as a
win in the disk numbers and a loss in every generation.

`from_pipe` gets both properties at once: both pipelines exist at module scope so
both are packed, and they share module objects so there is only one set of tensors
*to* pack. Strictly better on both axes.

Recorded rather than dropped because the arithmetic makes it an obvious idea, and
the next person to have it deserves the reason it does not work rather than
silence.

### 8.4 `enable_model_cpu_offload()` must NOT be used

This supersedes an earlier draft of this plan, which called it mandatory. It is the opposite.

`enable_model_cpu_offload()` and `enable_sequential_cpu_offload()` are **counterproductive on
ZeroGPU**, for a specific mechanical reason: any submodule accelerate parks on CPU is **never
registered for packing**. Consequently it —

- stays resident in **main-process RAM** (which is our tightest budget, §2),
- is **fork-inherited by every worker**, multiplying that cost, and
- is **re-copied host→device on every hook fire**, on top of the packer's own streaming.

So we pay RAM we do not have, to duplicate work the platform already does better. The
`model_cpu_offload_seq` string recorded in §5.7 is upstream metadata we do not act on.

### 8.5 Never return CUDA tensors across the `@spaces.GPU` boundary

Returning a CUDA tensor from a `@spaces.GPU` function makes unpickling in the main process
trigger a **blocked lazy CUDA init**, and **the call hangs** — no error, no timeout message, a
dead request.

**Rule: return file paths, or `.cpu()` tensors. Always.** `zerogpu.py` owns this boundary and it
is a test case.

### 8.6 Tier and quota

We declare `size="xlarge"` **explicitly** rather than being silently promoted.

Packed bf16 weights (67.2 GiB) exceed `xlarge_threshold = 42.00 GiB`, so **ZeroGPU auto-promotes
us to `xlarge` and doubles quota cost with no code change** (§2.6). Declaring it explicitly means
the cost is visible in the source rather than being a surprise in the quota dashboard.

**Quota arithmetic:** PRO gives 2400 s/day. At `duration=75` on `xlarge` (2×), each run is
charged ~150 s → **~16 runs/day**. Dropping under 42.00 GiB via quantization would return us to
`large` at 1× — that is the entire quota argument for §2.6's fallback, and it is worth
approximately double the runs.

Two further traps from §5.8: the hidden **1.5× `duration_factor` on `large`** (so `duration=60`
schedules as 90 s; `xlarge` skips the factor but doubles the request instead), and the fact that
the **maximum duration is a server-side, tier-dependent, unpublished cap** — discover it, never
hardcode it. ~120 s is the empirical safe ceiling; the official Lightricks LTX Space ships
`@spaces.GPU(duration=75, size="xlarge")`.

### 8.6.1 Where §8 actually executes — and why the decorator is unconditional

Everything above is a rule about `app.py`, and for a while every one of them existed only as
prose: no executable `@spaces.GPU`, no executable `.to("cuda")`, and `zerogpu.py` imported by
nothing. The app would have deployed, started, and never touched a GPU.

What it looks like now, and each line is load-bearing:

```python
import spaces                      # FIRST, before anything that initialises CUDA
CONFIG = load_config()             # unguarded: a bad LTX25_* reaches the operator
preflight.run(CONFIG, strict=ON_ZEROGPU)
pipelines.get_pipeline(..., device="cuda")     # module scope -> the packer runs
_GPU_KWARGS = zerogpu.gpu_decorator_kwargs(spaces.GPU, duration=..., size=...)

@spaces.GPU(**_GPU_KWARGS)         # literal, static, UNCONDITIONAL
def run_generation(**params):
    return zerogpu.assert_returnable(_GENERATE(**params))

zerogpu.check_duration_signature(run_generation, zerogpu.estimate_duration)
generate_module.set_gpu_dispatch(run_generation)
```

**The decorator is unconditional on purpose.** Off-Space, `spaces` is a stub whose `GPU()` returns
the function unchanged, so the wiring is identical everywhere and only its *effect* differs. A
decorator behind an `if` is precisely how a documented rule quietly stops being applied: the Space
path stops being the path anyone runs, nothing local exercises it, and the docstring keeps
reassuring. Making the stub carry the difference means **the line you run locally is the line that
runs on the Space**.

`_GENERATE` is bound at import rather than resolved by attribute, because a wrapper installed
after startup would otherwise send the worker back through itself.

### 8.7 Local (Apple Silicon / MPS) is a different regime

MPS has no ZeroGPU packer, no `spaces` decorator semantics, and unified memory changes the
arithmetic entirely. Device strategy is selected in `config.py`, the `spaces` import is guarded,
and the local path is expected to be **capability-limited rather than identical**. Do not assume
a code path that works on ZeroGPU works on MPS, or vice versa.

---

## 9. Testing strategy

We run an **H3 TDD harness**. The gate is:

```
.venv/bin/python -m pytest -q
```

**Ownership:** a separate teammate owns `tests/**`. The implementer does not create or edit
anything under `tests/`. This plan's job is to make the units *testable*.

### 9.1 Testable without a GPU, without weights, without network

| unit | how |
|---|---|
| `constraints.py` | pure functions over ints — dimension snapping, frame-count validation (with an injected compression ratio), CRF selection from a stubbed `model_type`, sigma schedule shape |
| `model_files.py` | data assertions: every entry has repo/revision/subfolder/filename; no duplicates; the §2.3 subset sums to 72.185; the dtype switch enumerates the three §2.6 configs |
| `preflight.py` | inject fake free-RAM/free-disk numbers → assert a specific error naming both numbers; assert the diffusers-version check fires on a wrong version |
| `lora/inspect.py` | **synthesised safetensors headers** — small fixtures, no real LoRA (this is why §4.3 stage 2 is header-only) |
| `lora/validate.py` | stages 1–6 against crafted good/bad headers, including the wrong-architecture case |
| `modes.py` | registry completeness (11 modes over 4 workflows); each mode's kwarg builder produces expected kwargs |
| `errors.py` | every error type's message contains the actionable detail (filename, URL, found-prefixes, measured numbers) |
| `conditioning.py` | pixel→latent index conversion (§5.7 #1); out-of-range raises (§5.7 #2) |
| `outputs.py` | 20 GB oldest-first eviction over a temp dir |
| `generate.py` | stub pipeline returning nothing → assert a specific exception, **not** an empty path |
| `zerogpu.py` | the return-boundary rejects/converts CUDA-tensor returns; `duration=` callable signature matches the decorated function |

### 9.1.1 The lesson this build actually taught

> **A module can be complete, documented, specced and green, and still be connected to nothing.**
>
> *And it can be imported, and still be connected to nothing that matters.*

The corollary is the part that took five separate blockers to learn, every one of them found
**after** the gate first went green:

| # | What was disconnected | How it looked |
|---|---|---|
| 1 | `generate` ↔ `handlers` disagreed about what success is | every successful run in the three flagship modes reported as an internal error |
| 2 | `zerogpu` wired to nothing | the app would have deployed, started, and **never touched a GPU** |
| 3 | `generate` ↔ its own GPU entrypoint | the worker resolved `generate` by attribute and re-entered itself |
| 4 | `conditioning` reached only from a UI readout | raw dataframe rows went to the pipeline; the pixel→latent conversion never ran |
| 5 | `lora/apply` — **importer count zero**, 19 passing specs | the entire LoRA validator was itself never applied |

**#5 is the one to remember:** the machine built to prevent silent wrongness was itself silently
doing nothing, and **19 green specs said otherwise**. Nine validation stages, a header-only
parser, a per-module dimension check — a user would upload an adapter, watch it validate,
generate, receive base-model output, and be told it worked.

**They were not five unrelated mistakes. They were one mistake with five faces**, and the reason
it recurred matters more than the instances: *everyone verified the thing they had just built,
using a check they designed, against the mental model they already held.* Unit specs passed.
Hand-written probes passed. AST scans and `sys.modules` checks passed. Every one of them tested a
**belief** rather than the system.

The only checks that found anything were the ones whose *shape* did not depend on the checker
being right — seam tests driving real layer A into real layer B, and the evict-and-diff sweep
that asks the program what it actually reached. Stated as the person who kept missing them:

> **I probed the path I had in mind rather than the paths other code takes.**

**The numbers make it concrete, and this is the load-bearing fact of the whole build:
the gate was green at 1102 passing tests with all five blockers still in the tree.** Nothing was
failing. The app could not reach a GPU, could not apply a LoRA, and reported successful runs as
internal errors — at 1102 passing, zero failing. It stands at ~1800 now, and the difference is almost entirely **seam and
reachability tests — not more unit coverage**. More tests of the same shape would have found none
of them.

**Why the usual checks miss this class.** Unit tests prove a module correct *in isolation*, which
is exactly what a disconnected module is. A probe you design tests the mental model you already
have — and a wrong mental model is precisely what a seam bug is made of. `conditioning` was
`import app: yes` and would have passed any "is it imported" check, because the UI's live readout
loaded it; it was the *generation path* that never touched it.

**The check that does catch it**, and the only one here that does not depend on the checker
already suspecting the answer:

```
import app                                    # 1. load the entrypoint
evict every ltx25_studio.* that import loaded  # 2. clear the slate
run one real generation                        # 3. image + shot row + staged LoRA
diff sys.modules again                         # 4. what loaded is what the path reaches
```

Not *"is it imported"* but **"is it reached by the path that does the work"**. It cannot miss a
module nobody remembered to think about, which is the whole point. `tests/test_module_reachability.py`
pins it.

### 9.1.2 Two rules for prose, since no test can fail on it

**All four confidently-wrong artefacts on this build lived in the documentation
layer** — `expected_weights=53.20GB`, `weights cached : 37/37`, *"blows the
ceiling"*, and a §2.5/risk-#1 contradiction where the body stated a fact as known
and the risk table stated the same fact as unverified. Both sentences were
well-formed markdown and the gate has no opinion about prose.

Two rules, both cheap, both applied rather than merely recorded:

**1. A claim stated in two places is a claim that will diverge.** Where a number
or a mechanism appears more than once, make **one place authoritative** and have
the others point at it. §2.5 now carries that marker explicitly; §7.4, §8.3, the
resource table and the risk register reference it instead of restating it.

**2. Mark provenance on load-bearing claims** — *verified against `spaces` 0.51.1
source*, *measured on hinode*, *inferred*. `53.20GB` and `37/37` were both numbers
with **nothing visible behind them**, so no reader could tell there was nothing to
check.

And the rule that generalises furthest, from the contradiction above:

> ### A claim moving from **known → unknown** is nearly always a regression.
>
> Documents accrete facts and rarely lose them. When an edit is about to record
> something as uncertain, that is the moment to search the document for it being
> **known** — because the likeliest explanation is that someone already
> established it and the current author has not read it.

That is what happened here: the answer was already in `PLAN.md`, and it was
overwritten with "unverified" by an author who had the file open and searchable.

### 9.1.3 Process finding: history that misdescribes itself

`git log -- ltx25_studio/outputs.py` returns a commit whose message describes `conditioning.py`
and `lora/apply.py`. An uncommitted change by another agent was swept into `3d53dd3` by a
`git add ltx25_studio/` that was not path-scoped.

Recorded because the next reader deserves the warning: **history that misdescribes its own
contents is a trap for whoever reads it next, and it is the same failure as documentation that
reassures — just in the log instead of the source.** The root cause was structural — three agents sharing one worktree with files
reassigned mid-flight — so the mitigation is structural too: stage by explicit path, never by
directory, whenever more than one agent can write.

### 9.1.4 Two more rules, both of the §9.1.1 shape

Neither is here because it sounds wise. Both are here because their *shape* does not
depend on the author already suspecting the answer — the property §9.1.1 identifies as
the only one that found anything.

**3. Ask of every input: "if this were silently ignored, would the run still pass?"**

Wherever the answer is *yes*, a green run is not evidence of anything. The mode still
renders, the file still carries the right streams, every check passes — and the field
under test was never read. Auditing the request fields on this build turned up **five**
such inputs (`enable_stage_2`, `loras`, `auto_duration`, `noise_scale`, `strength`), and
in every case the remedy is not a sharper assertion but **a second run differing in
exactly that field**. `docs/TEST_PLAN.md` §A.4.1 is authoritative for the per-field
table and the runs it implies.

The rule earns its place by having **caught itself**. The six comparison runs it
generated would every one of them have passed vacuously: `seed=-1` means *draw a fresh
seed*, so both halves of each pair would have run at different seeds and differed for
that reason alone. Established by running it rather than by reasoning about it — two
runs at the default returned `777437012` and `3499445304`. **A rule about inputs that
are silently ignored was nearly defeated by an input that is silently ignored**, which
is the most compact argument available for checking a test can fail before trusting that
it passed.

**4. Record a fact once; derive every consumer of it.**

Rule 1 says a claim stated in two places will diverge. That is not a property of prose —
it is a property of *copies*, and code is not exempt. `model_files._SPECS` records the
live shard sets **once**, read off the repo's own index files; both consumers are then
derived from it, the runtime download and the Space `preload_from_hub:` list alike,
through `download_allow_list()`. A drift test compares the committed README against that
function's output, so the two cannot separate quietly.

The payoff was not theoretical. The 37-file, 72.185 GB list was **correct on hinode — a
machine nobody wrote it for** — including the duplicate-shard trap it exists to dodge:
`transformer/` ships a 4-shard *and* an 8-shard copy of the same weights, `connectors/` a
2-shard and a single-file copy, and a naive fetch of those folders pays **44.33 GB for
nothing**. A hand-maintained second copy would have been accurate about the machine its
author had in front of them — the only machine that never needed it to be right.

### 9.2 GPU-gated tests

Anything needing real weights is marked `gpu` and deselected by default. CI and the default local
run must be green with zero weights on disk.

### 9.3 The invariant tests that matter most

Ranked by how much silent wrongness they prevent:

1. **A LoRA for the wrong architecture is rejected** — specifically at stage 7 if it survives
   stages 1–6 (§4.3).
2. **A generation that produces no file raises** — never returns an empty path, and there is no
   newest-file fallback (§5.3).
3. **`denormalize=False` on the diffusion decode** — the double-denormalization corruption
   (§7.4) is silent and total.
4. **Pixel→latent index conversion** — off-by-8 is guaranteed without it (§5.7 #1).
5. **Audio/video length desync** with a supplied `audio_latents` (§5.7 #11), concrete case.
6. **The diffusers-version assertion fires** on 0.39.0 (§1.3) — otherwise the failure is late
   and expensive.
7. **`vae.temporal_compression_ratio` tripwire** raises if upstream changes it (§5.5).
8. **`preload_from_hub` names only live shards** — assertable against the index JSON with no
   download (§2.1).
9. **`fuse_lora` passes both components** (§4.4).

---

## 10. Resolved questions and technical risks

All six open questions from the first draft are **resolved**. What remains is one ranked risk
register.

| # | Question | Resolution |
|---|---|---|
| **OQ-1** | Pin a diffusers release, or `main`? | **RESOLVED: pin the exact SHA** `ac56fa2f…` (§1.3). A SHA is more pinned than a range, not less. Plus a startup version assertion, because 0.39.0 fails *late*. |
| **OQ-2** | Does `diffusion_decoder` have a consumer? | **Moot** — the SHA provides it. Both `diffusion_decoder` (0.834) and `duration_head` (0.004) stay in the subset; duration prediction ships. |
| **OQ-3** | Is `num_frames % 8 == 1` enforced? | **RESOLVED: no.** Upstream silently floors. We raise; the modulus is a **parameter with an `8` fallback** for pre-load validation, plus a load-time tripwire (§5.5). |
| **OQ-4** | Gemma-3 or Gemma-4? | **CLOSED ON EVIDENCE: Gemma-4.** `model_index.json` says `text_encoder: ['transformers', 'Gemma4UnifiedForConditionalGeneration']`. The diffusers annotation is a union (`Gemma3ForConditionalGeneration \| Gemma4UnifiedForConditionalGeneration`); the *weights* resolve it. So CRF provably takes the `gemma4` branch → **18**. **Still derive family and CRF at runtime; hardcode neither** (§5.7 #8). |
| **OQ-5** | Audio length from duration or latent shape? | **RESOLVED: latent shape wins.** Upstream computes it from duration, then overrides from `audio_latents.shape` when supplied. The desync is real and directional (§5.7 #11). |
| **OQ-6** | Output retention bound? | **RESOLVED: 20 GB**, oldest-first, swept on startup and after each run (§2.7). |

### Technical risk register

| # | Risk | Why it is uncertain | How it gets settled |
|---|---|---|---|
| **1** | ~~bf16 fits ZeroGPU's transient disk~~ — **SETTLED: it does not.** | **Not a risk any more, a finding.** The snapshot **IS** freed after packing (verified, `spaces` 0.51.1) — and the unlink runs *strictly after* `_pack()`, so the 144.37 GB peak precedes it, and with an 8–12 GB image the total is **152–156 GB against a 150 GB cap**. See §2.5 for the authoritative arithmetic. The ~68 GB RAM ceiling remains a community cgroup observation, separate and still open. | **No longer settled by measurement — it is settled.** The routes are persistent storage, a public Space with `preload_from_hub` (§6.2), or smaller checkpoints. All three are operator decisions, none is a knob we can turn. **ZeroGPU is parked; hinode is the critical path.** |
| 2 | ~~the packer handles quantised tensor subclasses~~ — **RESOLVED: it does, for bitsandbytes** | `spaces/zero/torch/bitsandbytes.py` no-ops for bnb ≥ 0.46.0, wired in at `patching.py:328/355/445`. **No equivalent module exists for torchao** — bnb is supported, torchao is not, and that decides the library. *Verified, `spaces` 0.51.1.* | Not a risk. Recorded because the asymmetry is invisible from outside. See `docs/ZEROGPU_DISK.md`. |
| 3 | ~~`preload_from_hub` 10-entry cap~~ | **RETIRED.** We are one repo → one entry with comma-separated filenames. Fits comfortably (§6.2). | — |
| 4 | Max `@spaces.GPU` duration on our tier | server-side, tier-dependent, **unpublished** | discover empirically; never hardcode (§5.8 #4) |
| 5 | Multishot via prompt structure produces usable results | no API support, and the enhancer system prompts actively discourage scene cuts (§3.4) | presented to users as unverified technique, not a feature |
| **7** | ~~The diffusion decoder fetches NATTEN from the Hub at generation time~~ — **ANSWERED: once per process.** | ✅ **Measured on hinode**: two stage-2 generations in one process, **exactly one** kernel fetch. So a worker needs outbound network at its **first** stage-2 use, not on every press of Generate. | **Still a real dependency** — each cold worker pays it once, and `get_kernel("shi-labs/natten", version=1)` is hardcoded inside diffusers so we cannot pin it by revision. Lazy construction keeps it off the startup path for the eight modes that never reach stage 2. |
| 6 | An LTX-**2.3** LoRA attaches cleanly to the **2.5 distilled** transformer | the model card says most 2.3 LoRAs run on 2.5 unchanged, but our R3 fixture is a 2.3 LoRA and our transformer is 2.5-distilled; the official 2.5 LoRA targets the **DEV** transformer, so it is not a valid control | **TEST_PLAN R3** — stage 7 asserts attachment count > 0 against real weights |
| **8** | **`temporal_latent_upsampler` has no documented role** — we download 0.2619 GB and `preflight` carries a dedicated error path for it | It is the **same class** as the spatial one (`LTX2LatentUpsamplerModel`), differing only by `spatial_upsample`/`temporal_upsample` and `use_rational_resampler`. But upstream's component table lists **only** `latent_upsampler/` as belonging to the two-stage recipe, and neither documented recipe nor `LTX25AutoBlocks` mentions the temporal one. *Read off the model card at revision `871165de…` and the configs themselves, not inferred* | **Deliberately not wired.** Inventing a role for it in a graph upstream does not describe is worse than 0.26 GB of unused download. If it proves to be dead weight the allow-list can drop it — but on evidence, not on tidiness |

### ✅ Blocker cleared

**`Lightricks/LTX-2.5-Diffusers` gate is OPEN** (§6.3), verified by an authenticated HTTP 200 on
`model_index.json`. Weights work is unblocked.

Remaining access note, **not** a blocker: the Lightricks *LoRA* repos are separately gated and
still shut. §4.5 shows the ungated community LoRAs make a better R3 fixture anyway.

---

## 11. Build order

1. ✅ **This document**, committed.
2. ✅ **Scaffold** — venv, `pyproject.toml`, package stubs, `.gitignore`.
3. ✅ **`requirements.txt`** — the §5.1 pin set, with **no `spaces` line**.
4. ✅ **[`docs/TEST_PLAN.md`](TEST_PLAN.md)** — the quota-bounded GPU matrix. **Must exist and be
   reported before the first GPU call.**
5. **TDD, innermost-out**, against the test teammate's failing specs:
   `errors` → `constraints` → `model_files` → `preflight` → `lora/` → `zerogpu` → `conditioning`
   → `modes` → `audio` → `generate` → `outputs` → `ui`.
6. **GPU runs on the Space**, in the TEST_PLAN order (R1 first — it is the riskiest), logging
   actual cost after each and stopping hard at 1920 GPU-seconds.
