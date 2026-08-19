# LTX-2.5 Studio — Design Plan

**Status:** design locked, implementation not started.
**Branch:** `soma/ltx25-studio` · **Remote:** `git@github.com:techfreakworm/ltx2.5-studio.git`
**Target:** private Hugging Face Space `techfreakworm/LTX2.5-Studio` (gradio SDK, ZeroGPU).

This document is the contract for the build. Every section below is either a **decision**
(settled, build to it) or an **open question** (§10, blocks work, needs an answer). Where a
claim was verified against real source or real repo metadata, the evidence is cited inline so a
future reader can re-check it instead of trusting it.

**Prime directive:** this app must not repeat the failure modes of its predecessor
(`ltx2.3-AIO-generator`). Those failures were not model failures. They were *packaging,
pinning, and silent-failure* failures. §5 is therefore not a nice-to-have section; it is the
section the build will be judged on.

---

## 1. Decision 1 — Integration path: **Diffusers**

We build on `diffusers`, loading `Lightricks/LTX-2.5-Diffusers`.

### 1.1 Options considered

| | Path | What it is | Verdict |
|---|---|---|---|
| **A** | `ltx-pipelines` | Lightricks' own repo, installed via `uv sync` of a git workspace | **Rejected** |
| **B** | ComfyUI runtime | Vendor ComfyUI + custom nodes, drive it via API graphs | **Rejected** |
| **C** | `diffusers` | `diffusers.pipelines.ltx2`, weights from `Lightricks/LTX-2.5-Diffusers` | **CHOSEN** |

### 1.2 Why C

**The `ltx2` module is real and complete enough.** `diffusers` ships a first-class
`pipelines/ltx2` package. Verified present in the installed source tree:
`pipeline_ltx2.py`, `pipeline_ltx2_condition.py`, `pipeline_ltx2_image2video.py`,
`pipeline_ltx2_latent_upsample.py`, plus `connectors.py`, `latent_upsampler.py`,
`vocoder.py`, `export_utils.py`, `utils.py`. Models `AutoencoderKLLTX2Video`,
`AutoencoderKLLTX2Audio`, `LTX2VideoTransformer3DModel` are exported from the top-level
`diffusers` namespace.

**Path A is a heavy, fragile install for a Space.** It requires `uv sync` of a git workspace,
CUDA >= 12.7 and torch ~= 2.7. A Space build that depends on resolving a git workspace at
image-build time is a build we cannot reproduce or pin.

**Path B is the documented cause of the fragility we are told to avoid.** The 2.3 project's
`requirements.txt` vendors an entire ComfyUI runtime — roughly 50 dependencies, nearly all
unpinned: `spandrel`, `torchsde`, `kornia`, `PyOpenGL`, `glfw`, `SQLAlchemy`, `alembic`,
`Mako`, `blake3`, `comfyui-frontend-package`, `comfyui-workflow-templates`,
`comfyui-embedded-docs`, plus a hand-maintained block of *custom-node deps we discovered are
required but aren't auto-installed by their respective nodes* (`gguf`, `imageio_ffmpeg`,
`opencv-python`, `matplotlib`, `scikit-image`, `onnxruntime`, `yt-dlp`). That comment block is
itself the evidence: the dependency surface was discovered by breakage, not by design.
Diffusers replaces all of it with ~15 pinned direct deps and zero custom-node surface.

**Decisive coupling to §2.** The `*-comfy-int8-convrot` quantised weights are **ComfyUI-only**
— the model card says so explicitly. Choosing A or C means those 36.9 GB of int8 weights are
unusable to us. Path choice and disk budget are therefore one decision, not two, and §2 is
computed on the bf16 Diffusers pack.

### 1.3 The single-file escape hatch: investigated and REJECTED

It is tempting to skip the Diffusers pack and load the 200.85 GB single-file release through
`from_single_file`. That does not work, and the reason is specific.

`diffusers/loaders/single_file_utils.py` does contain an LTX-2 route:
`CHECKPOINT_KEY_NAMES["ltx2"]` → `model_type = "ltx2-dev"` → config repo
`{"pretrained_model_name_or_path": "Lightricks/LTX-2"}`. But note the repo it points at:
`LTX-2`, i.e. **2.0, not 2.5**. And the converters it ships are only:

- `convert_ltx2_transformer_to_diffusers` (`LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT`)
- `convert_ltx2_vae_to_diffusers` (`LTX_2_0_VIDEO_VAE_RENAME_DICT`)
- `convert_ltx2_audio_vae_to_diffusers` (`LTX_2_0_AUDIO_VAE_RENAME_DICT`)

There is **no** single-file converter for 2.5's `connectors`, `diffusion_decoder`, `vocoder`,
`duration_head` or `latent_upsampler`. Even a successful single-file load would give us a
pipeline missing the components that make 2.5 what it is. We use the Diffusers pack. This is
settled; do not revisit it without new upstream converters.

---

## 2. Decision 2 — Disk budget: **~72.2 GB of a 150 GB cap**

ZeroGPU gives us a 150 GB persistent-storage cap. The `Lightricks/LTX-2.5-Diffusers` repo
totals **164.19 GB across 56 files** — which would blow the cap. That total is *not* what we
download, and understanding why is a correctness requirement, not an optimisation.

### 2.1 Trap A — duplicate shard sets

Two folders in the repo ship the *same weights twice*, in two different shardings:

| folder | set 1 | set 2 | note |
|---|---|---|---|
| `transformer/` | 4 shards: 9.932 + 9.919 + 9.986 + 8.138 = **37.975 GB** | 8 shards: 4.991 + 4.944 + 4.994 + 4.928 + 4.977 + 4.995 + 4.994 + 3.163 = **37.986 GB** | only **one** `diffusion_pytorch_model.safetensors.index.json` |
| `connectors/` | single file: **6.344 GB** | 2 shards: 5.000 + 1.344 = **6.344 GB** | same |

A naive `snapshot_download` of those folders pulls **both** sets and roughly doubles the
footprint (+44.33 GB of pure duplication). `from_pretrained` does the right thing: it reads
the index JSON and fetches only the shards the index references.

**Two consequences that are binding on the build:**

1. We rely on the index. We do **not** hand-roll a `snapshot_download` of `transformer/`.
2. Any `preload_from_hub:` list in the Space README must name **only the live shards** — the
   ones the index references. A wrong list here silently doubles disk and fails the deploy at
   72 GB in, which is the worst possible time to find out.

### 2.2 Trap B — `transformer_full/` is the DEV transformer

`transformer_full/` is a further **37.975 GB** and is the *dev* (non-distilled) transformer. We
ship the distilled 8-step path by default, so we do not ship it. It is documented as an opt-in
(§2.5), not deleted from our mental model.

### 2.3 The subset we ship

| component | GB | why |
|---|---:|---|
| `transformer/` (distilled, one shard set) | 37.975 | the model |
| `text_encoder/` (5 shards: 4.961 + 4.967 + 4.884 + 4.986 + 4.121) | 23.919 | Gemma-3 12B (see §10 OQ-4) |
| `connectors/` | 6.344 | text → transformer conditioning; also a LoRA target |
| `vae/` | 1.452 | video VAE |
| `audio_vae/` | 0.107 | audio VAE (mel ↔ latent) |
| `vocoder/` | 0.258 | mel → waveform |
| `latent_upsampler/` | 0.996 | stage-2 spatial upscale |
| `temporal_latent_upsampler/` | 0.262 | stage-2 temporal upscale |
| `diffusion_decoder/` | 0.834 | high-quality decode (see §10 OQ-1) |
| `duration_head/` | 0.004 | duration prediction (see §10 OQ-1) |
| `tokenizer/` | 0.032 | |
| **TOTAL** | **72.185** | **≈ 72.19 GB** |

### 2.4 What we exclude, and why

| excluded | GB | reason |
|---|---:|---|
| `transformer_full/` | 37.975 | the DEV transformer; not needed for the distilled 8-step path |
| duplicate shard sets (transformer 8-shard + connectors 2-shard) | 44.330 | Trap A — same weights, different sharding |
| `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (repo root) | 9.699 | the distilled LoRA for **DEV** workflows; useless when we already ship the distilled transformer |

**Arithmetic check:** 72.185 + 37.975 + 44.330 + 9.699 = **164.189 GB** — which closes exactly
against the repo's stated 164.19 GB total. The accounting above is complete; there is no
unexplained remainder.

### 2.5 Headroom

`150 − 72.19 ≈ **77.8 GB**` left for the pip environment, user-uploaded LoRAs, and generated
outputs. The dev transformer (`transformer_full/`, 37.98 GB) fits inside that headroom, which is
why it is an *opt-in* rather than an impossibility — but enabling it leaves ~40 GB, and outputs
must then be pruned aggressively.

**Implied requirement:** outputs are not immortal. `outputs/` needs a retention policy (bounded
count or bounded bytes, oldest-first eviction) or the Space will fill up in normal use.

---

## 3. Decision 3 — Mode coverage: all 11, honestly

All 11 requested modes are reachable. Several are reachable only in a way that costs more than
the name suggests, and the plan states that plainly rather than shipping a fast-looking button
that is secretly slow.

### 3.1 Three facts that shape the whole registry

**(a) One conditioning mechanism covers both image and video.**
`LTX2VideoCondition` is a dataclass whose `frames` field is typed
`PIL.Image.Image | list[PIL.Image.Image] | np.ndarray | torch.Tensor` — "the image (**or
video**) to condition the video on". It carries `index` (the frame position at which the
conditioning takes effect, default `0`) and `strength` (default `1.0`). So image-conditioning
and video-conditioning are *the same code path* with a different payload. The UI should expose
it that way: a list of conditions, each with a source, an index and a strength.

**(b) The model is jointly audio-visual. There is no video-off switch.**
The return type is `LTX2PipelineOutput(frames=..., audio=...)` — both fields are non-optional
`torch.Tensor`. Every call spends video compute and returns a video. This is the single most
important cost fact in the app, and it is why the "audio" modes below carry a warning.

**(c) Audio input is not a first-class kwarg.**
There is no `audio=` argument. Audio goes in as `audio_latents` (waveform → mel →
`audio_vae` → latents), combined with `noise_scale`. `prepare_audio_latents` routes through
`_create_noised_state(latents, noise_scale, generator)`, whose body is
`noised_latents = noise_scale * noise + (1 - noise_scale) * latents` — i.e. classic
img2img-style strength, where `noise_scale=1.0` discards the input entirely and `0.0` keeps it.
Audio frame count and video frame count are computed independently, so **they can desync**;
§5.5 makes validating that a hard requirement.

### 3.2 The mode registry

| # | Mode | How it is reached | Cost note |
|---|---|---|---|
| 1 | text → video | `LTX2Pipeline(prompt=…)` | first-class |
| 2 | image → video | `LTX2ConditionPipeline` with one `LTX2VideoCondition(frames=PIL, index=0)`, empty prompt | first-class |
| 3 | image + text → video | as #2 with a prompt | first-class |
| 4 | video → video | `LTX2ConditionPipeline` with `LTX2VideoCondition(frames=<video>, strength<1.0)` | first-class |
| 5 | text → audio+video | #1 — the audio track is always produced | first-class; **this is the default shape** |
| 6 | image → audio+video | #2 | first-class |
| 7 | image + text → audio+video | #3 | first-class |
| 8 | text → audio | #1, export only the audio track | ⚠️ **still pays full video compute** |
| 9 | video → audio | #4, export only the audio track | ⚠️ same |
| 10 | audio → audio | `audio_latents` + `noise_scale`, export only audio | ⚠️ same |
| 11 | audio → video | `audio_latents` + `noise_scale`, export only video | ⚠️ pays audio encode on top |

**Mitigation for modes 8–11** (mandatory, not optional):
run at the **minimum legal resolution**, **skip the stage-2 spatial upscale**, and export only
the requested track. The UI must say, in words, that audio-only modes still render video
internally — a user who does not know that will read the runtime as a bug.

### 3.3 Multishot is a UI affordance, not a 12th mode

2.5's native multishot is driven by (i) prompt structure — per the LTX prompting guide — and
(ii) multiple `LTX2VideoCondition`s at different `index` values. It is not a separate pipeline.
We therefore build it as a **first-class UI affordance** on top of the condition list: add a
shot, give it a prompt segment and an anchor frame index. Building it as a separate "mode"
would duplicate the condition-list code for no gain.

### 3.4 Prompt enhancement: **deferred, and we ship no dead toggle**

`enable_prompt_enhancement=True` requires a separate `prompt_enhancer` component
(`google/gemma-4-E2B-it`). It is optional in the pipeline constructor and **is not present in
the Diffusers pack**. Shipping a toggle that cannot work is exactly the kind of silent lie §5
exists to prevent. Documented as deferred/opt-in with that reason; no UI control until the
component is actually fetchable and wired.

### 3.5 Duration prediction: ship it (subject to OQ-1)

`duration_head` is an optional constructor component and **is** in the pack (0.004 GB). Usage:
omit `num_frames` and pass `min_seconds` / `max_seconds`; the head predicts the duration.
This is a real differentiator and we ship it — **conditional on OQ-1 (§10), which is currently
unresolved.**

---

## 4. Decision 4 — LoRA: the headline feature

User-uploadable LoRA at inference time is the reason this app exists. It must be the most
robust thing in it.

### 4.1 Verified mechanics

- All three generation pipelines (`LTX2Pipeline`, `LTX2ConditionPipeline`,
  `LTX2ImageToVideoPipeline`) are declared as
  `class LTX2…Pipeline(DiffusionPipeline, FromSingleFileMixin, LTX2LoraLoaderMixin)`.
- `LTX2LoraLoaderMixin` sets `_lora_loadable_modules = ["transformer", "connectors"]`. Note
  **`connectors`** — a LoRA may legitimately touch the text-conditioning path, not just the
  transformer.
- **Both key conventions load.** `lora_state_dict()` detects
  `is_non_diffusers_format = any(k.startswith("diffusion_model.") for k in state_dict)` and runs
  `_convert_non_diffusers_ltx2_lora_to_diffusers(state_dict)`; separately it detects
  `has_connector = any(k.startswith("text_embedding_projection.") for k in state_dict)` and
  converts those into the connectors sub-dict. So ComfyUI-style and diffusers-style checkpoints
  both work.
- `lora_state_dict(..., return_lora_metadata=True)` returns the file's `__metadata__`.

### 4.2 The danger: validation is almost non-existent upstream

The **only** built-in check is a substring test, which raises
`ValueError("Invalid LoRA checkpoint. Make sure all LoRA param names contain `'lora'` substring.")`.

A LoRA built for a *completely different architecture* passes that check — it has `lora` in its
key names — and then attaches to **zero** modules. Generation proceeds. Output looks like the
base model. **No error is raised.** That is silent garbage, and it is precisely the class of
failure the operator has told us to eliminate.

We also have a real compatibility signal available. Reading the official LoRA's safetensors
header directly: 3320 tensors, all under prefix `diffusion_model.`, peft-style
`lora_A.weight` / `lora_B.weight`, targeting `to_q` / `to_k` / `to_v` / `to_gate_logits`,
`adaln_single`, `audio_adaln_single`; and
`__metadata__ = {'lora_rank': '450', 'lora_alpha': '450', 'model_version': '2.5.0', 'license': …}`.
**`model_version` is machine-readable** and we should use it.

### 4.3 The validator (7 stages — each stage is a test unit)

Accepts either a **Hub repo-id** or a **direct file upload**.

| # | Stage | Rule | On failure |
|---|---|---|---|
| 1 | **Size bound** | reject above a configured max | refuse before any download/parse |
| 2 | **Header-only parse** | read the safetensors header, **never** load the tensor payload to validate | — |
| 3 | **Format** | must be safetensors with a parseable header | specific message: not-safetensors vs unparseable-header |
| 4 | **Version** | read `__metadata__.model_version`; compare to 2.5 | **warn** on absent, **reject** on explicit mismatch |
| 5 | **Convention** | classify keys: `diffusion_model.` / `transformer.` / `text_embedding_projection.` / unknown | on unknown, the error **names the prefixes actually found** |
| 6 | **Pairing** | require `lora_A`/`lora_B` (or `lora_down`/`lora_up`) to pair up | name the unpaired keys |
| 7 | **Attach assertion** | **after** `load_lora_weights`, assert the adapter attached to **> 0** modules | raise a specific error if zero — *this is the anti-silent-garbage check* |

Stage 2 is a hard rule, not a preference: a multi-GB upload must never be fully materialised
just to find out it is the wrong architecture.

Stage 7 is the one that upstream does not do for us and the one that actually protects the
user. It must run on every load, including the built-in LoRAs.

---

## 5. Decision 5 — Resilience doctrine

Five rules. These are binding on all code in the repo.

### 5.1 Pin every dependency, with upper bounds

Evidence from the predecessor's `requirements.txt`: `spaces>=0.30.0` and `torch>=2.4.0` — floors
with **no ceiling** — alongside exactly one hard pin, `torchaudio==2.10.0`, carrying this
comment:

> *"HF Spaces ships torch==2.10.0; bare 'torchaudio' lets pip pick an older wheel whose C++ ABI
> doesn't match → import crashes the whole audio code path… Pin the matching version."*

That comment is a post-mortem. It documents a break that already happened, and the fix was
applied to exactly one line while ~50 other unbounded lines were left. Every direct dependency
here gets a floor **and** a ceiling. `requirements.txt` is pending exact pins from the lead and
is deliberately not written yet.

### 5.2 One central model-file indirection

Every Hub repo-id, revision, subfolder and filename lives in **one** module —
`ltx25_studio/model_files.py` — as data, never inlined at a call site. An upstream rename must
be a one-line change in one file.

A missing file must raise an error naming **both the file and its URL**. This is not
hypothetical: Lightricks deleted a 2.3 checkpoint from the Hub with no warning and broke a live
Space. When it happens again, the error message should tell the operator exactly which URL to
go look at.

Pinning a **revision** (commit SHA) per repo is part of this module's job — a repo whose content
can change under us is not pinned just because its name is stable.

### 5.3 Never return an empty output

Any generation that completes without producing a file **must** raise a specific, actionable
error that the UI renders visibly.

The 2.3 outage hid for a **week** because a mid-graph load failure returned an empty video path,
and the UI rendered an empty path as a silent form reset. The user saw "nothing happened". No
error, no log, no signal. Structurally: a success path that can return a falsy artifact is a
bug, regardless of whether it is currently reachable.

### 5.4 Read audio parameters from config, never hardcode

Mel/audio parameters come from `audio_vae.config` at runtime:
`audio_sampling_rate`, `audio_hop_length`, `audio_mel_bins`, `audio_vae_mel_compression_ratio`,
`audio_vae_temporal_compression_ratio`, `audio_latent_channels`.

This mirrors what the pipeline itself does — its constructor reads
`self.vae.spatial_compression_ratio`, `self.vae.temporal_compression_ratio`,
`self.audio_vae.mel_compression_ratio`, `self.audio_vae.temporal_compression_ratio` and
`self.transformer.config.patch_size` from the loaded components, using literals only as
fallbacks when a component is absent. We follow the same discipline.

### 5.5 Enforce and surface the hard constraints

| constraint | source | handling |
|---|---|---|
| `height % 32 == 0` and `width % 32 == 0` | enforced upstream: `if height % 32 != 0 or width % 32 != 0: raise ValueError(...)` in all three pipelines *and* in the latent-upsample pipeline | validate **before** the call, snap in the UI, show the snapped value |
| `num_frames % 8 == 1` | **not** hard-raised upstream; it falls out of `latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1`, which silently floors a bad value | validate ourselves. **Derive the modulus from `vae.temporal_compression_ratio`, do not hardcode 8** — same principle as §5.4 |
| image conditioning CRF | `LTX2_5_IMAGE_CRF = 18` for 2.5, vs the generic `DEFAULT_IMAGE_CRF = 33` | 2.5 image conditioning uses **18**; the default is wrong for us |
| audio/video length agreement | audio and video frame counts are computed independently (§3.1c) | validate they agree; raise rather than emit a desynced file |

**Verified constants** (`diffusers/pipelines/ltx2/utils.py`):

```python
DISTILLED_SIGMA_VALUES         = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]  # 8 steps, CFG=1
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875]                                        # 3 steps
```

These live in `constraints.py` as re-exports/asserts against the installed diffusers, so a
schedule change upstream fails a test instead of silently changing output quality.

---

## 6. Decision 6 — Deploy shape

| | |
|---|---|
| **Space** | `techfreakworm/LTX2.5-Studio`, **private** |
| **SDK** | gradio |
| **Hardware** | ZeroGPU — `suggested_hardware: zero-a10g` (the identifier HF uses); account is PRO |
| **Weights** | README frontmatter `preload_from_hub:` (the mechanism the 2.3 Space uses) |
| **Secret** | `HF_TOKEN` — required, because `Lightricks/LTX-2.5-Diffusers` is gated |
| **Git** | GitHub `git@github.com:techfreakworm/ltx2.5-studio.git`; Space and GitHub stay in sync |

Naming precedent in the operator's account: `techfreakworm/LTX2.3-Studio`, `wan-studio`
(private), `SAM3-cutout-studio` (private).

**`preload_from_hub` must list only the live shards** — see §2.1 Trap A. Getting this list wrong
does not fail fast; it fails after tens of GB of download.

### 6.1 🚨 BLOCKER — repo gate not accepted

`Lightricks/LTX-2.5-Diffusers` is gated (`gated: auto`) and the operator's token has **not**
accepted that gate. Verified response:

> `Access to model Lightricks/LTX-2.5-Diffusers is restricted and you are not in the authorized list.`

This is a **separate acceptance** from `Lightricks/LTX-2.5`, which *is* already accepted. One
click resolves it:

**https://huggingface.co/Lightricks/LTX-2.5-Diffusers**

Until it is accepted, nothing that touches real weights can be built or tested — local or Space.
Escalated to the orchestrator.

---

## 7. Architecture

### 7.1 Module layout (proposed)

```
app.py                       # entrypoint: local + Space. Thin — builds UI, launches.
ltx25_studio/
  __init__.py
  config.py                  # runtime config: device/dtype selection, paths, env, retention policy
  model_files.py             # §5.2 — THE single source of truth for repo ids, revisions,
                             #   subfolders, filenames, expected sizes. Data only, no logic.
  errors.py                  # typed exception hierarchy; every error carries an actionable message
  constraints.py             # §5.5 — dimension/frame validation & snapping, CRF, sigma schedules
  pipelines.py               # pipeline construction, component wiring, caching, offload policy
  conditioning.py            # UI inputs -> list[LTX2VideoCondition]; multishot anchoring (§3.3)
  audio.py                   # waveform <-> mel <-> audio_vae latents; all params from config (§5.4)
  modes.py                   # §3.2 registry: 11 modes as data + per-mode kwarg builders
  generate.py                # orchestration: mode -> stage 1 -> optional stage 2 -> outputs.
                             #   Owns the never-empty-output guarantee (§5.3).
  outputs.py                 # encode/mux video+audio, audio-only export, retention/eviction
  lora/
    __init__.py
    inspect.py               # header-only safetensors parse; metadata + key-convention detection
    validate.py              # the 7-stage validator (§4.3 stages 1-6)
    apply.py                 # load_lora_weights + the post-attach assertion (§4.3 stage 7)
  ui/
    __init__.py
    blocks.py                # Gradio Blocks assembly
    components.py            # reusable widgets (condition list, LoRA picker, constraint hints)
    handlers.py              # thin adapters: UI event -> generate.py -> UI update. No logic.
```

### 7.2 Layering rules

- **Dependencies point inward.** `ui/` may import anything; `generate.py` must not import `ui/`.
  `model_files.py`, `errors.py` and `constraints.py` import nothing from the package.
- **`modes.py` is data, not branching.** Adding a mode should mean adding a registry entry, not
  adding an `if` to `generate.py`.
- **`ui/handlers.py` contains no business logic.** Every handler is: marshal inputs → call one
  function in `generate.py` or `lora/` → marshal outputs. This is what makes the app testable
  without a GPU and without Gradio.
- **Errors cross layers as typed exceptions**, and the UI is the only place that turns an
  exception into a string for a human. `generate.py` never returns an error-shaped success.

### 7.3 Rationale for the shape

The predecessor put backend, models, modes and workflow in four flat top-level modules with
`ui.py` importing all of them. That works until the failure modes get interesting: the empty-
output bug (§5.3) lived precisely in the seam between "workflow returned something" and "UI
rendered it", and there was no layer boundary at which to assert. Here, `generate.py` owns that
invariant and can be tested for it directly.

---

## 8. VRAM and offload

**The weights do not fit in VRAM.** ~72 GB of components against a ZeroGPU allocation an order
of magnitude smaller. Therefore:

- **`enable_model_cpu_offload()` is mandatory**, not a tuning option. It is part of pipeline
  construction in `pipelines.py`, not a user-facing toggle.
- **Peak GPU residency ≈ transformer (37.98 GB) + activations.** The transformer is the
  irreducible resident component during denoising; everything else (text encoder, VAEs, vocoder,
  upsamplers, decoder) can be paged in and out around it.
- **The text encoder is the second-largest component (23.9 GB) and is needed only once**, before
  denoising. It should be evicted after prompt encoding rather than held.
- **Stage 2 (latent upsample) is a separate pipeline** — `LTX2LatentUpsamplePipeline` — which
  means it is a separate residency window and can be skipped entirely for the audio-only modes
  (§3.2 mitigation).
- **Exact ZeroGPU numbers pending** from the lead; the offload strategy above does not depend on
  them, but the maximum supported resolution × frame count does.

**Local (Apple Silicon / MPS) is a different regime.** MPS has no `enable_model_cpu_offload()`
equivalent with the same semantics and unified memory changes the arithmetic entirely. Device
strategy is selected in `config.py` and the local path is expected to be capability-limited
rather than identical. Do not assume a code path that works on ZeroGPU works on MPS.

---

## 9. Testing strategy

We are running an **H3 TDD harness**. The gate is:

```
.venv/bin/python -m pytest -q
```

**Ownership:** a separate teammate owns `tests/**`. The implementer does not create or edit
anything under `tests/`. This plan's job is to make the units *testable*; writing the tests is
someone else's.

### 9.1 What must be testable without a GPU and without weights

This is the design constraint that shapes §7. The following must be exercisable with no CUDA,
no downloaded weights, and no network:

| unit | how it is tested without weights |
|---|---|
| `constraints.py` | pure functions over ints — dimension snapping, frame-count validation, sigma schedule shape |
| `model_files.py` | data-only assertions: every entry has a repo, revision, subfolder, filename; no duplicates; the §2.3 subset sums to the stated total |
| `lora/inspect.py` | synthesised safetensors **headers** — small fixtures, no real LoRA needed (this is why stage 2 is header-only) |
| `lora/validate.py` | the 7 stages against crafted good/bad headers, including the wrong-architecture case |
| `modes.py` | registry completeness (all 11 present), and each mode's kwarg builder produces the expected kwargs for given inputs |
| `errors.py` | every error type produces a message containing the actionable detail (filename, URL, found-prefixes) |
| `outputs.py` retention | eviction logic over a temp dir |
| `generate.py` never-empty invariant | inject a stub pipeline that returns nothing → assert a specific exception, **not** an empty path |

### 9.2 GPU-gated tests

Anything that needs real weights is marked and skipped by default (the predecessor used a
`gpu` marker plus a `--gpu` flag; same pattern is appropriate here). CI and the default local
run must be green with zero weights on disk.

### 9.3 The invariant tests that matter most

Three tests earn their keep more than all the others:

1. **A LoRA for the wrong architecture is rejected** — and specifically, that it is rejected at
   stage 7 if it survives stages 1–6 (§4.3).
2. **A generation that produces no file raises** — never returns an empty path (§5.3).
3. **The `preload_from_hub` list contains only live shards** — assertable against the index JSON
   without downloading anything (§2.1).

---

## 10. Open questions and implementer push-back

Recorded rather than silently resolved. **OQ-1 is blocking.**

### 🚨 OQ-1 (BLOCKING) — "pin everything" and "ship duration_head" are currently in conflict

**Evidence.** Inspecting an installed **released** `diffusers==0.37.1`:

- `diffusers/pipelines/__init__.py` exports exactly four LTX2 pipelines:
  `LTX2Pipeline`, `LTX2ConditionPipeline`, `LTX2ImageToVideoPipeline`,
  `LTX2LatentUpsamplePipeline`. Plus models `AutoencoderKLLTX2Video`,
  `AutoencoderKLLTX2Audio`, `LTX2VideoTransformer3DModel`.
- **Not present at any version on disk:** `LTX2InContextPipeline`, `LTX2ReferenceCondition`,
  `LTX2HDRPipeline`, `LTX2HDRReferenceCondition`, `LTX2VideoDiffusionDecodePipeline`,
  `LTX2DurationHead`, `LTX2VocoderWithBWE`, `LTX2VideoHDRProcessor`.
- `LTX2ConditionPipeline.__init__` in 0.37.1 takes exactly:
  `scheduler, vae, audio_vae, text_encoder, tokenizer, connectors, transformer, vocoder`.
  There is **no `duration_head` parameter and no `prompt_enhancer` parameter**. Grepping the
  whole `pipelines/ltx2/` tree for `duration_head`, `prompt_enhancer`, `min_seconds`,
  `max_seconds`, `enable_prompt_enhancement` returns **zero hits**.
- `LTX2_5_IMAGE_CRF` also does not exist in 0.37.1 — only `DISTILLED_SIGMA_VALUES` and
  `STAGE_2_DISTILLED_SIGMA_VALUES` are in `utils.py` (both matching §5.5 exactly).

**Conclusion.** The export list in the brief is from `diffusers` **main (unreleased)**, not from
any published release. That is not a problem in itself — but it collides head-on with §5.1
"pin every dependency with upper bounds". As written, we cannot both pin diffusers to a release
*and* ship `duration_head` (§3.5), the diffusion decoder (§2.3, 0.834 GB) or `LTX2_5_IMAGE_CRF`
(§5.5).

**Recommendation — pin to a git commit SHA, not to `main`:**

```
diffusers @ git+https://github.com/huggingface/diffusers@<exact-sha>
```

A SHA is *more* pinned than a version range, not less: it is byte-for-byte reproducible, it
cannot drift, and it gets us the full 2.5 feature set. It costs a git clone at build time and it
means we own upgrades manually — both acceptable. The alternative, `diffusers>=0.37,<0.38`, is a
smaller build but silently drops duration prediction, the diffusion decoder, HDR and in-context.

**This needs the lead's ruling before `requirements.txt` is written or `pipelines.py` is built,**
because it decides which components `model_files.py` lists and which modes `modes.py` can offer.

### OQ-2 — the `diffusion_decoder` in the disk budget has no consumer at 0.37.1

The 0.834 GB `diffusion_decoder/` is used via `LTX2VideoDiffusionDecodePipeline`, which does not
exist in the released package (see OQ-1). If OQ-1 resolves to a release pin, `diffusion_decoder`
and `duration_head` should come out of the §2.3 subset (−0.838 GB) and the plan should say so
rather than shipping weights nothing loads.

### OQ-3 — `num_frames % 8 == 1` is a derived rule, not an upstream check

Upstream computes `latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1`
and **silently floors** a non-conforming value — there is no raise. Two implications:

1. We must validate it ourselves; nothing upstream will catch it.
2. The `8` is `vae.temporal_compression_ratio`, read from the loaded VAE with `8` only as a
   fallback default. Hardcoding `8` in our validator would violate §5.4's own principle. §5.5 is
   written to derive it; flagging it because the brief stated it as a literal.

### OQ-4 — text encoder is Gemma-**3**, not Gemma-4

The pipeline constructors type the text encoder as `Gemma3ForConditionalGeneration` with a
`GemmaTokenizer | GemmaTokenizerFast`. The brief says "Gemma4-12B". Cosmetic for the disk budget
(the 23.92 GB figure is unaffected), but `model_files.py` and any user-facing label should not
claim a model family the loader does not use. Note this is *separate* from the prompt enhancer,
which genuinely is `google/gemma-4-E2B-it` (§3.4) — that one is a different model and is not in
the pack anyway.

### OQ-5 — audio/video length desync: which direction?

§3.1(c) states audio frame count is inferred from supplied latents' shape. In 0.37.1's
`LTX2Pipeline` the derivation reads `audio_num_frames = round(duration_s * audio_latents_per_second)`
— derived from *duration*, not from the latent shape. The desync risk is real either way, so
§5.5's "validate that they agree" stands regardless; but the exact mechanism should be re-read on
whichever diffusers revision OQ-1 selects before `audio.py` is written.

### OQ-6 — output retention is unspecified

§2.5 leaves ~78 GB of headroom, shared between the pip environment, user LoRA uploads and
outputs. Video outputs are large and a Space that never evicts will fill. Proposing bounded-bytes
retention with oldest-first eviction in `outputs.py`, configured in `config.py`. Needs a number.

---

## 11. Build order

1. ✅ **This document**, committed.
2. **Scaffold** — venv, `pyproject.toml` (metadata + ruff + pytest with `pythonpath = ["."]`),
   package with importable stubs matching §7.1, `.gitignore`. **No app logic.**
3. **Blocked on OQ-1** — `requirements.txt` (pins from the lead) and `model_files.py`.
4. **Blocked on §6.1** — anything that touches real weights.
5. Then, TDD, innermost-out: `errors` → `constraints` → `model_files` → `lora/` → `modes` →
   `generate` → `outputs` → `ui`.
