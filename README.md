---
title: LTX-2.5 Studio
emoji: 🎬
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.24.0
python_version: "3.12"
app_file: app.py
# Legacy hardware id string. The hardware is NOT an A10G — ZeroGPU is an NVIDIA
# RTX PRO 6000 Blackwell Server Edition, MIG-partitioned (sm_120). This is the id
# the Spaces API still reports for ZeroGPU. Do NOT "correct" it to an h200 value:
# that moves the Space off ZeroGPU onto paid dedicated hardware. See docs/PLAN.md §8.1.
suggested_hardware: zero-a10g
pinned: false
short_description: Every LTX-2.5 mode, with user-uploadable LoRA at inference time.
#
# GENERATED — do not hand-edit. Emitted by `model_files.preload_from_hub_entry()`,
# the single source of truth for every repo id, revision, subfolder and filename
# (PLAN §5.2). `tests/test_deploy_drift.py` asserts this block equals what that
# function returns, so an upstream re-shard is a regenerate, never a hand-patch.
#
# Why it must be generated: the model repo ships the same weights TWICE in two
# different shardings (PLAN §2.1, "Trap A"). A hand-written list naming the dead
# shard set does not fail fast — it fails after tens of gigabytes of download.
# The live shard names are read from the index JSONs rather than hardcoded.
#
# Why it is ONE entry: `preload_from_hub` is capped at 10 entries and this is 37
# files; one repo with comma-separated filenames fits, a file-per-entry list
# would not (PLAN §6.2). The trailing SHA is the pinned revision — without it the
# preload and the runtime resolve DIFFERENT cache keys and the Space re-downloads
# 72 GB it has already preloaded.
#
# Regenerate with:
#   .venv/bin/python -c "from ltx25_studio import model_files; \
#     print(model_files.preload_from_hub_entry())"
preload_from_hub:
  - Lightricks/LTX-2.5-Diffusers audio_vae/config.json,audio_vae/diffusion_pytorch_model.safetensors,connectors/config.json,connectors/diffusion_pytorch_model-00001-of-00002.safetensors,connectors/diffusion_pytorch_model-00002-of-00002.safetensors,connectors/diffusion_pytorch_model.safetensors.index.json,diffusion_decoder/config.json,diffusion_decoder/diffusion_pytorch_model.safetensors,duration_head/config.json,duration_head/diffusion_pytorch_model.safetensors,latent_upsampler/config.json,latent_upsampler/diffusion_pytorch_model.safetensors,model_index.json,scheduler/scheduler_config.json,temporal_latent_upsampler/config.json,temporal_latent_upsampler/diffusion_pytorch_model.safetensors,text_encoder/config.json,text_encoder/generation_config.json,text_encoder/model-00001-of-00005.safetensors,text_encoder/model-00002-of-00005.safetensors,text_encoder/model-00003-of-00005.safetensors,text_encoder/model-00004-of-00005.safetensors,text_encoder/model-00005-of-00005.safetensors,text_encoder/model.safetensors.index.json,tokenizer/chat_template.jinja,tokenizer/tokenizer.json,tokenizer/tokenizer_config.json,transformer/config.json,transformer/diffusion_pytorch_model-00001-of-00004.safetensors,transformer/diffusion_pytorch_model-00002-of-00004.safetensors,transformer/diffusion_pytorch_model-00003-of-00004.safetensors,transformer/diffusion_pytorch_model-00004-of-00004.safetensors,transformer/diffusion_pytorch_model.safetensors.index.json,vae/config.json,vae/diffusion_pytorch_model.safetensors,vocoder/config.json,vocoder/diffusion_pytorch_model.safetensors 871165de037793df7d75c23a02dd7f45fd364c97
---

# 🎬 LTX-2.5 Studio

A Gradio studio for [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) — Lightricks'
22B joint **video + audio** generation model — covering every mode the model supports,
with **user-uploadable LoRA at inference time**.

Built on `diffusers`, pinned to an exact commit. Deploys to a private Hugging Face Space
on ZeroGPU.

---

## The one thing to understand first

**LTX-2.5 is jointly audio-visual and there is no video-off switch.**

The pipeline returns `LTX2PipelineOutput(frames=..., audio=...)` — both non-optional
tensors. Every call spends full video compute and produces a video, whichever track you
asked for. There is no flag that turns the video half off.

That is the most important cost fact in this app, so the UI states it in words on the
screen where you pick a mode, marks the affected modes with ⚠️ in the selector itself,
repeats it in the mode briefing, and repeats it again after the run finishes. A user who
does not know this reads the runtime as a bug.

For the audio-only modes the app applies the mitigations it can — minimum legal
resolution, stage-2 refinement skipped, only the requested track exported — and it tells
you it is doing so rather than hiding it.

---

## The 11 modes

Eleven modes, but **not eleven code paths**: they are a presentation layer over four real
upstream workflows plus the audio-latent input route and single-track export. The UI is
generated from that registry, so the modes below and the widgets on screen cannot drift
apart.

| # | Mode | Workflow | Exports | Cost |
|---|---|---|---|---|
| 1 | Text → Video | `text2video` | video | first-class |
| 2 | Image → Video | `condition` | video | first-class |
| 3 | Image + Text → Video | `condition` | video | first-class |
| 4 | Video → Video | `condition` | video | first-class |
| 5 | Text → Audio + Video | `text2video` | audio + video | **the model's native shape** |
| 6 | Image → Audio + Video | `condition` | audio + video | first-class |
| 7 | Image + Text → Audio + Video | `condition` | audio + video | first-class |
| 8 | Text → Audio | `text2video` | audio | ⚠️ **still renders video internally** |
| 9 | Video → Audio | `condition` | audio | ⚠️ same |
| 10 | Audio → Audio | `condition` | audio | ⚠️ same |
| 11 | Audio → Video | `condition` | video | ⚠️ pays audio encode on top |

Audio is not a first-class argument — there is no `audio=` kwarg anywhere in the API.
Input audio is encoded to `audio_latents` and mixed with `noise_scale`, which behaves like
img2img strength: `0.0` keeps your audio, `1.0` discards it entirely. The slider says so.

---

## LoRA — the headline feature

Bring your own adapter, by **Hub repo id** or **direct upload**. Both routes are offered
side by side.

The reason this gets so much care: upstream's *only* validation is a substring test that
every LoRA key contains `lora`. An adapter built for a completely different architecture
passes that test, attaches to **zero** modules, and then generation proceeds normally and
produces output that looks like the base model — with no error raised anywhere. Silent
garbage.

So every adapter goes through a staged validator, and the UI tells you exactly which stage
rejected one and why, quoting the validator's own message rather than flattening it to
"invalid file":

| stage | check |
|---|---|
| 1 | size bound — refused before any download |
| 2 | header-only parse — a multi-GB file is **never** materialised just to validate it |
| 3 | format — safetensors with a parseable header (not-safetensors and bad-header are different messages) |
| 4 | version — reads `model_version`, `ss_ltx_version`, `modelspec.architecture`, `ss_base_model_version` in priority order |
| 5 | key convention — on unknown, the error **names the prefixes actually found** |
| 5.5 | in-context adapters — an IC-LoRA on the plain pipeline neither errors nor works |
| 6 | pairing — `lora_A`/`lora_B` (or `lora_down`/`lora_up`) must pair |
| 6.5 | shape fit and target modules — wrong dimension and wrong module are different diagnoses |
| 7 | **attach assertion** — after loading, assert the adapter attached to more than zero modules |

Stage 7 is the one upstream does not do, and it runs on every load.

The panel shows the detected key convention, rank, alpha and `model_version` when present.
**"Not declared" is the normal, expected reading** — of three real public LTX LoRAs
measured for this design, none carried `model_version` and two had no metadata block at
all. Absence warns; it does not reject, because rejecting on absence would reject the most
downloaded LTX adapter on the Hub.

When no adapter is applied, the panel says so in words. It never expresses that as a blank
area — a blank panel and an adapter that loaded but attached to nothing look identical.

---

## Constraints the UI enforces for you

Two hard model constraints, and they fail very differently:

**`width % 32 == 0` and `height % 32 == 0`** — upstream raises. You lose a run, but you
find out.

**`num_frames % 8 == 1`** — upstream does **not** raise. It computes
`(num_frames − 1) // 8 + 1` and **silently floors**. Ask for 120 frames and you get 113,
with no warning, ever. This app's frame hint is the only warning you will get, so it names
the number you would have silently received.

Both show live guidance and offer a **Snap to legal values** button. The app snaps when you
ask it to and shows you the result — it never rewrites what you typed behind your back.

The `8` is not a magic number: it is `vae.temporal_compression_ratio`, used as a documented
pre-load default so the UI can validate before any VAE exists, with a load-time tripwire
that raises if upstream ever changes it.

### Multishot

Offered as a shot list — a prompt segment, an anchor and a strength per shot — and labelled
honestly: **there is no multishot or scene API in LTX-2.5.** It is prompt-structure driven
and unverified, and the 2.5 prompt enhancer's system prompts explicitly instruct the model
*not* to describe scene cuts, so it works against the enhancer rather than with it. It is a
documented technique here, not a button implying support that does not exist.

The anchor is a **latent** index, not a pixel frame number — the conversion is
`pixel ≈ max((latent − 1) × 8 + 1, 0)`, so typing a pixel frame number there is an off-by-8
every time. The UI labels it; `conditioning.py` does the conversion.

---

## Running it locally

```bash
python3.12 -m venv .venv                       # 3.12 is required, not preferred
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e ".[dev]"    # pytest + ruff
.venv/bin/python app.py
```

Tests — no GPU, no weights, no network:

```bash
.venv/bin/python -m pytest -q
```

You need a `HF_TOKEN` with access to `Lightricks/LTX-2.5-Diffusers`; the repo is gated.

Two honest caveats:

- **Python 3.12 is a hard requirement.** numpy 2.5.2 needs ≥ 3.12 and `av` 18 needs ≥ 3.11.
  The default 3.10 image resolves both *downward* rather than failing, which is worse.
- **The 22B model does not run on macOS.** Apple Silicon / MPS is a genuinely different
  regime — no ZeroGPU packer, different memory arithmetic — and the local path is expected
  to be capability-limited rather than identical. Real generations happen on the Space.

`spaces` is deliberately **absent** from `requirements.txt`. The platform pins its own
version and a conflicting pin breaks resolution at build time; the import is guarded in
code so local dev works without it.

---

## ZeroGPU and quota — the actual numbers

Generation runs inside a `@spaces.GPU` reservation on shared, MIG-partitioned hardware.
This is metered, and the meter is the resource you will actually run out of.

| | |
|---|---|
| Hardware | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120), not the A10G the id string implies |
| Tier | **`xlarge`** — forced: packed bf16 weights are 67.23 GiB and the promotion threshold is 42.00 GiB |
| Multiplier | **2×**. A 75-second run costs ~150 GPU-seconds |
| PRO quota | 2400 GPU-seconds/day → **~16 runs/day** |

Two consequences worth knowing as a user:

- **A failed run still costs quota.** So does a timeout.
- **Timeouts are never retried automatically.** The captured request's `X-IP-Token` ages
  past its TTL during the first attempt, so a retry always fails with "Expired ZeroGPU proxy
  token" — masquerading as the root cause and burning a second reservation to tell you
  something false. The app surfaces the original timeout instead.

Stage-2 upscale-and-refine is a real quota cost rather than a free quality win. It is not
yet exposed as a control — `generate.py` does not run a stage-2 pass today, and a checkbox
that changes nothing is the kind of dead toggle this project refuses to ship.

---

## Design documents

The design is written down, with the evidence, before the code:

- [`docs/PLAN.md`](docs/PLAN.md) — the build contract. Integration path and the exact
  pinned SHA (§1), the disk/RAM budget and its two traps (§2), the mode registry (§3),
  the LoRA validator (§4), the resilience doctrine (§5), architecture and layering (§7),
  and the ZeroGPU execution model, which inverts several habits from normal GPU boxes (§8).
- [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) — the quota-bounded GPU matrix. Seven runs cover
  every distinct code path; everything else is a unit test. Budget, stop line and run log.

The governing rule behind both, and behind this app's noisier error messages: the
predecessor's worst outage was not a model failure. A generation failed, four layers each
turned that failure into a falsy value, and the UI rendered an empty path as what looked
like a form quietly resetting itself. No error, no log, no signal. **It hid for a week.**

So: an empty output is an error at the source, every layer propagates it, there is no
"newest file in the output directory" fallback, each run gets its own output directory, and
there is no code path in which you press Generate, wait, and see nothing change.

---

## License

MIT. The LTX-2.5 weights are licensed separately by Lightricks — see the
[model card](https://huggingface.co/Lightricks/LTX-2.5).
