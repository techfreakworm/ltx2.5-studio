# Standing the app up on a real GPU host (hinode)

`wakeuser@hinode` is an SSH box with the **same silicon as ZeroGPU** — RTX PRO 6000
Blackwell Server Edition — but **unpartitioned**, with no quota meter. It exists to
answer one question and it cannot answer the other:

| question | hinode |
|---|---|
| Is the application correct? | **yes, this is what it proves** |
| Does it fit ZeroGPU? | **no — see the gap table below** |

Never report a green run here as "works on the Space". Label every claim
**HOST-PROVEN** or **ZEROGPU-PENDING**.

## What hinode cannot prove

* **Tensor packing.** ZeroGPU intercepts `.to("cuda")`, rewrites it to `.to("cpu")`
  and packs every tensor to `~/.zerogpu/tensors`. Here `.to("cuda")` is a real VRAM
  move. The packed-copy disk profile is untestable on this host.
* **The 150 GB ceiling.** hinode has ~136 GiB free on `/home/wakeuser` and no
  packing double-copy.
* **RAM headroom.** 124 GB here vs ~68 GB empirical on ZeroGPU. A run that fits
  here can still OOM there.
* **`@spaces.GPU`** — the fork boundary, quota metering, `size=` promotion, Space
  secrets. `spaces` is not installed here, so the stub path is what runs.
* **MIG partitioning.** The card here is one unpartitioned 97887 MiB device;
  ZeroGPU serves MIG slices (large=48 GB, xlarge=96 GB).

## Two traps that are not in the machine's specs

1. **`/` and `/tmp` are a 32 G overlay with only ~23 G free.** A uv/pip build tree
   or a download temp file lands there and dies mid-way. `TMPDIR` must be
   redirected under `/home/wakeuser` for **every** session, and verified:
   `python -c "import tempfile;print(tempfile.gettempdir())"`.
2. **`/home/wakeuser/.cache` already holds ~56 GB** belonging to `ComfyUI` and
   `comfyui-supervisor-remote`, which are **not ours**. Check free space before and
   after any download and report both. Do not delete anything you did not create.

## The environment file

Everything lives under `/home/wakeuser`. Nothing may land on the root overlay.
`/home/wakeuser/ltx25-env.sh` is sourced by every session:

```sh
export TMPDIR=/home/wakeuser/tmp
export HF_HOME=/home/wakeuser/.cache/huggingface
export HF_HUB_CACHE=/home/wakeuser/.cache/huggingface/hub
export UV_CACHE_DIR=/home/wakeuser/.cache/uv
export XDG_DATA_HOME=/home/wakeuser/.local/share
export XDG_CACHE_HOME=/home/wakeuser/.cache
export PIP_CACHE_DIR=/home/wakeuser/.cache/pip
export LTX25_STUDIO_DIR=/home/wakeuser/ltx25-studio
export PYTHONPATH=/home/wakeuser/ltx25-studio
```

`HF_HOME` is exported **explicitly** rather than relying on the default — the
default happens to be the same path here, and a setup that works by coincidence
stops working the first time someone runs as another user.

`PYTHONPATH` is set because `ltx25_studio` is not installed into the venv — the
repo is synced, not `pip install -e`'d. Any script run from outside the repo root,
or with `sys.path[0]` pointing at its own directory, fails to
`import ltx25_studio` without it.

## Getting the code onto the box: `git archive`, not rsync

**`rsync` is not installed on hinode and there is no passwordless sudo**, so
rsync-over-ssh cannot work — it needs the binary on both ends. `tar`, `gzip` and
`zstd` are present.

`scripts/sync_to_hinode.sh` ships a **committed ref** over ssh:

```sh
./scripts/sync_to_hinode.sh          # ships HEAD
./scripts/sync_to_hinode.sh <ref>    # ships any committed ref
```

Shipping a committed ref rather than the working tree is the stronger choice
independently of rsync being missing: every measurement taken on the host maps to
exactly one SHA, and a teammate mid-edit in the shared worktree cannot leak WIP
into a recorded run. The script prints a warning when the working tree is dirty so
nobody mistakes what was shipped, and writes the SHA to
`/home/wakeuser/ltx25-studio/.synced_sha` on the box.

Extraction is staged in `$TMPDIR` and then swapped, so a transfer that dies partway
leaves the previous tree intact instead of a half-empty directory. `.venv/` and
`outputs/` live in the destination and are never touched.

## Python

System python is **3.14.4**; the app targets **3.12** (numpy ≥2.5.2 needs ≥3.12).
`uv venv --python 3.12` fetches the toolchain, and with `XDG_DATA_HOME` set it
caches under `/home/wakeuser/.local/share/uv/python` — verified, not assumed.

## The HF token

The repo is gated. The token is passed **via stdin into a launcher that exports it
into the environment**, and never any other way:

```sh
./scripts/hinode_run.sh host_download_weights.py
SMOKE_MODE=image_text_to_video ./scripts/hinode_run.sh host_smoke.py
```

`hinode_run.sh` reads the token from `~/.cache/huggingface/token`, pipes it into
the remote launcher's **stdin**, and the launcher exports it into the environment
of the `nohup`'d process. Detached, because an ssh drop must not kill a 72 GB
download or a multi-minute generation.

* Not in a file in the repo.
* Not in `argv` — `ps` is world-readable on a shared box, so
  `env HF_TOKEN=... python ...` would publish it to every other user.
* The launcher reads it with `IFS= read -r TOK || true`. The `|| true` matters: a
  token file with no trailing newline makes `read` return non-zero at EOF **even
  though it assigned the value**, and `set -e` then kills the script with no
  output at all.

## Torch on Blackwell — HOST-PROVEN, and no pin change needed

`requirements.txt` pins `torch==2.11.0` / `torchvision==0.26.0` /
`torchaudio==2.11.0` with no index directive. The concern was that the default
PyPI wheel might predate cu128 and give
`no kernel image is available for execution on the device` — discoverable only
after a 72 GB download. It does not:

| measurement | value |
|---|---|
| `torch.__version__` | `2.11.0+cu130` |
| `torch.version.cuda` | `13.0` |
| `torch.cuda.get_device_name()` | `NVIDIA RTX PRO 6000 Blackwell Server Edition` |
| `torch.cuda.get_device_capability()` | `(12, 0)` |
| `torch.cuda.get_arch_list()` | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` |
| real bf16 4096² matmul | launched, finite result |
| real bf16 `Conv3d` | launched, finite result |
| `torchaudio` / `torchvision` | `2.11.0+cu130` / `0.26.0+cu130` |

Default PyPI ships **cu130**, which is *newer* than the cu128 floor sm_120 needs,
and `sm_120` is in the compiled arch list. **No `--index-url` override is
required, and `requirements.txt` needs no index directive for non-Space hosts.**

One wording note: the `requirements.txt` comment says the trio is "CUDA 12.8".
That is what ZeroGPU ships; the same pin resolves to cu130 from PyPI. Both satisfy
`cu128+`, so the pin is portable — the comment is just narrower than reality.

A version string is not proof of a working kernel. Always launch one.

## The weight download — HOST-PROVEN

Run through the app's own resolver (`scripts/host_download_weights.py` →
`preflight.ensure_weights` → `model_files.download_allow_list()`), not a
hand-written file list. Running the real resolver on a second host is what
independently proves it.

| measurement | value |
|---|---|
| files | **37 / 37** |
| wall clock | **774.2 s (12.90 min)**, ~93 MB/s |
| free before | 140,107,329,536 B (140.107 GB) |
| free after | 67,920,973,824 B (67.921 GB) |
| **consumed** | **72,186,355,712 B (72.186 GB)** |
| resolver's prediction | 72.185 GB — agreement to ~1 MB |
| `missing_cached_files()` | 0 |
| `require_cache_complete()` | PASS |

Free space on `/home/wakeuser` was **145,616,388,096 B (145.616 GB)** before any
setup; the venv and the torch trio account for the ~5.5 GB between that and the
pre-download figure. The shared `.cache` went from ~56 GB to 110 GiB total, of
which 68 GiB is our snapshot.

**The cache is provably complete for construction.** Every load since has run with
`HF_HUB_OFFLINE=1`, and the pipeline constructs entirely from cache — the Hub is
unreachable and it still works:

```
Couldn't connect to the Hub: ... offline mode is enabled ... Will try to load from local cache.
```

That is the permission-bug class from PLAN §6.2 eliminated by measurement rather
than by argument: if nothing is ever fetched lazily, a lazy fetch cannot hit a
read-only cache.

Two notes about that shared directory. It holds ~20 GB of stale
`models--ByteDance--Sa2VA-*/blobs/*.incomplete` files belonging to another
project — **left alone**. And the `xet` chunk cache stayed negligible (2.5 MB), so
`hf_xet` added no meaningful disk here.

## What one loaded copy costs — HOST-PROVEN

`LTX2ConditionPipeline` (`BASE_KIND`), bf16, moved to the card:

| | |
|---|---|
| `torch.cuda.memory_allocated()` | 70,190,959,616 B (**65.370 GiB**) |
| `torch.cuda.memory_reserved()` | 66.607 GiB |
| `nvidia-smi` used | 69,810 MiB |
| peak host RSS | **20.05 GiB** only — safetensors is mmap-backed |
| components on `cuda:0` | all 7 weight-bearing ones |

Per component (parameters + buffers), summing to 65.249 GiB:

| component | GiB |
|---|---|
| `transformer` | 35.368 |
| `text_encoder` | 22.277 |
| `connectors` | 5.909 |
| `vae` | 1.352 |
| `vocoder` | 0.240 |
| `audio_vae` | 0.099 |
| `duration_head` | 0.004 |

Peak during a 512×320×25 generation was **68.879 GiB** — roughly 3.5 GiB of
activations on top of the weights at that size.

**ZEROGPU-PENDING inference, not a host result:** 65.370 GiB of weights cannot fit
a `large` (48 GB) MIG slice, so `xlarge` promotion is structural rather than
optional. The footprint is measured; the slice behaviour is not testable here.

## `derive_pipeline` and `from_pipe` — the two-copy trap

`get_pipeline(pipeline="text2video", device="cuda")` **OOMs the card** unless the
dtype is passed to `from_pipe`. The cause is in diffusers:

```
pipeline_utils.py:2155   dtype = _resolve_dtype(kwargs.pop("dtype", None), torch_dtype) or torch.float32
pipeline_utils.py:2253   new_pipeline.to(dtype=dtype)
```

With no dtype, `from_pipe` defaults to **float32** and casts every shared module.
`t.to(float32)` on bf16 allocates new tensors at twice the bytes, so the "derived"
pipeline is a second copy — and because `nn.Module.to()` mutates in place and the
modules really are shared, **the base pipeline would silently become fp32 too** if
it ever fit. Passing the base's own dtype makes every `t.to()` a no-op returning
`self`:

| `from_pipe(base, dtype=base.dtype)` | measured |
|---|---|
| components shared by identity | **9 / 9** |
| cuda allocated delta | **−1024 B** (allocator bookkeeping) |
| total for BOTH pipelines | 65.370 GiB |
| `base.dtype` afterwards | still `torch.bfloat16` |
| `only in base` | `['audio_scheduler']` — as `derive_pipeline` predicts |

`scripts/host_probe_from_pipe.py` asserts all three properties. It compares the
delta against a 1 MiB tolerance rather than zero: a real second copy is tens of
GiB, while the allocator's own bookkeeping moves by a KiB in either direction, and
a check that fails on noise teaches its reader to ignore it.

## Stage 2 is unreachable — HOST-PROVEN

`get_pipeline` always builds `StageRunner(built, kind=pipeline)`, so `upsampler`
and `decoder` stay `None`. Any `enable_stage_2=True` request raises:

```
ModelLoadError: failed to load pipeline component 'latent_upsampler': stage 2 was
requested but the latent upsampler or the diffusion decoder was not loaded.
Refusing to silently fall back to a single-stage render, which would look like it
worked and be visibly worse.
```

An honest refusal rather than a silent downgrade — but the headline stage-2 quality
path (PLAN §7.4) has nothing behind it yet. This is wiring, not a missing
dependency: `latent_upsampler` and `temporal_latent_upsampler` are downloaded, and
`LTX2LatentUpsamplePipeline` / `LTX2VideoDiffusionDecodePipeline` both import.
Note also that `model_index.json` declares `diffusion_decoder` while
`LTX2ConditionPipeline` rejects it — *"config attributes {'diffusion_decoder': …}
were passed … but are not expected and will be ignored"* — so it must be built
separately.

## First real generation — HOST-PROVEN

`SMOKE_MODE=image_text_to_video SMOKE_W=512 SMOKE_H=320 SMOKE_FRAMES=25`. The
`condition` pipeline with a prompt and no image attached — `_conditions()` returns
empty, so this is text-to-video through the base workflow, chosen because
`text_to_video` routes to `text2video` and is blocked by the `from_pipe` trap
above.

```
pipeline loaded and moved to cuda in 14.1s
cuda allocated 65.370 GiB  reserved 66.607 GiB
ASSERT every weight-bearing component is on cuda: PASS
=== GENERATION COMPLETED in 5.3s ===
output_path : outputs/run-e8872ced78e4/output.mp4      93,351 bytes
peak cuda   : 68.879 GiB
container   : mov,mp4,m4a,3gp,3g2,mj2 (QuickTime / MOV)   duration 1.042 s
  [0] type=video codec=h264  512x320 pix_fmt=yuv420p avg_rate=24 declared_frames=25
DECODED     : {'video': 25}
```

25 frames declared **and 25 decoded** — the header is not the check. 8 distilled
steps in 5.3 s.

The dropped-audio warning fired and reached the result, which is PLAN §5.3
working rather than incidental: the model is jointly audio-visual with no video-off
switch, so a video-only mode renders audio and pays for it either way, and the user
is told instead of silently losing it.

## Part A results — HOST-PROVEN

Baseline **704 × 480 × 121**, seed pinned to **1234**, one model load per batch
(13.8 s, ~68 GiB off disk). `verify_streams` exit 0 = MATCHES throughout; exit 2
would be "could not probe" and is not a pass.

| # | mode | generate | bytes | streams observed | verify |
|---|---|---|---|---|---|
| H1 | `text_to_video` | 25.7 s | 865,447 | h264 704x480@24, **no audio** | 0 |
| H2 | `image_to_video` | 26.3 s | 577,051 | h264 704x480@24, **no audio** | 0 |
| H3 | `image_text_to_video` | 26.6 s | 1,040,007 | h264 704x480@24, **no audio** | 0 |
| H4 | `video_to_video` (strength 0.7) | 27.9 s | 726,998 | h264 704x480@24, **no audio** | 0 |
| H5 | `text_to_audio_video` | 25.8 s | 1,017,916 | h264 + **aac 2ch 48000** | 0 |
| H6 | `image_to_audio_video` | 26.5 s | 707,694 | h264 + **aac 2ch 48000** | 0 |
| H7 | `image_text_to_audio_video` | 26.7 s | 926,923 | h264 + **aac 2ch 48000** | 0 |
| H8 | `text_to_audio` | 5.5 s | 961,998 | **pcm_s16le 2ch 48000, no video** | 0 |
| H9 | `video_to_audio` | 6.0 s | 961,998 | **pcm_s16le 2ch 48000, no video** | 0 |
| H15 | duration_head | 20.2 s | 754,150 | h264, **97 frames / 4.042 s** | 0 |
| H16 | `text_to_audio_video` 1216×704 | 68.9 s | 2,075,143 | h264 1216x704 + aac 2ch | 0 |

**H1–H4 carry no audio stream and H8/H9 carry no video stream**, so `exports`
decides container contents in *both* directions — the direction that has been weak
here historically is the undeclared-stream-present one, and it holds.

**H15 is checked on the property the stream verifier cannot see.** 97 frames /
4.042 s against H1's 121 / 5.042 s from the same prompt and seed: inside the
2–6 s bounds, and 97 = 12·8+1, a legal count. A stream check alone would have
passed a mode that silently ignored `auto_duration`.

Peak VRAM: 68.879 GiB (audio-only, 256²) → 70.5 GiB (baseline) → 71.008 GiB
(video conditioning) → **76.653 GiB (1216×704)**. Against 94.97 GiB usable, the
high-water mark leaves ~18 GiB.

### The LoRA validator — both directions

| # | adapter | outcome |
|---|---|---|
| H12 | `joyfox/LTX-2.3-Transition-LORA` | **ACCEPTED**, stage 7 attached **576 modules** |
| H13a | `charlied98/…/lora_weights_step_03000` | **REJECTED at stage 6.5**, key `transformer.…` |
| H13b | `charlied98/…/comfy_lora_weights_step_03000` | **REJECTED at stage 6.5**, key `diffusion_model.…` |
| H12c | `AhsanHareem/reelbids-ltx25-camera-lora` | ACCEPTED, 384 modules (verdict unspecified) |

```
LoRADimensionError: 2048 at transformer_blocks.0.attn1.to_k, expected 4096
```

**H13a and H13b reach the same stage by the same mechanism**, differing only in key
prefix. That is the check worth making: two conventions normalised into one
dimension comparison, rather than one refused early at stage 5 for its convention
and the other at 6.5 — two mechanisms producing one verdict reads exactly like a
pass. H12 passing is the false-positive guard: its mixed dims (4096 attention,
2048 `audio_ff`, 16384 `ff.net.2`) are legitimate and did not trip 6.5.

`connectors: 0` on every accepted adapter. Stage 7 asserts *total* attached > 0, so
it passes on the transformer alone — probably correct for these adapters, but the
assertion cannot distinguish "connectors legitimately unused" from "connectors
silently missed".

### What is NOT proven

* **`get_pipeline` as `app.py` calls it.** Every run above used
  `MATRIX_BYPASS_STAGE2=1` — `_base_pipeline` plus a `StageRunner` injected through
  `generate(pipeline_factory=...)`, the app's own seam. The real path cannot load
  offline (below), so the whole set must be re-run once that is fixed.
* **H10 / H11** — `audio.load_waveform` calls `torchaudio.load`, which at pinned
  torchaudio 2.11.0 dispatches to TorchCodec and raises
  `ImportError: TorchCodec is required for load_with_torchcodec`. Installing
  `torchcodec` does not fix it: it loads system FFmpeg shared libraries
  (`libavutil.so.56`) that are absent, with no sudo to add them. **`av` is already
  a required pin and bundles its own FFmpeg**, and reads the file fine — note `s16`
  is packed, so a frame is `(1, n*channels)` interleaved and needs de-interleaving
  to the `(channels, samples)` layout `encode_waveform_to_latents` expects.
* **H14 / stage 2** — blocked by the conflict below.
* **"Output looks right?"** Nothing here establishes that the video looks good or
  the audio sounds right. Streams, frame counts, rates and channel layouts are
  verified; aesthetic correctness is not, and it is where every confidently-wrong
  artifact on this build came from.

## The offline/NATTEN deadlock

Two requirements that cannot both be met as the code stands:

* **The model load needs `HF_HUB_OFFLINE=1`.** Online, `from_pretrained` fetches
  the full 48-file snapshot and pulls the 44.33 GB of dead shards PLAN §2.1 exists
  to avoid.
* **Stage 2 needs the Hub.** `get_kernel("shi-labs/natten", version=1)` resolves
  version → ref through the Hub (`kernels/_versions.py:72`). Pre-downloading the
  kernel is not enough — the *resolution* is not cached:

```
ValueError: Version 1 of 'shi-labs/natten' is not available in the local cache
and Hugging Face Hub is in offline mode.
```

Because `_stage_2_parts` is called unconditionally from `get_pipeline`, **nothing
loads at all**, including the eight modes that never touch stage 2.

Also note the offline path works only while `trees/<revision>.json` is absent from
the cache. Once written, the completeness check refuses:

> *the cached snapshot … is incomplete: 14 file(s) are missing*

The cached manifest lists **56 files, 164.19 GB** — the whole repo does not fit
ZeroGPU's 150 GB cap with nothing else on disk, so a "complete snapshot" is
unreachable for us by design and our 37-file subset is the right answer. That
manifest is parked at
`/home/wakeuser/ltx25-logs/saved-metadata/tree-871165de.json`.

**Loading from the local snapshot directory** —
`preflight.cache_repo_dir() / "snapshots" / REVISION` — bypasses hub resolution,
the tree manifest, the completeness check and lazy fetching at once, and lets the
network stay up so NATTEN resolves. `local_files_only=True` does the opposite: it
is what *triggers* the strict check.

## Final state — everything through the real `get_pipeline` path

The whole matrix re-ran with **no `HF_HUB_OFFLINE`, no bypass, and
`trees/<revision>.json` restored to the cache**. That last detail is the one worth
keeping: the load previously worked only while that file was *absent*, and it now
works with it present, so the accident is gone rather than avoided.

```
load 14.0s (one-off)
H1  text_to_video             25.7s  verify=0     H9  video_to_audio      6.0s  verify=0
H2  image_to_video            26.3s  verify=0     H14 stage 2            84.3s  verify=0
H3  image_text_to_video       26.6s  verify=0     H15 duration_head      19.4s  verify=0
H4  video_to_video            27.0s  verify=0     H16 1216x704           68.7s  verify=0
H5  text_to_audio_video       25.0s  verify=0
H6  image_to_audio_video      26.6s  verify=0
H7  image_text_to_audio_video 26.7s  verify=0
H8  text_to_audio              5.6s  verify=0
```

Peak VRAM by workload: 68.879 GiB (audio-only) → ~70.5 (baseline) → 71.008 (video
conditioning) → 76.653 (1216×704) → **83.897 GiB (stage 2)**. Against 94.97 GiB
usable, the high-water mark leaves ~11 GiB.

### Stage 2 changes the output geometry

A 704 × 480 request with `enable_stage_2=True` produces a **1408 × 960** artifact —
the latent upsampler doubles both axes, which is what stage 2 is. Correct, but
`verify_streams` passes because the expected layout for `text_to_video` does not
assert resolution, so **the size the user asked for is not the size they get and
nothing says so.**

### NATTEN resolves once per process, not per generation

Two stage-2 generations in one process (85.0 s, 83.4 s, different seeds) produced
**exactly one** kernel fetch. `get_kernel("shi-labs/natten", version=1)` resolves
version → ref over the network even with the kernel fully cached, and `version=1`
is hardcoded inside diffusers so it is not ours to pin. Consequence: a worker needs
outbound network at its **first** stage-2 use, not at every generation — and
`_stage_2_parts` caches into `_CACHE`, so each cold worker pays it once.

## Two open product defects the stream checks could not see

### `video_to_audio` returns near-silence — and it is model-side

`scripts/host_audio_levels.py` over every artifact:

| artifact | mode | peak | rms | dBFS |
|---|---|---|---|---|
| `run-781803794dad/output.wav` | **video_to_audio** | 0.0146 | 0.001779 | **−55.0** |
| `run-c1b06b0cf577/output.wav` | **video_to_audio** | 0.0128 | 0.001880 | **−54.5** |
| `run-b5f1517ba8f0/output.wav` | text_to_audio | 1.0000 | 0.549740 | −5.2 |
| `run-ffb3ae939601/output.wav` | text_to_audio | 1.0000 | 0.546838 | −5.2 |

Localised by measuring the tensor **before** encode via `generate(encoder=...)`:

```
text_to_audio  : audio (1, 2, 240480) bf16  min -1.000000  max 1.000000  rms 0.541253
video_to_audio : audio (1, 2, 240480) bf16  min -0.002319  max 0.002411  rms 0.000370
```

The tensor is already near-silent, so **`outputs.py` is faithful and exonerated**;
the audio branch produces almost nothing when a video condition is present. Three
reproducible runs. `verify_streams` passes all of them — correct codec, 2 channels,
48000 Hz, no video stream — which is why this needed a level measurement rather
than a structural one. A user picking *"Video to audio"* gets a perfect, inaudible
file.

### The audio-input modes cannot start: no mel front-end exists

`load_waveform` works (via `av`). The next step does not:

> *the loaded audio_vae carries no mel front-end (looked for `mel_spectrogram`,
> `mel_transform`, `_mel_spectrogram`) … Pass `to_mel=` explicitly.*

The real `AutoencoderKLLTX2Audio` exposes only `mel_compression_ratio` (an int),
and **upstream ships no mel front-end anywhere** in `diffusers.pipelines.ltx2` —
all 16 modules enumerated, `utils` has no mel members, and the audio-VAE module
source has zero mel/stft/hann/n_fft references.

**But the window size is declared — by the vocoder, not the audio VAE:**

| `audio_vae/config.json` | `vocoder/config.json` |
|---|---|
| `mel_bins = 64` | **`window_length = 512`** |
| `mel_hop_length = 160` | `hop_length = 80` |
| `sample_rate = 16000` | `num_mel_channels = 64` |
| `in_channels = 2` (stereo required) | `input_sampling_rate = 16000` |
| `latent_channels = 8` | `output_sampling_rate = 48000` |

`mel_front_end_for` looks only at the `audio_vae` and concludes the parameters do
not exist. **One caution:** the hops disagree by 2× (160 vs 80), and the vocoder's
`upsample_factors` multiply to exactly 160 — so reading `window_length` across is
defensible, reading a hop across is not, and the VAE's own `mel_hop_length=160` is
the value for the forward transform.

Note the input path is **16 kHz** even though every artifact written is 48 kHz, and
that `in_channels = 2` means **the audio VAE requires stereo** — a mono upload
would otherwise reach `conv_in` and raise a raw tensor-shape `RuntimeError` from
inside diffusers.

**Why output works and input does not:** `decode_latents_to_waveform` goes latents →
mel → vocoder and needs no STFT. Only the forward direction needs an analysis
window. Not a symmetric gap.

## One conclusion I got wrong, recorded because the method matters

I reported that the audio export clipped, reasoning from PCM wavs peaking at
exactly 1.0000 — "the signature of hard clipping at the int16 boundary". The tensor
showed `min -1.000000 / max 1.000000` with **zero** samples exceeding |1.0|: the
model's own output is already saturated at full scale and the write is faithful.
The aac containers decoding to 1.78 are lossy-codec overshoot on full-scale input,
not something our path introduces.

I read a property off a finished file and attributed a cause. The tensor was one
cheap run away and it reversed the conclusion. **Measure at the layer that owns the
value, not at the artifact.**

## Harness constructs that were wrong in the same way three times

- `pkill -f host_matrix.py` matched the ssh command line **carrying the pattern**
  and killed the session issuing it. Use `pgrep -af "host_matri[x]"` or a PID.
- `until grep -q "SUMMARY"; do sleep; done` has no failure exit, so a crash becomes
  an indefinite hang indistinguishable from slow progress. Every wait needs three:
  success, **the process no longer existing**, and a timeout.
- Fixture *absence* was non-fatal while a fixture that *raised* still cancelled the
  batch, so a shape bug in the audio helper took down two modes that never wanted
  audio.

Each guarded against the failure that was imagined rather than the failures
available. **A harness construct validated only against success is not validated.**

## `video_to_audio` near-silence — cause found: the condition strength default

Sweeping the one lever, same seed 1234 throughout:

| `strength` | rms | dBFS | |
|---|---|---|---|
| 0.99 | 0.011720 | −38.6 | near-silent |
| **0.80** — the default that produced H9 | 0.001779 | **−55.0** | near-silent |
| **0.50** | 0.556122 | **−5.1** | **healthy, matches text-only** |
| 0.20 | 0.264710 | −11.5 | healthy |
| 0.05 | 0.218849 | −13.2 | healthy |

**The culprit is `GenerationRequest.strength = 0.8`, not `MAX_VIDEO_CONDITION_STRENGTH`.**
`modes._conditions` computes `min(request.strength, MAX_VIDEO_CONDITION_STRENGTH)`, so
with strength defaulting to 0.8 the 0.99 ceiling never binds — lowering that
constant would change nothing.

Two things to hold when choosing a fix:

* **It is not monotonic.** 0.8 gives −55.0 while 0.99 gives −38.6, and 0.5 is louder
  than 0.2. There is a cliff between 0.5 and 0.8, not a gradient, so a new default
  should not be interpolated from these points.
* **H4 `video_to_video` at strength 0.7 produced good video.** Strong conditioning is
  exactly what video-to-video fidelity wants and what ruins the audio branch, so one
  global default cannot serve both. A per-mode condition strength is the shape that
  fits.

Corroborating, and free — the levels sort by **condition type**, not by mode:

| condition | dBFS |
|---|---|
| none (text only) — H8, H5, H16 | −5.2, −5.2, −5.9, −6.0, −6.1, −6.2 |
| image, one frame at index 0 — H6, H7 | −8.2, −12.1, −12.9, −15.2 |
| video, all 121 frames at 0.8 — H9 | −54.5, −55.0 |

Which also rules something out: all three joint audio+video modes ran with the same
SFT guidance defaults (`audio_guidance_scale` at 7.0 rather than the distilled 1.0)
and H5 came out at −6.0. **That deviation is real and worth fixing on its own
merits, but it is not what silences H9.** The image-condition modes sitting 3–9 dB
below text-only is the same effect, much milder — one anchored frame against 121.

## Output quality sign-off, and the one exception

The operator reviewed the artifacts and signed off: *"visuals are good, audios are
good. All verified visually."* That closes the *"output looks right?"* column for
prompt adherence, motion coherence and audio correctness across the real-path
modes, the 1216×704 flagship, and both audio-only modes — and it validates the
16000/48000 rate handling **by ear**, which is the strongest form that claim could
take.

**One exception, recorded rather than reconciled:**

| | |
|---|---|
| **CONFIRMED — OPERATOR-VERIFIED** | the five healthy audio artifacts and every video mode |
| **PENDING — not operator-verified** | H9 `video_to_audio` |

The sign-off predates the dBFS measurements, and H9 sat among five healthy
artifacts in that gallery. At −55 dBFS with a 1.46% peak it is effectively
inaudible, and **near-silence in one card among twenty-five does not announce
itself** — so it is likelier it went unregistered than that it was judged fine.
Both facts belong side by side: signed off by ear, and measured 50 dB below its
siblings. A human verdict is worth what it was actually applied to.

## The pre-guidance-fix baseline

Every figure in this document is from the current pins with the **SFT guidance
defaults in force** — `stg_scale 1.0`, `modality_scale 3.0`,
`audio_guidance_scale 7.0`, `audio_stg_scale 1.0`, `audio_modality_scale 3.0` —
and with `num_inference_steps=8` and no `sigmas`, so **only H14 used the distilled
schedule**. When the distilled-guidance fix lands, levels and timings should be
expected to move and this table is the *before*.

## Provenance of the H9 condition clip

Recorded because it was briefly unrecoverable, and because it changes what a human
reviewer would be judging.

```
/home/wakeuser/ltx25-studio/outputs/run-938787102035/output.mp4
  704 x 480, 121 frames, h264 yuv420p, 24 fps, 865,447 bytes
  streams: [('video', 'h264')]        HAS AUDIO STREAM: False
```

* It is **H1's own generated output** — `text_to_video`, sports-car sunset prompt,
  seed 1234, from the first (bypassed) batch. **A generated clip, not real footage.**
* **H4 and H9 shared it**, and both H9 runs (`run-781803794dad`,
  `run-c1b06b0cf577`) used this same file.
* It carries **no audio stream**, and in any case **only the frames reach the
  model** — `load_video_frames` decodes to PIL images, so the condition is purely
  visual by construction.

So *"does the generated audio match the video that conditioned it?"* would be asked
of a synthetic, silent, model-generated clip. That is a weaker test than real
footage: the visual content is itself a model artifact, so a poor audio match could
reflect the source rather than the conditioning path. **The strong version of that
test needs real footage with sustained audio-relevant motion**, which is not on the
box.

## "Very quiet", not "near-silent"

The −55 dBFS figure survived every check: per-channel (ch0 −55.0, ch1 −55.0, so not
a summing artifact), both H9 files (−55.0 and −54.5), correct `int16 ÷ 32768`
normalisation (the same code yields `peak 1.000000` exactly for H8, which a broken
normalisation could not), an independent tool agreeing at int16 RMS **58.3**, and a
tensor measurement upstream of the file entirely.

**But −55 dBFS is faint, not inaudible.** Turned up it is plainly audible, so a
listener reporting "not silent" and a meter reporting "50 dB below its siblings" are
both right. The flag in `host_audio_levels.py` now reads **VERY-QUIET** rather than
NEAR-SILENT, because the stronger word invited a disagreement the measurement never
supported.

What the measurement does establish is narrower and still decisive: **the level is
not a property of the input.** Same mode, same silent generated clip, same seed,
only `strength` changed — 0.8 gives −55.0 dBFS and 0.5 gives −5.1. The mode is
capable of a healthy level from that exact condition.

One discrepancy kept rather than smoothed: the tensor probe read `rms 0.000370`
while the files read `0.001779`, about 4.8×. Different runs of the same mode, so
run-to-run variation rather than a conflict between methods. Both are ~50 dB down.

## Run provenance is now durable

Every run previously wrote `<script>.log` and the next run destroyed it, which is
how the fixture behind a recorded result became unreadable. Logs are timestamped,
and each launch appends the SHA, script and forwarded environment to an append-only
`/home/wakeuser/ltx25-logs/RUNS.manifest`; a symlink keeps the stable name for
polling. It records the **inputs** rather than the output, because that was the
irrecoverable half — a result whose inputs are unknown is not a measurement.

---

# Corrected results — after the distilled-recipe fix (`d513de5`)

**Everything above this line is the pre-fix baseline and is superseded.** Before
`d513de5`, `DISTILLED_SIGMA_VALUES` reached the pipeline **only when
`enable_stage_2` was set**, so H14 was the one run of twelve that used the distilled
schedule; the rest got a generic linear fallback plus five guidance parameters left
at upstream's **SFT** defaults (`audio_guidance_scale` 7.0 against 1.0,
`modality_scale` 3.0 against 1.0, both STG scales 1.0 against 0.0).

## The `video_to_audio` anomaly was never its own defect

| | BEFORE | AFTER |
|---|---|---|
| **H9** `video_to_audio` | **−55.0 dBFS**, peak 0.0146, \|L-R\| 3.07% | **−17.8 dBFS**, peak 0.5078, \|L-R\| **19.22%** |
| **H8** `text_to_audio` | −5.2 dBFS, **peak 1.0000** (railed) | −15.3 dBFS, peak 0.6953 |
| gap | **~50 dB** | **2.5 dB** |

One mode 50 dB down and another driven into the rail were **the same
misconfiguration seen from opposite ends**. H8 was never a valid control: a peak of
exactly 1.0000 is a signal against the rail, not a healthy reference.

### ⚠️ The strength attribution recorded above is WRONG

The earlier section concludes *"the culprit is `GenerationRequest.strength = 0.8`"*.
**It is not.** The corrected H9 ran at the **unchanged default 0.8** and came out
healthy. **No change to `strength` or `MAX_VIDEO_CONDITION_STRENGTH` is warranted.**

The sweep measured something real — 0.8 → −55.0 and 0.5 → −5.1 at fixed seed, one
lever — but that was an **interaction between condition strength and the broken SFT
guidance**, not a root cause. The pipeline source says the conditioning mask is
applied to `noise_pred_video` only and never touches `noise_pred_audio`, so the
video condition *cannot* suppress audio directly; that citation was correct and this
measurement did not contradict it.

**The lesson is the transferable part: a single-lever same-seed sweep is strong
evidence that something moves and no evidence about why.** Treating a clean
correlation as a mechanism, against a source citation, is the error — and it is the
same shape as reading a property off a finished file and inferring a cause.

## BEFORE / AFTER pairs

| # | mode | BEFORE | AFTER | pixel mean\|diff\| |
|---|---|---|---|---|
| H1 | `text_to_video` | `run-691abb4502d1` | `run-d6b415a8d334` | 53.54 |
| H2 | `image_to_video` | `run-d761a8862653` | `run-f2a85282c802` | 15.36 |
| H3 | `image_text_to_video` | `run-f8bc1f282edc` | `run-58a3e6db6812` | 55.27 |
| H4 | `video_to_video` | `run-caf43b631aa2` | `run-10ac6e0ca315` | **3.71** |
| H5 | `text_to_audio_video` | `run-7fceeaf2175d` | `run-8619ac8155f8` | 56.50 |
| H6 | `image_to_audio_video` | `run-ea3906db261d` | `run-5e1867c7a94b` | 16.40 |
| H7 | `image_text_to_audio_video` | `run-8bcbf7ff85ca` | `run-d41a0cee3225` | 58.11 |
| H8 | `text_to_audio` | `run-b5f1517ba8f0` | `run-46eab07d0c7e` | audio only |
| H9 | `video_to_audio` | `run-781803794dad` | `run-89bf420a98b1` | audio only |
| H14 | stage 2 | `run-4a2412a9522c` | `run-8fa1e765bc47` | 58.81 |
| H15 | duration_head | `run-bca114fb639d` | `run-2ede72c48af5` | 39.97 |
| H16 | 1216×704 | `run-96e42dab977a` | `run-0dd906803af1` | 36.68 |

**All twelve moved. Nothing silently survived the defect** — which is the finding, and
it would have been lost by reporting only the differences.

**The magnitudes order by conditioning strength**, which is why they are coherent
rather than noise: text-only and image+text moved most (37–59), image-conditioned
moderately (15–16), and **video-conditioned least at 3.71** — the strongest condition
pins the output hardest and leaves the guidance least room to move it.

**Quiet corroboration that the anchor conversion is right:** on every
image-conditioned mode **frame 0 barely moved** — H2 0.65, H3 0.85, H6 0.73,
H7 0.78 — while later frames moved 15–77. Frame 0 *is* the condition anchor at
`index=0`. It stayed put while everything downstream changed.

## Corrected audio, and the clipping was a symptom after all

```
peaks BEFORE : 1.7811  1.6452  1.4017  1.3557  1.1535  1.0054  1.0024  1.0000
peaks AFTER  : 1.0050  0.9693  0.8778  0.7088  0.6953  0.5078
levels AFTER : -9.5 to -17.8 dBFS   (was -5.2 to -55.0)
```

The earlier withdrawal of the clipping claim was **half right**. Our path never added
gain — but material sitting at ±1.0 was itself the defect. With the correct recipe
nothing reaches the rail, **so no limiter is needed** and `_write_audio`'s "what comes
back is what the vocoder produced" principle stays intact.

## ~3× faster, and the flagship is now 21 seconds

| | before | after |
|---|---|---|
| H1 | 25.7 s | **8.4 s** |
| H8 | 5.6 s | **1.8 s** |
| H14 stage 2 | 84.3 s | **35.2 s** |
| H16 **1216×704 + stereo** | 68.7 s | **21.0 s** |

The distilled recipe runs unguided, dropping the extra forward passes SFT guidance
required. VRAM essentially unchanged: H14 peak **83.899 GiB**, H16 **77.586 GiB**.

## The H14 brightness falloff was the defect, not stage 2

```
BEFORE  frame 0 → 120 : mean 99.48 → 34.44   std 63.16 → 26.07   (collapsing)
AFTER   frame 0 → 120 : mean 82.52 → 110.47  std 85.40 → 89.80   (stable)
```

Flagged earlier as needing a human eye; it no longer does. `std` now sits ~85–90 on
every mode against 50–60 before — materially more contrast and detail.

## H17 — the mel round-trip, and why H10/H11 passing would not have been enough

The forward mel does not exist in the checkpoint, so `build_mel_front_end`
constructs it. **Five parameters are read; the rest are chosen:**

| read from the checkpoint | | convention, declared nowhere |
|---|---|---|
| `n_fft` 512 | `vocoder.filter_length` | `power` 1.0 |
| `win_length` 512 | `vocoder.window_length` | `f_min` 0.0 / `f_max` None |
| `hop_length` 160 | `audio_vae.mel_hop_length` | `mel_scale` / `norm` slaney |
| `n_mels` 64 | `audio_vae.mel_bins` | `center` True |
| `sample_rate` 16000 | `audio_vae.sample_rate` | log floor 1e-05 |

Nothing offline can validate the right column, and a wrong choice there is
*"audible as degraded audio, reported as nothing"*. **H10/H11 will produce
well-formed audio through a wrong analysis window** — so they are not evidence the
mel is right, and the round trip is what is.

**Stage 1 passes.** `mel (2, 64, 401) float32, min −11.5129, max −0.0854,
mean −8.3448` — real dynamic range rather than a constant, and `min` is exactly
`log(1e-05)`, so the floor engages at the quiet end as intended.

### The fixture, and why it is tones

Two tones an octave apart per channel plus an amplitude envelope — **not noise and
not silence.** A wrong analysis window smears narrow spectral peaks, so tones are
what make the damage measurable; broadband noise has no peaks to smear and silence
has no content, and both reconstruct to something that looks fine. Same trap as a
`float32` fixture over a `bfloat16` defect.

Confirmed working from stage 1: energy lands at 203 / 219 / 234 / 1312 Hz on ch0
(tones at 220 and 1320) and 422 / 438 / 453 / 2641 Hz on ch1 (tones at 440 and
2640). The spread across adjacent bins is the analysis resolution, not the signal.

Comparison is **spectral, not sample-wise**: the vocoder does not preserve phase and
the round trip is lossy by design, so a sample-wise difference would fail on a
correct implementation. Each side is measured at **its own rate** — 16000 in, 48000
out — so the resampler does not end up inside the result. Reporting is per stage,
because a fault in the mel, the VAE or the vocoder all look identical at the output.

### ⛔ Blocked at stage 2 by a dtype boundary

```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
  autoencoder_kl_ltx2_audio.py:446  forward -> self.conv_in(hidden_states)
```

`to_mel` computes in float32 — **correct**, an STFT in bf16 would lose precision
exactly where the log floor matters — and returns float32, while the VAE is bf16 and
`encode_waveform_to_latents` casts nothing before `audio_vae.encode`. One cast fixes
it; the better home is arguably that call site rather than inside `to_mel`, since it
already owns the handoff and would also protect an explicitly-passed `to_mel=` from
a caller who does not know the VAE's dtype.

## Three dtype-boundary defects, one shape

1. `from_pipe` defaulting to fp32 and silently recasting the shared bf16 modules.
2. `_as_array` calling `.numpy()` on a bf16 tensor.
3. A float32 mel handed to a bf16 conv.

Each sits where a value crosses between our code and something with its own opinion
about precision, and **each was invisible to the offline gate because the fixtures
are float32 and float32 is the dtype that works.** Not a criticism of the gate: a
bf16 build needs at least one genuinely bf16 fixture, or this family recurs one
boundary at a time.

### H17 result — FAIL, and not on the conventions

Four structural defects sit in front of the conventions and mask them completely.

**1. The dtype cast is necessary AND sufficient.** Wrapping `build_mel_front_end`
with `.to(dtype=vae.dtype)` through the documented `to_mel=` seam gets
`audio_vae.encode` running. Verified rather than assumed, so the fix ships already
tested.

**2. `AUDIO_LATENT_DIMS` is transposed.** Proven by halving the input duration and
watching which axis moved:

| input | mel | latents |
|---|---|---|
| 4.0 s | `(1, 2, 64, 401)` | `(2, 8, 16, **100**)` |
| 2.0 s | `(1, 2, 64, 201)` | `(2, 8, 16, **50**)` |

Axis 3 tracks duration; axis 2 stays at 16 = `mel_bins / mel_compression_ratio`.
So the VAE produces **`(batch, channels, mel, frames)`** while `AUDIO_LATENT_DIMS`
declares `(batch, channels, frames, mel)`. `audio_frames_from_latents` therefore
reads **16** where the true frame count is **100**, and upstream's own
`_, _, audio_num_frames, _ = audio_latents.shape` reads the same wrong axis. PLAN
§5.7 row 5 is precisely this axis, and its whole point is that getting it wrong is
silent.

**3. The reconstruction length is fixed regardless of input.** 4.0 s and 2.0 s
inputs both produce exactly **29,280 samples (0.61 s)**. The decode path is not
tracking the latent frame count — the same defect as #2 from the other end.

**4. `match_channel_count` misreads a batched waveform as mono.** Given
`(1, 2, 64000)` it duplicated "the single channel" and produced batch = 2, having
read the batch dim as the channel count.

**5. The rank contract is self-contradictory.** 4D latents require a batched
waveform; `match_channel_count` requires an unbatched one and corrupts a batched
one. **No input shape currently satisfies both.** The shape guard did its job — its
docstring warns that a wrongly-ranked tensor "can be silently misinterpreted", and
it caught a rank error the VAE itself accepted without complaint.

### Why the A.5.1 candidate list was NOT followed

The comparison looks catastrophic — tones at 220/1320 Hz returning at
2297/2344 Hz, 22 dB down, no tone recovered — and A.5.1 offers an ordered list of
conventions to start changing (log/floor → `mel_scale`/`norm` → `power` → `center`).

**None was touched.** With the batch axis duplicated into channels, the frames axis
transposed and decode emitting a fixed 0.61 s, the reconstruction could not have
resembled the input whatever those values are. Changing a convention against a
structurally broken round trip is how a correct default gets "fixed" into a wrong
one, and the failure would persist looking like the next candidate's fault.

**This is the strength-sweep trap one layer along: characterising against a broken
baseline points at the wrong mechanism.** Having walked into it once, the ordered
list was an invitation to do it again.

**So the conventions remain UNVALIDATED.** H17 cannot answer that question yet, and
H10/H11 passing would establish even less than before — not that the mel is right,
and now not even that the plumbing is.

### H17 — PASS, and what that does and does not establish

`23f7e8b` ("the mel must reach the encoder batched AND time-major") and `5aa5d7b`
(the dtype cast) together fixed every structural finding above. On the natural
unbatched `(2, 64000)` that `load_waveform` returns:

```
mel      (1, 2, 401, 64) float32   min -11.5129  max -0.0854   <- batched AND time-major
latents  (1, 8, 101, 16) bfloat16  absmax 8.2500  rms 2.7778    <- frames on axis 2, as declared
recon    4.01 s out from 4.00 s in                             <- length TRACKS the input
ch0  -16.6 dB   tones recovered: [220, 1320] of [220, 1320]
ch1  -11.1 dB   tones recovered: [440] of [440, 2640]
```

**So the five conventions nobody declared — `power=1.0`, `f_min=0.0`,
`mel_scale`/`norm=slaney`, `center=True`, log floor `1e-05` — reconstruct a known
signal.** That claim is earned by measurement. Moving a tensor between devices does
not change the arithmetic, so this conclusion survives the caveat below.

All three structural findings confirmed fixed against the tree: frames on axis 2,
and the reconstruction length tracking the input where it had been pinned at 0.61 s
for *any* input. `encoder_layout` states the trap better than the original report
did — *"correcting only the batch axis yields a 4D tensor that satisfies the rank
check and is a transposed spectrogram — audio out, silently wrong."*

**Caveat on the instrument, not the result.** `peak_bins` reports only the **top 4**
peaks, and ch1's energy is dominated by the 440 Hz region — so 2640 Hz being absent
from the recovered list is **not** evidence of absent energy. Recorded as unresolved
rather than as high-frequency loss; inferring a defect from a limitation of one's own
tool is the error this file keeps returning to.

### ⛔ The device half of the same hand-off

```
RuntimeError: Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type) should be the same
  generate.py:318   _prepared_audio
  audio.py:706      encode_waveform_to_latents
```

`load_waveform` returns a **CPU** tensor, the mel is computed on CPU, and
`_matched_dtype` does `return cast(dtype=dtype)` — **dtype only** — while the VAE's
weights are on CUDA. It matches one of the two things that must match. One line:
read `device` off the component alongside `dtype`, and rename, because a function
called `_matched_dtype` that also moves devices will have the device half tidied out
by someone matching it to its name.

### The harness hid it, and that is the same failure it exists to catch

H17 originally did `wave.to(device=vae.device, dtype=vae.dtype)` before encoding, so
it **performed a step the application omits** and passed while the app raised. That
is a float32-fixture-over-a-bfloat16-defect in a new costume, authored here.

Fixed by writing the fixture to a wav and reading it back through
`load_waveform`, so the round trip now begins exactly where a user's upload begins —
same decoder, same dtype, same device. H17 now reproduces the device defect instead
of stepping over it, and `load_waveform` is inside what it covers rather than
alongside it.

### Four boundary defects, one hand-off, all four attributes

1. `from_pipe` recasting shared bf16 modules to fp32 — **dtype**, silently.
2. `_as_array` calling `.numpy()` on bf16 — dtype, the way **out**.
3. float32 mel into a bf16 `conv_in` — dtype, the way **in**.
4. CPU mel into CUDA weights — **device**, same hand-off.

Plus rank and axis order, both found in the same function within the hour. **At a
hand-off between our code and a model, every attribute that can disagree eventually
does — dtype, device, rank, axis order — and fixing them one at a time as they
surface is not the same as enumerating them.** All four attributes of a single
tensor hand-off failed today, in sequence, and each was found only by running it.

---

# Final state — all eleven modes execute; one produces silence

`825bfde` (device), `466bbd5` (batching in the product) and `7a8b408` (one
rearrangement, not two) together make the audio-input path structurally correct.

## H17 — PASS, on a criterion that took three attempts to get right

```
duration: in 4.000s -> out 4.010s (0.25% drift, tolerance 2%)
ch0  220 Hz  78.3% RECOVERED     1320 Hz 14.1% RECOVERED
ch1  440 Hz 110.8% RECOVERED     2640 Hz  5.4% LOST (reported, not gated)
criteria: (1) length scales with input (2) each fundamental recovered at its own
          frequency (3) no tone displaced
```

**The criterion history is kept because both earlier versions were wrong in
opposite directions**, and a threshold nobody can see is how a green result comes
to mean less than it appears to:

* **v1 — "≥1 tone per channel."** Passed with 3 of 4 tones missing **by design**.
* **v2 — "every tone."** Failed on a tone that recovers at **47.5%** once the
  pairing is swapped, so the loss is the fixture's tone combination against a
  log-spaced 64-bin mel, not a convention error.
* **v3 — placement and structure.** A wrong `mel_scale`, `f_min`/`f_max`, `norm` or
  `center` **displaces** frequencies; it does not attenuate one harmonic and leave
  the rest in place. So the gates are length-scaling, fundamentals at their own
  frequencies, and no displacement. Harmonic retention is **reported as fidelity
  data**.

**v3 is a loosening from v2 and is labelled as one.** Its justification is the swap
measurement, not that it makes the test green — that distinction is the whole point.

**The length check is the strongest of the three.** The broken path returned exactly
**29,280 samples for a 4.0 s input and for a 2.0 s input** — invariant to duration —
and no shape assertion could see it, because the rank was right and
`validate_audio_latents_shape` **accepted both the correct and the transposed
tensor.** It catches any axis-order error in either direction without needing to
know which axis is which.

### The ch1 question, closed

Running the swap discriminator without the transpose:

| pairing | result | 2640 Hz |
|---|---|---|
| `ch0 (220, 1320)` `ch1 (440, 2640)` | 3/4 | **5.6%** on ch1 |
| `ch0 (220, 2640)` `ch1 (440, 1320)` | **4/4** | **47.5%** on ch0 |

**Neither channel-dependent nor simply frequency-dependent.** ch1 recovered 1320 Hz
at 33.3% when it carried it; 2640 Hz retained 47.5% on ch0 against 5.6% on ch1. It
tracks the *pairing* — a log-spaced 64-bin mel shares bins differently for
440 + 2640 than for 440 + 1320. **The same path recovers 4/4 with a different
pairing**, so nothing structural drops high frequencies. Closed, not carried.

## ⚠️ `audio_to_audio` produces silence — the one open defect

| | `noise_scale` | peak | rms | dBFS |
|---|---|---|---|---|
| **input fixture** | — | 0.4000 | 0.182794 | **−14.8** |
| H10 | 0.5 | 0.0002 | 0.000040 | **−87.9** |
| H10b_lo | 0.1 | 0.0003 | 0.000045 | **−87.0** |
| H10b_hi | 0.9 | 0.0005 | 0.000066 | **−83.6** |
| H8 `text_to_audio` | — | 0.6953 | 0.172677 | −15.3 |
| H9 `video_to_audio` | — | 0.5078 | 0.128512 | −17.8 |

~70 dB below every other audio mode. **The input fixture was verified rather than
assumed** — 2ch @ 48000, 5.00 s, peak 0.4000, −14.8 dBFS. Input healthy, output not.
All three pass `verify_streams` with exit 0.

### H10b is a vacuous pass, and that is the finding

The halves **differ** — `rms 0.000045` vs `0.000066`, a factor of 1.5 — so
`noise_scale` reaches the model and the audio-input join is genuinely wired. **But
the difference is between two silences.** Through a broken path two wrong answers
can still differ, and "differs" is the verdict H10b reports as success.

**Only reporting RMS from both halves makes that visible.** The differ/identical
verdict alone would have shipped this as the eleventh mode proven and the
acceptance test passed. A same-seed comparison establishes that a lever moves
something; it cannot establish that either result is right.

### What narrows it

`audio_to_video` consumed the **same** audio input and produced a valid
327,420-byte video, so the latents are not breaking the pipeline generally.
`audio_to_audio` is the **only** mode that passes `audio_latents` into the pipeline;
`text_to_audio` and `video_to_audio` do not and both are healthy.

**A direction, not a diagnosis:** H17 shows the latents are correct *for
`audio_vae.decode`*. One candidate for why they are wrong *for the pipeline* is
scale — H17 measured them at `rms 3.2548`, and if the pipeline expects normalised
audio latents while receiving raw VAE output, `noise_scale * noise +
(1 − noise_scale) * latents` is dominated by a mis-scaled term at every
`noise_scale`, which matches all three readings being silent. **Confirm against
upstream's audio branch before changing any scale factor** — reasoning from
correlation to cause has cost this build two changes today.
