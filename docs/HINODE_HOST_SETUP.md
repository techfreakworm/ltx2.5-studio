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
