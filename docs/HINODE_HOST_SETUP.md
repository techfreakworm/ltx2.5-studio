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

`PYTHONPATH` is needed because the runner scripts live in
`/home/wakeuser/ltx25-runners/`, **outside** the repo, so that a re-sync cannot
wipe them. `sys.path[0]` is then the runner's directory, not the repo, and
`import ltx25_studio` fails without it.

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
cat ~/.cache/huggingface/token | ssh wakeuser@hinode '/home/wakeuser/ltx25-runners/launch_download.sh'
```

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
