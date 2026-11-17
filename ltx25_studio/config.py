"""Runtime configuration, resolved from the environment with loud validation.

PLAN §2.6, §2.7, §5.6, §8.

**Why the environment and not constants.** Hugging Face Spaces lets an operator
set Variables and Secrets from the web UI, taking effect on the next restart
**with no git push**. Anything we might plausibly need to tune on a live Space
therefore lives here as an environment variable with a plan-matching default:
if deploy #1 is evicted or OOMs, the fix is a variable, not a code change and a
round-trip through whoever holds push rights.

**Loud validation is the point.** An unrecognised value raises at startup,
naming the variable and the legal values. Silently falling back to a default
would hide a misconfiguration behind a working-but-wrong run, which is the
class of failure §5 exists to eliminate.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from . import model_files
from .errors import ConfigError

__all__ = [
    "CPU_DTYPE_CHOICES",
    "DEVICE_CHOICES",
    "ENV_CPU_DTYPE",
    "ENV_DEVICE",
    "ENV_GPU_DURATION_FLOOR_SECONDS",
    "ENV_GPU_DURATION_SECONDS",
    "ENV_LORA_MAX_GB",
    "ENV_MODEL_SNAPSHOT_DIR",
    "ENV_MPS_SELF_TEST",
    "ENV_RESIDENCY",
    "ENV_TRANSFORMER_GGUF",
    "ENV_QUANTIZATION",
    "MPS_SELF_TEST_CHOICES",
    "RESIDENCY_CHOICES",
    "DECODER_CHOICES",
    "DEFAULT_DECODER",
    "ENV_DECODER",
    "QUANTIZATION_CHOICES",
    "ENV_OUTPUT_DIR",
    "ENV_OUTPUT_RETENTION_GB",
    "ENV_PREFETCH_ENHANCER",
    "ENV_PROMPT_ENHANCER_REPO",
    "ENV_ZEROGPU_SIZE",
    "RuntimeConfig",
    "load_config",
]

# --- variable names, in one place so the deploy request can quote them -------
ENV_ZEROGPU_SIZE = "LTX25_ZEROGPU_SIZE"
#: No longer read for the reservation, which follows each run's size and length
#: (``zerogpu.estimate_duration``); when set, startup says it is ignored.
ENV_GPU_DURATION_SECONDS = "LTX25_GPU_DURATION_SECONDS"
#: Optional, unset by default: every reservation is at least this many seconds. The
#: operator's lever if an estimate ever proves short.
ENV_GPU_DURATION_FLOOR_SECONDS = "LTX25_GPU_DURATION_FLOOR_SECONDS"
ENV_OUTPUT_RETENTION_GB = "LTX25_OUTPUT_RETENTION_GB"
ENV_OUTPUT_DIR = "LTX25_OUTPUT_DIR"
ENV_LORA_MAX_GB = "LTX25_LORA_MAX_GB"
ENV_QUANTIZATION = "LTX25_QUANTIZATION"
ENV_PROMPT_ENHANCER_REPO = "LTX25_PROMPT_ENHANCER_REPO"
ENV_PREFETCH_ENHANCER = "LTX25_PREFETCH_ENHANCER"
ENV_DEVICE = "LTX25_DEVICE"
ENV_CPU_DTYPE = "LTX25_CPU_DTYPE"
ENV_MODEL_SNAPSHOT_DIR = "LTX25_MODEL_SNAPSHOT_DIR"
ENV_MPS_SELF_TEST = "LTX25_MPS_SELF_TEST"
ENV_TRANSFORMER_GGUF = "LTX25_TRANSFORMER_GGUF"
ENV_RESIDENCY = "LTX25_RESIDENCY"
ENV_DECODER = "LTX25_DECODER"

#: ``auto`` passes no ``size`` and the platform then allocates its default,
#: ``large`` (47.375 GiB measured). ZeroGPU does NOT promote by itself: its
#: ``startup()`` promotion only changes the duration factor, never the size it
#: schedules (zerogpu.py docstring; docs/MIG_ALLOCATOR_SIGHTINGS.md §1).
ZEROGPU_SIZES = ("auto", "large", "xlarge")

DEFAULT_ZEROGPU_SIZE = "xlarge"
DEFAULT_OUTPUT_RETENTION_GB = 20.0
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_LORA_MAX_GB = 12.0

#: Weight precision for ``pipelines.QUANTIZED_COMPONENTS``. ``auto`` is int8 on
#: ZeroGPU, where bf16 does not fit the disk cap (packing peak 152.4-156.4 GB
#: against ``pipelines.ZEROGPU_DISK_CAP_GB``; docs/ZEROGPU_DISK.md §4.2), and
#: bf16 everywhere else. ``bf16``/``int8`` are honoured as given; int8 off CUDA
#: is refused at startup by ``device.resolve``. A real lever: it changes output
#: quality, so an int8-vs-bf16 pair at one seed must stay possible.
DEFAULT_QUANTIZATION = "auto"
QUANTIZATION_CHOICES = ("auto", "bf16", "int8")

#: The generation device. ``auto`` is CUDA on ZeroGPU or wherever CUDA exists,
#: and no generation device anywhere else (the UI starts; Generate explains how
#: to opt in). ``mps`` and ``cpu`` are explicit opt-ins for local runs.
DEFAULT_DEVICE = "auto"
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")

#: Weight dtype when the device is the CPU. fp32 is the numerical reference the
#: tiny-checkpoint tier compares everything against.
DEFAULT_CPU_DTYPE = "fp32"
CPU_DTYPE_CHOICES = ("fp32", "bf16")

#: What happens when the MPS numerics self-test fails: ``enforce`` refuses to
#: generate, ``warn`` runs anyway and says so in the log and the report.
DEFAULT_MPS_SELF_TEST = "enforce"
MPS_SELF_TEST_CHOICES = ("enforce", "warn")

#: How the local (mps/cpu) runner holds real weights: ``staged`` never keeps the
#: text encoder and the transformer resident together; ``resident`` keeps both.
#: Staged by default on macOS, where memory is shared with the GPU.
DEFAULT_RESIDENCY = "staged" if sys.platform == "darwin" else "resident"
RESIDENCY_CHOICES = ("staged", "resident")

#: Which decoder turns the final latents into frames. ``diffusion`` is the
#: model's native decoder, ``conv`` the convolutional VAE; ``auto`` picks
#: diffusion where it can run (CUDA, its kernel available, the video large
#: enough) and conv otherwise, saying which in the run's notes.
DEFAULT_DECODER = "auto"
DECODER_CHOICES = ("auto", "diffusion", "conv")

#: The prompt enhancer (PLAN §3.5), loaded lazily by ``enhance.py`` on first use
#: inside the GPU worker and then KEPT for the life of that worker -- freeing it
#: per call cost a measured +30.78 s every time. A variable rather than a
#: constant because it is the
#: one model this app resolves OUTSIDE the pinned pack -- ``model_index.json``
#: ships ``prompt_enhancer: [None, None]`` and there is no ``prompt_enhancer/``
#: folder in ``Lightricks/LTX-2.5-Diffusers``, so it is an ordinary
#: ``transformers`` load into the HF cache and nothing in ``model_files.py`` or
#: ``preflight.py`` knows or needs to know about it.
#:
#: ✅ **The default is the QAT build, checked against the Hub rather than
#: assumed**: ``google/gemma-4-E2B-it-qat-mobile-transformers`` is public, not
#: gated, one ``model.safetensors``, **2.490 GB total** -- against **10.279 GB**
#: for the bf16 ``google/gemma-4-E2B-it``. Its ``config.json`` carries
#: ``quantization_config.quant_method == "gemma"``, which ``transformers``
#: registers as ``GemmaQuantizer`` from 5.15.1 on; requirements.txt pins 5.23.0,
#: where ``enhance.py`` reads one per-layer scale back from the checkpoint. The
#: unpacking layers it installs are plain torch, with no CUDA or kernel
#: dependency, so the checkpoint runs on either device -- which is what let the
#: enhancer start on the CPU and what lets it run on the GPU now.
#:
#: ⚠️ Its ``config.json`` has **no** ``architectures`` key, so ``AutoModel*``
#: cannot dispatch on it. ``enhance.py`` therefore names
#: ``Gemma4ForConditionalGeneration`` explicitly, which is the class the LTX-2.5
#: pipeline types the component as anyway.
#:
#: Set this to ``google/gemma-4-E2B-it`` for the bf16 build.
#:
#: A repo other than this default loads at its Hub head, unpinned; the default
#: is pinned by ``enhance.ENHANCER_REVISION``.
DEFAULT_PROMPT_ENHANCER_REPO = model_files.ENHANCER_REPO_ID

#: Fetch the enhancer's weights into the HF cache **at startup**, so the first
#: enhanced generation in a container does not pay the download.
#:
#: **On by default, because the alternative is billed.** Enhancement now runs
#: inside the ``@spaces.GPU`` worker (§3.5), so without a prefetch the first
#: enhanced call in each container spends the measured **88 s** Hub fetch inside
#: the metered reservation -- roughly 176 GPU-seconds, at the xlarge multiplier,
#: to download a file. Defaulting this off would make the prefetch a feature
#: nobody gets, which is the inert-lever shape pointed the other way.
#:
#: ⚠️ **Why it is switchable at all: disk.** The prefetch is ~2.5 GB for the QAT
#: default, and it lands while ``demo.launch()`` is packing -- the exact window
#: ``pipelines.ZEROGPU_DISK_CAP_GB`` is measured against, where the int8 packing
#: peak is derived at 121.4-125.4 GB for transformer + text encoder
#: (``pipelines._quantization_config``; never measured). An operator who knows
#: this deploy will never enable enhancement can decline to spend that headroom,
#: and gets the un-prefetched behaviour back: still working, just slower on
#: first use. On the Space the README preloads these files into the image, so
#: there the lever only switches this fallback fetch off.
DEFAULT_PREFETCH_ENHANCER = True


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value not in allowed:
        raise ConfigError(variable=name, value=raw, legal=allowed)
    return value


def _flag(name: str, default: bool) -> bool:
    """A boolean environment variable, validated as loudly as every other one.

    ``bool(os.environ.get(name))`` would read ``"0"``, ``"false"`` and ``"no"``
    as **true** -- an operator who set the variable to turn something off would
    turn it on and get no complaint. So the accepted spellings are enumerated
    and anything else raises, naming them.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(variable=name, value=raw, legal=tuple(sorted(_TRUE | _FALSE)))


def _snapshot_dir(name: str) -> str | None:
    """A local diffusers-layout directory to load every component from, or None.

    Validated here, at startup: a typo would otherwise surface as a load
    failure on the first Generate, or worse, fall back to the pinned snapshot.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    path = os.path.realpath(os.path.expanduser(raw))
    if not os.path.isfile(os.path.join(path, "model_index.json")):
        raise ConfigError(
            variable=name,
            value=raw,
            legal=("an existing directory that contains model_index.json (a diffusers snapshot)",),
        )
    return path


def _gguf_file(name: str) -> str | None:
    """A local ``.gguf`` transformer file for the local runner, or None.

    The name is checked as given: in the HF cache ``snapshots/<rev>/x.gguf`` is a
    link to an extensionless blob, and the link (not the blob) is returned.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    path = os.path.abspath(os.path.expanduser(raw))
    if not (os.path.isfile(path) and path.endswith(".gguf")):
        raise ConfigError(variable=name, value=raw, legal=("an existing .gguf file",))
    return path


def _set_value(name: str) -> str | None:
    """The variable as set, stripped, or None when it is unset or empty."""
    return (os.environ.get(name) or "").strip() or None


def _optional_seconds(name: str) -> int | None:
    """A whole number of seconds, at least 1, or None when the variable is unset."""
    raw = _set_value(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ConfigError(
            variable=name, value=raw, legal=("a whole number of seconds >= 1", "unset")
        )
    return value


def _number(name: str, default: float, *, minimum: float, integer: bool = False):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default) if integer else default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise ConfigError(
            variable=name,
            value=raw,
            legal=(f"{'an integer' if integer else 'a number'} >= {minimum}", f"default {default}"),
        ) from None
    if value < minimum:
        raise ConfigError(variable=name, value=raw, legal=(f">= {minimum}", f"default {default}"))
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    """Everything tunable without a code change."""

    zerogpu_size: str
    output_retention_gb: float
    output_dir: str
    lora_max_gb: float
    quantization: str = DEFAULT_QUANTIZATION
    prompt_enhancer_repo: str = DEFAULT_PROMPT_ENHANCER_REPO
    prefetch_enhancer: bool = DEFAULT_PREFETCH_ENHANCER
    device: str = DEFAULT_DEVICE
    cpu_dtype: str = DEFAULT_CPU_DTYPE
    model_snapshot_dir: str | None = None
    mps_self_test: str = DEFAULT_MPS_SELF_TEST
    transformer_gguf: str | None = None
    residency: str = DEFAULT_RESIDENCY
    decoder: str = DEFAULT_DECODER
    #: ``LTX25_GPU_DURATION_FLOOR_SECONDS``: every reservation is at least this; None unset.
    gpu_duration_floor_seconds: int | None = None
    #: ``LTX25_GPU_DURATION_SECONDS`` as set, which nothing reads any more; None unset.
    ignored_gpu_duration_seconds: str | None = None

    @property
    def lora_max_bytes(self) -> int:
        return int(self.lora_max_gb * 1e9)

    @property
    def download_gb(self) -> float:
        """The pinned download set on disk: bf16 files, whatever the quantisation.

        Not the loaded footprint -- an int8 load holds less in memory and packs
        smaller, from the same files.
        """
        return model_files.total_size_gb()

    def describe(self) -> str:
        """One-line startup summary of the levers as configured (not resolved)."""
        snapshot = self.model_snapshot_dir or "pinned"
        floor = self.gpu_duration_floor_seconds
        return (
            f"zerogpu_size={self.zerogpu_size} "
            f"reservation=estimated{f',floor={floor}s' if floor else ''} "
            f"quantization={self.quantization} "
            f"device={self.device} "
            f"snapshot={snapshot} "
            f"outputs<={self.output_retention_gb:.0f}GB"
        )


def load_config() -> RuntimeConfig:
    """Resolve the runtime configuration, raising on anything unrecognised."""
    return RuntimeConfig(
        zerogpu_size=_choice(ENV_ZEROGPU_SIZE, DEFAULT_ZEROGPU_SIZE, ZEROGPU_SIZES),
        output_retention_gb=_number(
            ENV_OUTPUT_RETENTION_GB, DEFAULT_OUTPUT_RETENTION_GB, minimum=1.0
        ),
        output_dir=os.environ.get(ENV_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR,
        lora_max_gb=_number(ENV_LORA_MAX_GB, DEFAULT_LORA_MAX_GB, minimum=0.01),
        quantization=_choice(ENV_QUANTIZATION, DEFAULT_QUANTIZATION, QUANTIZATION_CHOICES),
        # Free text, not a `_choice`: any `org/name` on the Hub is legal here and
        # an allow-list would have to be edited before a smaller enhancer could
        # ever be tried. `enhance.py` raises `PromptEnhancementError` naming this
        # value if it does not resolve, which is a better error than a startup
        # refusal that cannot see whether the repo exists either.
        prompt_enhancer_repo=(os.environ.get(ENV_PROMPT_ENHANCER_REPO) or "").strip()
        or DEFAULT_PROMPT_ENHANCER_REPO,
        prefetch_enhancer=_flag(ENV_PREFETCH_ENHANCER, DEFAULT_PREFETCH_ENHANCER),
        device=_choice(ENV_DEVICE, DEFAULT_DEVICE, DEVICE_CHOICES),
        cpu_dtype=_choice(ENV_CPU_DTYPE, DEFAULT_CPU_DTYPE, CPU_DTYPE_CHOICES),
        model_snapshot_dir=_snapshot_dir(ENV_MODEL_SNAPSHOT_DIR),
        mps_self_test=_choice(ENV_MPS_SELF_TEST, DEFAULT_MPS_SELF_TEST, MPS_SELF_TEST_CHOICES),
        transformer_gguf=_gguf_file(ENV_TRANSFORMER_GGUF),
        residency=_choice(ENV_RESIDENCY, DEFAULT_RESIDENCY, RESIDENCY_CHOICES),
        decoder=_choice(ENV_DECODER, DEFAULT_DECODER, DECODER_CHOICES),
        gpu_duration_floor_seconds=_optional_seconds(ENV_GPU_DURATION_FLOOR_SECONDS),
        # Not validated: it is read only to say, once at startup, that it is ignored.
        ignored_gpu_duration_seconds=_set_value(ENV_GPU_DURATION_SECONDS),
    )
