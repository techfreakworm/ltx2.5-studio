"""Memory preflight for local runs: committed memory plus the next step, against a limit.

The operator's rule on the Mac (M5 Max, 128 GB, unified memory): **committed
memory plus what the next step adds stays at or under 90 GB**. "Committed" is
``vm_stat``'s active + wired + occupied-by-compressor pages times the page
size: never RSS (it misses MPS allocations and double-counts nothing useful)
and never "available" (it counts reclaimable cache as free).

Stdlib only; torch is imported lazily by :func:`mps_stats` alone, so this module
imports in the torch-less environment.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass

from .errors import ConfigError, LTXStudioError

__all__ = [
    "DEFAULT_DARWIN_LIMIT_GB",
    "FIXED_ACTIVATION_GB",
    "MemoryPreflightError",
    "PreflightResult",
    "activation_calibration",
    "activation_estimate_gb",
    "activation_terms",
    "committed_gb",
    "default_limit_gb",
    "load_spike_factor",
    "mps_stats",
    "parse_vm_stat",
    "preflight",
    "video_tokens",
]

GB = 1e9  # decimal GB throughout, as the 90 GB rule and the Hub's file sizes are stated

LIMIT_ENV = "LTX25_MEMORY_LIMIT_GB"
CALIBRATION_ENV = "LTX25_ACTIVATION_CALIBRATION"
SPIKE_ENV = "LTX25_LOAD_SPIKE_FACTOR"

#: The operator's committed-memory ceiling on the Mac.
DEFAULT_DARWIN_LIMIT_GB = 90.0
#: Until the lead measures the load spike (recon K §8.5 "worst case k = 2").
DEFAULT_LOAD_SPIKE_FACTOR = 2.0

# --- activation model constants (recon K §8.2-8.3, real transformer/vae configs) ---
#: VAE compression: 8x in time, 32x in space (vae config, patch_size 1).
TEMPORAL_COMPRESSION = 8
SPATIAL_COMPRESSION = 32
#: Transformer: 32 video attention heads of 128 -> hidden size 4096.
ATTENTION_HEADS = 32
HIDDEN_SIZE = 4096
#: torch 2.11 MPS SDPA materialises (1, heads, L, L) about three times over:
#: QK^T, the softmax, and the attention output buffer (recon K §8.3, H3).
ATTENTION_COPIES = 3
#: Hidden-size tensors alive at once in a block (recon K §8.3 estimate).
LIVE_HIDDEN_TENSORS = 40
#: Conv-VAE decode: about six live 256-channel tensors at a quarter of the frame
#: size (recon K §8.3), tiled by the app's envelope (pipelines.VAE_TILE_*).
VAE_LIVE_TENSORS = 6
VAE_DECODE_CHANNELS = 256
VAE_DECODE_DOWNSCALE = 4
VAE_TILE_FRAMES = 56
VAE_TILE_HEIGHT = 480
VAE_TILE_WIDTH = 704
#: The decoded video lands as one float32 RGB buffer.
OUTPUT_BYTES_PER_SAMPLE = 4
#: Process slack that does not scale with size (recon K §8.3).
FIXED_ACTIVATION_GB = 1.0


class MemoryPreflightError(LTXStudioError):
    """A step would push committed memory over the limit, so it was not started."""

    def __init__(self, result: PreflightResult | None = None, *, reason: str | None = None):
        self.result = result
        if result is None:
            message = reason or "the memory check could not run"
        else:
            message = (
                f"Not enough memory for '{result.phase}': {result.committed:.1f} GB is already in "
                f"use and this step needs about {result.need:.1f} GB more, {result.total:.1f} GB "
                f"in all, which is over the {result.limit:.1f} GB limit by "
                f"{-result.headroom:.1f} GB. Nothing was loaded. Close other apps, pick a smaller "
                f"size or fewer frames, or raise {LIMIT_ENV} if you have measured that this "
                f"machine can take it."
            )
        super().__init__(message)


# --------------------------------------------------------------------------
# committed memory
# --------------------------------------------------------------------------

_PAGE_SIZE = re.compile(r"page size of (\d+) bytes")
_COUNTER = re.compile(r'^"?([^":]+)"?:\s+(\d+)\.?\s*$')
_COMMITTED_COUNTERS = ("Pages active", "Pages wired down", "Pages occupied by compressor")


def parse_vm_stat(text: str) -> tuple[int, dict[str, int]]:
    """``(page_size, {counter: pages})`` from ``vm_stat`` output."""
    match = _PAGE_SIZE.search(text)
    if not match:
        raise MemoryPreflightError(reason="could not read the page size from vm_stat output")
    counters = {}
    for line in text.splitlines():
        found = _COUNTER.match(line.strip())
        if found:
            counters[found.group(1).strip()] = int(found.group(2))
    return int(match.group(1)), counters


def committed_gb(*, vm_stat_text: str | None = None, platform: str | None = None) -> float | None:
    """Active + wired + compressor-occupied pages, in GB. ``None`` off macOS."""
    if (platform or sys.platform) != "darwin":
        return None
    if vm_stat_text is None:
        try:
            vm_stat_text = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise MemoryPreflightError(
                reason=f"could not run vm_stat to measure memory: {exc}"
            ) from exc
    page_size, counters = parse_vm_stat(vm_stat_text)
    missing = [name for name in _COMMITTED_COUNTERS if name not in counters]
    if missing:
        raise MemoryPreflightError(
            reason=f"vm_stat output is missing {', '.join(missing)}; cannot measure memory"
        )
    return sum(counters[name] for name in _COMMITTED_COUNTERS) * page_size / GB


def _live_committed(platform: str | None) -> float | None:
    return committed_gb(platform=platform)


# --------------------------------------------------------------------------
# environment knobs
# --------------------------------------------------------------------------


def _env_float(name: str, *, default, minimum: float, legal: str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        value = None
    if value is None or value != value or value < minimum:
        raise ConfigError(variable=name, value=raw, legal=(legal,))
    return value


def default_limit_gb(*, platform: str | None = None) -> float | None:
    """``LTX25_MEMORY_LIMIT_GB``, else 90 on macOS, else no limit."""
    fallback = DEFAULT_DARWIN_LIMIT_GB if (platform or sys.platform) == "darwin" else None
    return _env_float(LIMIT_ENV, default=fallback, minimum=1.0, legal="a number of GB >= 1")


def activation_calibration() -> float:
    """``LTX25_ACTIVATION_CALIBRATION``: the lead's measured/estimated ratio, default 1.0."""
    return _env_float(CALIBRATION_ENV, default=1.0, minimum=0.01, legal="a positive number")


def load_spike_factor() -> float:
    """``LTX25_LOAD_SPIKE_FACTOR``: transient bytes per resident byte while loading."""
    return _env_float(
        SPIKE_ENV, default=DEFAULT_LOAD_SPIKE_FACTOR, minimum=1.0, legal="a number >= 1"
    )


# --------------------------------------------------------------------------
# activation estimate
# --------------------------------------------------------------------------


def video_tokens(width: int, height: int, num_frames: int) -> int:
    """``((F-1)/8 + 1) * (H/32) * (W/32)``: the transformer's video sequence length."""
    latent_frames = (int(num_frames) - 1) // TEMPORAL_COMPRESSION + 1
    return (
        latent_frames * (int(height) // SPATIAL_COMPRESSION) * (int(width) // SPATIAL_COMPRESSION)
    )


def activation_terms(
    width: int, height: int, num_frames: int, *, dtype_bytes: int = 2
) -> dict[str, float]:
    """Each size-dependent term of the estimate, in GB."""
    tokens = video_tokens(width, height, num_frames)
    tile_frames = min(int(num_frames), VAE_TILE_FRAMES)
    tile_height = min(int(height), VAE_TILE_HEIGHT) // VAE_DECODE_DOWNSCALE
    tile_width = min(int(width), VAE_TILE_WIDTH) // VAE_DECODE_DOWNSCALE
    return {
        "attention": ATTENTION_COPIES * ATTENTION_HEADS * tokens**2 * dtype_bytes / GB,
        "hidden": LIVE_HIDDEN_TENSORS * tokens * HIDDEN_SIZE * dtype_bytes / GB,
        "vae_decode_tiles": VAE_LIVE_TENSORS
        * VAE_DECODE_CHANNELS
        * tile_frames
        * tile_height
        * tile_width
        * dtype_bytes
        / GB,
        "vae_output": int(num_frames) * int(height) * int(width) * 3 * OUTPUT_BYTES_PER_SAMPLE / GB,
    }


def activation_estimate_gb(
    width: int,
    height: int,
    num_frames: int,
    *,
    dtype_bytes: int = 2,
    calibration: float | None = None,
) -> float:
    """Peak activations for one denoise + decode call, in GB.

    ``calibration * (attention + hidden + vae tiles + vae output) + 1 GB``. The
    transformer and VAE terms are summed although the phases are sequential:
    the MPS allocator keeps freed blocks cached, so the sum is the safe side.
    ``calibration=None`` reads ``LTX25_ACTIVATION_CALIBRATION`` (default 1.0).
    """
    scale = activation_calibration() if calibration is None else float(calibration)
    terms = activation_terms(width, height, num_frames, dtype_bytes=dtype_bytes)
    return scale * sum(terms.values()) + FIXED_ACTIVATION_GB


# --------------------------------------------------------------------------
# the preflight
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightResult:
    phase: str
    committed: float | None
    need: float
    limit: float | None
    headroom: float | None
    verdict: str  # "OK", "FAIL" or "REPORT"

    @property
    def total(self) -> float | None:
        return None if self.committed is None else self.committed + self.need

    def describe(self) -> str:
        head = f"preflight[{self.phase}]: "
        if self.committed is None:
            return f"{head}committed n/a + need {self.need:.1f} GB (not measured here; report only)"
        sum_text = f"committed {self.committed:.1f} + need {self.need:.1f} = {self.total:.1f} GB"
        if self.limit is None:
            return f"{head}{sum_text} (no limit; report only)"
        if self.verdict == "OK":
            return f"{head}{sum_text} <= {self.limit:.1f} GB OK (headroom {self.headroom:.1f})"
        return f"{head}{sum_text} > {self.limit:.1f} GB FAIL (short by {-self.headroom:.1f})"

    def require(self) -> PreflightResult:
        if self.verdict == "FAIL":
            raise MemoryPreflightError(self)
        return self


_MEASURE = object()


def preflight(
    phase: str,
    *,
    need_gb: float,
    limit_gb: float | None | object = _MEASURE,
    committed_gb: float | None | object = _MEASURE,
    platform: str | None = None,
) -> PreflightResult:
    """Judge one step. ``limit_gb`` defaults to :func:`default_limit_gb`.

    ``committed_gb`` defaults to the live gauge; pass a number to judge a
    projected state (the dry run does).
    """
    limit = default_limit_gb(platform=platform) if limit_gb is _MEASURE else limit_gb
    committed = _live_committed(platform) if committed_gb is _MEASURE else committed_gb
    need = float(need_gb)
    if committed is None or limit is None:
        return PreflightResult(phase, committed, need, limit, None, "REPORT")
    headroom = float(limit) - (float(committed) + need)
    verdict = "OK" if headroom >= 0 else "FAIL"
    return PreflightResult(phase, float(committed), need, float(limit), headroom, verdict)


def mps_stats() -> dict[str, float] | None:
    """MPS allocator counters in GB, or ``None`` without torch or MPS."""
    try:
        import torch
    except ImportError:
        return None
    mps = getattr(torch, "mps", None)
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None or backend is None or not backend.is_available():
        return None
    try:
        return {
            "allocated_gb": mps.current_allocated_memory() / GB,
            "driver_gb": mps.driver_allocated_memory() / GB,
            "recommended_max_gb": mps.recommended_max_memory() / GB,
        }
    except (AttributeError, RuntimeError):
        return None
