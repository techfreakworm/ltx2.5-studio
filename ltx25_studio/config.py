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
from dataclasses import dataclass

from . import model_files
from .errors import ConfigError

__all__ = [
    "ENV_GPU_DURATION_SECONDS",
    "ENV_LORA_MAX_GB",
    "ENV_OUTPUT_DIR",
    "ENV_OUTPUT_RETENTION_GB",
    "ENV_ZEROGPU_SIZE",
    "RuntimeConfig",
    "load_config",
]

# --- variable names, in one place so the deploy request can quote them -------
ENV_ZEROGPU_SIZE = "LTX25_ZEROGPU_SIZE"
ENV_GPU_DURATION_SECONDS = "LTX25_GPU_DURATION_SECONDS"
ENV_OUTPUT_RETENTION_GB = "LTX25_OUTPUT_RETENTION_GB"
ENV_OUTPUT_DIR = "LTX25_OUTPUT_DIR"
ENV_LORA_MAX_GB = "LTX25_LORA_MAX_GB"

#: ``auto`` means "do not pass ``size`` at all". ZeroGPU promotes us to xlarge
#: by itself once packed weights exceed its 42.00 GiB threshold, and passing the
#: kwarg to a platform ``spaces`` that predates it is an import-time TypeError.
ZEROGPU_SIZES = ("auto", "large", "xlarge")

#: NOTE: there is deliberately NO precision/quantisation variable here.
#: One existed and was wired to nothing -- `pipelines.get_pipeline` never read
#: the config, so setting it changed no behaviour while `expected_weights_gb`
#: confidently reported 53.20 GB for a load that was still 72.19 GB. An
#: advertised lever attached to nothing is worse than no lever: it does not fail
#: silently, it *speaks*, and it misdirects the next decision. Wiring it properly
#: needs bitsandbytes or torchao -- neither pinned, neither verifiable without a
#: GPU, and both using custom tensor subclasses whose interaction with ZeroGPU's
#: packer is unknown (PLAN §2.6). So it is gone rather than half-present.
DEFAULT_ZEROGPU_SIZE = "auto"
DEFAULT_GPU_DURATION_SECONDS = 60
DEFAULT_OUTPUT_RETENTION_GB = 20.0
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_LORA_MAX_GB = 12.0


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value not in allowed:
        raise ConfigError(variable=name, value=raw, legal=allowed)
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
    gpu_duration_seconds: int
    output_retention_gb: float
    output_dir: str
    lora_max_gb: float

    @property
    def lora_max_bytes(self) -> int:
        return int(self.lora_max_gb * 1e9)

    @property
    def expected_weights_gb(self) -> float:
        """On-disk weight footprint. Always the full bf16 subset.

        There is no precision switch, so this is simply what
        ``model_files`` declares -- and being a single honest number beats being
        a configurable one that reported a footprint nobody was loading.
        """
        return model_files.total_size_gb()

    def describe(self) -> str:
        """One-line startup summary."""
        return (
            f"zerogpu_size={self.zerogpu_size} "
            f"duration={self.gpu_duration_seconds}s "
            f"expected_weights={self.expected_weights_gb:.2f}GB "
            f"outputs<={self.output_retention_gb:.0f}GB"
        )


def load_config() -> RuntimeConfig:
    """Resolve the runtime configuration, raising on anything unrecognised."""
    return RuntimeConfig(
        zerogpu_size=_choice(ENV_ZEROGPU_SIZE, DEFAULT_ZEROGPU_SIZE, ZEROGPU_SIZES),
        gpu_duration_seconds=_number(
            ENV_GPU_DURATION_SECONDS, DEFAULT_GPU_DURATION_SECONDS, minimum=1, integer=True
        ),
        output_retention_gb=_number(
            ENV_OUTPUT_RETENTION_GB, DEFAULT_OUTPUT_RETENTION_GB, minimum=1.0
        ),
        output_dir=os.environ.get(ENV_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR,
        lora_max_gb=_number(ENV_LORA_MAX_GB, DEFAULT_LORA_MAX_GB, minimum=0.01),
    )
