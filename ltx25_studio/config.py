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
from types import MappingProxyType

from . import model_files
from .errors import LTXStudioError

__all__ = [
    "ENV_GPU_DURATION_SECONDS",
    "ENV_LORA_MAX_GB",
    "ENV_OUTPUT_DIR",
    "ENV_OUTPUT_RETENTION_GB",
    "ENV_TEXT_ENCODER_PRECISION",
    "ENV_TRANSFORMER_PRECISION",
    "ENV_ZEROGPU_SIZE",
    "PRECISION_SIZE_FACTOR",
    "RuntimeConfig",
    "load_config",
]

# --- variable names, in one place so the deploy request can quote them -------
ENV_TRANSFORMER_PRECISION = "LTX25_TRANSFORMER_PRECISION"
ENV_TEXT_ENCODER_PRECISION = "LTX25_TEXT_ENCODER_PRECISION"
ENV_ZEROGPU_SIZE = "LTX25_ZEROGPU_SIZE"
ENV_GPU_DURATION_SECONDS = "LTX25_GPU_DURATION_SECONDS"
ENV_OUTPUT_RETENTION_GB = "LTX25_OUTPUT_RETENTION_GB"
ENV_OUTPUT_DIR = "LTX25_OUTPUT_DIR"
ENV_LORA_MAX_GB = "LTX25_LORA_MAX_GB"

TRANSFORMER_PRECISIONS = ("bf16", "fp8")
TEXT_ENCODER_PRECISIONS = ("bf16", "int8")
#: ``auto`` means "do not pass ``size`` at all". ZeroGPU promotes us to xlarge
#: by itself once packed weights exceed its 42.00 GiB threshold, and passing the
#: kwarg to a platform ``spaces`` that predates it is an import-time TypeError.
ZEROGPU_SIZES = ("auto", "large", "xlarge")

#: Rough on-disk multiplier per precision, for the startup footprint estimate.
#: Deliberately approximate: it drives a log line and a preflight comparison,
#: not an allocation.
PRECISION_SIZE_FACTOR = MappingProxyType({"bf16": 1.0, "fp8": 0.5, "int8": 0.5})

#: Defaults match the plan of record. bf16 is the documented configuration;
#: quantisation is a fallback whose interaction with ZeroGPU's tensor packer is
#: unverified (PLAN §2.6, technical risk #1).
DEFAULT_TRANSFORMER_PRECISION = "bf16"
DEFAULT_TEXT_ENCODER_PRECISION = "bf16"
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
        raise LTXStudioError(
            f"{name}={raw!r} is not a recognised value. Legal values are: "
            f"{', '.join(allowed)} (default {default!r}). Refusing to fall back silently."
        )
    return value


def _number(name: str, default: float, *, minimum: float, integer: bool = False):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default) if integer else default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise LTXStudioError(
            f"{name}={raw!r} is not a number. Expected "
            f"{'an integer' if integer else 'a number'} >= {minimum} (default {default})."
        ) from None
    if value < minimum:
        raise LTXStudioError(
            f"{name}={raw!r} is below the minimum of {minimum} (default {default})."
        )
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    """Everything tunable without a code change."""

    transformer_precision: str
    text_encoder_precision: str
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
        """Approximate on-disk weight footprint for the resolved precisions.

        Logged next to the preflight's measured free RAM and disk so that, when
        a Space misbehaves, the first log lines say which configuration was
        actually running.
        """
        total = 0.0
        for name in model_files.DEFAULT_COMPONENTS:
            spec = model_files.get_component(name)
            if name == "transformer":
                factor = PRECISION_SIZE_FACTOR[self.transformer_precision]
            elif name == "text_encoder":
                factor = PRECISION_SIZE_FACTOR[self.text_encoder_precision]
            else:
                factor = 1.0
            total += spec.size_gb * factor
        return total

    @property
    def is_quantised(self) -> bool:
        return (
            self.transformer_precision != DEFAULT_TRANSFORMER_PRECISION
            or self.text_encoder_precision != DEFAULT_TEXT_ENCODER_PRECISION
        )

    def describe(self) -> str:
        """One-line startup summary."""
        return (
            f"transformer={self.transformer_precision} "
            f"text_encoder={self.text_encoder_precision} "
            f"zerogpu_size={self.zerogpu_size} "
            f"duration={self.gpu_duration_seconds}s "
            f"expected_weights={self.expected_weights_gb:.2f}GB "
            f"outputs<={self.output_retention_gb:.0f}GB"
        )


def load_config() -> RuntimeConfig:
    """Resolve the runtime configuration, raising on anything unrecognised."""
    return RuntimeConfig(
        transformer_precision=_choice(
            ENV_TRANSFORMER_PRECISION, DEFAULT_TRANSFORMER_PRECISION, TRANSFORMER_PRECISIONS
        ),
        text_encoder_precision=_choice(
            ENV_TEXT_ENCODER_PRECISION, DEFAULT_TEXT_ENCODER_PRECISION, TEXT_ENCODER_PRECISIONS
        ),
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
