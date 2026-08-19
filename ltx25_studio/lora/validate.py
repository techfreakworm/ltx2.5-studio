"""Validator stages 1-6.5, run before any weights are touched.

PLAN §4.3. Upstream's entire validation is one substring test -- every stage
here is something nobody else does for us.

Accepts **either** a local path or a Hub repo id, and validates both
identically. The Hub route resolves through an injected ``fetcher`` so the
header can be range-read *before* downloading the payload (§4.6), and so the
test suite runs with no network.

Two rules that look like leniency and are not:

* **An absent version signal warns, it does not reject.** Not one of the three
  real community adapters measured in §4.5 carries ``model_version``; two carry
  no metadata at all. Being strict here would break the headline feature for the
  majority of real uploads.
* **Stage 6.5 is skipped when no transformer is available** -- the pre-load UI
  case, and the Hub case before anything is downloaded. Skipping is correct;
  skipping *silently* is not, so the report says so.

A clean report is not permission to generate. Stage 7 (``lora/apply.py``) is the
check that actually protects the user, and it cannot run here.

Dependency-light on purpose: stdlib plus ``inspect``. Only ``apply.py`` may
import diffusers or peft.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..errors import (
    LoRAConventionError,
    LoRADimensionError,
    LoRAFormatError,
    LoRAInContextError,
    LoRAModuleNotFoundError,
    LoRAPairingError,
    LoRATooLargeError,
    LoRAVersionError,
)
from .inspect import KEY_PREFIXES, LoRAInspection, inspect_lora

__all__ = [
    "DEFAULT_MAX_LORA_BYTES",
    "IC_LORA_MARKER",
    "LORA_SUFFIX_PAIRS",
    "SUPPORTED_ARCHITECTURE_TOKEN",
    "SUPPORTED_SERIES",
    "LoRAValidation",
    "validate_lora",
]

#: Upper bound on an accepted adapter. Must clear the official distilled LoRA at
#: 9.699 GB -- a limit below that rejects the adapter the model itself ships --
#: while staying well inside the ~78 GB of headroom the disk budget leaves.
DEFAULT_MAX_LORA_BYTES = 12_000_000_000

#: The metadata key that marks an in-context LoRA (stage 5.5).
IC_LORA_MARKER = "reference_downscale_factor"

#: ``(down, up)`` suffix pairs. peft writes A/B; kohya and sd-scripts write
#: down/up. Both are common and both load.
LORA_SUFFIX_PAIRS = (("lora_A", "lora_B"), ("lora_down", "lora_up"))

#: The series this pipeline is. Compared on ``major.minor`` only.
SUPPORTED_SERIES = "2.5"

#: Substring an ``modelspec.architecture`` value must contain to be this family.
SUPPORTED_ARCHITECTURE_TOKEN = "ltx2"

_LORA_SUBSTRING = "lora"


@dataclass(frozen=True)
class LoRAValidation:
    """What the header proved, and what could not be checked here."""

    path: Path
    inspection: LoRAInspection
    convention: str | None
    model_version: str | None
    warnings: tuple[str, ...]


def _resolve(source, fetcher: Callable | None) -> Path:
    """A local file is used as-is; anything else is a Hub reference."""
    path = Path(source)
    if path.exists():
        return path
    if fetcher is not None:
        return Path(fetcher(str(source)))
    return path


def _series(value: str) -> str:
    parts = str(value).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(value)


def _check_version(inspection: LoRAInspection, source: str, warnings: list[str]) -> None:
    """Stage 4 -- four signals in priority order (§4.5)."""
    signals = inspection.version_signals
    if not signals:
        warnings.append(
            "no model_version or other version signal in the adapter metadata, so compatibility "
            "could not be confirmed from the header. This is normal for community adapters."
        )
        return

    declared = signals.get("model_version")
    if declared is not None and _series(declared) != SUPPORTED_SERIES:
        raise LoRAVersionError(source=source, found=declared, expected=SUPPORTED_SERIES)

    architecture = signals.get("modelspec.architecture")
    if architecture is not None and SUPPORTED_ARCHITECTURE_TOKEN not in architecture.lower():
        raise LoRAVersionError(
            source=source,
            found=architecture,
            expected=f"an LTX2 architecture (LTX2/lora) for LTX-{SUPPORTED_SERIES}",
        )

    # Secondary trainer-written signals only warn. The model card states that
    # most LTX-2.3 adapters run on 2.5 unchanged, and our own reference fixture
    # is one -- so a 2.3 here is a note, not a defect.
    for key in ("ss_ltx_version", "ss_base_model_version"):
        value = signals.get(key)
        if value and key == "ss_ltx_version" and _series(value) != SUPPORTED_SERIES:
            warnings.append(
                f"the adapter reports {key}={value!r}, an earlier LTX series. Most such adapters "
                f"load unchanged, but treat unexpected output as suspect."
            )


def _check_convention(inspection: LoRAInspection, source: str) -> None:
    """Stage 5 -- and the error names what it actually found."""
    if inspection.convention is not None:
        return
    raise LoRAConventionError(
        source=source,
        found_prefixes=inspection.prefixes,
        known_prefixes=tuple(KEY_PREFIXES),
    )


def _check_in_context(inspection: LoRAInspection, source: str) -> None:
    """Stage 5.5 -- a header check, so it fires on the Hub route too."""
    if IC_LORA_MARKER in inspection.metadata:
        raise LoRAInContextError(
            source=source, marker=IC_LORA_MARKER, value=inspection.metadata[IC_LORA_MARKER]
        )


def _check_lora_substring(inspection: LoRAInspection, source: str) -> None:
    """Upstream's only check, done earlier and with a better message."""
    stray = [key for key in inspection.keys if _LORA_SUBSTRING not in key]
    if stray:
        raise LoRAFormatError(
            source=source,
            reason=(
                f"{len(stray)} tensor(s) are not LoRA parameters -- every key must contain "
                f"'lora'. First offender: {stray[0]!r}"
            ),
        )


def _split_suffix(key: str) -> tuple[str, str, int] | None:
    """``(base_module, side_token, side_index)`` for a low-rank factor key."""
    for pair_index, pair in enumerate(LORA_SUFFIX_PAIRS):
        for side, token in enumerate(pair):
            marker = f".{token}."
            if marker in key:
                return key.split(marker)[0], f"{pair_index}", side
            if key.endswith(f".{token}"):
                return key[: -len(token) - 1], f"{pair_index}", side
    return None


def _pairs(inspection: LoRAInspection) -> dict[str, dict[int, str]]:
    """Group keys by target module -> {side_index: key}."""
    grouped: dict[str, dict[int, str]] = {}
    for key in inspection.keys:
        split = _split_suffix(key)
        if split is None:
            continue
        base, _, side = split
        grouped.setdefault(base, {})[side] = key
    return grouped


def _check_pairing(grouped: dict[str, dict[int, str]], source: str) -> None:
    """Stage 6 -- every down needs its up, and the error names only the orphans."""
    orphans = [sides[next(iter(sides))] for sides in grouped.values() if len(sides) < 2]
    if orphans:
        raise LoRAPairingError(source=source, unpaired_keys=tuple(sorted(orphans)))


def _linear_dims(transformer) -> dict[str, tuple[int, int]]:
    """Walk the transformer for modules that look like ``nn.Linear``.

    Built from the loaded model, never a hardcoded table -- that is what keeps
    stage 6.5 correct across checkpoints, and what stops a single
    ``num_attention_heads x attention_head_dim`` number from being mistaken for
    the rule.
    """
    dims: dict[str, tuple[int, int]] = {}
    for name, module in transformer.named_modules():
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if isinstance(in_features, int) and isinstance(out_features, int):
            dims[name] = (in_features, out_features)
    return dims


def _check_dimensions(
    inspection: LoRAInspection,
    grouped: dict[str, dict[int, str]],
    transformer,
    source: str,
    warnings: list[str],
) -> None:
    """Stage 6.5 -- per module, against this transformer's own dimensions."""
    if transformer is None:
        warnings.append(
            "per-module dimension compatibility (stage 6.5) was skipped: no transformer was "
            "available to compare against. An adapter built for a different LTX model can still "
            "pass every header check."
        )
        return

    expected = _linear_dims(transformer)
    if not expected:
        warnings.append(
            "per-module dimension compatibility (stage 6.5) was skipped: the transformer exposed "
            "no linear modules to compare against."
        )
        return

    targeted, missing = 0, []
    for base, sides in grouped.items():
        prefix = next((p for p in KEY_PREFIXES if base.startswith(p)), None)
        # Connector-targeting keys are not checked against the transformer.
        if prefix is None or KEY_PREFIXES[prefix] not in ("native", "diffusers"):
            continue
        module = base[len(prefix) :]
        targeted += 1
        if module not in expected:
            missing.append(module)
            continue
        in_features, out_features = expected[module]
        for side, key in sides.items():
            shape = inspection.shapes.get(key, (None, ()))[1]
            if len(shape) != 2:
                continue
            # lora_A/down is [rank, in_features]; lora_B/up is [out_features, rank].
            actual = shape[1] if side == 0 else shape[0]
            want = in_features if side == 0 else out_features
            if actual != want:
                raise LoRADimensionError(
                    source=source, module=module, expected=want, actual=actual, key=key
                )

    if missing and len(missing) == targeted:
        raise LoRAModuleNotFoundError(
            source=source, missing_modules=tuple(sorted(missing)), module_count=targeted
        )
    if missing:
        warnings.append(
            f"{len(missing)} of {targeted} targeted modules are absent from this transformer "
            f"and will not receive the adapter: {', '.join(sorted(missing)[:5])}."
        )


def validate_lora(
    source,
    *,
    max_bytes: int = DEFAULT_MAX_LORA_BYTES,
    fetcher: Callable | None = None,
    transformer=None,
) -> LoRAValidation:
    """Run stages 1-6.5 and return a report, or raise a specific error."""
    path = _resolve(source, fetcher)
    label = str(path)
    warnings: list[str] = []

    # Stage 1 runs BEFORE any parse, so an oversized file is refused for its
    # size rather than for whatever its contents turn out to be.
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise LoRAFormatError(source=label, reason=f"cannot be read ({exc.strerror})") from exc
    if size_bytes > max_bytes:
        raise LoRATooLargeError(source=label, size_bytes=size_bytes, max_bytes=max_bytes)

    inspection = inspect_lora(path)  # stages 2-3

    _check_version(inspection, label, warnings)  # stage 4
    _check_convention(inspection, label)  # stage 5
    _check_in_context(inspection, label)  # stage 5.5
    _check_lora_substring(inspection, label)

    grouped = _pairs(inspection)
    _check_pairing(grouped, label)  # stage 6
    _check_dimensions(inspection, grouped, transformer, label, warnings)  # stage 6.5

    return LoRAValidation(
        path=path,
        inspection=inspection,
        convention=inspection.convention,
        model_version=inspection.model_version,
        warnings=tuple(warnings),
    )
