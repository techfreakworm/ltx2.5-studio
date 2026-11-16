"""Validator stages 1-6.5, run before any weights are touched.

PLAN §4.3. Accepts a local path or a Hub repo id. The Hub route resolves through
an injected ``fetcher`` (``lora.fetch_lora`` in the app), so the suite runs with
no network.

* **An absent version signal warns, it does not reject**: most real community
  adapters carry no ``model_version`` at all. An LTX-2.3 version also only
  warns; any other series is refused.
* **Stage 6.5 runs against the checked-in shape manifest of the pinned
  transformer** (``lora/manifest.py``), so it needs no weights and runs where
  the UI validates. Adapter keys are first resolved the way diffusers resolves
  them (``lora/keys.py``), so native names such as ``adaln_single`` are looked
  up under their diffusers names. A live transformer, when given, is checked
  against the manifest; a disagreement means the table no longer describes the
  deployed model and is an error.

Stage 7 (``lora/apply.py``) still runs at generation time: a clean report says
the file fits the model, not that it attached.

Dependency-light on purpose: stdlib plus this package. Only ``apply.py`` may
import diffusers or peft (``keys.py`` uses diffusers' key converter when it is
importable and an identical table when it is not).
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..errors import (
    LoRAConventionError,
    LoRADimensionError,
    LoRAError,
    LoRAFormatError,
    LoRAInContextError,
    LoRAModuleNotFoundError,
    LoRAPairingError,
    LoRATooLargeError,
    LoRAVersionError,
)
from .inspect import CONNECTOR_CONVENTIONS, KEY_PREFIXES, LoRAInspection, inspect_lora
from .keys import resolve_keys, split_suffix

__all__ = [
    "DEFAULT_MAX_LORA_BYTES",
    "DISTILL_MIN_RANK",
    "IC_LORA_MARKER",
    "LOADER_FACTOR_NAMES",
    "LORA_SUFFIX_PAIRS",
    "MIN_RESOLVED_FRACTION",
    "SUPPORTED_ARCHITECTURE_TOKEN",
    "SUPPORTED_SERIES",
    "UNREADABLE_FACTOR_NAMES",
    "WARN_RESOLVED_FRACTION",
    "WARNED_SERIES",
    "LoRAInContextNameError",
    "LoRAKeyNamingError",
    "LoRAValidation",
    "configured_max_bytes",
    "validate_lora",
]

#: Fallback upper bound on an accepted adapter, used only when the configured
#: ``LTX25_LORA_MAX_GB`` cannot be read. Clears the official distilled LoRA
#: (8.90 GB measured).
DEFAULT_MAX_LORA_BYTES = 12_000_000_000

#: The metadata key that marks an in-context LoRA (stage 5.5).
IC_LORA_MARKER = "reference_downscale_factor"

#: ``(down, up)`` suffix pairs. peft writes A/B; kohya and sd-scripts write down/up.
LORA_SUFFIX_PAIRS = (("lora_A", "lora_B"), ("lora_down", "lora_up"))

#: The factor names the pinned diffusers loads, and the ones it cannot: with
#: ``lora_down``/``lora_up`` it finds no rank and fails inside the load.
LOADER_FACTOR_NAMES = ("lora_A", "lora_B")
UNREADABLE_FACTOR_NAMES = ("lora_down", "lora_up")

#: The series this pipeline is. Compared on ``major.minor`` only.
SUPPORTED_SERIES = "2.5"

#: Earlier series whose adapters run on 2.5 (the vendor card says most do, and
#: measured 2.3 adapters resolve every module): a declared one warns.
WARNED_SERIES = ("2.3",)

#: Substring an ``modelspec.architecture`` value must contain to be this family.
SUPPORTED_ARCHITECTURE_TOKEN = "ltx2"

#: Fraction of an adapter's targeted modules that must exist in this model
#: after diffusers' key rename. Below the lower one the adapter is refused; below
#: the upper one it is accepted with a warning.
MIN_RESOLVED_FRACTION = 0.5
WARN_RESOLVED_FRACTION = 0.9

#: A full-coverage adapter at this rank or above is shaped like a distillation
#: adapter (the official one is rank 450 on all 1660 Linears).
DISTILL_MIN_RANK = 256

_LORA_SUBSTRING = "lora"
_IC_NAME = re.compile(r"(?<![A-Za-z0-9])ic[-_ ]?lora", re.IGNORECASE)
_IC_TAG = re.compile(r"^(ic[-_ ]?lora|in[-_ ]?context[-_ ]?lora)$", re.IGNORECASE)
_DISTILL_NAME = re.compile(r"distill", re.IGNORECASE)

#: Sentinel: "check against the checked-in manifest".
_MANIFEST = object()


class LoRAInContextNameError(LoRAInContextError):
    """Stage 5.5 by name, tag or base model, for an IC-LoRA whose metadata is empty."""

    def __init__(self, *, source, signal, value):
        self.source = source
        self.marker = signal
        self.value = value
        LoRAError.__init__(
            self,
            f"LoRA {source} is an in-context (IC) LoRA, marked as one by its {signal} "
            f"({value!r}). IC-LoRAs condition on a reference video or image and only work "
            f"through the in-context pipeline, which this app does not run. Applied to the "
            f"standard pipeline this adapter would load without error and have no effect.",
        )


class LoRAKeyNamingError(LoRAConventionError):
    """Stage 6: low-rank pairs named ``lora_down``/``lora_up``, which the loader cannot read.

    The keys are not converted: the file is refused before a run, naming both namings.
    """

    def __init__(self, *, source, keys):
        self.source = source
        self.keys = tuple(keys)
        self.found_prefixes = ()
        self.known_prefixes = ()
        found, wanted = UNREADABLE_FACTOR_NAMES, LOADER_FACTOR_NAMES
        LoRAError.__init__(
            self,
            f"LoRA {source} names its weight pairs {found[0]} and {found[1]} (the kohya "
            f"naming, in {len(self.keys)} tensors). This model's loader reads only pairs "
            f"named {wanted[0]} and {wanted[1]}, so it cannot load this file. Save the "
            f"adapter with {wanted[0]}/{wanted[1]} names and validate it again.",
        )


@dataclass(frozen=True)
class LoRAValidation:
    """What the header proved, and what could not be checked here."""

    path: Path
    inspection: LoRAInspection
    convention: str | None
    model_version: str | None
    warnings: tuple[str, ...]
    #: Transformer modules the adapter targets, and how many of them exist in
    #: the model after diffusers' key rename (stage 6.5).
    modules_targeted: int = 0
    modules_resolved: int = 0
    #: Distinct low-rank ranks found in the file.
    ranks: tuple[int, ...] = ()
    #: Set on the Hub route: the repo and the commit the file was fetched at.
    hub_repo: str | None = None
    hub_revision: str | None = None

    @property
    def rank(self) -> int | None:
        return max(self.ranks) if self.ranks else None

    @property
    def alpha(self) -> str | None:
        metadata = self.inspection.metadata
        return metadata.get("lora_alpha") or metadata.get("ss_network_alpha")

    @property
    def num_tensors(self) -> int:
        return len(self.inspection.keys)

    @property
    def targets_connectors(self) -> bool:
        return self.inspection.targets_connectors

    @property
    def content_key(self) -> str:
        """sha256 of the whole file: the same adapter under any file name, and two
        adapters that share a layout but not their weights kept apart."""
        return self.inspection.content_sha256


def configured_max_bytes() -> int:
    """``LTX25_LORA_MAX_GB`` in bytes, from the runtime config."""
    from ..config import load_config

    return load_config().lora_max_bytes


def _resolve(source, fetcher: Callable | None, upload: bool | None):
    """``(local path, fetched-or-None)`` for the route the caller named.

    ``upload=True``: a local file, never fetched. ``upload=False``: a Hub repo
    id, shape-checked as text before any filesystem call. ``None`` (the old
    contract): an existing local file is used as-is, anything else is fetched.
    """
    if upload is False:
        from .fetch import fetch_lora, parse_hub_reference

        parse_hub_reference(source)
        fetched = (fetcher or fetch_lora)(str(source))
    elif upload is True or fetcher is None:
        return Path(source), None
    else:
        path = Path(source)
        if path.exists():
            return path, None
        fetched = fetcher(str(source))
    hub = fetched if getattr(fetched, "repo_id", None) else None
    return Path(os.fspath(fetched)), hub


def _series(value: str) -> str:
    parts = str(value).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(value)


def _earlier_series(key: str, value: str) -> str:
    return (
        f"the adapter reports {key}={value!r}, an earlier LTX series. Most such adapters "
        f"load unchanged, but treat unexpected output as suspect."
    )


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
        if _series(declared) not in WARNED_SERIES:
            raise LoRAVersionError(source=source, found=declared, expected=SUPPORTED_SERIES)
        warnings.append(_earlier_series("model_version", declared))

    architecture = signals.get("modelspec.architecture")
    if architecture is not None and SUPPORTED_ARCHITECTURE_TOKEN not in architecture.lower():
        raise LoRAVersionError(
            source=source,
            found=architecture,
            expected=f"an LTX2 architecture (LTX2/lora) for LTX-{SUPPORTED_SERIES}",
        )

    value = signals.get("ss_ltx_version")
    if value and _series(value) != SUPPORTED_SERIES:
        warnings.append(_earlier_series("ss_ltx_version", value))


def _check_convention(inspection: LoRAInspection, source: str) -> None:
    """Stage 5 -- and the error names what it actually found."""
    if inspection.convention is not None:
        return
    raise LoRAConventionError(
        source=source,
        found_prefixes=inspection.prefixes,
        known_prefixes=tuple(KEY_PREFIXES),
    )


def _check_in_context(inspection: LoRAInspection, source: str, path: Path, hub) -> None:
    """Stage 5.5 -- the metadata marker, then the names and Hub card signals."""
    if IC_LORA_MARKER in inspection.metadata:
        raise LoRAInContextError(
            source=source, marker=IC_LORA_MARKER, value=inspection.metadata[IC_LORA_MARKER]
        )
    if _IC_NAME.search(path.name):
        raise LoRAInContextNameError(source=source, signal="file name", value=path.name)
    if hub is None:
        return
    for tag in getattr(hub, "tags", ()) or ():
        if _IC_TAG.match(str(tag)):
            raise LoRAInContextNameError(source=source, signal="Hub card tag", value=str(tag))
    if _IC_NAME.search(hub.repo_id):
        raise LoRAInContextNameError(source=source, signal="Hub repo name", value=hub.repo_id)
    for model in getattr(hub, "base_models", ()) or ():
        if _IC_NAME.search(str(model)):
            raise LoRAInContextNameError(source=source, signal="base model", value=str(model))


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


def _pairs(inspection: LoRAInspection) -> dict[str, dict[int, str]]:
    """Group keys by target -> {side_index: key}, for the pairing check."""
    grouped: dict[str, dict[int, str]] = {}
    for key in inspection.keys:
        split = split_suffix(key)
        if split is None:
            continue
        base, side = split
        grouped.setdefault(base, {})[side] = key
    return grouped


def _check_pairing(grouped: dict[str, dict[int, str]], source: str) -> None:
    """Stage 6 -- every down needs its up, and the error names only the orphans."""
    orphans = [sides[next(iter(sides))] for sides in grouped.values() if len(sides) < 2]
    if orphans:
        raise LoRAPairingError(source=source, unpaired_keys=tuple(sorted(orphans)))


def _check_factor_names(inspection: LoRAInspection, source: str) -> None:
    """Stage 6, after pairing: every factor carries a name the loader reads."""
    unreadable = [
        key
        for key in inspection.keys
        if any(f".{name}." in key or key.endswith(f".{name}") for name in UNREADABLE_FACTOR_NAMES)
    ]
    if unreadable:
        raise LoRAKeyNamingError(source=source, keys=unreadable)


def _ranks(inspection: LoRAInspection) -> tuple[int, ...]:
    ranks = set()
    for key, (_dtype, shape) in inspection.shapes.items():
        split = split_suffix(key)
        if split is not None and split[1] == 0 and len(shape) == 2:
            ranks.add(int(shape[0]))
    return tuple(sorted(ranks))


def _linear_dims(transformer) -> dict[str, tuple[int, int]]:
    """``name -> (in_features, out_features)`` for modules that look like ``nn.Linear``."""
    named_modules = getattr(transformer, "named_modules", None)
    if not callable(named_modules):
        return {}
    dims: dict[str, tuple[int, int]] = {}
    for name, module in named_modules():
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if isinstance(in_features, int) and isinstance(out_features, int):
            dims[name] = (in_features, out_features)
    return dims


def _shape(dims) -> str:
    return "absent" if dims is None else f"[{dims[1]}, {dims[0]}]"


def _cross_check(reference: dict, live: dict, targeted) -> None:
    """The live transformer must agree with the manifest wherever both speak."""
    from .manifest import LoRAManifestDriftError

    for name in sorted(reference.keys() & live.keys()):
        if reference[name] != live[name]:
            raise LoRAManifestDriftError(
                module=name, manifest=_shape(reference[name]), live=_shape(live[name])
            )
    for name in targeted:
        if (name in reference) != (name in live):
            raise LoRAManifestDriftError(
                module=name, manifest=_shape(reference.get(name)), live=_shape(live.get(name))
            )


def _dropped_module(key: str) -> str:
    """A key diffusers will ignore, named by the module it meant."""
    _, _, rest = key.partition(".")
    split = split_suffix(rest)
    return split[0] if split is not None else rest


def _check_dimensions(
    inspection: LoRAInspection, source: str, warnings: list[str], *, transformer, manifest
) -> tuple[int, int, int]:
    """Stage 6.5. Returns ``(targeted, resolved, reference size)``."""
    resolution = resolve_keys(inspection.keys)
    targeted = sorted(resolution.transformer)
    connector_prefixes = tuple(p for p, c in KEY_PREFIXES.items() if c in CONNECTOR_CONVENTIONS)
    dropped = sorted(
        {
            _dropped_module(key)
            for key in resolution.dropped
            if not key.startswith(connector_prefixes)
        }
    )
    targeted_count = len(targeted) + len(dropped)

    if manifest is _MANIFEST:
        from .manifest import load_manifest

        manifest = load_manifest()
    live = _linear_dims(transformer) if transformer is not None else {}
    if manifest is not None:
        reference = dict(manifest.dims)
        if live:
            _cross_check(reference, live, targeted)
    elif live:
        reference = live
    else:
        warnings.append(
            "per-module dimension compatibility (stage 6.5) was skipped: no transformer or "
            "shape table was available to compare against. An adapter built for a different "
            "LTX model can still pass every header check."
        )
        return targeted_count, 0, 0

    for module in targeted:
        if module not in reference:
            continue
        in_features, out_features = reference[module]
        for side, key in sorted(resolution.transformer[module].items()):
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

    missing = [module for module in targeted if module not in reference] + dropped
    resolved = targeted_count - len(missing)
    if not targeted_count or not missing:
        return targeted_count, resolved, len(reference)

    resolved_fraction = resolved / targeted_count
    if resolved_fraction < MIN_RESOLVED_FRACTION:
        raise LoRAModuleNotFoundError(
            source=source, missing_modules=tuple(sorted(missing)), module_count=targeted_count
        )
    severity = (
        "Most of the adapter will apply, but treat unexpected output as suspect"
        if resolved_fraction < WARN_RESOLVED_FRACTION
        else "This is a small enough fraction to be a normal version difference"
    )
    warnings.append(
        f"{len(missing)} of {targeted_count} targeted modules "
        f"({resolved_fraction:.0%} resolved) are absent from this transformer and will not "
        f"receive the adapter: {', '.join(sorted(missing)[:5])}. {severity}."
    )
    return targeted_count, resolved, len(reference)


def _distillation_warning(path: Path, hub, ranks, resolved: int, reference_size: int):
    """A distillation adapter is built for the dev transformer or another schedule."""
    names = [path.name] + ([hub.repo_id] if hub is not None else [])
    reason = None
    named = next((name for name in names if _DISTILL_NAME.search(name)), None)
    if named is not None:
        reason = f"its name, {named!r}, says so"
    elif reference_size and resolved == reference_size and ranks and max(ranks) >= DISTILL_MIN_RANK:
        reason = f"it covers every one of the model's {reference_size} layers at rank {max(ranks)}"
    if reason is None:
        return None
    return (
        f"this looks like a distillation adapter ({reason}). Those are made for the full, "
        f"non-distilled LTX-2.5 model or for a different number of sampling steps; this app "
        f"already runs the distilled model, so adding one usually makes results worse rather "
        f"than adding a style."
    )


def validate_lora(
    source,
    *,
    max_bytes: int | None = None,
    fetcher: Callable | None = None,
    transformer=None,
    upload: bool | None = None,
    manifest=_MANIFEST,
) -> LoRAValidation:
    """Run stages 1-6.5 and return a report, or raise a specific error.

    ``max_bytes=None`` uses the configured ``LTX25_LORA_MAX_GB``. ``upload``
    names the route: ``True`` for an uploaded file, ``False`` for a Hub repo id
    (only ``org/name[/file]`` is accepted), ``None`` for the old behaviour where
    an existing local path wins. ``manifest=None`` checks against ``transformer``
    alone instead of the checked-in shape table.
    """
    if max_bytes is None:
        max_bytes = configured_max_bytes()
    path, hub = _resolve(source, fetcher, upload)
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
    _check_in_context(inspection, label, path, hub)  # stage 5.5
    _check_lora_substring(inspection, label)

    _check_pairing(_pairs(inspection), label)  # stage 6
    _check_factor_names(inspection, label)
    targeted, resolved, reference_size = _check_dimensions(  # stage 6.5
        inspection, label, warnings, transformer=transformer, manifest=manifest
    )
    ranks = _ranks(inspection)
    distill = _distillation_warning(path, hub, ranks, resolved, reference_size)
    if distill is not None:
        warnings.append(distill)

    return LoRAValidation(
        path=path,
        inspection=inspection,
        convention=inspection.convention,
        model_version=inspection.model_version,
        warnings=tuple(warnings),
        modules_targeted=targeted,
        modules_resolved=resolved,
        ranks=ranks,
        hub_repo=getattr(hub, "repo_id", None),
        hub_revision=getattr(hub, "revision", None),
    )
