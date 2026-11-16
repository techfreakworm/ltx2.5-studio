"""User-uploadable LoRA -- the headline feature, and the most defensive code here.

PLAN §4. Split three ways so each stage is a unit testable against synthesised
safetensors headers, with no real LoRA file, no GPU and no network:

``inspect``
    Header parse, metadata, key-convention detection, the safetensors layout
    checks, and the content hash. Stdlib only; no tensor is ever built.
``validate``
    Stages 1-6.5, including the per-module dimension check against the
    checked-in transformer shape table (``manifest``, run as
    ``python -m ltx25_studio.lora.manifest`` to re-derive it), keys resolved
    the way diffusers resolves them (``keys``).
``fetch``
    Stage 0, the Hub route: repo ids only, size-bounded, pinned, cached.
``staging``
    The staged-adapter entry the UI keeps until Generate.
``apply``
    ``load_lora_weights`` plus stage 7, the post-attach assertion. The only
    module here that loads weights.

This package's exports are the authoritative interface; import from
``ltx25_studio.lora``, not from the submodules.
"""

from ..errors import (
    LoRAConventionError,
    LoRADimensionError,
    LoRAError,
    LoRAFetchError,
    LoRAFormatError,
    LoRAHeaderError,
    LoRAInContextError,
    LoRAModuleNotFoundError,
    LoRANotAttachedError,
    LoRAPairingError,
    LoRATooLargeError,
    LoRATruncatedError,
    LoRAVersionError,
)
from .fetch import (
    DIFFUSERS_LORA_WEIGHT_NAME,
    EVICTION_GRACE_SECONDS,
    FetchedLoRA,
    evict_lora_cache,
    fetch_lora,
    lora_cache_dir,
    mark_used,
)
from .inspect import LoRAInspection, inspect_lora, read_safetensors_header
from .staging import LoRAWeightError, StagedLoRA, check_weight, find_duplicate, staged_entry
from .validate import (
    DEFAULT_MAX_LORA_BYTES,
    MIN_RESOLVED_FRACTION,
    WARN_RESOLVED_FRACTION,
    LoRAInContextNameError,
    LoRAKeyNamingError,
    LoRAValidation,
    configured_max_bytes,
    validate_lora,
)

#: Re-exported so a caller catching LoRA failures needs one import, not two.
#: The error types live in ``ltx25_studio.errors`` (one hierarchy, one root);
#: these are aliases, not duplicates.
__all__ = [
    "DEFAULT_MAX_LORA_BYTES",
    "DIFFUSERS_LORA_WEIGHT_NAME",
    "EVICTION_GRACE_SECONDS",
    "MIN_RESOLVED_FRACTION",
    "WARN_RESOLVED_FRACTION",
    "FetchedLoRA",
    "LoRAConventionError",
    "LoRADimensionError",
    "LoRAError",
    "LoRAFetchError",
    "LoRAFormatError",
    "LoRAHeaderError",
    "LoRAInContextError",
    "LoRAInContextNameError",
    "LoRAInspection",
    "LoRAKeyNamingError",
    "LoRAModuleNotFoundError",
    "LoRANotAttachedError",
    "LoRAPairingError",
    "LoRATooLargeError",
    "LoRATruncatedError",
    "LoRAValidation",
    "LoRAVersionError",
    "LoRAWeightError",
    "StagedLoRA",
    "check_weight",
    "configured_max_bytes",
    "evict_lora_cache",
    "fetch_lora",
    "find_duplicate",
    "inspect_lora",
    "lora_cache_dir",
    "mark_used",
    "read_safetensors_header",
    "staged_entry",
    "validate_lora",
]
