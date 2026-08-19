"""User-uploadable LoRA -- the headline feature, and the most defensive code here.

PLAN §4. Split three ways so each stage is a unit testable against synthesised
safetensors headers, with no real LoRA file, no GPU and no network:

``inspect``
    Header-only parse, metadata, key-convention detection. Stdlib only, which is
    what lets the Hub route range-read a remote adapter's header *before*
    downloading it (§4.6).
``validate``
    Stages 1-6.5, including the per-module dimension check that catches an
    adapter built for a different LTX model.
``apply``
    ``load_lora_weights`` plus stage 7, the post-attach assertion. The only
    module here that may import diffusers or peft.

This package's exports are the authoritative interface; import from
``ltx25_studio.lora``, not from the submodules.
"""

from .inspect import LoRAInspection, inspect_lora, read_safetensors_header
from .validate import DEFAULT_MAX_LORA_BYTES, LoRAValidation, validate_lora

__all__ = [
    "DEFAULT_MAX_LORA_BYTES",
    "LoRAInspection",
    "LoRAValidation",
    "inspect_lora",
    "read_safetensors_header",
    "validate_lora",
]
