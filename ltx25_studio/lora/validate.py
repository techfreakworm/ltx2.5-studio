"""Validator stages 1-6, run before any weights are touched.

PLAN §4.3:
  1. size bound            reject oversized uploads before download or parse
  2. header-only parse     never load the payload (see ``inspect``)
  3. format                safetensors with a parseable header, or a specific error
  4. version               warn if ``model_version`` is absent, reject on mismatch
  5. convention            on unknown, the error NAMES the prefixes actually found
  6. pairing               ``lora_A``/``lora_B`` (or ``lora_down``/``lora_up``) must pair

Accepts either a Hub repo id or a direct file upload.
"""
