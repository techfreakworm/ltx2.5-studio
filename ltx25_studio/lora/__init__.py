"""User-uploadable LoRA — the headline feature, and the most defensive code here.

PLAN §4. Split three ways so each stage is a unit that can be tested against
synthesised safetensors headers, with no real LoRA file and no GPU (§9.1):

  ``inspect``   header-only parse, metadata, key-convention detection
  ``validate``  stages 1-6 of the validator
  ``apply``     ``load_lora_weights`` plus stage 7, the post-attach assertion
"""
