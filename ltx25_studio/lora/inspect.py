"""Header-only safetensors inspection: metadata and key-convention detection.

PLAN §4.3 stage 2 is a hard rule, not a preference: read the safetensors header
and NEVER materialise the tensor payload to validate. A multi-GB upload must not
be fully loaded just to discover it is the wrong architecture.

What we read:
  * ``__metadata__.model_version`` — a real, machine-readable compatibility
    signal (the official LoRA reports ``'2.5.0'``, rank 450, alpha 450).
  * the key prefix, to classify the convention: ``diffusion_model.`` (ComfyUI
    style), ``transformer.`` (diffusers style), ``text_embedding_projection.``
    (targets the connectors), or unknown.

Both conventions do load — ``lora_state_dict()`` converts non-diffusers keys —
so detection is for validation and for telling the user what we found, not for
gatekeeping a format we cannot handle.
"""
