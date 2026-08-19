"""Load the adapter, then assert it actually attached. Stage 7.

PLAN §4.2 / §4.3 stage 7 — the check upstream does not do for us.

The only built-in validation is a substring test that raises
``"Invalid LoRA checkpoint. Make sure all LoRA param names contain `'lora'`
substring."``. A LoRA built for a completely different architecture PASSES that
test — it has ``lora`` in its key names — then attaches to ZERO modules, and
generation proceeds looking like the base model with no error raised.

So: after ``load_lora_weights``, assert the adapter attached to more than zero
modules and raise a specific error if it did not. Run this on every load,
including the built-in LoRAs.

``LTX2LoraLoaderMixin._lora_loadable_modules == ["transformer", "connectors"]`` —
note ``connectors``: a LoRA may legitimately touch the text-conditioning path.
"""
