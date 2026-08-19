"""Pipeline construction, component wiring, caching, and the offload policy.

PLAN §8. ``enable_model_cpu_offload()`` is mandatory here, not a user-facing
toggle: ~72 GB of weights against a ZeroGPU allocation an order of magnitude
smaller. The transformer (37.98 GB) is the irreducible resident during
denoising; the text encoder (23.9 GB) is needed once and should be evicted
after prompt encoding.

Stage 2 (``LTX2LatentUpsamplePipeline``) is a separate pipeline and therefore a
separate residency window — which is what lets the audio-only modes skip it
(§3.2).
"""
