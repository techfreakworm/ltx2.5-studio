"""Encode and mux video+audio, export a single track, and evict old outputs.

PLAN §2.5 / §10 OQ-6: the Space has ~78 GB of headroom shared between the pip
environment, user LoRA uploads and generated video. Outputs are not immortal —
bounded retention with oldest-first eviction, configured in ``config.py``.
"""
