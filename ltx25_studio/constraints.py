"""Dimension and frame-count validation, CRF selection, sigma schedules.

PLAN §5.5. Pure functions over numbers — no torch, no weights, no I/O — which
is what makes them testable with zero GPU (§9.1).

Notes that are easy to get wrong:
  * ``height % 32 == 0`` and ``width % 32 == 0`` ARE enforced upstream (raise).
  * ``num_frames % 8 == 1`` is NOT. It falls out of
    ``(num_frames - 1) // vae.temporal_compression_ratio + 1``, which silently
    floors a bad value. We validate it ourselves, and we derive the modulus from
    ``vae.temporal_compression_ratio`` rather than hardcoding 8 (§10 OQ-3).
  * LTX-2.5 image conditioning uses CRF 18, not the generic default of 33.

Imports nothing else from this package.
"""
