"""THE single source of truth for every Hub repo id, revision, subfolder and filename.

PLAN §5.2. Data only — no logic, no I/O. An upstream rename must be a one-line
change here and nowhere else. A missing file must produce an error naming both
the file and its URL (§5.2), because Lightricks has deleted a checkpoint from
the Hub with no warning before.

The shipped subset and its arithmetic are §2.3; the duplicate-shard trap that
makes the ``preload_from_hub`` list dangerous to get wrong is §2.1.

Blocked on §10 OQ-1 (which diffusers revision) and OQ-2 (whether
``diffusion_decoder`` / ``duration_head`` have a consumer at all).

Imports nothing else from this package.
"""
