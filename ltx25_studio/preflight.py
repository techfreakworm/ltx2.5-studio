"""Startup assertions that fail loudly before anything expensive happens.

PLAN §5.6 and §1.3. Two jobs:

1. **Version fail-fast.** ``diffusers.__version__`` must be ``"0.40.0.dev0"``
   (the pinned SHA). Release 0.39.0 imports ``LTX2Pipeline`` fine and only fails
   LATE, on a missing ``temporal_latent_upsampler``, after ~70 GB of download.

2. **Resource preflight.** Measure free RAM and free disk, compare against the
   requirement computed from the ``model_files`` table, and raise an error that
   NAMES THE NUMBERS if short -- e.g. "need ~67.2 GiB RAM for the bf16 load, saw
   41.3 GiB". This turns an anonymous OOM, or a "Workload evicted, storage limit
   exceeded (150G)", into a diagnosable message that names the fix (PLAN §2.6's
   quantised fallbacks).

Both are test cases: inject fake numbers, assert the specific error.
"""
