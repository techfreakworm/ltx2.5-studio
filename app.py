"""Entrypoint for the Hugging Face Space and for local runs.

``import spaces`` MUST come before anything that initialises CUDA. ZeroGPU
raises *"CUDA has been initialized before importing the spaces package"*
otherwise, and ``ltx25_studio`` pulls in torch transitively -- so the rule covers
our own package too, not just the obvious third-party names. That ordering is
the entire reason this file exists as something other than two lines, and it is
why the ``spaces`` import lives here rather than in ``ltx25_studio/zerogpu.py``:
a deploy-environment package imported from library code makes the library
unimportable off-Space.

Deliberately thin (PLAN §7.1): build the UI, launch it. Everything else lives in
the package.
"""

try:  # noqa: I001 - import order is load-bearing; see the module docstring
    import spaces  # noqa: F401
except ImportError:  # local development, where ZeroGPU does not exist
    spaces = None

import logging

from ltx25_studio import outputs
from ltx25_studio.config import load_config
from ltx25_studio.ui import build_ui

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("ltx25_studio")

# PLAN §2.7: sweep on startup as well as after each run. ZeroGPU's disk is
# ephemeral but a restart is not a guaranteed clean slate, and the predecessor
# went unhealthy by filling it. Non-fatal: a retention failure must not stop the
# app from starting.
try:
    _config = load_config()
    for _evicted in outputs.enforce_retention(
        _config.output_dir, max_bytes=int(_config.output_retention_gb * 1e9)
    ):
        _log.info("startup eviction: removed old run %s", _evicted)
except Exception as _exc:  # noqa: BLE001 - startup must survive a bad sweep
    _log.warning("startup output sweep skipped: %s", _exc)

demo = build_ui()


def main() -> None:
    demo.launch()


if __name__ == "__main__":
    main()
