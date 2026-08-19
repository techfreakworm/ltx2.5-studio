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

from ltx25_studio.ui import build_ui

demo = build_ui()


def main() -> None:
    demo.launch()


if __name__ == "__main__":
    main()
