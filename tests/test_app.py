"""Spec for ``app.py`` and ``ltx25_studio/ui/`` — PLAN §6, §7.1, §7.2.

Two things are being caught here, and neither needs a GPU:

1. **The app builds headlessly.** Assembling the Blocks with no weights and no
   CUDA is the cheapest possible smoke test for import-order breaks, wiring
   mistakes and handlers that reference a component that no longer exists. It is
   also the failure that costs the most to find on the Space instead of here.

2. **``import spaces`` comes first.** ZeroGPU raises *"CUDA has been initialized
   before importing the spaces package"* if anything touches torch first — and
   ``import ltx25_studio`` pulls in torch transitively, so the ordering rule
   covers our own package too, not just the obvious third-party names.

The gradio-dependent tests skip until gradio is installed; the import-order test
is pure static analysis and always runs.
"""

from __future__ import annotations

import ast
import importlib
import sys

import pytest
from conftest import REPO_ROOT

APP_PATH = REPO_ROOT / "app.py"

#: Importing any of these initialises CUDA (directly or transitively).
#: ``ltx25_studio`` is on the list because it imports torch downstream of
#: ``pipelines.py``.
HEAVY_IMPORTS = {"torch", "diffusers", "transformers", "ltx25_studio", "accelerate", "torchaudio"}


def _module_level_imports(path):
    """(lineno, root module name) for every import in the file, in source order.

    Descends into ``try`` blocks so the common
    ``try: import spaces / except ImportError: spaces = None`` shape is seen.
    """
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                continue
            if node.module:
                found.append((node.lineno, node.module.split(".")[0]))
    return sorted(found)


# --------------------------------------------------------------------------
# ZeroGPU import order (static — always runs)
# --------------------------------------------------------------------------


def test_app_imports_the_spaces_package():
    """PLAN §6: the target is a ZeroGPU Space. ``spaces`` is not optional there."""
    roots = {root for _, root in _module_level_imports(APP_PATH)}
    assert "spaces" in roots, (
        "app.py must import `spaces` — ZeroGPU needs it, and it must be imported before "
        "anything that initialises CUDA"
    )


def test_spaces_is_imported_before_anything_that_initialises_cuda():
    imports = _module_level_imports(APP_PATH)
    spaces_lines = [lineno for lineno, root in imports if root == "spaces"]
    heavy = [(lineno, root) for lineno, root in imports if root in HEAVY_IMPORTS]
    if not heavy:
        pytest.skip("app.py imports nothing heavy yet")
    assert spaces_lines, "app.py imports torch/diffusers/ltx25_studio but never imports spaces"
    first_heavy_line, first_heavy_name = heavy[0]
    assert min(spaces_lines) < first_heavy_line, (
        f"`import {first_heavy_name}` on line {first_heavy_line} precedes `import spaces` on "
        f"line {min(spaces_lines)}. ZeroGPU raises 'CUDA has been initialized before importing "
        "the spaces package'."
    )


def test_app_is_thin():
    """PLAN §7.1: "Thin — builds UI, launches." Logic belongs in the package."""
    tree = ast.parse(APP_PATH.read_text())
    top_level_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert len(top_level_defs) <= 2, (
        f"app.py defines {[n.name for n in top_level_defs]} — the entrypoint should build the "
        "UI and launch it, nothing else (PLAN §7.1)"
    )


def test_app_does_not_launch_at_import_time():
    """A module-level ``launch()`` makes ``app.py`` unimportable by anything —
    including this test file and any future tooling."""
    tree = ast.parse(APP_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert name != "launch", "launch() must sit behind a __main__ guard"


# --------------------------------------------------------------------------
# headless build (needs gradio; skipped until it is installed)
# --------------------------------------------------------------------------


def test_the_blocks_build_with_no_gpu_and_no_weights():
    gradio = pytest.importorskip("gradio")
    from ltx25_studio.ui import blocks

    demo = blocks.build_ui()
    assert isinstance(demo, gradio.Blocks)


def test_importing_the_app_does_not_launch_a_server(monkeypatch):
    gradio = pytest.importorskip("gradio")
    launched = []
    monkeypatch.setattr(
        gradio.Blocks, "launch", lambda self, *a, **k: launched.append(True), raising=False
    )
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    assert not launched, "importing app.py must not start a server"
    assert hasattr(module, "main") or hasattr(module, "demo"), (
        "app.py should expose the built UI (`demo`) or an explicit `main()` entrypoint"
    )


def test_building_the_blocks_twice_is_safe():
    """Gradio keeps global state; a build that only works once breaks reloads."""
    pytest.importorskip("gradio")
    from ltx25_studio.ui import blocks

    assert blocks.build_ui() is not blocks.build_ui()


def test_launching_bounds_the_upload_size(monkeypatch):
    """Uploads land on gradio's tmp disk BEFORE any check of ours can run, and
    the retention sweep covers outputs/ only -- so an unbounded launch lets one
    request fill the disk with a file nothing ever reclaims. The bound must
    cover the LoRA validator's own ceiling (the largest upload the app has a
    use for) so a ceiling-sized adapter is refused by stage 1's message naming
    the limit, not by gradio's blunter transport error -- and it must still BE
    a bound, not a shrug."""
    gradio = pytest.importorskip("gradio")
    captured = {}
    monkeypatch.setattr(
        gradio.Blocks,
        "launch",
        lambda self, *a, **k: captured.update(k) or None,
        raising=False,
    )
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    module.main()

    from ltx25_studio.config import load_config

    ceiling = load_config().lora_max_bytes
    assert captured.get("max_file_size") is not None, (
        "demo.launch() must pass max_file_size -- without it any-size uploads reach tmp disk"
    )
    assert captured["max_file_size"] >= ceiling
    assert captured["max_file_size"] <= ceiling * 2
