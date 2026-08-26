"""Layering rules from PLAN §7.2, as tests.

§7.3 is explicit about why these matter: the predecessor's empty-output bug lived
"precisely in the seam between 'workflow returned something' and 'UI rendered
it', and there was no layer boundary at which to assert." These tests are that
boundary, kept honest automatically.

One deliberate reading: §7.2 says ``model_files.py``, ``errors.py`` and
``constraints.py`` "import nothing from the package". Taken literally that
contradicts §5.5/§4.3, which require ``constraints`` to raise the typed errors
defined in ``errors``. The rule enforced here is the one that makes both true:
**they may import ``errors`` and nothing else.** ``errors`` itself imports
nothing internal at all.
"""

from __future__ import annotations

import ast

import pytest
from conftest import PACKAGE_ROOT, package_python_files

LEAF_MODULES = ("errors.py", "constraints.py", "model_files.py")


def _internal_imports(path):
    """Names of ``ltx25_studio`` submodules this file imports."""
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ltx25_studio"):
                    found.add(alias.name.replace("ltx25_studio.", "").split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                found.add(node.module.split(".")[0])
            elif node.level and not node.module:
                found.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith("ltx25_studio"):
                remainder = node.module.replace("ltx25_studio", "").lstrip(".")
                if remainder:
                    found.add(remainder.split(".")[0])
                else:
                    found.update(alias.name for alias in node.names)
    return found


def _third_party_imports(path):
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module.split(".")[0])
    return found


# --------------------------------------------------------------------------
# dependencies point inward
# --------------------------------------------------------------------------


@pytest.mark.parametrize("filename", LEAF_MODULES)
def test_leaf_modules_import_nothing_internal_but_errors(filename):
    imports = _internal_imports(PACKAGE_ROOT / filename)
    assert imports <= {"errors"}, f"{filename} imports {sorted(imports - {'errors'})} (PLAN §7.2)"


def test_errors_imports_nothing_from_the_package():
    assert _internal_imports(PACKAGE_ROOT / "errors.py") == set()


def test_generate_does_not_import_the_ui():
    """PLAN §7.2: "``generate.py`` must not import ``ui/``." The UI is the only
    layer that turns an exception into a string for a human."""
    assert "ui" not in _internal_imports(PACKAGE_ROOT / "generate.py")


@pytest.mark.parametrize(
    "path",
    [p for p in package_python_files() if "ui" not in p.parts and p.name != "errors.py"],
    ids=lambda p: str(p.relative_to(PACKAGE_ROOT)),
)
def test_nothing_outside_the_ui_layer_imports_gradio(path):
    """``errors.py`` is deliberately absent from this sweep — it holds the one
    non-ui gradio licence, and the test below pins that licence to exactly one
    imported name rather than exempting the file wholesale."""
    assert "gradio" not in _third_party_imports(path), (
        f"{path.name} imports gradio — that is what makes the app untestable without it (§7.2)"
    )


def test_errors_gradio_licence_is_exactly_the_exception_base():
    """The ZeroGPU fork boundary transmits an exception *object* back to the
    parent only when it is a ``gr.Error`` (docs/SPACE_MODE_MATRIX.md §4);
    everything else is reduced to its class name, which is how a fully
    diagnosed ``DecodeError(reason=...)`` reached a live user as the string
    ``'DecodeError'``. The exception is constructed in the worker, in library
    code, so the isinstance property must hold at the root of the hierarchy —
    it cannot be granted from ``ui/`` after the fact. That makes ``errors.py``
    the one module outside ``ui/`` with a reason to touch gradio, and this
    test keeps the licence exactly that narrow: ``gradio.exceptions.Error``,
    nothing more. A second gradio name appearing here re-fails the layering
    rule rather than widening the exemption silently."""
    tree = ast.parse((PACKAGE_ROOT / "errors.py").read_text())
    gradio_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            gradio_imports += [a.name for a in node.names if a.name.split(".")[0] == "gradio"]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and not node.level
            and node.module.split(".")[0] == "gradio"
        ):
            gradio_imports += [f"{node.module}.{a.name}" for a in node.names]
    assert gradio_imports == ["gradio.exceptions.Error"], (
        f"errors.py's gradio licence covers exactly the exception base; found {gradio_imports}"
    )


@pytest.mark.parametrize(
    "path",
    [p for p in package_python_files() if "ui" not in p.parts],
    ids=lambda p: str(p.relative_to(PACKAGE_ROOT)),
)
def test_nothing_outside_the_ui_layer_imports_spaces(path):
    """``spaces`` is a deploy-environment package. Importing it from library code
    makes the library unimportable off-Space."""
    assert "spaces" not in _third_party_imports(path), path.name


def test_handlers_hold_no_business_logic():
    """PLAN §7.2: every handler marshals inputs, calls exactly one function, and
    marshals outputs. A handler that imports torch is doing something else."""
    handlers = PACKAGE_ROOT / "ui" / "handlers.py"
    imported = _third_party_imports(handlers)
    forbidden = imported & {"torch", "diffusers", "transformers", "safetensors", "huggingface_hub"}
    assert not forbidden, f"ui/handlers.py imports {sorted(forbidden)} — that is business logic"


# --------------------------------------------------------------------------
# the module layout is the one §7.1 describes
# --------------------------------------------------------------------------

EXPECTED_MODULES = (
    "config.py",
    "model_files.py",
    "errors.py",
    "constraints.py",
    "pipelines.py",
    "conditioning.py",
    "audio.py",
    "modes.py",
    "generate.py",
    "enhance.py",
    "outputs.py",
    "lora/inspect.py",
    "lora/validate.py",
    "lora/apply.py",
    "ui/blocks.py",
    "ui/components.py",
    "ui/handlers.py",
)


@pytest.mark.parametrize("relative", EXPECTED_MODULES)
def test_module_from_the_plan_exists(relative):
    assert (PACKAGE_ROOT / relative).is_file(), f"PLAN §7.1 lists {relative}"


@pytest.mark.parametrize(
    "path", package_python_files(), ids=lambda p: str(p.relative_to(PACKAGE_ROOT))
)
def test_every_module_is_importable_without_a_gpu(path):
    """The cheapest possible guard against an import-time ``torch.cuda`` call or a
    module-level model load."""
    import importlib

    relative = path.relative_to(PACKAGE_ROOT)
    dotted = ".".join(("ltx25_studio", *relative.with_suffix("").parts))
    if dotted.endswith(".__init__"):
        dotted = dotted[: -len(".__init__")]
    importlib.import_module(dotted)


@pytest.mark.parametrize(
    "path",
    [p for p in package_python_files() if p.name != "__init__.py"],
    ids=lambda p: str(p.relative_to(PACKAGE_ROOT)),
)
def test_no_module_has_a_docstring_only_stub_left_behind(path):
    """Every module in §7.1 owes the build something. A file that is still only a
    docstring at gate time is an unimplemented promise, not a design."""
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if not isinstance(node, ast.Expr)]
    assert body, f"{path.relative_to(PACKAGE_ROOT)} is still a docstring-only stub"
