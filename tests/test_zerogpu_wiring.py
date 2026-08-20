"""Spec for ZeroGPU actually being wired in — PLAN §8.2, §8.3, §8.5.

**The rule was written down at length and then not executed.** PLAN §8 argues
the ZeroGPU design in detail; ``zerogpu.py`` implements it carefully; and for a
while the app had **no executable ``@spaces.GPU`` anywhere** — all seven matches
in the tree were inside docstrings — ``zerogpu.py`` was imported by nothing that
runs, and there was no executable ``.to("cuda")``. The suite stayed green,
because ``test_app.py`` checks that ``spaces`` is *imported* and imported early,
and nothing checked that a decorated function *exists*.

A docstring that describes a decorator and a decorator are indistinguishable to
``grep``. They are not indistinguishable to ``ast``, which is why every static
check here reads the parsed tree and never the file's text.

What the platform actually requires, and what each of these pins:

* ``@spaces.GPU(...)`` applied **statically, at module scope**. Dynamic or inner
  decoration fails the startup analyzer with "No @spaces.GPU function detected
  during startup" — a failure that only ever appears on the Space.
* the ``.to("cuda")`` that triggers tensor packing at **module scope**, not
  lazily inside the decorated function (§8.3), where it skips packing entirely.
* the decorated entrypoint reachable from the UI, or every generation executes
  in the web worker and never reserves a slot.
* the entrypoint calling the **real** ``generate``, not whatever the module
  attribute points at later — see the last section, which pins a bug that got
  past 1283 tests and two hand-written runtime probes.

Runtime checks import ``app`` for real. That has global side effects (it
registers the GPU dispatch), so the fixture restores them: a spec that drives
the real thing must still not leave the process changed for whatever runs next.
"""

from __future__ import annotations

import ast
import importlib
import sys

import pytest
from conftest import PACKAGE_ROOT, REPO_ROOT, package_python_files

from ltx25_studio import generate as generate_module

APP_PATH = REPO_ROOT / "app.py"
ALL_SOURCE = [*package_python_files(), APP_PATH]


# --------------------------------------------------------------------------
# AST helpers — prose cannot satisfy any of these
# --------------------------------------------------------------------------


def _decorator_target(decorator):
    """The thing being applied, unwrapping ``@d(...)`` to ``d``."""
    return decorator.func if isinstance(decorator, ast.Call) else decorator


def _is_spaces_gpu(decorator) -> bool:
    """True for ``@spaces.GPU`` / ``@spaces.GPU(...)``, in code only."""
    target = _decorator_target(decorator)
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "GPU"
        and isinstance(target.value, ast.Name)
        and target.value.id == "spaces"
    )


def gpu_decorated_functions(path):
    """``(name, lineno, is_module_scope)`` for every ``@spaces.GPU`` function."""
    tree = ast.parse(path.read_text())
    module_scope = {id(node) for node in tree.body}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(_is_spaces_gpu(decorator) for decorator in node.decorator_list):
            found.append((node.name, node.lineno, id(node) in module_scope))
    return found


def cuda_moves(path):
    """``(lineno, argument)`` for every executable ``.to("cuda")``-shaped call."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "to"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == "cuda":
                found.append((node.lineno, "cuda"))
            elif isinstance(arg, ast.Name):
                # ``built.to(device)`` — a variable move, which is the real one
                # in pipelines.py. Recorded so the spec below can accept either.
                found.append((node.lineno, arg.id))
    return found


@pytest.fixture
def imported_app():
    """Import ``app`` for real, then put the process back as it was.

    Importing registers the GPU dispatch on ``generate``. Leaving that set would
    quietly reroute every later test's ``generate()`` call through the worker
    entrypoint — harmless today, and exactly the kind of inherited global state
    that makes a suite order-dependent.
    """
    pytest.importorskip("gradio")
    previous_dispatch = generate_module.gpu_dispatch()
    sys.modules.pop("app", None)
    try:
        yield importlib.import_module("app")
    finally:
        generate_module.set_gpu_dispatch(previous_dispatch)


# --------------------------------------------------------------------------
# a decorated function has to actually exist
# --------------------------------------------------------------------------


def test_an_executable_spaces_gpu_decorated_function_exists():
    """The one the whole ZeroGPU design depends on, and the one that was absent.

    ``grep -rn '@spaces.GPU'`` once returned seven matches, every one of them
    inside a docstring, while the running app had none. This reads the AST, so a
    paragraph describing the decorator cannot satisfy it.
    """
    decorated = [
        (path.relative_to(REPO_ROOT), name, lineno)
        for path in ALL_SOURCE
        for name, lineno, _ in gpu_decorated_functions(path)
    ]

    assert decorated, (
        "no module in the tree defines a function decorated with @spaces.GPU. Without one, "
        "ZeroGPU's startup analyzer reports 'No @spaces.GPU function detected during startup' "
        "and no generation ever reserves a GPU slot — while every docstring in the repo goes "
        "on describing the decorator (PLAN §8.2)."
    )


def test_the_gpu_decorator_is_applied_statically_at_module_scope():
    """Dynamic or inner decoration fails the startup analyzer.

    A decorator applied inside a function, or reassigned at runtime, is invisible
    to the analyzer that scans the module at import.
    """
    nested = [
        (path.relative_to(REPO_ROOT), name, lineno)
        for path in ALL_SOURCE
        for name, lineno, module_scope in gpu_decorated_functions(path)
        if not module_scope
    ]

    assert not nested, (
        f"{nested} applies @spaces.GPU to a function that is not defined at module scope. "
        f"ZeroGPU's analyzer only sees module-scope decoration (PLAN §8.2)."
    )


def test_the_gpu_decorator_is_not_hidden_behind_a_conditional():
    """PLAN §8: "a *conditional* decorator is how a documented rule quietly stops
    being applied — the Space path stops being the path anyone runs."

    Off-Space ``spaces`` is a stub whose ``GPU`` returns the function unchanged,
    so the wiring can be identical everywhere and only its effect differs. There
    is therefore no reason for the decoration itself to be conditional.
    """
    tree = ast.parse(APP_PATH.read_text())
    conditional = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                _is_spaces_gpu(decorator) for decorator in inner.decorator_list
            ):
                conditional.append((inner.name, inner.lineno))

    assert not conditional, (
        f"{conditional} in app.py is decorated with @spaces.GPU inside an `if`. The stub makes "
        f"the unconditional form safe off-Space; a conditional one makes the Space path "
        f"untested (PLAN §8.2)."
    )


def test_the_tensor_move_that_triggers_packing_is_executable():
    """PLAN §8.3: build on CPU, then ``.to("cuda")`` at **module scope**.

    ZeroGPU intercepts that call, rewrites it to ``.to("cpu")`` and packs every
    tensor to disk. Doing it lazily inside the decorated function skips packing
    entirely and re-transfers from RAM on every cold worker — slower, and silent.
    """
    moves = [
        (path.relative_to(REPO_ROOT), lineno, arg)
        for path in ALL_SOURCE
        for lineno, arg in cuda_moves(path)
    ]

    assert moves, (
        'no executable `.to("cuda")` (or `.to(device)`) exists anywhere in the tree, so '
        "ZeroGPU never packs the weights (PLAN §8.3). Every docstring describing the packing "
        "model is describing something that does not happen."
    )


def test_the_app_asks_for_a_cuda_device_at_module_scope():
    """The call site, specifically: ``app.py`` must request the device itself.

    ``pipelines.get_pipeline`` takes ``device=`` and moves only when given one,
    so the packing happens exactly when the entrypoint asks for it — and nowhere
    else. If ``app.py`` stops asking, the move becomes dead code that still
    passes the check above.
    """
    tree = ast.parse(APP_PATH.read_text())
    asked = [
        node.lineno
        for node in ast.walk(tree)
        for keyword in getattr(node, "keywords", [])
        if isinstance(node, ast.Call)
        and keyword.arg == "device"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "cuda"
    ]

    assert asked, (
        'app.py never calls anything with device="cuda", so nothing at module scope triggers '
        "ZeroGPU's packing (PLAN §8.3)."
    )


# --------------------------------------------------------------------------
# and it has to be reachable at runtime, not merely present
# --------------------------------------------------------------------------


def test_importing_the_app_loads_the_zerogpu_module(imported_app):
    """``zerogpu.py`` implements PLAN §8 and was imported by nothing that runs.

    A module the entrypoint never loads is a design document with a test suite —
    which is precisely what it was.
    """
    assert "ltx25_studio.zerogpu" in sys.modules, (
        "importing app.py does not load ltx25_studio.zerogpu, so none of the §8 boundary "
        "logic — the size-kwarg guard, the CUDA-return check, the duration estimator — is "
        "on the path the Space actually runs."
    )


def test_importing_the_app_runs_preflight(imported_app):
    """PLAN §5.6: fail loudly on resources *before* the platform kills us."""
    assert "ltx25_studio.preflight" in sys.modules, (
        "importing app.py does not load ltx25_studio.preflight, so the version assertion and "
        "the cache-completeness check never run at startup."
    )


def test_the_app_registers_its_decorated_entrypoint_as_the_gpu_dispatch(imported_app):
    """The wiring that turns a decorated function into the path a run takes.

    Without it the UI calls ``generate()`` directly, the generation executes in
    the web worker, and no slot is ever reserved — while a perfectly good
    ``@spaces.GPU`` function sits there decorated and uncalled.
    """
    dispatch = generate_module.gpu_dispatch()
    assert dispatch is not None, (
        "app.py defines a @spaces.GPU entrypoint but never registers it, so generate() runs "
        "in the web worker and never reserves a ZeroGPU slot."
    )
    assert dispatch is imported_app.run_generation, (
        f"the registered dispatch is {dispatch!r}, not app.run_generation — the decorated "
        f"function is not the one generations go through."
    )


def test_the_duration_estimator_matches_the_entrypoint_signature(imported_app):
    """A drifted estimator TypeErrors at *schedule* time, once the user is waiting."""
    from ltx25_studio import zerogpu

    zerogpu.check_duration_signature(imported_app.run_generation, zerogpu.estimate_duration)


# --------------------------------------------------------------------------
# the entrypoint must call the real generate, not the module attribute
# --------------------------------------------------------------------------


def _wrapper_supplying_the_gpu_seams(real, tmp_path, depth):
    """A wrapper on ``generate`` that supplies defaults, as instrumentation does.

    Entry has to be through ``generate()`` — the way the UI calls it — not
    through ``run_generation`` directly. That is the whole loop: UI → wrapper →
    real generate → dispatch → entrypoint → *wrapper again*. Calling the
    entrypoint first skips the outbound leg and the bug never fires, which is
    worth knowing: a behavioural test aimed at the wrong end of a cycle looks
    like a passing test.
    """
    from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

    def wrapper(**params):
        depth["current"] += 1
        depth["max"] = max(depth["max"], depth["current"])
        try:
            return real(
                pipeline_factory=factory_returning(FakePipe()),
                encoder=one_muxed_file_encoder,
                output_root=tmp_path,
                retention_bytes=10**9,
                **params,
            )
        finally:
            depth["current"] -= 1

    return wrapper


@pytest.mark.parametrize("mode_id", ["text_to_video", "text_to_audio_video"])
def test_a_wrapper_installed_after_startup_does_not_break_a_run(
    mode_id, imported_app, monkeypatch, tmp_path
):
    """The bug that passed 1283 tests and two hand-written runtime probes.

    ``run_generation`` resolved ``generate`` by **module attribute**, so a
    wrapper installed on ``generate`` after startup — instrumentation, a
    decorator, a seam test — sent the GPU worker straight back through
    ``generate()``, which dispatched to the worker again, which re-entered the
    wrapper. The wrapper then supplied ``pipeline_factory`` a second time, so the
    symptom was ``TypeError: got multiple values for keyword argument
    'pipeline_factory'`` — not a recursion error, which is exactly what made it
    read as a fixture problem rather than a product defect.
    """
    depth = {"current": 0, "max": 0}
    real = generate_module.generate
    monkeypatch.setattr(
        generate_module, "generate", _wrapper_supplying_the_gpu_seams(real, tmp_path, depth)
    )

    # Entering the way the UI does: through generate(), not the entrypoint.
    result = generate_module.generate(mode=mode_id, seed=7)

    assert result.video_path is not None
    assert depth["max"] <= 1, (
        f"the wrapper was re-entered {depth['max']} times for {mode_id!r}. run_generation "
        f"resolves generate by module attribute, so the GPU worker is sent back through the "
        f"dispatch and into the wrapper again."
    )


def test_a_run_through_the_dispatch_still_reserves_the_gpu_exactly_once(
    imported_app, monkeypatch, tmp_path
):
    """The other half: the fix must not simply bypass the entrypoint.

    Binding the real ``generate`` at import stops the loop. Binding it and then
    never routing through the decorated function would also stop the loop — and
    would silently give up the GPU slot, which is the defect the whole file
    exists to prevent.
    """
    from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

    entries = []
    real_entrypoint = generate_module.gpu_dispatch()

    def counting_entrypoint(**params):
        entries.append(params)
        return real_entrypoint(**params)

    generate_module.set_gpu_dispatch(counting_entrypoint)

    generate_module.generate(
        mode="text_to_video",
        seed=7,
        pipeline_factory=factory_returning(FakePipe()),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    assert len(entries) == 1, (
        f"the generation went through the @spaces.GPU entrypoint {len(entries)} times; it "
        f"must be exactly once — zero means no slot was reserved, more than one means the "
        f"worker re-entered the dispatch."
    )


def test_the_entrypoint_binds_generate_at_import_rather_than_resolving_it_late():
    """Stated structurally too, because the behavioural pins above can be satisfied
    by luck of ordering while the fragile shape remains.

    ``generate_module.generate(...)`` inside the decorated function is the shape
    that broke. A name bound at module scope is the shape that cannot.
    """
    tree = ast.parse(APP_PATH.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_is_spaces_gpu(decorator) for decorator in node.decorator_list):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "generate"
                and isinstance(inner.func.value, ast.Name)
            ):
                offenders.append((node.name, inner.lineno, inner.func.value.id))

    assert not offenders, (
        f"{offenders}: the @spaces.GPU entrypoint resolves `generate` through a module "
        f"attribute at call time. Any wrapper installed after startup then routes the GPU "
        f"worker back through the dispatch and into itself. Bind the real function at import."
    )


# --------------------------------------------------------------------------
# the return boundary
# --------------------------------------------------------------------------


def test_the_entrypoint_guards_what_it_returns(imported_app):
    """PLAN §8.5: a CUDA tensor crossing the boundary **hangs** the call.

    No error, no timeout, a dead request — the worst failure shape available. The
    guard exists in ``zerogpu.assert_returnable``; this pins that the entrypoint
    actually calls it.
    """
    source = ast.parse(APP_PATH.read_text())
    guarded = [
        node.name
        for node in ast.walk(source)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_is_spaces_gpu(decorator) for decorator in node.decorator_list)
        and any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "attr", getattr(inner.func, "id", None)) == "assert_returnable"
            for inner in ast.walk(node)
        )
    ]

    assert guarded, (
        "the @spaces.GPU entrypoint does not call zerogpu.assert_returnable on what it "
        "returns. Unpickling a CUDA tensor in the main process triggers a blocked lazy CUDA "
        "init and the call hangs with no error at all (PLAN §8.5)."
    )


def test_the_zerogpu_module_lives_in_the_package_not_the_entrypoint():
    """Guards the layering the wiring depends on: ``app.py`` stays thin."""
    assert (PACKAGE_ROOT / "zerogpu.py").is_file()
