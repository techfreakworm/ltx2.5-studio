"""Spec: every module PLAN §7.1 declares is actually REACHED, not merely present.

**Four of this build's blockers had one signature**: a module that exists, whose
own tests are green, whose docstring correctly describes what it does — and that
nothing ever calls.

* ``zerogpu.py`` — the whole §8 boundary, never imported by the running app.
* ``outputs.enforce_retention`` — correct, tested, and with zero call sites.
* ``conditioning.py`` — reachable only from a UI readout, so the *generation*
  path converted nothing: raw shot rows reached the pipeline unconverted, the
  pixel→latent conversion never ran, and upstream silently skips a malformed
  condition.
* ``lora/apply.py`` — no importer anywhere, so stage 7, the entire point of the
  validator, never ran and a user could receive base-model output.

**None of them failed a test, and none of them could.** A dead module's unit
tests pass perfectly, because the tests are the thing calling it. The only check
that catches this class is one that asks the *program* what it reached.

Two levels, because the second is the one that actually bit us:

* **Presence** — the module is in ``sys.modules`` after ``import app``.
  Necessary, and demonstrably **not sufficient**: ``conditioning.py`` passed
  this check for the whole time it was dead, because ``ui/handlers.py`` imported
  it for a live readout. It was used everywhere except where it mattered.
* **Execution** — the module's code actually runs during a real ``generate()``.
  Recorded with :func:`sys.settrace`, which reports what *ran* rather than what
  was imported. That distinction is the entire lesson of the conditioning bug.

A lazily-imported module is legitimate, but "lazy" must never become a synonym
for "dead". So :data:`LAZY_MODULES` names, for each one, **the call site that
loads it and the condition under which that fires** — and a test asserts the
named call site really contains that import, so the allow-list cannot rot into
an excuse list.

Offline by construction: no GPU, no weights, no network. ``import app`` works
off-Space because ``spaces`` falls back to the local stub, which leaves
``ON_ZEROGPU`` false and skips the CUDA path entirely.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import PACKAGE_ROOT, REPO_ROOT

av = pytest.importorskip("av", reason="the media stack is a pinned runtime dependency")
np = pytest.importorskip("numpy")

#: Every module PLAN §7.1 lists, as an importable name.
PLAN_MODULES = (
    "ltx25_studio.config",
    "ltx25_studio.model_files",
    "ltx25_studio.errors",
    "ltx25_studio.preflight",
    "ltx25_studio.zerogpu",
    "ltx25_studio.constraints",
    "ltx25_studio.pipelines",
    "ltx25_studio.conditioning",
    "ltx25_studio.audio",
    "ltx25_studio.modes",
    "ltx25_studio.generate",
    "ltx25_studio.outputs",
    "ltx25_studio.lora.inspect",
    "ltx25_studio.lora.validate",
    "ltx25_studio.lora.apply",
    "ltx25_studio.ui.blocks",
    "ltx25_studio.ui.components",
    "ltx25_studio.ui.handlers",
)

#: Modules that legitimately load late, each with **the call site that loads it
#: and when**. Naming the site is the point: an entry here is a claim that can be
#: checked, not a permission slip. ``test_the_lazy_allow_list_names_a_real_call_site``
#: verifies the file really imports the module, and the execution tests below
#: verify the lazy load actually fires on the path that is supposed to trigger it.
LAZY_MODULES = {
    "ltx25_studio.audio": (
        "ltx25_studio/generate.py",
        "_export_sampling_rate(), on every run whose pipeline exposes a vocoder",
    ),
    "ltx25_studio.lora.apply": (
        "ltx25_studio/generate.py",
        "_apply_loras(), on every run whose request stages an adapter",
    ),
    # Both reached through the package __init__: ui/handlers.py does
    # `from ..lora import validate_lora` on an upload and generate.py does
    # `from .lora import apply`, either of which executes lora/__init__.py and
    # pulls both submodules in. Naming handlers.py here would have been a claim
    # the file could not back — this entry was wrong until the check caught it.
    "ltx25_studio.lora.validate": (
        "ltx25_studio/lora/__init__.py",
        "the lora package __init__, executed by any `from ..lora import ...` — "
        "the UI's upload handler, or generate.py staging an adapter",
    ),
    "ltx25_studio.lora.inspect": (
        "ltx25_studio/lora/__init__.py",
        "the same package __init__; also imported directly by validate.py, which "
        "parses the safetensors header",
    ),
}

EAGER_MODULES = tuple(m for m in PLAN_MODULES if m not in LAZY_MODULES)


# --------------------------------------------------------------------------
# level 1: presence after `import app`
# --------------------------------------------------------------------------


def _modules_loaded_by_importing_app() -> frozenset[str]:
    """What a real ``import app`` pulls in, measured in a clean interpreter.

    A subprocess rather than an in-process reload: this suite has already
    imported most of the package by the time it gets here, so an in-process
    check would pass trivially — and purging ``sys.modules`` mid-run to fix that
    would hand every other test a different copy of classes it holds references
    to. The isolation is the point; one interpreter start is the price.
    """
    script = "import app, json, sys; print(json.dumps(sorted(sys.modules)))"
    finished = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if finished.returncode != 0:
        pytest.fail(f"`import app` failed in a clean interpreter:\n{finished.stderr[-2000:]}")
    return frozenset(json.loads(finished.stdout.strip().splitlines()[-1]))


@pytest.fixture(scope="module")
def loaded_modules() -> frozenset[str]:
    return _modules_loaded_by_importing_app()


@pytest.mark.parametrize("module", EAGER_MODULES)
def test_every_plan_module_is_reached_by_importing_the_app(module, loaded_modules):
    """``zerogpu.py`` shipped fully written and never imported. This is that check."""
    assert module in loaded_modules, (
        f"{module} is declared in PLAN §7.1 but `import app` never loads it. Either the app "
        f"does not use it — in which case it is dead code wearing a docstring, which is how "
        f"zerogpu.py, conditioning.py and lora/apply.py all shipped — or it is loaded lazily "
        f"and belongs in LAZY_MODULES with the call site that loads it named."
    )


@pytest.mark.parametrize("module", sorted(LAZY_MODULES))
def test_the_lazy_allow_list_names_a_real_call_site(module):
    """The allow-list must not rot into an excuse list.

    An entry claims a specific file imports this module. If that import moves or
    disappears, the claim is false and the module may now be dead — exactly the
    state the allow-list would otherwise disguise.
    """
    relative, when = LAZY_MODULES[module]
    source = (REPO_ROOT / relative).read_text()
    tail = module.rsplit(".", 1)[-1]
    tree = ast.parse(source)

    def names_the_module(node) -> bool:
        """True for every spelling that actually loads ``tail``.

        Three forms, and missing one is how this check would quietly stop
        checking: ``from .tail import symbol`` puts the module in ``node.module``
        and the *symbols* in ``node.names``, which is the form that matters most
        here and the one an obvious matcher misses.
        """
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")
            return tail in module or any(a.name == tail for a in node.names)
        if isinstance(node, ast.Import):
            return any(tail in a.name.split(".") for a in node.names)
        return False

    imports_it = any(names_the_module(node) for node in ast.walk(tree))
    assert imports_it, (
        f"LAZY_MODULES claims {relative} loads {module} ({when}), but that file contains no "
        f"such import. Either the call site moved — update the entry — or nothing loads "
        f"{module} any more and it is dead."
    )


# --------------------------------------------------------------------------
# the instrument, and proof that it works
# --------------------------------------------------------------------------


def executed_package_files(call) -> set[Path]:
    """Package files whose code actually RAN during ``call()``.

    ``sys.settrace`` fires on every Python call, so this measures execution
    rather than import — which is the whole distinction ``conditioning.py``
    taught us. The local trace function is ``None`` so line events never fire;
    the overhead stays proportional to call count, not statement count.
    """
    seen: set[Path] = set()
    root = str(PACKAGE_ROOT)

    def tracer(frame, event, _arg):
        if event == "call":
            filename = frame.f_code.co_filename
            if filename.startswith(root):
                seen.add(Path(filename))
        return None

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        call()
    finally:
        sys.settrace(previous)
    return seen


def module_file(dotted: str) -> Path:
    relative = dotted.replace("ltx25_studio.", "").replace(".", "/")
    return PACKAGE_ROOT / f"{relative}.py"


def test_the_instrument_distinguishes_executed_from_merely_present():
    """Validate the probe before trusting it — the ZeroGPU harness lesson.

    A tracer that reported *imported* rather than *executed* would pass every
    test in this file while proving nothing, because the conditioning bug was
    precisely a module that was imported and never run. So: one helper that is
    called and one that is not, in the same already-imported module.
    """
    from ltx25_studio import conditioning, constraints

    ran = executed_package_files(lambda: constraints.is_valid_num_frames(121))

    assert module_file("ltx25_studio.constraints") in ran, (
        "the tracer failed to notice a function that definitely ran — it cannot be trusted "
        "to notice one that did not"
    )
    assert module_file("ltx25_studio.conditioning") not in ran, (
        f"the tracer reported {conditioning.__name__} as executed when nothing in it was "
        f"called. It is measuring imports, not execution, and would pass this whole file "
        f"while proving nothing."
    )


# --------------------------------------------------------------------------
# level 2: execution on the generation path
# --------------------------------------------------------------------------

SAMPLING_RATE = 48000
FRAMES = 2
SIZE = 64


class _Output:
    """Stereo, and big enough for the real ``av`` writer to encode."""

    def __init__(self):
        self.frames = np.zeros((FRAMES, SIZE, SIZE, 3), np.uint8)
        samples = SAMPLING_RATE // 20
        tone = np.linspace(0.0, 1.0, samples, dtype=np.float32)
        self.audio = np.stack([tone, tone * 0.5])


class _Pipe:
    """Duck-types just enough: a vocoder to read the rate off, and a call."""

    vocoder = type("V", (), {"config": type("C", (), {"output_sampling_rate": SAMPLING_RATE})})()

    def __call__(self, **kwargs):
        return _Output()


def run_generation(tmp_path, **overrides):
    from ltx25_studio import generate as generate_module

    return generate_module.generate(
        mode=overrides.pop("mode", "text_to_video"),
        seed=7,
        width=SIZE,
        height=SIZE,
        pipeline_factory=lambda **_: _Pipe(),
        encoder=None,
        output_root=tmp_path,
        retention_bytes=10**9,
        **overrides,
    )


@pytest.mark.parametrize(
    "module",
    ["ltx25_studio.modes", "ltx25_studio.constraints", "ltx25_studio.outputs"],
)
def test_the_module_runs_during_a_real_generation(module, tmp_path):
    """These three are on the path of every single run, with no conditions on it."""
    ran = executed_package_files(lambda: run_generation(tmp_path))
    assert module_file(module) in ran, (
        f"{module} never executed during a complete generate() call. It is imported, so a "
        f"presence check would call it healthy — which is exactly how conditioning.py stayed "
        f"dead while being in sys.modules the whole time."
    )


def test_conditioning_runs_when_the_request_carries_shots(tmp_path):
    """**The regression that bit us**, as an assertion.

    ``conditioning.py`` was imported by the UI's live readout and by nothing on
    the generation path, so raw shot rows reached the pipeline unconverted: the
    pixel→latent index conversion never ran, and an out-of-range anchor was
    never caught. Upstream skips such a condition with a log line and generates
    anyway, so the shot silently did nothing and the metered run was spent.
    """
    shots = [["a wide establishing shot", 8, 0.8]]
    ran = executed_package_files(
        lambda: run_generation(tmp_path, mode="image_to_video", conditions=shots)
    )
    assert module_file("ltx25_studio.conditioning") in ran, (
        "a generation carrying shot rows never executed conditioning.py, so the pixel-to-latent "
        "index conversion did not run and the rows reached the pipeline raw. This is the exact "
        "shape of the original bug."
    )


def test_audio_is_reached_for_the_sampling_rate_it_owns(tmp_path):
    """``audio.py`` is allow-listed as lazy; this proves the lazy load fires.

    Without it, "lazy" and "dead" are indistinguishable from outside — which is
    the ambiguity the allow-list exists to remove rather than create.
    """
    ran = executed_package_files(lambda: run_generation(tmp_path))
    assert module_file("ltx25_studio.audio") in ran, (
        "audio.py is on the LAZY_MODULES list claiming generate._export_sampling_rate loads "
        "it on every run with a vocoder. It did not run, so either the claim is stale or the "
        "rate is being sourced from somewhere it should not be."
    )


# --------------------------------------------------------------------------
# level 3: discovered, not declared — "reached by nothing" for the whole package
# --------------------------------------------------------------------------
#
# Levels 1 and 2 check modules somebody thought to list. This level globs the
# package and asserts nothing is reached by *neither* startup nor a generation,
# so a module added tomorrow and wired to nothing fails without anyone having to
# remember to add it here. That is the property the hand-written checks cannot
# have: ``lora/apply.py`` shipped dead carrying 19 passing specs precisely
# because nobody thought to look for it.


def _package_modules() -> set[str]:
    """Every importable module under the package, discovered from disk."""
    found = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = [p for p in path.relative_to(PACKAGE_ROOT).with_suffix("").parts if p != "__init__"]
        found.add(".".join(("ltx25_studio", *parts)))
    return found


def _synthetic_lora(directory: Path) -> Path:
    """A well-formed adapter that attaches to ``conftest``'s fake transformer.

    Built from the shared fixtures rather than a second implementation, so it
    cannot drift from what the LoRA suite considers a valid file.
    """
    from conftest import (
        NATIVE_PREFIX,
        OFFICIAL_METADATA,
        TRANSFORMER_MODULE_PATHS,
        build_safetensors_bytes,
        lora_tensors,
    )

    target = directory / "adapter.safetensors"
    target.write_bytes(
        build_safetensors_bytes(
            lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS), OFFICIAL_METADATA
        )
    )
    return target


class _LoRAPipe(_Pipe):
    """A pipe an adapter can actually attach to, so stage 7 really runs."""

    def __init__(self):
        from conftest import FakePipeline

        self.pipeline = FakePipeline()
        self.pipeline.vocoder = _Pipe.vocoder


@pytest.fixture(scope="module")
def executed_by_a_full_generation(tmp_path_factory) -> set[Path]:
    """Package files that RAN during one generation exercising every path.

    An image mode, a shot row and a staged LoRA in a single call — the three
    inputs that reach ``conditioning.py`` and ``lora/apply.py``, the two modules
    that shipped dead. Execution rather than import, because by the time this
    runs the suite has already imported most of the package and an import-based
    measure would report everything as reached.
    """
    from ltx25_studio import generate as generate_module

    directory = tmp_path_factory.mktemp("full")
    lora = _synthetic_lora(directory)

    def call():
        generate_module.generate(
            mode="image_to_video",
            seed=7,
            width=SIZE,
            height=SIZE,
            conditions=[["a wide establishing shot", 8, 0.8]],
            loras=({"source": str(lora), "weight": 1.0},),
            pipeline_factory=lambda **_: _LoRAPipe(),
            encoder=None,
            output_root=directory,
            retention_bytes=10**9,
        )

    return executed_package_files(call)


def test_lora_apply_runs_when_the_request_stages_an_adapter(executed_by_a_full_generation):
    """Blocker #5, as an assertion.

    ``lora/apply.py`` had no importer anywhere in the package while carrying 19
    passing specs. A user could upload an adapter, watch it validate, generate,
    and receive base-model output — the exact silent garbage the whole validator
    exists to prevent, arriving because the last step was never called.
    """
    assert module_file("ltx25_studio.lora.apply") in executed_by_a_full_generation, (
        "a generation staging a LoRA never executed lora/apply.py, so stage 7 — the attach "
        "assertion that is the entire point of the validator — did not run. An adapter that "
        "attaches to zero modules would produce base-model output and report success."
    )


def test_no_package_module_is_reached_by_nothing(loaded_modules, executed_by_a_full_generation):
    """The check that cannot miss a module nobody thought to list.

    Every module under ``ltx25_studio/`` must be reached by **something**: loaded
    when the app starts, or executed by a generation. A module in neither set is
    dead — complete, documented, possibly fully specced, and wired to nothing.
    That was true of ``zerogpu.py``, of ``lora/apply.py``, and of
    ``enforce_retention``, and none of them failed a test.

    Discovered by globbing rather than declared, so a module added tomorrow and
    wired to nothing fails here without anyone remembering to update a list.
    """
    reached_files = executed_by_a_full_generation
    every = _package_modules()

    def directly_reached(module: str) -> bool:
        return module in loaded_modules or module_file(module) in reached_files

    def is_reached(module: str) -> bool:
        if directly_reached(module):
            return True
        # A package ``__init__`` holds only re-exports, so its module-level code
        # runs once per process -- already done by the time this suite gets
        # here, which means the tracer cannot watch it run again. It is also
        # unreachable-by-definition-impossible: importing ``lora.apply``
        # executes ``lora/__init__.py`` first, necessarily. So a package counts
        # as reached exactly when something inside it is.
        prefix = f"{module}."
        return any(c.startswith(prefix) and directly_reached(c) for c in every)

    orphans = sorted(module for module in every if not is_reached(module))
    assert not orphans, (
        f"these modules are reached by nothing — neither `import app` nor a generation that "
        f"exercises an image, a shot row and a staged LoRA: {orphans}. Either wire them in or "
        f"delete them; a module that exists and is called by nothing is the shape that shipped "
        f"four blockers on this build, every one of them with green tests."
    )
