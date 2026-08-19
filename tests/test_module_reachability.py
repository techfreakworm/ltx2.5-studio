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
