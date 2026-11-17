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
    "ltx25_studio.device",
    "ltx25_studio.local_runtime",
    "ltx25_studio.memory",
    "ltx25_studio.constraints",
    "ltx25_studio.pipelines",
    "ltx25_studio.conditioning",
    "ltx25_studio.audio",
    "ltx25_studio.modes",
    "ltx25_studio.generate",
    "ltx25_studio.enhance",
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
    # Note the "unconditionally" and the "every": the toggle is off by default,
    # but the CALL is not conditional on it. `enhance_if_requested` returns
    # immediately when the flag is clear, which is what keeps this module on the
    # executed path of a default run rather than on the path of a checkbox
    # almost nobody ticks. See test_the_module_runs_during_a_real_generation.
    "ltx25_studio.enhance": (
        "ltx25_studio/generate.py",
        "enhance_if_requested(), called unconditionally on every generation, inside the "
        "GPU worker wherever there is one",
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


#: Run in a **fresh interpreter**: import the app, record what that loaded, then
#: drive one real generation under ``sys.settrace`` and record what executed.
#:
#: Both halves must happen in the same clean process. Measuring reachability from
#: inside the suite is order-dependent in a way that is easy to miss: a package
#: ``__init__.py`` executes only on first import, so once any earlier test has
#: imported ``ltx25_studio.lora`` the tracer can never watch it run again, and
#: the package looks unreachable when it is reached on every LoRA generation.
#: This file shipped exactly that false positive. A subprocess has no history to
#: be confused by.
_PROBE = r"""
import json, sys, tempfile
sys.path.insert(0, "tests")

import app                                    # what a real startup loads
after_import = sorted(sys.modules)

import numpy as np
from pathlib import Path
from conftest import (
    NATIVE_PREFIX, OFFICIAL_METADATA, TRANSFORMER_MODULE_PATHS,
    FakePipeline, audio_file, build_safetensors_bytes, image_input, inputs_for, lora_tensors,
)
from ltx25_studio import generate as generate_module

PKG = str(Path(app.__file__).parent / "ltx25_studio")
RATE, COUNT, SIZE = 48000, 2, 64


class Out:
    def __init__(self):
        self.frames = np.zeros((COUNT, SIZE, SIZE, 3), np.uint8)
        tone = np.linspace(0.0, 1.0, RATE // 20, dtype=np.float32)
        self.audio = np.stack([tone, tone * 0.5])


class Pipe:
    def __init__(self):
        self.pipeline = FakePipeline()
        vocoder = type("V", (), {"config": type("C", (), {"output_sampling_rate": RATE})})()
        self.pipeline.vocoder = vocoder
        self.vocoder = vocoder

    def __call__(self, **kwargs):
        return Out()


executed = set()
functions = set()


def tracer(frame, event, _arg):
    if event == "call" and frame.f_code.co_filename.startswith(PKG):
        executed.add(frame.f_code.co_filename)
        # co_name was always right here and was always discarded. A module can
        # be reached by one function while another in the same file is dead --
        # which is exactly how `encode_waveform_to_latents` sat at importer
        # count zero inside a module that ran on every audio export.
        functions.add(frame.f_code.co_name)
    return None


with tempfile.TemporaryDirectory() as directory:
    adapter = Path(directory) / "adapter.safetensors"
    adapter.write_bytes(
        build_safetensors_bytes(
            lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS), OFFICIAL_METADATA
        )
    )
    sys.settrace(tracer)
    try:
        generate_module.generate(
            mode="image_to_video", seed=7, width=SIZE, height=SIZE,
            **inputs_for("image_to_video"),
            conditions=[{"source": image_input(), "index": 8, "strength": 0.8}],
            loras=({"source": str(adapter), "weight": 1.0},),
            pipeline_factory=lambda **_: Pipe(), encoder=None,
            output_root=directory, retention_bytes=10**9,
        )
        # A second generation, because one mode cannot exercise eleven. The
        # audio-input path is reachable from NO other mode, so a probe that
        # drives only image_to_video is blind to it by construction.
        from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM
        from ltx25_studio import audio as audio_module

        audio_module.load_waveform = lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
        generate_module.generate(
            mode="audio_to_audio", seed=7, width=SIZE, height=SIZE,
            audio=audio_file(),
            pipeline_factory=lambda **_: Pipe(), encoder=None,
            output_root=directory, retention_bytes=10**9,
        )
    finally:
        sys.settrace(None)

print("@@PROBE@@" + json.dumps({
    "after_import": after_import,
    "executed": sorted(executed),
    "functions": sorted(functions),
}))
"""


def _probe_a_clean_interpreter() -> dict:
    """What startup loads and what a full generation executes, with no history.

    One subprocess covers both questions. Purging ``sys.modules`` in-process to
    get the same isolation would hand every other test a different copy of
    classes it holds references to — a subtle and horrible bug to debug — so the
    interpreter start is the price of not doing that.
    """
    finished = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    marker = "@@PROBE@@"
    if finished.returncode != 0 or marker not in finished.stdout:
        pytest.fail(
            "the clean-interpreter reachability probe failed:\n"
            f"stdout tail:\n{finished.stdout[-1500:]}\n\nstderr tail:\n{finished.stderr[-2500:]}"
        )
    payload = json.loads(finished.stdout.split(marker, 1)[1].splitlines()[0])
    return {
        "after_import": frozenset(payload["after_import"]),
        "executed": {Path(f) for f in payload["executed"]},
        "functions": set(payload["functions"]),
    }


@pytest.fixture(scope="module")
def clean_probe() -> dict:
    return _probe_a_clean_interpreter()


def _modules_loaded_by_importing_app(probe) -> frozenset[str]:
    return probe["after_import"]


@pytest.fixture(scope="module")
def loaded_modules(clean_probe) -> frozenset[str]:
    return _modules_loaded_by_importing_app(clean_probe)


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
    """The file a dotted name lives in — ``__init__.py`` for a package.

    Getting this wrong is not a cosmetic bug: mapping ``ltx25_studio.lora`` to a
    ``lora.py`` that does not exist makes the package permanently unmatchable by
    file, so its reachability silently falls through to a weaker check. That was
    the mechanism behind this file's own order-dependent false positive.
    """
    relative = dotted.removeprefix("ltx25_studio").strip(".").replace(".", "/")
    direct = PACKAGE_ROOT / f"{relative}.py" if relative else None
    if direct is not None and direct.exists():
        return direct
    return (PACKAGE_ROOT / relative / "__init__.py") if relative else PACKAGE_ROOT / "__init__.py"


def test_module_file_resolves_a_package_to_its_init():
    """Regression: a package lives in ``__init__.py``, not ``<name>.py``.

    This mapping was wrong, and the way it was wrong is worth remembering,
    because it did not look like a bug: ``ltx25_studio.lora`` resolved to a
    ``lora.py`` that does not exist, so the package could never match by file
    and its reachability silently fell through to a weaker check. A lookup that
    returns a plausible path for a file that is not there fails *quietly* —
    which is the whole class this file exists to catch, reproduced inside the
    instrument meant to catch it.
    """
    for dotted in ("ltx25_studio", "ltx25_studio.lora", "ltx25_studio.ui"):
        resolved = module_file(dotted)
        assert resolved.name == "__init__.py", f"{dotted} is a package; got {resolved.name}"
        assert resolved.is_file(), f"{dotted} resolved to {resolved}, which does not exist"
    for dotted in ("ltx25_studio.audio", "ltx25_studio.lora.apply"):
        resolved = module_file(dotted)
        assert resolved.is_file() and resolved.name != "__init__.py", resolved


def test_a_package_reached_only_by_a_lazy_generation_import_is_not_a_false_positive(clean_probe):
    """The order-dependent false positive this file shipped, pinned.

    ``ltx25_studio.lora`` is imported by nothing at startup and by ``generate``
    only when a request stages an adapter. Its ``__init__.py`` therefore executes
    exactly once, on that first lazy import — so an in-suite tracer that runs
    after any earlier test has already imported it can never watch it execute,
    and the package looks dead while being reached on every LoRA run.

    Measuring in a clean interpreter is what makes this answerable at all. If
    someone moves the discovery back in-process to save the subprocess, this is
    the test that should stop them.
    """
    lora = "ltx25_studio.lora"
    assert lora not in clean_probe["after_import"], (
        f"{lora} is now loaded at startup, so this test no longer covers the lazy case it "
        f"was written for — point it at whatever is lazy now, do not delete it"
    )
    reached = module_file(lora) in clean_probe["executed"] or any(
        module_file(m) in clean_probe["executed"]
        for m in _package_modules()
        if m.startswith(f"{lora}.")
    )
    assert reached, (
        f"{lora} was not reached by a generation that stages an adapter. Either stage 7 is "
        f"disconnected again, or the probe has stopped measuring first-import execution."
    )


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
    from conftest import inputs_for

    from ltx25_studio import generate as generate_module

    mode = overrides.pop("mode", "text_to_video")
    return generate_module.generate(
        mode=mode,
        seed=7,
        width=SIZE,
        height=SIZE,
        pipeline_factory=lambda **_: _Pipe(),
        encoder=None,
        output_root=tmp_path,
        retention_bytes=10**9,
        **{**inputs_for(mode), **overrides},
    )


@pytest.mark.parametrize(
    "module",
    [
        "ltx25_studio.modes",
        "ltx25_studio.constraints",
        "ltx25_studio.outputs",
        # `enhance` is on this list even though the toggle is off by default,
        # and that is the point rather than an oversight. `generate()` calls
        # `enhance_if_requested` unconditionally and it returns immediately when
        # the toggle is off, precisely so the module is not one that only a
        # non-default checkbox reaches -- the state four of this build's
        # blockers shipped in. Gating the call on the flag would be the obvious
        # tidy-up and would put the module straight back into that class, with
        # nothing failing. This is what fails.
        "ltx25_studio.enhance",
    ],
)
def test_the_module_runs_during_a_real_generation(module, tmp_path):
    """These are on the path of every single run, with no conditions on it."""
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
    from conftest import image_input

    shots = [{"source": image_input(), "index": 8, "strength": 0.8}]
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


#: Non-UI modules that legitimately never run during a generation, each with the
#: reason it does not. Everything else outside ``ui/`` must be reached by the
#: generation path itself.
#:
#: Measured, not assumed — and kept short and checkable on purpose. A rule with a
#: vague exception set is a rule somebody disables; a rule whose exceptions each
#: carry a testable reason survives contact with a future reader who thinks the
#: list looks wrong.
STARTUP_ONLY = {
    "ltx25_studio": "the package __init__; re-exports only",
    "ltx25_studio.config": (
        "resolved once at startup; generation receives the values, not the module"
    ),
    "ltx25_studio.errors": (
        "executes only when something raises, and a successful run raises nothing"
    ),
    "ltx25_studio.model_files": (
        "the weight table; read by preflight and by pipelines loading real weights"
    ),
    "ltx25_studio.pipelines": (
        "bypassed offline by the injected pipeline_factory; exercised for real on the Space"
    ),
    "ltx25_studio.preflight": "startup resource and version checks, by definition",
    "ltx25_studio.log_filter": (
        "installed by app.py's logging set-up; the forked GPU worker inherits the filter"
    ),
    "ltx25_studio.local_runtime": (
        "imported by pipelines at module scope; runs only for a local mps/cpu generation with "
        "real weights, which the offline probe cannot make"
    ),
    "ltx25_studio.memory": "the local runner's memory preflight; same reach as local_runtime",
}


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
    from conftest import image_input, inputs_for

    from ltx25_studio import generate as generate_module

    directory = tmp_path_factory.mktemp("full")
    lora = _synthetic_lora(directory)

    def call():
        generate_module.generate(
            mode="image_to_video",
            seed=7,
            width=SIZE,
            height=SIZE,
            **inputs_for("image_to_video"),
            conditions=[{"source": image_input(), "index": 8, "strength": 0.8}],
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


def test_no_package_module_is_reached_by_nothing(clean_probe):
    """The check that cannot miss a module nobody thought to list.

    Every module under ``ltx25_studio/`` must be reached by **something**: loaded
    when the app starts, or executed by a generation. A module in neither set is
    dead — complete, documented, possibly fully specced, and wired to nothing.
    That was true of ``zerogpu.py``, of ``lora/apply.py``, and of
    ``enforce_retention``, and none of them failed a test.

    Discovered by globbing rather than declared, so a module added tomorrow and
    wired to nothing fails here without anyone remembering to update a list.
    """
    # From the clean-interpreter probe, NOT the in-process trace: measured
    # inside the suite, a package __init__ that some earlier test already
    # imported can never be seen to execute again, and this assertion reported
    # a false positive on exactly that. A fresh process has no such history.
    loaded_modules = clean_probe["after_import"]
    reached_files = clean_probe["executed"]
    every = _package_modules()

    def on_generation_path(module: str) -> bool:
        return module_file(module) in reached_files

    def directly_reached(module: str) -> bool:
        """Reached in a way that counts *for this module*.

        Scoped deliberately, and the scoping is the point. ``conditioning.py``
        sat in ``sys.modules`` the entire time it was dead, because
        ``ui/handlers.py`` imported it for a live readout — so for business
        logic, "loaded" launders exactly the defect being hunted. Non-``ui/``
        modules must therefore be reached by the **generation path**; only
        ``ui/`` modules and the documented :data:`STARTUP_ONLY` set may qualify
        on being loaded alone.

        **Do not widen this back to "loaded or executed".** That version was
        written first and it was silent on a business-logic module imported only
        by ``ui/handlers.py`` — blocker #4's shape for any module without a
        hand-written targeted test. It was caught by someone treating this file
        as a black box and asking what it *should* detect; reasoning forward
        from the implementation reproduces the blind spot instead of finding it.
        """
        if on_generation_path(module):
            return True
        is_ui = module.split(".")[:2] == ["ltx25_studio", "ui"]
        if is_ui or module in STARTUP_ONLY:
            return module in loaded_modules
        return False

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


# --------------------------------------------------------------------------
# a live module is not a live function
# --------------------------------------------------------------------------

#: Functions that ARE the path they belong to, in modules that other functions
#: keep alive. A file-level sweep cannot see these: ``audio.py`` runs on every
#: audio-exporting mode via ``export_sampling_rate_from``, so the module was
#: always "reached" while ``encode_waveform_to_latents`` sat at importer count
#: zero and the user's uploaded audio was silently discarded.
CONTRACT_FUNCTIONS = (
    (
        "encode_waveform_to_latents",
        "the user's uploaded audio is never encoded, so the "
        "model generates a track they did not ask for and the run reports success",
    ),
    ("_prepared_audio", "nothing joins request.audio to audio_latents"),
    ("build_conditions", "shot rows reach the pipeline unconverted"),
    ("apply_lora", "an adapter validates and is never attached"),
)


@pytest.mark.parametrize(
    "name,consequence", CONTRACT_FUNCTIONS, ids=[f[0] for f in CONTRACT_FUNCTIONS]
)
def test_a_function_that_is_the_whole_point_of_a_path_actually_runs(name, consequence, clean_probe):
    """The escalation of the module sweep, and the reason it was needed.

    Four blockers were dead *modules*; this one was a dead **function inside a
    live module**, which the file-level check passes by construction. The tracer
    always had ``co_name`` in hand and discarded it -- the resolution was thrown
    away, not missing.
    """
    assert name in clean_probe["functions"], (
        f"{name}() never ran during a real generation, so {consequence}. "
        f"A module-level check cannot see this: its module was executed by other "
        f"functions in the same file."
    )


def test_the_probe_drives_a_mode_that_takes_audio_input():
    """One mode cannot exercise eleven.

    The audio-input path is reachable from no other mode, so a probe that drives
    only ``image_to_video`` is blind to it however fine its granularity. Both
    halves were needed; either alone would still have missed this.
    """
    assert 'mode="audio_to_audio"' in _PROBE, (
        "the reachability probe no longer drives an audio-input mode, so nothing "
        "watches the path that was connected to nothing"
    )
