"""Spec for the shot-list seam: UI dataframe -> conditioning.py -> pipeline.

PLAN §3.1a, §5.7 rows 1 and 2, §7.2.

**The same shape as the exports blocker, a third time.** ``conditioning.py`` is
complete, documented and thoroughly tested in isolation. The shot-list UI is
complete and thoroughly tested in isolation. And **nothing connects them**: the
Gradio dataframe's rows travel from ``GenerationRequest.conditions`` through
``modes._conditions`` — which does ``specs.extend(request.conditions)`` — and
arrive at the pipeline as *raw list rows*, next to the properly built
``ConditionSpec`` for the uploaded image::

    conditions reaching the pipeline:
        (ConditionSpec(source='img.png', kind='image', index=0, strength=1.0),
         ['a second shot', 3])

``conditioning.build_conditions`` and ``conditioning.normalise_condition`` are
called by nothing in the package. Both sides' tests pass. Neither side is wrong.
There is simply no test across the boundary — which is the exact failure the
exports blocker was.

What that costs, in PLAN's own words:

* §5.7 row 1, "the severe one": ``LTX2VideoCondition.index`` is a **latent**
  index. The conversion lives in ``conditioning.py`` and is never applied, so a
  user's anchor lands wherever the raw number happens to point.
* §5.7 row 2: an out-of-range condition is **silently skipped upstream** with
  nothing but a ``logger.warning``. The user supplies a shot, the shot does
  nothing, nobody is told, and the metered run is spent anyway.

**Deliberately not asserted here:** whether the anchor column is a latent index
or a pixel frame. ``fields.SHOT_COLUMNS`` labels it "Anchor (LATENT index)" and
``handlers.on_shots_change`` converts latent→pixel for the readout, while
``conditioning.normalise_condition`` treats the same field as a pixel frame and
converts pixel→latent. The two halves disagree, and picking the winner is a
design call, not a test's to make. These specs hold under either answer.
"""

from __future__ import annotations

import pytest
from conftest import code_tokens, package_python_files
from test_exports_contract import factory_returning, one_muxed_file_encoder

from ltx25_studio import conditioning, modes
from ltx25_studio import generate as generate_module
from ltx25_studio.errors import LTXStudioError
from ltx25_studio.ui.fields import SHOT_ANCHOR_COLUMN, SHOT_COLUMNS

#: Modes that accept conditioning input, so a shot list is meaningful for them.
CONDITIONABLE_MODES = tuple(
    mode_id
    for mode_id, spec in modes.MODES.items()
    if spec.pipeline == "condition" and {"image", "video"} & set(spec.inputs)
)

NUM_FRAMES = 121


class CapturingPipe:
    """Records the kwargs it was called with, so the condition list can be read."""

    def __init__(self):
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return type("Output", (), {"frames": ("frame",), "audio": ("sample",)})()


def shot_row(anchor, prompt="a second shot", strength=1.0):
    """One row exactly as ``gr.Dataframe(type="array")`` hands it to the handler."""
    row = [None] * len(SHOT_COLUMNS)
    row[0] = prompt
    row[SHOT_ANCHOR_COLUMN] = anchor
    row[2] = strength
    return row


def run_with_shots(mode_id, rows, tmp_path, *, num_frames=NUM_FRAMES):
    pipe = CapturingPipe()
    generate_module.generate(
        mode=mode_id,
        seed=7,
        image="conditioning-input.image",
        num_frames=num_frames,
        conditions=rows,
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    return pipe.kwargs.get("conditions", ())


# --------------------------------------------------------------------------
# a raw dataframe row is not a condition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_no_raw_dataframe_row_reaches_the_pipeline(mode_id, tmp_path):
    """The observable form of "conditioning.py is not wired in".

    A bare ``list`` from ``gr.Dataframe`` is not a condition under any
    convention. It has no ``source``, no ``index`` and no ``strength`` — it is
    three cells in display order, and the pipeline cannot read it as anything.
    """
    conditions = run_with_shots(mode_id, [shot_row(3)], tmp_path)

    raw = [c for c in conditions if isinstance(c, (list, tuple))]
    assert not raw, (
        f"mode {mode_id!r} handed the pipeline {raw!r} — raw UI dataframe rows, unconverted. "
        f"conditioning.normalise_condition exists to turn these into conditions and is called "
        f"by nothing in the package (PLAN §5.7 rows 1 and 2)."
    )


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_every_condition_reaching_the_pipeline_carries_a_source_and_an_index(
    mode_id, tmp_path
):
    """Whatever the convention, a built condition has these three fields."""
    conditions = run_with_shots(mode_id, [shot_row(3)], tmp_path)

    for condition in conditions:
        for attribute in ("source", "index", "strength"):
            value = (
                condition.get(attribute)
                if isinstance(condition, dict)
                else getattr(condition, attribute, None)
            )
            assert value is not None, (
                f"mode {mode_id!r} handed the pipeline a condition with no {attribute!r}: "
                f"{condition!r}"
            )


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_the_users_shot_actually_arrives_as_a_condition(mode_id, tmp_path):
    """A shot the user typed must not be dropped between the form and the model.

    The uploaded image already becomes one condition. A row in the shot list must
    become another — a list that silently comes back the same length as if the
    row were never there is the quietest possible failure.
    """
    without = run_with_shots(mode_id, [], tmp_path)
    with_shot = run_with_shots(mode_id, [shot_row(3)], tmp_path)

    assert len(with_shot) == len(without) + 1, (
        f"mode {mode_id!r}: adding a shot row changed the condition count from "
        f"{len(without)} to {len(with_shot)}; it must add exactly one."
    )


# --------------------------------------------------------------------------
# an anchor that cannot work must say so, not be skipped upstream
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_an_out_of_range_anchor_is_refused_rather_than_silently_skipped(mode_id, tmp_path):
    """PLAN §5.7 row 2, and ``conditioning.validate_condition_index`` exists for it.

    Upstream skips an out-of-range condition with only a log line and generates
    anyway: the shot has no effect, nobody is told, and the metered run is spent.
    9999 is out of range for a 121-frame clip under *either* reading of the
    anchor column, so this holds whichever convention wins.
    """
    with pytest.raises(LTXStudioError):
        run_with_shots(mode_id, [shot_row(9999)], tmp_path, num_frames=NUM_FRAMES)


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_a_negative_anchor_is_refused(mode_id, tmp_path):
    """The other end of the same range check."""
    with pytest.raises(LTXStudioError):
        run_with_shots(mode_id, [shot_row(-5)], tmp_path, num_frames=NUM_FRAMES)


# --------------------------------------------------------------------------
# the module that owns the severe conversion has to be on the path at all
# --------------------------------------------------------------------------


#: The modules a generation actually runs through. ``ui/`` is excluded on
#: purpose: ``handlers.on_shots_change`` already imports ``conditioning`` for the
#: live readout, and that import is exactly what makes this defect invisible —
#: the module *is* used, just nowhere on the path that generates anything.
GENERATION_PATH = ("generate.py", "modes.py", "pipelines.py", "outputs.py")


def imported_names(path):
    """Every module name imported by a file, from its AST. Prose cannot fake it."""
    import ast

    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[-1])
            names.update(alias.name for alias in node.names)
    return names


def test_the_conditioning_module_is_reachable_from_the_generation_path():
    """``conditioning.py`` is imported by no module that runs a generation.

    Stated structurally because each behavioural spec above can be closed by a
    local patch, while the real problem is that the module holding the §5.7
    conversion is not wired into anything. A module nothing calls is not a
    safeguard; it is documentation with a test suite.

    Read from the AST, not the text: ``modes.py``'s own docstring says
    "``conditioning.py`` owns the actual ``LTX2VideoCondition`` construction",
    which is true, aspirational, and invisible to the interpreter.
    """
    callers = [
        path.name
        for path in package_python_files()
        if path.name in GENERATION_PATH and "conditioning" in imported_names(path)
    ]

    assert callers, (
        f"no module on the generation path ({', '.join(GENERATION_PATH)}) imports "
        f"conditioning.py, so build_conditions / normalise_condition / "
        f"validate_condition_index are never called by a generation. The pixel<->latent "
        f"conversion PLAN §5.7 calls 'the severe one' is implemented, tested, and unreachable."
    )


def test_the_conditioning_helpers_are_called_by_something():
    """Names the two functions specifically, so the fix is unambiguous.

    ``code_tokens`` drops comments and string literals, so a docstring promising
    the call cannot satisfy it.
    """
    required = ("normalise_condition", "build_conditions")
    found = {name: [] for name in required}
    for path in package_python_files(exclude=("conditioning.py",)):
        tokens = set(code_tokens(path))
        for name in required:
            if name in tokens:
                found[name].append(path.name)

    unused = sorted(name for name, callers in found.items() if not callers)
    assert not unused, (
        f"{', '.join(unused)} in conditioning.py is called by nothing else in the package. "
        f"The UI collects shot anchors and the pipeline receives them unconverted."
    )


def test_the_conversion_is_not_reimplemented_outside_conditioning():
    """One implementation of the latent<->pixel arithmetic, or the two drift.

    Wiring the seam by copying ``(index - 1) * ratio + 1`` into ``generate.py``
    would close every behavioural spec above and reopen the bug the module's own
    docstring warns about: a readout that disagrees with the generator is worse
    than no readout at all.

    Detected as arithmetic on the AST rather than as text, because ``errors.py``
    quotes the very same formula in a docstring — correctly, and it is not a
    second implementation.
    """
    import ast

    offenders = []
    for path in package_python_files(exclude=("conditioning.py", "constraints.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.BinOp):
                continue
            operands = [node.left, node.right]
            if any(
                isinstance(operand, ast.Name) and operand.id == "temporal_compression_ratio"
                for operand in operands
            ):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        f"{', '.join(offenders)} does latent<->pixel arithmetic outside conditioning.py. "
        f"That conversion belongs there, once — call pixel_index_for / latent_index_for."
    )


def test_the_conditioning_module_still_exposes_what_the_seam_needs():
    """Guards the names the specs above are written against."""
    for name in ("normalise_condition", "build_conditions", "validate_condition_index"):
        assert hasattr(conditioning, name), f"conditioning.{name} has gone missing"
