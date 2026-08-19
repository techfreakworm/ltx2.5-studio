"""Stronger versions of two deploy guards that a mutation walked straight past.

PLAN §5.1, §5.2. **Neither existing test is wrong** — each is too coarse to catch
the shape it is aimed at, and the fix is a stronger test beside it rather than an
edit to it.

**1. "Bounded above and below" was a substring check.**
``test_deploy_drift.py::test_every_requirement_is_bounded_above_and_below``
asserts ``">=" in line and "<" in line``. PEP 508 puts environment markers on the
same line, so::

    accelerate>=1.14.0 ; python_version<"3.14"

passes green with a floor and **no ceiling at all** — the ``<`` belongs to the
marker, not to a version specifier. That is precisely the unbounded requirement
§5.1 exists to forbid, wearing a character the check was looking for.
Superseded here by parsing the requirement instead of scanning it.

**2. Criterion 3 has an f-string bypass.**
``test_model_files.py::_is_a_hardcoded_filename`` early-returns ``False`` for the
bare string ``".safetensors"`` — correctly, because ``file_types=[".safetensors"]``
is an upload filter describing what a *user* may upload, and no Lightricks rename
touches it. But an f-string is stored as its *fragments*, so::

    _STEM = "diffusion_pytorch_model-00001-of-00004"
    return f"{_STEM}.safetensors"

presents the checker with ``"diffusion_pytorch_model-00001-of-00004"`` (does not
end with the extension) and ``".safetensors"`` (exempt). Both fragments pass, and
a hardcoded shard filename is live outside ``model_files.py``.

Commit ``b5f94e8`` removed the prior rule that covered f-string fragments, so
this is a coverage regression rather than an original gap. It defeats the
operator's third named criterion — the one that exists because Lightricks
deleted a checkpoint from the Hub without warning.

Closed from both ends, because either alone is bypassable:

* the **fragment** rule catches a shard name however it is later assembled;
* the **template** rule reconstructs f-strings and concatenations and asks
  whether the *result* is a filename.

All file parsing. Nothing here downloads or imports anything.
"""

from __future__ import annotations

import ast
import re

import pytest
from conftest import REPO_ROOT, package_python_files

REQUIREMENTS = REPO_ROOT / "requirements.txt"
SOURCE_FILES = [*package_python_files(exclude=("model_files.py",)), REPO_ROOT / "app.py"]


# ==========================================================================
# 1. requirement bounds, parsed rather than scanned
# ==========================================================================

#: A PEP 508 direct reference pinned to a 40-hex commit is *more* pinned than a
#: range, so it satisfies §5.1 with no version specifier at all.
DIRECT_REFERENCE = re.compile(r"@\s*git\+\S+@[0-9a-f]{40}\b")

_SPECIFIER = re.compile(r"(===|==|~=|!=|>=|<=|>|<)\s*([^,;\s\]]+)")

#: ``~=1.2`` means ``>=1.2, ==1.*`` — a floor and a ceiling in one operator.
#: ``==`` and ``===`` pin exactly, which bounds both ways. ``!=`` excludes a
#: version and bounds neither.
LOWER_BOUND_OPERATORS = {">=", ">", "==", "===", "~="}
UPPER_BOUND_OPERATORS = {"<=", "<", "==", "===", "~="}


def requirement_lines():
    return [
        line.split("#")[0].strip()
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def version_specifier_operators(line: str) -> set[str]:
    """The operators in the *version specifier* only.

    Everything from the first ``;`` is a PEP 508 environment marker and is not a
    version constraint on this distribution — ``python_version<"3.14"`` says
    nothing whatsoever about which ``accelerate`` pip may install. Extras in
    ``[...]`` are dropped for the same reason.
    """
    requirement = line.split(";")[0]
    requirement = re.sub(r"\[[^\]]*\]", "", requirement)
    return {match.group(1) for match in _SPECIFIER.finditer(requirement)}


@pytest.mark.parametrize("line", requirement_lines(), ids=lambda line: line.split()[0])
def test_every_requirement_really_is_bounded_above(line):
    """Supersedes ``test_every_requirement_is_bounded_above_and_below``.

    That test scans the raw line for the characters ``>=`` and ``<``. This one
    parses the version specifier and ignores the environment marker, which is
    where the ``<`` that satisfied it was hiding.
    """
    if DIRECT_REFERENCE.search(line):
        return
    operators = version_specifier_operators(line)
    assert operators & UPPER_BOUND_OPERATORS, (
        f"{line!r} has no upper bound in its version specifier (operators found: "
        f"{sorted(operators) or 'none'}). Any `<` elsewhere on the line belongs to a PEP 508 "
        f"environment marker and does not constrain the version pip may resolve. The "
        f"predecessor shipped ~50 unbounded requirements and got burned by a silent "
        f"torch/torchaudio ABI break (PLAN §5.1)."
    )


@pytest.mark.parametrize("line", requirement_lines(), ids=lambda line: line.split()[0])
def test_every_requirement_really_is_bounded_below(line):
    """The other half, parsed the same way."""
    if DIRECT_REFERENCE.search(line):
        return
    operators = version_specifier_operators(line)
    assert operators & LOWER_BOUND_OPERATORS, (
        f"{line!r} has no lower bound in its version specifier (operators found: "
        f"{sorted(operators) or 'none'}). A ceiling with no floor resolves to something "
        f"ancient the moment an old version becomes the only one pip can satisfy."
    )


def test_an_environment_marker_is_not_mistaken_for_a_ceiling():
    """The bypass itself, pinned so the parser cannot regress into a substring scan.

    This is the exact line the mutation used. If ``version_specifier_operators``
    ever starts reading the marker again, this fails before any real requirement
    has to.
    """
    bypass = 'accelerate>=1.14.0 ; python_version<"3.14"'
    operators = version_specifier_operators(bypass)

    assert operators == {">="}, (
        f"the marker's `<` leaked into the version specifier: {sorted(operators)}"
    )
    assert not operators & UPPER_BOUND_OPERATORS, (
        "a requirement whose only `<` is inside an environment marker must read as unbounded "
        "above; this is the exact line that passed the old substring check."
    )


@pytest.mark.parametrize(
    ("line", "lower", "upper"),
    [
        ("torch==2.11.0", True, True),
        ("gradio~=5.49.0", True, True),
        ("numpy>=2.1.0,<3", True, True),
        ("huggingface-hub>=0.36.0", True, False),
        ("pillow<12", False, True),
        ('accelerate>=1.14.0 ; python_version<"3.14"', True, False),
        ("something!=1.2.3", False, False),
    ],
)
def test_the_bound_parser_reads_each_operator_correctly(line, lower, upper):
    """Anchors the parser on hand-checked cases, before real pins are judged by it.

    ``~=`` is the one worth stating: ``~=1.2`` means ``>=1.2, ==1.*``, so it is
    both bounds in one operator, and a parser that treats it as neither would
    fail a perfectly good pin.
    """
    operators = version_specifier_operators(line)
    assert bool(operators & LOWER_BOUND_OPERATORS) is lower
    assert bool(operators & UPPER_BOUND_OPERATORS) is upper


# ==========================================================================
# 2. the f-string bypass in criterion 3
# ==========================================================================

#: Extensions that name a weight file. A rename upstream moves all of them.
MODEL_EXTENSIONS = (".safetensors", ".bin", ".ckpt", ".pth", ".onnx")

#: Sharded-checkpoint naming, which is a Hub coordinate wherever it appears —
#: whether or not the extension is attached to the same literal.
SHARD_NAME = re.compile(
    r"(diffusion_pytorch_model|pytorch_model|model)[-_]\d{4,5}[-_]of[-_]\d{4,5}"
)


def _string_constants(tree):
    """Every string literal, including the fragments inside an f-string."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _template_of(node) -> str | None:
    """Reconstruct an f-string or ``+`` chain, with ``{}`` for each interpolation.

    ``f"{_STEM}.safetensors"`` becomes ``"{}.safetensors"`` — which is filename
    shaped, whatever ``_STEM`` turns out to hold. That is the point: the checker
    should not have to resolve the variable to know a filename is being built.
    """
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("{}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _template_of(node.left), _template_of(node.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
        return "{}"
    return None


def _is_filename_shaped(template: str) -> bool:
    """A built string that names a file rather than describing one.

    The bare extension is still exempt — ``file_types=[".safetensors"]`` is an
    upload filter, and no upstream rename touches it. What is not exempt is a
    template that puts *something* in front of the extension, because that
    something is a coordinate.
    """
    stripped = template.strip()
    if stripped in MODEL_EXTENSIONS:
        return False
    if any(character.isspace() for character in stripped):
        return False
    return stripped.endswith(MODEL_EXTENSIONS)


@pytest.mark.parametrize(
    "path", SOURCE_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_shard_filename_fragment_lives_outside_model_files(path):
    """Catches the stem however it is later assembled.

    ``_STEM = "diffusion_pytorch_model-00001-of-00004"`` is a Hub coordinate on
    its own. Whether the extension is glued on by an f-string, by ``+``, by
    ``%`` or by ``str.join`` is irrelevant — the coordinate is already in the
    file, and an upstream rename must be a one-line change in one file
    (PLAN §5.2).
    """
    offenders = [
        (lineno, value)
        for lineno, value in _string_constants(ast.parse(path.read_text()))
        if SHARD_NAME.search(value)
    ]
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} contains sharded-checkpoint names {offenders}. "
        f"Those are Hub coordinates and belong in model_files.py, even when the extension "
        f"is attached separately (PLAN §5.2)."
    )


@pytest.mark.parametrize(
    "path", SOURCE_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_filename_is_assembled_from_fragments_outside_model_files(path):
    """Supersedes the literal-only half of ``test_no_hardcoded_hub_strings_outside_model_files``.

    An f-string is stored as its fragments, so a checker that only inspects whole
    string constants sees ``".safetensors"`` — which is deliberately exempt — and
    never sees the filename that was actually built.
    """
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.JoinedStr, ast.BinOp)):
            continue
        template = _template_of(node)
        if template and _is_filename_shaped(template):
            offenders.append((node.lineno, template))

    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} builds a weight filename from fragments: {offenders}. "
        f"The pieces each look innocent to a literal-by-literal check; the assembled string is "
        f"a Hub coordinate and belongs in model_files.py (PLAN §5.2)."
    )


# --------------------------------------------------------------------------
# the detector itself, anchored on the exact bypass and on what must stay legal
# --------------------------------------------------------------------------

_BYPASS = '''
_STEM = "diffusion_pytorch_model-00001-of-00004"


def shard():
    return f"{_STEM}.safetensors"
'''

_CONCATENATION_BYPASS = '''
def shard(stem):
    return stem + ".safetensors"
'''

_LEGITIMATE = '''
UPLOAD_FILTER = [".safetensors"]
HELP = "Upload a .safetensors adapter, or paste a Hub repo id."


def describe(name):
    return f"validating {name} as a LoRA adapter"
'''


def test_the_detector_catches_the_f_string_bypass():
    """The exact injection that passes today, asserted directly.

    Without this, the sweep above could be weakened back to a literal check and
    nothing would notice until someone injected the bypass again.
    """
    tree = ast.parse(_BYPASS)

    templates = [
        _template_of(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.JoinedStr, ast.BinOp))
    ]
    assert any(template and _is_filename_shaped(template) for template in templates), (
        'f"{_STEM}.safetensors" was not recognised as a built filename'
    )

    fragments = [value for _, value in _string_constants(tree) if SHARD_NAME.search(value)]
    assert fragments, "the shard stem was not recognised as a Hub coordinate on its own"


def test_the_detector_catches_plain_concatenation_too():
    """``stem + ".safetensors"`` is the same bypass without the f-string."""
    templates = [
        _template_of(node)
        for node in ast.walk(ast.parse(_CONCATENATION_BYPASS))
        if isinstance(node, ast.BinOp)
    ]
    assert any(template and _is_filename_shaped(template) for template in templates)


def test_the_detector_leaves_the_legitimate_cases_alone():
    """The exemptions the narrowing commit was right to introduce.

    A checker that flags ``file_types=[".safetensors"]`` or a sentence mentioning
    the extension gets switched off by whoever it annoys, and then catches
    nothing at all. Both must stay green.
    """
    tree = ast.parse(_LEGITIMATE)

    for _, value in _string_constants(tree):
        assert not SHARD_NAME.search(value), f"{value!r} is prose, not a shard name"

    for node in ast.walk(tree):
        if isinstance(node, (ast.JoinedStr, ast.BinOp)):
            template = _template_of(node)
            assert not (template and _is_filename_shaped(template)), (
                f"{template!r} is prose and must not be flagged"
            )


def test_the_bare_extension_stays_exempt():
    """``.safetensors`` alone describes what a user may upload, not where a file is."""
    assert not _is_filename_shaped(".safetensors")
    assert _is_filename_shaped("{}.safetensors")
    assert _is_filename_shaped("transformer/diffusion_pytorch_model-00001-of-00004.safetensors")
    assert not _is_filename_shaped("upload a .safetensors adapter")
