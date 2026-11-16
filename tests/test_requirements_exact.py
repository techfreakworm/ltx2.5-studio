"""Spec for exact requirement pins: a Space rebuild installs what was verified.

``requirements.txt`` used to give each dependency a floor and a ceiling. A range is
resolved again on every Space rebuild, so the image gets whatever is newest that
day rather than what was tested. That is how transformers 5.23.0 arrived and broke
the prompt enhancer. Each line now names one version (the set verified on real
weights on 2026-11-16) or, for the git dependency, one commit.

The only exemption is a package the Spaces platform installs itself, and such a
line must say so with a ``# platform-provided`` comment. ``spaces`` is that kind of
package and is left out of the file entirely (``test_deploy_drift.py``).

File parsing only, like the rest of the drift suite. The parser is local rather
than imported from another spec file, as in ``test_torch_trio_drift.py``.
"""

from __future__ import annotations

import re
import tomllib

import pytest
from conftest import REPO_ROOT

REQUIREMENTS = REPO_ROOT / "requirements.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The comment that exempts a line from exact pinning.
PLATFORM_PROVIDED = "platform-provided"

#: A PEP 508 direct reference pinned to a 40-hex commit names one exact tree.
PINNED_DIRECT_REFERENCE = re.compile(r"@\s*git\+\S+@[0-9a-f]{40}\b")

_SPECIFIER = re.compile(r"(===|==|~=|!=|>=|<=|>|<)\s*([^,;\s\]]+)")


def requirement_entries() -> list[tuple[str, str]]:
    """``(requirement, comment)`` for every requirement line, comment kept for the marker."""
    entries = []
    for raw in REQUIREMENTS.read_text().splitlines():
        requirement, _, comment = raw.partition("#")
        if requirement.strip():
            entries.append((requirement.strip(), comment.strip()))
    return entries


def canonical(name: str) -> str:
    """PEP 503 normalisation: ``huggingface_hub`` and ``huggingface-hub`` are one package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def name_of(requirement: str) -> str:
    return canonical(re.split(r"[=<>!~ @\[;]", requirement, maxsplit=1)[0].strip())


def specifiers(requirement: str) -> list[tuple[str, str]]:
    """``(operator, version)`` pairs of the version specifier only.

    Everything from the first ``;`` is an environment marker, whose ``<`` says nothing
    about which version pip may install. Extras in ``[...]`` are dropped too.
    """
    version_part = re.sub(r"\[[^\]]*\]", "", requirement.split(";")[0])
    return [(m.group(1), m.group(2)) for m in _SPECIFIER.finditer(version_part)]


def pins_one_version(requirement: str) -> bool:
    """True when pip can install exactly one thing for this line."""
    if PINNED_DIRECT_REFERENCE.search(requirement):
        return True
    found = specifiers(requirement)
    return (
        len(found) == 1
        and found[0][0] in {"==", "==="}
        and "*" not in found[0][1]  # `==2.*` is a range
    )


def exact_pins() -> dict[str, str]:
    """``{package: requirement}`` for every line of ``requirements.txt``."""
    return {name_of(requirement): requirement for requirement, _ in requirement_entries()}


# --------------------------------------------------------------------------
# the parser, anchored on hand-checked lines before real pins are judged by it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requirement", "exact"),
    [
        ("torch==2.11.0", True),
        ("hf_xet===1.8.0", True),
        ('accelerate==1.15.0 ; python_version<"3.14"', True),
        ("diffusers @ git+https://github.com/huggingface/diffusers@" + "a" * 40, True),
        ("diffusers @ git+https://github.com/huggingface/diffusers@main", False),
        ("numpy>=2.5.2,<3", False),
        ("gradio~=6.24.0", False),
        ("pillow<13", False),
        ("kernels>=0.16.0", False),
        ("numpy==2.*", False),
        ("peft==0.20.0,<0.21", False),
        ("tokenizers!=0.23.0", False),
        ("transformers", False),
    ],
)
def test_the_pin_parser_reads_each_line_correctly(requirement, exact):
    """A marker's ``<`` is not a range, ``==2.*`` is one, and a bare name pins nothing."""
    assert pins_one_version(requirement) is exact


# --------------------------------------------------------------------------
# every line names one version
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requirement", "comment"),
    requirement_entries(),
    ids=[requirement.split(" ")[0] for requirement, _ in requirement_entries()],
)
def test_every_requirement_is_pinned_to_one_version(requirement, comment):
    """A range is re-resolved on every rebuild; that is how an untested
    transformers reached the Space. Only a line marked platform-provided may use
    one, because the platform, not this file, decides that version."""
    if PLATFORM_PROVIDED in comment:
        return
    assert pins_one_version(requirement), (
        f"{requirement!r} does not pin one version (specifiers: {specifiers(requirement)}). "
        f"Pin the exact version that was verified, or mark a line the Spaces platform "
        f"installs itself with `# {PLATFORM_PROVIDED}`."
    )


def test_tokenizers_is_pinned_beside_transformers():
    """tokenizers decides whether transformers imports at all: transformers 5.15.1
    raises ImportError with tokenizers 0.23.3. Left to float, it can break the
    import on a rebuild with transformers itself unchanged."""
    pins = exact_pins()
    for name in ("transformers", "tokenizers"):
        assert name in pins, f"requirements.txt does not name {name}"
        assert pins_one_version(pins[name]), pins[name]


# --------------------------------------------------------------------------
# requirements.txt and pyproject.toml agree
# --------------------------------------------------------------------------


def pyproject_requirements() -> list[str]:
    """Every requirement ``pyproject.toml`` declares, runtime and optional."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    declared = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        declared.extend(group)
    return declared


def compact(requirement: str) -> str:
    return re.sub(r"\s+", "", requirement)


def test_pyproject_pins_whatever_it_repeats_exactly_as_requirements_does():
    """A package named in both files must carry the same pin in both: two
    versions of one dependency is one more drift the deploy cannot see."""
    pins = exact_pins()
    repeated = [req for req in pyproject_requirements() if name_of(req) in pins]
    for requirement in repeated:
        assert compact(requirement) == compact(pins[name_of(requirement)]), (
            f"pyproject.toml declares {requirement!r} but requirements.txt pins "
            f"{pins[name_of(requirement)]!r}"
        )


def test_a_pyproject_runtime_list_mirrors_requirements_exactly():
    """``requirements.txt`` is the file the Space build reads. If pyproject ever
    declares runtime dependencies too, they must be the same set at the same pins."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    if "dependencies" not in project:
        return
    declared = {name_of(req): compact(req) for req in project["dependencies"]}
    expected = {name: compact(req) for name, req in exact_pins().items()}
    assert declared == expected
