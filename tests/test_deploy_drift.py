"""Spec for deploy-config drift — PLAN §5.1, §5.2, §6, §2.1.

Three files have to agree or the Space misbehaves in ways that cost tens of
gigabytes to discover:

* ``README.md`` frontmatter — what the Space platform actually installs and
  preloads;
* ``requirements.txt`` — what pip is allowed to resolve;
* ``ltx25_studio/model_files.py`` — the single source of truth for weights.

Each pair below has a specific, expensive failure mode when it drifts, and none
of them fails fast on its own. That is what makes them worth a test rather than
a convention.

Nothing here downloads anything; it is all file parsing.
"""

from __future__ import annotations

import re

import pytest
from conftest import REPO_ROOT

from ltx25_studio import model_files

README = REPO_ROOT / "README.md"
REQUIREMENTS = REPO_ROOT / "requirements.txt"


# --------------------------------------------------------------------------
# parsers (no pyyaml — it is not a test dependency)
# --------------------------------------------------------------------------


def frontmatter_lines():
    text = README.read_text()
    if not text.startswith("---"):
        pytest.fail("README.md has no YAML frontmatter; the Space needs it (PLAN §6)")
    _, _, rest = text.partition("---\n")
    body, _, _ = rest.partition("\n---")
    return body.splitlines()


def frontmatter_scalars():
    values = {}
    for line in frontmatter_lines():
        if not line.strip() or line.lstrip().startswith("#") or line.startswith((" ", "\t", "-")):
            continue
        key, sep, value = line.partition(":")
        if sep:
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def frontmatter_list(key):
    """The ``- `` items under a top-level key, comments skipped."""
    items = []
    collecting = False
    for line in frontmatter_lines():
        if line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t", "-")) and line.strip():
            collecting = line.split(":")[0].strip() == key
            continue
        if collecting and line.strip().startswith("- "):
            items.append(line.strip()[2:].strip().strip('"').strip("'"))
    return items


def requirement_lines():
    return [
        line.split("#")[0].strip()
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def requirement_names():
    names = {}
    for line in requirement_lines():
        name = re.split(r"[=<>!~ @\[]", line, maxsplit=1)[0].strip()
        if name:
            names[name.lower().replace("_", "-")] = line
    return names


# --------------------------------------------------------------------------
# gradio: the frontmatter and the pin must be the SAME version
# --------------------------------------------------------------------------


def test_the_gradio_requirement_is_pinned_exactly():
    """PLAN §5.1, sharpened: a range is not a pin here.

    ``gradio>=6.24.0,<7`` brackets 6.24.0 today and resolves to 6.30 tomorrow
    while ``sdk_version`` still says 6.24.0. The Space then serves one gradio and
    the requirements permit another — the drift class this whole file exists to
    eliminate.
    """
    line = requirement_names().get("gradio")
    assert line, "requirements.txt does not mention gradio"
    assert "==" in line and "," not in line, (
        f"gradio must be pinned exactly to match README sdk_version; got {line!r}"
    )


def test_the_gradio_pin_equals_the_readme_sdk_version():
    sdk_version = frontmatter_scalars().get("sdk_version")
    line = requirement_names().get("gradio", "")
    pinned = line.partition("==")[2].strip()
    assert sdk_version, "README frontmatter has no sdk_version"
    assert pinned == sdk_version, (
        f"requirements.txt pins gradio {pinned!r} but README sdk_version is {sdk_version!r}. "
        "The Space installs the frontmatter version; they must be the same."
    )


# --------------------------------------------------------------------------
# preload_from_hub: generated from model_files, revision included
# --------------------------------------------------------------------------


def test_the_readme_preload_entry_matches_model_files():
    """PLAN §2.1 Trap A: a hand-written list that names the dead shard set does
    not fail fast — it fails after tens of gigabytes of download."""
    entries = frontmatter_list("preload_from_hub")
    assert entries, (
        "README frontmatter has no preload_from_hub entries. Generate the block from "
        "model_files.preload_from_hub_entry() (PLAN §5.2, §6.2)."
    )
    expected = model_files.preload_from_hub_entry()
    assert entries == [expected], (
        f"README preload_from_hub has drifted from model_files.py.\n"
        f"  README:      {entries}\n"
        f"  model_files: [{expected!r}]"
    )


def test_the_preload_entry_carries_the_pinned_revision():
    """Without the SHA the preload and the app resolve **different cache keys**,
    so the Space re-downloads 72 GB at runtime having already preloaded it."""
    expected = model_files.preload_from_hub_entry()
    assert model_files.REVISION in expected, (
        f"preload entry omits revision {model_files.REVISION}: {expected!r}"
    )


def test_there_is_exactly_one_preload_entry():
    """PLAN §6.2: ``preload_from_hub`` is capped at 10 entries; one file per entry
    would not fit. One repo, comma-separated filenames."""
    entries = frontmatter_list("preload_from_hub")
    if not entries:
        pytest.skip("preload_from_hub not generated yet — covered by the drift test above")
    assert len(entries) == 1, f"expected one comma-joined entry, got {len(entries)}"


# --------------------------------------------------------------------------
# every direct dependency has a floor AND a ceiling (PLAN §5.1)
# --------------------------------------------------------------------------

#: A PEP 508 direct reference pinned to a 40-hex commit is *more* pinned than a
#: range, so it satisfies §5.1 without a version specifier.
DIRECT_REFERENCE = re.compile(r"@\s*git\+\S+@[0-9a-f]{40}\b")


@pytest.mark.parametrize("line", requirement_lines(), ids=lambda line: line.split()[0])
def test_every_requirement_is_bounded_above_and_below(line):
    """The post-mortem in §5.1: the predecessor shipped ~50 unbounded requirements
    and got burned by a silent torch/torchaudio ABI break. A floor with no
    ceiling is how that happens."""
    if DIRECT_REFERENCE.search(line):
        return
    if "==" in line:
        return
    assert ">=" in line and "<" in line, (
        f"{line!r} has no upper bound. Every direct dependency gets a floor and a ceiling "
        "(PLAN §5.1)."
    )


def test_spaces_is_deliberately_not_pinned():
    """PLAN §5.1: the platform pins its own ``spaces`` and a conflicting pin here
    breaks pip resolution at build time. Its absence is a decision, not an
    oversight — so it is worth asserting, or someone will "fix" it."""
    assert "spaces" not in requirement_names()


def test_torch_and_torchaudio_are_pinned_to_the_same_version():
    """The predecessor's one hard pin carried a post-mortem comment: a mismatched
    torchaudio crashes the whole audio path on import via a C++ ABI mismatch."""
    names = requirement_names()
    torch = names.get("torch", "").partition("==")[2].strip()
    torchaudio = names.get("torchaudio", "").partition("==")[2].strip()
    assert torch, "torch must be pinned exactly"
    assert torchaudio == torch, (
        f"torchaudio {torchaudio!r} must match torch {torch!r} exactly, or the C++ ABI "
        "mismatch crashes the audio path on import"
    )


def test_diffusers_is_pinned_to_a_commit_sha():
    """PLAN §1.3 / OQ-1: LTX-2.5 is in no release, and a SHA is more pinned than a
    range, not less."""
    line = requirement_names().get("diffusers", "")
    assert DIRECT_REFERENCE.search(line), (
        f"diffusers must be a direct reference pinned to a 40-hex SHA; got {line!r}"
    )


# --------------------------------------------------------------------------
# the rest of the Space contract (PLAN §6)
# --------------------------------------------------------------------------


def test_the_space_declares_the_gradio_sdk():
    assert frontmatter_scalars().get("sdk") == "gradio"


def test_the_space_points_at_the_real_entrypoint():
    app_file = frontmatter_scalars().get("app_file")
    assert app_file, "README frontmatter has no app_file"
    assert (REPO_ROOT / app_file).is_file(), f"app_file {app_file!r} does not exist"


def test_the_space_pins_python_312():
    """The default 3.10.13 image silently resolves numpy and av **downward**
    rather than failing — a wrong-but-working build is the worst kind."""
    assert frontmatter_scalars().get("python_version") == "3.12"


def test_the_space_requests_zerogpu_hardware():
    """``zero-a10g`` is a legacy identifier for hardware that is not an A10G.
    "Correcting" it moves the Space onto paid dedicated hardware."""
    assert frontmatter_scalars().get("suggested_hardware") == "zero-a10g"


def test_the_python_requirement_agrees_with_the_frontmatter():
    import tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    requires = pyproject["project"]["requires-python"]
    assert "3.12" in requires, (
        f"pyproject requires-python {requires!r} should agree with the Space's python_version"
    )
