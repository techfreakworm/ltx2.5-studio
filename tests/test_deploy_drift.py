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


def test_the_readme_declares_no_preload_block():
    """A PRIVATE Space cannot preload a gated repo.

    The preload runs as a separate platform-managed init step that does not see
    Space secrets, so the request arrives unauthenticated and the repo answers
    **401** — not 403 — however valid the configured token is. Verified: the
    same URL at the same revision fetches 200 by hand, and the public 2.3 Space
    preloads a *more* restrictive `gated=manual` repo successfully. Privacy is
    the differentiating variable.

    So the block must stay absent, and this test exists to stop someone
    reinstating it from the plan and getting a RUNTIME_ERROR that says nothing
    about privacy.
    """
    assert not frontmatter_list("preload_from_hub"), (
        "README declares preload_from_hub, which cannot work for a private Space: "
        "the preload init step runs unauthenticated and a gated repo returns 401. "
        "Weights are fetched at runtime by preflight.ensure_weights()."
    )


def test_the_commented_preload_block_has_not_drifted():
    """The disabled block must still equal what ``model_files`` generates.

    **A commented-out block drifts more readily than a live one.** Nothing
    exercises it, so no failure ever points at it — and it will be trusted on the
    day someone re-enables it, which is precisely when a stale file list is most
    expensive. That is the same drift class that produced the short_description
    rejection, aimed at a target that cannot fail loudly by itself.

    So this test is the only thing standing between the block and rot.
    """
    commented = [
        line.lstrip("#").strip()
        for line in README.read_text().splitlines()
        if line.lstrip().startswith("#") and model_files.REPO_ID in line
    ]
    entries = [line.lstrip("- ").strip() for line in commented if line.startswith("- ")]
    assert entries, (
        "the commented preload_from_hub block is gone from README.md. It is kept "
        "deliberately so the Space can flip back to preloading when it goes public; "
        "regenerate it from model_files.preload_from_hub_entries()."
    )
    expected = list(model_files.preload_from_hub_entries())
    assert entries == expected, (
        "the COMMENTED preload block has drifted from model_files.py (one entry per "
        "revision). Nothing exercises it, so this test is the only thing that would notice.\n"
        + "\n".join(f"  README   : {entry[:70]}… {entry[-8:]}" for entry in entries)
        + "\n"
        + "\n".join(f"  generated: {entry[:70]}… {entry[-8:]}" for entry in expected)
    )


def test_the_commented_block_says_why_it_is_disabled_and_when_to_re_enable():
    """A disabled mechanism with no stated re-enable condition is a trap.

    Someone will eventually find it, see no reason attached, and either turn it
    on into the same 401 or delete it as dead weight.
    """
    text = README.read_text()
    assert "401" in text, "the commented block must say WHY it is disabled"
    assert "public" in text and "persistent storage" in text, (
        "the commented block must state the conditions under which it is re-enabled"
    )


def test_the_runtime_download_still_comes_from_model_files():
    """The single-source property survives the move off `preload_from_hub`.

    The guard is repointed, not deleted: whatever fetches the weights must take
    its file list from `model_files.download_allow_list()`, so an upstream
    re-shard is still a regenerate rather than a hand-patch (PLAN §5.2, §2.1).
    """
    allow_list = model_files.download_allow_list()
    assert len(allow_list) == 37, f"the allow-list is {len(allow_list)} paths, expected 37"
    assert not any("-of-00008" in path for path in allow_list), (
        "a dead transformer shard is in the download list (PLAN §2.1, Trap A)"
    )

    source = (REPO_ROOT / "ltx25_studio" / "preflight.py").read_text()
    assert "download_allow_list" in source, (
        "preflight.ensure_weights no longer reads model_files.download_allow_list(); "
        "the download list has drifted away from its single source."
    )
    assert "download_plan" in source, (
        "the runtime download does not fetch each component at its pinned revision. "
        "Without it the download and the app resolve different cache keys and the fetch "
        "is wasted (PLAN §6.2)."
    )


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


# --------------------------------------------------------------------------
# limits the PLATFORM enforces — not just fields we agree with ourselves about
# --------------------------------------------------------------------------
#
# Everything above this line checks that the frontmatter is *internally
# consistent*: sdk_version matches the gradio pin, the preload entry matches
# model_files, the revision is carried. All true, all local, and all of it
# passed while a field was four characters too long for the platform to accept.
#
# HF rejected the push at YAML metadata verification with:
#     short_description length must be less than or equal to 60 characters long
#
# That is the same shape as every other defect on this build: the cheap half was
# thorough and the expensive half was absent. The expensive half here is
# "constraints the platform actually enforces", and the only reason we know this
# one exists is that it rejected us. Anything asserted below is a limit we have
# SEEN enforced or that HF documents — deliberately not a guess, because
# inventing a limit would fail a valid deploy for a rule nobody imposes.

#: Proven by rejection on 2026-08-19.
MAX_SHORT_DESCRIPTION = 60

#: Documented Space SDKs.
VALID_SDKS = {"gradio", "streamlit", "docker", "static"}

#: Documented card colours.
VALID_COLORS = {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}


def test_short_description_is_within_the_platform_limit():
    """HF rejects the push outright above 60 characters.

    Cheap to fail and cheap to fix — but it fails at *push* time, which on a
    build this size means a round trip through whoever holds push rights.
    """
    value = frontmatter_scalars().get("short_description", "")
    assert len(value) <= MAX_SHORT_DESCRIPTION, (
        f"short_description is {len(value)} characters; HF rejects the push above "
        f"{MAX_SHORT_DESCRIPTION}. Value: {value!r}"
    )


def test_the_sdk_is_one_the_platform_accepts():
    assert frontmatter_scalars().get("sdk") in VALID_SDKS


@pytest.mark.parametrize("field", ["colorFrom", "colorTo"])
def test_the_card_colours_are_ones_the_platform_accepts(field):
    """An unrecognised colour is rejected at the same verification step."""
    value = frontmatter_scalars().get(field)
    assert value in VALID_COLORS, f"{field}={value!r} is not one of {sorted(VALID_COLORS)}"


def test_the_preload_block_stays_within_the_ten_entry_cap():
    """PLAN §6.2: the platform caps preload_from_hub at 10 entries.

    We use exactly one, so this has enormous headroom — it exists to fail if
    anyone ever splits the block per component, which is the shape that would
    quietly exceed it.
    """
    entries = frontmatter_list("preload_from_hub")
    assert len(entries) <= 10, f"preload_from_hub has {len(entries)} entries; the cap is 10"


# --------------------------------------------------------------------------
# every advertised env var must reach the path that does the work
# --------------------------------------------------------------------------
#
# `LTX25_TRANSFORMER_PRECISION` resolved cleanly, logged cleanly, and changed no
# behaviour: `pipelines.get_pipeline` never read the config. Worse, the startup
# report then stated a footprint of 53.20 GB for a load that was still 72.19 GB —
# so it did not fail silently, it SPOKE, and it would have misdirected the next
# decision. Two of these were about to be reached for under disk pressure.
#
# This is "is it reached by the path that does the work" (test_module_reachability)
# applied to a config value rather than a module: setting it must observably change
# something. Discovered by iterating load_config's own fields, so a lever added
# tomorrow and wired to nothing fails here without anyone updating a list.


def _config_fields():
    from dataclasses import fields

    from ltx25_studio.config import RuntimeConfig

    return {f.name for f in fields(RuntimeConfig)}


def test_every_config_field_is_consumed_somewhere_outside_config():
    """A resolved value nothing reads is a lever attached to nothing."""

    package = REPO_ROOT / "ltx25_studio"
    sources = {
        path: path.read_text()
        for path in list(package.rglob("*.py")) + [REPO_ROOT / "app.py"]
        if path.name != "config.py"
    }
    # A field consumed through a derived property is consumed: `lora_max_gb` is
    # read by nothing directly, but `lora_max_bytes` is `int(lora_max_gb * 1e9)`
    # and callers use that. Resolving properties keeps the guard accurate rather
    # than merely strict -- a guard that fails on correct code gets disabled by
    # whoever it blocks, which is the same defect pointed the other way.
    import inspect

    from ltx25_studio.config import RuntimeConfig

    aliases: dict[str, set[str]] = {}
    for name, attr in vars(RuntimeConfig).items():
        if isinstance(attr, property):
            body = inspect.getsource(attr.fget)
            for field in _config_fields():
                if f"self.{field}" in body:
                    aliases.setdefault(field, set()).add(name)

    unread = []
    for field in sorted(_config_fields()):
        names = {field} | aliases.get(field, set())
        if not any(any(n in text for n in names) for text in sources.values()):
            unread.append(field)
    assert not unread, (
        f"these RuntimeConfig fields are read by nothing outside config.py: {unread}. "
        f"Each is an advertised environment variable that resolves cleanly and changes "
        f"no behaviour — the shape that made LTX25_TRANSFORMER_PRECISION worse than "
        f"having no lever at all, because it reported a footprint nobody was loading."
    )


def test_the_startup_report_cache_count_cannot_default_to_complete():
    """The line that distinguishes preload-worked from preload-did-nothing.

    Its default must be "go and look", not "nothing is missing" — a count that
    can only ever say 37/37 is not a check, and this is the single line we will
    be reading from the Space when a build goes wrong.
    """
    import inspect

    from ltx25_studio import pipelines

    signature = inspect.signature(pipelines.startup_report)
    default = signature.parameters["missing_files"].default
    assert default is None, (
        f"startup_report(missing_files=) defaults to {default!r}. With an empty default "
        f"every caller that omits it gets a confident 37/37 with nothing checked."
    )


# --------------------------------------------------------------------------
# package data: the LoRA shape manifest must ship wherever the package does
# --------------------------------------------------------------------------


def test_the_lora_shape_manifest_is_found_from_the_checkout():
    """The Space runs from the repo checkout; this is the path it reads."""
    from ltx25_studio.lora import manifest

    assert manifest.MANIFEST_PATH.is_file(), manifest.MANIFEST_PATH


def test_the_lora_shape_manifest_ships_in_the_wheel(tmp_path):
    """An installed wheel holds only what ``package-data`` names: without it the
    manifest (a ``.json`` beside the code) is silently left out and stage 6.5
    loses its reference shapes. Built offline from a copy of the tree, with the
    environment's own setuptools (no build isolation, no network)."""
    pytest.importorskip("setuptools")
    import shutil
    import subprocess
    import sys
    import zipfile

    source = tmp_path / "src"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy(REPO_ROOT / name, source / name)
    shutil.copytree(
        REPO_ROOT / "ltx25_studio",
        source / "ltx25_studio",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    built = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from setuptools import build_meta; print(build_meta.build_wheel(sys.argv[1]))",
            str(wheels),
        ],
        cwd=str(source),
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert built.returncode == 0, built.stderr[-2000:]
    wheel = next(wheels.glob("*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    shipped = sorted(p.name for p in (REPO_ROOT / "ltx25_studio" / "lora").glob("*.json"))

    assert shipped, "the LoRA package has no JSON to ship"
    for name in shipped:
        assert f"ltx25_studio/lora/{name}" in names, f"{name} is missing from {wheel.name}"
