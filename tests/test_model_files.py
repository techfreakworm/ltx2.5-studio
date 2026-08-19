"""Spec for ``ltx25_studio/model_files.py`` — PLAN §5.2, §2.1, §2.3, §2.4.

Why this module is worth an architecture test and not just a data test:
Lightricks deleted a 2.3 checkpoint from the Hub with no warning and broke a live
Space. The mitigation is not "be careful" — it is that **every repo id, revision,
subfolder and filename lives in exactly one file**, so the next rename is a
one-line change. ``test_no_hardcoded_hub_strings_outside_model_files`` is what
turns that intention into something that stays true.

The contract:

* ``REPO_ID``, ``REVISION`` — the pinned coordinates. A revision that is a mutable
  branch ref is not a pin (§5.2).
* ``COMPONENTS: Mapping[str, ComponentSpec]`` — data only, no I/O, no logic.
  Each spec carries ``name``, ``subfolder``, ``repo_id``, ``revision``,
  ``size_gb``, ``default`` and a resolvable ``url``.
* ``DEFAULT_COMPONENTS`` — the §2.3 shipped subset, summing to ~72.19 GB.
* ``get_component(name)`` raises ``UnknownComponentError``; ``file_url(...)`` and
  ``require_file(...)`` name **both the filename and its URL** on failure.
"""

from __future__ import annotations

import ast

import pytest
from conftest import PACKAGE_ROOT, REPO_ROOT, code_string_constants, package_python_files

from ltx25_studio import model_files
from ltx25_studio.errors import MissingModelFileError, UnknownComponentError

#: PLAN §2.3 — the shipped subset, plus ``scheduler`` (config-only, 0 GB, needed
#: by every pipeline constructor but absent from the §2.3 size table).
EXPECTED_DEFAULT_COMPONENTS = {
    "transformer",
    "text_encoder",
    "tokenizer",
    "connectors",
    "vae",
    "audio_vae",
    "vocoder",
    "latent_upsampler",
    "temporal_latent_upsampler",
    "diffusion_decoder",
    "duration_head",
    "scheduler",
}

#: PLAN §2.3 sizes in GB, verified against the repo listing.
EXPECTED_SIZES_GB = {
    "transformer": 37.975,
    "text_encoder": 23.919,
    "connectors": 6.344,
    "vae": 1.452,
    "audio_vae": 0.107,
    "vocoder": 0.258,
    "latent_upsampler": 0.996,
    "temporal_latent_upsampler": 0.262,
    "diffusion_decoder": 0.834,
    "duration_head": 0.004,
    "tokenizer": 0.032,
}

MUTABLE_REFS = {"main", "master", "HEAD", "refs/heads/main", "latest", ""}


# --------------------------------------------------------------------------
# repo coordinates
# --------------------------------------------------------------------------


def test_repo_id_is_the_diffusers_pack_not_the_single_file_release():
    assert model_files.REPO_ID == "Lightricks/LTX-2.5-Diffusers"


def test_revision_is_pinned_to_something_immutable():
    """PLAN §5.2: a repo whose content can change under us is not pinned just
    because its name is stable."""
    revision = getattr(model_files, "REVISION", None)
    assert isinstance(revision, str) and revision, "model_files.REVISION must exist"
    assert revision not in MUTABLE_REFS, (
        f"REVISION={revision!r} is a mutable ref. Pin a commit SHA (PLAN §5.2)."
    )


def test_transformer_full_is_declared_but_not_shipped_by_default():
    """PLAN §2.2/§2.4: the DEV transformer is 37.98 GB. Known about, not downloaded."""
    spec = model_files.get_component("transformer_full")
    assert spec.default is False
    assert "transformer_full" not in set(model_files.DEFAULT_COMPONENTS)
    assert round(spec.size_gb, 2) == 37.98


# --------------------------------------------------------------------------
# the component set (PLAN §2.3)
# --------------------------------------------------------------------------


def test_default_component_set_matches_the_plan():
    assert set(model_files.DEFAULT_COMPONENTS) == EXPECTED_DEFAULT_COMPONENTS


@pytest.mark.parametrize("name", sorted(EXPECTED_DEFAULT_COMPONENTS))
def test_default_component_is_declared(name):
    spec = model_files.get_component(name)
    assert spec.name == name


@pytest.mark.parametrize("name", sorted(EXPECTED_DEFAULT_COMPONENTS))
def test_default_component_is_marked_default(name):
    assert model_files.get_component(name).default is True


@pytest.mark.parametrize("name,size_gb", sorted(EXPECTED_SIZES_GB.items()))
def test_component_size_matches_the_plan(name, size_gb):
    """PLAN §2.3 quotes three decimals; the module may hold a more precise figure
    from the repo listing. The tolerance is 5 MB — enough for that rounding, far
    too little to hide a wrong shard set (Trap A costs tens of GB)."""
    assert model_files.get_component(name).size_gb == pytest.approx(size_gb, abs=0.005)


def test_shipped_subset_sums_to_the_stated_total():
    """PLAN §2.3: 72.185 GB. If this drifts, the 150 GB cap arithmetic in §2.4/§2.5
    no longer closes and the deploy discovers it 72 GB into a download."""
    total = sum(model_files.get_component(n).size_gb for n in model_files.DEFAULT_COMPONENTS)
    assert total == pytest.approx(72.185, abs=0.01)


def test_shipped_subset_fits_the_persistent_storage_cap_with_headroom():
    total = sum(model_files.get_component(n).size_gb for n in model_files.DEFAULT_COMPONENTS)
    assert total < 150.0
    assert 150.0 - total > 70.0  # PLAN §2.5 headroom


@pytest.mark.parametrize("name", sorted(EXPECTED_DEFAULT_COMPONENTS))
def test_every_component_carries_the_required_fields(name):
    spec = model_files.get_component(name)
    for field in ("name", "subfolder", "repo_id", "revision", "size_gb", "default", "url"):
        assert hasattr(spec, field), f"{name} spec is missing {field!r}"
    assert spec.repo_id, name
    assert spec.revision, name
    assert isinstance(spec.size_gb, (int, float)), name


def test_component_names_are_unique():
    names = [model_files.get_component(n).name for n in model_files.COMPONENTS]
    assert len(names) == len(set(names))


def test_component_subfolders_are_unique():
    subfolders = [model_files.get_component(n).subfolder for n in model_files.COMPONENTS]
    assert len(subfolders) == len(set(subfolders))


def test_the_module_is_data_not_logic():
    """PLAN §5.2/§7.1: "Data only — no logic, no I/O." Lookups and URL formatting
    are fine; downloading, opening files or importing torch is not."""
    source = (PACKAGE_ROOT / "model_files.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = imported & {
        "torch",
        "diffusers",
        "transformers",
        "huggingface_hub",
        "requests",
        "gradio",
        "safetensors",
    }
    assert not forbidden, f"model_files.py must not import {sorted(forbidden)} — it is data (§5.2)"


def test_model_files_imports_nothing_from_the_package_except_errors():
    """PLAN §7.2: ``model_files`` sits at the bottom of the dependency graph."""
    tree = ast.parse((PACKAGE_ROOT / "model_files.py").read_text())
    internal = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.level or "ltx25_studio" in node.module:
            internal.add(node.module)
    allowed = {"errors", "ltx25_studio.errors"}
    assert not (internal - allowed), f"model_files.py imports {sorted(internal - allowed)}"


# --------------------------------------------------------------------------
# failure naming (PLAN §5.2)
# --------------------------------------------------------------------------


def test_unknown_component_raises_and_names_the_known_ones():
    with pytest.raises(UnknownComponentError) as excinfo:
        model_files.get_component("prompt_enhancer")
    message = str(excinfo.value)
    assert "prompt_enhancer" in message
    assert "transformer" in message, f"the error must list what IS available; got {message!r}"


def test_missing_file_error_names_both_the_filename_and_its_url():
    """The 2.3 lesson: when Lightricks deletes a file, the message must tell the
    operator exactly which URL to go and look at."""
    with pytest.raises(MissingModelFileError) as excinfo:
        model_files.require_file("transformer", "diffusion_pytorch_model-00099-of-00004.safetensors")
    message = str(excinfo.value)
    assert "diffusion_pytorch_model-00099-of-00004.safetensors" in message
    assert "https://huggingface.co/Lightricks/LTX-2.5-Diffusers" in message


def test_missing_file_error_url_is_resolvable_shaped():
    with pytest.raises(MissingModelFileError) as excinfo:
        model_files.require_file("vae", "nope.safetensors")
    message = str(excinfo.value)
    assert "https://" in message
    assert "vae" in message


def test_file_url_points_at_the_pinned_revision():
    url = model_files.file_url("vae", "config.json")
    assert url.startswith("https://huggingface.co/Lightricks/LTX-2.5-Diffusers/")
    assert model_files.REVISION in url
    assert url.endswith("vae/config.json")


def test_component_url_points_at_its_subfolder():
    spec = model_files.get_component("transformer")
    assert spec.url.startswith("https://huggingface.co/Lightricks/LTX-2.5-Diffusers/")
    assert spec.url.rstrip("/").endswith("transformer")


# --------------------------------------------------------------------------
# THE architecture test (PLAN §5.2)
# --------------------------------------------------------------------------

#: Substrings that are never legitimate in code outside ``model_files.py``.
FORBIDDEN_ANYWHERE = ("Lightricks/", "huggingface.co")


def _is_a_hardcoded_filename(value):
    """True when the literal is a *filename*, not prose that mentions the extension.

    The rule is about coordinates, not vocabulary. A literal naming an actual
    file — ``"transformer/diffusion_pytorch_model-00001-of-00004.safetensors"`` —
    is a coordinate and belongs in ``model_files.py``. Two things are not:

    * prose that mentions the extension ("upload a `.safetensors` adapter") — the
      UI is the layer whose whole job is sentences;
    * the bare extension used as an upload filter (``file_types=[".safetensors"]``)
      — that describes what a *user* may upload, and no Lightricks rename touches it.
    """
    stripped = value.strip()
    if stripped == ".safetensors":
        return False
    return stripped.endswith(".safetensors") and not any(c.isspace() for c in stripped)


SOURCE_FILES_TO_SWEEP = package_python_files(exclude=("model_files.py",)) + [REPO_ROOT / "app.py"]


@pytest.mark.parametrize(
    "path", SOURCE_FILES_TO_SWEEP, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_hardcoded_hub_strings_outside_model_files(path):
    """An upstream rename must be a one-line change in one file (PLAN §5.2).

    Filenames, repo ids and Hub URLs belong to ``model_files.py`` and nowhere
    else. Comments and docstrings are exempt; code is not.
    """
    offenders = []
    for lineno, value in code_string_constants(path):
        for literal in FORBIDDEN_ANYWHERE:
            if literal in value:
                offenders.append((lineno, value, literal))
        if _is_a_hardcoded_filename(value):
            offenders.append((lineno, value, ".safetensors"))
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} hardcodes Hub coordinates: {offenders}. "
        "Move them into model_files.py (PLAN §5.2)."
    )
