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
* ``DEFAULT_COMPONENTS`` — the files the Space loads, summing to 71.92 GB, each
  file with its exact size (``file_size``).
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
#: by every pipeline constructor but absent from the §2.3 size table). Without
#: ``temporal_latent_upsampler``: nothing on any path loads it (#23).
EXPECTED_DEFAULT_COMPONENTS = {
    "transformer",
    "text_encoder",
    "tokenizer",
    "connectors",
    "vae",
    "audio_vae",
    "vocoder",
    "latent_upsampler",
    "diffusion_decoder",
    "duration_head",
    "scheduler",
}

#: Every file the Space loads, with its size in bytes, from the Hub's own listing
#: of ``Lightricks/LTX-2.5-Diffusers`` at the pinned revisions (2026-11-17): the
#: decoder's two files at ``aba1b702``, everything else at ``871165de``.
EXPECTED_FILE_BYTES = {
    "audio_vae/config.json": 505,
    "audio_vae/diffusion_pytorch_model.safetensors": 106_507_972,
    "connectors/config.json": 817,
    "connectors/diffusion_pytorch_model-00001-of-00002.safetensors": 4_999_998_520,
    "connectors/diffusion_pytorch_model-00002-of-00002.safetensors": 1_344_490_576,
    "connectors/diffusion_pytorch_model.safetensors.index.json": 30_629,
    "diffusion_decoder/config.json": 1_089,
    "diffusion_decoder/diffusion_pytorch_model.safetensors": 834_313_392,
    "duration_head/config.json": 251,
    "duration_head/diffusion_pytorch_model.safetensors": 3_808_466,
    "latent_upsampler/config.json": 301,
    "latent_upsampler/diffusion_pytorch_model.safetensors": 995_743_320,
    "model_index.json": 859,
    "scheduler/scheduler_config.json": 489,
    "text_encoder/config.json": 4_594,
    "text_encoder/generation_config.json": 255,
    "text_encoder/model-00001-of-00005.safetensors": 4_961_189_118,
    "text_encoder/model-00002-of-00005.safetensors": 4_966_688_198,
    "text_encoder/model-00003-of-00005.safetensors": 4_884_113_678,
    "text_encoder/model-00004-of-00005.safetensors": 4_986_341_908,
    "text_encoder/model-00005-of-00005.safetensors": 4_121_215_522,
    "text_encoder/model.safetensors.index.json": 66_306,
    "tokenizer/chat_template.jinja": 18_683,
    "tokenizer/tokenizer.json": 32_169_626,
    "tokenizer/tokenizer_config.json": 3_735,
    "transformer/config.json": 1_440,
    "transformer/diffusion_pytorch_model-00001-of-00004.safetensors": 9_932_336_016,
    "transformer/diffusion_pytorch_model-00002-of-00004.safetensors": 9_919_463_480,
    "transformer/diffusion_pytorch_model-00003-of-00004.safetensors": 9_986_144_288,
    "transformer/diffusion_pytorch_model-00004-of-00004.safetensors": 8_138_277_304,
    "transformer/diffusion_pytorch_model.safetensors.index.json": 446_505,
    "vae/config.json": 1_480,
    "vae/diffusion_pytorch_model.safetensors": 1_452_233_194,
    "vocoder/config.json": 1_478,
    "vocoder/diffusion_pytorch_model.safetensors": 258_305_056,
}

#: The sum of the table above: 35 files, 71.92 GB (66.98 GiB).
EXPECTED_TOTAL_BYTES = 71_923_919_050

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
    """71.924 GB (PLAN §2.3's 72.185 less the unloaded temporal upsampler). If this
    drifts, the 150 GB cap arithmetic no longer closes and the deploy discovers it
    72 GB into a download."""
    total = sum(model_files.get_component(n).size_gb for n in model_files.DEFAULT_COMPONENTS)
    assert total == pytest.approx(71.924, abs=0.001)


# --------------------------------------------------------------------------
# the download set, file by file (#23): one source for the list AND the sizes
# --------------------------------------------------------------------------


def test_the_download_set_is_exactly_the_files_the_space_loads():
    assert set(model_files.download_allow_list()) == set(EXPECTED_FILE_BYTES)


@pytest.mark.parametrize("path,size", sorted(EXPECTED_FILE_BYTES.items()))
def test_every_file_in_the_set_carries_its_exact_size(path, size):
    """The start-up names how much it must download when the image lacks files;
    the size of each file therefore lives beside its name, not in a second table."""
    assert model_files.file_size(path) == size


def test_the_set_totals_the_hubs_listing_to_the_byte():
    assert model_files.total_size_bytes() == EXPECTED_TOTAL_BYTES
    assert sum(map(model_files.file_size, model_files.download_allow_list())) == (
        EXPECTED_TOTAL_BYTES
    )
    assert model_files.total_size_gb() == EXPECTED_TOTAL_BYTES / 1e9


@pytest.mark.parametrize("name", sorted(EXPECTED_DEFAULT_COMPONENTS))
def test_a_components_size_is_the_sum_of_its_files(name):
    spec = model_files.get_component(name)
    assert spec.size_gb == sum(EXPECTED_FILE_BYTES[path] for path in spec.paths) / 1e9


def test_a_path_outside_the_set_has_no_size():
    with pytest.raises(MissingModelFileError) as caught:
        model_files.file_size("temporal_latent_upsampler/diffusion_pytorch_model.safetensors")
    assert "temporal_latent_upsampler" in str(caught.value)


def test_the_temporal_upsampler_is_declared_but_never_fetched():
    """It is the spatial upsampler's class on the other axis, and nothing loads it:
    ``StageRunner`` wires only ``latent_upsampler``. Fetching it put 0.26 GB on
    every start for nothing."""
    spec = model_files.get_component("temporal_latent_upsampler")
    assert spec.default is False
    assert "never loaded" in spec.excluded_because
    assert not any(
        path.startswith("temporal_latent_upsampler/")
        for path in model_files.download_allow_list()
    )


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
        model_files.require_file(
            "transformer", "diffusion_pytorch_model-00099-of-00004.safetensors"
        )
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
      — that describes what a *user* may upload, and no Lightricks rename touches it;
    * the diffusers ecosystem's LoRA weight-name convention
      (``pytorch_lora_weights.safetensors``, ``lora_base.py``'s
      ``LORA_WEIGHT_NAME_SAFE``) — the name a *community* repo gives its
      adapter, which the Hub fetch route honours when a multi-file repo is
      otherwise ambiguous. It is a contract with the ecosystem, like the
      upload filter, not a coordinate of ours: no Lightricks rename touches
      it, and moving it into ``model_files.py`` would misfile a diffusers
      convention among our checkpoint pins.
    """
    stripped = value.strip()
    if stripped == ".safetensors":
        return False
    if stripped == "pytorch_lora_weights.safetensors":
        return False
    return stripped.endswith(".safetensors") and not any(c.isspace() for c in stripped)


SOURCE_FILES_TO_SWEEP = package_python_files(exclude=("model_files.py",)) + [REPO_ROOT / "app.py"]


@pytest.mark.parametrize("path", SOURCE_FILES_TO_SWEEP, ids=lambda p: str(p.relative_to(REPO_ROOT)))
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


# --------------------------------------------------------------------------
# the diffusion decoder at its own revision (F50)
#
# Lightricks replaced the decoder's weights after the pinned revision; the file
# at REVISION differs in 405 of 407 tensors. Only that component is re-pinned.
# --------------------------------------------------------------------------

DECODER_REVISION = "aba1b70255c292c8a06702008dd376f0d3007264"


def test_exactly_one_component_is_pinned_to_a_different_revision():
    off_pin = {
        name: spec.revision
        for name, spec in model_files.COMPONENTS.items()
        if spec.revision != model_files.REVISION
    }
    assert off_pin == {"diffusion_decoder": DECODER_REVISION}, off_pin
    assert model_files.REVISION == "871165de037793df7d75c23a02dd7f45fd364c97"


def test_the_download_plan_fetches_the_decoder_at_its_own_revision():
    plan = dict(model_files.download_plan())

    assert list(plan) == [model_files.REVISION, DECODER_REVISION]
    assert plan[DECODER_REVISION] == (
        "diffusion_decoder/config.json",
        "diffusion_decoder/diffusion_pytorch_model.safetensors",
    )
    assert "model_index.json" in plan[model_files.REVISION]
    assert not any(path.startswith("diffusion_decoder/") for path in plan[model_files.REVISION])
    assert sum(len(paths) for paths in plan.values()) == len(model_files.download_allow_list())


def test_the_startup_download_asks_for_each_file_at_its_revision(monkeypatch, tmp_path):
    import sys
    import types

    from ltx25_studio import preflight

    asked = []
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: asked.append(kwargs) or f"/x/{kwargs['filename']}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delenv("LTX25_MODEL_SNAPSHOT_DIR", raising=False)
    # An empty cache of the test's own: the start-up fetches only what is missing,
    # so whatever this machine's own cache holds must not decide what is asked for.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    # Nor this machine's free disk: room for the whole set, so its room check passes.
    import shutil

    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: types.SimpleNamespace(total=0, used=0, free=10**12)
    )

    preflight.ensure_weights()

    revision_of = {call["filename"]: call["revision"] for call in asked}
    assert revision_of["diffusion_decoder/diffusion_pytorch_model.safetensors"] == DECODER_REVISION
    assert revision_of["diffusion_decoder/config.json"] == DECODER_REVISION
    assert revision_of["vae/diffusion_pytorch_model.safetensors"] == model_files.REVISION
    assert len(asked) == len(model_files.download_allow_list())


def test_the_preload_entries_are_one_per_revision_then_the_enhancers():
    entries = model_files.preload_from_hub_entries()

    assert [entry.rsplit(" ", 1)[1] for entry in entries] == [
        model_files.REVISION,
        DECODER_REVISION,
        ENHANCER_REVISION,
    ]
    assert model_files.preload_from_hub_entry() == entries[0]


# --------------------------------------------------------------------------
# the prompt enhancer is preloaded too (#23): its own repo, at its own pin
#
# A first enhanced call in a fresh container whose start-up prefetch has not
# finished would pay the ~88 s fetch inside its GPU reservation. Its files come
# from the image instead; the prefetch thread stays as the fallback.
# --------------------------------------------------------------------------

ENHANCER_REPO = "google/gemma-4-E2B-it-qat-mobile-transformers"
ENHANCER_REVISION = "dd693ff40353f057ca5f07e945ad867f4afbf2ec"

#: What the loader reads from that repo at that pin (AutoProcessor: the processor,
#: image-processor and tokenizer configs, the tokenizer, the chat template;
#: Gemma4ForConditionalGeneration: the config, the generation config, the
#: single-file checkpoint), with sizes. The repo's README and .gitattributes are
#: not read.
EXPECTED_ENHANCER_FILE_BYTES = {
    "chat_template.jinja": 18_569,
    "config.json": 6_204,
    "generation_config.json": 209,
    "model.safetensors": 2_458_111_846,
    "preprocessor_config.json": 511,
    "processor_config.json": 1_689,
    "tokenizer.json": 32_169_626,
    "tokenizer_config.json": 3_082,
}


def test_the_enhancers_entry_names_exactly_its_files_at_its_pin():
    from ltx25_studio import config, enhance

    repo, files, revision = model_files.preload_from_hub_entries()[-1].split(" ")

    assert repo == ENHANCER_REPO == config.DEFAULT_PROMPT_ENHANCER_REPO
    assert revision == ENHANCER_REVISION == enhance.ENHANCER_REVISION
    assert files.split(",") == sorted(EXPECTED_ENHANCER_FILE_BYTES)


def test_the_enhancers_files_carry_their_sizes():
    assert dict(model_files.ENHANCER_SIZED_FILES) == EXPECTED_ENHANCER_FILE_BYTES
    assert sum(size for _name, size in model_files.ENHANCER_SIZED_FILES) == 2_490_311_736


def test_the_enhancers_list_is_the_files_its_loader_reads():
    """Tied to the loader by its own library's names, so a loader that starts reading
    another file shows up here, not as a download inside a GPU reservation."""
    utils = pytest.importorskip("transformers.utils")
    tokenization = pytest.importorskip("transformers.tokenization_utils_base")

    read = {
        utils.CONFIG_NAME,
        utils.GENERATION_CONFIG_NAME,
        utils.SAFE_WEIGHTS_NAME,
        utils.PROCESSOR_NAME,
        utils.IMAGE_PROCESSOR_NAME,
        utils.CHAT_TEMPLATE_FILE,
        tokenization.TOKENIZER_CONFIG_FILE,
        tokenization.FULL_TOKENIZER_FILE,
    }
    assert set(model_files.ENHANCER_FILES) == read


@pytest.mark.parametrize(
    "path", SOURCE_FILES_TO_SWEEP, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_the_enhancers_repo_and_pin_are_written_in_model_files_only(path):
    """One source for what the preload fetches and what the loader asks for."""
    offenders = [
        (lineno, value)
        for lineno, value in code_string_constants(path)
        if ENHANCER_REPO in value or ENHANCER_REVISION in value
    ]
    assert not offenders, f"{path.relative_to(REPO_ROOT)}: {offenders}"


def test_the_decoder_directory_is_its_own_revisions_snapshot(monkeypatch, tmp_path):
    from ltx25_studio import config, preflight

    monkeypatch.delenv(config.ENV_MODEL_SNAPSHOT_DIR, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = preflight.cache_repo_dir()

    assert preflight.component_dir("diffusion_decoder") == (
        repo / "snapshots" / DECODER_REVISION / "diffusion_decoder"
    )
    assert preflight.component_dir("vae") == repo / "snapshots" / model_files.REVISION / "vae"


def test_the_snapshot_override_still_holds_every_component(monkeypatch, tmp_path):
    from ltx25_studio import config, preflight

    (tmp_path / "model_index.json").write_text("{}")
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path))

    assert preflight.component_dir("diffusion_decoder") == tmp_path / "diffusion_decoder"
    assert preflight.component_dir("latent_upsampler") == tmp_path / "latent_upsampler"


def test_the_completeness_check_looks_for_the_decoder_under_its_revision(monkeypatch, tmp_path):
    from ltx25_studio import config, preflight

    monkeypatch.delenv(config.ENV_MODEL_SNAPSHOT_DIR, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = preflight.cache_repo_dir()
    for path in model_files.download_allow_list():
        target = repo / "snapshots" / model_files.REVISION / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")

    missing = preflight.missing_cached_files()

    assert set(missing) == {
        "diffusion_decoder/config.json",
        "diffusion_decoder/diffusion_pytorch_model.safetensors",
    }, "the decoder at the old revision must not count as present"
    with pytest.raises(MissingModelFileError) as caught:
        preflight.require_cache_complete()
    assert DECODER_REVISION in str(caught.value)


# --------------------------------------------------------------------------
# the strict load: decoder.type_emb tolerated, nothing else (real classes)
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_decoder_dir(tmp_path, monkeypatch):
    """A tiny random decoder saved to an override snapshot, as safetensors."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers import LTX2VideoDiffusionDecoderModel

    from ltx25_studio import config

    (tmp_path / "model_index.json").write_text("{}")
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path))
    torch.manual_seed(0)
    model = LTX2VideoDiffusionDecoderModel(
        latent_channels=8,
        decoder_head_dim=8,
        decoder_stage_channels=[64, 32, 16, 16, 8],
        decoder_stage_depths=[1, 1, 1, 1, 1],
        decoder_t_emb_dim=32,
    )
    model.save_pretrained(tmp_path / "diffusion_decoder", safe_serialization=True)
    return tmp_path / "diffusion_decoder"


def _edit_weights(directory, edit):
    from safetensors.torch import load_file, save_file

    path = directory / "diffusion_pytorch_model.safetensors"
    tensors = load_file(str(path))
    edit(tensors)
    save_file(tensors, str(path), metadata={"format": "pt"})


def _load_decoder():
    import torch
    from diffusers import LTX2VideoDiffusionDecoderModel

    from ltx25_studio import pipelines

    return pipelines._load_part(
        LTX2VideoDiffusionDecoderModel, "diffusion_decoder", dtype=torch.float32
    )


def test_the_decoders_one_extra_key_is_tolerated(tiny_decoder_dir):
    import torch

    _edit_weights(tiny_decoder_dir, lambda t: t.update({"decoder.type_emb": torch.zeros(8)}))

    model = _load_decoder()

    assert not hasattr(model.decoder, "type_emb")


@pytest.mark.parametrize("change", ["unexpected", "missing"])
def test_anything_else_off_the_class_is_a_load_error(tiny_decoder_dir, change):
    import torch

    from ltx25_studio.errors import ModelLoadError

    def edit(tensors):
        tensors["decoder.type_emb"] = torch.zeros(8)
        if change == "unexpected":
            tensors["decoder.something_new"] = torch.zeros(4)
        else:
            tensors.pop("decoder.conv_out.bias")

    _edit_weights(tiny_decoder_dir, edit)

    with pytest.raises(ModelLoadError) as caught:
        _load_decoder()
    assert "diffusion_decoder" in str(caught.value)


def test_nothing_is_fetched_when_the_snapshot_override_is_set(monkeypatch, tmp_path):
    """Review space-path-3: nothing loads the pinned repo under the override."""
    import sys
    import types

    from ltx25_studio import preflight

    asked = []
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: asked.append(kwargs) or pytest.fail("fetched")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    (tmp_path / "model_index.json").write_text("{}")
    monkeypatch.setenv("LTX25_MODEL_SNAPSHOT_DIR", str(tmp_path))

    assert preflight.ensure_weights(token=None) == ()
    assert asked == []
