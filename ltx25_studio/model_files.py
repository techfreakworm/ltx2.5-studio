"""THE single source of truth for every Hub coordinate this app uses.

PLAN §5.2, §2.1, §2.3, §2.4. **Data only** -- no I/O, no torch, no
``huggingface_hub``. Lookups and URL formatting are the only logic permitted.

**Why this file exists at all.** Lightricks deleted an LTX-2.3 checkpoint from
the Hub with no warning and broke a live Space. The mitigation is not "be
careful", it is structural: every repo id, revision, subfolder and filename
lives here and nowhere else, so the next rename is a one-line change. The
architecture test ``test_no_hardcoded_hub_strings_outside_model_files`` is what
keeps that true after everyone forgets this paragraph.

**Two loading mechanisms, deliberately modelled.** Most components are listed in
``model_index.json`` and arrive with the pipeline. ``latent_upsampler`` is
**not** in the index -- it loads by subfolder through its own
``from_pretrained`` call (as would ``temporal_latent_upsampler``, which nothing
loads and which is therefore not fetched). A resolver that assumes one mechanism
silently misses the component the stage-2 quality path needs, so
:class:`ComponentSpec` records which is which.

**Every file carries its size in bytes**, from the Hub's own listing at the
pinned revision: the download set, the Space's ``preload_from_hub`` list and the
start-up's "how much is missing" all read this one table.

**The duplicate-shard trap (PLAN §2.1).** ``transformer/`` ships a 4-shard set
AND an 8-shard set of the same weights; ``connectors/`` ships a 2-shard set AND
a single-file copy. Only one of each is referenced by the index. The live sets
recorded below were read from the index files themselves:

* ``transformer/`` -> the **4-shard** set is live (the 8-shard set is 37.99 GB of
  dead weight)
* ``connectors/`` -> the **2-shard** set is live (the single 6.34 GB file is dead)

A naive ``snapshot_download`` of those folders pays 44.33 GB for nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .errors import MissingModelFileError, UnknownComponentError

__all__ = [
    "COMPONENTS",
    "DEFAULT_COMPONENTS",
    "DEFAULT_DTYPE",
    "DIFFUSION_DECODER_REVISION",
    "HUB_BASE_URL",
    "REPO_ID",
    "REVISION",
    "ROOT_FILES",
    "TOLERATED_UNEXPECTED_KEYS",
    "TINY_REPO_ID",
    "TINY_REVISION",
    "ComponentSpec",
    "component_dir",
    "config_filename",
    "download_allow_list",
    "download_plan",
    "file_size",
    "file_url",
    "get_component",
    "local_path",
    "missing_config_files",
    "preload_from_hub_entries",
    "preload_from_hub_entry",
    "require_file",
    "revision_of_path",
    "revisions",
    "total_size_bytes",
    "total_size_gb",
]

#: The Diffusers pack. NOT ``Lightricks/LTX-2.5`` (the single-file release):
#: diffusers has no single-file converter for 2.5's connectors, diffusion
#: decoder, vocoder, duration head or latent upsampler (PLAN §1.4).
REPO_ID = "Lightricks/LTX-2.5-Diffusers"

#: Pinned commit SHA. A branch name is not a pin -- the content behind ``main``
#: can change under us without any signal (PLAN §5.2).
REVISION = "871165de037793df7d75c23a02dd7f45fd364c97"

#: The diffusion decoder alone is pinned later: Lightricks replaced its weights
#: on 2026-10-08 ("Decoder weights match the source bitwise"); the file at
#: :data:`REVISION` differs in 405 of 407 tensors. Its config is unchanged and
#: the new file adds one tensor the pinned class does not have
#: (:data:`TOLERATED_UNEXPECTED_KEYS`). The global revision stays: the newer
#: ``model_index.json`` declares components this app neither fetches nor nulls.
DIFFUSION_DECODER_REVISION = "aba1b70255c292c8a06702008dd376f0d3007264"

#: Keys a component's checkpoint may carry that its pinned diffusers class does
#: not load. Anything else unexpected, missing or mismatched is a load error.
TOLERATED_UNEXPECTED_KEYS = MappingProxyType({"diffusion_decoder": ("decoder.type_emb",)})

HUB_BASE_URL = "https://huggingface.co"

#: A public ~95 MB random-weight checkpoint with this pipeline's layout, for the
#: real-pipeline test tier only (``scripts/local_smoke.py``). Never a default.
TINY_REPO_ID = "optimum-intel-internal-testing/tiny-random-ltx2.5"
TINY_REVISION = "3f43040b4719f6dad170a6d07b30921ee1a8f405"

#: The checkpoint's on-disk dtype. Quantisation is a load-time choice
#: (``LTX25_QUANTIZATION``), not a different download.
DEFAULT_DTYPE = "bfloat16"

#: Repo-root files needed alongside the components, with their sizes in bytes.
_ROOT_FILE_BYTES = MappingProxyType({"model_index.json": 859})

#: Repo-root files needed alongside the components.
ROOT_FILES = tuple(_ROOT_FILE_BYTES)


@dataclass(frozen=True)
class ComponentSpec:
    """One model component: where it lives, what it weighs, how it loads."""

    name: str
    subfolder: str
    default: bool
    #: Every file we must fetch for this component -- and ONLY the live shards --
    #: each with its size in bytes at ``revision``, from the Hub's listing.
    sized_files: tuple[tuple[str, int], ...] = ()
    #: True when ``model_index.json`` lists this component, i.e. it loads as
    #: part of the pipeline. False when it loads via its own subfolder call.
    in_model_index: bool = True
    #: Why it is excluded, for components with ``default=False``.
    excluded_because: str = ""
    repo_id: str = REPO_ID
    revision: str = REVISION
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def files(self) -> tuple[str, ...]:
        """The file names, without their sizes."""
        return tuple(name for name, _size in self.sized_files)

    @property
    def size_bytes(self) -> int:
        """The component's download size: the sum of its files."""
        return sum(size for _name, size in self.sized_files)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    @property
    def url(self) -> str:
        """Browsable URL for the component's folder at the pinned revision."""
        return f"{HUB_BASE_URL}/{self.repo_id}/tree/{self.revision}/{self.subfolder}"

    @property
    def paths(self) -> tuple[str, ...]:
        """Repo-relative paths for every file this component needs."""
        return tuple(f"{self.subfolder}/{name}" for name in self.files)


_SPECS = (
    # -- what the Space loads (PLAN §2.3, less the temporal upsampler) --------
    ComponentSpec(
        name="transformer",
        subfolder="transformer",
        default=True,
        sized_files=(
            ("config.json", 1_440),
            ("diffusion_pytorch_model.safetensors.index.json", 446_505),
            ("diffusion_pytorch_model-00001-of-00004.safetensors", 9_932_336_016),
            ("diffusion_pytorch_model-00002-of-00004.safetensors", 9_919_463_480),
            ("diffusion_pytorch_model-00003-of-00004.safetensors", 9_986_144_288),
            ("diffusion_pytorch_model-00004-of-00004.safetensors", 8_138_277_304),
        ),
    ),
    ComponentSpec(
        name="text_encoder",
        subfolder="text_encoder",
        default=True,
        sized_files=(
            ("config.json", 4_594),
            ("generation_config.json", 255),
            ("model.safetensors.index.json", 66_306),
            ("model-00001-of-00005.safetensors", 4_961_189_118),
            ("model-00002-of-00005.safetensors", 4_966_688_198),
            ("model-00003-of-00005.safetensors", 4_884_113_678),
            ("model-00004-of-00005.safetensors", 4_986_341_908),
            ("model-00005-of-00005.safetensors", 4_121_215_522),
        ),
    ),
    ComponentSpec(
        name="connectors",
        subfolder="connectors",
        default=True,
        sized_files=(
            ("config.json", 817),
            ("diffusion_pytorch_model.safetensors.index.json", 30_629),
            ("diffusion_pytorch_model-00001-of-00002.safetensors", 4_999_998_520),
            ("diffusion_pytorch_model-00002-of-00002.safetensors", 1_344_490_576),
        ),
    ),
    ComponentSpec(
        name="vae",
        subfolder="vae",
        default=True,
        sized_files=(("config.json", 1_480), ("diffusion_pytorch_model.safetensors", 1_452_233_194)),
    ),
    ComponentSpec(
        name="latent_upsampler",
        subfolder="latent_upsampler",
        default=True,
        in_model_index=False,
        sized_files=(("config.json", 301), ("diffusion_pytorch_model.safetensors", 995_743_320)),
    ),
    ComponentSpec(
        name="diffusion_decoder",
        subfolder="diffusion_decoder",
        default=True,
        sized_files=(("config.json", 1_089), ("diffusion_pytorch_model.safetensors", 834_313_392)),
        revision=DIFFUSION_DECODER_REVISION,
    ),
    ComponentSpec(
        name="vocoder",
        subfolder="vocoder",
        default=True,
        sized_files=(("config.json", 1_478), ("diffusion_pytorch_model.safetensors", 258_305_056)),
    ),
    ComponentSpec(
        name="audio_vae",
        subfolder="audio_vae",
        default=True,
        sized_files=(("config.json", 505), ("diffusion_pytorch_model.safetensors", 106_507_972)),
    ),
    ComponentSpec(
        name="tokenizer",
        subfolder="tokenizer",
        default=True,
        sized_files=(
            ("tokenizer.json", 32_169_626),
            ("tokenizer_config.json", 3_735),
            ("chat_template.jinja", 18_683),
        ),
    ),
    ComponentSpec(
        name="duration_head",
        subfolder="duration_head",
        default=True,
        sized_files=(("config.json", 251), ("diffusion_pytorch_model.safetensors", 3_808_466)),
    ),
    ComponentSpec(
        name="scheduler",
        subfolder="scheduler",
        default=True,
        sized_files=(("scheduler_config.json", 489),),
    ),
    # -- declared, deliberately NOT shipped (PLAN §2.4) -----------------------
    ComponentSpec(
        name="temporal_latent_upsampler",
        subfolder="temporal_latent_upsampler",
        default=False,
        in_model_index=False,
        excluded_because=(
            "never loaded: StageRunner wires only the spatial latent_upsampler, and neither "
            "documented recipe uses the temporal one. Fetching it cost 0.26 GB on every start."
        ),
        sized_files=(("config.json", 299), ("diffusion_pytorch_model.safetensors", 261_943_760)),
    ),
    ComponentSpec(
        name="transformer_full",
        subfolder="transformer_full",
        default=False,
        excluded_because=(
            "the DEV (non-distilled) transformer. The distilled 8-step path does not need it. "
            "It fits the remaining headroom, so it is an opt-in rather than an impossibility -- "
            "but enabling it leaves ~40 GB and makes output eviction urgent."
        ),
        sized_files=(
            ("config.json", 1_440),
            ("diffusion_pytorch_model.safetensors.index.json", 446_505),
            ("diffusion_pytorch_model-00001-of-00004.safetensors", 9_932_336_016),
            ("diffusion_pytorch_model-00002-of-00004.safetensors", 9_919_463_480),
            ("diffusion_pytorch_model-00003-of-00004.safetensors", 9_986_144_288),
            ("diffusion_pytorch_model-00004-of-00004.safetensors", 8_138_277_304),
        ),
    ),
)

#: Every declared component, shipped or not. Read-only by construction.
COMPONENTS = MappingProxyType({spec.name: spec for spec in _SPECS})

#: The components the Space loads, and so fetches. Ordered largest-first so a
#: preflight or a progress display surfaces the expensive downloads first.
DEFAULT_COMPONENTS = tuple(spec.name for spec in _SPECS if spec.default)


def get_component(name: str) -> ComponentSpec:
    """Look up a component, or raise naming what IS declared."""
    try:
        return COMPONENTS[name]
    except KeyError:
        raise UnknownComponentError(name=name, known=tuple(COMPONENTS)) from None


def file_url(component: str, filename: str) -> str:
    """Resolvable URL for one file, at the pinned revision."""
    spec = get_component(component)
    return f"{HUB_BASE_URL}/{spec.repo_id}/resolve/{spec.revision}/{spec.subfolder}/{filename}"


def require_file(component: str, filename: str) -> str:
    """Return the URL for a declared file, or raise naming the file AND its URL.

    The 2.3 lesson made executable: when an upstream file disappears, the
    message must tell the operator exactly which page to open.
    """
    spec = get_component(component)
    url = file_url(component, filename)
    if filename not in spec.files:
        raise MissingModelFileError(filename=filename, url=url)
    return url


def file_size(path: str, components: tuple[str, ...] | None = None) -> int:
    """Bytes of one repo-relative path in a component set's download, at its revision.

    Raises naming the path and its folder's URL when the path is not in the set.
    """
    if path in _ROOT_FILE_BYTES:
        return _ROOT_FILE_BYTES[path]
    names = DEFAULT_COMPONENTS if components is None else components
    for name in names:
        spec = get_component(name)
        for filename, size in spec.sized_files:
            if f"{spec.subfolder}/{filename}" == path:
                return size
    raise MissingModelFileError(
        filename=f"{path} (not in the download set)",
        url=f"{HUB_BASE_URL}/{REPO_ID}/tree/{REVISION}/{path.split('/', 1)[0]}",
    )


def total_size_bytes(components: tuple[str, ...] | None = None) -> int:
    """Bytes of a component set's whole download, root files included."""
    return sum(file_size(path, components) for path in download_allow_list(components))


def total_size_gb(components: tuple[str, ...] | None = None) -> float:
    """Total download size for a component set, defaulting to what the Space loads."""
    return total_size_bytes(components) / 1e9


def component_dir(name: str, *, repo_cache, override=None) -> Path:
    """The directory component ``name`` loads from. Every reader resolves it here.

    ``repo_cache`` is the repo's HF cache directory (``models--org--name``); the
    HF cache keeps each revision in its own ``snapshots/<revision>``, so a
    component lives under ITS revision. ``override`` (``LTX25_MODEL_SNAPSHOT_DIR``)
    holds every component. Path arithmetic only: nothing is read.
    """
    spec = get_component(name)
    if override is not None:
        return Path(override) / spec.subfolder
    return Path(repo_cache) / "snapshots" / spec.revision / spec.subfolder


def local_path(path: str, *, repo_cache, override=None) -> Path:
    """Where one allow-listed repo-relative path lives on disk (see :func:`component_dir`)."""
    if override is not None:
        return Path(override) / path
    return Path(repo_cache) / "snapshots" / revision_of_path(path) / path


def revisions(components: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The revisions a component set is fetched at: :data:`REVISION` first."""
    names = DEFAULT_COMPONENTS if components is None else components
    others = sorted({get_component(name).revision for name in names} - {REVISION})
    return (REVISION, *others)


def revision_of_path(path: str) -> str:
    """The revision a repo-relative allow-list path is fetched at."""
    subfolder = path.split("/", 1)[0]
    for spec in _SPECS:
        if spec.subfolder == subfolder and path in spec.paths:
            return spec.revision
    return REVISION


def download_plan(
    components: tuple[str, ...] | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((revision, paths), ...)``: every allow-listed path, grouped by its revision.

    The HF cache keeps each revision in its own snapshot directory, so a file is
    fetched, checked and loaded at the revision its component is pinned to.
    """
    paths = download_allow_list(components)
    return tuple(
        (revision, tuple(path for path in paths if revision_of_path(path) == revision))
        for revision in revisions(components)
    )


def download_allow_list(components: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Every repo-relative path to fetch, sorted -- and only the live shards.

    Across revisions: :func:`download_plan` groups them by the revision each is
    fetched at.

    PLAN §5.2: the runtime download and the Space README ``preload_from_hub:``
    list are generated from this one function, never hand-maintained in two
    places. A drift test compares the committed README against this output, so
    the duplicate-shard trap (§2.1) cannot silently reappear after an upstream
    re-shard.
    """
    names = DEFAULT_COMPONENTS if components is None else components
    incomplete = missing_config_files(names)
    if incomplete:
        raise MissingModelFileError(
            filename=", ".join(f"{n}/{config_filename(n)}" for n in incomplete),
            url=f"{HUB_BASE_URL}/{REPO_ID}/tree/{REVISION}",
        )
    paths: list[str] = list(ROOT_FILES)
    for name in names:
        paths.extend(get_component(name).paths)
    # Sorted, so the emitted string is canonical. A drift test comparing the
    # README entry to this output compares STRINGS; an unstable order would make
    # it fail for no reason, or worse, make someone "fix" it by loosening the
    # comparison to a set and lose the revision check with it.
    return tuple(sorted(paths))


#: The config file each component needs in order to be constructed. Most use
#: ``config.json``; schedulers and tokenizers follow their own conventions.
CONFIG_FILENAMES = MappingProxyType(
    {
        "scheduler": "scheduler_config.json",
        "tokenizer": "tokenizer_config.json",
    }
)
DEFAULT_CONFIG_FILENAME = "config.json"


def config_filename(component: str) -> str:
    """The config file that ``from_pretrained`` needs for this component.

    ``scheduler/`` uses ``scheduler_config.json``, not ``config.json`` -- a
    preload list built from a naive ``{component}/config.json`` pattern misses
    it and the Space fails to construct the scheduler.
    """
    return CONFIG_FILENAMES.get(component, DEFAULT_CONFIG_FILENAME)


def missing_config_files(components: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Components whose file list omits their config file.

    A weight-map index tells you which *weights* are live. It says nothing about
    the sibling config files -- so a file list generated from the index alone
    silently drops ``config.json`` for every sharded component, and
    ``from_pretrained`` cannot construct a component without it. That failure
    surfaces only after the full download.

    This is the cheap self-check that catches it.
    """
    names = DEFAULT_COMPONENTS if components is None else components
    return tuple(name for name in names if config_filename(name) not in get_component(name).files)


def preload_from_hub_entries(components: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The ``preload_from_hub:`` entries for the Space README, one per revision."""
    return tuple(
        f"{REPO_ID} {','.join(paths)} {revision}" for revision, paths in download_plan(components)
    )


def preload_from_hub_entry(components: tuple[str, ...] | None = None) -> str:
    """The ``preload_from_hub:`` entry for :data:`REVISION` (see :func:`preload_from_hub_entries`).

    Format is ``"repo_id file1,file2,... commit_sha"``. One entry per revision
    keeps us inside the platform's 10-entry cap despite needing far more than 10
    files.

    **The revision is not optional.** If the preload fetches ``main`` while the
    app loads a pinned SHA, those are *different cache keys*: the preload is
    wasted and the app re-downloads 72 GB at runtime, on top of the copy already
    on disk. Any drift test must compare the whole entry, revision included --
    not just the file list.
    """
    return preload_from_hub_entries(components)[0]
