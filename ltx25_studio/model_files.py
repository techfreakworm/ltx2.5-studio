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
``model_index.json`` and arrive with the pipeline. ``latent_upsampler`` and
``temporal_latent_upsampler`` are **not** in the index -- they load by subfolder
through their own ``from_pretrained`` calls. A resolver that assumes one
mechanism silently misses exactly the two components the stage-2 quality path
needs, so :class:`ComponentSpec` records which is which.

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
from types import MappingProxyType

from .errors import MissingModelFileError, UnknownComponentError

__all__ = [
    "COMPONENTS",
    "DEFAULT_COMPONENTS",
    "DEFAULT_DTYPE",
    "HUB_BASE_URL",
    "REPO_ID",
    "REVISION",
    "ROOT_FILES",
    "ComponentSpec",
    "download_allow_list",
    "file_url",
    "get_component",
    "preload_from_hub_entry",
    "require_file",
    "total_size_gb",
]

#: The Diffusers pack. NOT ``Lightricks/LTX-2.5`` (the single-file release):
#: diffusers has no single-file converter for 2.5's connectors, diffusion
#: decoder, vocoder, duration head or latent upsampler (PLAN §1.4).
REPO_ID = "Lightricks/LTX-2.5-Diffusers"

#: Pinned commit SHA. A branch name is not a pin -- the content behind ``main``
#: can change under us without any signal (PLAN §5.2).
REVISION = "871165de037793df7d75c23a02dd7f45fd364c97"

HUB_BASE_URL = "https://huggingface.co"

#: Weight precision. Default is bf16: it is what the model card documents and
#: what the official Space pattern uses. Quantised variants are a documented
#: fallback rather than the default, because ZeroGPU's tensor packer is
#: unverified against bitsandbytes/torchao custom tensor subclasses (PLAN §2.6,
#: technical risk #1). Switching lives here so it stays a one-line change.
DEFAULT_DTYPE = "bfloat16"

#: Repo-root files needed alongside the components.
ROOT_FILES = ("model_index.json",)


@dataclass(frozen=True)
class ComponentSpec:
    """One model component: where it lives, what it weighs, how it loads."""

    name: str
    subfolder: str
    size_gb: float
    default: bool
    #: Every file we must fetch for this component -- and ONLY the live shards.
    files: tuple[str, ...] = ()
    #: True when ``model_index.json`` lists this component, i.e. it loads as
    #: part of the pipeline. False when it loads via its own subfolder call.
    in_model_index: bool = True
    #: Why it is excluded, for components with ``default=False``.
    excluded_because: str = ""
    repo_id: str = REPO_ID
    revision: str = REVISION
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def url(self) -> str:
        """Browsable URL for the component's folder at the pinned revision."""
        return f"{HUB_BASE_URL}/{self.repo_id}/tree/{self.revision}/{self.subfolder}"

    @property
    def paths(self) -> tuple[str, ...]:
        """Repo-relative paths for every file this component needs."""
        return tuple(f"{self.subfolder}/{name}" for name in self.files)


_SPECS = (
    # -- the shipped subset (PLAN §2.3) ---------------------------------------
    ComponentSpec(
        name="transformer",
        subfolder="transformer",
        size_gb=37.9762,
        default=True,
        files=(
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model-00001-of-00004.safetensors",
            "diffusion_pytorch_model-00002-of-00004.safetensors",
            "diffusion_pytorch_model-00003-of-00004.safetensors",
            "diffusion_pytorch_model-00004-of-00004.safetensors",
        ),
    ),
    ComponentSpec(
        name="text_encoder",
        subfolder="text_encoder",
        size_gb=23.9195,
        default=True,
        files=(
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ),
    ),
    ComponentSpec(
        name="connectors",
        subfolder="connectors",
        size_gb=6.3445,
        default=True,
        files=(
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model-00001-of-00002.safetensors",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
        ),
    ),
    ComponentSpec(
        name="vae",
        subfolder="vae",
        size_gb=1.4522,
        default=True,
        files=("config.json", "diffusion_pytorch_model.safetensors"),
    ),
    ComponentSpec(
        name="latent_upsampler",
        subfolder="latent_upsampler",
        size_gb=0.9957,
        default=True,
        in_model_index=False,
        files=("config.json", "diffusion_pytorch_model.safetensors"),
    ),
    ComponentSpec(
        name="diffusion_decoder",
        subfolder="diffusion_decoder",
        size_gb=0.8343,
        default=True,
        files=("config.json", "diffusion_pytorch_model.safetensors"),
    ),
    ComponentSpec(
        name="temporal_latent_upsampler",
        subfolder="temporal_latent_upsampler",
        size_gb=0.2619,
        default=True,
        in_model_index=False,
        files=("config.json", "diffusion_pytorch_model.safetensors"),
    ),
    ComponentSpec(
        name="vocoder",
        subfolder="vocoder",
        size_gb=0.2583,
        default=True,
        files=("config.json", "diffusion_pytorch_model.safetensors"),
    ),
    ComponentSpec(
        name="audio_vae",
        subfolder="audio_vae",
        size_gb=0.1065,
        default=True,
        files=("config.json", "diffusion_pytorch_model.safetensors"),
    ),
    ComponentSpec(
        name="tokenizer",
        subfolder="tokenizer",
        size_gb=0.0322,
        default=True,
        files=("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"),
    ),
    ComponentSpec(
        name="duration_head",
        subfolder="duration_head",
        size_gb=0.0038,
        default=True,
        files=("config.json", "diffusion_pytorch_model.safetensors"),
    ),
    ComponentSpec(
        name="scheduler",
        subfolder="scheduler",
        size_gb=0.0,
        default=True,
        files=("scheduler_config.json",),
    ),
    # -- declared, deliberately NOT shipped (PLAN §2.4) -----------------------
    ComponentSpec(
        name="transformer_full",
        subfolder="transformer_full",
        size_gb=37.9766,
        default=False,
        excluded_because=(
            "the DEV (non-distilled) transformer. The distilled 8-step path does not need it. "
            "It fits the remaining headroom, so it is an opt-in rather than an impossibility -- "
            "but enabling it leaves ~40 GB and makes output eviction urgent."
        ),
        files=(
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model-00001-of-00004.safetensors",
            "diffusion_pytorch_model-00002-of-00004.safetensors",
            "diffusion_pytorch_model-00003-of-00004.safetensors",
            "diffusion_pytorch_model-00004-of-00004.safetensors",
        ),
    ),
)

#: Every declared component, shipped or not. Read-only by construction.
COMPONENTS = MappingProxyType({spec.name: spec for spec in _SPECS})

#: The shipped subset (PLAN §2.3). Ordered largest-first so a preflight or a
#: progress display surfaces the expensive downloads first.
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


def total_size_gb(components: tuple[str, ...] | None = None) -> float:
    """Total download size for a component set, defaulting to the shipped subset."""
    names = DEFAULT_COMPONENTS if components is None else components
    return sum(get_component(name).size_gb for name in names)


def download_allow_list(components: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Every repo-relative path to fetch -- and only the live shards.

    PLAN §5.2: the runtime download and the Space README ``preload_from_hub:``
    list are generated from this one function, never hand-maintained in two
    places. A drift test compares the committed README against this output, so
    the duplicate-shard trap (§2.1) cannot silently reappear after an upstream
    re-shard.
    """
    names = DEFAULT_COMPONENTS if components is None else components
    paths: list[str] = list(ROOT_FILES)
    for name in names:
        paths.extend(get_component(name).paths)
    return tuple(paths)


def preload_from_hub_entry(components: tuple[str, ...] | None = None) -> str:
    """The single ``preload_from_hub:`` entry for the Space README.

    One repo, so one entry with comma-separated filenames -- which is what keeps
    us inside the platform's 10-entry cap despite needing far more than 10 files.
    """
    return f"{REPO_ID} {','.join(download_allow_list(components))}"
