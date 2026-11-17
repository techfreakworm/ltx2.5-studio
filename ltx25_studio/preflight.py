"""Startup assertions that fail loudly before anything expensive happens.

PLAN §1.3, §5.6, §6.2.1. Three jobs, all cheap, all run before a single weight
byte is touched:

1. **Version fail-fast.** ``diffusers.__version__`` must be ``"0.40.0.dev0"``
   (the pinned SHA). Release 0.39.0 imports ``LTX2Pipeline`` fine and only fails
   LATE, on a missing ``temporal_latent_upsampler``, after ~70 GB of download.

2. **Cache completeness.** Every path in ``model_files.download_allow_list()``
   must already be in the Hub cache. This is the real fix for the read-only
   preload problem: preloaded files belong to the *build* user and are read-only
   to runtime uid 1000, so a lazy ``hf_hub_download`` fails with
   ``Permission denied (os error 13)``. If nothing ever needs fetching lazily,
   that failure cannot happen -- so we check the invariant instead of hoping for
   it, and a miss names the file AND its URL.

   Note we deliberately do NOT move ``HF_HOME``, and there are two reasons --
   the second sharper than the first. ``preload_from_hub`` writes to the default
   cache and does not follow a moved ``HF_HOME``, so relocating it makes the app
   look somewhere empty and re-download 72 GB. **And ZeroGPU's post-pack cleanup
   unlinks blobs matching** ``~/.cache/huggingface/hub/models--*/blobs/*``
   **specifically** -- move the cache out from under that pattern and the 72 GB
   snapshot is never freed, silently (PLAN §6.2, §8.3).

3. **Resource report.** The startup report (``pipelines.startup_report``) logs
   the disk floor; there is no RAM check here (local runs preflight memory in
   ``memory.py``).

Stdlib only. A preflight that needs a heavy import to tell you the environment
is wrong has already lost.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import model_files
from .errors import LTXStudioError, MissingModelFileError, ModelFileError

__all__ = [
    "EXPECTED_DIFFUSERS_VERSION",
    "cache_repo_dir",
    "component_dir",
    "snapshot_dir",
    "snapshot_override",
    "check_diffusers_version",
    "missing_cached_files",
    "ensure_weights",
    "local_missing_files",
    "require_cache_complete",
    "run",
]

#: The pinned diffusers SHA reports this. A release (0.39.0) would import
#: ``LTX2Pipeline`` successfully and fail much later and more expensively.
EXPECTED_DIFFUSERS_VERSION = "0.40.0.dev0"

#: ZeroGPU's packed copy, written at launch after the weights are in: 40.6 GB
#: measured on the 2026-11-13 boot (41.24 derived), plus the 0.83 GB decoder.
ZEROGPU_PACKED_COPY_GB = 42.1

#: Room a fallback download must leave on the disk.
DOWNLOAD_MARGIN_GB = 10.0


def ensure_weights(
    *,
    token=None,
    components: tuple[str, ...] | None = None,
    hub_cache=None,
    progress=None,
    on_zerogpu: bool = False,
) -> tuple[Path, ...]:
    """Check every allow-listed file is in the Hub cache; download only what is not.

    On the Space the files come from the image: ``preload_from_hub`` in the
    README front matter fetches this same set
    (``model_files.preload_from_hub_entries()``) before the app starts. With
    every file present nothing is downloaded and no network call is made; one
    line says the weights came from the image.

    Any file the cache lacks -- a preload that failed, stopped short or names a
    different set -- is downloaded here, at its pinned revision, with the
    Space's token, after one warning that says how many files and how many GB.
    A broken preload makes the start slower, as in the first deploy, not broken
    -- unless the disk cannot hold the download (:func:`_require_room`): then
    start-up stops before a byte is written, saying why and what to do.
    On the Space the runtime user is root (``/root/.cache`` in its logs), so it
    can add files to a cache the preload wrote.

    Fails loudly per file, naming the file and its URL, on the same contract as
    :func:`require_cache_complete`. A partial download must not become a
    confusing error three layers later. Returns the local path of every file.
    """
    import logging

    log = logging.getLogger("ltx25_studio")
    override = snapshot_override()
    if override is not None:
        # Nothing loads the pinned repo under the override (review space-path-3):
        # fetching it would download 72 GB for nothing, or stop start-up on a 401.
        log.info("LTX25_MODEL_SNAPSHOT_DIR is set: weights load from %s; nothing fetched", override)
        return ()

    # Each path at the revision its component is pinned to (model_files.REVISION
    # for all but the diffusion decoder): download_plan groups them.
    plan = [
        (revision, path)
        for revision, group in model_files.download_plan(components)
        for path in group
    ]
    repo_cache = cache_repo_dir(hub_cache)
    local = [model_files.local_path(path, repo_cache=repo_cache) for _revision, path in plan]
    missing = [
        (rev, path) for (rev, path), there in zip(plan, local, strict=True) if not there.exists()
    ]
    total_gb = model_files.total_size_gb(components)
    if not missing:
        log.info(
            "weights: all %d files (%.2f GB) are already in %s -- from the image "
            "(preload_from_hub); nothing downloaded",
            len(plan),
            total_gb,
            repo_cache,
        )
        return tuple(local)

    missing_bytes = sum(model_files.file_size(path, components) for _rev, path in missing)
    _require_room(missing_bytes, len(missing), repo_cache, on_zerogpu=on_zerogpu)
    missing_gb = missing_bytes / 1e9
    log.warning(
        "weights: %d of %d files (%.2f GB) are not in %s, which the image's preload should "
        "have filled; downloading them now, so this start is slower. Missing: %s",
        len(missing),
        len(plan),
        missing_gb,
        repo_cache,
        ", ".join(path for _rev, path in missing[:5]) + (" ..." if len(missing) > 5 else ""),
    )
    from huggingface_hub import hf_hub_download

    for index, (revision, path) in enumerate(missing, 1):
        # Progress is not decoration. A full set is a 72 GB download, and a log
        # that goes quiet for fifteen minutes is indistinguishable from a log
        # that has stopped.
        log.info("  [%2d/%d] %s @ %s", index, len(missing), path, revision[:8])
        if progress is not None:
            progress(index / len(missing), f"fetching {path}")
        try:
            hf_hub_download(
                repo_id=model_files.REPO_ID,
                filename=path,
                revision=revision,
                token=token,
                cache_dir=str(hub_cache) if hub_cache else None,
            )
        except Exception as exc:
            raise MissingModelFileError(
                filename=f"{path} ({index} of {len(missing)} missing; {exc.__class__.__name__})",
                url=f"{model_files.HUB_BASE_URL}/{model_files.REPO_ID}/blob/{revision}/{path}",
            ) from exc
    log.info("fetched the %d missing files; all %d are in the cache", len(missing), len(plan))
    return tuple(local)


def _require_room(missing_bytes: int, count: int, repo_cache: Path, *, on_zerogpu: bool) -> None:
    """Refuse a fallback download the disk cannot hold, before any of it is written.

    A download that fills the disk part-way is the worst failure a public Space
    can have; a refusal that names the cause and the remedy is better. Needed:
    the missing bytes, the margin and, on ZeroGPU, the packed copy still to be
    written. Free: the filesystem's free space and, on ZeroGPU, the room left
    under its cap after the image and everything already in the Hub cache --
    there the filesystem is the host's (terabytes), so it alone never refuses.
    """
    probe = Path(repo_cache)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    where = "free"
    need = missing_bytes + DOWNLOAD_MARGIN_GB * 1e9
    extras = f"a {DOWNLOAD_MARGIN_GB:.0f} GB margin"
    if on_zerogpu:
        from . import pipelines

        need += ZEROGPU_PACKED_COPY_GB * 1e9
        extras = f"ZeroGPU's {ZEROGPU_PACKED_COPY_GB} GB packed copy and {extras}"
        cap = pipelines.ZEROGPU_DISK_CAP_GB
        room = (cap - pipelines.IMAGE_FOOTPRINT_GB) * 1e9 - pipelines._directory_bytes(
            Path(repo_cache).parent
        )
        if room < free:
            free, where = room, f"free under ZeroGPU's {cap:.0f} GB cap"
    if free >= need:
        return
    raise ModelFileError(
        f"Not downloading the {count} weight files missing from {repo_cache} "
        f"({missing_bytes / 1e9:.2f} GB): there are {max(free, 0) / 1e9:.1f} GB {where}, and "
        f"the download needs {need / 1e9:.1f} GB with {extras}, so it would fill the disk "
        f"part-way. On the Space, remove the preload_from_hub block from README.md so "
        f"start-up downloads every file into an empty disk, as the first deploy did; on "
        f"another machine, free space there."
    )


def run(runtime_config, *, strict: bool = False, spaces_module=None, resolution=None) -> str:
    """Log the startup report, and hard-check the environment when about to generate.

    ``strict`` gates the checks that only make sense with real weights present:
    the pinned-revision assertion and cache completeness. A UI-only machine gets
    the report alone; a machine that will generate gets a named failure now
    instead of one 70 GB into a download.

    ``resolution`` is the ``device.DeviceResolution`` app.py resolved; the MPS
    self-test result is reported when that device is ``mps``. Before it, the
    MPS allocator's watermark pair: under a pair PyTorch refuses every GPU
    operation fails, the self-test's too, so the self-test is not run and the
    pair is named instead (``strict``: refused with ``memory.MpsWatermarkError``).
    """
    import logging

    from . import pipelines, zerogpu

    log = logging.getLogger("ltx25_studio")
    local = _local_runner_device(resolution)
    if local:
        absent = local_missing_files(runtime_config)
        missing = tuple(str(path) for path, _command in absent)
    else:
        missing = missing_cached_files()
    self_test = watermarks = None
    if resolution is not None and resolution.device == "mps":
        from . import device, memory

        watermarks = memory.mps_watermarks()
        if watermarks.problem is None:
            self_test = device.mps_self_test()
        elif not strict:
            log.warning("MPS self-test not run: %s", watermarks.problem)
    log.info(
        "%s",
        pipelines.startup_report(
            runtime_config,
            missing_files=missing,
            resolution=resolution,
            spaces_version=zerogpu.spaces_version(spaces_module),
            mps_self_test=self_test,
        ),
    )
    if not strict:
        return "reported"

    if watermarks is not None:
        watermarks.require()
    import diffusers

    check_diffusers_version(diffusers.__version__)
    if local:
        if absent:
            from .local_runtime import LocalFilesMissingError

            raise LocalFilesMissingError(missing=absent)
    else:
        require_cache_complete()
    return "verified"


def _local_runner_device(resolution) -> bool:
    """True where generation goes to the local runner (mps/cpu, real weights)."""
    from .pipelines import _uses_local_runner

    device = getattr(resolution, "device", None)
    return device in ("mps", "cpu") and _uses_local_runner()


def local_missing_files(runtime_config) -> list:
    """What the local runner loads and is not on disk: ``[(path, download command)]``.

    The small set, the text encoder, and the safetensors transformer or the
    configured GGUF -- not the Space's whole allow-list: stage 2 is refused at
    real sizes off CUDA, and a GGUF replaces the transformer shards.
    """
    from . import local_runtime

    return local_runtime.missing_files(
        snapshot_dir(), transformer_gguf=getattr(runtime_config, "transformer_gguf", None)
    )


def check_diffusers_version(found: str) -> None:
    """Raise unless the installed diffusers is the pinned dev build."""
    if found != EXPECTED_DIFFUSERS_VERSION:
        raise LTXStudioError(
            f"diffusers {found!r} is installed, but LTX-2.5 needs the pinned revision, which "
            f"reports {EXPECTED_DIFFUSERS_VERSION!r}. A release build imports LTX2Pipeline "
            f"successfully and then fails on a missing temporal_latent_upsampler, after the "
            f"download. Install the pin from requirements.txt."
        )


def cache_repo_dir(hub_cache: str | os.PathLike[str] | None = None) -> Path:
    """The ``huggingface_hub`` cache directory for the model repo.

    Reads the default cache location. We never relocate ``HF_HOME``; see the
    module docstring.
    """
    if hub_cache is not None:
        root = Path(hub_cache)
    else:
        root = Path(
            os.environ.get("HF_HUB_CACHE")
            or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        )
    return root / f"models--{model_files.REPO_ID.replace('/', '--')}"


def component_dir(name: str, hub_cache: str | os.PathLike[str] | None = None) -> Path:
    """The directory component ``name`` loads from: ``model_files.component_dir`` here.

    ``LTX25_MODEL_SNAPSHOT_DIR`` holds every component when set (and no explicit
    cache is asked for); otherwise each component's own revision's snapshot.
    """
    override = snapshot_override() if hub_cache is None else None
    return model_files.component_dir(name, repo_cache=cache_repo_dir(hub_cache), override=override)


def snapshot_override() -> Path | None:
    """``LTX25_MODEL_SNAPSHOT_DIR``, validated, or None when unset."""
    from .config import load_config

    override = load_config().model_snapshot_dir
    return Path(override) if override else None


def snapshot_dir(
    hub_cache: str | os.PathLike[str] | None = None,
    revision: str | None = None,
) -> Path:
    """The local snapshot directory for the pinned revision.

    ``LTX25_MODEL_SNAPSHOT_DIR`` replaces it when set and no explicit cache or
    revision is asked for (tests and local work on another checkpoint).

    **Construct every pipeline from this path, never from the repo id.** Passing
    the repo id sends ``from_pretrained`` through hub resolution, which fetches
    the repo's tree manifest and then downloads *everything the manifest lists*
    -- 48 files, including the dead 8-shard ``transformer/`` set and the dead
    single-file ``connectors/`` copy: **44.33 GB** that ``model_files`` already
    documents as paid "for nothing". A complete 37/37 cache does not prevent it.

    ``local_files_only=True`` is NOT the fix. It switches on a strict
    completeness check against that same manifest, which demands the 14 dead
    files the allow-list excludes **on purpose** -- so it turns a working load
    into a hard refusal. A snapshot directory has no manifest and no repo to
    resolve against, so the refetch becomes structurally impossible rather than
    merely disabled.

    Nor may we rely on what happens to work today: with no cached
    ``trees/<revision>.json`` diffusers falls back to *"Will try to load from
    local cache"* and succeeds. The first run that reaches the Hub writes that
    manifest, and every load afterwards behaves differently. **Depending on the
    absence of a cache metadata file is not a design.**
    """
    if hub_cache is None and revision is None:
        override = snapshot_override()
        if override is not None:
            return override
    return cache_repo_dir(hub_cache) / "snapshots" / (revision or model_files.REVISION)


def _declared_components(snapshot: Path) -> tuple[str, ...]:
    """Components an override snapshot's ``model_index.json`` says it has."""
    import json

    index = json.loads((snapshot / "model_index.json").read_text())
    return tuple(
        name
        for name, value in index.items()
        if not name.startswith("_")
        and isinstance(value, list)
        and len(value) == 2
        and value[1] is not None
    )


def missing_cached_files(
    hub_cache: str | os.PathLike[str] | None = None,
    revision: str | None = None,
) -> tuple[str, ...]:
    """Allow-list paths that are not present in the cache snapshot.

    With ``LTX25_MODEL_SNAPSHOT_DIR`` set, the components that directory's own
    ``model_index.json`` declares and that are absent or empty instead.
    """
    snapshot = snapshot_dir(hub_cache, revision)
    if hub_cache is None and revision is None and snapshot_override() is not None:
        return tuple(
            name
            for name in _declared_components(snapshot)
            if not (snapshot / name).is_dir() or not any((snapshot / name).iterdir())
        )
    if revision is not None:
        return tuple(
            path for path in model_files.download_allow_list() if not (snapshot / path).exists()
        )
    repo_cache = cache_repo_dir(hub_cache)
    return tuple(
        path
        for path in model_files.download_allow_list()
        if not model_files.local_path(path, repo_cache=repo_cache).exists()
    )


def require_cache_complete(
    hub_cache: str | os.PathLike[str] | None = None,
    revision: str | None = None,
) -> None:
    """Raise naming the first missing file AND its URL.

    Better to stop here than to let a lazy download attempt hit a read-only
    preload cache and surface as ``Permission denied (os error 13)`` halfway
    through a generation.
    """
    missing = missing_cached_files(hub_cache, revision)
    if not missing:
        return
    override = snapshot_override() if hub_cache is None and revision is None else None
    if override is not None:
        raise MissingModelFileError(
            filename=f"{missing[0]}/ ({len(missing)} declared component(s) missing)",
            url=str(override / missing[0]),
        )
    first = missing[0]
    url = (
        f"{model_files.HUB_BASE_URL}/{model_files.REPO_ID}/blob/"
        f"{revision or model_files.revision_of_path(first)}/{first}"
    )
    raise MissingModelFileError(
        filename=f"{first} ({len(missing)} of {len(model_files.download_allow_list())} missing)",
        url=url,
    )
