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
from pathlib import Path

from . import model_files
from .errors import LTXStudioError, MissingModelFileError

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


def ensure_weights(
    *,
    token=None,
    components: tuple[str, ...] | None = None,
    hub_cache=None,
    progress=None,
) -> tuple[Path, ...]:
    """Download every allow-listed file **at runtime, as the runtime user**.

    This replaces ``preload_from_hub``, which cannot work for a **private**
    Space: the preload runs as a separate platform-managed init step that does
    not see Space secrets, so a gated repo returns ``401 Unauthenticated`` even
    with a valid token configured. (401, not 403 -- the request arrives with no
    credentials at all, which is why the same URL fetches fine by hand.)

    Downloading here costs a cold start per restart, and buys two things:

    * **The token is available.** This process has the Space secrets.
    * **It removes the permission bug class entirely** (PLAN §6.2). Preloaded
      files are owned by the *build* user and read-only to runtime uid 1000, so
      any later lazy fetch fails with ``Permission denied (os error 13)`` -- the
      2.3 failure. When the runtime user downloads its own cache, it owns it.

    Fails loudly per file, naming the file and its URL, on the same contract as
    :func:`require_cache_complete`. A partial download must not become a
    confusing error three layers later.
    """
    import logging

    log = logging.getLogger("ltx25_studio")
    override = snapshot_override()
    if override is not None:
        # Nothing loads the pinned repo under the override (review space-path-3):
        # fetching it would download 72 GB for nothing, or stop start-up on a 401.
        log.info("LTX25_MODEL_SNAPSHOT_DIR is set: weights load from %s; nothing fetched", override)
        return ()

    from huggingface_hub import hf_hub_download

    # Each path at the revision its component is pinned to (model_files.REVISION
    # for all but the diffusion decoder): download_plan groups them.
    plan = [
        (revision, path)
        for revision, group in model_files.download_plan(components)
        for path in group
    ]
    paths = model_files.download_allow_list(components)
    total_gb = model_files.total_size_gb(components)
    log.info(
        "fetching %d files (%.2f GB) at revisions %s",
        len(paths),
        total_gb,
        ", ".join(model_files.revisions(components)),
    )
    fetched: list[Path] = []
    for index, (revision, path) in enumerate(plan, 1):
        # Progress is not decoration. This is a 72 GB cold start on every restart,
        # and a log that goes quiet for forty minutes is indistinguishable from a
        # log that has stopped.
        log.info("  [%2d/%d] %s @ %s", index, len(plan), path, revision[:8])
        if progress is not None:
            progress(index / len(plan), f"fetching {path}")
        try:
            fetched.append(
                Path(
                    hf_hub_download(
                        repo_id=model_files.REPO_ID,
                        filename=path,
                        revision=revision,
                        token=token,
                        cache_dir=str(hub_cache) if hub_cache else None,
                    )
                )
            )
        except Exception as exc:
            raise MissingModelFileError(
                filename=f"{path} ({index} of {len(plan)}; {exc.__class__.__name__})",
                url=f"{model_files.HUB_BASE_URL}/{model_files.REPO_ID}/blob/{revision}/{path}",
            ) from exc
    log.info("fetched %d/%d files", len(fetched), len(paths))
    return tuple(fetched)


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
