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

   Note we deliberately do NOT move ``HF_HOME`` to dodge the permission problem.
   ``preload_from_hub`` writes to the default cache and does not follow a moved
   ``HF_HOME``, so relocating it makes the app look somewhere empty and
   re-download 72 GB (PLAN §6.2).

3. **Resource preflight.** Measure free RAM and free disk against the
   requirement computed from the resolved config, and raise an error that NAMES
   THE NUMBERS if short -- "need ~67.2 GiB for the bf16 load, saw 41.3 GiB".
   That turns an anonymous OOM, or a "Workload evicted, storage limit exceeded
   (150G)", into a message that names the fix (PLAN §2.6's quantised fallbacks).

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
    "check_diffusers_version",
    "missing_cached_files",
    "require_cache_complete",
]

#: The pinned diffusers SHA reports this. A release (0.39.0) would import
#: ``LTX2Pipeline`` successfully and fail much later and more expensively.
EXPECTED_DIFFUSERS_VERSION = "0.40.0.dev0"


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


def missing_cached_files(
    hub_cache: str | os.PathLike[str] | None = None,
    revision: str | None = None,
) -> tuple[str, ...]:
    """Allow-list paths that are not present in the cache snapshot."""
    snapshot = cache_repo_dir(hub_cache) / "snapshots" / (revision or model_files.REVISION)
    return tuple(
        path for path in model_files.download_allow_list() if not (snapshot / path).exists()
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
    first = missing[0]
    url = (
        f"{model_files.HUB_BASE_URL}/{model_files.REPO_ID}/blob/"
        f"{revision or model_files.REVISION}/{first}"
    )
    raise MissingModelFileError(
        filename=f"{first} ({len(missing)} of {len(model_files.download_allow_list())} missing)",
        url=url,
    )
