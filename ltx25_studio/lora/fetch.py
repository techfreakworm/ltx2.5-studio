"""Stage 0: resolve a Hub repo id to a local safetensors file, bounded first.

Runs in the WEB process, so the GPU worker only ever reads a local file and a
Hub download can never land inside the billed ``@spaces.GPU`` window.

What it guarantees:

1. **Only repo ids get here.** ``org/name`` or ``org/name/path/to/file`` --
   checked as text before any filesystem or network call, so a server path
   typed into the repo-id box is refused rather than read.
2. **The size bound is enforced from the listing, before any download**, and
   the download is pinned to the commit that listing saw (``revision=sha``),
   so the file that was measured is the file that arrives.
3. **Gated repos say so.** They list fine and fail at the download (HTTP 401,
   ``GatedRepoError``); both steps map that to the same actionable message.
4. **The cache is bounded.** Adapters download into their own directory
   (:func:`lora_cache_dir`, beside the Hub cache, never inside another model's
   folder), each fetch stamps its repo folder, and :func:`evict_lora_cache`
   removes the least recently used ones over a byte budget. Only folders this
   module stamped are ever counted or removed.

The listing also carries the repo's card tags and ``base_model`` references,
which ``validate_lora`` uses to recognise IC-LoRAs whose file metadata is empty.

Pre-download header validation (range-read, then validate, then download) is
not done here yet: a wrongly-built adapter costs one bounded, cached download.

``huggingface_hub`` is imported inside the functions; the listing and the
download are injected seams so the module tests offline.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..errors import LoRAFetchError, LoRATooLargeError
from .validate import configured_max_bytes

logger = logging.getLogger(__name__)

__all__ = [
    "DIFFUSERS_LORA_WEIGHT_NAME",
    "LAST_USED_MARKER",
    "FetchedLoRA",
    "RepoListing",
    "evict_lora_cache",
    "fetch_lora",
    "listing_from_model_info",
    "lora_cache_dir",
    "parse_hub_reference",
]

#: The file name diffusers' own loader defaults to when given a bare repo id
#: (``lora_base.py``: ``LORA_WEIGHT_NAME_SAFE``), so picking it out of a
#: multi-file repo means the same file to us as to ``load_lora_weights``.
DIFFUSERS_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

#: Directory name, under ``HF_HOME``, that adapters are downloaded into.
LORA_CACHE_DIRNAME = "ltx25-lora"

#: Stamped into each repo folder this module fetched into; its mtime is the
#: last use. Folders without it are never counted or evicted.
LAST_USED_MARKER = ".ltx25-last-used"

_ACCEPTED_FORMS = (
    "a Hub repo id has the form `org/name`, or `org/name/path/to/file.safetensors` to pick "
    "a file inside the repo"
)
#: Hub owner and repo names: letters, digits, ``-``, ``_``, ``.``; not starting
#: or ending with ``-`` or ``.``.
_HUB_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$")


@dataclass(frozen=True)
class RepoListing:
    """What one listing of a repo said: file sizes, the commit, and card signals."""

    files: Mapping
    sha: str | None = None
    tags: tuple[str, ...] = ()
    base_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class FetchedLoRA:
    """A downloaded adapter: usable as a path, and carrying where it came from."""

    path: Path
    repo_id: str
    filename: str
    revision: str | None = None
    tags: tuple[str, ...] = ()
    base_models: tuple[str, ...] = ()

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


def parse_hub_reference(source) -> tuple[str, str | None]:
    """``org/name[/path/to/file]`` -> ``(repo_id, filename)``, or a refusal.

    Text only: nothing here touches the filesystem, so a local path typed into
    the repo-id box is refused without being looked at.
    """
    text = str(source).strip()
    if "://" in text:
        raise LoRAFetchError(source=source, reason=f"URLs are not accepted here; {_ACCEPTED_FORMS}")
    refusal = LoRAFetchError(
        source=source,
        reason=(
            f"{_ACCEPTED_FORMS}. To use a file from your computer, upload it instead of "
            f"typing its path"
        ),
    )
    if not text or text.startswith(("/", "~", ".")) or "\\" in text or "\x00" in text:
        raise refusal
    parts = text.rstrip("/").split("/")
    if len(parts) < 2:
        raise refusal
    org, name, *rest = parts
    for part in (org, name):
        if not _HUB_NAME.match(part) or "--" in part or ".." in part:
            raise refusal
    for part in rest:
        if part in ("", ".", "..") or any(ord(ch) < 32 for ch in part):
            raise refusal
    return f"{org}/{name}", "/".join(rest) or None


def _choose_file(source: str, filenames_to_sizes: Mapping, filename: str | None) -> str:
    """Pick the one file to fetch, or refuse naming what is actually there."""
    if filename is not None:
        if filename in filenames_to_sizes:
            if not filename.endswith(".safetensors"):
                raise LoRAFetchError(
                    source=source,
                    reason=f"{filename!r} is not a .safetensors file; only safetensors "
                    f"adapters are accepted",
                )
            return filename
        raise LoRAFetchError(
            source=source,
            reason=(
                f"the repo has no file named {filename!r}. Files present: "
                f"{_listed(filenames_to_sizes)}"
            ),
        )

    candidates = sorted(n for n in filenames_to_sizes if n.endswith(".safetensors"))
    if not candidates:
        raise LoRAFetchError(
            source=source,
            reason=(
                f"the repo contains no .safetensors file at all. Files present: "
                f"{_listed(filenames_to_sizes)}"
            ),
        )
    if len(candidates) == 1:
        return candidates[0]
    if DIFFUSERS_LORA_WEIGHT_NAME in candidates:
        return DIFFUSERS_LORA_WEIGHT_NAME
    raise LoRAFetchError(
        source=source,
        reason=(
            f"the repo holds {len(candidates)} .safetensors files and none is the "
            f"diffusers-convention {DIFFUSERS_LORA_WEIGHT_NAME!r}, so choosing one would "
            f"be a guess. Name it explicitly as `{source}/<file>`. Candidates: "
            f"{', '.join(candidates[:5])}{' and more' if len(candidates) > 5 else ''}"
        ),
    )


def _listed(filenames_to_sizes: Mapping) -> str:
    names = sorted(filenames_to_sizes)
    shown = ", ".join(names[:5]) or "(none)"
    return f"{shown}{' and more' if len(names) > 5 else ''}"


def listing_from_model_info(info) -> RepoListing:
    """A :class:`RepoListing` from ``HfApi().model_info(..., files_metadata=True)``."""
    tags = tuple(str(tag) for tag in (getattr(info, "tags", None) or ()))
    base_models: list[str] = []
    card = getattr(info, "card_data", None)
    declared = getattr(card, "base_model", None) if card is not None else None
    if isinstance(declared, str):
        base_models.append(declared)
    elif declared:
        base_models.extend(str(model) for model in declared)
    for tag in tags:
        if tag.startswith("base_model:"):
            base_models.append(tag.rsplit(":", 1)[-1])
    return RepoListing(
        files={s.rfilename: s.size for s in (getattr(info, "siblings", None) or [])},
        sha=getattr(info, "sha", None),
        tags=tags,
        base_models=tuple(dict.fromkeys(base_models)),
    )


def _default_repo_files(repo_id: str) -> RepoListing:
    from huggingface_hub import HfApi

    return listing_from_model_info(HfApi().model_info(repo_id, files_metadata=True))


def _default_download(repo_id: str, filename: str, revision=None, *, cache_dir=None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def lora_cache_dir() -> Path:
    """Where Hub adapters are downloaded: ``$HF_HOME/ltx25-lora``."""
    try:
        from huggingface_hub import constants

        home = Path(constants.HF_HOME)
    except Exception:  # noqa: BLE001 -- the same default huggingface_hub uses
        home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return home / LORA_CACHE_DIRNAME


def _repo_folder(root: Path, repo_id: str) -> Path:
    return root / ("models--" + repo_id.replace("/", "--"))


def _as_listing(listed) -> RepoListing:
    if isinstance(listed, RepoListing):
        return listed
    return RepoListing(files=dict(listed))


def fetch_lora(
    source,
    *,
    max_bytes: int | None = None,
    repo_files=None,
    download=None,
    cache_dir=None,
    cache_max_bytes: int | None = None,
) -> FetchedLoRA:
    """Resolve ``org/name[/file]`` to a local, size-bounded safetensors file.

    ``max_bytes`` defaults to the configured ``LTX25_LORA_MAX_GB``.
    ``repo_files(repo_id)`` returns a :class:`RepoListing` (or a plain
    ``{filename: size}``); ``download(repo_id, filename, revision=...)`` returns
    a local path. Both default to huggingface_hub. ``cache_dir`` and
    ``cache_max_bytes`` (default twice ``max_bytes``) bound the adapter cache;
    eviction runs on the real download path or when ``cache_dir`` is given.
    """
    if max_bytes is None:
        max_bytes = configured_max_bytes()

    repo_id, filename = parse_hub_reference(source)

    list_files = repo_files or _default_repo_files
    try:
        listing = _as_listing(list_files(repo_id))
    except LoRAFetchError:
        raise
    except Exception as exc:
        raise _mapped_hub_error(source, repo_id, exc) from exc

    available = dict(listing.files)
    chosen = _choose_file(source, available, filename)

    size = available.get(chosen)
    if size is None:
        raise LoRAFetchError(
            source=source,
            reason=(
                f"the Hub reported no size for {chosen!r}, so the {max_bytes}-byte cap "
                f"cannot be enforced before downloading. Refusing to download unbounded"
            ),
        )
    if size > max_bytes:
        raise LoRATooLargeError(
            source=f"{repo_id}/{chosen}", size_bytes=int(size), max_bytes=int(max_bytes)
        )

    root = Path(cache_dir) if cache_dir is not None else None
    if download is None:
        root = root or lora_cache_dir()

        def fetch(repo, name, revision=None):
            return _default_download(repo, name, revision, cache_dir=root)
    else:
        fetch = download
    try:
        local = Path(fetch(repo_id, chosen, revision=listing.sha))
    except Exception as exc:
        raise _mapped_download_error(source, repo_id, exc) from exc

    if root is not None:
        _mark_used(root, repo_id)
        budget = cache_max_bytes if cache_max_bytes is not None else 2 * int(max_bytes)
        try:
            evict_lora_cache(budget, cache_dir=root, keep=(repo_id,))
        except OSError as exc:
            logger.warning("LoRA cache eviction under %s failed: %r", root, exc)

    return FetchedLoRA(
        path=local,
        repo_id=repo_id,
        filename=chosen,
        revision=listing.sha,
        tags=tuple(listing.tags),
        base_models=tuple(listing.base_models),
    )


def _mark_used(root: Path, repo_id: str) -> None:
    folder = _repo_folder(root, repo_id)
    if folder.is_dir():
        (folder / LAST_USED_MARKER).touch()


def _tree_bytes(folder: Path) -> int:
    """Bytes of regular files under ``folder``; symlinks (snapshots) are not followed."""
    total = 0
    for directory, _dirs, files in os.walk(folder, followlinks=False):
        for name in files:
            path = os.path.join(directory, name)
            if not os.path.islink(path):
                total += os.lstat(path).st_size
    return total


def evict_lora_cache(max_bytes: int, *, cache_dir=None, keep=()) -> tuple[str, ...]:
    """Remove least-recently-used adapters until the cache is within ``max_bytes``.

    Only ``models--*`` folders carrying :data:`LAST_USED_MARKER` count; nothing
    else under ``cache_dir`` is touched. Repo ids in ``keep`` are never removed.
    Returns the evicted repo ids, oldest first.
    """
    root = Path(cache_dir) if cache_dir is not None else lora_cache_dir()
    if not root.is_dir():
        return ()
    kept = set(keep)
    entries = []
    for folder in root.iterdir():
        marker = folder / LAST_USED_MARKER
        if not (folder.is_dir() and folder.name.startswith("models--") and marker.is_file()):
            continue
        repo_id = folder.name[len("models--") :].replace("--", "/", 1)
        entries.append((marker.stat().st_mtime, repo_id, folder, _tree_bytes(folder)))
    total = sum(size for *_, size in entries)
    evicted = []
    for _stamp, repo_id, folder, size in sorted(entries):
        if total <= max_bytes:
            break
        if repo_id in kept:
            continue
        shutil.rmtree(folder)
        shutil.rmtree(root / ".locks" / folder.name, ignore_errors=True)
        total -= size
        evicted.append(repo_id)
        logger.info("evicted LoRA %s from %s (%d bytes)", repo_id, root, size)
    return tuple(evicted)


def _gated(source, repo_id: str) -> LoRAFetchError:
    return LoRAFetchError(
        source=source,
        reason=(
            f"{repo_id!r} is gated. Accept its terms on the Hub with this Space's "
            f"token's account, or use an ungated adapter"
        ),
    )


def _is_gated(exc: Exception) -> bool:
    if "GatedRepoError" in {cls.__name__ for cls in type(exc).__mro__}:
        return True
    return getattr(getattr(exc, "response", None), "status_code", None) == 401


def _mapped_download_error(source, repo_id: str, exc: Exception) -> LoRAFetchError:
    """Gated repos list fine and fail here, so the download step maps them too."""
    if _is_gated(exc):
        return _gated(source, repo_id)
    return LoRAFetchError(
        source=source, reason=f"the download failed ({type(exc).__name__}: {exc})"
    )


def _mapped_hub_error(source, repo_id: str, exc: Exception) -> LoRAFetchError:
    """Typed, actionable messages for the listing's failure shapes.

    Matched by class name in the MRO, so the mapping cannot crash when
    huggingface_hub is absent or renames a module. ``GatedRepoError`` subclasses
    ``RepositoryNotFoundError`` and is checked first.
    """
    if _is_gated(exc):
        return _gated(source, repo_id)
    if "RepositoryNotFoundError" in {cls.__name__ for cls in type(exc).__mro__}:
        return LoRAFetchError(
            source=source,
            reason=(
                f"{repo_id!r} was not found on the Hub. The Hub reports private repos "
                f"the same way, so: check the spelling, or check the repo's visibility "
                f"against this Space's token"
            ),
        )
    return LoRAFetchError(
        source=source,
        reason=f"listing the repo failed ({type(exc).__name__}: {exc})",
    )
