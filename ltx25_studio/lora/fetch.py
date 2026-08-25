"""Stage 0: resolve a Hub repo id to a local safetensors file, bounded first.

PLAN §4.6's Hub route, previously an advertised direction wired to nothing:
``validate_lora`` carried a ``fetcher`` seam and the UI carried a repo-id
textbox, and no fetcher existed -- so every repo id anyone typed was rejected
with "cannot be read", including the UI's own placeholder.

Three properties are load-bearing here, in order:

1. **The size bound is enforced BEFORE any download.** The Hub's file
   metadata reports every file's size, so an adapter over the configured cap
   is refused from the listing alone -- the §4.6 "stage 1 comes free" half.
   Discovering the bound by downloading past it would spend the bandwidth the
   bound exists to protect.
2. **This runs in the WEB process.** The staged entry ends up holding a local
   path, exactly like the upload direction, so the GPU worker only ever reads
   a file. A Hub download inside the billed ``@spaces.GPU`` window would meter
   wall-clock spent on bandwidth -- tens to hundreds of GPU-seconds for
   nothing.
3. **Downloads land in huggingface_hub's default cache** (under ``HF_HOME``),
   so re-validating the same repo id -- and the later generation that reads
   the staged path -- costs no second download.

What this deliberately does NOT yet do: §4.6's full pre-download validation
(range-read the safetensors header, run stages 2-6.5, only then download).
That remains the documented next step; this module is shaped so it slots in
front of the download without changing any caller -- but a wrongly-built
adapter currently costs one bounded, cached download before its rejection,
rather than one range request.

Import posture matches the rest of ``lora/``: ``huggingface_hub`` is imported
inside the functions that need it, and both the listing and the download are
injected seams so the whole module tests offline.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import LoRAFetchError, LoRATooLargeError
from .validate import DEFAULT_MAX_LORA_BYTES

__all__ = ["DIFFUSERS_LORA_WEIGHT_NAME", "fetch_lora"]

#: The file name diffusers' own loader defaults to when given a bare repo id
#: (``lora_base.py``: ``LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"``).
#: Picking it out of a multi-file repo is therefore not a guess -- it is the
#: one file ``load_lora_weights(repo_id)`` itself would take.
DIFFUSERS_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def _split_source(source: str) -> tuple[str, str | None]:
    """``org/name`` or ``org/name/<path/to/file>`` -> ``(repo_id, filename)``.

    The two accepted shapes are exactly what the UI's helper text promises.
    URLs are refused with the accepted forms named, rather than half-parsed:
    a pasted ``huggingface.co`` URL carries revision segments this module does
    not honour, and silently ignoring them would fetch something other than
    what the user pointed at.
    """
    cleaned = str(source).strip().strip("/")
    if "://" in cleaned:
        raise LoRAFetchError(
            source=source,
            reason=(
                "URLs are not accepted here. Use the repo id, `org/name`, or "
                "`org/name/file.safetensors` to pick a file inside it"
            ),
        )
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) < 2:
        raise LoRAFetchError(
            source=source,
            reason="a Hub repo id has the form `org/name` (optionally `/file.safetensors`)",
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:]) or None
    return repo_id, filename


def _choose_file(source: str, filenames_to_sizes: dict, filename: str | None) -> str:
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
        # See DIFFUSERS_LORA_WEIGHT_NAME: this is the published convention,
        # not a heuristic. Anything else ambiguous is refused below.
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


def _listed(filenames_to_sizes: dict) -> str:
    names = sorted(filenames_to_sizes)
    shown = ", ".join(names[:5]) or "(none)"
    return f"{shown}{' and more' if len(names) > 5 else ''}"


def _default_repo_files(repo_id: str) -> dict:
    """``{filename: size_in_bytes_or_None}`` for a Hub model repo."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    return {s.rfilename: s.size for s in (info.siblings or [])}


def _default_download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def fetch_lora(
    source,
    *,
    max_bytes: int | None = None,
    repo_files=None,
    download=None,
):
    """Resolve ``org/name[/file]`` to a local, size-bounded safetensors path.

    The shape ``validate_lora``'s ``fetcher`` seam expects: one positional
    source in, one local path back, every failure a typed ``LoRAError`` whose
    message the UI surfaces verbatim. ``repo_files`` and ``download`` are the
    injected seams (metadata listing and cached download respectively); the
    defaults are huggingface_hub's real ones.
    """
    if max_bytes is None:
        try:
            from ..config import load_config

            max_bytes = load_config().lora_max_bytes
        except Exception:
            max_bytes = DEFAULT_MAX_LORA_BYTES  # the validator's own default

    repo_id, filename = _split_source(source)

    list_files = repo_files or _default_repo_files
    try:
        available = dict(list_files(repo_id))
    except LoRAFetchError:
        raise
    except Exception as exc:
        raise _mapped_hub_error(source, repo_id, exc) from exc

    chosen = _choose_file(source, available, filename)

    size = available.get(chosen)
    if size is None:
        # "Could not check" must not quietly become "checked and fine": a
        # listing without a size gives the bound nothing to bind, and the
        # bound is the whole point of looking before downloading.
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

    fetch = download or _default_download
    try:
        return Path(fetch(repo_id, chosen))
    except Exception as exc:
        raise LoRAFetchError(
            source=source,
            reason=f"the download failed ({type(exc).__name__}: {exc})",
        ) from exc


def _mapped_hub_error(source, repo_id: str, exc: Exception) -> LoRAFetchError:
    """Typed, actionable messages for the Hub's own failure shapes.

    Checked by name-in-MRO rather than by import so the mapping cannot crash
    when huggingface_hub is absent (the offline gate) or renames a module.
    ``GatedRepoError`` is checked FIRST -- it subclasses
    ``RepositoryNotFoundError``, and the gated message is the one that names
    an action the user can actually take.
    """
    mro_names = {cls.__name__ for cls in type(exc).__mro__}
    if "GatedRepoError" in mro_names:
        return LoRAFetchError(
            source=source,
            reason=(
                f"{repo_id!r} is gated. Accept its terms on the Hub with this Space's "
                f"token's account, or use an ungated adapter"
            ),
        )
    if "RepositoryNotFoundError" in mro_names:
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
