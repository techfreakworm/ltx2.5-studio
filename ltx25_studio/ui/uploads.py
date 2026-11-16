"""App-owned copies of uploaded LoRA files, under the LoRA cache's own budget.

Gradio keeps uploads in its cache, and ``build_ui`` sets ``delete_cache`` so
that cache is pruned. A staged adapter must outlive the pruning until Generate,
so an upload is copied first and the copy is what gets validated and staged.

Where and how much: the copy lives in the Hub adapter cache
(``lora.lora_cache_dir()``, ``$HF_HOME/ltx25-lora``) as a
``models--upload--<id>`` folder stamped with the cache's last-used marker. So
uploads and Hub downloads share one least-recently-used budget -- the one
``lora.fetch_lora`` already applies, twice the configured adapter size cap --
and ``lora.evict_lora_cache`` removes the oldest of either kind. A copy that is
rejected, or that duplicates a staged adapter, is deleted at once.

Gradio-free.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

__all__ = ["UPLOAD_PREFIX", "budget_bytes", "discard", "own_copy"]

#: Folder prefix inside the LoRA cache; ``evict_lora_cache`` reads the id after
#: ``models--`` as ``upload/<id>``.
UPLOAD_PREFIX = "models--upload--"


def budget_bytes() -> int:
    """The cache budget ``fetch_lora`` uses: twice the configured size cap."""
    from .. import lora

    return 2 * int(lora.configured_max_bytes())


def own_copy(source: str | os.PathLike) -> Path:
    """Copy an uploaded file into the LoRA cache, then keep the cache in budget."""
    from .. import lora
    from ..lora.fetch import LAST_USED_MARKER

    path = Path(source)
    root = lora.lora_cache_dir()
    key = uuid.uuid4().hex[:16]
    folder = root / f"{UPLOAD_PREFIX}{key}"
    folder.mkdir(parents=True)
    target = folder / path.name
    shutil.copyfile(path, target)
    (folder / LAST_USED_MARKER).touch()
    lora.evict_lora_cache(budget_bytes(), cache_dir=root, keep=(f"upload/{key}",))
    return target


def discard(copy: str | os.PathLike) -> None:
    """Remove a copy made by :func:`own_copy` (rejected, or already staged)."""
    folder = Path(copy).parent
    if folder.name.startswith(UPLOAD_PREFIX):
        shutil.rmtree(folder, ignore_errors=True)
