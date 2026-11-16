"""App-owned copies of uploaded LoRA files.

Gradio keeps uploads in its own cache, and ``build_ui`` sets ``delete_cache``
so that cache is pruned. A staged adapter must outlive that until Generate, so
the upload is copied here first and the copy is what gets validated and
staged. Copies older than :data:`KEEP_SECONDS` are removed on each new copy.

Gradio-free.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

__all__ = ["KEEP_SECONDS", "UPLOADS_ENV", "own_copy", "uploads_dir"]

#: Overrides where uploaded adapters are kept.
UPLOADS_ENV = "LTX25_LORA_UPLOAD_DIR"

#: How long a copied upload is kept: a working day.
KEEP_SECONDS = 24 * 3600


def uploads_dir() -> Path:
    """The folder uploaded adapters are copied into."""
    configured = os.environ.get(UPLOADS_ENV)
    root = Path(configured) if configured else Path(tempfile.gettempdir()) / "ltx25-lora-uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prune(root: Path, *, now: float) -> None:
    for entry in root.iterdir():
        try:
            if entry.is_dir() and now - entry.stat().st_mtime > KEEP_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


def own_copy(source: str | os.PathLike) -> Path:
    """Copy an uploaded file into :func:`uploads_dir`, keeping its file name."""
    path = Path(source)
    root = uploads_dir()
    _prune(root, now=time.time())
    target_dir = root / uuid.uuid4().hex[:12]
    target_dir.mkdir()
    target = target_dir / path.name
    shutil.copyfile(path, target)
    return target
