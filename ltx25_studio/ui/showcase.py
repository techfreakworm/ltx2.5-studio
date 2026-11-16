"""Examples and sample outputs, read from an optional manifest.

The directory is ``LTX25_EXAMPLES_DIR`` (default ``examples/``) and the file is
``manifest.json`` inside it::

    {
      "modes": {
        "<mode id>": {
          "examples": [
            {"label": "Fishing harbour at dawn",
             "prompt": "...",
             "image": "media/harbour.png",          # optional; also "video", "audio"
             "settings": {"shape": "16:9", "seconds": 5.0, "upscale": false,
                          "seed": 42, "enhance": false, "strength": 0.8,
                          "noise": "Moderately", "auto_duration": false}}
             # "width"/"height" give an exact size and win over "shape".
             # "auto" is accepted for "auto_duration"; "noise" names a change
             # choice (Audio -> sound, or Video -> video).
             # An example writes only what it names; nothing else changes.
          ],
          "sample": {"file": "samples/t2av.mp4", "caption": "...",
                     "settings": "704 × 480, 5 s", "seed": 42}
                     # "settings" may instead be {"size": "704 × 480", "seconds": 5,
                     # "upscale": true, "enhance": true, "auto_duration": true}:
                     # the caption then says each in words.
        }
      }
    }

Paths are relative to the directory and must stay inside it. An entry naming
a missing file is skipped with a log line rather than shown broken. With no
manifest at all the page shows a designed idle monitor and no example row.

Gradio-free.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["EXAMPLES_ENV", "Example", "Sample", "Showcase", "load_showcase"]

EXAMPLES_ENV = "LTX25_EXAMPLES_DIR"
DEFAULT_DIR = "examples"
MANIFEST = "manifest.json"

_MEDIA = ("image", "video", "audio")
_SETTINGS = {
    "shape",
    "width",
    "height",
    "seconds",
    "upscale",
    "seed",
    "enhance",
    "strength",
    "noise",
    "auto",
    "auto_duration",
}


@dataclass(frozen=True)
class Example:
    label: str
    prompt: str = ""
    media: dict[str, Path] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    path: Path
    caption: str = ""
    #: Free text, or a mapping of the settings the caption names in words.
    settings: str | dict = ""
    seed: int | None = None

    @property
    def is_video(self) -> bool:
        return self.path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}


@dataclass(frozen=True)
class Showcase:
    root: Path | None = None
    examples: dict[str, tuple[Example, ...]] = field(default_factory=dict)
    samples: dict[str, Sample] = field(default_factory=dict)

    @property
    def has_examples(self) -> bool:
        return any(self.examples.values())

    def for_mode(self, mode_id: str) -> tuple[Example, ...]:
        return self.examples.get(mode_id, ())

    def sample_for(self, mode_id: str) -> Sample | None:
        return self.samples.get(mode_id)


def _inside(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        return None
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        logger.warning("showcase: skipping %r (missing, or outside %s)", relative, root)
        return None
    return candidate


def load_showcase(directory: str | os.PathLike | None = None, *, known_modes=()) -> Showcase:
    """Read the manifest; an absent or unreadable one is an empty showcase."""
    root = Path(directory or os.environ.get(EXAMPLES_ENV) or DEFAULT_DIR)
    manifest = root / MANIFEST
    if not manifest.is_file():
        return Showcase()
    root = root.resolve()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("showcase: %s could not be read (%s); showing no examples", manifest, exc)
        return Showcase()

    examples: dict[str, tuple[Example, ...]] = {}
    samples: dict[str, Sample] = {}
    for mode_id, entry in (data.get("modes") or {}).items():
        if known_modes and mode_id not in known_modes:
            logger.warning("showcase: unknown mode %r in %s", mode_id, manifest)
            continue
        if not isinstance(entry, dict):
            continue
        found = []
        for raw in entry.get("examples") or ():
            if not isinstance(raw, dict) or not raw.get("label"):
                continue
            media = {}
            complete = True
            for kind in _MEDIA:
                if raw.get(kind):
                    path = _inside(root, raw[kind])
                    if path is None:
                        complete = False
                        break
                    media[kind] = path
            if not complete:
                continue
            settings = {k: v for k, v in (raw.get("settings") or {}).items() if k in _SETTINGS}
            found.append(
                Example(
                    label=str(raw["label"]),
                    prompt=str(raw.get("prompt") or ""),
                    media=media,
                    settings=settings,
                )
            )
        if found:
            examples[mode_id] = tuple(found)
        sample = entry.get("sample")
        if isinstance(sample, dict):
            path = _inside(root, sample.get("file"))
            if path is not None:
                seed = sample.get("seed")
                raw_settings = sample.get("settings") or ""
                samples[mode_id] = Sample(
                    path=path,
                    caption=str(sample.get("caption") or ""),
                    settings=dict(raw_settings)
                    if isinstance(raw_settings, dict)
                    else str(raw_settings),
                    seed=int(seed) if isinstance(seed, int) else None,
                )
    return Showcase(root=root, examples=examples, samples=samples)
