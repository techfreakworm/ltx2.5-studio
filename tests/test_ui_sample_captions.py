"""The shipped samples' captions are true and in a creator's words (review round 4, minors 11, 8).

* Minor 11: modes 10 and 11 called 4.7 s "the recording's length" for a 5.0 s recording (the
  run is the longest length that fits it), mode 11's "the example's recording and prompt" did
  not say which of its two examples, and "single pass" is builder shorthand for "no upscale".
  Now every caption names the example it was made from, every length the settings state is the
  sample's own, a recording's length is the recording's, and "no upscale" is said in words.

* Minor 8: the samples were made on the operator's Mac, and on the Space the same prompt and
  seed give another take (mode 05 at seed 12: crewmen on deck there, the ship from above here),
  yet the captions said "Same prompt and seed as ...". Now each says it was made earlier by this
  app from that example, with its prompt and seed, and promises no repeat.

The captions are read from the shipped manifest; lengths are measured off the files when the
media were fetched (a clone without Git LFS holds pointers, and the length checks skip).
"""

from __future__ import annotations

import json
import re

import pytest
from conftest import REPO_ROOT

from ltx25_studio import modes
from ltx25_studio.ui import showcase

EXAMPLES = REPO_ROOT / showcase.DEFAULT_DIR
MANIFEST = json.loads((EXAMPLES / showcase.MANIFEST).read_text(encoding="utf-8"))["modes"]
POINTER_START = b"version https://git-lfs.github.com/spec/v1"


def fetched(path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(POINTER_START)) != POINTER_START


def seconds_of(path) -> float:
    av = pytest.importorskip("av")
    with av.open(str(path)) as container:
        video = next(iter(container.streams.video), None)
        if video is not None:
            return float(video.frames) / float(video.average_rate)
        return float(container.duration) / 1e6


@pytest.mark.parametrize("mode_id", sorted(MANIFEST))
def test_every_caption_names_its_example_in_plain_words(mode_id):
    entry = MANIFEST[mode_id]
    caption = entry["sample"]["caption"]
    labels = [example["label"] for example in entry["examples"]]
    named = [label for label in labels if f"“{label}”" in caption]
    assert len(named) == 1, (mode_id, caption, labels)
    example = next(e for e in entry["examples"] if e["label"] == named[0])
    assert example["settings"].get("seed") == entry["sample"]["seed"], "made with its seed"
    for shorthand in ("single pass", "the example's", "the example"):
        assert shorthand not in caption, (mode_id, shorthand)
    if "video" in modes.get_mode(mode_id).exports:
        assert "no upscale" in caption, (mode_id, caption)


@pytest.mark.parametrize("mode_id", sorted(MANIFEST))
def test_every_length_a_caption_states_is_the_samples_own(mode_id):
    entry = MANIFEST[mode_id]
    sample = EXAMPLES / entry["sample"]["file"]
    if not fetched(sample):
        pytest.skip("the media were not fetched (Git LFS pointers)")
    settings = str(entry["sample"]["settings"])
    stated = float(re.search(r"(\d+(?:\.\d+)?) s\b", settings).group(1))
    assert abs(stated - seconds_of(sample)) < 0.1, (mode_id, settings, seconds_of(sample))
    for found in re.finditer(r"the (\d+(?:\.\d+)?) s recording", settings):
        recording = next(EXAMPLES / e["audio"] for e in entry["examples"] if e.get("audio"))
        assert abs(float(found.group(1)) - seconds_of(recording)) < 0.05, (mode_id, settings)
    assert "the recording's length" not in settings, (mode_id, settings)


@pytest.mark.parametrize("mode_id", sorted(MANIFEST))
def test_every_caption_says_the_sample_was_made_earlier_and_promises_no_repeat(mode_id):
    caption = MANIFEST[mode_id]["sample"]["caption"]
    assert caption.startswith("Made earlier by this app from “"), (mode_id, caption)
    for promise in ("Same prompt and seed", "same seed", "Same seed"):
        assert promise not in caption, (mode_id, promise)
