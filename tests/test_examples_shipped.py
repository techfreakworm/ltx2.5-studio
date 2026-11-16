"""The examples shipped in ``examples/``, the directory the loader reads by default.

``examples/media/**`` and ``examples/samples/**`` are Git LFS files (``.gitattributes``): a
clone with Git LFS holds the media themselves, a clone without it (or an export made with
``GIT_LFS_SKIP_SMUDGE=1``) a three-line pointer in place of each. The files that ship are
pinned by name, the manifest must name every mode with an example and a sample, and exactly
the files that are there; the loader must return every entry when the media were fetched, and
skip each entry that needs a pointer with its log line when they were not; the page must build
either way. Which of the two this checkout is, is read from the first bytes of one sample.

Audio to video does not lip-sync, so nothing it shows may suggest it does. Measured on real
weights, the mouth's opening frame by frame against the speech level of the sound track:
+0.845 in image + text to video with sound, where the model voices the words itself, and
-0.379 in audio to video with a portrait and a recording of speech. Every audio-to-video run
measured lay between -0.38 and +0.07, no better than an unrelated sound track scores
(docs/SPACE_MODE_MATRIX.md §10.2). So its examples ask for no speech and bring no recording
named as speech, and its sample's caption promises no speaking face.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import PurePosixPath

import pytest
from conftest import REPO_ROOT

from ltx25_studio import modes
from ltx25_studio.ui import showcase

EXAMPLES = REPO_ROOT / showcase.DEFAULT_DIR
MANIFEST = json.loads((EXAMPLES / showcase.MANIFEST).read_text(encoding="utf-8"))["modes"]
MEDIA = ("image", "video", "audio")
#: The first line of every Git LFS pointer (the spec: git-lfs, docs/spec.md).
POINTER_START = b"version https://git-lfs.github.com/spec/v1"
#: The file whose first bytes say whether this checkout fetched the media.
PROBE = EXAMPLES / "samples" / "text_to_video.mp4"
#: Ignored by git (.gitignore), so it never ships.
IGNORED = {".DS_Store"}

#: Every media file that ships, and only these: a change here is a change to what the
#: Space downloads.
SHIPPED = {
    "media/cabin.jpg",
    "media/cabin_clip.mp4",
    "media/captain.jpg",
    "media/captain_silent.mp4",
    "media/ocean.wav",
    "media/tram.jpg",
    "samples/audio_to_audio.m4a",
    "samples/audio_to_video.mp4",
    "samples/image_text_to_audio_video.mp4",
    "samples/image_text_to_video.mp4",
    "samples/image_to_audio_video.mp4",
    "samples/image_to_video.mp4",
    "samples/text_to_audio.m4a",
    "samples/text_to_audio_video.mp4",
    "samples/text_to_video.mp4",
    "samples/video_to_audio.m4a",
    "samples/video_to_video.mp4",
}

#: Words that ask for speech, at the start of a word in any case ("speaks", "Shouting",
#: "voices"); "sing" but not "single".
SPEECH = re.compile(r"\b(?:speak|say|said|shout|talk|sing(?!l)|voice)", re.IGNORECASE)
#: A quoted line: any double quote, or single quotes around words (not an apostrophe).
QUOTED_LINE = re.compile(r"[\"“”„«»]|(?<!\w)['‘][^'’\n]+['’](?!\w)")
#: A recording whose name says it is speech.
SPEECH_FILE = re.compile(r"(?:^|[^a-z])(?:line|speech|voice|welcome)", re.IGNORECASE)
#: Words a caption would use to promise a face that speaks.
FACE = re.compile(r"\b(?:face|lip|mouth)", re.IGNORECASE)


def _is_pointer(path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(POINTER_START)) == POINTER_START


FETCHED = not _is_pointer(PROBE)


def _media_of(example: dict) -> list[str]:
    return [example[kind] for kind in MEDIA if example.get(kind)]


def _named() -> set[str]:
    named = {entry["sample"]["file"] for entry in MANIFEST.values()}
    for entry in MANIFEST.values():
        for example in entry["examples"]:
            named.update(_media_of(example))
    return named


def _files_under(folder) -> set[str]:
    return {
        path.relative_to(EXAMPLES).as_posix()
        for path in folder.rglob("*")
        if path.is_file() and path.name not in IGNORED
    }


def _shipped() -> set[str]:
    return _files_under(EXAMPLES / "media") | _files_under(EXAMPLES / "samples")


def _not_fetched(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "not fetched" in r.getMessage()]


def _speech_in(text: str) -> list[str]:
    return [m.group() for pattern in (SPEECH, QUOTED_LINE) for m in pattern.finditer(text)]


def test_the_manifest_names_all_eleven_modes_each_with_an_example_and_a_sample():
    assert set(MANIFEST) == set(modes.MODES)
    for mode_id, entry in MANIFEST.items():
        assert entry["examples"], mode_id
        assert all(example.get("label") for example in entry["examples"]), mode_id
        assert entry["sample"].get("file"), mode_id


def test_every_file_the_manifest_names_is_in_examples_and_every_media_file_is_named():
    root = EXAMPLES.resolve()
    for relative in sorted(_named()):
        path = (EXAMPLES / relative).resolve()
        assert path.is_relative_to(root) and path.is_file(), relative
    assert _shipped() == _named(), "a media file nobody names would ship as dead weight"


def test_the_shipped_files_are_these_seventeen_and_all_sit_where_git_lfs_tracks_them():
    assert _named() == SHIPPED, "the manifest names other media than the ones pinned here"
    assert _files_under(EXAMPLES) == SHIPPED | {showcase.MANIFEST, "PROVENANCE.md"}, (
        "a file elsewhere in examples/ is not a Git LFS file and would ship as a plain blob"
    )


def test_the_media_are_git_lfs_files_and_the_manifest_is_plain_text():
    rules = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    for pattern in ("examples/media/**", "examples/samples/**"):
        assert f"{pattern} filter=lfs diff=lfs merge=lfs -text" in rules, pattern
    for name in (showcase.MANIFEST, "PROVENANCE.md"):
        assert not _is_pointer(EXAMPLES / name), name
    states = {relative: _is_pointer(EXAMPLES / relative) for relative in sorted(_shipped())}
    assert set(states.values()) == {not FETCHED}, (
        f"{PROBE.name} says the media were {'' if FETCHED else 'not '}fetched; these disagree: "
        + ", ".join(name for name, pointer in states.items() if pointer == FETCHED)
    )


def test_audio_to_video_asks_for_no_speech_and_its_sample_promises_no_speaking_face():
    entry = MANIFEST["audio_to_video"]
    offending = {}
    for example in entry["examples"]:
        found = _speech_in(f"{example['label']}\n{example.get('prompt', '')}")
        name = PurePosixPath(example.get("audio", "")).name
        if SPEECH_FILE.search(name):
            found.append(name)
        if found:
            offending[example["label"]] = found
    assert offending == {}, "an audio-to-video example that asks for speech"
    caption = entry["sample"]["caption"]
    promised = [m.group() for pattern in (SPEECH, FACE) for m in pattern.finditer(caption)]
    assert promised == [], caption
    seeds = {example["settings"].get("seed") for example in entry["examples"]}
    assert entry["sample"]["seed"] in seeds, "the sample is what one of these examples makes"


def test_the_loader_returns_what_was_fetched_and_skips_each_pointer_with_its_line(caplog):
    with caplog.at_level(logging.WARNING, logger=showcase.__name__):
        shown = showcase.load_showcase(EXAMPLES, known_modes=tuple(modes.MODES))
    lines = _not_fetched(caplog)
    if FETCHED:
        assert lines == []
        root = EXAMPLES.resolve()
        for mode_id, entry in MANIFEST.items():
            labels = [example["label"] for example in entry["examples"]]
            assert [e.label for e in shown.for_mode(mode_id)] == labels, mode_id
            for example, raw in zip(shown.for_mode(mode_id), entry["examples"], strict=True):
                got = {kind: p.relative_to(root).as_posix() for kind, p in example.media.items()}
                assert got == {kind: raw[kind] for kind in MEDIA if raw.get(kind)}, raw["label"]
            sample = shown.sample_for(mode_id)
            assert sample is not None, mode_id
            assert sample.path == (EXAMPLES / entry["sample"]["file"]).resolve()
        return
    skipped = []
    for mode_id, entry in MANIFEST.items():
        kept = [example["label"] for example in entry["examples"] if not _media_of(example)]
        assert [e.label for e in shown.for_mode(mode_id)] == kept, mode_id
        assert shown.sample_for(mode_id) is None, mode_id
        skipped += [_media_of(example)[0] for example in entry["examples"] if _media_of(example)]
        skipped.append(entry["sample"]["file"])
    assert len(lines) == len(skipped), "one line per entry skipped"
    for relative, line in zip(skipped, lines, strict=True):
        assert repr(relative) in line and "Git LFS pointer" in line, (relative, line)


def test_the_page_builds_from_the_default_directory_either_way(monkeypatch, caplog):
    pytest.importorskip("gradio", reason="gradio is a declared dev dependency")
    from ltx25_studio.ui import build_ui
    from ltx25_studio.ui.blocks import DEFAULT_MODE

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv(showcase.EXAMPLES_ENV, raising=False)
    with caplog.at_level(logging.WARNING, logger=showcase.__name__):
        demo = build_ui()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    buttons = [
        b.value
        for b in demo.blocks.values()
        if "mr-example" in (getattr(b, "elem_classes", None) or [])
        and getattr(b, "visible", True) is not False
    ]
    entry = MANIFEST[DEFAULT_MODE]
    if FETCHED:
        assert by_id["monitor-idle"].visible is False
        assert by_id["video-out"].visible is not False and by_id["sample-jump"].visible is not False
        assert buttons == [example["label"] for example in entry["examples"]]
        assert _not_fetched(caplog) == []
    else:
        assert by_id["monitor-idle"].visible is not False, "the designed idle monitor"
        assert by_id["video-out"].visible is False and by_id["video-out"].value is None
        assert by_id["sample-jump"].visible is False
        assert buttons == [e["label"] for e in entry["examples"] if not _media_of(e)]
        assert _not_fetched(caplog), "each skipped entry is logged"
