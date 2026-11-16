"""Examples and sample outputs come from an optional manifest (R6, AC-20, AC-21).

No manifest: the page shows a designed idle monitor and no example row.
A manifest: examples per mode, each filling the form and running nothing, and a
sample output per mode for the idle monitor. Entries that name a missing file,
or a path outside the examples folder, are skipped rather than shown broken.
"""

from __future__ import annotations

import json

import pytest

from ltx25_studio import modes
from ltx25_studio.ui import showcase


def write(directory, data) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / showcase.MANIFEST).write_text(json.dumps(data))


def test_no_manifest_is_an_empty_showcase(tmp_path):
    shown = showcase.load_showcase(tmp_path)
    assert not shown.has_examples and shown.samples == {}


def test_a_manifest_gives_examples_and_samples(tmp_path):
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "harbour.png").write_bytes(b"png")
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "t2av.mp4").write_bytes(b"mp4")
    write(
        tmp_path,
        {
            "modes": {
                "image_to_audio_video": {
                    "examples": [
                        {
                            "label": "Harbour",
                            "image": "media/harbour.png",
                            "settings": {"shape": "16:9", "seconds": 5, "bogus": 1},
                        }
                    ],
                    "sample": {
                        "file": "samples/t2av.mp4",
                        "caption": "c",
                        "settings": "704 × 480",
                        "seed": 4,
                    },
                }
            }
        },
    )
    shown = showcase.load_showcase(tmp_path, known_modes=tuple(modes.MODES))
    (example,) = shown.for_mode("image_to_audio_video")
    assert example.label == "Harbour"
    assert example.media["image"].name == "harbour.png"
    assert example.settings == {"shape": "16:9", "seconds": 5}, "unknown settings are dropped"
    sample = shown.sample_for("image_to_audio_video")
    assert sample.is_video and sample.seed == 4 and sample.settings == "704 × 480"


@pytest.mark.parametrize("bad", ["missing.png", "../outside.png", "/etc/hosts"])
def test_an_entry_naming_a_missing_or_outside_file_is_skipped(tmp_path, bad):
    (tmp_path.parent / "outside.png").write_bytes(b"x")
    write(
        tmp_path / "ex",
        {
            "modes": {
                "image_to_video": {
                    "examples": [{"label": "Broken", "image": bad}],
                    "sample": {"file": bad},
                }
            }
        },
    )
    shown = showcase.load_showcase(tmp_path / "ex")
    assert shown.for_mode("image_to_video") == () and shown.sample_for("image_to_video") is None


def test_an_unknown_mode_and_an_unreadable_manifest_are_ignored(tmp_path):
    write(tmp_path / "a", {"modes": {"no_such_mode": {"examples": [{"label": "x"}]}}})
    assert not showcase.load_showcase(tmp_path / "a", known_modes=tuple(modes.MODES)).has_examples
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / showcase.MANIFEST).write_text("{not json")
    assert not showcase.load_showcase(tmp_path / "b").has_examples


def test_the_directory_comes_from_the_environment(tmp_path, monkeypatch):
    write(tmp_path, {"modes": {"text_to_video": {"examples": [{"label": "Tram"}]}}})
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path))
    assert showcase.load_showcase().for_mode("text_to_video")[0].label == "Tram"


def test_without_a_manifest_the_page_has_no_example_row_and_an_idle_monitor(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "none"))
    from ltx25_studio.ui import build_ui

    demo = build_ui()
    ids = {getattr(b, "elem_id", None) for b in demo.blocks.values()}
    assert "examples" not in ids, "'Try an example' is not rendered without examples"
    idle = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "monitor-idle")
    video = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "video-out")
    assert idle.visible is not False and video.visible is False
    assert isinstance(idle, gr.HTML) and "Your result plays here." in idle.value


def test_with_a_manifest_the_examples_say_nothing_runs(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    write(
        tmp_path,
        {"modes": {"text_to_audio_video": {"examples": [{"label": "Harbour", "prompt": "p"}]}}},
    )
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path))
    from ltx25_studio.ui import build_ui

    demo = build_ui()
    texts = [getattr(b, "value", "") for b in demo.blocks.values()]
    assert "Fills the form. Nothing runs until you press Generate." in texts
    labels = [
        getattr(b, "value", None)
        for b in demo.blocks.values()
        if "mr-example" in (getattr(b, "elem_classes", None) or [])
    ]
    assert "Harbour" in labels
