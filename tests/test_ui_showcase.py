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


def test_an_example_writes_only_what_it_names(tmp_path):
    from pathlib import Path

    from ltx25_studio.ui import handlers

    image = tmp_path / "captain.jpg"
    image.write_bytes(b"jpg")
    example = showcase.Example(
        label="The captain speaks",
        prompt="He shouts: all hands on deck!",
        media={"image": image},
        settings={"shape": "3:2", "seconds": 5.0, "seed": 16},
    )
    out = handlers.example_values(example, "image_text_to_audio_video")
    assert out == {
        "prompt": "He shouts: all hands on deck!",
        "image": str(Path(image)),
        "width": 704,
        "height": 480,
        "preset": "3:2",
        "num_frames": 121,
        "seconds": 5.0,
        "auto_duration": False,
        "seed": 16,
    }, "enhance, upscale, other media, strength and the noise level stay as the user left them"


def test_an_exact_size_wins_over_the_shape_and_clears_the_preset():
    from ltx25_studio.ui import handlers

    example = showcase.Example(
        label="Portrait", prompt="p", settings={"shape": "9:16", "width": 480, "height": 704}
    )
    out = handlers.example_values(example, "image_text_to_audio_video")
    assert (out["width"], out["height"]) == (480, 704)
    assert out["preset"] is None, "480 × 704 is not one of the shape presets"


def test_a_pinned_or_promptless_mode_ignores_what_it_cannot_use():
    from ltx25_studio.ui import handlers

    example = showcase.Example(label="x", prompt="p", settings={"shape": "16:9", "upscale": True})
    assert handlers.example_values(example, "image_to_video") == {
        "width": 960,
        "height": 544,
        "preset": "16:9",
        "enable_stage_2": True,
    }
    assert handlers.example_values(example, "text_to_audio") == {"prompt": "p"}


def test_the_summary_says_where_the_length_came_from_and_how_much_the_sound_changed():
    from ltx25_studio import modes
    from ltx25_studio.ui import handlers, render
    from ltx25_studio.ui.fields import FIELDS

    moderately = dict(render.NOISE_CHOICES)["Moderately"]
    params = {f.key: f.default for f in FIELDS} | {"auto_duration": True, "noise_scale": moderately}

    class Result:
        seed = 3

    facts = handlers.run_facts(
        handlers.RunOutcome(
            video=None,
            audio="a.wav",
            status="",
            mode_id="audio_to_audio",
            params=params,
            result=Result(),
        )
    )
    assert facts.length_source == "audio" and facts.audio_level == moderately
    summary = render.run_summary(facts)
    assert "Matched to your audio" in summary and "Moderately" in summary and "0.73" in summary
    facts = handlers.run_facts(
        handlers.RunOutcome(
            video="v",
            audio=None,
            status="",
            mode_id="text_to_video",
            params=params,
            result=Result(),
        )
    )
    assert facts.length_source == "model" and facts.audio_level is None
    assert modes.get_mode("text_to_video").label


# --------------------------------------------------------------------------
# pending (b): every mode's example and sample come from the manifest alone
# --------------------------------------------------------------------------

#: Media per mode, as the lead's manifest will name them: a clip for 04 and 09, an
#: audio file for 10 and 11, an image as well for 11.
FIXTURE_MEDIA = {
    "image_to_video": {"image": "media/still.png"},
    "image_text_to_video": {"image": "media/still.png"},
    "image_to_audio_video": {"image": "media/still.png"},
    "image_text_to_audio_video": {"image": "media/still.png"},
    "video_to_video": {"video": "media/clip.mp4"},
    "video_to_audio": {"video": "media/clip.mp4"},
    "audio_to_audio": {"audio": "media/voice.wav"},
    "audio_to_video": {"audio": "media/voice.wav", "image": "media/still.png"},
}
#: The optional settings an entry may carry, and the words a sample caption gives them.
OPTIONAL = {
    "upscale": "upscaled ×2",
    "enhance": "prompt enhanced",
    "auto_duration": "length chosen by the model",
}


def _fixture_manifest(root):
    """A text-only manifest for all 11 modes; the media files are placeholders made here."""
    for name in ("media/still.png", "media/clip.mp4", "media/voice.wav"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"placeholder")
    entries = {}
    for mode_id, spec in modes.MODES.items():
        settings = {"seed": 21}
        if not spec.skip_stage_2:
            settings["upscale"] = True
        if modes.accepts(spec, "text"):
            settings["enhance"] = True
        if not (modes.accepts(spec, "video") or modes.requires(spec, "audio")):
            settings["auto_duration"] = True
        sample_settings = {
            "size": "704 × 480",
            **{k: v for k, v in settings.items() if k in OPTIONAL},
        }
        (root / "samples").mkdir(exist_ok=True)
        (root / "samples" / f"{mode_id}.mp4").write_bytes(b"placeholder")
        entries[mode_id] = {
            "examples": [
                {
                    "label": f"Example for {mode_id}",
                    "prompt": "a lighthouse at dusk" if modes.accepts(spec, "text") else "",
                    **FIXTURE_MEDIA.get(mode_id, {}),
                    "settings": settings,
                }
            ],
            "sample": {"file": f"samples/{mode_id}.mp4", "settings": sample_settings, "seed": 21},
        }
    write(root, {"modes": entries})
    return entries


def test_a_manifest_with_entries_for_all_eleven_modes_loads_without_a_code_change(tmp_path):
    from ltx25_studio.ui import handlers

    entries = _fixture_manifest(tmp_path)
    shown = showcase.load_showcase(tmp_path, known_modes=tuple(modes.MODES))
    assert set(shown.examples) == set(modes.MODES) == set(shown.samples)
    for mode_id, entry in entries.items():
        (example,) = shown.for_mode(mode_id)
        assert {k: p.name for k, p in example.media.items()} == {
            k: v.split("/")[-1] for k, v in FIXTURE_MEDIA.get(mode_id, {}).items()
        }
        out = handlers.example_values(example, mode_id)
        view = handlers.mode_view(mode_id)
        for kind in example.media:
            assert out.get(kind) == str(example.media[kind]), (mode_id, kind)
        settings = entry["examples"][0]["settings"]
        if "upscale" in settings:
            assert out["enable_stage_2"] is True, mode_id
        if "enhance" in settings:
            assert out["enhance_prompt"] is True, mode_id
        if "auto_duration" in settings:
            assert out["auto_duration"] is True and view.active["auto_duration"], mode_id


def test_the_optional_settings_show_in_the_sample_caption(tmp_path):
    from ltx25_studio.ui import blocks

    entries = _fixture_manifest(tmp_path)
    shown = showcase.load_showcase(tmp_path, known_modes=tuple(modes.MODES))
    for mode_id, entry in entries.items():
        caption = blocks._sample_caption(modes.get_mode(mode_id), shown.sample_for(mode_id))
        assert caption.startswith("**Example output, not your run.**")
        for key, words in OPTIONAL.items():
            assert (words in caption) == (key in entry["sample"]["settings"]), (
                mode_id,
                key,
                caption,
            )
        assert "704 × 480" in caption and "seed 21" in caption


def test_clicking_an_example_fills_the_form_in_every_mode(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from test_ui_form import Form

    from ltx25_studio.ui import build_ui

    _fixture_manifest(tmp_path)
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path))
    form = Form(build_ui())
    buttons = [
        b
        for b in form.demo.blocks.values()
        if "mr-example" in (getattr(b, "elem_classes", None) or [])
    ]
    for mode_id, spec in modes.MODES.items():
        form.pick_mode(mode_id)
        form.click(buttons[0])
        for kind, name in FIXTURE_MEDIA.get(mode_id, {}).items():
            assert str(form.field(kind)).endswith(name.split("/")[-1]), (mode_id, kind)
        if not spec.skip_stage_2:
            assert form.field("enable_stage_2") is True, mode_id
        if modes.accepts(spec, "text"):
            assert form.field("enhance_prompt") is True, mode_id
