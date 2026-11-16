"""Assistive tech hears what the page shows (review round 2, F15; minor 14's sample jump).

* The status region is in the page from the first paint (empty), so its role is set before
  the first message and both "Running" and "Done" are announced; the Done line it carries is
  visually hidden, never ``display: none``.
* The column cost notes are outside every ``aria-hidden`` subtree, and each cell of a noted
  column is described by its note.
The accessibility snapshot itself is checked by the capture rig (assertion F); these tests
pin the markup it depends on.
"""

from __future__ import annotations

import re

import pytest

from ltx25_studio import modes
from ltx25_studio.ui import chrome, render

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


@pytest.fixture(scope="module")
def demo():
    from ltx25_studio.ui import build_ui

    return build_ui()


def by_id(demo, elem_id):
    return next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == elem_id)


def test_the_status_region_is_in_the_page_from_the_first_paint(demo):
    status = by_id(demo, "run-status")
    assert status.visible is True, "a region mounted with its first message is not announced"
    assert status.value in ("", None)
    script = chrome.head_html()
    assert "status.setAttribute('role', 'status')" in script


def test_the_done_line_is_read_but_not_shown():
    css = chrome._css()
    assert "#run-status .mr-report-head { display: none" not in css
    rule = re.search(r"#run-status \.mr-report-head, [^{]*\{([^}]*)\}", css)
    assert rule and "clip: rect(0 0 0 0)" in rule.group(1) and "display" not in rule.group(1)
    head = render.success(
        type("R", (), {"seed": 3, "warnings": ()})(), modes.get_mode("text_to_video")
    )
    assert (
        f'<p class="mr-report-head">Done: {modes.get_mode("text_to_video").label}. Seed 3.</p>'
        in head
    )


def test_the_column_notes_are_outside_every_hidden_subtree_and_describe_their_cells():
    frame = chrome.matrix_frame_html()
    assert not re.search(r'<div class="mx-frame"[^>]*aria-hidden', frame)
    cells = chrome.matrix_cells()
    notes = re.findall(r'<p class="mx-note" id="(mx-note-\d)" data-modes="([^"]*)"', frame)
    assert len(notes) == len(render.COLUMN_NOTES)
    for note_id, listed in notes:
        column = int(note_id.rsplit("-", 1)[1])
        assert set(listed.split()) == {m for m, (_r, c, _n) in cells.items() if c == column}
    for text in render.COLUMN_NOTES.values():
        assert text in frame
    for header in re.findall(r'<div class="mx-(?:col|row|empty)[^"]*"[^>]*>', frame):
        assert 'aria-hidden="true"' in header, header
    assert "setAttribute('aria-describedby', note.id)" in chrome.head_html()


def test_the_sample_jump_shows_only_while_the_monitor_shows_a_sample(tmp_path, monkeypatch):
    import json

    from test_ui_form import Form

    from ltx25_studio.ui import build_ui, showcase

    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "t2av.mp4").write_bytes(b"mp4")
    (tmp_path / showcase.MANIFEST).write_text(
        json.dumps({"modes": {"text_to_audio_video": {"sample": {"file": "samples/t2av.mp4"}}}})
    )
    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path))
    form = Form(build_ui())
    jump = form.by_id("sample-jump")
    assert form.visible(jump) and jump.value == render.SAMPLE_JUMP
    form.pick_mode("text_to_video")  # no sample for 01 in this manifest
    assert not form.visible(jump)
    form.pick_mode("text_to_audio_video")
    assert form.visible(jump)
    assert (
        "@media (min-width: 1024px) { #sample-jump { display: none !important; } }" in chrome._css()
    )
