"""Follow-ons from the other owners' fixes and the verdict's section 9 (brief W3-U R3d, item 6).

* "Matches your N s clip" measures a clip by its picture, as the backend plans it since 16bbbd3:
  a WebM whose sound outlasts its picture was said to match its sound's length.
* The result player starts the finished run by itself: it is told to before the file arrives.
* A run that finished but could not be shown is listed in Recent runs, and its card sends the
  user to Show before it suggests running again.
* Longest (Advanced, the model-chosen length) stops at the model's longest, and the page keeps
  Shortest below Longest, so neither field-name refusal can be reached from the page.
* The shot-anchor readout (kept for the API's shot list) says an out-of-range anchor is refused
  before the run, which is what happens now.
"""

from __future__ import annotations

import pytest
from conftest import inputs_for

from ltx25_studio import constraints, modes
from ltx25_studio import generate as generate_module
from ltx25_studio.ui import blocks, handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


def form_values(mode_id: str, **overrides) -> dict:
    return {spec.key: spec.default for spec in FIELDS} | inputs_for(mode_id) | overrides


# --------------------------------------------------------------------------
# a clip is as long as its picture
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["webm", "mkv", "mp4"])
def test_a_clip_whose_sound_outlasts_its_picture_matches_its_picture(tmp_path, kind):
    pytest.importorskip("av")
    from test_picture_and_sound_length import clip_with_long_sound

    clip = clip_with_long_sound(tmp_path, kind, frames=9, audio_seconds=2.0)
    picture = generate_module._video_seconds(clip)
    assert picture < 0.5, "the fixture is a 9-frame picture under 2 s of sound"
    assert handlers.clip_seconds(clip, kind="video") == pytest.approx(picture)
    assert handlers.frames_matching_clip(handlers.clip_seconds(clip)) == 9


def test_match_my_clip_states_what_will_run_for_such_a_webm(tmp_path):
    """Through the page: in Video to audio the line and the frames are the picture's, and the
    backend plans the same frames for the same clip."""
    pytest.importorskip("av")
    from test_picture_and_sound_length import clip_with_long_sound
    from test_ui_form import Form

    clip = clip_with_long_sound(tmp_path, "webm", frames=9, audio_seconds=2.0)
    form = Form(blocks.build_ui())
    form.pick_mode("video_to_audio")
    form.user(form.fields["video"], clip)
    seconds = generate_module._video_seconds(clip)
    said = [form.get(b).replace("\u00a0", " ") for b in form.demo.blocks.values() if isinstance(form.get(b), str)]
    assert f"Matches your {seconds:.1f} s clip: 9 frames." in said
    assert form.field("num_frames") == 9
    sent = dict(zip(FIELD_KEYS, form.form_values(), strict=True))
    request = generate_module.GenerationRequest(mode="video_to_audio", **sent)
    prepared, _spec, _notes = generate_module.prepare_request(request)
    assert prepared.num_frames == 9, "the page and the backend plan the same length"


def test_an_unreadable_clip_has_no_length():
    assert handlers.clip_seconds("/no/such/clip.webm", kind="video") is None
    assert handlers.clip_seconds("/no/such/audio.wav", kind="audio") is None
    assert handlers.clip_seconds(None) is None


# --------------------------------------------------------------------------
# the finished run plays by itself
# --------------------------------------------------------------------------


def test_the_result_player_is_set_to_autoplay_before_the_file_arrives():
    demo = blocks.build_ui()
    run = next(f for f in demo.fns.values() if f.api_name == "on_generate")
    start = demo.fns[run.trigger_after]
    video = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "video-out")
    update = start.fn("text_to_audio_video", False, "a harbour at dawn")[start.outputs.index(video)]
    assert update["autoplay"] is True and update["value"] is None and update["visible"] is True
    assert video in run.outputs, "the file then arrives through /on_generate, into a player told to play"


# --------------------------------------------------------------------------
# a run finished but not shown: listed, and Show is the step
# --------------------------------------------------------------------------


class _Result:
    seed = 7
    warnings = ()
    run_dir = "unused"
    record = None


def _finished(tmp_path, *, keep_file: bool = True) -> handlers.RunOutcome:
    video = tmp_path / "output.mp4"
    if keep_file:
        video.write_bytes(b"x")
    return handlers.RunOutcome(
        video=str(video),
        audio=str(video),
        status="Done",
        mode_id="text_to_audio_video",
        params=form_values("text_to_audio_video", width=704, height=480, num_frames=49),
        result=_Result(),
        took=208.0,
    )


def test_an_unshown_run_whose_file_is_kept_sends_the_user_to_show(tmp_path):
    card = handlers.unshown_report(_finished(tmp_path), "the server's message")
    words = " ".join(card.split())
    assert "Its file is kept: press Show in Recent runs." in words
    assert "If that fails too, report it with the detail below." in words
    assert "Press Generate again" not in words, "a paid run is never the first step"
    elsewhere = tmp_path / "evicted"
    elsewhere.mkdir()
    gone = handlers.unshown_report(_finished(elsewhere, keep_file=False), "m")
    assert "Press Generate again" in " ".join(gone.split()), "no file kept: run again"


def test_the_failure_step_lists_an_unshown_run_with_its_show_button(tmp_path):
    from test_ui_form import Form

    form = Form(blocks.build_ui())

    class Request:
        session_hash = "unshown"

    outcome = _finished(tmp_path)
    blocks._remember(Request(), outcome)
    blocks._note_failure("unshown", RuntimeError("Cannot move the file outside allowed_paths"))
    form.run(form.fn("_finish_failed"), request=Request())
    recent = form.fn("_finish").inputs[1]
    entries = form.get(recent)
    assert len(entries) == 1 and entries[0]["video"] == outcome.video
    show = next(b for b in form.fn("_finish").outputs if isinstance(b, gr.Button) and b.value == "Show")
    assert form.visible(show), "Show serves the kept file without a paid rerun"
    status = form.get(form.by_id("run-status"))
    assert "press Show in Recent runs" in status
    assert form.get(form.by_id("monitor-caption")) == render.monitor_caption(kind="unshown")
    form.click(show)
    assert form.get(form.by_id("video-out")) == outcome.video
    assert form.get(form.by_id("monitor-caption")).startswith("**Earlier run.**")


# --------------------------------------------------------------------------
# Longest stops at the model's longest; Shortest stays below Longest
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shortest", "longest", "edited", "expected"),
    [
        (2.0, 25.0, "longest", (2.0, 20.0)),
        (2.0, 5.0, "longest", (2.0, 5.0)),
        (6.0, 5.0, "shortest", (6.0, 6.1)),
        (6.0, 3.0, "longest", (2.9, 3.0)),
        (25.0, 5.0, "shortest", (19.9, 20.0)),
        (5.0, 5.0, "longest", (4.9, 5.0)),
        (None, 5.0, "longest", (None, 5.0)),
    ],
)
def test_the_page_keeps_shortest_below_longest_and_longest_within_the_model(shortest, longest, edited, expected):
    got = handlers.ordered_seconds(shortest, longest, edited=edited)
    for value, want in zip(got, expected, strict=True):
        assert (value is None and want is None) or value == pytest.approx(want), (got, expected)
    assert constraints.MAX_SECONDS == 20.0


def test_neither_field_name_refusal_can_be_reached_from_the_page():
    """Driven through the built page: whatever is typed into Shortest or Longest, the request
    the page holds passes the backend's own length checks."""
    from test_ui_form import Form

    form = Form(blocks.build_ui())
    form.pick_mode("text_to_video")
    form.user(form.fields["prompt"], "a tram")
    form.user(form.fields["auto_duration"], True)
    for key, value in (("max_seconds", 25.0), ("min_seconds", 30.0), ("max_seconds", 1.0), ("min_seconds", 7.5)):
        form.user(form.fields[key], value)
        shortest, longest = form.field("min_seconds"), form.field("max_seconds")
        assert longest <= constraints.MAX_SECONDS and shortest < longest, (key, value, shortest, longest)
        facts = handlers.review("text_to_video", dict(zip(FIELD_KEYS, form.form_values(), strict=True)))
        problems = " ".join(facts.problems or ()) + " " + str(facts.reason or "")
        assert "max_seconds" not in problems and "min_seconds" not in problems, problems
    info = form.props[form.fields["max_seconds"]._id]["info"]
    assert f"{constraints.MAX_SECONDS:g} s at most" in info


# --------------------------------------------------------------------------
# the shot-anchor readout says what happens now
# --------------------------------------------------------------------------


def test_an_anchor_outside_the_clip_is_said_to_be_refused_before_the_run():
    readout = render.shot_anchor_readout(
        [{"frame": 400, "latent": 50, "first_frame": 393, "last_frame": 400, "exact": False, "shared": True}],
        out_of_range=[400],
    )
    assert "Upstream" not in readout and "generates anyway" not in readout
    assert "outside the clip" in readout and "refused before" in readout
    assert modes.get_mode("text_to_video")  # the readout names no mode: it is the API's
