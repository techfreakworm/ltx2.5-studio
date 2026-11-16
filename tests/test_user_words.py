"""Everything a user reads after pressing Generate is in the UI's words (UI review 2, B-1).

The run report prints ``result.warnings`` verbatim above a collapsed Diagnostics
block (``ui/render.py::success``), and the progress bar shows the phase labels
``generate.ProgressReporter`` emits. The readiness check shows the ``str()`` of
any refusal ``prepare_request`` raises. All three are read by the person who
pressed the button, so none may carry the builder's vocabulary: a mode id, the
encoder's verbs, the model's internals, a path, a Python repr, or words about
payment after the run (cost is the cost panel's job, before the run).

Technical facts -- the decoder policy, evictions, the partition, VRAM laps, the
run folder, the mode id -- live in ``record.diagnostics``, which this file does
not police; it checks they are NOT above it.
"""

from __future__ import annotations

import re

import pytest
from conftest import audio_file, image_input, inputs_for, video_file
from test_encoder_contract import FRAME_SIZE, VocoderPipe

from ltx25_studio import audio_conditioning, modes
from ltx25_studio import generate as generate_module
from ltx25_studio.errors import LTXStudioError

MODE_IDS = tuple(modes.MODES)

#: Words a note, a phase label or a refusal must never carry.
BANNED_WORDS = (
    *MODE_IDS,
    "mux",
    "latent",
    "stage 2",
    "stage-2",
    "kwarg",
    "upstream",
    "vram",
    "num_frames",
    "audio_latents",
    "paid",
    "billed",
    "quota",
)
_ABSOLUTE_PATH = re.compile(r"/(?:Users|var|tmp|home|data|private)/")
#: A word in repr quotes: 'audio_to_audio'. An apostrophe inside a word
#: (model's, it's) is not one.
_REPR_QUOTED = re.compile(r"(?<![A-Za-z])'[^'\s][^']*'(?![A-Za-z])")


def offences(text: str) -> list[str]:
    lowered = text.lower()
    found = [word for word in BANNED_WORDS if word.lower() in lowered]
    if _ABSOLUTE_PATH.search(text):
        found.append("an absolute path")
    quoted = _REPR_QUOTED.findall(text)
    if quoted:
        found.append(f"repr quotes {quoted}")
    return found


def assert_in_the_uis_words(texts, what: str) -> None:
    bad = {text: offences(text) for text in texts if offences(text)}
    assert not bad, f"{what} carries words the UI does not use:\n" + "\n".join(
        f"  {text!r}: {words}" for text, words in bad.items()
    )


class Labels:
    """A ``gr.Progress``-shaped handle that keeps every label it is shown."""

    def __init__(self):
        self.labels: list[str] = []

    def __call__(self, position, desc=None):
        if desc:
            self.labels.append(str(desc))


@pytest.fixture(autouse=True)
def _uploads_load_without_torch(audio_input_ready):
    """The audio modes get a real upload; its loader is stubbed so no torch is needed."""


def run(mode_id, tmp_path, **overrides):
    """One run through the production encoder; returns ``(result, phase labels)``."""
    labels = Labels()
    fields = {**inputs_for(mode_id), **overrides}
    retention = fields.pop("retention_bytes", 10**9)
    result = generate_module.generate(
        mode=mode_id,
        seed=7,
        width=fields.pop("width", FRAME_SIZE),
        height=fields.pop("height", FRAME_SIZE),
        pipeline_factory=lambda **_: VocoderPipe(),
        encoder=None,
        enhancer=lambda prompt, image=None: f"{prompt}, golden hour, slow dolly in",
        output_root=tmp_path,
        retention_bytes=retention,
        progress=labels,
        **fields,
    )
    return result, labels.labels


def shown(result, labels) -> list[str]:
    """What the UI prints above Diagnostics, plus the progress labels."""
    return [str(note) for note in result.warnings] + list(labels)


# --------------------------------------------------------------------------
# every mode, and the variants that add notes
# --------------------------------------------------------------------------

VARIANTS = [(mode_id, {}) for mode_id in MODE_IDS] + [
    ("text_to_audio_video", {"enable_stage_2": True, "width": 128, "height": 128}),
    ("image_text_to_audio_video", {"enhance_prompt": True}),
    ("audio_to_video", {"image": image_input()}),
    ("audio_to_video", {"auto_duration": True}),
    ("text_to_video", {"auto_duration": True}),
    ("text_to_video", {"decoder": "conv"}),
    *[("audio_to_audio", {"noise_scale": level}) for level in audio_conditioning.VARIATION_LEVELS],
    ("audio_to_audio", {"noise_scale": 0.5}),
    *[
        ("video_to_video", {"noise_scale": level, "video": video_file(frames=17)})
        for level in audio_conditioning.VIDEO_VARIATION_LEVELS
    ],
    ("video_to_video", {"noise_scale": 0.6, "video": video_file(frames=17)}),
]


@pytest.mark.parametrize(
    ("mode_id", "overrides"),
    VARIANTS,
    ids=[f"{m}-{'-'.join(sorted(o)) or 'default'}" for m, o in VARIANTS],
)
def test_every_note_and_phase_label_is_in_the_uis_words(mode_id, overrides, tmp_path):
    result, labels = run(mode_id, tmp_path, **overrides)

    assert_in_the_uis_words(shown(result, labels), f"{mode_id} {overrides}")


def test_a_longer_upload_is_trimmed_and_said_in_seconds(tmp_path, monkeypatch):
    from conftest import STUB_SAMPLING_RATE, waveform_of

    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module,
        "load_waveform",
        lambda source, **_: (waveform_of(2, samples=6 * STUB_SAMPLING_RATE), STUB_SAMPLING_RATE),
    )
    result, labels = run("audio_to_video", tmp_path, audio=audio_file(seconds=6.0))

    assert_in_the_uis_words(shown(result, labels), "a trimmed upload")
    assert any("trimmed from" in note for note in result.warnings), result.warnings


def test_the_technical_facts_are_below_diagnostics_not_above(tmp_path):
    # A first run to evict, then a second with a cap that forces it out.
    run("text_to_video", tmp_path / "out")
    result, _labels = run("text_to_video", tmp_path / "out", retention_bytes=1)

    above = " ".join(result.warnings).lower()
    assert "evicted" not in above and "run-" not in above, result.warnings
    assert "decoder" not in above, result.warnings
    assert any("evicted" in line for line in result.record.diagnostics), result.record.diagnostics
    # Which decoder ran, and why, is a diagnostic too (tests/test_decode_policy.py).


def test_a_track_the_encoder_offered_unasked_is_a_diagnostic(tmp_path):
    """An over-generous encoder is the builder's business, said below Diagnostics."""
    from test_exports_contract import FILENAME_FOR, TRACKS, _write

    def encoder(output, *, run_dir, exports, **kwargs):
        return {track: _write(run_dir, FILENAME_FOR[track]) for track in TRACKS}

    result = generate_module.generate(
        mode="text_to_video",
        seed=7,
        **inputs_for("text_to_video"),
        pipeline_factory=lambda **_: VocoderPipe(),
        encoder=encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    assert not any("encoder" in note for note in result.warnings), result.warnings
    assert any("encoder produced audio" in line for line in result.record.diagnostics), (
        result.record.diagnostics
    )


# --------------------------------------------------------------------------
# the exact sentences the lead ruled
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", ["text_to_video", "image_to_video", "image_text_to_video"])
def test_a_silent_mode_names_the_mode_that_keeps_the_sound(mode_id, tmp_path):
    result, _labels = run(mode_id, tmp_path)
    counterpart = modes.counterpart_label_for(mode_id)

    assert (
        "this mode returns silent video. The sound the model made with it was left out; "
        f"{counterpart} keeps it." in result.warnings
    ), result.warnings


def test_video_to_video_names_no_remedy_it_does_not_have(tmp_path):
    result, _labels = run("video_to_video", tmp_path, video=video_file(frames=17))

    assert (
        "this mode returns silent video. The sound the model made with it was left out."
        in result.warnings
    ), result.warnings


@pytest.mark.parametrize(
    ("level", "words"),
    [(0.421875, "a little"), (0.725, "moderately"), (0.909375, "a lot")],
)
def test_audio_to_audio_says_the_sound_is_the_models_and_how_much_it_changed(
    level, words, tmp_path
):
    result, _labels = run("audio_to_audio", tmp_path, noise_scale=level)

    assert "this sound is the model's, made from yours; it is not your file." in result.warnings
    assert f"your audio was changed {words}." in result.warnings, result.warnings


@pytest.mark.parametrize(
    ("mode_id", "asked", "words"),
    [
        ("audio_to_audio", 0.42, "your audio was changed a little."),
        ("audio_to_audio", 0.73, "your audio was changed moderately."),
        ("video_to_video", 0.91, "your clip was changed a lot."),
    ],
)
def test_a_request_that_reads_as_the_level_used_says_only_what_was_used(
    mode_id, asked, words, tmp_path
):
    """Never "You asked for 0.42 ... 0.42 is the nearest" (measured by the lead)."""
    extra = {"video": video_file(frames=17)} if mode_id == "video_to_video" else {}
    result, _labels = run(mode_id, tmp_path, noise_scale=asked, **extra)

    assert words in result.warnings, result.warnings
    assert not any("not one of the levels" in note for note in result.warnings), result.warnings


@pytest.mark.parametrize(
    ("mode_id", "asked", "words", "requested", "used"),
    [
        ("audio_to_audio", 0.5, "your audio was changed a little.", "0.50", "0.42"),
        ("video_to_video", 0.8, "your clip was changed moderately.", "0.80", "0.73"),
    ],
)
def test_a_request_snapped_to_a_different_level_says_what_was_asked_and_used(
    mode_id, asked, words, requested, used, tmp_path
):
    extra = {"video": video_file(frames=17)} if mode_id == "video_to_video" else {}
    result, labels = run(mode_id, tmp_path, noise_scale=asked, **extra)

    assert words in result.warnings, result.warnings
    assert (
        f"you asked for {requested}, which is not one of the levels; the nearest, {used}, "
        f"was used." in result.warnings
    ), result.warnings
    assert_in_the_uis_words(shown(result, labels), f"{mode_id} at {asked}")


def test_audio_to_video_says_the_sound_is_yours_and_the_length_it_set(tmp_path):
    result, _labels = run("audio_to_video", tmp_path, auto_duration=True)

    assert "the sound in this file is your audio, unchanged." in result.warnings
    assert any(
        re.fullmatch(r"length set from your audio: \d+\.\d s\.", n) for n in result.warnings
    ), result.warnings


@pytest.mark.parametrize(
    ("level", "words"),
    [(0.421875, "a little"), (0.725, "moderately"), (0.909375, "a lot")],
)
def test_video_to_video_says_how_much_the_clip_changed(level, words, tmp_path):
    result, _labels = run(
        "video_to_video", tmp_path, noise_scale=level, video=video_file(frames=17)
    )

    assert f"your clip was changed {words}." in result.warnings, result.warnings


def test_upscaling_is_shown_as_the_ui_names_it():
    # A fake pipeline does not report the upscale itself; the runner calls these.
    labels = Labels()
    reporter = generate_module.ProgressReporter(labels)
    reporter.sampling(index=1, passes=2)
    reporter.upscaling()
    reporter.sampling(index=2, passes=2)
    reporter.decoding()

    assert "upscaling ×2" in labels.labels, labels.labels
    # Every label the bar can show, whichever path emits it.
    assert_in_the_uis_words(
        [*labels.labels, *generate_module._PHASE_BANDS], "a progress label"
    )


def test_the_enhancers_rewrite_is_not_repeated_as_a_note(tmp_path):
    result, _labels = run("image_text_to_audio_video", tmp_path, enhance_prompt=True)

    assert not any("golden hour" in note for note in result.warnings), result.warnings
    assert "golden hour" in result.record.prompt


# --------------------------------------------------------------------------
# refusals the readiness check shows
# --------------------------------------------------------------------------


def _refusal(**fields):
    with pytest.raises(LTXStudioError) as caught:
        generate_module.prepare_request(generate_module.GenerationRequest(**fields))
    return str(caught.value)


def _missing_input_cases():
    for mode_id in MODE_IDS:
        spec = modes.get_mode(mode_id)
        for kind in spec.inputs:
            fields = inputs_for(mode_id)
            fields.pop({"text": "prompt"}.get(kind, kind))
            yield pytest.param(mode_id, fields, id=f"{mode_id}-no-{kind}")


@pytest.mark.parametrize(("mode_id", "fields"), list(_missing_input_cases()))
def test_a_missing_input_is_refused_in_the_uis_words(mode_id, fields):
    message = _refusal(mode=mode_id, **fields)

    assert_in_the_uis_words([message], f"{mode_id} missing input")


def _broken(tmp_path, suffix):
    path = tmp_path / f"broken{suffix}"
    path.write_bytes(b"not a media file")
    return str(path)


def test_wrong_inputs_are_refused_in_the_uis_words(tmp_path, monkeypatch):
    monkeypatch.setenv("LTX25_DEVICE", "cpu")
    messages = [
        _refusal(mode="audio_to_video", audio=_broken(tmp_path, ".wav")),
        _refusal(mode="audio_to_audio", audio=_broken(tmp_path, ".wav")),
        _refusal(mode="video_to_video", video=_broken(tmp_path, ".mp4")),
        _refusal(mode="video_to_audio", video=_broken(tmp_path, ".mp4")),
        # B-2: a set length longer than the audio, and a recording no length fits
        _refusal(mode="audio_to_video", audio=audio_file(seconds=4.0), num_frames=121),
        _refusal(mode="audio_to_audio", audio=audio_file(seconds=4.0), num_frames=121),
        _refusal(mode="audio_to_video", audio=audio_file(seconds=0.34), num_frames=121),
        _refusal(mode="audio_to_audio", audio=audio_file(seconds=0.34), num_frames=121),
        # a shot placed past the end of the clip
        _refusal(
            mode="image_to_video",
            image=image_input(),
            num_frames=9,
            conditions=({"source": image_input(), "kind": "image", "index": 17},),
        ),
        # enhance with nothing to enhance
        _refusal(mode="text_to_video", prompt="", enhance_prompt=True),
        # Upscale on this machine at a real size
        _refusal(
            mode="text_to_video",
            prompt="p",
            width=704,
            height=480,
            num_frames=121,
            enable_stage_2=True,
        ),
        # Best quality chosen where it cannot run at this size
        _refusal(
            mode="text_to_video",
            prompt="p",
            width=704,
            height=480,
            num_frames=121,
            decoder="diffusion",
        ),
    ]

    assert_in_the_uis_words(messages, "a refusal")


def test_a_set_length_longer_than_the_audio_is_refused_with_the_length_that_fits():
    message = _refusal(mode="audio_to_video", audio=audio_file(seconds=4.0), num_frames=121)

    assert message == (
        'Your audio is 4.0 s and the length is set to 5.0 s. Choose "Match my clip", '
        "or set 3.7 s or less."
    )


@pytest.mark.parametrize(
    ("mode_id", "label"), [("audio_to_video", "Audio to video"), ("audio_to_audio", "Audio to audio")]
)
def test_a_recording_no_length_fits_is_called_too_short_with_the_shortest_that_works(
    mode_id, label
):
    """Never "set 0.0 s or less": a single frame is no length a user can set."""
    message = _refusal(mode=mode_id, audio=audio_file(seconds=0.34), num_frames=121)

    assert message == (
        f"Your audio is 0.3 s, too short for {label}: the shortest recording it can use is 0.4 s."
    )


@pytest.mark.parametrize(
    ("mode_id", "label"),
    [("audio_to_video", "Audio to video"), ("audio_to_audio", "Audio to audio")],
)
def test_match_my_clip_calls_a_recording_too_short_in_the_same_words(mode_id, label):
    """With "Match my clip" such a recording once ran as one frame; now it is refused alike."""
    matched = _refusal(mode=mode_id, audio=audio_file(seconds=0.34), auto_duration=True)

    assert matched == _refusal(mode=mode_id, audio=audio_file(seconds=0.34), num_frames=121)
    assert matched == (
        f"Your audio is 0.3 s, too short for {label}: the shortest recording it can use is 0.4 s."
    )
    assert_in_the_uis_words([matched], "Match my clip with a recording too short")


def test_the_shortest_usable_recording_is_one_that_runs():
    """The 0.4 s it names is enough: refused with no length below it, run with one at it."""
    assert "or set 0.3 s or less" in _refusal(
        mode="audio_to_video", audio=audio_file(seconds=0.4), num_frames=121
    )
    scoped, _spec, _notes = generate_module.prepare_request(
        generate_module.GenerationRequest(
            mode="audio_to_video", audio=audio_file(seconds=0.4), num_frames=9
        )
    )
    assert scoped.num_frames == 9


def test_upscale_on_this_machine_is_refused_as_the_ui_names_it(monkeypatch):
    monkeypatch.setenv("LTX25_DEVICE", "cpu")
    message = _refusal(
        mode="text_to_video",
        prompt="p",
        width=704,
        height=480,
        num_frames=121,
        enable_stage_2=True,
    )

    assert message == "Upscale ×2 needs a CUDA GPU; this machine runs on cpu."


def test_the_banned_word_check_catches_what_it_is_for():
    """The checker itself: each banned shape is caught, ordinary English is not."""
    assert offences("mode 'text_to_video' exports video only")
    assert offences("dropped rather than muxed")
    assert offences("written to /Users/someone/outputs/run-1")
    assert offences("rendered and paid for either way")
    assert not offences("this sound is the model's, made from yours; it is not your file.")
    assert not offences('Choose "Match my clip", or set 3.7 s or less.')
