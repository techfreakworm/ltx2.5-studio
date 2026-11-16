"""A run is validated before a GPU is reserved (F02, F03, F05, F06, F12).

On the Space every ``generate()`` call is routed through ``@spaces.GPU``, which
queues for a ZeroGPU slot and pre-checks quota at the full reservation before
the worker runs a line. Every refusal that can be decided from the request
alone -- a missing input, an illegal size, an anchor off the end of the clip,
an audio file that will not open, a LoRA file that has gone -- must therefore
be decided in the web process, before the dispatch is called.

Each spec here registers a dispatch that records its calls and a pipeline
factory that records its calls, and requires both lists to stay empty.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import audio_file, image_input, inputs_for, video_file, waveform_of
from test_exports_contract import MODE_IDS, FakePipe, factory_returning, one_muxed_file_encoder

from ltx25_studio import constraints, errors, modes
from ltx25_studio import generate as generate_module
from ltx25_studio.conditioning import ConditionSpec
from ltx25_studio.errors import ConstraintError, LoRAError, LTXStudioError

# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.dispatched: list[dict] = []
        self.built: list[dict] = []

    def dispatch(self, **kwargs):
        self.dispatched.append(kwargs)
        raise AssertionError("the GPU dispatch was reached by a request that should be refused")

    def factory(self, **kwargs):
        self.built.append(kwargs)
        raise AssertionError("the pipeline was built for a request that should be refused")


@pytest.fixture
def refused(monkeypatch, tmp_path):
    """Call ``generate()`` with a dispatch registered; return the typed error."""

    def go(expected=LTXStudioError, **fields):
        recorder = Recorder()
        monkeypatch.setattr(generate_module, "_GPU_DISPATCH", recorder.dispatch)
        with pytest.raises(expected) as caught:
            generate_module.generate(
                seed=7,
                pipeline_factory=recorder.factory,
                encoder=one_muxed_file_encoder,
                output_root=tmp_path,
                retention_bytes=10**9,
                **fields,
            )
        assert recorder.dispatched == [], "the refusal came after the GPU was reserved"
        assert recorder.built == [], "the refusal came after the pipeline was built"
        assert not isinstance(caught.value, AssertionError)
        return caught.value

    return go


@pytest.fixture
def accepted(monkeypatch, tmp_path, audio_input_ready):
    """Call ``generate()`` through an in-process dispatch; return (result, pipe kwargs)."""

    def go(**fields):
        pipe = FakePipe()
        real = generate_module.generate

        def dispatch(**kwargs):
            token = generate_module._IN_GPU_WORKER.set(True)
            try:
                return real(**kwargs)
            finally:
                generate_module._IN_GPU_WORKER.reset(token)

        monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
        result = generate_module.generate(
            seed=7,
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path,
            retention_bytes=10**9,
            **fields,
        )
        return result, pipe.calls[-1]

    return go


#: How the error names each kind of input to a creator.
KIND_WORDS = {"text": "prompt", "image": "image", "video": "video", "audio": "audio"}

REQUIRED = [(m, kind) for m in MODE_IDS for kind in modes.get_mode(m).inputs]


# --------------------------------------------------------------------------
# F02: required inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("mode_id", "kind"), REQUIRED, ids=[f"{m}-no-{k}" for m, k in REQUIRED])
def test_a_missing_required_input_is_refused_before_the_dispatch(mode_id, kind, refused):
    fields = inputs_for(mode_id)
    fields.pop({"text": "prompt"}.get(kind, kind))

    error = refused(mode=mode_id, **fields)

    missing_input_error = getattr(errors, "MissingInputError", None)
    assert missing_input_error is not None, "errors.MissingInputError does not exist"
    assert isinstance(error, missing_input_error), f"{type(error).__name__}: {error}"
    message = str(error)
    spec = modes.get_mode(mode_id)
    assert spec.label in message, f"the error does not name the mode: {message!r}"
    assert KIND_WORDS[kind] in message, f"the error does not name the missing input: {message!r}"
    assert "supplied" not in message and "none was" not in message, message
    # A suggested mode must be one the user can run with what is already supplied.
    suggestion = message.partition(" switch to ")[2].rstrip(".")
    if suggestion:
        (other,) = [m for m in MODE_IDS if modes.get_mode(m).label == suggestion]
        assert other != mode_id, message
        assert set(modes.get_mode(other).inputs) <= set(supplied_kinds(fields)), (
            f"{mode_id} suggests {other!r}, which is just as stuck: {message!r}"
        )


def supplied_kinds(fields) -> set:
    kinds = {kind for kind in ("image", "video", "audio") if fields.get(kind) is not None}
    if str(fields.get("prompt") or "").strip():
        kinds.add("text")
    return kinds


def test_the_missing_input_sentence_states_the_need_plainly(refused):
    error = refused(mode="text_to_video")
    assert str(error) == "Text to video needs a prompt."


def test_a_mode_is_suggested_only_when_what_is_supplied_runs_it(refused):
    # An image to video run with a prompt but no image: Text to video runs on the prompt.
    error = refused(mode="image_text_to_video", prompt="a lighthouse")
    assert str(error) == (
        "Image + text to video needs an image. To use what you have, switch to Text to video."
    )


def test_no_mode_is_suggested_when_nothing_supplied_would_run_one(refused):
    error = refused(mode="text_to_audio")
    assert str(error) == f"{modes.get_mode('text_to_audio').label} needs a prompt."


def test_a_blank_prompt_counts_as_missing(refused):
    error = refused(mode="text_to_video", prompt="   \n ")
    assert type(error).__name__ == "MissingInputError", f"{type(error).__name__}: {error}"


def test_every_missing_input_is_named_at_once(refused):
    error = refused(mode="image_text_to_video")
    message = str(error)
    assert "image" in message and "prompt" in message, message


def test_the_missing_input_error_crosses_the_fork_intact():
    import pickle

    error = errors.MissingInputError(
        mode="Image to video", missing=("image",), alternative="Text to video"
    )
    back = pickle.loads(pickle.dumps(error))
    assert type(back) is type(error) and str(back) == str(error)
    assert vars(back) == vars(error)
    assert isinstance(error, errors.ConstraintError)


# --------------------------------------------------------------------------
# F03 / F06: geometry and length, with the new upper bounds
# --------------------------------------------------------------------------


def test_an_illegal_width_is_refused_before_the_dispatch(refused):
    refused(errors.DimensionError, mode="text_to_video", prompt="p", width=700)


def test_an_illegal_frame_count_is_refused_before_the_dispatch(refused):
    refused(errors.FrameCountError, mode="text_to_video", prompt="p", num_frames=120)


def test_the_upper_bounds_exist_with_the_agreed_values():
    assert constraints.MAX_DIMENSION == 1920
    assert constraints.MAX_NUM_FRAMES == 481
    assert constraints.MAX_SECONDS == 20.0


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"width": 1952}, ("1952", "1920")),
        ({"height": 2048}, ("2048", "1920")),
        ({"num_frames": 489}, ("489", "481")),
        ({"auto_duration": True, "max_seconds": 25.0}, ("25", "20")),
    ],
    ids=["width", "height", "num_frames", "max_seconds"],
)
def test_a_request_over_an_upper_bound_is_refused_naming_the_limit(fields, named, refused):
    error = refused(ConstraintError, mode="text_to_video", prompt="p", **fields)
    message = str(error)
    for token in named:
        assert token in message, f"{token!r} missing from {message!r}"


@pytest.mark.parametrize(
    "fields",
    [
        {"width": 1920, "height": 1920},
        {"num_frames": 481},
        {"auto_duration": True, "max_seconds": 20.0},
    ],
    ids=["dimension", "frames", "seconds"],
)
def test_a_request_exactly_at_a_bound_runs(fields, accepted):
    result, _sent = accepted(mode="text_to_video", prompt="p", **fields)
    assert Path(result.output_path).exists()


def test_an_anchor_off_the_end_of_the_clip_is_refused_before_the_dispatch(refused):
    condition = ConditionSpec(source=image_input(), kind="image", index=200)
    refused(ConstraintError, mode="image_to_video", image=image_input(), conditions=(condition,))


# --------------------------------------------------------------------------
# F01's refusals: shot rows that carry text, and enhance with nothing to enhance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows",
    [
        [["a second shot from above", 9, 0.5]],
        (("a second shot", 9, 0.5),),
        {"headers": ["Prompt segment", "Anchor (frame)", "Strength"], "data": [["a shot", 9, 0.5]]},
    ],
    ids=["list", "tuple", "dataframe-payload"],
)
def test_shot_rows_that_carry_prompt_text_are_refused_in_plain_words(rows, refused):
    error = refused(ConstraintError, mode="image_to_video", image=image_input(), conditions=rows)
    message = str(error).lower()
    assert "prompt" in message and "shot" in message, message
    assert "frame" in message, message


def test_enhance_with_an_empty_prompt_is_refused_before_the_dispatch(refused, monkeypatch):
    called = []
    error = refused(
        ConstraintError,
        mode="video_to_video",
        video=video_file(),
        prompt="  ",
        enhance_prompt=True,
        enhancer=lambda prompt, image=None: called.append(prompt),
    )
    assert "prompt is empty" in str(error).lower(), str(error)
    assert called == []


# --------------------------------------------------------------------------
# F03: the audio file is checked web-side
# --------------------------------------------------------------------------


def test_an_audio_file_that_will_not_open_is_refused_before_the_dispatch(refused, tmp_path):
    bogus = tmp_path / "not-audio.wav"
    bogus.write_text("this is not a wav file")
    error = refused(ConstraintError, mode="audio_to_video", audio=str(bogus))
    assert "not-audio.wav" in str(error), str(error)


def test_an_audio_path_that_does_not_exist_is_refused_before_the_dispatch(refused, tmp_path):
    error = refused(ConstraintError, mode="audio_to_audio", audio=str(tmp_path / "gone.wav"))
    assert "gone.wav" in str(error), str(error)


def test_a_file_with_no_audio_stream_is_refused_before_the_dispatch(refused):
    error = refused(ConstraintError, mode="audio_to_video", audio=video_file())
    assert "audio stream" in str(error).lower(), str(error)


# --------------------------------------------------------------------------
# F03: staged LoRA files still exist
# --------------------------------------------------------------------------


def test_a_staged_lora_whose_file_has_gone_is_refused_before_the_dispatch(refused, tmp_path):
    gone = tmp_path / "adapter.safetensors"
    error = refused(
        LoRAError,
        mode="text_to_video",
        prompt="p",
        loras=({"source": "adapter.safetensors", "path": str(gone), "weight": 1.0},),
    )
    assert "adapter.safetensors" in str(error), str(error)


def test_a_kohya_named_adapter_is_refused_before_the_dispatch(refused, tmp_path):
    """F59: the loader cannot read lora_down/lora_up; that is said before a GPU is reserved."""
    from lora_fixtures import OFFICIAL_METADATA, REAL_MODULES, adapter_tensors, write_adapter

    path = write_adapter(
        tmp_path / "kohya.safetensors",
        adapter_tensors(REAL_MODULES[:3], style="kohya"),
        OFFICIAL_METADATA,
    )
    error = refused(LoRAError, mode="text_to_video", prompt="p", loras=({"source": str(path)},))

    assert "lora_down" in str(error) and "lora_A" in str(error), str(error)
    assert errors.categorise(error) == errors.Failure(errors.ADAPTER, False)


# --------------------------------------------------------------------------
# F05: when audio is the input, the audio sets the length
# --------------------------------------------------------------------------


@pytest.fixture
def loads_as_probed(monkeypatch):
    """The worker-side loader returns a waveform as long as the file is."""
    from ltx25_studio import audio as audio_module

    def load(seconds):
        monkeypatch.setattr(
            audio_module,
            "load_waveform",
            lambda source: (waveform_of(2, samples=round(seconds * 16000)), 16000),
        )

    return load


@pytest.mark.parametrize(
    ("seconds", "frame_rate", "expected"),
    [
        (8.16, 24.0, 193),  # the operator's upload
        (7.96, 24.0, 185),
        (7.96, 25.0, 193),
        (193 / 24.0, 24.0, 193),  # exactly on a legal length
        (193 / 24.0 - 0.002, 24.0, 185),  # a hair short of it
    ],
)
def test_auto_duration_in_an_audio_mode_takes_the_length_from_the_audio(
    seconds, frame_rate, expected, accepted, loads_as_probed
):
    loads_as_probed(seconds)
    result, sent = accepted(
        mode="audio_to_video",
        audio=audio_file(seconds),
        frame_rate=frame_rate,
        auto_duration=True,
    )
    assert sent.get("num_frames") == expected, sorted(sent)
    assert "min_seconds" not in sent and "max_seconds" not in sent, (
        "the text-only duration head cannot hear the audio and must not be asked"
    )
    notes = [n for n in result.warnings if "length set from your audio" in n]
    assert notes == [f"length set from your audio: {expected / frame_rate:.1f} s."], result.warnings


def test_audio_longer_than_the_maximum_is_capped_and_says_so(accepted, loads_as_probed):
    loads_as_probed(25.0)
    result, sent = accepted(mode="audio_to_audio", audio=audio_file(25.0), auto_duration=True)
    assert sent["num_frames"] == constraints.MAX_NUM_FRAMES
    note = next(n for n in result.warnings if "length set from your audio" in n)
    assert note.startswith("length set from your audio: 20.0 s.") and "maximum" in note, note


#: Modes 10 and 11, the two that take their length from a recording ("Match my clip").
AUDIO_MODES = [("audio_to_audio", "Audio to audio"), ("audio_to_video", "Audio to video")]


def _too_short(seconds: str, label: str) -> str:
    return (
        f"Your audio is {seconds}, too short for {label}: the shortest recording it can use "
        f"is 0.4 s."
    )


@pytest.mark.parametrize(("mode_id", "label"), AUDIO_MODES)
def test_match_my_clip_refuses_a_recording_too_short_for_the_mode(refused, mode_id, label):
    """Never a one-frame video: refused before dispatch, in the words a set length gets."""
    matched = refused(
        errors.AudioVideoDesyncError, mode=mode_id, audio=audio_file(0.03), auto_duration=True
    )
    at_a_set_length = refused(
        errors.AudioVideoDesyncError, mode=mode_id, audio=audio_file(0.03), num_frames=121
    )

    assert str(matched) == str(at_a_set_length) == _too_short("0.03 s", label)
    assert errors.categorise(matched) == errors.Failure(errors.INPUT, False)


@pytest.mark.parametrize("mode_id", [mode_id for mode_id, _ in AUDIO_MODES])
def test_match_my_clip_runs_the_shortest_usable_recording_at_nine_frames(
    accepted, loads_as_probed, mode_id
):
    loads_as_probed(9 / 24.0)
    result, sent = accepted(mode=mode_id, audio=audio_file(9 / 24.0), auto_duration=True)

    assert sent["num_frames"] == 9
    assert "length set from your audio: 0.4 s." in result.warnings


@pytest.mark.parametrize(("mode_id", "label"), AUDIO_MODES)
def test_match_my_clip_refuses_a_recording_one_frame_shorter(refused, mode_id, label):
    error = refused(
        errors.AudioVideoDesyncError, mode=mode_id, audio=audio_file(8 / 24.0), auto_duration=True
    )

    assert str(error) == _too_short("0.3 s", label)


@pytest.fixture
def refused_in_the_worker(monkeypatch, tmp_path, audio_input_ready):
    """``generate()`` through an in-process dispatch; return what the worker refused with."""

    def go(**fields):
        real = generate_module.generate

        def dispatch(**kwargs):
            token = generate_module._IN_GPU_WORKER.set(True)
            try:
                return real(**kwargs)
            finally:
                generate_module._IN_GPU_WORKER.reset(token)

        monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
        with pytest.raises(LTXStudioError) as caught:
            generate_module.generate(
                seed=7,
                pipeline_factory=factory_returning(FakePipe()),
                encoder=one_muxed_file_encoder,
                output_root=tmp_path,
                retention_bytes=10**9,
                **fields,
            )
        return caught.value

    return go


@pytest.mark.parametrize(("mode_id", "label"), AUDIO_MODES)
def test_the_workers_backstop_refuses_a_short_decode_in_the_same_words(
    refused_in_the_worker, loads_as_probed, mode_id, label
):
    """The file says nine frames' worth (0.375 s); it decodes to 0.32 s, past the
    0.05 s the worker allows, so the worker refuses it as the gate would have."""
    loads_as_probed(0.32)
    error = refused_in_the_worker(mode=mode_id, audio=audio_file(9 / 24.0), auto_duration=True)

    assert type(error) is errors.AudioVideoDesyncError
    assert str(error) == _too_short("0.3 s", label)
    assert errors.categorise(error) == errors.Failure(errors.INPUT, True), "it ran in the worker"


@pytest.mark.parametrize("mode_id", [mode_id for mode_id, _ in AUDIO_MODES])
def test_a_decode_one_frame_short_is_within_the_workers_tolerance_and_runs(
    accepted, loads_as_probed, mode_id
):
    """0.333 s decoded against nine frames (0.375 s) is 0.042 s short: within 0.05 s."""
    loads_as_probed(8 / 24.0)
    _result, sent = accepted(mode=mode_id, audio=audio_file(9 / 24.0), auto_duration=True)

    assert sent["num_frames"] == 9


def test_auto_duration_with_pre_encoded_audio_latents_is_refused(refused):
    error = refused(
        ConstraintError,
        mode="audio_to_video",
        audio_latents=SimpleNamespace(shape=(1, 8, 121, 16)),
        auto_duration=True,
    )
    assert "num_frames" in str(error) or "frame count" in str(error).lower(), str(error)


class SizedOutput:
    def __init__(self, frames: int):
        import numpy as np

        self.frames = np.zeros((1, frames, 8, 8, 3), dtype=np.uint8)
        self.audio = ("sample",)


class ChoosingPipe(FakePipe):
    """Returns a clip of a length nobody asked for, as the duration head does."""

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SizedOutput(97)


def test_the_length_the_duration_head_chose_is_reported(tmp_path):
    pipe = ChoosingPipe()
    result = generate_module.generate(
        mode="text_to_video",
        prompt="p",
        seed=7,
        auto_duration=True,
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    notes = [n for n in result.warnings if "the model chose" in n]
    assert notes == ["the model chose the length: 4.0 s."], result.warnings


def test_no_chosen_length_note_when_the_user_set_the_length(tmp_path):
    result = generate_module.generate(
        mode="text_to_video",
        prompt="p",
        seed=7,
        pipeline_factory=factory_returning(FakePipe()),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    assert not any("the model chose" in n for n in result.warnings)


def test_under_the_duration_head_a_video_condition_is_decoded_up_to_max_seconds(accepted):
    """Not up to the hidden Frames box: the head may pick anything up to max_seconds.

    video_to_audio: its clip is a condition (video_to_video varies its clip instead).
    """
    _result, sent = accepted(
        mode="video_to_audio",
        video=video_file(frames=60),
        auto_duration=True,
        min_seconds=1.0,
        max_seconds=2.0,
        num_frames=17,
    )
    (condition,) = sent["conditions"]
    assert len(condition.source) == 48, (
        f"decoded {len(condition.source)} frames; under the duration head the cap is "
        f"max_seconds x frame_rate = 48"
    )


def test_under_the_duration_head_an_anchor_must_fit_the_shortest_clip_it_can_pick(refused):
    """min_seconds=1.0 at 24 fps: the head can return as few as 25 frames."""
    condition = ConditionSpec(source=image_input(), kind="image", index=30)
    refused(
        ConstraintError,
        mode="image_to_video",
        image=image_input(),
        conditions=(condition,),
        auto_duration=True,
        min_seconds=1.0,
        max_seconds=5.0,
    )


# --------------------------------------------------------------------------
# F12: spaces 0.52 wraps every worker exception
# --------------------------------------------------------------------------


class ZeroGPUTaskException(Exception):
    """Shaped like spaces 0.52.0's wrapper: the real error rides on ``.exception``."""

    def __init__(self, message, exception, traceback):
        super().__init__(message)
        self.message = message
        self.exception = exception
        self.traceback = traceback


def _dispatching(monkeypatch, raised):
    def dispatch(**kwargs):
        raise raised

    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)


def test_a_wrapped_typed_error_reaches_the_caller_as_itself(monkeypatch, tmp_path):
    inner = errors.DecodeError(mode="text_to_video", reason="the sampler produced NaNs")
    wrapper = ZeroGPUTaskException(message="DecodeError", exception=inner, traceback="...")
    _dispatching(monkeypatch, wrapper)

    with pytest.raises(errors.DecodeError) as caught:
        generate_module.generate(mode="text_to_video", prompt="p", output_root=tmp_path)

    assert caught.value is inner
    assert caught.value.__cause__ is wrapper


def test_a_wrapped_plain_error_reaches_the_caller_as_itself(monkeypatch, tmp_path):
    inner = RuntimeError("CUDA error: an illegal memory access was encountered")
    _dispatching(
        monkeypatch, ZeroGPUTaskException(message="RuntimeError", exception=inner, traceback="")
    )

    with pytest.raises(RuntimeError) as caught:
        generate_module.generate(mode="text_to_video", prompt="p", output_root=tmp_path)

    assert caught.value is inner


def test_the_handler_renders_the_unwrapped_reason(monkeypatch):
    from test_request_scoping import values_for

    from ltx25_studio.ui import handlers

    inner = errors.DecodeError(mode="text_to_video", reason="the sampler produced NaNs")
    _dispatching(
        monkeypatch, ZeroGPUTaskException(message="DecodeError", exception=inner, traceback="")
    )

    _video, _audio, status = handlers.on_generate("text_to_video", *values_for("text_to_video"))

    assert "the sampler produced NaNs" in status, status
    assert "ZeroGPUTaskException" not in status, status


@pytest.mark.parametrize("payload", [None, "DecodeError", 42])
def test_a_wrapper_without_a_real_exception_is_left_alone(payload, monkeypatch, tmp_path):
    wrapper = ZeroGPUTaskException(message="X", exception=payload, traceback="")
    _dispatching(monkeypatch, wrapper)

    with pytest.raises(ZeroGPUTaskException) as caught:
        generate_module.generate(mode="text_to_video", prompt="p", output_root=tmp_path)

    assert caught.value is wrapper


# --------------------------------------------------------------------------
# the video decode is handed the run's frame rate, when it takes one
# --------------------------------------------------------------------------


def _spy_decoder(monkeypatch, *, takes_frame_rate):
    import numpy as np

    from ltx25_studio import conditioning

    seen: list[dict] = []
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    if takes_frame_rate:

        def decode(source, *, max_frames=None, frame_rate=None):
            seen.append({"max_frames": max_frames, "frame_rate": frame_rate})
            return frames
    else:

        def decode(source, *, max_frames=None):
            seen.append({"max_frames": max_frames})
            return frames

    monkeypatch.setattr(conditioning, "decode_video_frames", decode)
    return seen


@pytest.mark.parametrize("takes_frame_rate", [True, False])
def test_the_video_decode_gets_the_frame_rate_only_if_it_takes_one(
    takes_frame_rate, monkeypatch, accepted
):
    seen = _spy_decoder(monkeypatch, takes_frame_rate=takes_frame_rate)
    accepted(mode="video_to_audio", video=video_file(), frame_rate=25.0, num_frames=97)

    assert len(seen) == 1, seen
    assert seen[0]["max_frames"] == 97
    if takes_frame_rate:
        assert seen[0]["frame_rate"] == 25.0
    else:
        assert "frame_rate" not in seen[0]


# --------------------------------------------------------------------------
# the gate, by its public name, for the UI's readiness check
# --------------------------------------------------------------------------


def test_the_public_gate_refuses_a_request_missing_a_required_input():
    request = generate_module.GenerationRequest(mode="image_to_video")
    with pytest.raises(errors.MissingInputError) as caught:
        generate_module.prepare_request(request)
    assert "Image to video" in str(caught.value) and "image" in str(caught.value)


def test_the_public_gate_returns_the_scoped_request_and_its_notes():
    request = generate_module.GenerationRequest(
        mode="text_to_video", prompt="a lighthouse", image=image_input()
    )
    scoped, spec, notes = generate_module.prepare_request(request)
    assert spec.id == "text_to_video"
    assert scoped.image is None
    assert len(notes) == 1 and "Text to video" in notes[0], notes
    assert "prepare_request" in generate_module.__all__
    assert generate_module._prepare_request is generate_module.prepare_request


# --------------------------------------------------------------------------
# review findings: refusals that cost a GPU slot (request-path-1, space-path-1,
# media-io-2) and gate checks over values a run never uses (request-path-3..6)
# --------------------------------------------------------------------------


def _wav(path: Path, *, seconds: float, channels: int = 2, rate: int = 48000) -> str:
    import wave

    import numpy as np

    count = int(round(seconds * rate))
    samples = (np.sin(np.arange(count) / 10.0)[:, None] * 8000).repeat(channels, axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.astype("<i2").tobytes())
    return str(path)


@pytest.mark.parametrize("mode_id", ["audio_to_video", "audio_to_audio"])
def test_audio_shorter_than_a_set_length_is_refused_before_dispatch(mode_id, refused, tmp_path):
    error = refused(
        errors.AudioVideoDesyncError,
        mode=mode_id,
        audio=_wav(tmp_path / "short.wav", seconds=2.0),
        num_frames=121,
    )
    assert error.gpu_used is False
    assert error.largest_fitting_num_frames == 41
    # UI review 2, B-2: the UI's words, and the longest set length that fits.
    assert str(error) == (
        'Your audio is 2.0 s and the length is set to 5.0 s. Choose "Match my clip", '
        "or set 1.7 s or less."
    )


@pytest.mark.parametrize("mode_id", ["audio_to_video", "audio_to_audio"])
def test_a_four_second_wav_against_a_set_five_seconds_is_refused_for_free(
    mode_id, refused, tmp_path
):
    error = refused(
        errors.AudioVideoDesyncError,
        mode=mode_id,
        audio=_wav(tmp_path / "four.wav", seconds=4.0),
        num_frames=121,
    )
    assert error.gpu_used is False
    assert str(error) == (
        'Your audio is 4.0 s and the length is set to 5.0 s. Choose "Match my clip", '
        "or set 3.7 s or less."
    )


def test_the_set_length_boundary_exact_fit_runs_and_the_next_length_is_refused(
    accepted, refused, tmp_path
):
    # 97 frames at 24 fps is 4.0417 s: audio of exactly that length fits.
    accepted(
        mode="audio_to_video", audio=_wav(tmp_path / "fit.wav", seconds=97 / 24), num_frames=97
    )
    # The next legal length (105 frames, 4.375 s) is past it by more than the tolerance.
    error = refused(
        errors.AudioVideoDesyncError,
        mode="audio_to_video",
        audio=_wav(tmp_path / "over.wav", seconds=97 / 24),
        num_frames=105,
    )
    assert error.largest_fitting_num_frames == 97


def test_audio_just_inside_the_tolerance_is_not_refused(accepted, tmp_path):
    accepted(mode="audio_to_video", audio=_wav(tmp_path / "ok.wav", seconds=4.995), num_frames=121)


def test_audio_with_more_than_two_channels_is_refused_before_dispatch(refused, tmp_path):
    error = refused(
        ConstraintError,
        mode="audio_to_video",
        audio=_wav(tmp_path / "six.wav", seconds=6.0, channels=6),
        num_frames=121,
    )
    assert "6 audio channels" in str(error)


def test_a_missing_video_is_refused_before_dispatch_in_a_condition_mode(refused, tmp_path):
    error = refused(ConstraintError, mode="video_to_audio", video=str(tmp_path / "gone.mp4"))
    assert "could not be opened as a video" in str(error)


@pytest.mark.parametrize("mode_id", ["video_to_video", "video_to_audio"])
def test_a_sound_file_given_as_the_video_is_refused_before_dispatch(mode_id, refused, tmp_path):
    """Review request-path-2: a WAV in the video slot opens, and has no picture."""
    error = refused(ConstraintError, mode=mode_id, video=_wav(tmp_path / "talk.wav", seconds=2.0))
    assert str(error).startswith("talk.wav has no picture in it"), str(error)
    assert error.gpu_used is False


def test_audio_matched_length_ignores_the_hidden_seconds_fields(accepted, tmp_path):
    """The length comes from the audio; min/max seconds belong to the text modes."""
    accepted(
        mode="audio_to_video",
        audio=_wav(tmp_path / "five.wav", seconds=5.0),
        auto_duration=True,
        min_seconds=30.0,
        max_seconds=25.0,
    )


@pytest.mark.parametrize("mode_id", ["text_to_audio", "audio_to_audio"])
def test_a_pinned_size_mode_ignores_a_hidden_size_over_the_limit(mode_id, accepted):
    _result, sent = accepted(mode=mode_id, **inputs_for(mode_id), width=2048, height=2048)
    assert (sent["width"], sent["height"]) == (modes.MIN_RESOLUTION, modes.MIN_RESOLUTION)


def test_a_required_image_given_as_a_keyframe_is_not_a_missing_input(accepted):
    condition = ConditionSpec(source=image_input(), kind="image", index=16)
    _result, sent = accepted(mode="image_to_video", conditions=(condition,), num_frames=17)
    assert [c.kind for c in sent["conditions"]] == ["image"]


@pytest.mark.parametrize(
    "field,value,limit",
    [("width", 1950, 1920), ("height", 1921, 1920), ("num_frames", 485, 481)],
)
def test_past_the_limit_the_refusal_names_only_a_legal_value(refused, field, value, limit):
    error = refused(errors.OverLimitError, mode="text_to_video", prompt="p", **{field: value})
    assert error.nearest == limit, error
    assert str(limit + (32 if field != "num_frames" else 8)) not in str(error)


@pytest.mark.parametrize("seed", [2**64, 2**32, -2])
def test_a_seed_out_of_range_is_refused_before_dispatch(seed):
    """torch's generator raises an untyped overflow in the worker past 2**64."""
    with pytest.raises(ConstraintError) as caught:
        generate_module.prepare_request(
            generate_module.GenerationRequest(mode="text_to_video", prompt="p", seed=seed)
        )
    assert "seed" in str(caught.value) and str(generate_module.MAX_SEED) in str(caught.value)


# --------------------------------------------------------------------------
# call sites asked for by the lora / conditioning / audio owner
# --------------------------------------------------------------------------


def _staged_entry(path):
    Path(path).write_bytes(b"x")
    return {"source": str(path), "path": str(path), "weight": 1.0, "report": object()}


def test_a_staged_adapter_is_marked_in_use_at_the_gate(monkeypatch, tmp_path):
    """lora-5: eviction cannot remove an adapter a queued run is about to load."""
    from ltx25_studio import lora

    marked = []
    monkeypatch.setattr(lora, "mark_used", lambda path: marked.append(str(path)) or True)
    entry = _staged_entry(tmp_path / "a.safetensors")

    generate_module.prepare_request(
        generate_module.GenerationRequest(mode="text_to_video", prompt="p", loras=(entry,))
    )

    assert marked == [entry["path"]]


def test_an_unvalidated_adapter_is_validated_before_dispatch(refused, tmp_path):
    """Validation hashes the whole file; it belongs before the GPU window."""
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"x")
    refused(LoRAError, mode="text_to_video", prompt="p", loras=(str(junk),))


def test_a_validated_adapter_carries_its_report_into_the_worker(monkeypatch, tmp_path):
    from ltx25_studio import lora

    report = object()
    monkeypatch.setattr(lora, "validate_lora", lambda source, **kwargs: report)
    path = tmp_path / "b.safetensors"
    path.write_bytes(b"x")

    scoped, _spec, _notes = generate_module.prepare_request(
        generate_module.GenerationRequest(mode="text_to_video", prompt="p", loras=(str(path),))
    )

    ((entry,),) = [scoped.loras]
    assert entry["report"] is report and entry["path"] == str(path)


def test_a_video_condition_is_decoded_at_the_generation_size_and_its_notes_reach_the_run(
    monkeypatch, accepted
):
    """media-io-3/4: reduced while decoding; an HDR clip's note is told to the user."""
    import numpy as np

    from ltx25_studio import conditioning

    seen = []

    def decode(source, *, max_frames=None, frame_rate=None, notes=None, size=None):
        seen.append(size)
        notes.append("your clip is HDR; it was converted to SDR colour.")
        return np.zeros((2, 8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(conditioning, "decode_video_frames", decode)
    result, _sent = accepted(
        mode="video_to_audio", video=video_file(), num_frames=9, enable_stage_2=False
    )

    assert seen == [(modes.MIN_RESOLUTION, modes.MIN_RESOLUTION)]
    assert any("HDR" in note for note in result.warnings), result.warnings


def test_an_audio_upload_is_decoded_only_as_far_as_the_run_uses(monkeypatch, tmp_path):
    """media-io-5: an hour-long upload to a five-second run is not decoded whole."""
    from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM, FakeAudioVae

    from ltx25_studio import audio as audio_module

    asked = {}

    def load(source, *, max_seconds=None):
        asked["max_seconds"] = max_seconds
        return STUB_WAVEFORM, STUB_SAMPLING_RATE

    monkeypatch.setattr(audio_module, "load_waveform", load)
    pipe = FakePipe()
    pipe.audio_vae = FakeAudioVae()
    generate_module.generate(
        mode="audio_to_video",
        seed=7,
        audio=_wav(tmp_path / "long.wav", seconds=20.0),
        num_frames=17,
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path / "out",
        retention_bytes=10**9,
    )

    assert asked["max_seconds"] == pytest.approx(audio_module.upload_seconds_needed(17, 24.0))
