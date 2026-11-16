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
    others = [modes.get_mode(m).label for m in MODE_IDS if m != mode_id]
    assert any(label in message for label in others), (
        f"the error gives no way out: it should name a mode that does not need it: {message!r}"
    )


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
    assert len(notes) == 1, result.warnings
    assert f"{expected} frames" in notes[0], notes[0]


def test_audio_longer_than_the_maximum_is_capped_and_says_so(accepted, loads_as_probed):
    loads_as_probed(25.0)
    result, sent = accepted(mode="audio_to_audio", audio=audio_file(25.0), auto_duration=True)
    assert sent["num_frames"] == constraints.MAX_NUM_FRAMES
    note = next(n for n in result.warnings if "length set from your audio" in n)
    assert "481 frames" in note and "20" in note, note


def test_very_short_audio_is_clamped_to_the_minimum_length(accepted, loads_as_probed):
    loads_as_probed(0.03)
    _result, sent = accepted(mode="audio_to_video", audio=audio_file(0.03), auto_duration=True)
    assert sent["num_frames"] == 1


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
    assert notes == ["the model chose 97 frames (4.04 s)."], result.warnings


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
