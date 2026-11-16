"""Spec for the run record and the failure categories — what the UI renders from facts.

``GenerationResult.record`` is a frozen ``RunRecord`` of builtins that crosses
the ZeroGPU fork; every failure carries a stable ``category`` and ``gpu_used``,
and ``errors.categorise`` classifies anything that reaches the UI, ours or the
platform's. Each category is produced here by its real trigger through
``generate()`` with fakes.
"""

from __future__ import annotations

import dataclasses
import pickle

import pytest
from conftest import inputs_for
from test_generate import (
    FakePipe,
    LoRAFakePipe,
    _staged,
    a_cuda,
    a_request,
    factory_for,
    good_encoder,
    install_fake_torch,
)

from ltx25_studio import errors
from ltx25_studio import generate as generate_module
from ltx25_studio.errors import categorise


class Shaped:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __len__(self):
        return self.shape[0]


class Output:
    """Decoded frames as an array ``[B, F, H, W, C]``, with an audio track."""

    def __init__(self, frames=9, height=64, width=96):
        self.frames = Shaped((1, frames, height, width, 3))
        self.audio = ("sample",)


def run(tmp_path, request=None, *, pipe=None, **seams):
    return generate_module.generate(
        request or a_request(width=96, height=64, num_frames=9),
        pipeline_factory=factory_for(pipe or FakePipe(output=Output())),
        encoder=good_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
        **seams,
    )


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def test_the_record_carries_what_actually_happened(tmp_path):
    result = run(tmp_path, enhancer=lambda prompt, image=None: "ENHANCED " + prompt, **{})

    record = result.record
    assert isinstance(record, generate_module.RunRecord)
    assert record.mode == "text_to_video"
    assert (record.requested_width, record.requested_height) == (96, 64)
    assert (record.width, record.height) == (96, 64)
    assert record.frames == 9 and record.frame_rate == 24.0 and record.seconds == 0.375
    assert record.seed == 7 and record.stage_2 is False
    assert record.length_source == generate_module.LENGTH_FROM_REQUEST
    assert record.adapters == () and record.audio_level is None
    assert record.elapsed_seconds >= 0 and record.peak_vram_gib is None


def test_the_record_holds_the_prompt_that_was_sent_after_enhancement(tmp_path):
    result = run(
        tmp_path,
        a_request(width=96, height=64, num_frames=9, enhance_prompt=True),
        enhancer=lambda prompt, image=None: "ENHANCED " + prompt,
    )

    assert result.record.prompt == "ENHANCED a lighthouse in fog"


def test_the_record_survives_pickling_and_holds_only_builtins(tmp_path):
    record = run(tmp_path).record

    assert pickle.loads(pickle.dumps(record)) == record

    def builtin(value):
        if isinstance(value, (tuple, list)):
            return all(builtin(item) for item in value)
        return value is None or isinstance(value, (str, int, float, bool))

    for field in dataclasses.fields(record):
        assert builtin(getattr(record, field.name)), field.name


def test_the_whole_result_survives_pickling(tmp_path):
    result = run(tmp_path)
    assert pickle.loads(pickle.dumps(result)).record == result.record


def test_the_length_chosen_by_the_model_is_recorded(tmp_path):
    result = run(tmp_path, a_request(auto_duration=True, min_seconds=1.0, max_seconds=3.0))
    assert result.record.length_source == generate_module.LENGTH_FROM_MODEL


@pytest.mark.usefixtures("audio_input_ready")
def test_the_length_derived_from_the_users_audio_is_recorded(tmp_path):
    from conftest import FakeAudioVae

    pipe = FakePipe(output=Output())
    pipe.audio_vae = FakeAudioVae()
    result = run(
        tmp_path,
        generate_module.GenerationRequest(
            mode="audio_to_video", seed=7, auto_duration=True, **inputs_for("audio_to_video")
        ),
        pipe=pipe,
    )

    assert result.record.length_source == generate_module.LENGTH_FROM_AUDIO
    assert result.record.audio_level == 0.0


def test_adapters_are_recorded_with_label_weight_and_attached_count(tmp_path):
    pipe = LoRAFakePipe(output=Output())

    result = run(
        tmp_path,
        a_request(
            width=96, height=64, num_frames=9, loras=(_staged(tmp_path / "Bts.safetensors", 0.8),)
        ),
        pipe=pipe,
    )

    assert result.record.adapters == (("Bts", 0.8, 1),)


def test_the_runner_says_which_decoder_ran_and_whether_stage_2_ran(tmp_path):
    from test_decode_policy import OK, runner_on

    runner, _pipeline, _built = runner_on("cuda")
    result = generate_module.generate(
        mode="text_to_video",
        seed=7,
        prompt="p",
        **OK,
        pipeline_factory=lambda **_: runner,
        encoder=good_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    assert result.record.decoder == "diffusion" and result.record.stage_2 is False


def test_peak_vram_is_recorded_where_it_was_measured(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch, a_cuda())
    assert run(tmp_path).record.peak_vram_gib == 73.0


def test_a_smaller_gpu_slice_than_asked_for_is_said_out_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("LTX25_ZEROGPU_SIZE", "xlarge")
    install_fake_torch(monkeypatch, a_cuda(total=47 * 1024**3))

    result = run(tmp_path)

    assert any("xlarge was requested" in line for line in result.record.diagnostics)
    assert not any("xlarge was requested" in note for note in result.warnings), result.warnings


def test_diagnostics_hold_the_run_folder_and_stay_out_of_the_notes(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch, a_cuda())

    result = run(tmp_path)

    diagnostics = result.record.diagnostics
    assert f"run folder: {result.run_dir}" in diagnostics
    assert any(line.startswith("peak VRAM, whole call") for line in diagnostics)
    assert set(diagnostics).isdisjoint(result.warnings)
    assert all(isinstance(line, str) for line in diagnostics)


# --------------------------------------------------------------------------
# a category on every failure, from its real trigger
# --------------------------------------------------------------------------


def test_an_out_of_memory_in_the_denoise_is_oom(tmp_path):
    with pytest.raises(errors.GpuMemoryError) as caught:
        run(tmp_path, pipe=FakePipe(raises=RuntimeError("CUDA out of memory. Tried to allocate")))
    assert categorise(caught.value) == errors.Failure(errors.OOM, True)


def test_an_mps_out_of_memory_is_oom_too(tmp_path):
    with pytest.raises(errors.GpuMemoryError):
        run(
            tmp_path, pipe=FakePipe(raises=RuntimeError("MPS backend out of memory (MPS allocated"))
        )


def test_a_refusal_before_dispatch_is_input_and_used_no_gpu(tmp_path):
    with pytest.raises(errors.FrameCountError) as caught:
        run(tmp_path, a_request(num_frames=120))
    assert categorise(caught.value) == errors.Failure(errors.INPUT, False)


def test_a_refusal_before_a_registered_dispatch_never_reaches_it(tmp_path, monkeypatch):
    dispatched = []
    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", lambda **kw: dispatched.append(kw))

    with pytest.raises(errors.MissingInputError) as caught:
        generate_module.generate(mode="image_to_video", seed=7)

    assert not dispatched
    assert categorise(caught.value) == errors.Failure(errors.INPUT, False)


def test_a_lora_that_does_not_attach_is_adapter(tmp_path):
    bad = tmp_path / "b.safetensors"
    with pytest.raises(errors.LoRANotAttachedError) as caught:
        run(
            tmp_path,
            a_request(loras=(_staged(bad),)),
            pipe=LoRAFakePipe(zero_attach_sources=(bad,)),
        )
    assert categorise(caught.value) == errors.Failure(errors.ADAPTER, True)


def test_an_enhancer_failure_is_enhancer(tmp_path):
    with pytest.raises(errors.PromptEnhancementError) as caught:
        run(tmp_path, a_request(enhance_prompt=True), enhancer=lambda prompt, image=None: "")
    assert categorise(caught.value).category == errors.ENHANCER


@pytest.mark.parametrize(
    "title,message,expected",
    [
        (
            "ZeroGPU quota exceeded",
            "You have exceeded your GPU quota (120s requested vs. 30s left).",
            errors.Failure(errors.QUOTA, False),
        ),
        (
            "ZeroGPU illegal duration",
            "The requested GPU duration (900s) is larger than the maximum allowed",
            errors.Failure(errors.LIMIT, False),
        ),
        ("ZeroGPU worker error", "GPU task aborted", errors.Failure(errors.TIMEOUT, True)),
        ("ZeroGPU queue timeout", "No GPU was available", errors.Failure(errors.QUEUE, False)),
        ("ZeroGPU client error", "Expired ZeroGPU proxy token", errors.Failure(errors.BUG, True)),
    ],
)
def test_the_platforms_errors_are_recognised_at_the_ui_boundary(
    tmp_path, monkeypatch, title, message, expected
):
    """spaces 0.51 raises ``gr.Error(message, title=...)``; generate() lets it through."""
    import gradio as gr

    def dispatch(**_kwargs):
        raise gr.Error(message, title=title)

    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
    with pytest.raises(gr.Error) as caught:
        generate_module.generate(a_request())

    assert categorise(caught.value) == expected


@pytest.mark.parametrize(
    "title,message,expected",
    [
        (
            "ZeroGPU illegal duration",
            "The requested GPU duration (900s) is larger than the maximum allowed",
            errors.Failure(errors.LIMIT, False),
        ),
        ("ZeroGPU queue timeout", "No GPU was available", errors.Failure(errors.QUEUE, False)),
    ],
)
def test_the_limit_and_the_queue_are_recognised_in_the_spaces_052_wrapped_form(
    title, message, expected
):
    """spaces 0.52 wraps the platform's ``gr.Error`` in a ZeroGPUTaskException."""
    import gradio as gr

    class ZeroGPUTaskException(Exception):
        def __init__(self, exception):
            super().__init__("task failed")
            self.exception = exception

    assert categorise(ZeroGPUTaskException(gr.Error(message, title=title))) == expected


#: (title, message, category): the platform's wordings as seen on this Space.
PLATFORM_WORDINGS = [
    (
        "ZeroGPU quota exceeded",
        "You have exceeded your Pro ZeroGPU quota (360s requested vs. "
        "118s left). Try again in 5:41:07.",
        errors.QUOTA,
    ),
    (
        "ZeroGPU quota exceeded",
        "You have exceeded your free ZeroGPU quota (120s requested vs. "
        "30s left). Try again in 0:01:00.",
        errors.QUOTA,
    ),
    ("ZeroGPU quota exceeded", "You have exceeded your ZeroGPU runs limit. ", errors.QUOTA),
    ("ZeroGPU quota exceeded", "Space app has reached its GPU limit. Retry later", errors.QUOTA),
    (
        "ZeroGPU illegal duration",
        "The requested GPU duration (1020s) is larger than the maximum allowed",
        errors.LIMIT,
    ),
    ("ZeroGPU queue timeout", "No GPU was available after 60s. Retry later", errors.QUEUE),
]


@pytest.mark.parametrize(("title", "message", "category"), PLATFORM_WORDINGS)
@pytest.mark.parametrize("form", ["title and message", "message only", "title only", "0.52"])
def test_the_platforms_refusals_are_recognised_by_title_or_message(title, message, category, form):
    """An API client receives only the MESSAGE; the UI sees the title too. Either is enough.

    Nothing ran on any of these, so no GPU time was used.
    """
    import gradio as gr

    class ZeroGPUTaskException(Exception):
        def __init__(self, exception):
            super().__init__("task failed")
            self.exception = exception

    if form == "title and message":
        raised = gr.Error(message, title=title)
    elif form == "message only":
        raised = gr.Error(message)
    elif form == "title only":
        raised = gr.Error("", title=title)
    else:
        raised = ZeroGPUTaskException(gr.Error(message, title=title))

    assert categorise(raised) == errors.Failure(category, False)


@pytest.mark.parametrize(("title", "message", "category"), PLATFORM_WORDINGS)
def test_the_platforms_words_match_in_any_case(title, message, category):
    assert categorise(RuntimeError(message.upper())) == errors.Failure(category, False)
    assert categorise(RuntimeError(title.lower())) == errors.Failure(category, False)


def test_the_new_categories_are_appended_so_existing_ones_keep_their_place():
    assert errors.CATEGORIES[:7] == (
        errors.QUOTA,
        errors.TIMEOUT,
        errors.OOM,
        errors.INPUT,
        errors.ADAPTER,
        errors.ENHANCER,
        errors.BUG,
    )
    assert errors.CATEGORIES[7:] == ("limit", "queue")


def test_a_foreign_exception_of_ours_is_a_bug():
    assert categorise(KeyError("x")) == errors.Failure(errors.BUG, True)


def test_categorise_unwraps_the_spaces_052_task_exception():
    class ZeroGPUTaskException(Exception):
        def __init__(self, exception):
            super().__init__("task failed")
            self.exception = exception

    wrapped = ZeroGPUTaskException(
        errors.DecoderTooSmallError(
            output=(64, 64, 9), minimum=(224, 224, 17), smallest=(128, 128, 17), stage_2=True
        )
    )
    assert categorise(wrapped).category == errors.INPUT


@pytest.mark.parametrize("gpu_used", [True, False])
def test_categorise_round_trips_the_fork(gpu_used):
    exc = errors.GpuMemoryError(
        mode="text_to_video", stage="generating", reason="CUDA out of memory"
    )
    exc.gpu_used = gpu_used

    back = pickle.loads(pickle.dumps(exc))

    assert categorise(back) == categorise(exc) == errors.Failure(errors.OOM, gpu_used)


def test_every_family_has_a_known_category():
    for name in errors.__all__:
        cls = getattr(errors, name)
        if isinstance(cls, type) and issubclass(cls, errors.LTXStudioError):
            assert cls.category in errors.CATEGORIES, name


# --------------------------------------------------------------------------
# small items on the same seam
# --------------------------------------------------------------------------


def test_a_device_refusal_from_the_factory_stays_typed(tmp_path):
    """Without the GPU entrypoint the factory's own typed error must not become ModelLoadError."""
    from ltx25_studio.device import GenerationDisabledError

    def factory(**_kwargs):
        raise GenerationDisabledError(reason="no device")

    with pytest.raises(GenerationDisabledError):
        generate_module.generate(
            a_request(), pipeline_factory=factory, encoder=good_encoder, output_root=tmp_path
        )


def test_an_unknown_progress_label_degrades_to_a_neutral_phase():
    seen = []
    reporter = generate_module.ProgressReporter(
        lambda position, desc=None: seen.append((position, desc))
    )

    reporter.phase("loading the model")
    reporter.phase("a phase nobody declared")

    assert seen[-1] == (seen[-2][0], "a phase nobody declared")


def test_prepare_request_refuses_a_stage_2_request_this_device_cannot_decode(monkeypatch):
    from ltx25_studio import device

    resolution = dataclasses.replace(device.resolve(None), device="cpu")
    monkeypatch.setattr(device, "current", lambda: resolution)

    with pytest.raises(device.StageTwoNeedsCudaError) as caught:
        generate_module.prepare_request(a_request(enable_stage_2=True))
    assert categorise(caught.value) == errors.Failure(errors.INPUT, True)

    cuda = dataclasses.replace(resolution, device="cuda")
    monkeypatch.setattr(device, "current", lambda: cuda)
    generate_module.prepare_request(a_request(enable_stage_2=True))


def test_adapters_are_attached_with_their_label_and_after_the_transformer_is_resident(
    tmp_path, monkeypatch
):
    from ltx25_studio.lora import apply as lora_apply

    calls = []
    real = lora_apply.apply_lora

    def spy(*args, **kwargs):
        calls.append(kwargs.get("label"))
        return real(*args, **kwargs)

    monkeypatch.setattr(lora_apply, "apply_lora", spy)
    resident = []

    class StagedRunner(LoRAFakePipe):
        def ensure_transformer(self):
            resident.append(True)

    run(
        tmp_path,
        a_request(
            width=96, height=64, num_frames=9, loras=(_staged(tmp_path / "Look.safetensors"),)
        ),
        pipe=StagedRunner(output=Output()),
    )

    assert calls == ["Look"] and resident == [True]


def test_an_unvalidated_adapter_on_an_overridden_model_is_checked_without_the_manifest(
    tmp_path, monkeypatch
):
    from ltx25_studio import config
    from ltx25_studio import lora as lora_package

    (tmp_path / "model_index.json").write_text("{}")
    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tmp_path))
    seen = {}

    def validate(source, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(lora_package, "validate_lora", validate)
    adapter = tmp_path / "raw.safetensors"
    adapter.touch()

    run(
        tmp_path / "out",
        a_request(width=96, height=64, num_frames=9, loras=(str(adapter),)),
        pipe=LoRAFakePipe(output=Output()),
    )

    assert "manifest" in seen and seen["manifest"] is None


def test_the_lora_dimension_error_is_true_at_load_time_too():
    message = str(
        errors.LoRADimensionError(source="x.safetensors", module="m", expected=4096, actual=2048)
    )
    assert "applying it would corrupt" not in message
    assert "refused" in message
