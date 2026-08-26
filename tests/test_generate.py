"""Spec for ``ltx25_studio/generate.py`` — PLAN §5.3, §7.3, §9.3.2.

**The invariant this whole build is judged on.**

The 2.3 outage: a mid-graph load failure returned an empty video path, the UI
rendered an empty path as a silent form reset, and the user saw "nothing
happened". No error, no log, no signal. It hid for a week.

So: *a generation that completes without producing a file raises a specific,
actionable, typed error.* Never ``None``. Never ``""``. Never a success-shaped
result carrying an empty path. Structurally — a success path that can return a
falsy artifact is a bug whether or not it is currently reachable.

The contract:

* ``generate(request, *, pipeline_factory=None, encoder=None, output_root=None)
  -> GenerationResult``
* ``GenerationResult`` carries ``output_path`` (the artifact the user receives),
  ``run_id``, ``run_dir`` and ``mode``.
* ``pipeline_factory`` and ``encoder`` are injected seams so this file can test
  failure at every depth with no GPU and no weights. The only pinned detail of
  their signatures is that ``generate`` passes ``run_dir=`` to the encoder as a
  keyword argument — everything else the fakes accept positionally or not at all.
* Each run writes into its **own run-scoped subdirectory** of ``output_root``.
  There is no "newest file in the output directory" fallback: that helper has no
  run correlation and will happily hand a failed run the *previous* run's video.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import PACKAGE_ROOT, code_string_constants, code_tokens

from ltx25_studio import generate as generate_module
from ltx25_studio.errors import (
    AudioVideoDesyncError,
    DecodeError,
    EmptyOutputError,
    FrameCountError,
    GenerationError,
    LoRANotAttachedError,
    LTXStudioError,
    ModelLoadError,
    UnknownModeError,
)


def generate(*args, **kwargs):
    """Late-bound so a missing ``generate()`` fails the tests that use it rather
    than aborting collection for the whole file."""
    return generate_module.generate(*args, **kwargs)


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeOutput:
    """Duck-types ``LTX2PipelineOutput(frames=..., audio=...)``."""

    def __init__(self, frames=("frame",), audio=("sample",)):
        self.frames = frames
        self.audio = audio


class NothingOutput:
    """A pipeline that ran, returned an output object, and decoded nothing."""

    frames = None
    audio = None


class FakePipe:
    def __init__(self, output=None, raises=None):
        self._output = FakeOutput() if output is None else output
        self._raises = raises
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._raises is not None:
            raise self._raises
        return self._output


def factory_for(pipe):
    def _factory(*args, **kwargs):
        return pipe

    return _factory


def exploding_factory(exc):
    def _factory(*args, **kwargs):
        raise exc

    return _factory


def _run_dir_of(kwargs):
    if "run_dir" not in kwargs:
        pytest.fail(
            "generate() must pass run_dir= to the encoder as a keyword argument, so every "
            "artifact is written inside a run-scoped directory (PLAN §5.3)"
        )
    return Path(kwargs["run_dir"])


def good_encoder(*args, **kwargs):
    run_dir = _run_dir_of(kwargs)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "output.mp4"
    target.write_bytes(b"\x00" * 2048)
    return target


def encoder_returning_a_path_it_never_wrote(*args, **kwargs):
    return _run_dir_of(kwargs) / "output.mp4"


def encoder_writing_a_zero_byte_file(*args, **kwargs):
    run_dir = _run_dir_of(kwargs)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "output.mp4"
    target.write_bytes(b"")
    return target


def encoder_returning_none(*args, **kwargs):
    _run_dir_of(kwargs)
    return None


def encoder_returning_empty_string(*args, **kwargs):
    _run_dir_of(kwargs)
    return ""


def encoder_raising(*args, **kwargs):
    _run_dir_of(kwargs)
    raise OSError("ffmpeg exited 1")


def a_request(**overrides):
    fields = {
        "mode": "text_to_video",
        "prompt": "a lighthouse in fog",
        "width": 704,
        "height": 480,
        "num_frames": 121,
        "seed": 7,
    }
    fields.update(overrides)
    return generate_module.GenerationRequest(**fields)


# --------------------------------------------------------------------------
# the happy path, so the failure specs mean something
# --------------------------------------------------------------------------


def test_a_successful_run_returns_a_real_file(tmp_path):
    result = generate(
        a_request(),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert result.output_path
    path = Path(result.output_path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_a_successful_run_writes_into_its_own_run_directory(tmp_path):
    result = generate(
        a_request(),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    path = Path(result.output_path)
    assert path.parent != tmp_path, "outputs must be run-scoped, not dumped in the root"
    assert tmp_path in path.parents


def test_two_runs_do_not_share_a_directory(tmp_path):
    first = generate(
        a_request(),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    second = generate(
        a_request(),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert Path(first.run_dir) != Path(second.run_dir)
    assert first.run_id != second.run_id
    assert Path(first.output_path).exists() and Path(second.output_path).exists()


def test_the_result_reports_the_mode_that_produced_it(tmp_path):
    result = generate(
        a_request(),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert result.mode == "text_to_video"


def test_the_pipeline_receives_the_kwargs_the_mode_built(tmp_path):
    pipe = FakePipe()
    generate(
        a_request(prompt="a red balloon"),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert pipe.calls, "generate must actually call the pipeline"
    _, kwargs = pipe.calls[0]
    assert kwargs.get("prompt") == "a red balloon"


# --------------------------------------------------------------------------
# failure at every depth — each with its OWN actionable error
# --------------------------------------------------------------------------


def test_a_model_load_failure_raises_model_load_error(tmp_path):
    with pytest.raises(ModelLoadError):
        generate(
            a_request(),
            pipeline_factory=exploding_factory(RuntimeError("connection reset by peer")),
            encoder=good_encoder,
            output_root=tmp_path,
        )


def test_the_model_load_error_keeps_the_original_cause(tmp_path):
    original = RuntimeError("connection reset by peer")
    with pytest.raises(ModelLoadError) as excinfo:
        generate(
            a_request(),
            pipeline_factory=exploding_factory(original),
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert excinfo.value.__cause__ is original, "swallowing the cause loses the diagnosis"


def test_a_pipeline_returning_nothing_raises_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        generate(
            a_request(),
            pipeline_factory=factory_for(FakePipe(output=NothingOutput())),
            encoder=good_encoder,
            output_root=tmp_path,
        )


def test_a_pipeline_returning_empty_frames_raises_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        generate(
            a_request(),
            pipeline_factory=factory_for(FakePipe(output=FakeOutput(frames=[], audio=[]))),
            encoder=good_encoder,
            output_root=tmp_path,
        )


def test_a_pipeline_returning_none_raises_decode_error(tmp_path):
    class NoneReturningPipe:
        def __call__(self, *args, **kwargs):
            return None

    with pytest.raises(DecodeError):
        generate(
            a_request(),
            pipeline_factory=factory_for(NoneReturningPipe()),
            encoder=good_encoder,
            output_root=tmp_path,
        )


def test_a_missing_output_file_raises_empty_output_error(tmp_path):
    with pytest.raises(EmptyOutputError):
        generate(
            a_request(),
            pipeline_factory=factory_for(FakePipe()),
            encoder=encoder_returning_a_path_it_never_wrote,
            output_root=tmp_path,
        )


def test_a_zero_byte_output_file_raises_empty_output_error(tmp_path):
    """A file that exists but is empty is the exact shape of the 2.3 bug: the UI
    gets a truthy path and renders nothing."""
    with pytest.raises(EmptyOutputError):
        generate(
            a_request(),
            pipeline_factory=factory_for(FakePipe()),
            encoder=encoder_writing_a_zero_byte_file,
            output_root=tmp_path,
        )


def test_an_encoder_returning_none_raises_rather_than_propagating_none(tmp_path):
    with pytest.raises(EmptyOutputError):
        generate(
            a_request(),
            pipeline_factory=factory_for(FakePipe()),
            encoder=encoder_returning_none,
            output_root=tmp_path,
        )


def test_an_encoder_returning_an_empty_string_raises(tmp_path):
    with pytest.raises(EmptyOutputError):
        generate(
            a_request(),
            pipeline_factory=factory_for(FakePipe()),
            encoder=encoder_returning_empty_string,
            output_root=tmp_path,
        )


def test_an_encoder_crash_is_translated_into_one_of_our_errors(tmp_path):
    with pytest.raises(GenerationError) as excinfo:
        generate(
            a_request(),
            pipeline_factory=factory_for(FakePipe()),
            encoder=encoder_raising,
            output_root=tmp_path,
        )
    assert not isinstance(excinfo.value, OSError), "an OSError must not reach the UI raw"


FAILURE_INJECTIONS = {
    "model_load": dict(
        pipeline_factory=exploding_factory(RuntimeError("connection reset by peer")),
        encoder=good_encoder,
    ),
    "decode_returned_nothing": dict(
        pipeline_factory=factory_for(FakePipe(output=NothingOutput())),
        encoder=good_encoder,
    ),
    "output_file_missing": dict(
        pipeline_factory=factory_for(FakePipe()),
        encoder=encoder_returning_a_path_it_never_wrote,
    ),
    "output_file_empty": dict(
        pipeline_factory=factory_for(FakePipe()),
        encoder=encoder_writing_a_zero_byte_file,
    ),
    "encoder_returned_none": dict(
        pipeline_factory=factory_for(FakePipe()),
        encoder=encoder_returning_none,
    ),
}


@pytest.mark.parametrize("name", sorted(FAILURE_INJECTIONS))
def test_every_injected_failure_raises_one_of_our_typed_errors(name, tmp_path):
    with pytest.raises(GenerationError):
        generate(a_request(), output_root=tmp_path, **FAILURE_INJECTIONS[name])


@pytest.mark.parametrize("name", sorted(FAILURE_INJECTIONS))
def test_no_injected_failure_ever_returns_a_value(name, tmp_path):
    """The invariant, stated as bluntly as it can be stated."""
    returned = "sentinel"
    try:
        returned = generate(a_request(), output_root=tmp_path, **FAILURE_INJECTIONS[name])
    except LTXStudioError:
        returned = "sentinel"
    assert returned == "sentinel", f"{name} returned {returned!r} instead of raising"


def test_the_failure_messages_are_all_different(tmp_path):
    """Five different failures must read as five different problems. One generic
    "generation failed" for all of them is not a diagnosis."""
    messages = {}
    for name, injection in FAILURE_INJECTIONS.items():
        with pytest.raises(GenerationError) as excinfo:
            generate(a_request(), output_root=tmp_path, **injection)
        messages[name] = str(excinfo.value)
    assert len(set(messages.values())) == len(messages), messages


@pytest.mark.parametrize("name", sorted(FAILURE_INJECTIONS))
def test_every_failure_message_is_actionable(name, tmp_path):
    with pytest.raises(GenerationError) as excinfo:
        generate(a_request(), output_root=tmp_path, **FAILURE_INJECTIONS[name])
    message = str(excinfo.value)
    assert len(message) > 20, f"{name}: {message!r}"
    assert message.strip() != "", name


# --------------------------------------------------------------------------
# validation happens before anything expensive
# --------------------------------------------------------------------------


def test_an_illegal_frame_count_is_rejected_before_the_model_is_loaded(tmp_path):
    called = []

    def factory(*args, **kwargs):
        called.append(True)
        return FakePipe()

    with pytest.raises(FrameCountError):
        generate(
            a_request(num_frames=120),
            pipeline_factory=factory,
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert not called, "constraints are checked before a 72 GB load, not after"


def test_an_illegal_dimension_is_rejected_before_the_model_is_loaded(tmp_path):
    from ltx25_studio.errors import DimensionError

    called = []

    def factory(*args, **kwargs):
        called.append(True)
        return FakePipe()

    with pytest.raises(DimensionError):
        generate(
            a_request(width=700),
            pipeline_factory=factory,
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert not called


def test_an_unknown_mode_is_rejected_before_the_model_is_loaded(tmp_path):
    called = []

    def factory(*args, **kwargs):
        called.append(True)
        return FakePipe()

    with pytest.raises(UnknownModeError):
        generate(
            a_request(mode="text_to_hologram"),
            pipeline_factory=factory,
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert not called


# --------------------------------------------------------------------------
# NO "newest file in the output directory" FALLBACK (PLAN §5.3)
# --------------------------------------------------------------------------


def test_a_failed_run_never_returns_a_previous_runs_output(tmp_path):
    """The 2.3 fallback had no run correlation. A failed run would pick up the
    last successful run's video and present it as this run's result — which is
    worse than an empty path, because it looks like it worked."""
    stale_dir = tmp_path / "run-0000-previous"
    stale_dir.mkdir()
    stale = stale_dir / "output.mp4"
    stale.write_bytes(b"\xff" * 4096)

    with pytest.raises(GenerationError):
        generate(
            a_request(),
            pipeline_factory=factory_for(FakePipe()),
            encoder=encoder_returning_a_path_it_never_wrote,
            output_root=tmp_path,
        )
    assert stale.read_bytes() == b"\xff" * 4096, "the stale file must not even be touched"


def test_a_successful_run_returns_its_own_file_not_the_newest_one(tmp_path):
    """A file written *after* this run's artifact must not be able to win."""
    result = generate(
        a_request(),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    decoy = tmp_path / "run-9999-decoy"
    decoy.mkdir()
    (decoy / "output.mp4").write_bytes(b"\xff" * 8192)
    assert Path(result.output_path).read_bytes() == b"\x00" * 2048


BANNED_IN_GENERATE = (
    "getmtime",
    "st_mtime",
    "iterdir",
    "glob",
    "listdir",
    "scandir",
    "walk",
)


def test_generate_never_goes_looking_for_a_file_it_did_not_write():
    """PLAN §5.3, as an architecture rule.

    ``generate.py`` knows exactly which path it asked the encoder to write. Any
    directory scan or mtime sort in this module is the 2.3 fallback growing back.
    (Oldest-first eviction in ``outputs.py`` legitimately needs mtime — this rule
    is scoped to ``generate.py`` alone.)
    """
    tokens = set(code_tokens(PACKAGE_ROOT / "generate.py"))
    offenders = sorted(tokens & set(BANNED_IN_GENERATE))
    assert not offenders, (
        f"generate.py references {offenders} — a 'newest file in the output dir' fallback "
        "has no run correlation and will show a previous run's video for a failed run "
        "(PLAN §5.3)."
    )


def test_generate_does_not_branch_on_mode_identifiers():
    """PLAN §7.2: "adding a mode should mean adding a registry entry, not adding
    an ``if`` to ``generate.py``"."""
    from ltx25_studio import modes

    literals = {value for _, value in code_string_constants(PACKAGE_ROOT / "generate.py")}
    hardcoded = sorted(set(modes.MODES) & literals)
    assert not hardcoded, (
        f"generate.py names modes {hardcoded} directly. The registry is data; dispatch "
        "through it (PLAN §7.2)."
    )


# --------------------------------------------------------------------------
# the seams exist and default to the real thing
# --------------------------------------------------------------------------


def test_generate_accepts_the_documented_injection_points():
    import inspect

    params = inspect.signature(generate_module.generate).parameters
    for name in ("pipeline_factory", "encoder", "output_root"):
        assert name in params, f"generate() must accept {name}= so it is testable without a GPU"
        assert params[name].default is not inspect.Parameter.empty, (
            f"{name} must default to the real implementation, not be required at every call site"
        )


def test_the_request_is_a_real_object_not_a_dict():
    """``ui/handlers.py`` marshals inputs into this; a dict would let a typo pass
    silently all the way to the pipeline."""
    request = a_request()
    assert not isinstance(request, dict)
    assert isinstance(request, object)
    assert request.mode == "text_to_video"


def test_the_result_is_not_a_bare_tuple(tmp_path):
    result = generate(
        a_request(),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert not isinstance(result, (tuple, list, str)), (
        "an unnamed tuple is how a path and an error message get swapped at a call site"
    )
    assert not isinstance(result, SimpleNamespace) or hasattr(result, "output_path")


# --------------------------------------------------------------------------
# the LoRA lifecycle on a warm worker -- attach, activate together, detach
# --------------------------------------------------------------------------
#
# The pipeline is a cached singleton (pipelines._CACHE), so a ZeroGPU worker
# that survives between requests hands every generation the SAME object.
# Anything a run leaves attached outlives the run: the next run with the same
# staged adapter dies on diffusers' duplicate-name ValueError, and a run with
# no staged adapters silently generates through the previous user's LoRA --
# the silent-garbage shape stage 7 exists to prevent, arriving through the
# lifecycle instead of the load.
#
# The fake below mirrors the two real diffusers behaviours these tests exist
# to pin, because a fake that cannot express the bug makes the test theatre
# (conftest's own rule): `load_lora_weights` refuses a duplicate adapter name
# exactly as peft does, and `set_adapters` REPLACES the active set rather
# than adding to it.


class _LoRAModule:
    def __init__(self):
        self.lora_A = {}


class _LoRAComponent:
    def __init__(self):
        self._modules = {"blocks.0.attn": _LoRAModule()}

    def named_modules(self):
        return iter(self._modules.items())


class LoRAFakePipe(FakePipe):
    def __init__(self, output=None, raises=None, zero_attach_sources=()):
        super().__init__(output=output, raises=raises)
        self.transformer = _LoRAComponent()
        self.adapters = []
        self.active = []
        self.active_weights = None
        self.deleted = []
        self.adapters_at_call = None
        self.active_at_call = None
        self.weights_at_call = None
        self._zero_attach_sources = tuple(str(s) for s in zero_attach_sources)

    def __call__(self, *args, **kwargs):
        # Snapshot what the pipeline would actually run with: assertions made
        # after generate() returns cannot tell "active during the run" from
        # "activated and then detached".
        self.adapters_at_call = list(self.adapters)
        self.active_at_call = list(self.active)
        self.weights_at_call = (
            list(self.active_weights) if self.active_weights is not None else None
        )
        return super().__call__(*args, **kwargs)

    def load_lora_weights(self, path, adapter_name=None):
        if adapter_name in self.adapters:
            # Verbatim shape of diffusers/loaders/peft.py -- a bare ValueError.
            raise ValueError(
                f"Adapter name {adapter_name} already in use in the model - "
                f"please select a new adapter name."
            )
        self.adapters.append(adapter_name)
        if str(path) in self._zero_attach_sources:
            return  # registered, attached to nothing -- the stage-7 shape
        for _, module in self.transformer.named_modules():
            module.lora_A[adapter_name] = object()

    def set_adapters(self, names, weights=None):
        names = [names] if isinstance(names, str) else list(names)
        if weights is None or isinstance(weights, (int, float)):
            # diffusers expands a scalar (or None) into one entry per adapter.
            weights = [1.0 if weights is None else float(weights)] * len(names)
        self.active = names  # REPLACE, not extend -- the real semantics
        self.active_weights = list(weights)

    def delete_adapters(self, names):
        for name in [names] if isinstance(names, str) else list(names):
            self.adapters.remove(name)
            for _, module in self.transformer.named_modules():
                module.lora_A.pop(name, None)
            self.active = [n for n in self.active if n != name]
            self.deleted.append(name)


def _staged(source, weight=1.0):
    # The UI stages validated adapters as dicts carrying their report; the
    # report's presence is what lets _apply_loras skip stages 1-6.5, which
    # would otherwise want a real safetensors file.
    return {"source": str(source), "weight": weight, "report": object()}


def test_a_second_run_with_the_same_staged_lora_works_on_a_warm_worker(tmp_path):
    """The fix for the duplicate-name death: run N+1 must not inherit run N's
    registration. Before the unload existed this raised 'Adapter name
    user_lora_0 already in use' out of the second load."""
    pipe = LoRAFakePipe()
    for _ in range(2):
        result = generate(
            a_request(loras=(_staged(tmp_path / "a.safetensors"),)),
            pipeline_factory=factory_for(pipe),
            encoder=good_encoder,
            output_root=tmp_path,
        )
        assert Path(result.output_path).exists()
    assert pipe.deleted == ["user_lora_0", "user_lora_0"]


def test_adapters_are_attached_during_the_run_and_gone_after_it(tmp_path):
    pipe = LoRAFakePipe()
    generate(
        a_request(loras=(_staged(tmp_path / "a.safetensors"),)),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert pipe.adapters_at_call == ["user_lora_0"], "the adapter must be live during the call"
    assert pipe.adapters == [], "and detached after it -- the singleton outlives the request"
    assert pipe.active == []


def test_a_run_without_loras_does_not_inherit_the_previous_runs_adapters(tmp_path):
    """The silent half, and the worse one: a leaked adapter does not raise, it
    quietly styles the NEXT user's generation."""
    pipe = LoRAFakePipe()
    generate(
        a_request(loras=(_staged(tmp_path / "a.safetensors"),)),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    generate(
        a_request(),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert pipe.adapters_at_call == [], (
        "a run with no staged adapters generated through the previous run's LoRA -- "
        "base-model output was promised and something else was delivered"
    )


def test_two_staged_loras_are_both_active_during_the_run(tmp_path):
    """set_adapters REPLACES the active set, so per-adapter activation leaves
    only the last adapter live while every attach report claims success. The
    collective call in _apply_loras is what makes this pass."""
    pipe = LoRAFakePipe()
    generate(
        a_request(
            loras=(
                _staged(tmp_path / "a.safetensors", weight=0.25),
                _staged(tmp_path / "b.safetensors", weight=0.75),
            )
        ),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert pipe.active_at_call == ["user_lora_0", "user_lora_1"]
    assert pipe.weights_at_call == [0.25, 0.75]
    assert pipe.adapters == [], "both must still be detached afterwards"


def test_a_failed_generation_still_detaches_its_adapters(tmp_path):
    pipe = LoRAFakePipe(raises=RuntimeError("CUDA out of memory"))
    with pytest.raises(DecodeError):
        generate(
            a_request(loras=(_staged(tmp_path / "a.safetensors"),)),
            pipeline_factory=factory_for(pipe),
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert pipe.adapters == [], "a failed run must not poison the next one"


def test_a_mid_staging_failure_detaches_what_was_already_attached(tmp_path):
    """Adapter 2 of 2 failing stage 7 must not leave adapter 1 -- or its own
    dead registration -- on the singleton; otherwise the RETRY dies on a
    duplicate name instead of the real error."""
    good, bad = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    pipe = LoRAFakePipe(zero_attach_sources=(bad,))
    with pytest.raises(LoRANotAttachedError):
        generate(
            a_request(loras=(_staged(good), _staged(bad))),
            pipeline_factory=factory_for(pipe),
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert pipe.adapters == [], "both the attached and the half-registered adapter must go"


def test_an_adapter_that_cannot_be_detached_fails_loudly(tmp_path):
    """A stuck adapter corrupts every later run on the worker, which is louder
    than any single run's outcome -- so the unload raises, never warns."""

    class StuckPipe(LoRAFakePipe):
        def delete_adapters(self, names):
            raise RuntimeError("peft refused")

    with pytest.raises(GenerationError) as excinfo:
        generate(
            a_request(loras=(_staged(tmp_path / "a.safetensors"),)),
            pipeline_factory=factory_for(StuckPipe()),
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert "user_lora_0" in str(excinfo.value)
    assert "singleton" in str(excinfo.value)


def test_a_pipeline_that_cannot_delete_adapters_is_refused(tmp_path):
    """No delete_adapters at all: the base state cannot be restored, so the
    run must say so rather than quietly leaving the adapter attached."""

    class NoDeletePipe(LoRAFakePipe):
        delete_adapters = None

    with pytest.raises(GenerationError) as excinfo:
        generate(
            a_request(loras=(_staged(tmp_path / "a.safetensors"),)),
            pipeline_factory=factory_for(NoDeletePipe()),
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert "delete_adapters" in str(excinfo.value)


# --------------------------------------------------------------------------
# supplied audio_latents are checked against the video length -- §5.7 row 11
# --------------------------------------------------------------------------
#
# Supplied audio_latents set the audio length while num_frames governs the
# video, and upstream never compares them: a mismatched pair generates to
# completion and emits a desynced file with no error anywhere. The check
# below was exported and documented as exactly this guard and called by
# nothing (SPACE_MODE_MATRIX §8 follow-up #6); these tests pin its wiring.
#
# Geometry chosen so the seconds are exact: hop 160 x temporal 4 / 16 kHz
# = 0.04 s per latent frame, so 121 latent frames = 4.84 s, and the video
# is 121 pixel frames at 25 fps = 4.84 s.


class _SyncAudioVAE:
    """The six §5.4 parameters where the real checkpoint keeps them: four in
    config under upstream's names, two as instance attributes."""

    config = {
        "sample_rate": 16000,
        "mel_hop_length": 160,
        "mel_bins": 64,
        "latent_channels": 8,
    }
    mel_compression_ratio = 4
    temporal_compression_ratio = 4


def _latents(frames):
    return SimpleNamespace(shape=(1, 8, frames, 64))


def _audio_request(**overrides):
    fields = {
        "mode": "audio_to_video",
        "frame_rate": 25.0,
        "num_frames": 121,
        "audio": object(),
        "noise_scale": 0.3,
    }
    fields.update(overrides)
    return a_request(**fields)


def test_matched_supplied_latents_generate_with_no_warning(tmp_path):
    pipe = FakePipe()
    pipe.audio_vae = _SyncAudioVAE()
    result = generate(
        _audio_request(audio_latents=_latents(121)),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert not any("NOT checked" in w for w in result.warnings)
    _, sent = pipe.calls[-1]
    assert sent.get("audio_latents") is not None, "the supplied latents must still reach the pipe"


def test_desynced_supplied_latents_are_refused_before_the_metered_call(tmp_path):
    """4.84 s of video against 3.84 s of audio must refuse -- and refuse BEFORE
    the pipeline runs, because the desynced generation costs real quota and
    the file it produces is the §5.5 failure: emitted, not refused."""
    pipe = FakePipe()
    pipe.audio_vae = _SyncAudioVAE()
    with pytest.raises(AudioVideoDesyncError) as excinfo:
        generate(
            _audio_request(audio_latents=_latents(96)),
            pipeline_factory=factory_for(pipe),
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert pipe.calls == [], "the refusal must land before any pipeline work"
    assert "4.84" in str(excinfo.value) and "3.84" in str(excinfo.value), (
        "the refusal must name both lengths so the user can fix one"
    )


def test_auto_duration_skips_the_check_because_nothing_governs_the_video(tmp_path):
    """With num_frames omitted the duration head picks the video length, so
    there is no independent length for the audio to disagree with."""
    pipe = FakePipe()
    pipe.audio_vae = _SyncAudioVAE()
    result = generate(
        _audio_request(audio_latents=_latents(96), auto_duration=True),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert not any("NOT checked" in w for w in result.warnings)


def test_no_audio_vae_reports_could_not_check_rather_than_passing_silently(tmp_path):
    """The check_mmap_backed rule: 'could not check' and 'checked and fine'
    must never look the same. A pipe with no audio_vae cannot yield the six
    timing parameters, so the run proceeds carrying a warning that says so."""
    result = generate(
        _audio_request(audio_latents=_latents(96)),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert any("NOT checked" in w and "audio_vae" in w for w in result.warnings)


def test_shapeless_latents_report_could_not_check(tmp_path):
    pipe = FakePipe()
    pipe.audio_vae = _SyncAudioVAE()
    result = generate(
        _audio_request(audio_latents=object()),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert any("NOT checked" in w and "shape" in w for w in result.warnings)


# --------------------------------------------------------------------------
# uploaded audio FILES: trim long (with a visible note), refuse short
# --------------------------------------------------------------------------
#
# The file branch deliberately accommodates where the supplied-latents branch
# refuses: supplied latents are a programmatic caller who meant something
# precise; a file upload is an end user who should not need to know the frame
# arithmetic. Geometry as in the latents tests: 0.04 s per latent frame, and
# 121 frames @ 25 fps = 4.84 s of video = 77440 samples at 16 kHz.


def _file_audio_pipe():
    from conftest import FakeAudioVae

    pipe = FakePipe()
    pipe.audio_vae = FakeAudioVae()
    return pipe


def _patched_upload(monkeypatch, samples):
    from conftest import waveform_of

    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (waveform_of(2, samples=samples), 16000)
    )


def test_a_long_upload_is_trimmed_and_the_user_is_told(monkeypatch, tmp_path):
    pipe = _file_audio_pipe()
    _patched_upload(monkeypatch, samples=96000)  # 6.0 s against a 4.84 s video
    result = generate(
        _audio_request(),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert "audio input trimmed from 6.00s to 4.84s to match 121 frames" in result.warnings
    # The encoder must have consumed the TRIMMED waveform -- the note without
    # the trim would be the inverted defect: reporting an alteration that
    # never happened.
    assert pipe.audio_vae.mels[-1].shape[-1] == 77440


def test_a_matching_upload_is_encoded_whole_with_no_note(monkeypatch, tmp_path):
    pipe = _file_audio_pipe()
    _patched_upload(monkeypatch, samples=77440)
    result = generate(
        _audio_request(),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert not any("trimmed" in w for w in result.warnings)
    assert pipe.audio_vae.mels[-1].shape[-1] == 77440


def test_a_short_upload_is_refused_before_the_metered_call(monkeypatch, tmp_path):
    """Refuse, never pad: padding fabricates input the user never gave, and
    the refusal must name the num_frames that would fit so the message is a
    fix, not a shrug. It lands before the pipeline so it costs no quota."""
    pipe = _file_audio_pipe()
    _patched_upload(monkeypatch, samples=48000)  # 3.0 s
    with pytest.raises(AudioVideoDesyncError) as excinfo:
        generate(
            _audio_request(),
            pipeline_factory=factory_for(pipe),
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert pipe.calls == []
    assert "num_frames=73" in str(excinfo.value)


def test_auto_duration_leaves_the_upload_length_alone(monkeypatch, tmp_path):
    """No num_frames, no independent video length, nothing to reconcile --
    the duration head may well derive the length FROM this audio."""
    pipe = _file_audio_pipe()
    _patched_upload(monkeypatch, samples=96000)
    result = generate(
        _audio_request(auto_duration=True),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert pipe.audio_vae.mels[-1].shape[-1] == 96000, "auto_duration must not trim"
    assert not any("trimmed" in w for w in result.warnings)


def test_a_hub_staged_adapter_is_loaded_from_its_fetched_local_path(tmp_path):
    """The web process resolves repo ids to local files precisely so this
    loop — inside the billed GPU window — only ever reads a file. 'path'
    (the fetched local file) must win over 'source' (what the user typed);
    an upload has no 'path' and falls back to 'source', which IS local."""

    class RecordingPipe(LoRAFakePipe):
        def __init__(self):
            super().__init__()
            self.loaded_from = []

        def load_lora_weights(self, path, adapter_name=None):
            self.loaded_from.append(str(path))
            return super().load_lora_weights(path, adapter_name=adapter_name)

    local = tmp_path / "fetched.safetensors"
    pipe = RecordingPipe()
    generate(
        a_request(
            loras=(
                {"source": "org/my-lora", "path": str(local), "report": object()},
                _staged(tmp_path / "uploaded.safetensors"),
            )
        ),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        output_root=tmp_path,
    )
    assert pipe.loaded_from == [str(local), str(tmp_path / "uploaded.safetensors")], (
        "a repo id must never reach load_lora_weights — the GPU window reads files only"
    )


# --------------------------------------------------------------------------
# peak-VRAM instrumentation
# --------------------------------------------------------------------------
#
# Two decisions on this path were taken on VRAM grounds with no VRAM number
# behind either. The enhancer is loaded and freed on every call for "VRAM
# safety", which costs +30.78 s of reload per enhanced generation; the VAE
# decode went tiled+framewise after a MIG allocator assert nobody has headroom
# for. Both are unmeasured trades. Neither can be revisited until a real run
# says what the partition has, so every generation now reports it.
#
# The spec below is deliberately about the *properties* rather than the prose:
# the reset happens first, the three quantities never collapse into one, the
# split is real, and none of it can take a run down.

_GiB = 1024**3


class FakeCuda:
    """A ``torch.cuda`` answering only the calls the probe is allowed to make.

    Anything the probe reaches for that is not defined here raises
    ``AttributeError`` -- which is how ``synchronize`` creeping onto the hot
    path is caught rather than argued about.
    """

    def __init__(self, *, total, resident, allocated, reserved, available=True):
        self.calls: list = []
        self._total = total
        self._resident = resident
        self._allocated = list(allocated)
        self._reserved = list(reserved)
        self._available = available

    def _next(self, values):
        # Sticky last value: a probe that samples more often than the test
        # scripted should report a plausible number, not blow up and disable
        # itself, or the test would be asserting on the wrong failure.
        return values.pop(0) if len(values) > 1 else values[0]

    def is_available(self):
        return self._available

    def current_device(self):
        self.calls.append("current_device")
        return 0

    def get_device_properties(self, index):
        self.calls.append(("get_device_properties", index))
        return SimpleNamespace(total_memory=self._total)

    def memory_allocated(self):
        self.calls.append("memory_allocated")
        return self._resident

    def max_memory_allocated(self):
        self.calls.append("max_memory_allocated")
        return self._next(self._allocated)

    def max_memory_reserved(self):
        self.calls.append("max_memory_reserved")
        return self._next(self._reserved)

    def reset_peak_memory_stats(self):
        self.calls.append("reset_peak_memory_stats")


class ExplodingCuda(FakeCuda):
    """Available, and broken in every direction. Nothing here may reach the user."""

    def __init__(self):
        super().__init__(total=1, resident=1, allocated=[1], reserved=[1])

    def _boom(self, *args, **kwargs):
        raise RuntimeError("NVML_SUCCESS == r INTERNAL ASSERT FAILED")

    current_device = _boom
    get_device_properties = _boom
    memory_allocated = _boom
    max_memory_allocated = _boom
    max_memory_reserved = _boom
    reset_peak_memory_stats = _boom


def install_fake_torch(monkeypatch, cuda):
    """Put a ``torch`` in ``sys.modules`` carrying ``cuda``, and nothing else it
    does not need. ``_seeded_generator`` also imports torch on this path, so the
    fake owes it a Generator; that is the whole surface."""
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.cuda = cuda

    class _Generator:
        def __init__(self, device=None):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    torch.Generator = _Generator
    monkeypatch.setitem(sys.modules, "torch", torch)
    return cuda


def a_cuda(**overrides):
    fields = {
        # A plausible MIG partition and a plausible flagship run: ~54 GiB of
        # int8 transformer already resident, a peak in the low seventies.
        "total": 79 * _GiB,
        "resident": 54 * _GiB,
        "allocated": [71 * _GiB],
        "reserved": [73 * _GiB],
    }
    fields.update(overrides)
    return FakeCuda(**fields)


def vram_notes(result):
    return [w for w in result.warnings if w.startswith("peak VRAM")]


def run_on(tmp_path, **overrides):
    return generate(
        a_request(**overrides),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
    )


def test_a_run_with_no_cuda_reports_no_vram_and_reads_exactly_as_before(tmp_path):
    """The offline gate runs on MPS and CPU. The probe must be invisible there.

    Not "reports zero" -- absent. A local run that grew three lines of GiB
    figures about a device it does not have would be a probe leaking into the
    thing it measures.
    """
    result = run_on(tmp_path)
    assert vram_notes(result) == [], (
        "a box with no CUDA reported VRAM figures, so the probe is either "
        "inventing numbers or initialising a device that is not there"
    )


def test_a_cuda_run_reports_allocated_and_reserved_and_total_without_conflating_them(
    tmp_path, monkeypatch
):
    """The three quantities, distinguishable, in one note.

    ``allocated`` is what the run needed, ``reserved`` is what the caching
    allocator holds and therefore what actually bounds the next allocation, and
    ``total`` is what the partition has. Headroom is ``total - reserved``.
    Quoting ``allocated`` against ``total`` overstates it by the whole of the
    allocator's slack -- and measuring the wrong quantity is precisely the
    mistake this build keeps making.
    """
    install_fake_torch(monkeypatch, a_cuda())
    notes = vram_notes(run_on(tmp_path))
    assert len(notes) == 1, notes
    note = notes[0]
    for figure, quantity in (("71.000", "allocated"), ("73.000", "reserved"), ("79.000", "total")):
        assert f"{figure} GiB" in note, (
            f"the {quantity} figure is missing from {note!r}; all three have to be present "
            f"or the reader cannot tell which one they are looking at"
        )
    assert "54.000 GiB" in note, (
        "the resident-on-entry figure is missing, so nothing separates what the run "
        "cost from what the transformer was already holding"
    )


def test_the_peak_is_reset_before_anything_is_read(tmp_path, monkeypatch):
    """A ZeroGPU worker is reused across generations and arrives carrying the
    previous run's peak. A probe that only reads reports the largest run the
    container ever served, and calls it this one."""
    cuda = install_fake_torch(monkeypatch, a_cuda())
    run_on(tmp_path)
    assert cuda.calls[0] == "reset_peak_memory_stats", (
        f"the first CUDA call of the run was {cuda.calls[0]!r}, not the reset — a reused "
        f"worker's stale peak is being reported as this run's"
    )


def test_enhancement_and_generation_are_measured_as_separate_windows(tmp_path, monkeypatch):
    """The split that answers the actual open question.

    The enhancer costs +30.78 s per enhanced generation because it is loaded and
    freed every call. Making it resident is only safe if its ~2.49 GB is *not*
    additive to the peak the denoise already reaches. One lap each side of the
    rewrite is what settles that, and a whole-call number cannot.
    """
    cuda = install_fake_torch(
        monkeypatch,
        a_cuda(allocated=[57 * _GiB, 71 * _GiB], reserved=[59 * _GiB, 73 * _GiB]),
    )
    result = generate(
        a_request(enhance_prompt=True),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
        enhancer=lambda prompt, image=None: "a weathered stone lighthouse in dense fog",
    )
    notes = vram_notes(result)
    assert len(notes) == 3, f"expected a lap each side plus the whole call, got {notes}"
    assert "prompt enhancement" in notes[0] and "57.000 GiB" in notes[0]
    assert "generation" in notes[1] and "71.000 GiB" in notes[1]
    assert cuda.calls.count("reset_peak_memory_stats") == 3, (
        "each lap must reset after it reads, or every window reports 'everything so "
        "far' and the second phase is credited with the first phase's peak"
    )


def test_the_whole_call_figure_is_the_largest_lap_not_the_last(tmp_path, monkeypatch):
    """Laps are windows, so the whole call is their maximum -- exactly, because
    the max over two touching windows is the max of the two maxima.

    Scripted here with the *enhancement* lap higher than the generation lap,
    which is the case that catches a "just report the last sample" shortcut. It
    is also the case that would matter most: an enhancer that peaks above the
    denoise is an enhancer that cannot be made resident.
    """
    install_fake_torch(
        monkeypatch,
        a_cuda(allocated=[77 * _GiB, 71 * _GiB], reserved=[78 * _GiB, 73 * _GiB]),
    )
    result = generate(
        a_request(enhance_prompt=True),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
        enhancer=lambda prompt, image=None: "a weathered stone lighthouse in dense fog",
    )
    whole = vram_notes(result)[-1]
    assert "whole call" in whole
    assert "77.000 GiB" in whole and "78.000 GiB" in whole, (
        f"the whole-call note is {whole!r}; it reported a later, smaller lap and so "
        f"understates the peak the partition actually had to hold"
    )


def test_a_run_that_never_enhances_gets_one_window_and_no_phantom_phase(tmp_path, monkeypatch):
    """``enhance_prompt`` off means no model was loaded, so there is no
    enhancement phase to attribute. A lap there would report the baseline under
    a label claiming a measurement happened."""
    install_fake_torch(monkeypatch, a_cuda())
    notes = vram_notes(run_on(tmp_path))
    assert len(notes) == 1 and "whole call" in notes[0], notes
    assert not any("prompt enhancement" in n for n in notes)


def test_enhancement_requested_on_an_empty_prompt_reports_no_enhancement_phase(
    tmp_path, monkeypatch
):
    """The other way the enhancer does not run. ``enhance_if_requested`` returns
    without loading anything when the prompt is blank, so the toggle being set
    is not evidence that a phase existed."""
    install_fake_torch(monkeypatch, a_cuda())
    result = generate(
        a_request(prompt="   ", enhance_prompt=True),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
        enhancer=lambda prompt, image=None: pytest.fail(
            "the enhancer must not run on a blank prompt"
        ),
    )
    assert not any("prompt enhancement" in n for n in vram_notes(result))


def test_a_probe_that_cannot_measure_never_takes_the_run_down(tmp_path, monkeypatch):
    """Log-only, and that is load-bearing rather than tidy.

    There is no baseline yet, so there is no threshold that could be right, and
    a wrong one breaks a working Space for the sake of a diagnostic. The exact
    failure scripted here is the one this partition actually produces: the NVML
    allocator assert, raised from inside the allocator bookkeeping the probe
    reads.
    """
    install_fake_torch(monkeypatch, ExplodingCuda())
    result = run_on(tmp_path)
    assert Path(result.output_path).exists(), "a broken probe killed a good generation"
    assert vram_notes(result) == [], "a probe that could not measure reported anyway"


def test_the_probe_does_not_synchronise_the_stream(tmp_path, monkeypatch):
    """A synchronize on the hot path would perturb both the number being
    measured and the wall-clock it is measured against -- and the wall-clock is
    the whole reason this instrumentation exists.

    ``FakeCuda`` defines no ``synchronize`` and no ``empty_cache``, so reaching
    for either raises ``AttributeError`` and fails the run rather than passing
    quietly. The source check is the belt to that braces: an unreached branch
    would slip past the fake.
    """
    install_fake_torch(monkeypatch, a_cuda())
    run_on(tmp_path)
    tokens = set(code_tokens(PACKAGE_ROOT / "generate.py"))
    assert "synchronize" not in tokens and "empty_cache" not in tokens, (
        "generate.py synchronises or empties the cache on the generation path; both cost "
        "real time on every run to make a free measurement slightly tidier"
    )


def test_the_vram_notes_are_plain_strings_because_they_cross_the_fork(tmp_path, monkeypatch):
    """They are built inside the ``@spaces.GPU`` worker and read in the web
    process, so they ride the res queue. ``tests/test_request_fork_boundary.py``
    owns the pickle proof; this pins the shape at the source."""
    install_fake_torch(monkeypatch, a_cuda())
    result = run_on(tmp_path)
    assert all(isinstance(w, str) for w in result.warnings)
    assert isinstance(result.warnings, tuple)


def test_the_probe_is_opened_after_validation_so_a_bad_request_costs_nothing(tmp_path, monkeypatch):
    """Everything cheap first still holds. A request refused for its geometry
    must not first construct a probe and touch a device."""
    cuda = install_fake_torch(monkeypatch, a_cuda())
    with pytest.raises(LTXStudioError):
        generate(
            a_request(num_frames=120),
            pipeline_factory=factory_for(FakePipe()),
            encoder=good_encoder,
            output_root=tmp_path,
        )
    assert cuda.calls == [], (
        f"a request refused by validation still touched CUDA ({cuda.calls}); the probe is "
        f"above the cheap refusals"
    )
