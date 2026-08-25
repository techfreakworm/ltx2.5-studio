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
