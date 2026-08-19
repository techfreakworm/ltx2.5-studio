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
    DecodeError,
    EmptyOutputError,
    FrameCountError,
    GenerationError,
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
