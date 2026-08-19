"""Spec for ``ltx25_studio/zerogpu.py`` — PLAN §8, §5.8.

Three rules, all of which fail *silently* when broken, and none of which needs a
GPU to test:

1. **Never return a CUDA tensor across the boundary.** Unpickling one in the main
   process triggers a blocked lazy CUDA init and **the call hangs** — no error, no
   timeout, a dead request. This is the worst failure shape in the whole app
   because there is nothing to read afterwards.
2. **Never pass ``size=`` unguarded.** The platform pins its own ``spaces`` and
   lags PyPI; against a build predating the parameter,
   ``@spaces.GPU(duration=..., size="xlarge")`` raises ``TypeError`` **at module
   import**, so the Space simply never starts.
3. **The ``duration=`` callable must accept the decorated function's exact
   signature**, or it TypeErrors at schedule time — after the user has waited.

Tested with fakes throughout: a "CUDA tensor" here is any object exposing a
truthy ``is_cuda`` or a ``device`` that stringifies as ``cuda*``, which is what a
real ``torch.Tensor`` does and what the detector must key on.
"""

from __future__ import annotations

import pytest

from ltx25_studio import zerogpu
from ltx25_studio.errors import LTXStudioError


class FakeTensor:
    def __init__(self, device="cpu"):
        self.device = device
        self.is_cuda = str(device).startswith("cuda")

    def cpu(self):
        return FakeTensor("cpu")


class FakeOutput:
    """Duck-types ``LTX2PipelineOutput`` — the object generation actually returns."""

    def __init__(self, frames, audio):
        self.frames = frames
        self.audio = audio


# --------------------------------------------------------------------------
# the return boundary
# --------------------------------------------------------------------------


def test_a_path_crosses_the_boundary_fine(tmp_path):
    zerogpu.assert_returnable(tmp_path / "run-1" / "output.mp4")


def test_plain_data_crosses_the_boundary_fine():
    zerogpu.assert_returnable({"video": "/tmp/run-1/output.mp4", "seed": 7, "frames": 121})


def test_a_cpu_tensor_crosses_the_boundary_fine():
    zerogpu.assert_returnable(FakeTensor("cpu"))


def test_a_cuda_tensor_is_refused():
    """The hang. A loud error here is strictly better than a dead request."""
    with pytest.raises(LTXStudioError):
        zerogpu.assert_returnable(FakeTensor("cuda:0"))


def test_the_refusal_says_what_to_do_instead():
    with pytest.raises(LTXStudioError) as excinfo:
        zerogpu.assert_returnable(FakeTensor("cuda:0"))
    message = str(excinfo.value).lower()
    assert "cuda" in message
    assert "cpu" in message or "path" in message, (
        f"the message should name the fix — return a path, or .cpu(); got {message!r}"
    )


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda t: (t,),
        lambda t: [t],
        lambda t: {"frames": t},
        lambda t: ("/tmp/x.mp4", t),
        lambda t: [{"frames": t}],
    ],
    ids=["tuple", "list", "dict", "mixed_tuple", "nested"],
)
def test_a_cuda_tensor_nested_in_the_return_value_is_refused(wrapper):
    """Generation returns a container, not a bare tensor. A boundary check that
    only looks at the top level is a boundary check that never fires."""
    with pytest.raises(LTXStudioError):
        zerogpu.assert_returnable(wrapper(FakeTensor("cuda:0")))


def test_a_cuda_tensor_on_a_pipeline_output_object_is_refused():
    output = FakeOutput(frames=FakeTensor("cuda:0"), audio=FakeTensor("cuda:0"))
    with pytest.raises(LTXStudioError):
        zerogpu.assert_returnable(output)


def test_an_all_cpu_pipeline_output_is_fine():
    zerogpu.assert_returnable(FakeOutput(frames=FakeTensor("cpu"), audio=FakeTensor("cpu")))


# --------------------------------------------------------------------------
# the defensive size= kwarg
# --------------------------------------------------------------------------


def test_size_is_passed_when_the_installed_spaces_accepts_it():
    def gpu(*, duration=None, size=None):  # the 0.51-era signature
        return None

    kwargs = zerogpu.gpu_decorator_kwargs(gpu, duration=60, size="xlarge")
    assert kwargs["duration"] == 60
    assert kwargs["size"] == "xlarge"


def test_size_is_dropped_when_the_installed_spaces_predates_it():
    """Live Spaces report 0.49.3 / 0.50.0 / 0.50.4 and we deliberately do not pin
    ``spaces``. Passing ``size`` to one of those raises ``TypeError`` at module
    import — which on a Space looks like a mysterious failure, not a version
    problem."""

    def gpu(*, duration=None):  # the older platform signature
        return None

    kwargs = zerogpu.gpu_decorator_kwargs(gpu, duration=60, size="xlarge")
    assert "size" not in kwargs
    assert kwargs["duration"] == 60


def test_dropping_size_costs_us_nothing():
    """ZeroGPU's own ``startup()`` promotes to xlarge past its 42.00 GiB
    threshold, and ours is 67.23 GiB — so the kwarg neither avoids nor incurs the
    2x quota cost. Dropping it must therefore never raise."""

    def gpu(*, duration=None):
        return None

    assert zerogpu.gpu_decorator_kwargs(gpu, duration=60, size="xlarge")["duration"] == 60


def test_an_unknown_kwarg_is_dropped_rather_than_exploding():
    def gpu(*, duration=None):
        return None

    kwargs = zerogpu.gpu_decorator_kwargs(gpu, duration=60, size="xlarge", some_future_kwarg=1)
    assert set(kwargs) == {"duration"}


# --------------------------------------------------------------------------
# the duration= callable signature
# --------------------------------------------------------------------------


def test_a_matching_duration_callable_is_accepted():
    def generate(prompt, width, height, *, seed=0):
        return None

    def duration(prompt, width, height, *, seed=0):
        return 60

    zerogpu.check_duration_signature(generate, duration)


def test_a_mismatched_duration_callable_is_refused_before_schedule_time():
    """It TypeErrors at schedule time otherwise — i.e. after the user has waited
    and after the reservation was made."""

    def generate(prompt, width, height, *, seed=0):
        return None

    def duration(prompt):
        return 60

    with pytest.raises(LTXStudioError):
        zerogpu.check_duration_signature(generate, duration)


def test_the_mismatch_message_names_both_signatures():
    def generate(prompt, width, height):
        return None

    def duration(prompt):
        return 60

    with pytest.raises(LTXStudioError) as excinfo:
        zerogpu.check_duration_signature(generate, duration)
    message = str(excinfo.value)
    assert "width" in message or "height" in message, message


def test_a_missing_trailing_kwarg_is_refused():
    """"the EXACT same positional signature as the decorated function, trailing
    kwargs included"."""

    def generate(prompt, *, seed=0, num_frames=121):
        return None

    def duration(prompt, *, seed=0):
        return 60

    with pytest.raises(LTXStudioError):
        zerogpu.check_duration_signature(generate, duration)


def test_a_constant_duration_is_not_a_signature_problem():
    def generate(prompt, width, height):
        return None

    zerogpu.check_duration_signature(generate, 60)


# --------------------------------------------------------------------------
# importability off-Space
# --------------------------------------------------------------------------


def test_the_module_imports_without_the_spaces_package():
    """``requirements.txt`` deliberately does not pin ``spaces`` and the dev box
    does not have it. A bare ``import spaces`` here makes the whole package
    unimportable locally — and this suite would be the first casualty."""
    import importlib
    import sys

    assert "spaces" not in sys.modules or sys.modules["spaces"] is not None
    importlib.reload(zerogpu)
