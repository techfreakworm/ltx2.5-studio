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
    """We deliberately do not pin ``spaces``. Passing ``size`` to a build that
    predates the parameter raises ``TypeError`` at module import — which on a
    Space looks like a mysterious failure, not a version problem."""

    def gpu(*, duration=None):  # the older platform signature
        return None

    kwargs = zerogpu.gpu_decorator_kwargs(gpu, duration=60, size="xlarge")
    assert "size" not in kwargs
    assert kwargs["duration"] == 60


def test_dropping_size_still_never_raises_and_the_cost_is_reported():
    """Dropping ``size`` is NOT free: ZeroGPU's ``startup()`` promotion never
    reaches the scheduler, so no ``size`` means the platform default ``large``
    (47.375 GiB measured, half of xlarge). The filter must still not raise, and
    the worker's measured partition is what makes the cost visible."""

    def gpu(*, duration=None):
        return None

    assert zerogpu.gpu_decorator_kwargs(gpu, duration=60, size="xlarge")["duration"] == 60
    assert zerogpu.partition_note(int(47.375 * 1024**3), "xlarge") is not None


# --------------------------------------------------------------------------
# the measured partition against the requested size
# --------------------------------------------------------------------------


def test_a_half_size_partition_under_xlarge_is_warned_about():
    note = zerogpu.partition_note(int(47.375 * 1024**3), "xlarge")

    assert note is not None and "xlarge" in note and "47.4" in note, note


def test_a_full_xlarge_partition_says_nothing():
    assert zerogpu.partition_note(int(94.971 * 1024**3), "xlarge") is None


def test_an_unrequested_size_names_the_platform_default():
    note = zerogpu.partition_note(int(47.375 * 1024**3), "auto")

    assert note is not None and "LTX25_ZEROGPU_SIZE=xlarge" in note, note


def test_an_explicit_large_is_not_second_guessed():
    assert zerogpu.partition_note(int(47.375 * 1024**3), "large") is None


def test_an_unmeasured_partition_says_nothing():
    assert zerogpu.partition_note(None, "xlarge") is None


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
    """ "the EXACT same positional signature as the decorated function, trailing
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


# --------------------------------------------------------------------------
# capability, not identity — and the ordering that keeps it safe on a Space
# --------------------------------------------------------------------------
#
# Gating the weight download, the preflight and the CUDA move on "is this a
# Space" meant a real 97 GB GPU box did none of them: no weights, no pipeline, no
# device move, and a non-strict preflight that therefore did not complain either.
# The UI came up and the first generation went nowhere.


class _SpiedProbe:
    """Records whether the CUDA probe was consulted at all."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.result


def test_a_space_uses_the_gpu_without_ever_probing_cuda(monkeypatch):
    """**The load-bearing assertion is the call count, not the return value.**

    ``import spaces`` must precede anything that touches CUDA — ZeroGPU raises
    "CUDA has been initialized before importing the spaces package" otherwise.
    So on a Space the probe must never run, and a refactor that reorders the
    short-circuit has to fail here rather than on the platform.
    """
    probe = _SpiedProbe(True)
    monkeypatch.setattr(zerogpu, "host_has_cuda", probe)

    class RealSpaces:
        __version__ = "0.51.1"

    assert zerogpu.will_use_gpu(RealSpaces()) is True
    assert probe.calls == 0, (
        "will_use_gpu probed CUDA on a Space. The spaces check must short-circuit "
        "first; touching torch.cuda at module scope there is fatal at startup."
    )


def test_a_plain_cuda_box_uses_the_gpu(monkeypatch):
    """hinode: same GPU, no `spaces` package. It must still do the work."""
    monkeypatch.setattr(zerogpu, "host_has_cuda", _SpiedProbe(True))
    assert zerogpu.will_use_gpu(zerogpu.local_spaces_stub()) is True


def test_a_box_with_no_cuda_does_not(monkeypatch):
    monkeypatch.setattr(zerogpu, "host_has_cuda", _SpiedProbe(False))
    assert zerogpu.will_use_gpu(zerogpu.local_spaces_stub()) is False
    assert zerogpu.will_use_gpu(None) is False


def test_the_cuda_probe_never_raises_without_torch(monkeypatch):
    """A dev box with no model stack must still import the app."""
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert zerogpu.host_has_cuda() is False


# --------------------------------------------------------------------------
# the mmap precondition — 72 GB that never gets freed, with nothing to blame
# --------------------------------------------------------------------------
#
# ZeroGPU reclaims the HF snapshot after packing by walking /proc/self/maps,
# mapping each tensor's data_ptr() back to the blob it came from, and unlinking
# those. A dtype cast, a .clone(), an eager load or a non-safetensors path
# materialises tensors into anonymous memory — nothing left to capture, nothing
# freed, and the Space dies at STEADY STATE with no error explaining why.
#
# We assert on the same basis the packer prunes on, rather than on a proxy.

BLOB = "/home/u/.cache/huggingface/hub/models--Lightricks--LTX-2.5-Diffusers/blobs/abc123"

MAPS = f"""\
7f0000000000-7f0000010000 r--s 00000000 00:1b 111 {BLOB}
7f0000020000-7f0000030000 rw-p 00000000 00:00 0 
7f0000040000-7f0000050000 r--p 00000000 00:1b 222 /usr/lib/x86_64-linux-gnu/libc.so.6
"""

MAPPED = 0x7F0000000100  # inside the blob
ANONYMOUS = 0x7F0000025000  # a cast or a clone lands here
WRONG_FILE = 0x7F0000045000  # file-backed, but not a prunable blob


class _Tensor:
    def __init__(self, address):
        self._address = address

    def data_ptr(self):
        return self._address


class _Module:
    def __init__(self, **params):
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())


def test_a_fully_mmapped_module_passes():
    module = _Module(**{f"blk.{n}.weight": _Tensor(MAPPED) for n in range(4)})
    checked, message = zerogpu.check_mmap_backed(module, name="base", maps_text=MAPS)
    assert checked
    assert "all parameters file-backed" in message


def test_a_cast_tensor_is_caught():
    """The mutation the operator asked for: force one parameter into anonymous
    memory, exactly as a dtype cast or a .clone() would."""
    module = _Module(good=_Tensor(MAPPED), **{"blk.7.weight": _Tensor(ANONYMOUS)})
    checked, message = zerogpu.check_mmap_backed(module, name="base", maps_text=MAPS)
    assert checked
    assert "not mmap'd" in message
    assert "blk.7.weight" in message, f"the offending parameter must be named: {message}"
    assert "anonymous memory" in message


def test_a_tensor_backed_by_the_wrong_file_is_caught():
    """File-backed is not sufficient — it must match the prune pattern.

    A tensor mapped from somewhere the autoprune glob does not cover is just as
    unfreeable as an anonymous one, and looks fine to any check that only asks
    "is this file-backed".
    """
    module = _Module(**{"blk.0.weight": _Tensor(WRONG_FILE)})
    checked, message = zerogpu.check_mmap_backed(module, name="base", maps_text=MAPS)
    assert checked and "not mmap'd" in message
    assert "libc" in message


def test_being_unable_to_check_is_reported_not_passed():
    """ "could not check" and "checked and fine" must never look the same.

    On macOS there is no /proc/self/maps. Returning a quiet pass there would be
    the same defect as the startup report that defaulted to 37/37.
    """
    module = _Module(**{"blk.0.weight": _Tensor(MAPPED)})
    checked, message = zerogpu.check_mmap_backed(
        module, name="base", maps_text=zerogpu.read_process_maps("/nonexistent/maps")
    )
    assert checked is False
    assert "NOT checked" in message


def test_anonymous_regions_are_not_parsed_as_backing():
    regions = zerogpu.parse_maps(MAPS)
    assert zerogpu.backing_file(MAPPED, regions) == BLOB
    assert zerogpu.backing_file(ANONYMOUS, regions) is None


# --------------------------------------------------------------------------
# the duration estimator reads the call the LIVE dispatch actually makes
# --------------------------------------------------------------------------
#
# generate() routes to the entrypoint as dispatch(request=GenerationRequest,
# pipeline_factory=..., ...), and spaces invokes the estimator with those
# same kwargs. So every request field the estimator wants lives on the
# `request` OBJECT -- a read of params.get("enable_stage_2") alone is an
# advertised lever attached to nothing on the live path: it resolves fine,
# changes no behaviour, and every live run reserves the base duration.
# These tests pin both shapes: the request-object read (the live path) and
# the top-level kwarg (direct callers), for stage 2 and for LoRA.


def _fake_request(**fields):
    from types import SimpleNamespace

    defaults = {"enable_stage_2": False, "loras": ()}
    defaults.update(fields)
    return SimpleNamespace(**defaults)


BASE_SECONDS = zerogpu.estimate_duration()


def test_stage_2_scales_the_reservation_through_the_request_object():
    """The live-path shape. A top-level-only read makes stage 2 reserve the
    base duration and die at the reservation kill, having proven nothing."""
    got = zerogpu.estimate_duration(request=_fake_request(enable_stage_2=True))
    assert got == int(BASE_SECONDS * 1.5)


def test_stage_2_still_scales_for_a_direct_top_level_caller():
    assert zerogpu.estimate_duration(enable_stage_2=True) == int(BASE_SECONDS * 1.5)


def test_a_lora_run_reserves_extra_time_through_the_request_object():
    """Inside the billed window a LoRA run additionally reads the adapter
    payload and runs peft injection over ~3320 modules -- absent headroom
    that spend can hit the reservation kill."""
    got = zerogpu.estimate_duration(request=_fake_request(loras=({"source": "a"},)))
    assert got == BASE_SECONDS + zerogpu.LORA_DURATION_SECONDS


def test_a_lora_run_reserves_extra_time_for_a_direct_top_level_caller():
    got = zerogpu.estimate_duration(loras=({"source": "a"},))
    assert got == BASE_SECONDS + zerogpu.LORA_DURATION_SECONDS


def test_no_staged_loras_means_no_lora_headroom():
    """An empty tuple is the request default; it must not buy headroom, or
    every run over-reserves and schedulability pays for nothing."""
    assert zerogpu.estimate_duration(request=_fake_request()) == BASE_SECONDS
    assert zerogpu.estimate_duration(loras=()) == BASE_SECONDS


def test_lora_headroom_is_additive_after_the_stage_2_multiplier():
    """The payload read and injection walk happen once, before any denoising:
    their cost does not scale with the refinement chain, so the increment is
    added after the multiplier, not multiplied by it."""
    got = zerogpu.estimate_duration(
        request=_fake_request(enable_stage_2=True, loras=({"source": "a"},))
    )
    assert got == int(BASE_SECONDS * 1.5) + zerogpu.LORA_DURATION_SECONDS


def test_an_explicit_duration_is_still_scaled_and_padded():
    got = zerogpu.estimate_duration(
        gpu_duration_seconds=100,
        request=_fake_request(enable_stage_2=True, loras=({"source": "a"},)),
    )
    assert got == 150 + zerogpu.LORA_DURATION_SECONDS


# --------------------------------------------------------------------------
# the mmap report is WIRED -- pipelines reads it at load, before the pack
# --------------------------------------------------------------------------
#
# check_mmap_backed sat exported, tested and called by NOTHING, parked on a
# note that said "ZeroGPU is parked" while ZeroGPU became the live
# deployment -- the §8 shape, third instance. These tests pin the wiring:
# pipelines.mmap_backing_report walks the pipeline's parameter-bearing
# components through the same check the tests above specify, and
# _base_pipeline calls it before the device move, which is the last moment
# the pack's precondition can be seen.


class _FakePipelineObject:
    def __init__(self, **components):
        self.components = components


def test_the_report_covers_every_parameter_bearing_component():
    from ltx25_studio import pipelines

    pipe = _FakePipelineObject(
        transformer=_Module(**{"blk.0.weight": _Tensor(MAPPED)}),
        vae=_Module(**{"decoder.weight": _Tensor(MAPPED)}),
        scheduler=object(),  # no parameters, nothing to check
    )
    report = pipelines.mmap_backing_report(pipe, maps_text=MAPS)
    assert len(report) == 2
    assert all(checked and not offending for checked, offending, _ in report)
    named = "".join(message for _, _, message in report)
    assert "transformer" in named and "vae" in named


def test_an_offending_component_is_marked_offending():
    from ltx25_studio import pipelines

    pipe = _FakePipelineObject(
        transformer=_Module(**{"blk.7.weight": _Tensor(ANONYMOUS)}),
        vae=_Module(**{"decoder.weight": _Tensor(MAPPED)}),
    )
    report = pipelines.mmap_backing_report(pipe, maps_text=MAPS)
    assert len(report) == 2, "the clean component must still be reported, not eclipsed"
    offending_rows = [message for _, offending, message in report if offending]
    assert len(offending_rows) == 1
    assert "blk.7.weight" in offending_rows[0], "the offender must be named, not counted"


def test_no_maps_yields_one_could_not_check_row_not_a_quiet_pass():
    """'Could not check' must neither look like 'checked and fine' nor be
    repeated per component: the answer is the platform's, once."""
    from ltx25_studio import pipelines

    pipe = _FakePipelineObject(
        transformer=_Module(**{"a": _Tensor(MAPPED)}),
        vae=_Module(**{"b": _Tensor(MAPPED)}),
        audio_vae=_Module(**{"c": _Tensor(MAPPED)}),
    )
    report = pipelines.mmap_backing_report(pipe, maps_text=None)
    assert len(report) == 1
    checked, offending, message = report[0]
    assert checked is False and offending is False
    assert "NOT checked" in message


def test_a_pipeline_without_a_components_dict_is_walked_by_known_names():
    from ltx25_studio import pipelines

    class _BarePipe:
        transformer = _Module(**{"blk.0.weight": _Tensor(MAPPED)})

    report = pipelines.mmap_backing_report(_BarePipe(), maps_text=MAPS)
    assert len(report) == 1
    assert "transformer" in report[0][2]


def test_base_pipeline_runs_the_report_before_the_device_move():
    """The wiring claim itself, and its placement. Packing happens later, at
    ``demo.launch()``; but after ``.to(device)`` on ZeroGPU ``named_parameters()``
    returns the platform's fakes, so the report must read the tensors before the
    move or it can no longer see the precondition it exists to check."""
    import inspect

    from ltx25_studio import pipelines

    source = inspect.getsource(pipelines._base_pipeline)
    called_at = source.find("mmap_backing_report(")
    moved_at = source.find("built.to(device)")
    assert called_at != -1, "check_mmap_backed's wiring point -- do not re-park it silently"
    assert moved_at != -1
    assert called_at < moved_at
