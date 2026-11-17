"""Which device generates, and what that device needs.

``LTX25_DEVICE`` = ``auto`` (default) | ``cuda`` | ``mps`` | ``cpu``.

* ``auto`` is CUDA on ZeroGPU or wherever CUDA exists, and **no generation
  device** anywhere else: the UI starts, and Generate raises
  :class:`GenerationDisabledError` saying how to opt in. Without this a machine
  with the weights on disk built the whole ~70 GB pipeline on the CPU at the
  first click.
* ``mps`` and ``cpu`` are explicit opt-ins for local runs. Neither has a
  bitsandbytes int8 path, so int8 there is a startup error.

Everything device-specific that the pipeline needs lives here: the dtype per
device, the RoPE handling MPS needs (it has no float64), the MPS numerics
self-test, and the size limit for stage 2's decoder off CUDA.

On ZeroGPU nothing here probes an accelerator: :func:`resolve` answers from the
``spaces`` module alone, because touching ``torch.cuda`` in the web process is
what the platform refuses.
"""

from __future__ import annotations

import inspect
import logging
import sys
from dataclasses import dataclass

from . import zerogpu
from .errors import ConfigError, ConstraintError, GenerationError, LTXStudioError

__all__ = [
    "DeviceConfigError",
    "DeviceResolution",
    "GenerationDisabledError",
    "MpsNumericsError",
    "MpsSelfTest",
    "OFF_CUDA_DECODER_TOKEN_LIMIT",
    "SelfTestCheck",
    "StageTwoNeedsCudaError",
    "check_stage_2_size",
    "compute_rope_on_cpu",
    "current",
    "host_has_mps",
    "MPS_CLEAN_TORCH",
    "mps_self_test",
    "require_mps_numerics",
    "unverified_note",
    "reset_mps_self_test",
    "resolve",
    "stage_2_decoder_tokens",
]


def _log():
    return logging.getLogger("ltx25_studio")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class DeviceConfigError(ConfigError):
    """A recognised setting that cannot work on this machine, with the reason."""

    def __init__(self, *, variable, value, legal, why):
        self.variable = variable
        self.value = value
        self.legal = tuple(legal)
        self.why = why
        LTXStudioError.__init__(
            self,
            f"{variable}={value!r} cannot be used here: {why} Use one of: {', '.join(self.legal)}.",
        )


class GenerationDisabledError(LTXStudioError):
    """This machine has no generation device; the UI runs, Generate does not."""

    def __init__(self, *, reason):
        self.reason = reason
        super().__init__(
            "Generation is not enabled on this machine: it has no NVIDIA (CUDA) GPU, and "
            "running the model on Apple silicon or on the CPU is something you switch on "
            "yourself. Set LTX25_DEVICE=mps (Apple silicon) or LTX25_DEVICE=cpu and restart "
            "the app. The full model needs roughly 70 GB of memory at 16-bit precision (Apple "
            "silicon), and about twice that on the CPU at its default 32-bit precision "
            "(LTX25_CPU_DTYPE=bf16 runs the CPU at 16-bit)."
        )


class MpsNumericsError(GenerationError):
    """The MPS self-test found this PyTorch computing wrong results on this Mac."""

    def __init__(self, *, checks, torch_version):
        self.checks = tuple(checks)
        self.torch_version = torch_version
        what = "; ".join(_PLAIN_CHECK_NAMES.get(name, name) for name in self.checks)
        super().__init__(
            f"Generation on this Mac's GPU is turned off: this PyTorch ({torch_version}) "
            f"computes Apple-GPU {what} wrongly, which would produce a broken video without "
            f"any error. Use PyTorch {MPS_CLEAN_TORCH} or newer for local runs, or generate on "
            f"the CPU with LTX25_DEVICE=cpu, or set LTX25_MPS_SELF_TEST=warn to proceed "
            f"unverified."
        )


class StageTwoNeedsCudaError(ConstraintError):
    """Stage 2's diffusion decoder at this size needs the CUDA build."""

    def __init__(self, *, device, tokens, limit):
        self.device = device
        self.tokens = tokens
        self.limit = limit
        # The UI's words (UI review 2, B-1): its control is "Upscale ×2". The
        # position counts stay on the instance for the logs.
        super().__init__(f"Upscale ×2 needs a CUDA GPU; this machine runs on {device}.")


_PLAIN_CHECK_NAMES = {
    "sdpa_scale": "attention (the sdpa_scale check)",
    "bf16_linear": "bf16 matrix multiplication (the bf16_linear check)",
}

#: The README's floor for local runs on Apple silicon: the SDPA scale-reuse
#: error is 8.8e-7 on 2.15.0 against 5.19 on 2.11.0, and 2.14.1 measured the
#: same as 2.15.0 in the numerics checks.
MPS_CLEAN_TORCH = "2.14"


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def host_has_mps() -> bool:
    """Is Apple's MPS backend usable here? False when torch is absent."""
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


@dataclass(frozen=True)
class DeviceResolution:
    """The generation device, why, and what the load must use on it."""

    requested: str
    #: ``"cuda"``, ``"mps"``, ``"cpu"``, or None: no generation on this machine.
    device: str | None
    reason: str
    on_zerogpu: bool
    #: The effective quantisation: ``"bf16"`` or ``"int8"``.
    quantization: str
    #: A ``torch`` dtype attribute name: ``"bfloat16"`` or ``"float32"``.
    weight_dtype: str

    @property
    def generation_enabled(self) -> bool:
        return self.device is not None

    def describe(self) -> str:
        if self.device is None:
            return f"none, UI only ({self.reason})"
        return f"{self.device} ({self.reason})"

    def annotate(self, result):
        """``result`` with the unverified-numerics warning added, when it applies.

        Only on MPS, only after a failed self-test that ``warn`` let through.
        """
        note = unverified_note() if self.device == "mps" else None
        warnings = getattr(result, "warnings", None)
        if note is None or warnings is None or note in warnings:
            return result
        import dataclasses

        return dataclasses.replace(result, warnings=(*warnings, note))

    def require_generation(self, *, injected: bool = False) -> str | None:
        """The device to build on, or raise when generation must not happen.

        ``injected`` is true when the caller supplies its own pipeline (the
        offline gate's fakes): nothing is built, so nothing is refused. On MPS
        the watermark pair and then the numerics (:func:`require_mps_numerics`).
        """
        if injected:
            return None
        if self.device is None:
            raise GenerationDisabledError(reason=self.reason)
        if self.device == "mps":
            require_mps_numerics()
        return self.device


def _config(runtime_config):
    if runtime_config is not None:
        return runtime_config
    from .config import load_config

    return load_config()


def resolve(spaces_module=None, runtime_config=None) -> DeviceResolution:
    """Resolve ``LTX25_DEVICE`` against this machine. Raises on impossible settings."""
    from .config import ENV_DEVICE

    cfg = _config(runtime_config)
    requested = cfg.device
    on_zerogpu = zerogpu.is_zerogpu(spaces_module)

    if on_zerogpu:
        # Answered from `spaces` alone: no torch, no CUDA probe in the web process.
        if requested not in ("auto", "cuda"):
            raise DeviceConfigError(
                variable=ENV_DEVICE,
                value=requested,
                legal=("auto", "cuda"),
                why="this is a ZeroGPU Space, which generates on its CUDA GPU only.",
            )
        device, reason = "cuda", "ZeroGPU Space"
    elif requested == "auto":
        if zerogpu.host_has_cuda():
            device, reason = "cuda", "CUDA GPU found"
        else:
            device, reason = (
                None,
                (
                    "no CUDA GPU; MPS and CPU generation are opt-in via LTX25_DEVICE=mps or "
                    "LTX25_DEVICE=cpu"
                ),
            )
    elif requested == "cuda":
        if not zerogpu.host_has_cuda():
            raise DeviceConfigError(
                variable=ENV_DEVICE,
                value=requested,
                legal=("auto", "mps", "cpu"),
                why="PyTorch sees no CUDA GPU on this machine.",
            )
        device, reason = "cuda", f"{ENV_DEVICE}=cuda"
    elif requested == "mps":
        if not host_has_mps():
            raise DeviceConfigError(
                variable=ENV_DEVICE,
                value=requested,
                legal=("auto", "cpu"),
                why="PyTorch reports no usable MPS (Apple silicon GPU) backend here.",
            )
        device, reason = "mps", f"{ENV_DEVICE}=mps"
    else:
        device, reason = "cpu", f"{ENV_DEVICE}=cpu"

    return DeviceResolution(
        requested=requested,
        device=device,
        reason=reason,
        on_zerogpu=on_zerogpu,
        quantization=_effective_quantization(cfg, device, on_zerogpu),
        weight_dtype=_weight_dtype(cfg, device),
    )


def current() -> DeviceResolution:
    """This process's resolution, read fresh.

    ZeroGPU is recognised by the platform package being imported (``app.py``
    imports it first); locally it is absent and the local rules apply.
    """
    return resolve(sys.modules.get("spaces"))


def _effective_quantization(cfg, device: str | None, on_zerogpu: bool) -> str:
    from .config import ENV_QUANTIZATION

    if cfg.quantization == "auto":
        return "int8" if on_zerogpu else "bf16"
    if cfg.quantization == "int8" and device in ("mps", "cpu"):
        raise DeviceConfigError(
            variable=ENV_QUANTIZATION,
            value="int8",
            legal=("auto", "bf16"),
            why=(
                f"int8 is bitsandbytes LLM.int8, which has no {device} kernels in this "
                f"build; it works on CUDA only."
            ),
        )
    return cfg.quantization


def _weight_dtype(cfg, device: str | None) -> str:
    from .pipelines import WEIGHT_DTYPE

    if device == "cpu" and cfg.cpu_dtype == "fp32":
        return "float32"
    return WEIGHT_DTYPE


def weight_dtype_for(device: str | None, runtime_config=None) -> str:
    """The ``torch`` dtype name a load on ``device`` uses."""
    return _weight_dtype(_config(runtime_config), device)


def quantization_for(device: str | None, runtime_config=None) -> str:
    """The effective quantisation for a load on ``device`` in this process."""
    on_zerogpu = zerogpu.is_zerogpu(sys.modules.get("spaces"))
    return _effective_quantization(_config(runtime_config), device, on_zerogpu)


# ---------------------------------------------------------------------------
# stage 2 off CUDA
# ---------------------------------------------------------------------------

#: The largest stage-5 decoder grid allowed off CUDA, in positions
#: (frames x H/4 x W/4 of the decoded video). Measured on the CPU's dense
#: FlexAttention path: 9216 (128x128x9 decoded) peaked at +2.1 GB; 17 frames at
#: that size passed 9 GB and was killed. Cost grows with the square of this.
OFF_CUDA_DECODER_TOKEN_LIMIT = 9216


def stage_2_decoder_tokens(width: int, height: int, num_frames: int) -> int:
    """Decoder positions for a stage-2 request of this size (output is 2x each axis)."""
    from .pipelines import STAGE_2_SPATIAL_SCALE

    scale = STAGE_2_SPATIAL_SCALE
    return int(num_frames) * (int(height) * scale // 4) * (int(width) * scale // 4)


def check_stage_2_size(device: str | None, width, height, num_frames) -> None:
    """Raise :class:`StageTwoNeedsCudaError` if this request cannot decode here."""
    if device in (None, "cuda") or not (width and height and num_frames):
        return
    tokens = stage_2_decoder_tokens(width, height, num_frames)
    if tokens > OFF_CUDA_DECODER_TOKEN_LIMIT:
        raise StageTwoNeedsCudaError(
            device=device, tokens=tokens, limit=OFF_CUDA_DECODER_TOKEN_LIMIT
        )


# ---------------------------------------------------------------------------
# RoPE on MPS: the float64 part runs on the CPU
# ---------------------------------------------------------------------------


def compute_rope_on_cpu(*targets) -> tuple[str, ...]:
    """Make every RoPE module under ``targets`` compute on the CPU.

    The LTX-2 RoPE builds its frequency grid in float64
    (``rope_double_precision``), which MPS does not have. Running the whole
    RoPE forward on the CPU keeps float64 -- the precision CUDA uses -- and the
    CPU's accurate ``cos``/``sin``; only the resulting tables move to the
    device. Upstream code is untouched: these are forward hooks on our loaded
    instances, so any loader works (``from_pretrained``, single-file, GGUF).

    ``targets`` are modules, ``(name, module)`` pairs, or pipelines (anything
    with a ``components`` dict). Idempotent: a module already handled is not
    hooked twice, so it is safe to call after every (re)load. Returns the names
    of every RoPE module now covered.
    """
    covered = []
    for root_name, root in _named_roots(targets):
        named_modules = getattr(root, "named_modules", None)
        if not callable(named_modules):
            continue
        for name, module in named_modules():
            if not hasattr(module, "double_precision"):
                continue
            if not getattr(module, "_ltx25_rope_cpu", False):
                module.register_forward_pre_hook(_rope_inputs_to_cpu, with_kwargs=True)
                module.register_forward_hook(_rope_outputs_to_device)
                module._ltx25_rope_cpu = True
            covered.append(f"{root_name}.{name}" if name else root_name)
    return tuple(covered)


def _named_roots(targets):
    for target in targets:
        if isinstance(target, tuple) and len(target) == 2 and isinstance(target[0], str):
            yield target
        elif isinstance(getattr(target, "components", None), dict):
            yield from target.components.items()
        else:
            yield type(target).__name__, target


def _rope_inputs_to_cpu(module, args, kwargs):
    import torch

    bound = inspect.signature(module.forward).bind(*args, **kwargs)
    target = bound.arguments.get("device")
    if target is None:
        target = next((a.device for a in bound.arguments.values() if torch.is_tensor(a)), None)
    module._ltx25_rope_target = target
    for key, value in list(bound.arguments.items()):
        if torch.is_tensor(value):
            bound.arguments[key] = value.to("cpu")
    if "device" in bound.signature.parameters:
        bound.arguments["device"] = torch.device("cpu")
    return bound.args, bound.kwargs


def _rope_outputs_to_device(module, args, output):
    target = getattr(module, "_ltx25_rope_target", None)
    if target is None:
        return output
    return tuple(tensor.to(target) for tensor in output)


# ---------------------------------------------------------------------------
# F21b: the MPS numerics self-test
# ---------------------------------------------------------------------------

#: Thresholds, relative max-abs error against a CPU float32 reference.
#:
#: SDPA runs in float32: MPS measures ~1e-6 against the CPU, and the torch 2.11
#: scale-reuse bug measures 1.6-10.9. 1e-3 sits three orders from each.
SDPA_RTOL = 1e-3
#: bf16 keeps 8 significant bits, so one rounding step is <= 2**-8 of a value.
#: Two identical calls may differ by at most that step.
LINEAR_REPEAT_RTOL = 2**-8
#: Against the fp32 reference, bf16 rounding of inputs, weights and output
#: measures ~1 step (0.3-0.4%); 2**-5 allows 8 steps before calling it wrong.
LINEAR_VS_CPU_RTOL = 2**-5

#: An odd shape no model attention uses: the test's first call is what the bug
#: caches for this shape, and it uses the default scale, so it poisons nothing.
_SDPA_SHAPE = (1, 1, 7, 8)


@dataclass(frozen=True)
class SelfTestCheck:
    name: str
    ok: bool
    detail: str
    #: A failed blocking check refuses generation; float64 is informational.
    blocking: bool = True


@dataclass(frozen=True)
class MpsSelfTest:
    torch_version: str
    checks: tuple[SelfTestCheck, ...]

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if c.blocking and not c.ok)

    @property
    def ok(self) -> bool:
        return not self.failed

    def describe(self) -> str:
        head = "passed" if self.ok else f"FAILED ({', '.join(self.failed)})"
        parts = "; ".join(
            f"{c.name} {'ok' if c.ok else ('FAIL' if c.blocking else 'no')}: {c.detail}"
            for c in self.checks
        )
        return f"{head} on torch {self.torch_version} -- {parts}"


_MPS_SELF_TEST: MpsSelfTest | None = None
_WARNED: set[str] = set()


def reset_mps_self_test() -> None:
    global _MPS_SELF_TEST
    _MPS_SELF_TEST = None
    _WARNED.clear()


def _relative(got, ref) -> float:
    return float((got.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-12))


def _check_sdpa(torch) -> SelfTestCheck:
    functional = torch.nn.functional
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(*_SDPA_SHAPE, generator=generator) for _ in range(3))
    errors = []
    for scale in (None, 1.0):
        ref = functional.scaled_dot_product_attention(q, k, v, scale=scale)
        got = functional.scaled_dot_product_attention(
            q.to("mps"), k.to("mps"), v.to("mps"), scale=scale
        ).cpu()
        errors.append(_relative(got, ref))
    return SelfTestCheck(
        "sdpa_scale",
        max(errors) <= SDPA_RTOL,
        f"relative error {errors[0]:.1e} (default scale), {errors[1]:.1e} (scale 1.0), "
        f"limit {SDPA_RTOL:.0e}",
    )


def _check_linear(torch) -> SelfTestCheck:
    functional = torch.nn.functional
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2, 33, 64, generator=generator)
    w = torch.randn(48, 64, generator=generator) * 0.1
    b = torch.randn(48, generator=generator) * 0.1
    ref = functional.linear(x, w, b)
    xm, wm, bm = (t.to("mps", torch.bfloat16) for t in (x, w, b))
    first = functional.linear(xm, wm, bm).cpu()
    second = functional.linear(xm, wm, bm).cpu()
    repeat, versus = _relative(second, first), _relative(first, ref)
    return SelfTestCheck(
        "bf16_linear",
        repeat <= LINEAR_REPEAT_RTOL and versus <= LINEAR_VS_CPU_RTOL,
        f"repeat {repeat:.1e} (limit {LINEAR_REPEAT_RTOL:.1e}), vs CPU {versus:.1e} "
        f"(limit {LINEAR_VS_CPU_RTOL:.1e})",
    )


def _check_float64(torch) -> SelfTestCheck:
    try:
        torch.zeros(1, dtype=torch.float64, device="mps")
    except Exception:
        return SelfTestCheck(
            "float64",
            False,
            "unavailable, so RoPE frequencies are computed on the CPU",
            blocking=False,
        )
    return SelfTestCheck("float64", True, "available", blocking=False)


def mps_self_test(*, force: bool = False) -> MpsSelfTest:
    """Run the MPS numerics checks once per process (~50 ms) and keep the result.

    Only ever called when the resolved device is ``mps``.
    """
    global _MPS_SELF_TEST
    if _MPS_SELF_TEST is not None and not force:
        return _MPS_SELF_TEST
    import torch

    checks = []
    for name, check, blocking in (
        ("sdpa_scale", _check_sdpa, True),
        ("bf16_linear", _check_linear, True),
        ("float64", _check_float64, False),
    ):
        try:
            checks.append(check(torch))
        except Exception as exc:  # noqa: BLE001 - a check that cannot run has failed
            checks.append(SelfTestCheck(name, False, f"could not run: {exc!r}", blocking))
    result = MpsSelfTest(torch_version=str(torch.__version__), checks=tuple(checks))
    _MPS_SELF_TEST = result
    (_log().info if result.ok else _log().warning)("MPS self-test %s", result.describe())
    return result


def unverified_note() -> str | None:
    """The user-facing warning for a run that ``warn`` let past a failed self-test."""
    from .config import load_config

    result = _MPS_SELF_TEST
    if result is None or result.ok or load_config().mps_self_test != "warn":
        return None
    what = ", ".join(_PLAIN_CHECK_NAMES.get(name, name) for name in result.failed)
    return (
        f"Unverified output: this PyTorch ({result.torch_version}) computes Apple-GPU {what} "
        f"wrongly, and LTX25_MPS_SELF_TEST=warn let the run go ahead. The result may be "
        f"wrong without any error; PyTorch {MPS_CLEAN_TORCH} or newer passes the self-test."
    )


def require_mps_numerics() -> MpsSelfTest:
    """The self-test result, or :class:`MpsNumericsError` under ``enforce``.

    The MPS allocator's watermark pair is checked first, whatever
    ``LTX25_MPS_SELF_TEST`` says: under a pair PyTorch refuses every GPU
    operation fails, the self-test's too, so it raises
    ``memory.MpsWatermarkError`` instead of blaming this PyTorch.
    """
    from . import memory
    from .config import load_config

    memory.mps_watermarks().require()
    result = mps_self_test()
    if result.ok:
        return result
    if load_config().mps_self_test == "warn":
        if result.describe() not in _WARNED:
            _WARNED.add(result.describe())
            _log().warning(
                "MPS self-test FAILED (%s) and LTX25_MPS_SELF_TEST=warn: generating anyway; "
                "output from this torch on this GPU is numerically unreliable",
                ", ".join(result.failed),
            )
        return result
    raise MpsNumericsError(checks=result.failed, torch_version=result.torch_version)
