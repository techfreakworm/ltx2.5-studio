"""The ZeroGPU boundary: the ``spaces`` shim, GPU decoration, and return safety.

PLAN §8. This module exists because ZeroGPU inverts normal GPU-box practice and
several of its rules are silent-failure rules.

  * ``import spaces`` must precede torch/diffusers/transformers, or startup dies
    with "CUDA has been initialized before importing the spaces package". The
    import here is guarded (``try/except ImportError``) so local dev works --
    ``requirements.txt`` deliberately does NOT pin ``spaces`` (PLAN §5.1).

  * ``@spaces.GPU(...)`` must be applied STATICALLY at module import. Dynamic or
    inner decoration fails the startup analyzer with "No @spaces.GPU function
    detected during startup".

  * **Never pass ``size=`` unguarded.** The parameter exists in ``spaces``
    0.51.x (the Space runs 0.51.3), but the platform pins its own version and we
    deliberately do not pin it. Against a platform build that predates it,
    ``@spaces.GPU(duration=..., size="xlarge")`` raises a **TypeError at module
    import**, which on a Space means the app simply never starts and looks like
    a mysterious failure rather than a version problem.

    ⛔ **What used to be written here was wrong, and it cost this build half its
    GPU.** The claim was: *"we do not need it anyway -- ZeroGPU's ``startup()``
    packs the weights and, past its 42.00 GiB threshold, sets
    ``_default_gpu_size`` to ``xlarge`` itself, so we are auto-promoted
    regardless."* The promotion is real and it does fire. It **does not reach the
    scheduler.** ``default_gpu_size()`` has exactly one consumer --
    ``client.py:109``, inside ``get_duration_seconds`` -- where it rebinds a
    *local* that feeds one branch: the ``duration_factor`` multiplier at
    ``client.py:111-112``. ``schedule()`` then forwards its **own** ``gpu_size``
    (still ``None``) to ``api.py:112-113``, which sends ``gpuSize`` **only when it
    is not None**. So ``auto`` sent no size at all and the platform allocated its
    default, ``large``: **47.375 GiB measured, against the 94.971 GiB of
    ``xlarge``.** The app ran its entire life on half a GPU, was reserved for as
    though it were on the whole one, and nobody noticed until a run finally
    printed ``total_memory``. Four-hop trace and the arithmetic:
    ``docs/MIG_ALLOCATOR_SIGHTINGS.md`` §1.

    So ``size`` is NOT optional, and ``LTX25_ZEROGPU_SIZE`` must be set to a real
    value -- ``xlarge`` is now the default. The defensive construction below is
    what makes that safe: build the decorator kwargs by checking
    ``inspect.signature(spaces.GPU).parameters``, include ``size`` only if the
    running ``spaces`` accepts it, log the version and the size VALUE passed
    (:func:`describe_gpu_kwargs`), and compare the partition the worker measures
    against the request (:func:`partition_note`).

    ⚠️ **And note what the filter does on an old ``spaces``: it silently drops
    ``size``, which is exactly the failure above.** It trades an import-time
    ``TypeError`` -- an unexplained dead Space -- for a half-size GPU that runs
    fine and says nothing. That is still the right trade, because one is
    recoverable by reading a log and the other is not, but it is a trade and not
    a free lunch. The startup log line is the only thing that distinguishes them.

  * **Never return CUDA tensors** across the boundary. Unpickling one in the main
    process triggers a blocked lazy CUDA init and THE CALL HANGS -- no error, no
    timeout, a dead request. Return file paths, or ``.cpu()``. Always. This
    module owns that boundary and it is a test case.

  * ``enable_model_cpu_offload()`` is NOT used -- see PLAN §8.4. It is
    counterproductive here, not merely suboptimal.

  * The ``duration=`` callable must accept the EXACT same positional signature as
    the decorated function, trailing kwargs included, or it TypeErrors at
    schedule time.

  * Quota attribution rides a contextvar (``gradio.context.LocalContext.request``)
    that does NOT cross a plain ``threading.Thread``. Without
    ``contextvars.copy_context()`` every user buckets as Space-anonymous and you
    get "Space app has reached its GPU limit" at zero minutes used.

  * Do NOT auto-retry a ZeroGPU timeout with a captured Gradio request: the
    ``X-IP-Token`` ages past TTL during attempt 1, so attempt 2 always 401s with
    "Expired ZeroGPU proxy token" -- masquerading as the root cause.

The Space hardware identifier is ``zero-a10g``. ``huggingface_hub``'s
``SpaceHardware`` enumerates ZeroGPU as exactly that string, and it is what the
README frontmatter must say. It is a **legacy identifier** -- the hardware is
Blackwell (RTX PRO 6000 Server Edition via MIG), not an A10G. The same
enumeration contains ``h200``, ``h200x2`` and friends, so nobody should ever
"correct" ``zero-a10g`` to an h200 value: that moves the Space off ZeroGPU onto
paid dedicated hardware.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from . import constraints
from .errors import LTXStudioError

__all__ = [
    "SPACE_HARDWARE",
    "AUTOPRUNE_PATTERN",
    "assert_returnable",
    "check_duration_signature",
    "describe_gpu_kwargs",
    "partition_note",
    "quota_multiplier",
    "unbacked_parameters",
    "read_process_maps",
    "parse_maps",
    "check_mmap_backed",
    "backing_file",
    "estimate_duration",
    "ENHANCE_DURATION_SECONDS",
    "LORA_DURATION_SECONDS",
    "RunShape",
    "check_envelope",
    "describe_reservation",
    "ignored_duration_note",
    "run_shape",
    "step_seconds",
    "gpu_decorator_kwargs",
    "host_has_cuda",
    "is_zerogpu",
    "local_spaces_stub",
    "spaces_version",
    "will_use_gpu",
]

#: The Spaces hardware identifier for ZeroGPU. A **legacy** string -- the
#: hardware is Blackwell (RTX PRO 6000 Server Edition via MIG), not an A10G.
#: ``huggingface_hub``'s ``SpaceHardware`` enumerates it as exactly this, and the
#: same enumeration contains ``h200``/``h200x2``: "correcting" this value moves
#: the Space off ZeroGPU onto paid dedicated hardware.
SPACE_HARDWARE = "zero-a10g"

#: Recursion bound for the return check. Deep enough for any real pipeline
#: output, shallow enough that a pathological object cannot hang the boundary.
_MAX_DEPTH = 8

_ATOMIC = (str, bytes, bytearray, int, float, bool, complex, type(None))


class _LocalSpacesStub:
    """Stand-in for ``spaces`` off-Space, so the wiring is identical everywhere.

    The point is that ``app.py`` can write a literal ``@spaces.GPU(...)`` -- the
    form ZeroGPU's startup analyzer looks for -- without that line becoming
    conditional. A conditional decorator is how "we wrote the rule down and then
    did not apply it" happens: the Space path stops being the path anyone runs.

    Locally ``GPU(...)`` returns the function unchanged.
    """

    __version__ = "local-stub"
    is_stub = True

    @staticmethod
    def GPU(*, duration=None, size=None, **kwargs):  # noqa: N802 - mirrors spaces.GPU
        def decorate(func):
            return func

        return decorate


def local_spaces_stub() -> _LocalSpacesStub:
    """The off-Space stand-in. See :class:`_LocalSpacesStub`."""
    return _LocalSpacesStub()


def is_zerogpu(spaces_module) -> bool:
    """True only when the real platform package is present."""
    return spaces_module is not None and not getattr(spaces_module, "is_stub", False)


# ---------------------------------------------------------------------------
# The mmap-backing guard, reported (log-only) by `pipelines.mmap_backing_report`
# from `_base_pipeline`. ZeroGPU-only in effect: macOS has no /proc maps.
# See docs/ZEROGPU_DISK.md for what it is for.
# ---------------------------------------------------------------------------

#: What ZeroGPU's post-pack cleanup unlinks, from `ZEROGPU_MMAP_AUTOPRUNE_PATTERN`.
#: We assert on the SAME basis it prunes on, rather than on a proxy for it.
AUTOPRUNE_PATTERN = "*/huggingface/hub/models--*/blobs/*"

#: Where the kernel reports this process's mappings. Linux only.
PROC_MAPS = "/proc/self/maps"


def parse_maps(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, path)`` for every FILE-BACKED region in a maps dump.

    Anonymous regions are dropped deliberately: a tensor living in one is
    exactly the failure we are looking for, and it shows up as an address that
    matches nothing here.
    """
    regions: list[tuple[int, int, str]] = []
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5].strip()
        if not path or path.startswith("["):
            continue
        bounds, _, _ = fields[0].partition(" ")
        start, _, end = bounds.partition("-")
        try:
            regions.append((int(start, 16), int(end, 16), path))
        except ValueError:
            continue
    return regions


def backing_file(address: int, regions) -> str | None:
    """The file an address is mapped from, or None if it is anonymous."""
    for start, end, path in regions:
        if start <= address < end:
            return path
    return None


def unbacked_parameters(module, regions, *, pattern: str = AUTOPRUNE_PATTERN) -> list[str]:
    """Named parameters whose storage is NOT backed by a prunable blob.

    ZeroGPU's post-pack cleanup cannot attribute these to a blob it may unlink.
    """
    import fnmatch

    offenders = []
    for name, tensor in module.named_parameters():
        try:
            address = tensor.data_ptr()
        except Exception:
            continue
        path = backing_file(address, regions)
        if path is None or not fnmatch.fnmatch(path, pattern):
            offenders.append(f"{name} ({'anonymous memory' if path is None else path})")
    return offenders


def read_process_maps(path: str = PROC_MAPS) -> str | None:
    """The maps dump, or None where the kernel does not provide one (macOS)."""
    try:
        with open(path) as handle:
            return handle.read()
    except OSError:
        return None


def check_mmap_backed(module, *, name: str, maps_text: str | None = None) -> tuple[bool, str]:
    """Are this module's tensors still mmap'd from prunable blobs?

    ZeroGPU reclaims the 72 GB HF snapshot after packing by walking
    ``/proc/self/maps``, mapping each tensor's ``data_ptr()`` back to the blob it
    came from, and unlinking those. **Anything that materialises tensors into
    anonymous memory** -- a dtype cast, a ``.clone()``, an eager
    ``low_cpu_mem_usage=False``, a non-safetensors path -- leaves nothing to
    capture. The prune then frees nothing, and the Space dies **at steady state**
    with 72 GB it cannot account for and **no error explaining why**.

    Returns ``(checked, message)``. ``checked`` is False where the kernel gives
    us no maps to read -- reported rather than passed silently, because "could
    not check" and "checked and fine" must never look the same.
    """
    text = maps_text if maps_text is not None else read_process_maps()
    if text is None:
        return False, (
            f"mmap backing for {name} was NOT checked: {PROC_MAPS} is unavailable on this "
            f"platform. The check is meaningful only where ZeroGPU packs."
        )
    offenders = unbacked_parameters(module, parse_maps(text))
    if offenders:
        shown = ", ".join(offenders[:3])
        more = f" and {len(offenders) - 3} more" if len(offenders) > 3 else ""
        return True, (
            f"{len(offenders)} parameters of {name} are not mmap'd from prunable HF blobs: "
            f"{shown}{more}. ZeroGPU's post-pack cleanup unlinks only file-backed tensors, so "
            f"the blobs behind these may not be freed. int8-quantised weights are materialised "
            f"by construction (the 2026-11-13 boot still freed 65.25 GB); anything else was "
            f"cast, cloned or eagerly loaded, and its dtype must match the checkpoint's."
        )
    return True, f"mmap backing for {name}: all parameters file-backed and prunable"


def host_has_cuda() -> bool:
    """Does this machine have a usable CUDA device?

    **Never call this on a Space.** Importing torch and touching
    ``torch.cuda.is_available()`` at module scope is exactly what the
    ``import spaces``-before-torch rule exists to prevent -- ZeroGPU raises
    "CUDA has been initialized before importing the spaces package". Which is
    why :func:`will_use_gpu` short-circuits on ``spaces`` first and this is only
    ever reached off-Space, where the probe is harmless.

    Returns False rather than raising when torch is absent, so a dev box with no
    model stack still imports the app.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def will_use_gpu(spaces_module) -> bool:
    """Will this process do GPU work? A **capability** question, not identity.

    The distinction is the whole point. Gating on "are we on ZeroGPU" meant a
    real 97 GB GPU box downloaded nothing, built nothing and moved nothing to
    CUDA, because it simply was not a Space. The same lines now do the right
    thing in both places: on ZeroGPU ``.to("cuda")`` is intercepted and packs to
    disk, on a plain CUDA box it is a real VRAM move. That difference belongs to
    the platform, not to us.

    **The ordering is load-bearing, not stylistic.** ``spaces`` is checked first
    so that on a Space we never probe CUDA at all -- see :func:`host_has_cuda`.
    """
    if is_zerogpu(spaces_module):
        return True
    return host_has_cuda()


# ---------------------------------------------------------------------------
# the reservation: what one run asks ZeroGPU for, from its size and length
# ---------------------------------------------------------------------------
#
# Measured on the Space (xlarge, int8, mode 05, seed 7, the diffusion decoder). Window =
# the Space log's schedule -> release; tokens N = (W/32) x (H/32) x ((F - 1)/8 + 1); the
# Upscale pass runs 3 steps at 2W x 2H. "reserved" is what the model below asks for.
#
#   run                           N pass 1 / 2       s/step 1 / 2    window  reserved
#   704x480x121, Upscale    M1     5,280 / 21,120     1.49 / 5.17        52       105
#   704x480x481             M2    20,130              4.83               65       115
#   1408x960x121            M3    21,120              5.14               69       115
#   1280x704x241            M4    27,280              6.80               88       145
#   704x480x241, Upscale    M5    10,230 / 40,920     ~2.6 / 11.12      100       165
#   960x544x241, Upscale    M6    15,810 / 63,240     3.79 / 19.96      154       240
#
# Deploy 1 (24 calls, 2026-11-16): windows 10.5-51.6 s at 704x480x121 or smaller, and
# 512x960x193 (12,000 tokens, 2.98 s/step) in 36.5 s; a fresh worker's first call put
# 25 s before the first step (the diffusion decoder's build); Enhance added 27 s of
# window, an adapter 5-15 s, the upsample between the passes 1-2 s. Every one of the 30
# runs is reserved at least 1.3 x its window (tests/test_reservation.py).

#: Seconds per denoise step at N tokens, ``t(N) = A + B x N + C x N^2``: least squares
#: through M1, M3 and M5 (5,280 / 21,120 / 40,920 tokens), within 0.2 s of every other
#: measured point.
STEP_SECONDS_AT_ZERO = 0.46
STEP_SECONDS_PER_TOKEN = 1.81e-4
STEP_SECONDS_PER_TOKEN_SQUARED = 1.94e-9

#: Before the first step: 5-8 s warm, up to 25 s measured on a fresh worker's first call
#: while it still built the diffusion decoder per run. The decoder is built at start-up
#: now; 25 s stays, for a cold first call.
PRE_SECONDS = 25.0
#: The latent upsample between the two passes: 1-2 s measured.
UPSAMPLE_SECONDS = 3.0
#: After the last step (decode, mux, write): this, plus the next per returned pixel-frame
#: (fitted 1.25e-7; 1.29e-7 at 241 frames).
POST_SECONDS = 1.0
POST_SECONDS_PER_PIXEL_FRAME = 1.3e-7
#: Enhance added 27 s of window with the enhancer's weights already on disk. The image
#: preloads them (``preload_from_hub``, with the start-up prefetch as the fallback), so a
#: first enhanced call does not pay the ~88 s Hub fetch inside its reservation;
#: LTX25_GPU_DURATION_FLOOR_SECONDS is the lever if one ever does.
ENHANCE_WINDOW_SECONDS = 60.0
#: An adapter added 5-15 s (the payload read and the injection walk). Flat for any number
#: of them: stacked adapters share one injection walk.
LORA_WINDOW_SECONDS = 30.0

#: The margin over the modelled window, the step the reservation is rounded up to, and
#: the smallest reservation.
RESERVATION_MARGIN = 1.3
RESERVATION_STEP_SECONDS = 5
MIN_RESERVATION_SECONDS = 45


def _reserved(window_seconds: float) -> int:
    """``window_seconds`` times the margin, rounded up to the reservation step."""
    scaled = round(float(window_seconds) * RESERVATION_MARGIN, 6)
    return int(math.ceil(scaled / RESERVATION_STEP_SECONDS) * RESERVATION_STEP_SECONDS)


#: What staged LoRA adapters add to the reservation (any number of them).
LORA_DURATION_SECONDS = _reserved(LORA_WINDOW_SECONDS)

#: What prompt enhancement adds to the reservation.
ENHANCE_DURATION_SECONDS = _reserved(ENHANCE_WINDOW_SECONDS)

#: Quota/billing factor per ZeroGPU size. Source: the Hugging Face ZeroGPU docs
#: (xlarge draws quota at twice large's rate); absent from the spaces client
#: (docs/TEST_PLAN.md B.1), so documented rather than measured. ``auto`` sends no
#: size, and the platform default is ``large``.
_QUOTA_MULTIPLIERS = {"large": 1, "xlarge": 2, "auto": 1}


def quota_multiplier(size: str | None = None) -> int:
    """How many quota seconds one GPU second costs at ``size``.

    ``None`` reads the configured ``LTX25_ZEROGPU_SIZE``. Torch-free and
    import-safe: the UI's cost panel calls it in the web process.
    """
    from .config import ENV_ZEROGPU_SIZE, ZEROGPU_SIZES, load_config
    from .errors import ConfigError

    if size is None:
        size = load_config().zerogpu_size
    try:
        return _QUOTA_MULTIPLIERS[size]
    except KeyError:
        raise ConfigError(variable=ENV_ZEROGPU_SIZE, value=size, legal=ZEROGPU_SIZES) from None


def _request_field(params, name):
    """Read a field from the call as the LIVE dispatch actually shapes it.

    ``generate()`` routes through the entrypoint as ``dispatch(request=
    GenerationRequest(...), ...)``, so the request's fields arrive on the
    ``request`` object -- NOT as top-level kwargs. An estimator that reads
    only ``params.get(name)`` is an advertised lever attached to nothing on
    the live path (the §8 defect class): it resolves fine, changes no
    behaviour, and every live run reserves the base duration. Top-level
    kwargs are still honoured first for direct callers.
    """
    value = params.get(name)
    if value is None:
        value = getattr(params.get("request"), name, None)
    return value


@dataclass(frozen=True)
class RunShape:
    """A run as it will execute: what it renders, for how long, and its passes."""

    width: int
    height: int
    frames: int
    frame_rate: float
    #: Denoise steps of the first pass (a clip variation runs only the schedule's tail).
    steps: int
    #: How much the Upscale pass enlarges each axis; None for one pass.
    stage_2_scale: int | None

    @property
    def tokens(self) -> int:
        return constraints.latent_tokens(self.width, self.height, self.frames)

    @property
    def stage_2_tokens(self) -> int | None:
        if not self.stage_2_scale:
            return None
        scale = self.stage_2_scale
        return constraints.latent_tokens(self.width * scale, self.height * scale, self.frames)

    @property
    def pixel_frames(self) -> int:
        """Pixel-frames of the returned video."""
        scale = self.stage_2_scale or 1
        return self.width * scale * self.height * scale * self.frames


def step_seconds(tokens: int) -> float:
    """Seconds one denoise step takes at ``tokens`` (the fit above)."""
    n = float(tokens)
    return (
        STEP_SECONDS_AT_ZERO + STEP_SECONDS_PER_TOKEN * n + STEP_SECONDS_PER_TOKEN_SQUARED * n * n
    )


def run_shape(**params) -> RunShape:
    """The run the estimator's arguments describe, read as it will really execute.

    Takes what :func:`estimate_duration` takes: a ``request`` (the live dispatch's
    shape) and/or request fields at top level, top level first. A sound-only mode
    renders at its pinned size; a mode that skips Upscale never runs it; a model-picked
    length is read at its longest end; a recording's or a clip's length is the one the
    gate set, and the longest the app allows until the gate has measured it; a clip
    variation runs only the schedule's tail from its level.
    """
    from . import audio_conditioning, modes
    from .generate import LENGTH_FROM_VIDEO, GenerationRequest
    from .pipelines import STAGE_2_SPATIAL_SCALE

    defaults = vars(GenerationRequest(mode=""))

    def field(name):
        value = _request_field(params, name)
        return defaults.get(name) if value is None else value

    spec = modes.MODES.get(field("mode"))
    width, height = int(field("width")), int(field("height"))
    if spec is not None and spec.min_resolution:
        width = height = modes.MIN_RESOLUTION
    frame_rate = float(field("frame_rate"))
    variation = spec is not None and spec.video_conditioning == "variation"
    if variation and (field("extras") or {}).get("length_source") != LENGTH_FROM_VIDEO:
        frames = constraints.MAX_NUM_FRAMES  # the clip's own length, not measured yet
    elif field("auto_duration") and spec is not None and modes.requires(spec, "audio"):
        frames = constraints.MAX_NUM_FRAMES  # the recording's own length, not measured yet
    elif field("auto_duration"):
        _shortest, frames = constraints.duration_head_frame_range(
            min_seconds=float(field("min_seconds")),
            max_seconds=float(field("max_seconds")),
            frame_rate=frame_rate,
        )
    else:
        frames = int(field("num_frames"))
    steps = len(constraints.DISTILLED_SIGMA_VALUES)
    if variation:
        level = field("noise_scale")
        level = audio_conditioning.DEFAULT_VIDEO_LEVEL if level is None else level
        steps = len(audio_conditioning.sigma_tail(audio_conditioning.snap_video_level(level)))
    upscale = bool(field("enable_stage_2")) and not (spec is not None and spec.skip_stage_2)
    return RunShape(
        width=width,
        height=height,
        frames=max(1, frames),
        frame_rate=frame_rate,
        steps=steps,
        stage_2_scale=STAGE_2_SPATIAL_SCALE if upscale else None,
    )


def _run_window(shape: RunShape) -> float:
    """Seconds of window for the first pass and the post at its size."""
    return (
        PRE_SECONDS
        + shape.steps * step_seconds(shape.tokens)
        + POST_SECONDS
        + POST_SECONDS_PER_PIXEL_FRAME * shape.width * shape.height * shape.frames
    )


def _upscale_window(shape: RunShape) -> float:
    """Seconds of window the Upscale pass adds: the upsample, its steps, the larger post."""
    steps = len(constraints.STAGE_2_DISTILLED_SIGMA_VALUES)
    single = shape.width * shape.height * shape.frames
    return (
        UPSAMPLE_SECONDS
        + steps * step_seconds(shape.stage_2_tokens)
        + POST_SECONDS_PER_PIXEL_FRAME * (shape.pixel_frames - single)
    )


def _floor_seconds(params) -> int | None:
    """``LTX25_GPU_DURATION_FLOOR_SECONDS``, or the caller's own ``gpu_duration_floor_seconds``."""
    if "gpu_duration_floor_seconds" in params:
        return params["gpu_duration_floor_seconds"]
    from .config import load_config

    try:
        return load_config().gpu_duration_floor_seconds
    except Exception:  # noqa: BLE001 -- startup already refused a bad value
        return None


def estimate_duration(**params) -> int:
    """Seconds to reserve for one generation, from its size and length.

    **Its signature must match the decorated function exactly** -- ZeroGPU calls
    the estimator with that function's own arguments, so a drifted signature
    raises at *schedule* time, once the user is already waiting. Both take
    ``**params``, and ``check_duration_signature`` asserts it at import.

    The modelled window (constants above): the run, ``PRE + steps x t(N1) + POST +
    POST_PER_PIXEL_FRAME x W x H x F``; Upscale, ``UPSAMPLE + 3 x t(N2)`` and the larger
    post; LoRA and Enhance, their flat seconds. Each term is reserved at
    ``RESERVATION_MARGIN`` times its window, rounded up to 5 s, so an extra always adds
    the seconds the cost panel names (the sum is at most 15 s above one rounding of the
    whole); the run term is never below ``MIN_RESERVATION_SECONDS``.
    ``LTX25_GPU_DURATION_FLOOR_SECONDS`` raises the total to at least that.

    Reservation and billing are different meters: the pre-check compares the
    **requested** duration against remaining quota, while billing is on the
    **effective** duration times :func:`quota_multiplier`. So an over-generous
    request costs nothing in billing and everything in schedulability.
    """
    try:
        shape = run_shape(**params)
    except (TypeError, ValueError) as exc:
        # Never raise at schedule time: the gate refuses such a request in the worker.
        import logging

        logging.getLogger(__name__).warning("reservation for an unreadable request: %s", exc)
        shape = run_shape()
    seconds = max(MIN_RESERVATION_SECONDS, _reserved(_run_window(shape)))
    if shape.stage_2_scale:
        seconds += _reserved(_upscale_window(shape))
    if _request_field(params, "loras"):
        seconds += LORA_DURATION_SECONDS
    if _request_field(params, "enhance_prompt"):
        # Enhancement runs inside the worker (`enhance.py`), so its time is reserved time.
        seconds += ENHANCE_DURATION_SECONDS
    return int(max(seconds, _floor_seconds(params) or 0))


def check_envelope(**params) -> None:
    """Raise ``constraints.BeyondEnvelopeError`` for a run past what the Space was measured
    to finish. Takes :func:`estimate_duration`'s arguments; the Space's gate calls it."""
    shape = run_shape(**params)
    constraints.validate_envelope(
        shape.width,
        shape.height,
        shape.frames,
        stage_2_scale=shape.stage_2_scale,
        frame_rate=shape.frame_rate,
    )


def describe_reservation(runtime_config) -> str:
    """The reservation rule as VALUES, for the startup log."""
    from .config import ENV_GPU_DURATION_FLOOR_SECONDS

    shape = run_shape()
    text = (
        f"reservation from each run's size and length ("
        f"{estimate_duration(gpu_duration_floor_seconds=None)} s for the default "
        f"{shape.width}x{shape.height}x{shape.frames} run, more for larger, longer or "
        f"upscaled runs; +{LORA_DURATION_SECONDS} s with LoRAs, +{ENHANCE_DURATION_SECONDS} s "
        f"with prompt enhancement)"
    )
    floor = getattr(runtime_config, "gpu_duration_floor_seconds", None)
    if floor:
        text += f", at least {floor} s ({ENV_GPU_DURATION_FLOOR_SECONDS})"
    return text


def ignored_duration_note(runtime_config) -> str | None:
    """One startup line when ``LTX25_GPU_DURATION_SECONDS`` is still set, or None."""
    from .config import ENV_GPU_DURATION_FLOOR_SECONDS, ENV_GPU_DURATION_SECONDS

    value = getattr(runtime_config, "ignored_gpu_duration_seconds", None)
    if not value:
        return None
    return (
        f"{ENV_GPU_DURATION_SECONDS}={value} is ignored: each run's reservation now follows "
        f"its size and length. {ENV_GPU_DURATION_FLOOR_SECONDS} sets a minimum if an "
        f"estimate ever proves short."
    )


def spaces_version(spaces_module) -> str | None:
    """The platform's ``spaces`` version, for the startup log.

    Takes the module rather than importing it: ``spaces`` is a deploy-environment
    package, and importing it from library code makes the library unimportable
    off-Space. The real package defines no ``__version__``, so the installed
    distribution's metadata answers; the local stub carries its own.
    """
    if spaces_module is None:
        return None
    version = getattr(spaces_module, "__version__", None)
    if version:
        return str(version)
    import importlib.metadata

    try:
        return importlib.metadata.version("spaces")
    except importlib.metadata.PackageNotFoundError:
        return None


#: Below this, a partition is not the ~95 GiB ``xlarge`` slice (``large`` is
#: 47.375 GiB measured).
XLARGE_MIN_GIB = 80.0


def partition_note(total_bytes, requested_size) -> str | None:
    """A plain warning when the measured GPU partition is not what was asked for.

    ``total_bytes`` is the worker's ``total_memory`` reading; ``None`` (not
    measured, e.g. off CUDA) says nothing.
    """
    if total_bytes is None:
        return None
    gib = total_bytes / 1024**3
    if gib >= XLARGE_MIN_GIB:
        return None
    if requested_size == "xlarge":
        return (
            f"xlarge was requested but this GPU slice has {gib:.1f} GiB (xlarge is about "
            f"95 GiB). Large videos and stage 2 may run out of memory on it."
        )
    if requested_size in (None, "auto"):
        return (
            f"no GPU size was requested, so the platform gave its default slice: {gib:.1f} "
            f"GiB, about half of xlarge. Set LTX25_ZEROGPU_SIZE=xlarge for the full GPU."
        )
    return None


def describe_gpu_kwargs(kwargs, runtime_config) -> str:
    """The ``@spaces.GPU`` arguments as VALUES, for the startup log."""
    from .config import ENV_ZEROGPU_SIZE

    duration = kwargs.get("duration")
    if callable(duration):
        name = getattr(duration, "__name__", "callable")
        seconds = f"duration={name} ({describe_reservation(runtime_config)})"
    else:
        seconds = f"duration={duration} s"
    if "size" not in kwargs:
        size = (
            f"size NOT passed: the installed spaces does not accept it, so the platform "
            f"default (large) applies whatever {ENV_ZEROGPU_SIZE} says"
        )
    elif kwargs["size"] is None:
        size = (
            f"size not requested ({ENV_ZEROGPU_SIZE}={runtime_config.zerogpu_size}): the "
            f"platform default (large, about half the GPU) applies"
        )
    else:
        size = f"size={kwargs['size']}"
    return f"{size}, {seconds}"


def _looks_like_a_cuda_tensor(value) -> bool:
    """True for anything a real ``torch.Tensor`` on CUDA would answer True to.

    A torch object's ``device`` is a ``torch.device``, never a ``str``: a plain
    string is a label (the run record names the device it ran on), not a tensor.
    """
    if getattr(value, "is_cuda", False):
        return True
    device = getattr(value, "device", None)
    return device is not None and not isinstance(device, str) and str(device).startswith("cuda")


def _find_cuda(value, depth: int, seen: set[int]):
    """Depth-first hunt for a device tensor anywhere in a return value."""
    if depth > _MAX_DEPTH or isinstance(value, _ATOMIC):
        return None
    marker = id(value)
    if marker in seen:
        return None
    seen.add(marker)

    if _looks_like_a_cuda_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _find_cuda(item, depth + 1, seen)
            if found is not None:
                return found
        return None
    if isinstance(value, Sequence) and not isinstance(value, _ATOMIC):
        for item in value:
            found = _find_cuda(item, depth + 1, seen)
            if found is not None:
                return found
        return None
    # Pipeline outputs are plain objects holding tensors on attributes.
    for item in vars(value).values() if hasattr(value, "__dict__") else ():
        found = _find_cuda(item, depth + 1, seen)
        if found is not None:
            return found
    return None


def assert_returnable(value):
    """Raise if anything in ``value`` is a CUDA tensor.

    Returning one across the ``@spaces.GPU`` boundary makes unpickling in the
    main process trigger a blocked lazy CUDA init, and **the call hangs** -- no
    error, no timeout, a dead request. It is the worst failure shape in the app
    because there is nothing to read afterwards, so a loud refusal here is
    strictly better.

    Generation returns a container, not a bare tensor, so the search is
    recursive: a boundary check that only looks at the top level never fires.
    """
    found = _find_cuda(value, 0, set())
    if found is None:
        return value
    raise LTXStudioError(
        f"refusing to return a CUDA tensor ({type(found).__name__} on "
        f"{getattr(found, 'device', 'cuda')}) across the ZeroGPU boundary: unpickling it in the "
        f"main process triggers a blocked lazy CUDA init and the request hangs with no error. "
        f"Return a file path, or move the tensor with .cpu() first."
    )


def gpu_decorator_kwargs(gpu_callable, **kwargs) -> dict:
    """Keep only the kwargs the installed ``spaces.GPU`` actually accepts.

    The platform pins its own ``spaces`` and we deliberately do not pin it.
    Passing ``size`` to a build that predates the parameter raises ``TypeError``
    **at module import**, which on a Space looks like a mysterious startup
    failure rather than a version problem.

    ⛔ **Dropping ``size`` does NOT cost nothing, whatever this docstring used to
    say.** The old text claimed ZeroGPU's ``startup()`` promotion made the kwarg
    redundant. It does not: the promoted value is read only by
    ``get_duration_seconds`` (``client.py:109``) and never reaches the
    ``gpuSize`` parameter the scheduler actually allocates on
    (``api.py:112-113``). Sending no size got us ``large`` -- ✅ 47.375 GiB
    measured, half of ``xlarge``'s 94.971 GiB -- for the life of the project. See
    the module docstring and ``docs/MIG_ALLOCATOR_SIGHTINGS.md`` §1.

    This filter therefore has a **silent** failure mode as well as the loud one
    it was built for: against a ``spaces`` too old for ``size``, it drops the
    kwarg and the Space runs at half capacity rather than not starting. Still the
    right trade -- a half-size GPU is diagnosable and a dead import is not -- but
    the caller must log what was passed, which ``app.py`` does with
    :func:`describe_gpu_kwargs`.
    """
    try:
        accepted = inspect.signature(gpu_callable).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in accepted}


def _signature_of(func) -> list[tuple[str, object]]:
    return [(name, p.kind) for name, p in inspect.signature(func).parameters.items()]


def check_duration_signature(func, duration) -> None:
    """Raise unless a callable ``duration=`` matches ``func`` exactly.

    ZeroGPU calls the estimator with the decorated function's own arguments, so a
    drifted signature TypeErrors **at schedule time** -- after the user has
    waited and after the reservation was made. A constant duration has no
    signature to drift and is always fine.
    """
    if not callable(duration):
        return
    wanted, given = _signature_of(func), _signature_of(duration)
    if wanted == given:
        return
    raise LTXStudioError(
        f"the duration estimator does not match the function it estimates for. "
        f"{getattr(func, '__name__', func)}{inspect.signature(func)} vs "
        f"{getattr(duration, '__name__', duration)}{inspect.signature(duration)}. "
        f"ZeroGPU calls the estimator with the decorated function's own arguments, so a "
        f"mismatch raises at schedule time, once the user is already waiting."
    )
