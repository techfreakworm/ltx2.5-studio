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
from collections.abc import Mapping, Sequence

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


#: Reservation headroom for a run with staged LoRA adapters, in seconds.
#: Inside the billed window a LoRA run additionally reads the adapter payload
#: off gradio's tmp disk (up to the 12 GB validator ceiling, though real
#: adapters measure 0.2-0.4 GB -- seconds of I/O) and runs peft injection
#: over ~3320 transformer modules. Both are seconds-to-low-tens, not minutes;
#: 60 covers them with headroom. Deliberately flat rather than per-adapter:
#: over-reserving costs schedulability, not billing, and stacked adapters
#: share one injection walk.
LORA_DURATION_SECONDS = 60

#: Reservation headroom for a run with prompt enhancement on, in seconds.
#:
#: **Sized for the DEGRADED case, not the happy one.** The enhancement itself is
#: a 600-token greedy decode, ~1-2 s on the GPU. What this number is really
#: covering is the first enhanced call in a container whose startup prefetch did
#: not finish or did not succeed: that call pays the measured **88 s** Hub fetch
#: inside its own reservation. ``enhance.prefetch`` is explicitly best-effort --
#: it must never block readiness or fail startup -- so "the prefetch worked" is
#: not something the estimator is allowed to assume.
#:
#: ⚠️ **Residency does NOT let this shrink, and the reason is structural rather
#: than cautious.** ``enhance.py`` now keeps the model loaded for the life of the
#: worker, so a *warm* enhanced call costs only the decode. The estimator cannot
#: spend that saving, because it does not know whether the call will be warm:
#: ``spaces`` computes the duration in the WEB process and hands it to
#: ``client.schedule()`` **before** a worker is selected -- ``schedule_response``
#: is what carries the ``idle`` flag that decides reuse, and it does not exist yet
#: when this function runs. So every reservation must still fit the cold path.
#:
#: ⛔ **Raised 120 -> 180, because the old arithmetic has been measured wrong.**
#: The 120 was justified as "88 s of download, a **sub-second load** and the
#: decode, with headroom". The load is not sub-second: it is the **~29 s** left
#: over from the ✅ measured +30.78 s per-call reload once the ~1-2 s decode is
#: taken out. The real degraded worst case was therefore 88 + 29 + 2 ~= 119 s
#: against a 120 s allowance -- about **one second** of margin, where the comment
#: claimed headroom. 120 was right by luck, not by construction, and the same
#: measurement that motivated residency is what exposed it.
#:
#: The asymmetry in the module docstring decides the rounding: reservation and
#: billing are different meters, so an over-generous request costs
#: schedulability and nothing at all in billing, while an under-generous one
#: kills the run at the boundary after the user has already waited through it.
#: 180 keeps ~50% headroom over the measured cold path. It is not free -- the
#: pre-check compares the *requested* duration against remaining quota, and on
#: xlarge that comparison is scaled by :func:`quota_multiplier` -- but a queued
#: run is recoverable and reports why, while a run killed at the boundary is not.
ENHANCE_DURATION_SECONDS = 180

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


def estimate_duration(**params) -> int:
    """Seconds to reserve for one generation.

    **Its signature must match the decorated function exactly** -- ZeroGPU calls
    the estimator with that function's own arguments, so a drifted signature
    raises at *schedule* time, once the user is already waiting. Both take
    ``**params``, and ``check_duration_signature`` asserts it at import.

    Reservation and billing are different meters: the pre-check compares the
    **requested** duration against remaining quota, while billing is on the
    **effective** duration times :func:`quota_multiplier`. So an over-generous
    request costs nothing in billing and everything in schedulability.
    """
    from .config import DEFAULT_GPU_DURATION_SECONDS, load_config

    seconds = params.get("gpu_duration_seconds")
    if seconds is None:
        # Read the CONFIG, not the constant. Reading the constant is what made
        # LTX25_GPU_DURATION_SECONDS an advertised lever attached to nothing --
        # it resolved fine and changed no behaviour.
        try:
            seconds = load_config().gpu_duration_seconds
        except Exception:
            seconds = DEFAULT_GPU_DURATION_SECONDS
    if _request_field(params, "enable_stage_2"):
        # Stage 2 adds an upsample pass plus three more denoise steps; a timeout
        # costs the whole reservation and proves nothing.
        seconds = int(seconds * 1.5)
    if _request_field(params, "loras"):
        # AFTER the stage-2 multiplier on purpose: the payload read and the
        # injection walk happen once, before any denoising, so their cost is
        # additive -- it does not scale with the refinement chain.
        seconds = int(seconds) + LORA_DURATION_SECONDS
    if _request_field(params, "enhance_prompt"):
        # Additive for the same reason, and it genuinely has to be here: prompt
        # enhancement used to run in the WEB process, outside the reservation
        # entirely, and this estimator had nothing to say about it. It now runs
        # INSIDE the worker (see `enhance.py`), so the time it takes is billed
        # time -- and a reservation that does not know that gets the run killed
        # at the boundary, which is a self-inflicted wound rather than a
        # platform limit.
        seconds = int(seconds) + ENHANCE_DURATION_SECONDS
    return int(seconds)


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

    base = runtime_config.gpu_duration_seconds
    duration = kwargs.get("duration")
    if callable(duration):
        seconds = (
            f"duration={getattr(duration, '__name__', 'callable')} (base {base} s; x1.5 with "
            f"stage 2, +{LORA_DURATION_SECONDS} s with LoRAs, +{ENHANCE_DURATION_SECONDS} s "
            f"with prompt enhancement)"
        )
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
