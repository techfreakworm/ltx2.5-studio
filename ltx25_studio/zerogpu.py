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
    0.51.1, but the platform pins its own version and lags -- live Spaces report
    0.49.3 / 0.50.0 / 0.50.4, and we deliberately do not pin it. Against a
    platform build that predates the parameter,
    ``@spaces.GPU(duration=..., size="xlarge")`` raises a **TypeError at module
    import**, which on a Space means the app simply never starts and looks like
    a mysterious failure rather than a version problem.

    We do not need it anyway: ZeroGPU's ``startup()`` packs the weights and, if
    the total exceeds its 42.00 GiB threshold, sets ``_default_gpu_size`` to
    ``xlarge`` itself. Ours is 67.23 GiB, so we are auto-promoted regardless --
    passing ``size`` neither avoids nor incurs the 2x quota cost.

    So: build the decorator kwargs defensively by checking
    ``inspect.signature(spaces.GPU).parameters``, include ``size`` only if the
    running ``spaces`` accepts it, log the detected version and whether it was
    passed, and log the size we actually got after packing so the quota cost is
    visible rather than inferred. Same treatment for any other newer kwarg.

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
    "unbacked_parameters",
    "read_process_maps",
    "parse_maps",
    "check_mmap_backed",
    "backing_file",
    "estimate_duration",
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
# PARKED — the mmap-backing guard. Not wired into any live path.
#
# ZeroGPU-only: it cannot fire off-Space, and ZeroGPU is parked until the app is
# proven on a dedicated GPU host. These helpers are kept rather than deleted
# because they are tested and mutation-checked, and re-enabling is one call in
# `pipelines._base_pipeline`. See docs/ZEROGPU_DISK.md for what they are for.
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

    Each one is 72 GB of disk that ZeroGPU will never reclaim.
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
            f"the {len(offenders)}-parameter shortfall means the HF snapshot is never freed -- "
            f"72 GB stays on disk with nothing to attribute it to. Something cast, cloned or "
            f"eagerly loaded these; the dtype must match the checkpoint's on-disk dtype."
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
    **effective** duration times the size multiplier. So an over-generous
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
    return int(seconds)


def spaces_version(spaces_module) -> str | None:
    """The platform's ``spaces`` version, for the startup log.

    Takes the module rather than importing it: ``spaces`` is a deploy-environment
    package, and importing it from library code makes the library unimportable
    off-Space. ``app.py`` owns that import -- it is also the only place where the
    before-torch ordering can actually be guaranteed.
    """
    return getattr(spaces_module, "__version__", None) if spaces_module is not None else None


def _looks_like_a_cuda_tensor(value) -> bool:
    """True for anything a real ``torch.Tensor`` on CUDA would answer True to."""
    if getattr(value, "is_cuda", False):
        return True
    device = getattr(value, "device", None)
    return device is not None and str(device).startswith("cuda")


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

    The platform pins its own ``spaces`` and lags PyPI -- live Spaces report
    0.49.3 / 0.50.0 / 0.50.4, and we deliberately do not pin it. Passing ``size``
    to a build that predates the parameter raises ``TypeError`` **at module
    import**, which on a Space looks like a mysterious startup failure rather
    than a version problem.

    Dropping ``size`` costs nothing: ZeroGPU's own ``startup()`` promotes to
    xlarge past its 42.00 GiB threshold and our packed weights are 67.23 GiB, so
    the kwarg neither avoids nor incurs the 2x quota cost.
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
