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

try:  # pragma: no cover - present only on a Space
    import spaces
except ImportError:  # the dev box, and the test suite
    spaces = None

__all__ = [
    "SPACE_HARDWARE",
    "assert_returnable",
    "check_duration_signature",
    "gpu_decorator_kwargs",
    "spaces_version",
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


def spaces_version() -> str | None:
    """The platform's ``spaces`` version, or None when it is not installed."""
    return getattr(spaces, "__version__", None) if spaces is not None else None


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
