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
