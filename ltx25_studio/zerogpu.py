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
"""
