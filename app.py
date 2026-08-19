"""Entrypoint for the Hugging Face Space and for local runs.

``import spaces`` MUST come before anything that initialises CUDA. ZeroGPU
raises *"CUDA has been initialized before importing the spaces package"*
otherwise, and ``ltx25_studio`` pulls in torch transitively -- so the rule covers
our own package too, not just the obvious third-party names. That ordering is
why the ``spaces`` import lives here rather than inside the library: a
deploy-environment package imported from library code makes the library
unimportable off-Space.

**This file is where the ZeroGPU design in PLAN §8 actually executes.** The
decorator below is a literal, unconditional ``@spaces.GPU(...)`` because that is
the form the platform's startup analyzer looks for, and because a *conditional*
decorator is how a documented rule quietly stops being applied -- the Space path
stops being the path anyone runs. Off-Space, ``spaces`` is a stub whose ``GPU``
returns the function unchanged, so the wiring is identical everywhere and only
its effect differs.
"""

import logging
import os

try:  # noqa: I001 - import order is load-bearing; see the module docstring
    import spaces
except ImportError:  # local development, where ZeroGPU does not exist
    from ltx25_studio import zerogpu as _zerogpu_bootstrap

    spaces = _zerogpu_bootstrap.local_spaces_stub()

from ltx25_studio import generate as generate_module
from ltx25_studio import outputs, pipelines, preflight, zerogpu
from ltx25_studio.config import load_config
from ltx25_studio.ui import build_ui

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("ltx25_studio")

# Unguarded on purpose. An unrecognised LTX25_* value must reach the operator as
# a startup failure naming the variable and the legal values -- swallowing it
# would leave the Space running in a configuration nobody chose (PLAN §5.1).
CONFIG = load_config()

ON_ZEROGPU = zerogpu.is_zerogpu(spaces)
_log.info("spaces=%s zerogpu=%s", zerogpu.spaces_version(spaces), ON_ZEROGPU)

# Weights are fetched HERE, at runtime, not by `preload_from_hub`. On a PRIVATE
# Space the preload init step runs unauthenticated and a gated repo answers 401,
# so it cannot work however valid the token is. This process has the secrets.
# It also removes the read-only-cache class (PLAN §6.2): the runtime user owns
# the cache it downloaded.
if ON_ZEROGPU:
    preflight.ensure_weights(token=os.environ.get("HF_TOKEN"))

# Version assertion, cache completeness and the resource report. Strict only
# on-Space — and now genuinely meaningful, because it verifies the download we
# just performed rather than asserting a preload we did not control.
preflight.run(CONFIG, strict=ON_ZEROGPU, spaces_module=spaces)

# PLAN §2.7: sweep on startup as well as after each run. Resolved from the same
# config the writer uses, so the two cannot drift onto different directories.
try:
    for _evicted in outputs.enforce_retention(
        generate_module.configured_output_root(),
        max_bytes=int(CONFIG.output_retention_gb * 1e9),
    ):
        _log.info("startup eviction: removed old run %s", _evicted)
except Exception as _exc:  # noqa: BLE001 - a bad sweep must not stop startup
    _log.warning("startup output sweep skipped: %s", _exc)

# PLAN §8.3: build on CPU, then move at MODULE SCOPE. ZeroGPU intercepts this
# call, rewrites it to .to("cpu") and packs every tensor to disk. Doing it lazily
# inside the decorated function skips packing entirely and re-transfers from RAM
# on every cold worker.
if ON_ZEROGPU:
    pipelines.get_pipeline(pipeline="condition", device="cuda")
    pipelines.get_pipeline(pipeline="text2video", device="cuda")

# `size` is filtered against the installed spaces signature: the platform pins
# its own version and lags PyPI, and passing an unknown kwarg raises TypeError
# at import, which on a Space looks like a mysterious startup failure.
_GPU_KWARGS = zerogpu.gpu_decorator_kwargs(
    spaces.GPU,
    duration=zerogpu.estimate_duration,
    size=None if CONFIG.zerogpu_size == "auto" else CONFIG.zerogpu_size,
)
_log.info("spaces.GPU kwargs: %s", sorted(_GPU_KWARGS))


# Bound at import, deliberately: the entrypoint must call the real generate, not
# whatever the module attribute happens to point at later. Resolving it by
# attribute lets any wrapper installed after startup send the worker straight
# back through itself.
_GENERATE = generate_module.generate


@spaces.GPU(**_GPU_KWARGS)
def run_generation(**params):
    """The only function that runs on the GPU. Decorated statically at import.

    Never returns a CUDA tensor: unpickling one in the main process triggers a
    blocked lazy CUDA init and the call **hangs** with no error at all, which is
    the worst failure shape available to us.
    """
    token = generate_module._IN_GPU_WORKER.set(True)
    try:
        return zerogpu.assert_returnable(_GENERATE(**params))
    finally:
        generate_module._IN_GPU_WORKER.reset(token)


# Raises at import if the estimator's signature ever drifts from the function's
# -- otherwise it raises at schedule time, once the user is already waiting.
zerogpu.check_duration_signature(run_generation, zerogpu.estimate_duration)

# Route every generation through the decorated entrypoint, so a run started from
# the UI actually reserves a GPU slot instead of executing in the web worker.
generate_module.set_gpu_dispatch(run_generation)

demo = build_ui()


def main() -> None:
    demo.launch()


if __name__ == "__main__":
    main()
