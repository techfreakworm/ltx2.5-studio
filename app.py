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

from ltx25_studio import device, log_filter, outputs, pipelines, preflight, zerogpu
from ltx25_studio import enhance as enhance_module
from ltx25_studio import generate as generate_module
from ltx25_studio.config import load_config
from ltx25_studio.ui import build_ui
from ltx25_studio.ui.blocks import served_outputs

logging.basicConfig(level=logging.INFO)
# bitsandbytes' int8 cast warning once per process, not once per matmul; the
# forked GPU worker inherits the filter with the logger.
log_filter.install()
_log = logging.getLogger("ltx25_studio")

# Unguarded on purpose. An unrecognised LTX25_* value must reach the operator as
# a startup failure naming the variable and the legal values -- swallowing it
# would leave the Space running in a configuration nobody chose (PLAN §5.1).
CONFIG = load_config()

ON_ZEROGPU = zerogpu.is_zerogpu(spaces)

# The generation device: CUDA on ZeroGPU or a CUDA box, an explicit LTX25_DEVICE
# opt-in for MPS/CPU, otherwise none (UI only). Unguarded like CONFIG: an
# impossible setting (int8 off CUDA, mps without MPS) stops startup by name.
#
# CAPABILITY, not identity, for the CUDA path: a real GPU box that is not a
# Space still downloads, preflights strictly and moves to CUDA. On a Space the
# resolver answers from `spaces` alone and never touches torch.cuda at module
# scope — the rule the whole import-order dance exists to protect.
DEVICE = device.resolve(spaces, CONFIG)
WILL_USE_GPU = DEVICE.device == "cuda"
_log.info(
    "spaces=%s zerogpu=%s device=%s",
    zerogpu.spaces_version(spaces),
    ON_ZEROGPU,
    DEVICE.describe(),
)

# The weights come from the image: README's `preload_from_hub` fetches the same
# set (`model_files`) before this runs. `ensure_weights` checks every file is in
# the cache, with no network call when it is, and downloads only what the image
# lacks, saying how much: a broken preload makes the start slower, not broken.
# A download the disk (on ZeroGPU, its cap) cannot hold is refused instead.
if WILL_USE_GPU:
    preflight.ensure_weights(token=os.environ.get("HF_TOKEN"), on_zerogpu=ON_ZEROGPU)

# Version assertion, cache completeness and the resource report.
#
# Strict wherever this process will generate: the pinned diffusers revision and
# cache completeness are enforced rather than reported, because a host about to
# generate must fail loudly on a missing file, not quietly render something
# wrong. A UI-only machine gets the report only. On CUDA this verifies the
# cache ensure_weights just checked or filled; on an MPS/CPU opt-in it verifies the
# snapshot (or LTX25_MODEL_SNAPSHOT_DIR) is complete. MPS also runs its
# numerics self-test here, once, and the report carries the result.
preflight.run(CONFIG, strict=DEVICE.generation_enabled, spaces_module=spaces, resolution=DEVICE)

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
# call and registers every tensor; it packs them to disk at demo.launch(). Doing
# it lazily inside the decorated function skips packing entirely and
# re-transfers from RAM on every cold worker. MPS/CPU build lazily, on the
# first Generate, through the same resolved device.
if DEVICE.device == "cuda":
    pipelines.get_pipeline(pipeline="condition", device="cuda")
    pipelines.get_pipeline(pipeline="text2video", device="cuda")
    # The diffusion decoder too, so it is packed with the rest and every run's
    # worker uses it instead of loading it again. Never raises: on a failure it
    # logs one line and each run builds its own, as before.
    pipelines.build_decoder_at_startup(device="cuda")

# The prompt enhancer's WEIGHTS, into the HF cache, on a daemon thread.
#
# Enhancement runs inside the `@spaces.GPU` worker now (see enhance.py for the
# measurement that moved it there), which means the 88 s Hub fetch would be
# BILLED on the first enhanced call in every container -- ~176 GPU-seconds spent
# downloading a file. Pulling it here spends that on wall-clock nobody is
# waiting on instead.
#
# ⚠️ It must not be in the packing window, and it is not: `snapshot_download`
# writes blobs and constructs no torch tensors, so there is nothing registered
# in `cuda_aliases` for the `torch.pack()` at `demo.launch()` to capture. Files
# on disk are not packable state. This is what lets the weights be resident
# before the first request without the enhancer re-entering the disk budget the
# original web-process placement was chosen to protect.
#
# ⚠️ It must not delay readiness, and it does not: the thread is a daemon, so a
# fetch still in flight blocks neither `demo.launch()` below nor process exit. A
# failure is logged and swallowed -- the Space still starts, un-enhanced
# generations are untouched, and the first enhanced run merely pays the download
# itself. Degraded, not broken.
#
# CONFIG.prefetch_enhancer exists for the deploy that will never turn
# enhancement on and would rather keep the ~2.5 GB of disk; see config.py.
if WILL_USE_GPU and CONFIG.prefetch_enhancer:
    enhance_module.start_prefetch(token=os.environ.get("HF_TOKEN"))

# `size` is filtered against the installed spaces signature: the platform pins
# its own version and lags PyPI, and passing an unknown kwarg raises TypeError
# at import, which on a Space looks like a mysterious startup failure.
_GPU_KWARGS = zerogpu.gpu_decorator_kwargs(
    spaces.GPU,
    duration=zerogpu.estimate_duration,
    size=None if CONFIG.zerogpu_size == "auto" else CONFIG.zerogpu_size,
)
_log.info("spaces.GPU: %s", zerogpu.describe_gpu_kwargs(_GPU_KWARGS, CONFIG))


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

    Refuses first, typed and before any load, on a machine with no generation
    device or an MPS torch that failed its self-test. Free on CUDA.
    """
    DEVICE.require_generation(injected=params.get("pipeline_factory") is not None)
    token = generate_module._IN_GPU_WORKER.set(True)
    try:
        return zerogpu.assert_returnable(DEVICE.annotate(_GENERATE(**params)))
    finally:
        generate_module._IN_GPU_WORKER.reset(token)


# Raises at import if the estimator's signature ever drifts from the function's
# -- otherwise it raises at schedule time, once the user is already waiting.
zerogpu.check_duration_signature(run_generation, zerogpu.estimate_duration)

# Route every generation through the decorated entrypoint, so a run started from
# the UI actually reserves a GPU slot instead of executing in the web worker.
generate_module.set_gpu_dispatch(run_generation)

demo = build_ui()

# The largest upload the app has any use for is a LoRA at the validator's own
# ceiling (config.lora_max_bytes; stage 1 refuses anything bigger BEFORE
# reading it) -- conditioning media is orders of magnitude smaller. Without a
# launch bound, any-size uploads land on gradio's tmp disk before any check of
# ours can run, and the retention sweep covers outputs/ only, so nothing ever
# reclaims that space. Reads the CONFIG, not the constant, so the
# LTX25_LORA_MAX_GB lever bounds both the validator and the transport. The
# 5% margin covers multipart framing so a ceiling-sized adapter is refused by
# the validator's stage 1 -- which names the limit and the actual size --
# rather than by gradio's blunter transport error.
MAX_UPLOAD_BYTES = int(CONFIG.lora_max_bytes * 1.05)


def main() -> None:
    # Gradio serves a returned file only from the working directory, the temp dir or
    # `allowed_paths`: an output folder outside both is passed (served_outputs).
    output_root = generate_module.configured_output_root()
    demo.launch(max_file_size=MAX_UPLOAD_BYTES, **served_outputs(output_root))


if __name__ == "__main__":
    main()
