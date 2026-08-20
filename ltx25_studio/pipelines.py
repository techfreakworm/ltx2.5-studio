"""Pipeline construction, the stage graph, and the ZeroGPU placement rules.

PLAN §7.4, §8, §5.5.

**Placement (§8).** ZeroGPU replaces CPU offload with its own O_DIRECT
disk-packing engine. The correct pattern inverts normal GPU-box practice: build
on CPU, call ``.to("cuda")`` at module scope -- where it is intercepted,
rewritten to ``.to("cpu")`` and every tensor is packed to disk -- and apply
``@spaces.GPU`` statically at import. ``enable_model_cpu_offload()`` is
**counterproductive** here and must not be used: anything accelerate parks on
CPU is never registered for packing, so it stays resident in main-process RAM
(our tightest budget), is fork-inherited by every worker, and is re-copied
host-to-device on every hook fire.

**The stage graph (§7.4).** The classic pipelines decode with the convolutional
VAE, but the diffusion decoder is the native default from LTX-2.5 on. A naive
call therefore silently discards the model's headline quality feature -- it does
not error, it is just permanently worse. So the decode is an explicit stage::

    stage 1   LTX2ConditionPipeline(sigmas=DISTILLED_SIGMA_VALUES,
                                    output_type="latent")      -> latents, audio
    optional  LTX2LatentUpsamplePipeline                        -> upsampled
    stage 2   same pipeline, latents=upsampled,
                             sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                             noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                             output_type="latent"
    decode V  LTX2VideoDiffusionDecodePipeline(denormalize=False)
    decode A  audio_vae.decode -> mel -> vocoder -> waveform

``denormalize=False`` is **mandatory**: ``output_type="latent"`` already returns
denormalized latents, so the default double-denormalizes and silently corrupts
the output.

Everything heavy is imported inside functions, so this module -- and therefore
the whole package -- stays importable with no GPU, no weights and no diffusers.
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from types import MappingProxyType

from . import model_files
from .constraints import (
    DEFAULT_FRAME_RATE,
    DEFAULT_SPATIAL_COMPRESSION_RATIO,
    DEFAULT_TEMPORAL_COMPRESSION_RATIO,
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    image_crf_for,
)
from .errors import ModelLoadError

__all__ = [
    "DIFFUSION_DECODE_DENORMALIZE",
    "PipelineBundle",
    "StageRunner",
    "WEIGHT_DTYPE",
    "WEIGHT_DTYPE_LABEL",
    "BASE_KIND",
    "assert_compression_ratios",
    "assert_frame_rate_default",
    "clear_cache",
    "derive_pipeline",
    "get_pipeline",
    "stage_1_kwargs",
    "stage_2_size_note",
    "STAGE_2_SPATIAL_SCALE",
    "distilled_kwargs",
    "DISTILLED_GUIDANCE",
    "stage_2_kwargs",
    "upsample_kwargs",
    "STAGE_2_STRIPPED_KEYS",
    "startup_report",
    "disk_footprint",
    "ZEROGPU_DISK_CAP_GB",
    "QUANTIZED_COMPONENTS",
    "peak_rss_gib",
]

#: Mandatory. See the module docstring -- the default silently corrupts output.
DIFFUSION_DECODE_DENORMALIZE = False

#: The dtype every component loads at, and its short label for the log.
#: **Not configurable.** A precision variable existed, was wired to nothing, and
#: reported a footprint nobody was loading (PLAN §2.6). Fixed here so the value
#: the loader uses and the value the startup report prints cannot disagree --
#: which is the whole reason the report states it at all.
WEIGHT_DTYPE = "bfloat16"
WEIGHT_DTYPE_LABEL = "bf16"

_CACHE: dict[str, object] = {}


def _log():
    import logging

    return logging.getLogger("ltx25_studio")


@dataclass(frozen=True)
class PipelineBundle:
    """A constructed pipeline plus the values we must read off it, not assume."""

    pipeline: object
    kind: str
    temporal_compression_ratio: int
    spatial_compression_ratio: int
    image_crf: int


def assert_compression_ratios(vae) -> tuple[int, int]:
    """Tripwire for the fallbacks ``constraints.py`` uses before a VAE exists.

    The UI must validate ``num_frames`` and the frame size before anything is
    loaded, so ``constraints.py`` carries 8 and 32 as documented pre-load
    defaults. If upstream ever changes them, silently generating at the wrong
    geometry would be far worse than failing here.
    """
    temporal = int(getattr(vae, "temporal_compression_ratio", DEFAULT_TEMPORAL_COMPRESSION_RATIO))
    spatial = int(getattr(vae, "spatial_compression_ratio", DEFAULT_SPATIAL_COMPRESSION_RATIO))
    if (
        temporal != DEFAULT_TEMPORAL_COMPRESSION_RATIO
        or spatial != DEFAULT_SPATIAL_COMPRESSION_RATIO
    ):
        raise ModelLoadError(
            component="vae",
            reason=(
                f"compression ratios changed upstream: temporal={temporal} "
                f"(expected {DEFAULT_TEMPORAL_COMPRESSION_RATIO}), spatial={spatial} "
                f"(expected {DEFAULT_SPATIAL_COMPRESSION_RATIO}). The pre-load validators in "
                f"constraints.py use the expected values, so generation would silently run at "
                f"the wrong geometry. Update constraints.py before continuing."
            ),
        )
    return temporal, spatial


def assert_frame_rate_default(pipeline_object) -> None:
    """Tripwire on upstream's own ``frame_rate`` default.

    Same shape as the compression-ratio tripwire: we thread one rate to both the
    generation and the container, and ``DEFAULT_FRAME_RATE`` is our copy of
    upstream's. If upstream changes its default, a caller that omits the kwarg
    would generate at one rate while the container declares another -- and a
    container written at the wrong rate plays at the wrong speed and reports no
    error.
    """
    import inspect

    try:
        default = inspect.signature(pipeline_object).parameters["frame_rate"].default
    except (TypeError, ValueError, KeyError):
        return  # nothing to compare against; not worth failing a load over
    if default is not inspect.Parameter.empty and float(default) != float(DEFAULT_FRAME_RATE):
        raise ModelLoadError(
            component="transformer",
            reason=(
                f"upstream's frame_rate default changed to {default} (we carry "
                f"{DEFAULT_FRAME_RATE}). Generation and the muxed container would be written at "
                f"different rates, which plays at the wrong speed without any error. Update "
                f"DEFAULT_FRAME_RATE in constraints.py before continuing."
            ),
        )


def _text_encoder_model_type(pipeline) -> str:
    encoder = getattr(pipeline, "text_encoder", None)
    config = getattr(encoder, "config", None)
    return str(getattr(config, "model_type", "")) if config is not None else ""


#: The workflow whose pipeline class accepts the SUPERSET of components, and so
#: the one we always load. ``LTX2ConditionPipeline`` takes everything
#: ``LTX2Pipeline`` takes plus ``audio_scheduler`` -- zero weights, ``None``
#: default -- so deriving in this direction can never drop a component, while
#: the reverse would silently lose the one the condition path actually uses.
BASE_KIND = "condition"

#: Cache key for the raw base pipeline, distinct from the StageRunner entries.
_BASE_CACHE_KEY = "__base__"

#: The stage-2 parts are built once and shared by every workflow, exactly as the
#: base is. They are ~2.1 GB of weights that no other component shares, so
#: building them per workflow would pay for them twice.
_STAGE_2_CACHE_KEY = "__stage_2__"


def derive_pipeline(base, target_cls):
    """Build ``target_cls`` from ``base``'s ALREADY-LOADED modules.

    ``from_pretrained`` twice materialises two complete, independent copies of
    the model -- ~67 GiB each. On ZeroGPU that is ~134 GiB packed to disk
    against :data:`ZEROGPU_DISK_CAP_GB`, and packing runs after both builds; on a 96 GiB
    card it is an outright OOM. The two pipeline classes take **identical**
    weight-bearing components at our pinned revision, so there is no version of
    this where two copies are wanted.

    Prefers ``from_pipe``, which exists at the pinned SHA and handles exactly
    this asymmetry -- it treats optional components carrying a default in the
    signature as safe to omit, which is precisely ``audio_scheduler``. Falls back
    to filtering ``base.components`` against the target's own signature, which is
    the same rule written out.
    """
    from_pipe = getattr(target_cls, "from_pipe", None)
    if callable(from_pipe):
        # ⚠️ `dtype=` IS REQUIRED. Without it diffusers resolves
        # `dtype ... or torch.float32` (pipeline_utils.py:2155) and then calls
        # `new_pipeline.to(dtype=dtype)` (:2253) — casting every module to fp32.
        #
        # The crash is the lesser half: an fp32 cast of a bf16 tensor allocates
        # twice the bytes, so the "derived" pipeline is a second ~130 GiB copy and
        # OOMs a 96 GiB card. The half that matters is that `nn.Module.to()`
        # mutates IN PLACE and the modules are genuinely shared — so had it fit,
        # **the base pipeline would silently have become fp32**, and every
        # generation after it would have run at the wrong precision with no error
        # and plausible-looking output.
        #
        # It would also have broken the ZeroGPU mmap precondition, since a cast
        # materialises tensors into anonymous memory and the post-pack prune then
        # frees nothing (PLAN §8.3). Same root cause, a second consequence.
        dtype = getattr(base, "dtype", None)
        return from_pipe(base) if dtype is None else from_pipe(base, dtype=dtype)
    accepted = inspect.signature(target_cls.__init__).parameters
    shared = {name: module for name, module in base.components.items() if name in accepted}
    return target_cls(**shared)


def peak_rss_gib() -> float | None:
    """Peak resident set size so far, in GiB, or ``None`` where unavailable.

    Logged around the model load because **a bf16 load is not a floor for a
    quantised one.** ✅ Measured: a full bf16 load peaked at 20.05 GiB against
    65.4 GiB of weights, because safetensors **mmaps** and the pages stay
    file-backed. A quantising load materialises new int8 tensors -- anonymous
    memory by construction, the same mechanism that defeats ZeroGPU's post-pack
    prune -- so that 20 GiB says nothing about this path.
    """
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux and bytes on macOS. The Space and the host are
    # Linux; the branch keeps a local run from reporting a nonsense number.
    if sys.platform == "darwin":
        return peak / 1024**3
    return peak / 1024**2


#: Which pipeline components load in int8. **Names must match
#: ``model_index.json``'s own keys** -- verified at revision ``871165de...``,
#: where ``text_encoder`` maps to ``Gemma4UnifiedForConditionalGeneration``.
#:
#: WARNING: a typo here fails SILENTLY. ``PipelineQuantizationConfig`` does not
#: validate these names -- ``pipe_quant_config.py:171`` is a bare
#: ``module_name in components_to_quantize`` membership test, so an unrecognised
#: entry never matches and that component quietly loads at full precision. That
#: is the advertised-lever-attached-to-nothing shape again, which is why
#: :func:`_quantization_config` checks them against ``model_files.COMPONENTS``.
QUANTIZED_COMPONENTS = ("transformer", "text_encoder")


def _quantization_config(quantization: str):
    """The quantisation config, or ``None`` for the bf16 default.

    **Transformer only, originally — and that reasoning was right at the time.**
    The transformer is the largest single component (37.98 GB of 72.19), and
    quantising it alone took the packing peak from 152.4-156.4 GB to
    133.4-137.4 against :data:`ZEROGPU_DISK_CAP_GB`. Widening to the text
    encoder bought headroom we did not need and cost something we did: **it
    would have made an int8-vs-bf16 comparison unable to say whether a quality
    change came from the denoiser or from prompt encoding.**

    ✅ **The ground moved and that argument no longer binds.** The live Space at
    sha ``a1791c9e`` gave a *measured* transformer-only baseline — **55.89 GiB
    peak RSS, 60.0 GB steady state.** Adding the text encoder is now
    attributable **against that baseline** rather than confounded with it:
    the very thing that made widening a confound is what a measured baseline
    turns into a control.

    **And it is load-bearing now rather than optional.** Disk was never the
    binding constraint — the run died with the disk line reporting **[OK]** at
    82.2 GB against :data:`ZEROGPU_DISK_CAP_GB`, killed at ~60.0 GB of RAM
    against a ceiling measurably above that (ours)
    and ~68 GB (⚠️ inherited). See ``docs/SPACE_GENERATION_FINDING.md``.

    ✅ **Revised arithmetic**, derived from ``model_files`` component sizes:

    * transformer only ....... packed 53.197 GB, peak 133.4 - 137.4
    * **+ text encoder** ..... packed **41.237 GB**, peak **121.4 - 125.4**

    ⚠️ ~11.96 GB less packed, and ~46.6 GB expected RSS **if** the 1.13x
    transient factor the measured run implies holds — **one data point, so an
    extrapolation rather than a rate.**
    """
    if quantization != "int8":
        return None

    # Checked BEFORE reaching for diffusers: validating our own constant does not
    # need a heavy optional dependency, and a config error should surface as a
    # config error rather than behind an ImportError on a machine without it.
    #
    # Guarded because upstream will not guard it: an unknown component name is
    # silently ignored, so a typo would deploy that component at full precision
    # and raise nothing. `model_files.COMPONENTS` is the registry the download
    # allow-list is built from, so a name that survives this check is one the
    # checkpoint really has.
    unknown = [c for c in QUANTIZED_COMPONENTS if c not in model_files.COMPONENTS]
    if unknown:
        raise ModelLoadError(
            component="transformer",
            reason=(
                f"QUANTIZED_COMPONENTS names {unknown}, which the checkpoint does not "
                f"have. PipelineQuantizationConfig ignores unknown names silently and "
                f"would load those components at full precision -- a lever attached to "
                f"nothing. Known: {sorted(model_files.COMPONENTS)}."
            ),
        )

    try:
        from diffusers import PipelineQuantizationConfig
    except ImportError as exc:
        raise ModelLoadError(
            component="transformer",
            reason=(
                f"int8 was requested and the quantisation stack is not importable ({exc}). "
                f"`bitsandbytes` is a pinned requirement; see requirements.txt for why the "
                f"floor is 0.46.0 rather than the 0.43.3 diffusers itself mandates."
            ),
        ) from exc

    # ⚠️ `PipelineQuantizationConfig`, NOT `BitsAndBytesConfig`. A bare
    # `BitsAndBytesConfig` is what a COMPONENT's `from_pretrained` takes;
    # `DiffusionPipeline.from_pretrained` rejects it outright:
    #
    #   ValueError: `quantization_config` must be an instance of
    #   `PipelineQuantizationConfig`.
    #
    # ✅ `quant_backend="bitsandbytes_8bit"` is a key in both backend mappings
    # at the pinned SHA (`quantizers/auto.py:45,57`), not a guessed string.
    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
        # Bound B, now expressed as an ARGUMENT rather than as an intent a
        # source-inspection test had to assert. Widening is one list entry and
        # the kwargs test sees it.
        components_to_quantize=list(QUANTIZED_COMPONENTS),
    )


def _snapshot_path():
    """The local directory every component loads from.

    Loading by repo id makes ``from_pretrained`` resolve against the Hub and
    pull the **whole** repo -- 44.33 GB of shards the allow-list exists to
    exclude -- even with a verified-complete 37/37 cache. See
    ``preflight.snapshot_dir`` for why ``local_files_only=True`` is not the fix.
    """
    from . import preflight

    path = preflight.snapshot_dir()
    if not path.is_dir():
        raise ModelLoadError(
            component=BASE_KIND,
            reason=(
                f"no local snapshot at {path}. Weights are fetched by "
                f"`preflight.ensure_weights()`, which must run before any pipeline is "
                f"constructed; loading by repo id instead would refetch the whole repo."
            ),
        )
    return path


def _base_pipeline(*, dtype=None, device: str | None = None, quantization: str | None = None):
    """Load the model ONCE. Every workflow is derived from this."""
    if _BASE_CACHE_KEY in _CACHE:
        return _CACHE[_BASE_CACHE_KEY]

    if quantization is None:
        from .config import load_config

        quantization = load_config().quantization

    try:
        import torch
        from diffusers import LTX2ConditionPipeline
    except ImportError as exc:
        raise ModelLoadError(
            component=BASE_KIND,
            reason=(
                f"the model stack is not installed ({exc}). Install the pinned requirements; "
                f"see docs/PLAN.md §1.3 for why the diffusers revision is exact."
            ),
        ) from exc

    try:
        # ⚠️ `torch_dtype` MUST match the checkpoint's on-disk dtype. It is not a
        # free parameter, and the reason is invisible from here.
        #
        # ZeroGPU frees the 72 GB HF snapshot after packing, but only by reading
        # /proc/self/maps, mapping each tensor's data_ptr() back to the blob it is
        # mmap'd from, and unlinking those. Anything that materialises tensors
        # into anonymous memory -- a dtype CAST, a .clone(), an eager
        # low_cpu_mem_usage=False, a non-safetensors path -- leaves nothing in the
        # map to capture. Then nothing is freed, 72 GB stays on disk forever, and
        # there is no error and no log line saying so.
        #
        # The checkpoint is bf16 and WEIGHT_DTYPE is bfloat16, so this is a
        # no-op load rather than a conversion. Do not "helpfully" add a cast.
        before = peak_rss_gib()
        quantization_config = _quantization_config(quantization)
        load_kwargs = {"torch_dtype": dtype or getattr(torch, WEIGHT_DTYPE)}
        if quantization_config is not None:
            # Applied to the transformer alone -- see `_quantization_config`.
            load_kwargs["quantization_config"] = quantization_config
        built = LTX2ConditionPipeline.from_pretrained(_snapshot_path(), **load_kwargs)
        after = peak_rss_gib()
        # Reported whatever happens. A bf16 load peaks far below the weight
        # footprint because safetensors mmaps; a quantising load materialises
        # tensors and does not, so this number is the one that decides whether
        # int8 is viable on a memory-capped platform -- not the disk arithmetic.
        _log().info(
            "loaded transformer as %s | peak RSS %.2f -> %.2f GiB",
            quantization,
            before if before is not None else float("nan"),
            after if after is not None else float("nan"),
        )
    except Exception as exc:
        raise ModelLoadError(component=BASE_KIND, reason=str(exc)) from exc

    # NOTE: the mmap-backing guard (`zerogpu.check_mmap_backed`) is deliberately
    # NOT wired in here. It is ZeroGPU-only -- it cannot fire anywhere else --
    # and ZeroGPU is parked pending the app being proven on a dedicated host.
    # The helpers stay tested and ready; re-enabling is this call plus a raise.
    # See docs/ZEROGPU_DISK.md.

    if device:
        # ZeroGPU intercepts this, rewrites it to .to("cpu") and packs every
        # tensor to disk -- which is why it must happen at module scope, not
        # lazily inside a @spaces.GPU function. Doing it there skips packing
        # entirely and re-transfers from RAM on every cold worker. Only the base
        # is moved: everything else shares these very modules.
        built.to(device)
        # The second reporting point, and it must be HERE. ZeroGPU intercepts
        # `.to(device)` and packs every tensor to disk inside that call, so this
        # is the first moment the packed copy exists. Reporting it before the
        # move would print "after packing" over a measurement taken before any
        # packing happened -- a mislabelled report, which is the defect this
        # whole check exists to remove.
        _log().info("disk %s", disk_footprint(phase="after packing"))

    vae = getattr(built, "vae", None)
    if vae is not None:
        assert_compression_ratios(vae)
    assert_frame_rate_default(built)

    _CACHE[_BASE_CACHE_KEY] = built
    return built


def _stage_2_parts(base, *, dtype=None, device: str | None = None):
    """Build the latent upsampler and the diffusion decoder, once, sharing the base.

    **The diffusion decoder is never a pipeline component.** ``model_index.json``
    declares it, but every ``LTX2*Pipeline`` rejects it -- upstream is explicit
    that it "is a diffusion model in its own right rather than a pipeline
    component, so it is driven by ``LTX2VideoDiffusionDecodePipeline``". Loading
    the base and expecting the decoder to arrive with it silently gets you a
    pipeline with no decoder and a warning nobody reads.

    Both stage-2 pipelines are thin wrappers: the upsampler shares the base's
    ``vae`` and the decoder shares its ``scheduler``, so the only new weights are
    the two model checkpoints themselves.
    """
    if _STAGE_2_CACHE_KEY in _CACHE:
        return _CACHE[_STAGE_2_CACHE_KEY]

    try:
        import torch
        from diffusers import LTX2LatentUpsamplePipeline, LTX2VideoDiffusionDecoderModel
        from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
            LTX2VideoVaeNeighborhoodNattenProcessor,
        )
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode import (
            LTX2VideoDiffusionDecodePipeline,
        )
    except ImportError as exc:
        raise ModelLoadError(
            component="diffusion_decoder",
            reason=(
                f"the stage-2 classes are not importable ({exc}). The diffusion decoder also "
                f"needs the `kernels` package, which fetches the NATTEN kernel from the Hub; "
                f"a local `natten` install is explicitly not used."
            ),
        ) from exc

    # Same rule as the base load, and the same reason: the checkpoints are bf16,
    # so this is a no-op rather than a conversion. See `_base_pipeline`.
    weight_dtype = dtype or getattr(torch, WEIGHT_DTYPE)

    def _load(cls, component):
        spec = model_files.get_component(component)
        try:
            return cls.from_pretrained(
                _snapshot_path(),
                subfolder=spec.subfolder,
                torch_dtype=weight_dtype,
            )
        except Exception as exc:
            raise ModelLoadError(component=component, reason=str(exc)) from exc

    upsampler_model = _load(LTX2LatentUpsamplerModel, "latent_upsampler")
    decoder_model = _load(LTX2VideoDiffusionDecoderModel, "diffusion_decoder")

    # ⚠️ Neither of these is an optimisation; both are required to run at all.
    #
    # The decoder's default FlexAttention processor is uncompiled, so it
    # materialises the full score matrix and needs **tens of GB** at video
    # resolutions. `set_attention_backend` is not an alternative -- no backend
    # other than `flex` accepts the BlockMask. Every attention module in the
    # decoder is the same neighborhood attention, so one call swaps them all.
    #
    # Two operational traps, both upstream's and neither settable from here:
    #   * it resolves `get_kernel("shi-labs/natten", version=1)` -- the version is
    #     HARDCODED in diffusers, so we cannot pin it by revision the way we pin
    #     everything else, and the version->ref lookup goes over the network even
    #     when the kernel files are already cached.
    #   * `DIFFUSERS_DISABLE_REMOTE_CODE` set in the environment makes this raise.
    # Constructing it lazily is what keeps both of those out of app startup.
    decoder_model.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
    # Tiling bounds peak memory by the tile size rather than by the video size.
    decoder_model.enable_tiling()

    if device:
        # These weights are NOT shared with the base, so `_base_pipeline`'s
        # `.to(device)` does not reach them. Without this the stage-2 path dies
        # on a CPU/CUDA mismatch the first time anyone ticks the box.
        upsampler_model.to(device)
        decoder_model.to(device)

    upsampler = LTX2LatentUpsamplePipeline(vae=base.vae, latent_upsampler=upsampler_model)
    # `vae` is deliberately omitted: it is optional here and consulted only to
    # denormalize, which `DIFFUSION_DECODE_DENORMALIZE` never asks for.
    decoder = LTX2VideoDiffusionDecodePipeline(
        diffusion_decoder=decoder_model, scheduler=base.scheduler
    )

    parts = (upsampler, decoder)
    _CACHE[_STAGE_2_CACHE_KEY] = parts
    return parts


def get_pipeline(*, pipeline: str = BASE_KIND, dtype=None, device: str | None = None):
    """The runner for one workflow, sharing one loaded copy of the model."""
    if pipeline in _CACHE:
        return _CACHE[pipeline]

    base = _base_pipeline(dtype=dtype, device=device)
    if pipeline == BASE_KIND:
        built = base
    else:
        from diffusers import LTX2Pipeline

        built = derive_pipeline(base, LTX2Pipeline)

    # NOT built here. The weights are already guaranteed by
    # `preflight.require_cache_complete()` -- all three components are in the
    # allow-list -- so the startup loudness is not lost. What cannot be checked
    # at startup is the NATTEN kernel, which resolves over the network, and
    # checking it would reimpose the very network dependency this defers.
    runner = StageRunner(
        built,
        kind=pipeline,
        stage_2_factory=lambda: _stage_2_parts(base, dtype=dtype, device=device),
    )
    _CACHE[pipeline] = runner
    return runner


def bundle_for(pipeline_object, kind: str) -> PipelineBundle:
    """Read the values we must never hardcode off a constructed pipeline."""
    vae = getattr(pipeline_object, "vae", None)
    temporal, spatial = (
        assert_compression_ratios(vae)
        if vae is not None
        else (
            DEFAULT_TEMPORAL_COMPRESSION_RATIO,
            DEFAULT_SPATIAL_COMPRESSION_RATIO,
        )
    )
    return PipelineBundle(
        pipeline=pipeline_object,
        kind=kind,
        temporal_compression_ratio=temporal,
        spatial_compression_ratio=spatial,
        image_crf=image_crf_for(_text_encoder_model_type(pipeline_object)),
    )


#: The distilled recipe's guidance settings, from the model card at ``871165de``.
#:
#: ⚠️ **The pipeline defaults are the SFT values** -- ``stg_scale=1.0``,
#: ``modality_scale=3.0``, ``audio_guidance_scale=7.0`` -- so omitting any of
#: these does not "leave it at the default", it selects a *different model's*
#: settings. Upstream is explicit: distilled inference "runs unguided --
#: guidance_scale=1.0 plus STG and modality guidance zeroed, since the pipeline
#: defaults are the SFT values."
DISTILLED_GUIDANCE = MappingProxyType(
    {
        "guidance_scale": 1.0,
        "audio_guidance_scale": 1.0,
        "stg_scale": 0.0,
        "audio_stg_scale": 0.0,
        "modality_scale": 1.0,
        "audio_modality_scale": 1.0,
    }
)


def distilled_kwargs() -> dict:
    """The distilled recipe. **Every** run gets this, single-stage included.

    This function exists because the previous arrangement put the sigmas on
    ``stage_1_kwargs()``, which is applied **only when stage 2 is enabled** -- so
    ``DISTILLED_SIGMA_VALUES`` was defined, documented as mandatory, quoted in
    this module's own docstring, and **had never been sent to a pipeline** until
    the first stage-2 run. Every other generation got upstream's fallback:

        sigmas is None -> np.linspace(1.0, 1/num_inference_steps, num_inference_steps)
        [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]

    against the distilled schedule's

        [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]

    -- five of eight steps above 0.97 and a floor of 0.42, versus one and 0.125.
    A different denoising trajectory, not a mistuning.

    ``num_inference_steps`` is deliberately NOT sent: upstream builds the linear
    schedule from it, and the model card warns that passing it "would hand the
    model a generic linear schedule and quietly cost quality".

    ⏸️ **OPEN, and deliberately not acted on.** For a run given ``noise_scale``
    -- only the audio-input modes -- ``sigmas[0]`` stays 1.0 while the latents
    are noised to ``noise_scale``. 📄 That contradicts upstream's own inference
    rule, ``noise_scale = sigmas[0] if sigmas is not None else 1.0``, and our
    ``stage_2_kwargs`` honours the same rule.

    **It was diagnosed as the cause of a silent `audio_to_audio` and it was
    not.** ✅ The silence was a spectrally-sparse test fixture; with realistic
    audio the mode is healthy at −10.4 dBFS **carrying the full mismatch**, and
    H10b spans −9.9 to −19.6 dBFS in the right direction with it in place.

    ⚠️ **Truncating the schedule at ``noise_scale`` is not a safe no-op.** It
    would take an eight-step run to **two** steps at 0.5 and **one** at 0.1 --
    values that measured healthy on eight. So the fix needs evidence that it
    *improves* something, not merely that it agrees with the contract, and no
    instrument we have measures what it would change.
    """
    return {"sigmas": list(DISTILLED_SIGMA_VALUES), **DISTILLED_GUIDANCE}


def stage_1_kwargs() -> dict:
    """Stage-1 denoise: the distilled recipe, latents out."""
    return {**distilled_kwargs(), "output_type": "latent"}


#: How much stage 2 enlarges each axis. 📄 ``latent_upsampler/config.json`` at
#: revision 871165de… declares ``spatial_upsample: true`` and
#: ``rational_spatial_scale: 2.0``; ✅ confirmed on the card, where a 704×480
#: request with stage 2 produced 1408×960.
#:
#: Named rather than written as a bare 2 because it appears in a message the
#: user reads, and a message that quietly disagrees with the model is worse than
#: no message.
STAGE_2_SPATIAL_SCALE = 2


def stage_2_size_note(width: int, height: int) -> str:
    """What to tell the user about the size stage 2 will actually return.

    **Requesting 704×480 with stage 2 gives a 1408×960 file**, which is correct
    -- doubling both axes is what stage 2 *is* -- but nothing said so, and
    ``verify_streams`` does not assert resolution, so the size a user asked for
    was not the size they got and no part of the system mentioned it.
    """
    return (
        f"stage 2 upsamples both axes, so this output is "
        f"{int(width) * STAGE_2_SPATIAL_SCALE}×{int(height) * STAGE_2_SPATIAL_SCALE}, "
        f"not the {int(width)}×{int(height)} requested. That is what stage 2 does: "
        f"it renders at half size and enlarges, which is why it looks better."
    )


def upsample_kwargs(latents) -> dict:
    """Spatial x2 latent upsample.

    ``output_type="latent"`` is **mandatory and is not the default**. The default
    is ``"pil"``, which makes the upsampler run ``vae.decode`` and hand back
    images -- the convolutional decode that stage 2 exists to replace -- which
    would then be passed to stage 2 as its ``latents``. Nothing raises: the call
    succeeds and the render is wrong.
    """
    return {"latents": latents, "output_type": "latent"}


#: Stage 2 takes its size from the upsampled latents, so these must not be sent.
#: Upstream's recipe is explicit: *"Stage 2 takes its size from the upsampled
#: latents, so pass no height/width."* Forwarding the request's stage-1 values
#: would ask for a render half the size of the latents it was handed.
STAGE_2_STRIPPED_KEYS = ("height", "width")


def stage_2_kwargs(latents, *, audio_latents=None) -> dict:
    """Stage-2 refinement over upsampled latents.

    ``noise_scale`` is the first stage-2 sigma: the upsampled latents are already
    partly denoised, so re-noising to 1.0 would discard stage 1 entirely.

    ``audio_latents`` carries stage 1's audio forward. Without it stage 2 starts
    the audio from scratch, so the track that ships is not the one the video was
    denoised alongside.
    """
    kwargs = {
        **distilled_kwargs(),
        "latents": latents,
        "sigmas": list(STAGE_2_DISTILLED_SIGMA_VALUES),
        "noise_scale": STAGE_2_DISTILLED_SIGMA_VALUES[0],
        "output_type": "latent",
    }
    if audio_latents is not None:
        kwargs["audio_latents"] = audio_latents
    return kwargs


def decode_kwargs(latents, *, generator=None) -> dict:
    """Diffusion decode. ``denormalize=False`` is not optional.

    The diffusion decoder is a diffusion model in its own right and **draws its
    own noise**, so it takes a generator like any other sampler. Omitting it
    makes the decode unseeded, and two runs at the same seed then differ for a
    reason that has nothing to do with what they were comparing -- which is
    exactly what the same-seed matrix in ``docs/TEST_PLAN.md`` A.4.1 relies on.
    """
    kwargs = {"latents": latents, "denormalize": DIFFUSION_DECODE_DENORMALIZE}
    if generator is not None:
        kwargs["generator"] = generator
    return kwargs


class StageRunner:
    """Callable that owns the stage graph, so ``generate.py`` owns no branches.

    ``generate.py`` calls this exactly as it would call a bare pipeline. The
    ``enable_stage_2`` kwarg is consumed here and never reaches diffusers.

    **Single stage** (default): one denoise, decoded by the pipeline itself.

    **Two stages**: stage 1 renders at half resolution and emits latents, the
    spatial x2 latent upsampler enlarges them, stage 2 refines with the
    three-step schedule at full resolution, and the **diffusion decoder** turns
    the result into frames -- not ``vae.decode``. Upstream's own
    ``LTX25AutoBlocks`` says the diffusion decoder "is the native default from
    LTX-2.5 on", so decoding with the convolutional VAE silently forfeits the
    single biggest 2.5 quality feature. It would not error. It would just be
    permanently worse.

    Only ``latent_upsampler`` (spatial) takes part. ``temporal_latent_upsampler``
    is a second checkpoint of the same class on the other axis, but upstream's
    component table lists only the spatial one as belonging to this recipe and
    neither documented recipe uses the temporal one -- so it is deliberately not
    wired in rather than guessed into the graph.

    **Every stage call here is shaped, not merely made.** The upsample must ask
    for ``output_type="latent"``, stage 2 must NOT be given a size, stage 1's
    audio latents must be carried forward, the decode must be seeded, and the
    decoder's video-only output must be reunited with the audio. Get any one
    wrong and the run still succeeds and still returns a video.
    """

    def __init__(self, pipeline, *, kind, upsampler=None, decoder=None, stage_2_factory=None):
        self.pipeline = pipeline
        self.kind = kind
        self.upsampler = upsampler
        self.decoder = decoder
        #: Built on the first stage-2 request, never at construction. The parts
        #: are ~2.1 GB of weights AND a NATTEN kernel resolved over the network,
        #: and eight of the eleven modes never touch them. Building eagerly
        #: turned a dependency of one branch into a hard requirement for every
        #: mode -- when NATTEN could not resolve, nothing loaded at all.
        self._stage_2_factory = stage_2_factory

    def _stage_2_ready(self):
        """Obtain the stage-2 parts, building them if this is the first request."""
        if (self.upsampler is None or self.decoder is None) and self._stage_2_factory:
            self.upsampler, self.decoder = self._stage_2_factory()
        if self.upsampler is None or self.decoder is None:
            raise ModelLoadError(
                component="latent_upsampler",
                reason=(
                    "stage 2 was requested but the latent upsampler or the diffusion decoder "
                    "was not loaded. Refusing to silently fall back to a single-stage render, "
                    "which would look like it worked and be visibly worse."
                ),
            )

    def __call__(self, *, enable_stage_2: bool = False, **kwargs):
        kwargs = _with_real_conditions(kwargs)
        if not enable_stage_2:
            # The distilled recipe applies here too. It used to apply ONLY on
            # the stage-2 path, which is how eleven of twelve host runs were
            # generated with a linear schedule and SFT guidance.
            return self.pipeline(**kwargs, **distilled_kwargs())
        self._stage_2_ready()

        stage_1 = self.pipeline(**kwargs, **stage_1_kwargs())
        upsampled = self.upsampler(**upsample_kwargs(_latents_of(stage_1)))
        stage_2 = self.pipeline(
            **{k: v for k, v in kwargs.items() if k not in STAGE_2_STRIPPED_KEYS},
            **stage_2_kwargs(_latents_of(upsampled), audio_latents=_audio_of(stage_1)),
        )
        decoded = self.decoder(
            **decode_kwargs(_latents_of(stage_2), generator=kwargs.get("generator"))
        )
        return self._with_audio(decoded, stage_2)

    def _with_audio(self, decoded, stage_2):
        """Reunite the diffusion-decoded frames with the audio stage 2 produced.

        **The diffusion decoder returns video only.** ``LTX2VideoDecodeOutput``
        has a ``frames`` field and no ``audio`` field at all, and
        ``output_type="latent"`` skipped the vocoder on the way in, so stage 2's
        audio is still latents. Returning the decoder's output as-is therefore
        exports a **silent file** on all six audio-bearing modes -- and
        ``outputs.py`` reads the track with ``getattr(output, "audio", None)``,
        so nothing raises, nothing logs, and the run reports success.

        Video-only modes have no audio to reunite, so they get the decoder's
        output unchanged rather than an ``audio=None`` field they never had.
        """
        audio_latents = _audio_of(stage_2)
        if audio_latents is None:
            return decoded

        audio_vae = getattr(self.pipeline, "audio_vae", None)
        vocoder = getattr(self.pipeline, "vocoder", None)
        if audio_vae is None or vocoder is None:
            raise ModelLoadError(
                component="vocoder",
                reason=(
                    "stage 2 produced audio latents but the pipeline exposes no audio_vae or "
                    "vocoder to finish them. Refusing to return a silent video for a mode that "
                    "exports audio."
                ),
            )

        # Upstream's own finish for this path. These latents are already
        # denormalized, which is what `audio_vae.decode` expects.
        mel = audio_vae.decode(audio_latents.to(audio_vae.dtype), return_dict=False)[0]
        audio = vocoder(mel)

        # Upstream's own class, taken off the object we already hold rather than
        # imported. Same type `LTX2PipelineOutput` that the single-stage path
        # returns, so `outputs.py` never branches on which path produced it --
        # and no `import diffusers` on this branch, which would make the one
        # piece of logic that reunites video with audio unrunnable in the
        # default local gate. A branch that cannot be exercised offline is how
        # the audio came to be dropped in the first place.
        return type(stage_2)(frames=_latents_of(decoded), audio=audio)


def _with_real_conditions(kwargs: dict) -> dict:
    """Turn normalised condition specs into real ``LTX2VideoCondition`` objects.

    The last possible moment, and the only layer that may import diffusers. The
    specs arriving here have already been range-checked and had their indices
    converted from pixel frames to latent indices by ``conditioning.py``; this
    only changes their type.
    """
    specs = kwargs.get("conditions")
    if not specs:
        return kwargs
    try:
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    except ImportError:
        return kwargs  # no diffusers: the specs travel as-is (tests, local dev)
    from . import conditioning

    # Specs arriving from generate() are already range-checked and already carry
    # LATENT indices, so re-normalising them would convert a second time and
    # move every anchor. Anything that is NOT a spec reached us by another route
    # and has had no checking at all -- normalise that, so a future call site
    # cannot hand diffusers a raw row and have it silently skipped upstream.
    num_frames = int(kwargs.get("num_frames") or 0)
    prepared = []
    for spec in specs:
        if isinstance(spec, conditioning.ConditionSpec):
            prepared.append(spec)
            continue
        fields = conditioning.normalise_condition(spec, num_frames=num_frames)
        prepared.append(
            conditioning.ConditionSpec(
                source=fields["source"], index=fields["index"], strength=fields["strength"]
            )
        )

    kwargs = dict(kwargs)
    kwargs["conditions"] = [
        LTX2VideoCondition(frames=spec.source, index=spec.index, strength=spec.strength)
        for spec in prepared
    ]
    return kwargs


def _latents_of(output):
    """``output_type="latent"`` returns the latents on the frames field."""
    latents = getattr(output, "frames", None)
    if latents is None:
        latents = getattr(output, "latents", None)
    if latents is None:
        raise ModelLoadError(
            component="stage graph",
            reason="a staged call returned no latents; refusing to continue with nothing",
        )
    return latents


def _audio_of(output):
    """Stage 1's audio latents, when the mode produces audio at all.

    ``output_type="latent"`` skips the vocoder, so this is a latent tensor rather
    than a waveform. Video-only modes have no such field, so absence is normal
    and must not raise -- unlike :func:`_latents_of`, where absence means a stage
    returned nothing and continuing would be meaningless.
    """
    return getattr(output, "audio", None)


#: ZeroGPU's ephemeral-disk ceiling. 📄 Documented platform constant, **not
#: readable from inside the container**: the cap is not a filesystem limit but a
#: quota the platform enforces by watching the workload, which is why exceeding
#: it arrives as an eviction rather than as ``ENOSPC``.
ZEROGPU_DISK_CAP_GB = 150.0

#: The container image itself — torch, CUDA, the wheels. ⚠️ **Not measurable
#: from inside**, so it is carried as a constant and shown as such. It is also
#: the term that pushes bf16 over: 72.2 + 72.2 alone is 144.4 and fits.
IMAGE_FOOTPRINT_GB = 10.0


def _directory_bytes(path) -> int:
    """Bytes under ``path``, or 0 if it does not exist."""
    total = 0
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total


def disk_footprint(*, phase: str = "before packing") -> str:
    """**What we consume against the cap** — and it is a floor, which it says.

    ⚠️ ``shutil.disk_usage(".")`` was here and could never see the constraint.
    On the Space it reported *"1969.8 GB free of 8589.9 GB"* — **the host
    filesystem, 8.6 TB** — while the workload was being evicted for exceeding
    :data:`ZEROGPU_DISK_CAP_GB`. It said that through **two evictions** and
    looked like a check the whole time.

    It is not fixable by pointing it somewhere better: **the cap is not a
    filesystem limit.** The platform enforces it by watching the workload, which
    is why exceeding it arrives as an eviction rather than ``ENOSPC``, and why
    no ``statvfs`` anywhere reports it.

    **The measured figure excludes the container image, which cannot be seen
    from inside — so it is a lower bound and the line says so.** A number
    presented as complete when it is a floor is exactly how the old line
    survived two evictions: it looked like a total. *A floor that declares
    itself is useful; a floor that does not is worse than nothing.*

    Reported at two points because they answer different questions:

    * **before packing** — is there room to pack? This is the report that would
      have predicted both evictions *before* they happened.
    * **after packing** — did we fit, and by how much? This is where the
      53.197 GB packed prediction gets tested from inside rather than inferred.
    """
    from . import preflight

    snapshot = _directory_bytes(preflight.cache_repo_dir()) / 1e9
    packed_dir = os.environ.get("ZEROGPU_OFFLOAD_DIR") or os.path.expanduser("~/.zerogpu/tensors")
    packed = _directory_bytes(packed_dir) / 1e9
    measured = snapshot + packed
    remaining = ZEROGPU_DISK_CAP_GB - measured
    verdict = "OVER" if measured + IMAGE_FOOTPRINT_GB > ZEROGPU_DISK_CAP_GB else "OK"
    return (
        f"[{phase}] {measured:.1f} GB measured of {ZEROGPU_DISK_CAP_GB:.0f} GB cap "
        f"[{verdict}] = snapshot {snapshot:.1f} + packed {packed:.1f}; "
        f"{remaining:.1f} GB left for the image (~{IMAGE_FOOTPRINT_GB:.0f} GB, "
        f"NOT measurable from inside) and anything else. "
        f"**Measured is a FLOOR, not the total.**"
    )


def startup_report(runtime_config, *, cached_files=None, missing_files=None, gpu_size=None) -> str:
    """The lines we will be reading at a distance when something goes wrong.

    Deliberately verbose. When R1 fails on a metered Space behind a 72 GB build,
    these lines are the difference between diagnosing it in one round trip and
    burning several -- and there is no opportunity to add a print statement
    afterwards.

    Reports the resolved precision and expected footprint (so the running
    configuration is never inferred), the measured RAM and disk (so an eviction
    or an OOM has numbers attached), **how many of the expected files were found
    in cache** (so "preload worked" is distinguishable from "preload silently did
    nothing"), and the ZeroGPU size actually obtained after packing (so the 2x
    quota cost is visible rather than assumed).
    """

    # `missing_files=None` means "go and look", NOT "nothing is missing". The
    # default used to be `()`, so any caller that omitted the argument got a
    # confident `37/37` without anything having been checked -- the cache count
    # is the line that distinguishes preload-worked from preload-did-nothing, and
    # a line that cannot report failure is not a check. Pass `()` explicitly to
    # assert completeness you have established yourself.
    if missing_files is None:
        from . import preflight

        missing_files = preflight.missing_cached_files()

    expected_files = len(model_files.download_allow_list())
    found = expected_files - len(missing_files) if cached_files is None else cached_files
    try:
        disk = disk_footprint()
    except OSError:  # pragma: no cover - unusual filesystems
        disk = "unavailable"

    lines = [
        "LTX-2.5 Studio starting",
        f"  config          : {runtime_config.describe()}",
        f"  dtype           : {WEIGHT_DTYPE_LABEL} (fixed; no precision switch)",
        f"  weights cached  : {found}/{expected_files} files"
        + (f"  MISSING {len(missing_files)}" if missing_files else ""),
        f"  disk            : {disk}",
        f"  zerogpu size    : {gpu_size or 'not yet known (reported after packing)'}",
        f"  model revision  : {model_files.REVISION}",
    ]
    if missing_files:
        lines.append(f"  first missing   : {missing_files[0]}")
    return "\n".join(lines)


def clear_cache() -> None:
    """Drop cached pipelines. Used by tests and by a precision switch."""
    _CACHE.clear()
