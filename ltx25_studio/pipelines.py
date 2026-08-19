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

from dataclasses import dataclass

from . import model_files
from .constraints import (
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
    "assert_compression_ratios",
    "clear_cache",
    "get_pipeline",
    "stage_2_kwargs",
]

#: Mandatory. See the module docstring -- the default silently corrupts output.
DIFFUSION_DECODE_DENORMALIZE = False

_CACHE: dict[str, object] = {}


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


def _text_encoder_model_type(pipeline) -> str:
    encoder = getattr(pipeline, "text_encoder", None)
    config = getattr(encoder, "config", None)
    return str(getattr(config, "model_type", "")) if config is not None else ""


def get_pipeline(*, pipeline: str = "condition", dtype=None, device: str | None = None):
    """Construct (or return the cached) pipeline for one upstream workflow.

    Built on CPU. The ``.to("cuda")`` that triggers ZeroGPU's packing belongs at
    module scope in the entrypoint, not here -- doing it lazily inside a
    ``@spaces.GPU`` function defeats the packer entirely.
    """
    if pipeline in _CACHE:
        return _CACHE[pipeline]

    try:
        import torch
        from diffusers import LTX2ConditionPipeline, LTX2Pipeline
    except ImportError as exc:
        raise ModelLoadError(
            component=pipeline,
            reason=(
                f"the model stack is not installed ({exc}). Install the pinned requirements; "
                f"see docs/PLAN.md §1.3 for why the diffusers revision is exact."
            ),
        ) from exc

    cls = LTX2Pipeline if pipeline == "text2video" else LTX2ConditionPipeline
    try:
        built = cls.from_pretrained(
            model_files.REPO_ID,
            revision=model_files.REVISION,
            torch_dtype=dtype or torch.bfloat16,
        )
    except Exception as exc:
        raise ModelLoadError(component=pipeline, reason=str(exc)) from exc

    if device:
        built.to(device)
    _CACHE[pipeline] = built
    return built


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


def stage_1_kwargs() -> dict:
    """Stage-1 denoise: distilled 8-step schedule, latents out."""
    return {"sigmas": list(DISTILLED_SIGMA_VALUES), "output_type": "latent"}


def stage_2_kwargs(latents) -> dict:
    """Stage-2 refinement over upsampled latents.

    ``noise_scale`` is the first stage-2 sigma: the upsampled latents are already
    partly denoised, so re-noising to 1.0 would discard stage 1 entirely.
    """
    return {
        "latents": latents,
        "sigmas": list(STAGE_2_DISTILLED_SIGMA_VALUES),
        "noise_scale": STAGE_2_DISTILLED_SIGMA_VALUES[0],
        "output_type": "latent",
    }


def decode_kwargs(latents) -> dict:
    """Diffusion decode. ``denormalize=False`` is not optional."""
    return {"latents": latents, "denormalize": DIFFUSION_DECODE_DENORMALIZE}


def clear_cache() -> None:
    """Drop cached pipelines. Used by tests and by a precision switch."""
    _CACHE.clear()
