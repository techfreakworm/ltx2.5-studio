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
from dataclasses import dataclass

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
    "StagedOutput",
    "WEIGHT_DTYPE",
    "WEIGHT_DTYPE_LABEL",
    "BASE_KIND",
    "assert_compression_ratios",
    "assert_frame_rate_default",
    "clear_cache",
    "derive_pipeline",
    "get_pipeline",
    "stage_1_kwargs",
    "stage_2_kwargs",
    "upsample_kwargs",
    "STAGE_2_STRIPPED_KEYS",
    "startup_report",
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
    against a 150 GB cap, and the packing runs after both builds; on a 96 GiB
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


def _base_pipeline(*, dtype=None, device: str | None = None):
    """Load the model ONCE. Every workflow is derived from this."""
    if _BASE_CACHE_KEY in _CACHE:
        return _CACHE[_BASE_CACHE_KEY]

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
        built = LTX2ConditionPipeline.from_pretrained(
            model_files.REPO_ID,
            revision=model_files.REVISION,
            torch_dtype=dtype or getattr(torch, WEIGHT_DTYPE),
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
                model_files.REPO_ID,
                revision=model_files.REVISION,
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

    upsampler, decoder = _stage_2_parts(base, dtype=dtype, device=device)
    runner = StageRunner(built, kind=pipeline, upsampler=upsampler, decoder=decoder)
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


def stage_1_kwargs() -> dict:
    """Stage-1 denoise: distilled 8-step schedule, latents out."""
    return {"sigmas": list(DISTILLED_SIGMA_VALUES), "output_type": "latent"}


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


@dataclass(frozen=True)
class StagedOutput:
    """What the two-stage graph returns: the two fields the export path reads.

    ``outputs.write_artifacts`` takes ``frames`` and ``audio`` off the pipeline
    output with ``getattr`` and touches nothing else, so these two fields are the
    entire contract and a single-stage output satisfies it already.

    Deliberately ours rather than upstream's ``LTX2PipelineOutput``. Constructing
    that would put an ``import diffusers`` on the stage-2 finish, making the one
    branch that reunites video with audio unreachable in any environment without
    the weights -- and PLAN §9.2 requires the default local run to be green with
    zero weights on disk. A branch that cannot be exercised offline is how the
    audio came to be dropped in the first place.
    """

    frames: object
    audio: object


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

    def __init__(self, pipeline, *, kind, upsampler=None, decoder=None):
        self.pipeline = pipeline
        self.kind = kind
        self.upsampler = upsampler
        self.decoder = decoder

    def __call__(self, *, enable_stage_2: bool = False, **kwargs):
        kwargs = _with_real_conditions(kwargs)
        if not enable_stage_2:
            return self.pipeline(**kwargs)
        if self.upsampler is None or self.decoder is None:
            raise ModelLoadError(
                component="latent_upsampler",
                reason=(
                    "stage 2 was requested but the latent upsampler or the diffusion decoder "
                    "was not loaded. Refusing to silently fall back to a single-stage render, "
                    "which would look like it worked and be visibly worse."
                ),
            )

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

        return StagedOutput(frames=_latents_of(decoded), audio=audio)


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
    import shutil

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
        usage = shutil.disk_usage(".")
        disk = f"{usage.free / 1e9:.1f} GB free of {usage.total / 1e9:.1f} GB"
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
