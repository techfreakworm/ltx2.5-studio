"""Pipeline construction, the stage graph, and the ZeroGPU placement rules.

PLAN §7.4, §8, §5.5.

**Placement (§8).** ZeroGPU replaces CPU offload with its own O_DIRECT
disk-packing engine. The correct pattern inverts normal GPU-box practice: build
on CPU, call ``.to("cuda")`` at module scope -- where it is intercepted and each
tensor is registered for packing; the platform writes the packed copy at
``demo.launch()`` -- and apply ``@spaces.GPU`` statically at import. Off CUDA
(``LTX25_DEVICE=mps|cpu``) the same build moves to that device; see
``device.py`` for what each device needs. ``enable_model_cpu_offload()`` is
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

from . import local_runtime, model_files
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
    "check_cpu_decoder_size",
    "check_decoder_minimum",
    "check_decoder_size",
    "clear_cache",
    "configure_vae_tiling",
    "decoder_minimum",
    "derive_pipeline",
    "get_pipeline",
    "mmap_backing_report",
    "pinned_decoder_config",
    "VAE_TILE_MIN_HEIGHT",
    "VAE_TILE_MIN_WIDTH",
    "VAE_TILE_MIN_NUM_FRAMES",
    "VAE_TILE_STRIDE_HEIGHT",
    "VAE_TILE_STRIDE_WIDTH",
    "VAE_TILE_STRIDE_NUM_FRAMES",
    "stage_1_kwargs",
    "stage_2_geometry",
    "stage_2_size_note",
    "STAGE_2_SPATIAL_SCALE",
    "distilled_kwargs",
    "DISTILLED_GUIDANCE",
    "stage_2_kwargs",
    "upsample_kwargs",
    "variation_kwargs",
    "STAGE_2_STRIPPED_KEYS",
    "startup_report",
    "disk_footprint",
    "ZEROGPU_DISK_CAP_GB",
    "QUANTIZED_COMPONENTS",
    "TRANSFORMER_INT8_SKIP_MODULES",
    "peak_rss_gib",
]

#: Mandatory. See the module docstring -- the default silently corrupts output.
DIFFUSION_DECODE_DENORMALIZE = False

#: The checkpoint's dtype and the load dtype on CUDA and MPS. int8 is applied on
#: top of it to ``QUANTIZED_COMPONENTS`` (``LTX25_QUANTIZATION``); a CPU load
#: may use float32 (``LTX25_CPU_DTYPE``). ``device.resolve`` decides per device.
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


# ---------------------------------------------------------------------------
# VAE decode tiling -- the envelope is the measured one, and the gate is
# upstream's own dispatch
# ---------------------------------------------------------------------------

#: The largest decode PROVEN to fit the 96 GB MIG slice untiled, from live
#: runs: 704x480x49 decoded clean; 1216x704x121 completed its denoise and then
#: died inside ``vae.decode``'s conv3d on the CUDACachingAllocator (~26 GB per
#: full-resolution intermediate at 121 frames, on ~35 GB of resident weights).
#:
#: These values are handed to ``AutoencoderKLLTX2.enable_tiling`` so that
#: **upstream's own dispatch is the size gate**: its ``_decode`` compares the
#: actual latent tensor against ``tile_sample_min_* // compression_ratio``
#: with strict ``>`` (read at the pinned ``ac56fa2f``, byte-identical in the
#: local 0.37.1 copy), so
#:
#: * anything at or inside the proven envelope -- 704x480 spatial, 49 frames
#:   temporal, which includes the sha256-banked 256x256x9 -- takes the SAME
#:   untiled code path as before this change, structurally byte-identical;
#: * anything beyond it decodes in tiles of exactly the proven-fit shape,
#:   temporal chunks re-dispatching through spatial tiling (upstream's
#:   ``_temporal_tiled_decode`` composes both), so the peak decode tensor is
#:   bounded by a shape the card has already survived.
#:
#: Keying on the tensor being decoded is also what makes stage 2 need no
#: branch of ours: its inputs are 2x the request, and 2x is what the dispatch
#: sees. (Stage 2's FINAL pixels come from the diffusion decoder, a separate
#: model this flag set does not govern -- see StageRunner.)
#:
#: Always-on with upstream's DEFAULTS was measured against the source and
#: rejected: the default 512px tile minimum would have re-routed the
#: proven-fitting 704x480 decode through the tiled blend path, perturbing a
#: working size for no memory benefit. The envelope values keep the change
#: invisible everywhere the platform has already been proven.
VAE_TILE_MIN_HEIGHT = 480
VAE_TILE_MIN_WIDTH = 704
#: 56 pixel frames -> a latent threshold of 56 // 8 = 7, and a 49-frame video
#: occupies (49 - 1) // 8 + 1 = 7 latent frames -- so ``7 > 7`` is False and
#: the proven 49-frame decode is EXACTLY the last untouched point. This is the
#: off-by-one that matters: 57 frames occupy 8 latent frames and chunk.
VAE_TILE_MIN_NUM_FRAMES = 56
#: Strides preserve upstream's own default overlaps -- 64 px spatial blend
#: (512 - 448) and 8-frame temporal blend (16 - 8) -- so the seam-blending
#: behaviour for oversized decodes is exactly the one upstream ships, at our
#: tile size.
VAE_TILE_STRIDE_HEIGHT = 416
VAE_TILE_STRIDE_WIDTH = 640
VAE_TILE_STRIDE_NUM_FRAMES = 48


def configure_vae_tiling(vae) -> None:
    """Enable tiled + framewise decoding with the proven-fit envelope.

    Set once on the ONE shared VAE at load: every derived pipeline and the
    stage-2 upsampler hold this same object, so the envelope governs each
    place a video tensor meets ``vae.decode`` or ``vae.encode`` -- the
    in-pipeline decode that died at 1216x704x121, and the conditioning
    encodes, whose flagship-size inputs carry the same arithmetic.

    Both framewise flags are set although upstream's dispatch currently reads
    only ``use_framewise_decoding`` (its ``_encode`` checks the *decoding*
    flag -- an upstream quirk, identical at the pinned SHA): setting the
    encoding flag too costs nothing today and keeps this correct the day the
    quirk is fixed.

    The divisibility tripwire is not pedantry: the dispatch converts these
    pixel values to latent units by floor division, so a threshold that does
    not divide exactly SHIFTS the envelope silently -- the proven boundary
    would drift without any test noticing. Raise instead, naming the drift.
    """
    spatial = int(getattr(vae, "spatial_compression_ratio", DEFAULT_SPATIAL_COMPRESSION_RATIO))
    temporal = int(getattr(vae, "temporal_compression_ratio", DEFAULT_TEMPORAL_COMPRESSION_RATIO))
    exact = {
        "VAE_TILE_MIN_HEIGHT": (VAE_TILE_MIN_HEIGHT, spatial),
        "VAE_TILE_MIN_WIDTH": (VAE_TILE_MIN_WIDTH, spatial),
        "VAE_TILE_STRIDE_HEIGHT": (VAE_TILE_STRIDE_HEIGHT, spatial),
        "VAE_TILE_STRIDE_WIDTH": (VAE_TILE_STRIDE_WIDTH, spatial),
        "VAE_TILE_MIN_NUM_FRAMES": (VAE_TILE_MIN_NUM_FRAMES, temporal),
        "VAE_TILE_STRIDE_NUM_FRAMES": (VAE_TILE_STRIDE_NUM_FRAMES, temporal),
    }
    off_grid = {name: value for name, (value, ratio) in exact.items() if value % ratio}
    if off_grid:
        raise ModelLoadError(
            component="vae",
            reason=(
                f"VAE tile thresholds must divide the compression ratios exactly "
                f"(spatial {spatial}, temporal {temporal}) or the floor division in "
                f"upstream's dispatch silently shifts the proven-fit envelope: {off_grid}. "
                f"Fix the constants in pipelines.py before continuing."
            ),
        )
    vae.enable_tiling(
        tile_sample_min_height=VAE_TILE_MIN_HEIGHT,
        tile_sample_min_width=VAE_TILE_MIN_WIDTH,
        tile_sample_min_num_frames=VAE_TILE_MIN_NUM_FRAMES,
        tile_sample_stride_height=VAE_TILE_STRIDE_HEIGHT,
        tile_sample_stride_width=VAE_TILE_STRIDE_WIDTH,
        tile_sample_stride_num_frames=VAE_TILE_STRIDE_NUM_FRAMES,
    )
    vae.use_framewise_decoding = True
    vae.use_framewise_encoding = True


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
#: The diffusion decoder alone: a single-stage diffusion decode needs no upsampler.
_DECODER_CACHE_KEY = "__decoder__"


def derive_pipeline(base, target_cls):
    """Build ``target_cls`` from ``base``'s ALREADY-LOADED modules, sharing them as-is.

    ``from_pretrained`` twice materialises two complete, independent copies of
    the model -- ~67 GiB each. On ZeroGPU that is ~134 GiB packed to disk
    against :data:`ZEROGPU_DISK_CAP_GB`, and packing runs after both builds; on a 96 GiB
    card it is an outright OOM. The pipeline classes take **identical**
    weight-bearing components at our pinned revision, so there is no version of
    this where two copies are wanted.

    Constructed directly from ``base.components``, filtered to what the target
    accepts (``audio_scheduler`` is the one asymmetry). **Not** ``from_pipe``:
    it calls ``.to(dtype)`` on every module, which a GGUF-quantized transformer
    refuses ("Casting a quantized model to a new dtype is unsupported"), which
    bitsandbytes skips with a warning, and which -- without a dtype -- casts the
    shared modules to float32 in place. Nothing here moves or casts a module.
    """
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

#: The transformer Linears that must stay bf16 under ``LLM.int8()``.
#:
#: ⚠️ **Read off the real checkpoint, not inferred from another DiT.** That
#: distinction is the entire finding. The obvious skip list for a PixArt-lineage
#: transformer is ``("norm_out.linear", "adaln_single", "caption_projection",
#: "proj_out")`` — and on LTX-2.5 the first three match **nothing at all**:
#:
#: * ``norm_out`` is ``nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)``
#:   (``transformer_ltx2.py:1357``) — **zero parameters, no ``.linear`` child.**
#: * ``adaln_single`` does not exist. The modulation Linears live in eight
#:   *top-level* :class:`LTX2AdaLayerNormSingle` modules, named individually
#:   below (``transformer_ltx2.py:1219-1257``). The blocks carry no adaLN Linear
#:   of their own — their modulation is a bare ``scale_shift_table``.
#: * ``caption_projection`` is gated on ``use_prompt_embeddings``, which is
#:   ``false`` in this checkpoint's ``transformer/config.json``.
#:
#: A list of three no-ops and one real name would have looked like a fix,
#: deployed, and produced the same garbage. The names here were derived from
#: the 4091-entry ``diffusion_pytorch_model.safetensors.index.json`` weight map
#: at revision ``871165de...`` and cross-checked against the module
#: constructors at the pinned diffusers SHA. ``test_int8_skip_modules.py``
#: re-derives them from that same index so this cannot rot.
#:
#: Every ``scale_shift_table`` is an ``nn.Parameter``, not a module, so bnb --
#: which only replaces ``nn.Linear`` -- can never reach one. They need no entry
#: and get none.
#:
#: ⚠️ **Do not shorten any of these.** bnb's matcher
#: (``quantizers/bitsandbytes/utils.py:58-62``) has two gates: the entry equals
#: a module's **local child name**, or ``"<entry>."`` appears as a **substring**
#: of the full dotted path. The second is substring containment, not a path
#: prefix test, so short names spread. Measured against the real tree:
#:
#: * ``"proj"`` alone skips **96** Linears — every feed-forward ``net.0.proj``
#:   in all 48 blocks, video and audio, because ``proj`` is their local child
#:   name and gate 1 fires. Those are the widest matrices in the model
#:   (``FeedForward`` expands ``inner_dim`` by its ``mult``), so that is not a
#:   safety margin; it un-quantises the single largest block of weights int8
#:   was turned on to shrink, and reports nothing while doing it.
#: * ``"linear"`` alone skips **8** — the ``LTX2AdaLayerNormSingle.linear``
#:   leaves — while silently missing all sixteen ``linear_1``/``linear_2``
#:   timestep-embedder Linears. Over-broad and under-broad at once.
#:
#: The list below is deliberately mixed: ``to_gate_logits`` is a leaf name and
#: matches only via gate 1; the ``time_embed``-style entries are parent
#: prefixes and match only via gate 2.
#:
#: ℹ️ **Two entries are redundant today and are kept anyway.** Because gate 2 is
#: substring containment, ``time_embed`` already matches
#: ``audio_time_embed.emb.timestep_embedder.linear_1`` — so ``audio_time_embed``
#: and, for the same reason, ``audio_prompt_adaln`` spare nothing their video
#: counterparts do not already spare. Removing them is measurably a no-op
#: (``test_which_entries_are_load_bearing_is_pinned``). They stay for two
#: reasons: the audio tower's coverage should be legible without deriving it
#: from a matcher's substring semantics, and if upstream ever tightens gate 2
#: into a real path-prefix test — which is what that matcher arguably should be
#: — these stop being redundant and become the only thing covering the audio
#: tower. ⚠️ The subsumption runs one way only: dropping ``prompt_adaln`` while
#: keeping ``audio_prompt_adaln`` leaves the VIDEO prompt adaLN quantised.
TRANSFORMER_INT8_SKIP_MODULES = (
    # --- conditioning: the eight top-level LTX2AdaLayerNormSingle modules ----
    # Each contributes `.emb.timestep_embedder.linear_1`, `.linear_2` and
    # `.linear`. These compute the shift/scale/gate that every block's
    # modulation is built from, so int8 error here is not attenuated -- it is
    # multiplied across all 48 blocks and both towers. This is the collapse
    # mechanism: quantised conditioning yields correctly-shaped, high-variance
    # noise rather than a degraded picture.
    "time_embed",
    "audio_time_embed",
    "prompt_adaln",
    "audio_prompt_adaln",
    "av_cross_attn_video_scale_shift",
    "av_cross_attn_audio_scale_shift",
    "av_cross_attn_video_a2v_gate",
    "av_cross_attn_audio_v2a_gate",
    # --- the input and output projections ------------------------------------
    # One Linear each. `proj_out` is the last op before the latent leaves the
    # transformer, so its error reaches the VAE unattenuated; upstream names it
    # as the canonical skip in its own comment at `bnb_quantizer.py:261`:
    # "We may keep some modules such as the `proj_out` in their original dtype
    # for numerical stability reasons." `proj_in` is the mirror argument at the
    # other end -- it is the only path the patchified latent has in.
    "proj_in",
    "proj_out",
    "audio_proj_in",
    "audio_proj_out",
    # --- attention gates ------------------------------------------------------
    # 288 Linears (6 attentions x 48 blocks), and the largest single block of
    # skips. `to_gate_logits` is `nn.Linear(query_dim, heads)`
    # (`transformer_ltx2.py:382`) -- a HEAD-COUNT-wide output, i.e. 32 numbers
    # from 4096. Rounding a per-head gate is not a small perturbation of a big
    # tensor; it is a coarse decision about whether an attention head
    # contributes at all.
    "to_gate_logits",
)


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

    ❌ **Both of those arguments were about memory, and the config that came
    out of them was blanket.** It was also the bug. Every frame the deployed
    Space ever produced was correctly-shaped, high-variance garbage — colour
    static and flat fields, at every resolution, in every mode,
    deterministically. The same code on local MPS bf16 looked good. The only
    difference in the deployed path is this function, and the config it
    returned named no ``llm_int8_skip_modules`` at all, so bnb replaced
    **every** ``nn.Linear`` in the transformer — including the adaLN/modulation
    heads and the input/output projections.

    Deterministic garbage at every setting is the signature of broken
    *conditioning*, not of lost precision. Quantisation noise degrades an image;
    it does not replace it with static. When the modulation path is rounded, the
    shift/scale/gate that every one of the 48 blocks is conditioned on stops
    carrying the timestep and the prompt, the denoiser is no longer steered by
    the schedule, and what comes out is a correctly-shaped sample from the wrong
    distribution — which is exactly what a high-variance field is.

    :data:`TRANSFORMER_INT8_SKIP_MODULES` is the fix and carries the derivation.
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
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
        from diffusers import PipelineQuantizationConfig
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
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
    # ✅ **`quant_mapping`, NOT `quant_backend`.** The blanket
    # `quant_backend="bitsandbytes_8bit"` form shipped every frame the Space
    # ever produced as high-variance garbage, because it quantises the
    # conditioning path (see `TRANSFORMER_INT8_SKIP_MODULES`). Three separate
    # reasons say the fix belongs in a mapping rather than in wider
    # `quant_kwargs`:
    #
    # 1. **The two components need different configs.** The transformer needs
    #    the skips; `text_encoder` is a Gemma LLM, which is precisely the case
    #    `LLM.int8()` was designed and validated for, and is not implicated in
    #    this bug. `quant_backend` can only express ONE config for all of
    #    `components_to_quantize`.
    #
    # 2. **`quant_backend` ALIASES the skip list across components.**
    #    `pipe_quant_config.py:_resolve_quant_config` splats the same
    #    `self.quant_kwargs` dict for every component, and
    #    `BitsAndBytesConfig.__init__` stores `llm_int8_skip_modules` without
    #    copying (`quantization_config.py:248`). `bnb_quantizer.py:261` then
    #    takes that same list -- again without copying -- and `.extend()`s it
    #    IN PLACE at `:267` and `:279`. So one shared list object accumulates
    #    `keep_in_fp32_modules` and offloaded keys from every component in
    #    turn. `quant_mapping` hands each component its own config object
    #    (`_resolve_quant_config` returns `quant_mapping[module_name]`
    #    verbatim), so the in-place mutation stays contained.
    #
    # 3. **`quant_backend` couples us to two libraries' signatures.**
    #    `pipe_quant_config.py:_validate_init_kwargs_in_backends` raises unless
    #    diffusers' and transformers' `BitsAndBytesConfig.__init__` parameter
    #    sets are EQUAL. They are identical today, but requirements.txt pins
    #    only `transformers>=5.15.1,<6` -- a minor release that adds one kwarg
    #    on either side turns int8 into a hard startup failure. The
    #    `quant_mapping` path skips that check entirely.
    #
    # ⚠️ The config CLASS differs per component and is not interchangeable. In
    # the granular branch `_resolve_quant_config` ignores its own `is_diffusers`
    # argument and returns the mapped object as-is, so each component's config
    # must be the one ITS OWN loader accepts: diffusers' class for the
    # diffusers-owned transformer, transformers' class for the Gemma text
    # encoder. Upstream's own docs are explicit -- "You need to import and use
    # `transformers.BitsAndBytesConfig` for components that come from
    # Transformers."
    #
    # `llm_int8_threshold` is left at its 6.0 default deliberately. It governs
    # a gradual outlier/fidelity tradeoff, not a collapse, and this change
    # already moves one variable.
    quant_mapping = {
        "transformer": DiffusersBitsAndBytesConfig(
            load_in_8bit=True,
            # `list(...)`, not the tuple, for two independent reasons: the
            # config's `post_init` rejects a non-list outright, and
            # `bnb_quantizer` mutates whatever it is handed in place -- so a
            # fresh copy per call keeps the module constant from growing
            # entries across loads.
            llm_int8_skip_modules=list(TRANSFORMER_INT8_SKIP_MODULES),
        ),
        "text_encoder": TransformersBitsAndBytesConfig(load_in_8bit=True),
    }

    # The mapping keys ARE `components_to_quantize` on this path, so the two
    # declarations must not be able to drift. `QUANTIZED_COMPONENTS` remains the
    # single source of truth -- it is what the guard above validates against
    # `model_files.COMPONENTS` and what the load log reports -- and this asserts
    # the mapping still covers exactly it. Without this, widening the constant
    # would update the guard and the log while quietly quantising nothing new.
    if set(quant_mapping) != set(QUANTIZED_COMPONENTS):
        raise ModelLoadError(
            component="transformer",
            reason=(
                f"the int8 quant_mapping covers {sorted(quant_mapping)} but "
                f"QUANTIZED_COMPONENTS declares {sorted(QUANTIZED_COMPONENTS)}. On the "
                f"quant_mapping path the mapping keys are the only thing that decides "
                f"what is quantised, so a component named in the constant but absent "
                f"from the mapping would load at full precision while the startup log "
                f"reported it as quantised."
            ),
        )

    return PipelineQuantizationConfig(quant_mapping=quant_mapping)


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


def _parameter_bearing_components(pipeline_object):
    """``(name, module)`` for every component that actually holds parameters.

    Reads ``.components`` where the pipeline offers one (diffusers does) and
    falls back to the names this stack is known to carry, so the report works
    against a duck-typed pipeline in the offline suite as well as the real
    object on the Space.
    """
    components = getattr(pipeline_object, "components", None)
    if isinstance(components, dict):
        items = list(components.items())
    else:
        names = ("transformer", "connectors", "vae", "audio_vae", "text_encoder", "vocoder")
        items = [(name, getattr(pipeline_object, name, None)) for name in names]
    return [
        (name, module)
        for name, module in items
        if callable(getattr(module, "named_parameters", None))
    ]


#: Sentinel for "read the live maps now": ``None`` is a real value here -- it
#: is what a platform without ``/proc/self/maps`` returns -- so it cannot
#: double as the default.
_READ_MAPS = object()


def mmap_backing_report(
    pipeline_object, *, maps_text=_READ_MAPS
) -> tuple[tuple[bool, bool, str], ...]:
    """``(checked, offending, message)`` per component, for the startup log.

    This is the wiring for ``zerogpu.check_mmap_backed`` -- previously exported,
    tested, documented as the detector for the silent 72 GB leak, and called by
    nothing (the §8 shape: an advertised guard reporting safety it never
    checked). The three-way verdict is the point of the tuple: "checked and
    fine", "checked and OFFENDING", and "could not check" must all be
    distinguishable in the log, because collapsing the third into the first is
    exactly the defect ``check_mmap_backed`` refuses internally.

    The maps snapshot is read once and shared: every component is judged
    against the same instant, and where the platform offers no maps at all the
    report short-circuits to that single line rather than repeating the same
    non-answer per component.
    """
    from . import zerogpu

    if maps_text is _READ_MAPS:
        maps_text = zerogpu.read_process_maps()
    report = []
    for name, module in _parameter_bearing_components(pipeline_object):
        checked, message = zerogpu.check_mmap_backed(module, name=name, maps_text=maps_text)
        # The substring is pinned by test_zerogpu's own assertions on
        # check_mmap_backed's offender message, so drift there fails tests
        # before it can misclassify here.
        report.append((checked, checked and "not mmap'd" in message, message))
        if not checked:
            break
    return tuple(report)


def _base_pipeline(*, dtype=None, device: str | None = None, quantization: str | None = None):
    """Load the model ONCE. Every workflow is derived from this."""
    if _BASE_CACHE_KEY in _CACHE:
        return _CACHE[_BASE_CACHE_KEY]

    from . import device as device_module

    if quantization is None:
        quantization = device_module.quantization_for(device)
    if device == "mps":
        # Before any weight is read: a torch that computes wrong results on this
        # GPU must refuse here, not after a load (enforce), or say so (warn).
        device_module.require_mps_numerics()

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
        # ⚠️ The load `dtype` MUST match the checkpoint's on-disk dtype. It is not a
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
        # (A CPU load at float32 is a deliberate cast; ZeroGPU never sees one.)
        before = peak_rss_gib()
        quantization_config = _quantization_config(quantization)
        weight_dtype = dtype or getattr(torch, device_module.weight_dtype_for(device))
        load_kwargs = {"dtype": weight_dtype}
        if quantization_config is not None:
            # Covers `QUANTIZED_COMPONENTS` -- see `_quantization_config`.
            load_kwargs["quantization_config"] = quantization_config
        built = LTX2ConditionPipeline.from_pretrained(_snapshot_path(), **load_kwargs)
        after = peak_rss_gib()
        # Reported whatever happens. A bf16 load peaks far below the weight
        # footprint because safetensors mmaps; a quantising load materialises
        # tensors and does not, so this number is the one that decides whether
        # int8 is viable on a memory-capped platform -- not the disk arithmetic.
        #
        # ⚠️ It named only the transformer while TWO components were quantised.
        # Not silent -- wrong in the direction that misdirects, because this is
        # the line read to decide whether the RAM hypothesis survives, and a new
        # RSS number beside "loaded transformer as int8" invites attributing the
        # whole change to the transformer.
        #
        # DERIVED from QUANTIZED_COMPONENTS, deliberately. That is the opposite
        # of the rule for a test's expectation -- a message interpolating the
        # constant it checks tracks any mutation and can never contradict it --
        # and the difference is what each one is FOR. A test asserts an
        # independent expectation; **a log reports what was requested**, so it
        # must move when the request moves. The guard in `_quantization_config`
        # is what makes "requested" and "what happened" the same thing for any
        # name that survives it. Do not "fix" this into a hardcoded string.
        if quantization_config is not None:
            what = f"quantised {', '.join(QUANTIZED_COMPONENTS)} as {quantization}"
        else:
            what = f"no components quantised; all at {_dtype_label(weight_dtype)}"
        _log().info(
            "%s | peak RSS %.2f -> %.2f GiB",
            what,
            before if before is not None else float("nan"),
            after if after is not None else float("nan"),
        )
    except Exception as exc:
        raise ModelLoadError(component=BASE_KIND, reason=str(exc)) from exc

    # The mmap-backing report -- the check that detects the silent 72 GB leak:
    # ZeroGPU reclaims the HF snapshot after packing only for tensors still
    # mmap'd from prunable blobs, and anything that materialised them into
    # anonymous memory leaves the Space dying at steady state with 72 GB it
    # cannot account for and no error explaining why. It sat parked here
    # citing "ZeroGPU is parked" -- but ZeroGPU is the live deployment, so the
    # one place the check can fire was the one place it did not run.
    #
    # LOG-ONLY this deploy, deliberately. ZEROGPU_DISK.md §6 designed it as a
    # raise, but a wrong raise here bricks a working Space at startup;
    # promotion to ModelLoadError is a separate decision, to be taken after
    # reading these lines from a live boot -- not a default smuggled in with
    # the wiring. Runs BEFORE the device move: on ZeroGPU, after `.to(device)`
    # `named_parameters()` returns the platform's fakes, not the mmap'd tensors.
    for checked, offending, message in mmap_backing_report(built):
        (_log().warning if offending else _log().info)("%s%s", "" if checked else "⚠️ ", message)

    if device == "mps":
        # MPS has no float64; the RoPE frequency grid is float64 by config.
        # Computed on the CPU at that precision, as CUDA computes it.
        handled = device_module.compute_rope_on_cpu(*_parameter_bearing_components(built))
        _log().info("RoPE computed on the CPU in float64 for %d modules", len(handled))

    if device:
        # On ZeroGPU this is intercepted at module scope and each tensor is
        # registered for packing; the packed copy is written at demo.launch().
        # Doing it lazily inside a @spaces.GPU function skips packing and
        # re-transfers from RAM on every cold worker. Only the base is moved:
        # everything else shares these very modules.
        built.to(device)
        # A pre-launch reading, labelled as one: nothing has been packed yet.
        _log().info("disk %s", disk_footprint(phase="after the device move, before launch"))

    vae = getattr(built, "vae", None)
    if vae is not None:
        assert_compression_ratios(vae)
        configure_vae_tiling(vae)
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
    the two model checkpoints themselves. The decoder is the same object a
    single-stage diffusion decode uses (:func:`_decoder_part`).
    """
    if _STAGE_2_CACHE_KEY in _CACHE:
        return _CACHE[_STAGE_2_CACHE_KEY]
    try:
        from diffusers import LTX2LatentUpsamplePipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    except ImportError as exc:
        raise ModelLoadError(
            component="latent_upsampler",
            reason=f"the stage-2 classes are not importable ({exc}).",
        ) from exc

    decoder = _decoder_part(base, dtype=dtype, device=device)
    upsampler_model = _load_part(
        LTX2LatentUpsamplerModel, "latent_upsampler", dtype=dtype, device=device
    )
    if device:
        # Not shared with the base, so `_base_pipeline`'s `.to(device)` does not
        # reach it.
        upsampler_model.to(device)
    upsampler = LTX2LatentUpsamplePipeline(vae=base.vae, latent_upsampler=upsampler_model)

    parts = (upsampler, decoder)
    _CACHE[_STAGE_2_CACHE_KEY] = parts
    return parts


def _load_part(cls, component, *, dtype=None, device=None):
    """One stage-2 checkpoint from its own directory, at the checkpoint's dtype.

    The directory comes from ``preflight.component_dir``: each component at the
    revision it is pinned to (the diffusion decoder's is later than the rest).
    The load is exact: anything missing, mismatched or unexpected -- other than
    ``model_files.TOLERATED_UNEXPECTED_KEYS`` -- is a load error.
    """
    import torch

    from . import device as device_module
    from . import preflight

    # Same rule as the base load, and the same reason: the checkpoints are bf16,
    # so this is a no-op rather than a conversion. See `_base_pipeline`.
    weight_dtype = dtype or getattr(torch, device_module.weight_dtype_for(device))
    directory = preflight.component_dir(component)
    if not directory.is_dir():
        raise ModelLoadError(
            component=component,
            reason=(
                f"no local files at {directory}. Weights are fetched by "
                f"`preflight.ensure_weights()`, each component at its pinned revision."
            ),
        )
    try:
        loaded = cls.from_pretrained(directory, dtype=weight_dtype, output_loading_info=True)
    except Exception as exc:
        raise ModelLoadError(component=component, reason=str(exc)) from exc
    model, info = loaded if isinstance(loaded, tuple) else (loaded, None)
    if info is not None:
        _require_exact_load(component, info)
    return model


def _require_exact_load(component: str, info: dict) -> None:
    """Refuse a checkpoint that does not match its class, beyond the tolerated keys."""
    tolerated = set(model_files.TOLERATED_UNEXPECTED_KEYS.get(component, ()))
    unexpected = sorted(set(info.get("unexpected_keys") or ()) - tolerated)
    problems = [
        f"{label} {sorted(map(str, values))[:5]}"
        for label, values in (
            ("missing keys", info.get("missing_keys") or ()),
            ("mismatched keys", info.get("mismatched_keys") or ()),
            ("load errors", info.get("error_msgs") or ()),
            ("unexpected keys", unexpected),
        )
        if values
    ]
    if problems:
        raise ModelLoadError(
            component=component,
            reason="its checkpoint does not match the pinned class: " + "; ".join(problems),
        )
    ignored = sorted(set(info.get("unexpected_keys") or ()) & tolerated)
    if ignored:
        _log().info("%s: ignored the expected extra key(s) %s", component, ", ".join(ignored))


def _decoder_part(base, *, dtype=None, device: str | None = None):
    """The diffusion decoder pipeline, built once and shared by every decode.

    Off CUDA the decoder keeps its default FlexAttention processor and stays on
    the CPU: NATTEN has no macOS build and FlexAttention refuses MPS tensors.
    That path is dense, so its size is bounded (``device.check_stage_2_size``,
    and the single-stage decode policy).
    """
    if _DECODER_CACHE_KEY in _CACHE:
        return _CACHE[_DECODER_CACHE_KEY]
    try:
        from diffusers import LTX2VideoDiffusionDecoderModel
        from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
            LTX2VideoVaeNeighborhoodNattenProcessor,
        )
        from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode import (
            LTX2VideoDiffusionDecodePipeline,
        )
    except ImportError as exc:
        raise ModelLoadError(
            component="diffusion_decoder",
            reason=(
                f"the decoder classes are not importable ({exc}). The diffusion decoder also "
                f"needs the `kernels` package, which fetches the NATTEN kernel from the Hub; "
                f"a local `natten` install is explicitly not used."
            ),
        ) from exc

    on_cuda = device in (None, "cuda")
    decoder_model = _load_part(
        LTX2VideoDiffusionDecoderModel, "diffusion_decoder", dtype=dtype, device=device
    )
    # ⚠️ Not an optimisation; required to run at all on CUDA.
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
    if on_cuda:
        try:
            decoder_model.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
        except Exception as exc:
            raise ModelLoadError(
                component="diffusion_decoder",
                reason=f"its NATTEN attention kernel could not be built ({exc})",
            ) from exc
    # Tiling bounds peak memory by the tile size rather than by the video size.
    decoder_model.enable_tiling()
    if device:
        # Off CUDA the decoder stays on the CPU (FlexAttention refuses MPS).
        decoder_model.to(device if on_cuda else "cpu")
    # `vae` is deliberately omitted: it is optional here and consulted only to
    # denormalize, which `DIFFUSION_DECODE_DENORMALIZE` never asks for.
    decoder = LTX2VideoDiffusionDecodePipeline(
        diffusion_decoder=decoder_model, scheduler=base.scheduler
    )
    _CACHE[_DECODER_CACHE_KEY] = decoder
    return decoder


def get_pipeline(*, pipeline: str = BASE_KIND, dtype=None, device: str | None = None):
    """The runner for one workflow, sharing one loaded copy of the model.

    With no ``device`` the process's resolved device is used, and a machine with
    no generation device (or an MPS torch that fails its self-test) is refused
    before anything is loaded -- never a silent full load on the CPU.

    On mps/cpu with real weights the local runner (``local_runtime``) is
    returned; a ``LTX25_MODEL_SNAPSHOT_DIR`` override keeps the resident
    :class:`StageRunner`. CUDA is unchanged.
    """
    if pipeline in _CACHE:
        return _CACHE[pipeline]

    from . import device as device_module

    if device is None:
        device = device_module.current().require_generation()
    elif device == "mps":
        device_module.require_mps_numerics()

    if device in ("mps", "cpu") and _uses_local_runner():
        from .config import load_config
        from .preflight import snapshot_dir

        runtime_config = load_config()
        return local_runtime.build_local_runner(
            kind=pipeline,
            device=device,
            dtype=dtype or device_module.weight_dtype_for(device, runtime_config),
            snapshot_dir=snapshot_dir(),
            transformer_gguf=runtime_config.transformer_gguf,
            residency=runtime_config.residency,
            log=_log().info,
        )

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
        device=device,
        stage_2_factory=lambda: _stage_2_parts(base, dtype=dtype, device=device),
        decoder_factory=lambda: _decoder_part(base, dtype=dtype, device=device),
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

    The audio-input modes keep this schedule too: their audio has its own
    level and schedule (``audio_conditioning``), so no run truncates it.
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


#: The request's own size never reaches stage 2. ``LTX2Pipeline`` takes its size
#: from the upsampled latents; ``LTX2ConditionPipeline`` builds its conditioning
#: mask from height/width (default 512x768) even when handed latents, so it is
#: given the upsampled size explicitly (:func:`stage_2_geometry`).
STAGE_2_STRIPPED_KEYS = ("height", "width")


def stage_2_geometry(latents, *, kind, spatial_ratio, temporal_ratio) -> dict:
    """Stage 2's ``num_frames`` (and, for the condition pipeline, its size), read off the latents.

    ``latents`` are the upsampled ``[B, C, F, H, W]`` latents. Under the duration
    head the request has no frame count, so it comes from stage 1's latent frames.
    """
    shape = getattr(latents, "shape", None)
    if shape is None or len(shape) != 5:
        if kind == "text2video":
            return {}
        raise ModelLoadError(
            component="stage graph",
            reason=(
                "the upsampled latents have no [B, C, F, H, W] shape to size stage 2 from; "
                "refusing to guess the size the condition pipeline builds its mask from"
            ),
        )
    _batch, _channels, frames, height, width = (int(n) for n in shape)
    geometry = {"num_frames": (frames - 1) * int(temporal_ratio) + 1}
    if kind != "text2video":
        geometry["height"] = height * int(spatial_ratio)
        geometry["width"] = width * int(spatial_ratio)
    return geometry


def decoder_minimum(config) -> tuple[int, int, int]:
    """Smallest decoded video ``(width, height, frames)`` the diffusion decoder accepts.

    Every neighbourhood-attention stage raises when its grid is smaller than its
    kernel. The grids are walked as ``LTX2VideoDiffusionDecoder3d`` builds them:
    trailing ghost frames padded, each upsample (dropping the duplicate leading
    frame on a temporal x2), the ghosts cropped before stage 5.
    """
    kernels = [tuple(int(k) for k in row) for row in config["decoder_stage_kernels"]]
    strides = [tuple(int(s) for s in row) for row in config["decoder_upsample_strides"]]
    stage5 = tuple(int(k) for k in config["decoder_stage5_kernel"])
    temporal_ratio = int(config["temporal_compression_ratio"])
    spatial_ratio = int(config["spatial_compression_ratio"])
    pad = (kernels[0][0] // 2) * 2

    def fits(axis: int, latent: int) -> bool:
        size = latent + (pad if axis == 0 else 0)
        for kernel, stride in zip(kernels, strides, strict=True):
            if size < kernel[axis]:
                return False
            size = size * stride[axis] - (1 if axis == 0 and stride[0] == 2 else 0)
        if axis == 0:
            size -= pad * temporal_ratio
        return size >= stage5[axis]

    def smallest(axis: int) -> int:
        latent = 1
        while not fits(axis, latent):
            latent += 1
        return latent

    frames, height, width = (smallest(axis) for axis in range(3))
    return width * spatial_ratio, height * spatial_ratio, (frames - 1) * temporal_ratio + 1


def _decoder_config(decoder):
    """The diffusion decoder model's config, or None (a stand-in in the offline gate)."""
    model = getattr(decoder, "diffusion_decoder", None)
    return getattr(model, "config", None)


def check_decoder_size(decoder, *, width, height, num_frames, scale: int, stage_2: bool) -> None:
    """Raise ``DecoderTooSmallError`` if the decoder cannot take this request.

    ``scale`` is how much the graph enlarges the request before decoding (2 for
    stage 2, 1 for a single-stage diffusion decode).
    """
    check_decoder_minimum(
        _decoder_config(decoder),
        width=width,
        height=height,
        num_frames=num_frames,
        scale=scale,
        stage_2=stage_2,
    )


def check_decoder_minimum(config, *, width, height, num_frames, scale: int, stage_2: bool) -> None:
    """:func:`check_decoder_size` against a decoder config (the loaded one, or its file)."""
    if config is None or not (width and height and num_frames):
        return
    from .constraints import DIMENSION_MULTIPLE
    from .errors import DecoderTooSmallError

    minimum = decoder_minimum(config)
    output = (int(width) * scale, int(height) * scale, int(num_frames))
    if all(got >= need for got, need in zip(output, minimum, strict=True)):
        return

    def request_side(side: int, need: int) -> int:
        per_request = -(-need // scale)
        return max(side, -(-per_request // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)

    smallest = (
        request_side(int(width), minimum[0]),
        request_side(int(height), minimum[1]),
        max(int(num_frames), minimum[2]),
    )
    raise DecoderTooSmallError(output=output, minimum=minimum, smallest=smallest, stage_2=stage_2)


def variation_kwargs(pipeline, frames, level, *, height, width, num_frames) -> dict:
    """A clip variation's first-denoise kwargs: the clip's latents, noised to ``level``.

    The clip is preprocessed exactly as a video condition is (the pipeline's own
    ``preprocess_conditions``), encoded with the video VAE to RAW ``[B, C, F, H, W]``
    latents (the pipeline normalises and packs supplied latents itself), and the
    call denoises only the tail of the distilled schedule from ``level``, with
    ``noise_scale=level`` and the geometry the latents imply. No condition.
    """
    import torch
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import (
        LTX2VideoCondition,
        retrieve_latents,
    )

    from .audio_conditioning import sigma_tail

    device = pipeline._execution_device
    condition = LTX2VideoCondition(frames=frames, index=0, strength=1.0)
    with _no_grad():
        pixels, _strengths, _indices, _counts = pipeline.preprocess_conditions(
            [condition],
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            device=device,
        )
        raw = retrieve_latents(pipeline.vae.encode(pixels[0]), sample_mode="argmax")
    latents = raw.to(device=device, dtype=torch.float32)
    _batch, _channels, latent_frames, latent_height, latent_width = latents.shape
    spatial = int(getattr(pipeline, "vae_spatial_compression_ratio", 32))
    temporal = int(getattr(pipeline, "vae_temporal_compression_ratio", 8))
    return {
        "latents": latents,
        "noise_scale": float(level),
        "sigmas": sigma_tail(level),
        "height": latent_height * spatial,
        "width": latent_width * spatial,
        "num_frames": (latent_frames - 1) * temporal + 1,
    }


def check_cpu_decoder_size(*, width, height, num_frames) -> None:
    """Off CUDA a single-stage diffusion decode runs on the CPU, at very small sizes only."""
    from . import device as device_module
    from .errors import DecoderUnavailableError

    tokens = (
        int(num_frames) * (int(height) // 4) * (int(width) // 4)
        if width and height and num_frames
        else None
    )
    limit = device_module.OFF_CUDA_DECODER_TOKEN_LIMIT
    if tokens is None or tokens > limit:
        raise DecoderUnavailableError(
            reason=(
                f"without an NVIDIA (CUDA) GPU it runs only on the CPU, for very small "
                f"videos (up to {limit} decoder positions; this one needs {tokens})"
            )
        )


def pinned_decoder_config() -> dict | None:
    """The diffusion decoder's config read from its file, or None where it is absent.

    For checks before a run (no model loaded); the loaded decoder is checked again.
    """
    import json

    from . import preflight
    from .errors import LTXStudioError

    try:
        path = preflight.component_dir("diffusion_decoder") / "config.json"
        return json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError, LTXStudioError):
        return None


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

    def __init__(
        self,
        pipeline,
        *,
        kind,
        upsampler=None,
        decoder=None,
        stage_2_factory=None,
        decoder_factory=None,
        device=None,
    ):
        self.pipeline = pipeline
        self.kind = kind
        self.upsampler = upsampler
        self.decoder = decoder
        #: The generation device; off CUDA the diffusion decoder runs on the CPU.
        self.device = device
        #: Built on the first stage-2 request, never at construction. The parts
        #: are ~2.1 GB of weights AND a NATTEN kernel resolved over the network,
        #: and eight of the eleven modes never touch them. Building eagerly
        #: turned a dependency of one branch into a hard requirement for every
        #: mode -- when NATTEN could not resolve, nothing loaded at all.
        self._stage_2_factory = stage_2_factory
        #: The diffusion decoder alone, for a single-stage diffusion decode.
        self._decoder_factory = decoder_factory
        #: Why the decoder could not be built, once that has been found out.
        self._decoder_unavailable: str | None = None
        #: The audio-conditioned derivative of ``pipeline``, built on first use.
        self._audio_pipeline = None
        #: What the last call actually did (builtins only): see :func:`run_facts`.
        self.last_run: dict | None = None

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

    def _decoder_ready(self):
        """Obtain the diffusion decoder alone, building it on first use."""
        if self.decoder is None and self._decoder_unavailable is None:
            try:
                if self._decoder_factory is not None:
                    self.decoder = self._decoder_factory()
                elif self._stage_2_factory is not None:
                    self.upsampler, self.decoder = self._stage_2_factory()
            except ModelLoadError as exc:
                self._decoder_unavailable = str(exc.reason)
                raise
        if self.decoder is None:
            raise ModelLoadError(
                component="diffusion_decoder",
                reason=self._decoder_unavailable or "no diffusion decoder was supplied",
            )

    def _single_stage_decoder(self, policy, kwargs) -> tuple[str, str | None]:
        """``(decoder, why conv)`` for a single-stage run; raises for an unusable explicit choice."""
        from .config import DECODER_CHOICES
        from .errors import ConstraintError, DecoderTooSmallError, DecoderUnavailableError

        if policy not in DECODER_CHOICES:
            raise ConstraintError(f"decoder={policy!r} is not one of {', '.join(DECODER_CHOICES)}.")
        if policy == "conv":
            return "conv", None
        width, height = kwargs.get("width"), kwargs.get("height")
        frames = _frames_lower_bound(kwargs)
        if self.device != "cuda":
            if policy == "auto":
                return "conv", OFF_CUDA_DECODER_NOTE
            check_cpu_decoder_size(width=width, height=height, num_frames=frames)
        try:
            self._decoder_ready()
        except ModelLoadError as exc:
            if policy == "auto":
                return "conv", f"the diffusion decoder could not be loaded here ({exc.reason})"
            raise DecoderUnavailableError(reason=f"it could not be loaded ({exc.reason})") from exc
        try:
            check_decoder_size(
                self.decoder, width=width, height=height, num_frames=frames, scale=1, stage_2=False
            )
        except DecoderTooSmallError as exc:
            if policy == "auto":
                w, h, f = exc.minimum
                return "conv", (
                    f"this video is smaller than the diffusion decoder's minimum of {w}×{h} "
                    f"pixels and {f} frames"
                )
            raise
        return "diffusion", None

    def __call__(self, *, enable_stage_2: bool = False, progress=None, **kwargs):
        """Run the graph. ``progress`` is consumed here and never reaches diffusers.

        **This is the only layer that knows how many sampling passes a run has**,
        which is why the progress reporting is wired here rather than in
        ``generate.py``. Stage 2 is a second denoise over the same schedule
        machinery: its bar genuinely starts again at zero, and a bar that
        silently restarts reads as a crash. So each pass is announced as "pass N
        of M" before it begins, and the two non-sampling stage-2 steps -- the
        latent upsample and the diffusion decode -- get their own labels rather
        than leaving the user on a finished stage-1 bar for the length of both.

        ``progress`` is a ``generate.ProgressReporter``; it is duck-typed here so
        this module keeps importing nothing from the orchestration layer above
        it, and it is optional so every direct caller (the offline gate, the
        ``scripts/`` harnesses) is unaffected.
        """
        kwargs = dict(_with_real_conditions(kwargs))
        # Ours, like enable_stage_2: the audio stream's own noise level, and
        # which decoder finishes the frames.
        audio_noise_level = kwargs.pop("audio_noise_level", None)
        policy = kwargs.pop("video_decoder", None) or "auto"
        variation = kwargs.pop("video_variation", None)
        pipeline, audio_kwargs = self._pipeline_for(audio_noise_level)
        kwargs.update(audio_kwargs)
        self.last_run = None
        # A clip variation replaces the first denoise's start: the clip's latents
        # at its level and the schedule's tail, merged last so they win.
        first = {}
        if variation is not None:
            from . import audio_conditioning

            if audio_conditioning.is_condition_pipeline(self.pipeline):
                first = self._variation(variation.source, variation.level, kwargs)
            else:
                first = {"video_variation": variation}  # a stand-in takes it as given
        if not enable_stage_2:
            decoder, why_conv = self._single_stage_decoder(policy, kwargs)
            self.last_run = run_facts(
                stage_2=False,
                decoder=decoder,
                why_conv=why_conv,
                policy=policy,
                audio_noise_level=audio_noise_level,
            )
            if decoder == "conv":
                # The distilled recipe applies here too. It used to apply ONLY
                # on the stage-2 path, which is how eleven of twelve host runs
                # were generated with a linear schedule and SFT guidance.
                return pipeline(
                    **_merged(kwargs, distilled_kwargs(), _sampling_kwargs(progress), first)
                )
            single = pipeline(
                **_merged(kwargs, stage_1_kwargs(), _sampling_kwargs(progress), first)
            )
            return self._diffusion_decoded(single, kwargs, progress)
        from . import device as device_module

        device_module.check_stage_2_size(
            self.device, kwargs.get("width"), kwargs.get("height"), _frames_upper_bound(kwargs)
        )
        self._stage_2_ready()
        check_decoder_size(
            self.decoder,
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            num_frames=_frames_lower_bound(kwargs),
            scale=STAGE_2_SPATIAL_SCALE,
            stage_2=True,
        )

        stage_1 = pipeline(
            **_merged(
                kwargs, stage_1_kwargs(), _sampling_kwargs(progress, index=1, passes=2), first
            )
        )
        if progress is not None:
            progress.upscaling()
        upsampled = _latents_of(self.upsampler(**upsample_kwargs(_latents_of(stage_1))))
        # Stage 2's own values win: the audio-input modes already carry
        # noise_scale and audio_latents in the request.
        stage_2 = pipeline(
            **_merged(
                {k: v for k, v in kwargs.items() if k not in STAGE_2_STRIPPED_KEYS},
                stage_2_geometry(
                    upsampled,
                    kind=self.kind,
                    spatial_ratio=self._ratio("spatial"),
                    temporal_ratio=self._ratio("temporal"),
                ),
                stage_2_kwargs(upsampled, audio_latents=_audio_of(stage_1)),
                _sampling_kwargs(progress, index=2, passes=2),
            )
        )
        self.last_run = run_facts(
            stage_2=True,
            decoder="diffusion",
            why_conv=None,
            policy=policy,
            audio_noise_level=audio_noise_level,
        )
        return self._diffusion_decoded(stage_2, kwargs, progress)

    def _variation(self, frames, level, kwargs) -> dict:
        """The clip's first-denoise kwargs (see :func:`variation_kwargs`)."""
        return variation_kwargs(
            self.pipeline,
            frames,
            level,
            height=kwargs.get("height"),
            width=kwargs.get("width"),
            num_frames=kwargs.get("num_frames"),
        )

    def _diffusion_decoded(self, staged, kwargs, progress):
        """Decode a latent-output call with the diffusion decoder and reunite its audio."""
        if progress is not None:
            progress.decoding()
        latents, generator = _latents_of(staged), kwargs.get("generator")
        if self.device == "mps":
            latents, generator = _for_the_cpu_decoder(latents, generator)
        decoded = self.decoder(**decode_kwargs(latents, generator=generator))
        return self._with_audio(decoded, staged)

    def _pipeline_for(self, audio_noise_level):
        """The pipeline for this call, and any kwargs it must also be given.

        With an audio level, a real condition pipeline is swapped for its
        audio-conditioned derivative (shared modules); a stand-in in the
        offline gate is handed the level instead.
        """
        if audio_noise_level is None:
            return self.pipeline, {}
        from . import audio_conditioning

        if not audio_conditioning.is_condition_pipeline(self.pipeline):
            return self.pipeline, {"audio_noise_level": audio_noise_level}
        self._audio_pipeline = audio_conditioning.audio_pipeline(
            self.pipeline, audio_noise_level, derived=self._audio_pipeline
        )
        return self._audio_pipeline, {}

    def _ratio(self, axis: str) -> int:
        """The VAE's compression ratio on ``axis``, read off the pipeline."""
        default = (
            DEFAULT_SPATIAL_COMPRESSION_RATIO
            if axis == "spatial"
            else DEFAULT_TEMPORAL_COMPRESSION_RATIO
        )
        value = getattr(self.pipeline, f"vae_{axis}_compression_ratio", None)
        return int(value) if isinstance(value, int) else default

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
        with _no_grad():
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


#: Why ``auto`` decodes with the conv VAE off CUDA.
OFF_CUDA_DECODER_NOTE = "the model's diffusion decoder runs only on an NVIDIA (CUDA) GPU here"


def run_facts(*, stage_2, decoder, why_conv, policy, audio_noise_level) -> dict:
    """What one runner call did, as builtins (it crosses the ZeroGPU fork)."""
    return {
        "stage_2": bool(stage_2),
        "decoder": decoder,
        "decoder_policy": policy,
        "why_conv": why_conv,
        "audio_noise_level": None if audio_noise_level is None else float(audio_noise_level),
    }


def decoder_note(facts: dict) -> str:
    """The run note naming the decoder that finished the frames."""
    if facts.get("decoder") == "diffusion":
        return "the frames were decoded by the model's diffusion decoder."
    why = facts.get("why_conv")
    if why:
        return f"the frames were decoded by the standard (convolutional) decoder: {why}."
    return "the frames were decoded by the standard (convolutional) decoder, as chosen."


def _frames_upper_bound(kwargs) -> int:
    """``num_frames``, or the most the duration head may choose when it is unset."""
    if kwargs.get("num_frames"):
        return int(kwargs["num_frames"])
    seconds = float(kwargs.get("max_seconds") or 0)
    rate = float(kwargs.get("frame_rate") or DEFAULT_FRAME_RATE)
    return int(seconds * rate) + 1


def _frames_lower_bound(kwargs) -> int | None:
    """``num_frames``, or the fewest the duration head may choose when it is unset."""
    if kwargs.get("num_frames"):
        return int(kwargs["num_frames"])
    if kwargs.get("min_seconds") is None or kwargs.get("max_seconds") is None:
        return None
    from .constraints import duration_head_frame_range

    shortest, _longest = duration_head_frame_range(
        min_seconds=float(kwargs["min_seconds"]),
        max_seconds=float(kwargs["max_seconds"]),
        frame_rate=float(kwargs.get("frame_rate") or DEFAULT_FRAME_RATE),
    )
    return shortest


def _merged(*mappings) -> dict:
    """One call's kwargs from several dicts; a later dict wins on a shared key."""
    merged: dict = {}
    for mapping in mappings:
        merged.update(mapping)
    return merged


def _no_grad():
    """``torch.no_grad()``, or a no-op where torch is absent (the offline gate)."""
    try:
        import torch
    except ImportError:
        import contextlib

        return contextlib.nullcontext()
    return torch.no_grad()


def _for_the_cpu_decoder(latents, generator):
    """MPS latents and generator, made usable by the decoder on the CPU.

    The CPU generator is seeded from the run's own seed, so a decode stays
    reproducible: an MPS generator cannot seed a CPU ``randn``.
    """
    import torch

    if getattr(generator, "device", None) is not None and generator.device.type != "cpu":
        generator = torch.Generator(device="cpu").manual_seed(generator.initial_seed())
    return latents.to("cpu"), generator


def _sampling_kwargs(progress, *, index: int = 1, passes: int = 1) -> dict:
    """``callback_on_step_end`` for one denoise pass, or nothing at all.

    A dict rather than a value so the kwarg is **absent** when there is nobody
    to report to, rather than present and ``None``. Upstream defaults it to
    ``None`` and would behave identically either way, but the house rule holds:
    a ``None`` in the kwargs is how an unset value becomes a confusing TypeError
    several layers away, and a fake pipeline in the gate should be handed exactly
    what it was handed before this existed.

    ✅ The kwarg is real, not hopeful: ``pipeline_ltx2_condition.py`` and
    ``pipeline_ltx2.py`` at the pinned ``ac56fa2f`` both accept
    ``callback_on_step_end`` and call it as ``callback_on_step_end(self, i, t,
    callback_kwargs)`` inside the denoise loop, popping ``latents`` off the dict
    it returns 📄.
    """
    if progress is None:
        return {}
    callback = progress.sampling(index=index, passes=passes)
    return {"callback_on_step_end": callback} if callback is not None else {}


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


def _snapshot_bytes() -> int:
    """Bytes of the snapshot weights load from, following its symlinks."""
    from . import preflight

    override = preflight.snapshot_override()
    if override is None:
        return _directory_bytes(preflight.cache_repo_dir())
    total = 0
    for root, _dirs, files in os.walk(override, followlinks=True):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def disk_footprint(*, phase: str = "at startup, before launch") -> str:
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

    Both readings are taken before ``demo.launch()``, which is when ZeroGPU
    writes the packed copy, and are labelled so: at startup, and after the base
    has moved to its device. An empty packed directory is reported as "not
    written yet", never as a measured ``0.0``.
    """
    snapshot = _snapshot_bytes() / 1e9
    packed_dir = os.environ.get("ZEROGPU_OFFLOAD_DIR") or os.path.expanduser("~/.zerogpu/tensors")
    packed_bytes = _directory_bytes(packed_dir)
    packed = packed_bytes / 1e9
    packed_text = (
        f"packed {packed:.1f}"
        if packed_bytes
        else "packed copy not written yet (ZeroGPU writes it at launch)"
    )
    measured = snapshot + packed
    remaining = ZEROGPU_DISK_CAP_GB - measured
    verdict = "OVER" if measured + IMAGE_FOOTPRINT_GB > ZEROGPU_DISK_CAP_GB else "OK"
    return (
        f"[{phase}] {measured:.1f} GB measured of {ZEROGPU_DISK_CAP_GB:.0f} GB cap "
        f"[{verdict}] = snapshot {snapshot:.1f} + {packed_text}; "
        f"{remaining:.1f} GB left for the image (~{IMAGE_FOOTPRINT_GB:.0f} GB, "
        f"NOT measurable from inside) and anything else. "
        f"**Measured is a FLOOR, not the total.**"
    )


def _dtype_label(dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    return {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}.get(name, name)


#: An override snapshot this small (the tiny test checkpoint) takes the resident
#: build; anything larger is real weights and goes through the local runner, with
#: its staging and memory preflights (review local-runtime-4).
RESIDENT_SNAPSHOT_MAX_BYTES = 1_000_000_000


def _uses_local_runner() -> bool:
    """Real weights on a local device go to the local runner; only a small override does not."""
    from .preflight import snapshot_override

    override = snapshot_override()
    return override is None or _weight_bytes(override) > RESIDENT_SNAPSHOT_MAX_BYTES


def _weight_bytes(path) -> int:
    """Bytes under ``path``, following links (an HF snapshot is links into its blobs)."""
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=True):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def runner_summary(resolution, runtime_config) -> str:
    """Which runner generation will use here, and where the transformer comes from."""
    if resolution is None:
        return "unknown (no device resolved)"
    if resolution.device is None:
        return "none (UI only)"
    if resolution.device == "cuda":
        return "StageRunner, the pipeline built at startup and resident on CUDA"
    if not _uses_local_runner():
        return (
            f"StageRunner, resident on {resolution.device}; every component from "
            f"{runtime_config.model_snapshot_dir} (LTX25_MODEL_SNAPSHOT_DIR)"
        )
    source = (
        f"the GGUF file {runtime_config.transformer_gguf} (LTX25_TRANSFORMER_GGUF)"
        if runtime_config.transformer_gguf
        else "the pinned snapshot's safetensors"
    )
    return (
        f"local runner, {runtime_config.residency} residency (LTX25_RESIDENCY), on "
        f"{resolution.device}; transformer from {source}"
    )


def precision_summary(resolution) -> str:
    """What each component is loaded as, from the resolved configuration."""
    if resolution is None:
        return "unknown (no device resolved)"
    base = _dtype_label(resolution.weight_dtype)
    if resolution.device is None:
        return f"not loaded (UI only); a load here would be {base}"
    if resolution.quantization == "int8":
        return (
            f"int8 for {', '.join(QUANTIZED_COMPONENTS)} (bitsandbytes LLM.int8; the "
            f"transformer keeps its conditioning, projection and gate Linears bf16 via "
            f"{len(TRANSFORMER_INT8_SKIP_MODULES)} skip patterns); {base} for every other "
            f"component"
        )
    return f"{base} for every component on {resolution.device}"


def startup_report(
    runtime_config,
    *,
    cached_files=None,
    missing_files=None,
    resolution=None,
    spaces_version=None,
    mps_self_test=None,
) -> str:
    """The lines we will be reading at a distance when something goes wrong.

    Deliberately verbose. When R1 fails on a metered Space behind a 72 GB build,
    these lines are the difference between diagnosing it in one round trip and
    burning several -- and there is no opportunity to add a print statement
    afterwards.

    Reports the resolved device and precision (so the running configuration is
    never inferred), the snapshot on disk and the measured disk floor, **how many
    of the expected files were found in cache**, the ZeroGPU size requested and
    the reservation, and the installed ``spaces``. The packed copy and the
    granted GPU slice cannot be known here: the platform packs at launch and
    the worker measures the slice on each run, and the report says so rather
    than printing a number nobody measured.
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

    from . import preflight
    from .zerogpu import ENHANCE_DURATION_SECONDS, LORA_DURATION_SECONDS

    override = preflight.snapshot_override()
    if override is None:
        expected_files = len(model_files.download_allow_list())
        unit, source = (
            "files",
            f"pinned {model_files.REPO_ID} @ {', '.join(r[:8] for r in model_files.revisions())}",
        )
    else:
        expected_files = len(preflight._declared_components(override))
        unit, source = "declared components", f"{override} (LTX25_MODEL_SNAPSHOT_DIR)"
    found = expected_files - len(missing_files) if cached_files is None else cached_files
    try:
        disk = disk_footprint()
        snapshot = f"{_snapshot_bytes() / 1e9:.2f} GB measured"
    except OSError:  # pragma: no cover - unusual filesystems
        disk = snapshot = "unavailable"

    if runtime_config.zerogpu_size == "auto":
        size = "not requested (auto): the platform default, large, applies"
    else:
        size = f"requested {runtime_config.zerogpu_size}"
    reservation = (
        f"reservation {runtime_config.gpu_duration_seconds} s per run (x1.5 with stage 2, "
        f"+{LORA_DURATION_SECONDS} s with LoRAs, +{ENHANCE_DURATION_SECONDS} s with prompt "
        f"enhancement); the granted slice is measured by the worker on each run"
    )

    lines = [
        "LTX-2.5 Studio starting",
        f"  config          : {runtime_config.describe()}",
        f"  device          : {resolution.describe() if resolution is not None else 'unknown'}",
        f"  precision       : {precision_summary(resolution)}",
        f"  runner          : {runner_summary(resolution, runtime_config)}",
        f"  weights cached  : {found}/{expected_files} {unit}"
        + (f"  MISSING {len(missing_files)}" if missing_files else ""),
        f"  snapshot on disk: {snapshot} ({source}; download set "
        f"{runtime_config.download_gb:.2f} GB)",
        f"  disk            : {disk}",
        "  packed copy     : not measured yet, ZeroGPU writes it at launch after this report",
        f"  zerogpu size    : {size}; {reservation}",
        f"  spaces          : {spaces_version or 'not installed'}",
    ]
    if mps_self_test is not None:
        lines.append(f"  mps self-test   : {mps_self_test.describe()}")
    lines.append(f"  model revision  : {model_files.REVISION}")
    lines.append(f"  decoder revision: {model_files.get_component('diffusion_decoder').revision}")
    if missing_files:
        lines.append(f"  first missing   : {missing_files[0]}")
    return "\n".join(lines)


def clear_cache() -> None:
    """Drop cached pipelines. Used by tests and by the local smoke tier."""
    _CACHE.clear()
