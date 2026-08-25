"""Spec for the VAE decode-tiling envelope — pipelines.configure_vae_tiling.

The live failure this exists for: 1216x704x121 completed its ~52 s denoise and
then died inside ``vae.decode``'s conv3d on the MIG allocator — full-resolution
decode intermediates are ~26 GB per tensor at 121 frames, on ~35 GB of
resident weights. 704x480x49 fits; 1216x704x121 does not.

The design under test: hand ``AutoencoderKLLTX2.enable_tiling`` thresholds
equal to the largest PROVEN-fit decode, so that upstream's own dispatch is the
size gate. Its ``_decode`` compares the actual latent tensor against
``tile_sample_min_* // compression_ratio`` with strict ``>`` (read at the
pinned ``ac56fa2f``; byte-identical in the local 0.37.1 copy), which gives the
two properties these tests pin:

* everything at or inside the proven envelope — the sha256-banked 256x256x9
  above all — takes the SAME untiled code path as before the change;
* everything beyond it decodes in tiles of exactly the proven-fit shape, the
  temporal chunks re-dispatching through spatial tiling.

What the offline gate can and cannot say, plainly: these tests verify the
plumbing, the threshold arithmetic, and that the dispatch predicate — as
replicated verbatim from the pinned source — routes the banked configs down
the untouched path. Byte-identity itself is a CUDA fact and is confirmed by
the sha256 re-check on the Space re-run, not here.
"""

from __future__ import annotations

import inspect as _inspect

import pytest

from ltx25_studio import pipelines
from ltx25_studio.constraints import (
    DEFAULT_SPATIAL_COMPRESSION_RATIO,
    DEFAULT_TEMPORAL_COMPRESSION_RATIO,
)
from ltx25_studio.errors import ModelLoadError


class FakeTilingVae:
    spatial_compression_ratio = DEFAULT_SPATIAL_COMPRESSION_RATIO
    temporal_compression_ratio = DEFAULT_TEMPORAL_COMPRESSION_RATIO

    def __init__(self):
        self.tiling_kwargs = None
        self.use_framewise_decoding = False
        self.use_framewise_encoding = False

    def enable_tiling(self, **kwargs):
        self.tiling_kwargs = kwargs


def _configured():
    vae = FakeTilingVae()
    pipelines.configure_vae_tiling(vae)
    return vae


# --------------------------------------------------------------------------
# the plumbing: what is set, exactly
# --------------------------------------------------------------------------


def test_the_envelope_is_handed_to_enable_tiling_verbatim():
    vae = _configured()
    assert vae.tiling_kwargs == {
        "tile_sample_min_height": pipelines.VAE_TILE_MIN_HEIGHT,
        "tile_sample_min_width": pipelines.VAE_TILE_MIN_WIDTH,
        "tile_sample_min_num_frames": pipelines.VAE_TILE_MIN_NUM_FRAMES,
        "tile_sample_stride_height": pipelines.VAE_TILE_STRIDE_HEIGHT,
        "tile_sample_stride_width": pipelines.VAE_TILE_STRIDE_WIDTH,
        "tile_sample_stride_num_frames": pipelines.VAE_TILE_STRIDE_NUM_FRAMES,
    }


def test_both_framewise_flags_are_set():
    """enable_tiling alone never chunks time — use_framewise_decoding is a
    separate flag it does not touch (read at the pinned SHA). The encoding
    flag rides along because upstream's _encode currently checks the DECODING
    flag (a quirk, identical at the pin): setting both is correct today and
    stays correct the day the quirk is fixed."""
    vae = _configured()
    assert vae.use_framewise_decoding is True
    assert vae.use_framewise_encoding is True


def test_off_grid_thresholds_are_refused_naming_the_drift():
    """The dispatch floor-divides these by the compression ratios; a value
    that does not divide exactly SHIFTS the proven envelope silently. A VAE
    whose ratios changed upstream must therefore refuse to load rather than
    tile at a boundary nobody measured."""

    class WeirdVae(FakeTilingVae):
        spatial_compression_ratio = 33

    with pytest.raises(ModelLoadError) as excinfo:
        pipelines.configure_vae_tiling(WeirdVae())
    assert "envelope" in str(excinfo.value)
    assert "VAE_TILE" in str(excinfo.value)


def test_base_pipeline_wires_the_configuration_in():
    source = _inspect.getsource(pipelines._base_pipeline)
    assert "configure_vae_tiling(" in source, (
        "the tiling envelope must be applied to the ONE shared VAE at load — "
        "an unwired configure function is the §8 inert-guard shape"
    )


# --------------------------------------------------------------------------
# the envelope arithmetic — upstream's dispatch, replicated verbatim
# --------------------------------------------------------------------------


def _dispatch(width, height, num_frames):
    """The pinned ``AutoencoderKLLTX2._decode`` routing, verbatim in latent
    units: framewise when latent frames exceed min_frames // temporal_ratio,
    tiled when either latent axis exceeds min // spatial_ratio, else plain."""
    spatial = DEFAULT_SPATIAL_COMPRESSION_RATIO
    temporal = DEFAULT_TEMPORAL_COMPRESSION_RATIO
    latent_w, latent_h = width // spatial, height // spatial
    latent_f = (num_frames - 1) // temporal + 1

    framewise = latent_f > pipelines.VAE_TILE_MIN_NUM_FRAMES // temporal
    tiled = (
        latent_w > pipelines.VAE_TILE_MIN_WIDTH // spatial
        or latent_h > pipelines.VAE_TILE_MIN_HEIGHT // spatial
    )
    return framewise, tiled


def test_the_sha256_banked_config_takes_the_untouched_path():
    """256x256x9 was sha256-proven byte-identical across three containers
    TODAY. Both comparisons are strict >, so this config decodes through the
    same plain path as before this change — structurally, not by luck."""
    assert _dispatch(256, 256, 9) == (False, False)


def test_the_largest_proven_fit_decode_is_exactly_the_last_untouched_point():
    """704x480x49 decoded clean on the slice; 49 frames occupy (49-1)//8+1 = 7
    latent frames and the threshold is 56//8 = 7, so 7 > 7 is False — the
    proven point is the boundary, and nothing that ever worked changes."""
    assert _dispatch(704, 480, 49) == (False, False)
    assert _dispatch(704, 480, 1) == (False, False)


def test_one_step_past_the_proven_envelope_engages_the_right_mechanism():
    assert _dispatch(704, 480, 57) == (True, False), "57 frames -> 8 latent -> framewise"
    assert _dispatch(736, 480, 49) == (False, True), "one tile-width past -> spatial"
    assert _dispatch(704, 512, 49) == (False, True), "one tile-height past -> spatial"


def test_the_live_failure_config_engages_both_mechanisms():
    """1216x704x121 — the run that died. Temporal chunks re-dispatch through
    spatial tiling upstream, so the peak decode tensor is bounded by a
    proven-fit shape."""
    assert _dispatch(1216, 704, 121) == (True, True)


def test_the_stage_2_decoded_size_is_gated_by_what_is_decoded_not_requested():
    """Stage 2 hands the VAE 2x the request (1408x960 from 704x480). The gate
    lives inside the dispatch and keys on the tensor itself, so the doubled
    size is what gets compared — no branch of ours reads the request."""
    assert _dispatch(1408, 960, 49) == (False, True)
    assert _dispatch(1408, 960, 121) == (True, True)


# --------------------------------------------------------------------------
# the envelope constants themselves
# --------------------------------------------------------------------------


def test_the_thresholds_sit_exactly_on_the_latent_grid():
    spatial = DEFAULT_SPATIAL_COMPRESSION_RATIO
    temporal = DEFAULT_TEMPORAL_COMPRESSION_RATIO
    assert pipelines.VAE_TILE_MIN_HEIGHT % spatial == 0
    assert pipelines.VAE_TILE_MIN_WIDTH % spatial == 0
    assert pipelines.VAE_TILE_STRIDE_HEIGHT % spatial == 0
    assert pipelines.VAE_TILE_STRIDE_WIDTH % spatial == 0
    assert pipelines.VAE_TILE_MIN_NUM_FRAMES % temporal == 0
    assert pipelines.VAE_TILE_STRIDE_NUM_FRAMES % temporal == 0


def test_the_strides_preserve_upstreams_own_blend_widths():
    """The overlap (min - stride) is the blend region. Upstream ships 64 px
    spatial (512-448) and 8 frames temporal (16-8); keeping those means the
    seam behaviour for oversized decodes is exactly the shipped one, only at
    our tile size."""
    assert pipelines.VAE_TILE_MIN_WIDTH - pipelines.VAE_TILE_STRIDE_WIDTH == 64
    assert pipelines.VAE_TILE_MIN_HEIGHT - pipelines.VAE_TILE_STRIDE_HEIGHT == 64
    assert pipelines.VAE_TILE_MIN_NUM_FRAMES - pipelines.VAE_TILE_STRIDE_NUM_FRAMES == 8


def test_strides_stay_strictly_inside_the_tile():
    """A stride >= its tile min means zero (or negative) overlap and a
    degenerate blend; upstream does not guard this, so we do."""
    assert 0 < pipelines.VAE_TILE_STRIDE_WIDTH < pipelines.VAE_TILE_MIN_WIDTH
    assert 0 < pipelines.VAE_TILE_STRIDE_HEIGHT < pipelines.VAE_TILE_MIN_HEIGHT
    assert 0 < pipelines.VAE_TILE_STRIDE_NUM_FRAMES < pipelines.VAE_TILE_MIN_NUM_FRAMES
