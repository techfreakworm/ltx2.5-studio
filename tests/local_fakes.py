"""Helpers for the local-runtime tests: the tiny checkpoints, never a download.

Everything here works on the tiny random-weight checkpoints already in the HF
cache. A helper that cannot find one raises ``pytest.skip`` with the reason,
so a machine without them skips rather than fetching.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: Tiny diffusers-layout LTX-2.5 (random weights, ~95 MB), pinned.
TINY_REPO = "optimum-intel-internal-testing/tiny-random-ltx2.5"
TINY_REVISION = "3f43040b4719f6dad170a6d07b30921ee1a8f405"


def tiny_snapshot() -> Path:
    """The tiny snapshot directory, or skip. Never downloads."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(TINY_REPO, revision=TINY_REVISION, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any failure means "not cached here"
        pytest.skip(
            f"tiny checkpoint {TINY_REPO}@{TINY_REVISION[:8]} is not in the HF cache ({exc})"
        )
    return Path(path)


def install_tiny_audio_encode_shim(audio_vae) -> None:
    """Tiny-checkpoint artifact, not an app fix.

    The tiny audio VAE has ``ch_mult=[1]`` (the real one ``[1, 2, 4]``), so its
    encoder never downsamples while the class hard-codes a compression of 4 on
    both axes; the pipeline then rejects the latents. Pooling the posterior 4x4
    gives the shape the real VAE produces, so the rest of the audio route runs.
    """
    import torch.nn.functional as F
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_outputs import AutoencoderKLOutput

    if getattr(audio_vae, "_tiny_shim", False):
        return
    original = audio_vae.encode

    def encode(x, return_dict=True):
        params = original(x, return_dict=True).latent_dist.parameters
        pooled = F.avg_pool2d(params.float(), kernel_size=4, ceil_mode=True).to(params.dtype)
        dist = DiagonalGaussianDistribution(pooled)
        return AutoencoderKLOutput(latent_dist=dist) if return_dict else (dist,)

    audio_vae.encode = encode
    audio_vae._tiny_shim = True


class RecordingEncoder:
    """Keeps the raw pipeline output, then writes the file with the real writer."""

    def __init__(self):
        self.outputs = []

    def __call__(self, output, **kwargs):
        from ltx25_studio import outputs

        self.outputs.append(output)
        return outputs.write_output(output, **kwargs)


def as_array(value):
    """A pipeline field (tensor, numpy, or list of PIL frames) as a float64 array."""
    import numpy as np

    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy().astype(np.float64)
    if isinstance(value, (list, tuple)):
        return np.stack([as_array(item) for item in value])
    return np.asarray(value, dtype=np.float64)


def transformer_inputs(config, device, dtype):
    """Fixed inputs for one transformer forward: 3x4x4 video tokens, 18 audio, 16 text."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(0)
    frames, height, width, audio, text = 3, 4, 4, 18, 16
    tensors = {
        "hidden_states": torch.randn(
            1, frames * height * width, config.in_channels, generator=generator
        ),
        "audio_hidden_states": torch.randn(1, audio, config.audio_in_channels, generator=generator),
        "encoder_hidden_states": torch.randn(
            1, text, config.cross_attention_dim, generator=generator
        ),
        "audio_encoder_hidden_states": torch.randn(
            1, text, config.audio_cross_attention_dim, generator=generator
        ),
    }
    inputs = {name: value.to(device, dtype) for name, value in tensors.items()}
    timestep = torch.tensor([600.0], device=device)
    mask = torch.ones(1, text, device=device)
    inputs.update(
        timestep=timestep,
        sigma=timestep / 1000.0,
        encoder_attention_mask=mask,
        audio_encoder_attention_mask=mask,
        num_frames=frames,
        height=height,
        width=width,
        fps=24.0,
        audio_num_frames=audio,
        use_cross_timestep=True,
        return_dict=False,
    )
    return inputs


# --------------------------------------------------------------------------
# a tiny transformer file with the REAL single-file key names
# --------------------------------------------------------------------------

#: diffusers -> original (ComfyUI / Lightricks single-file) names: the inverse of
#: diffusers' ``convert_ltx2_transformer_to_diffusers`` plus the two renames it
#: lacks. The probe validated this map against the real LTX-2.5 distilled
#: header: 4,349 names reproduced, 0 missing, 0 extra, 0 shape mismatches.
_TO_ORIGINAL = (
    ("audio_prompt_adaln.", "audio_prompt_adaln_single."),
    ("prompt_adaln.", "prompt_adaln_single."),
    ("audio_time_embed.", "audio_adaln_single."),
    ("time_embed.", "adaln_single."),
    ("audio_proj_in.", "audio_patchify_proj."),
    ("proj_in.", "patchify_proj."),
    ("av_cross_attn_video_scale_shift.", "av_ca_video_scale_shift_adaln_single."),
    ("av_cross_attn_video_a2v_gate.", "av_ca_a2v_gate_adaln_single."),
    ("av_cross_attn_audio_scale_shift.", "av_ca_audio_scale_shift_adaln_single."),
    ("av_cross_attn_audio_v2a_gate.", "av_ca_v2a_gate_adaln_single."),
    ("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video"),
    ("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio"),
    (".norm_q.", ".q_norm."),
    (".norm_k.", ".k_norm."),
)


def to_original_name(key: str) -> str:
    """A diffusers transformer key under its original single-file name (no prefix)."""
    out = key
    for diffusers_name, original in _TO_ORIGINAL:
        if diffusers_name.endswith(".") and not diffusers_name.startswith("."):
            if out.startswith(diffusers_name):
                out = original + out[len(diffusers_name) :]
        else:
            out = out.replace(diffusers_name, original)
    return out


def write_faithful_gguf(snapshot: Path, target: Path) -> dict:
    """The tiny transformer as a Q8_0 GGUF under the real key names.

    2-D Linear weights whose rows divide into Q8_0 blocks are quantised; the
    rest stay F32, as in the published files' non-Linear tensors. Returns the
    tiny model's own state dict (bf16) for comparison.
    """
    gguf = pytest.importorskip("gguf", reason="the GGUF route needs the `gguf` package")
    import torch
    from diffusers import LTX2VideoTransformer3DModel

    model = LTX2VideoTransformer3DModel.from_pretrained(
        snapshot, subfolder="transformer", torch_dtype=torch.float32, local_files_only=True
    )
    state = model.state_dict()
    writer = gguf.GGUFWriter(str(target), arch="ltxv")
    for key, tensor in state.items():
        array = tensor.float().numpy()
        name = to_original_name(key)
        if key.endswith(".weight") and array.ndim == 2 and array.shape[-1] % 32 == 0:
            writer.add_tensor(
                name,
                gguf.quants.quantize(array, gguf.GGMLQuantizationType.Q8_0),
                raw_dtype=gguf.GGMLQuantizationType.Q8_0,
            )
        else:
            writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {key: value.to(torch.bfloat16) for key, value in state.items()}
