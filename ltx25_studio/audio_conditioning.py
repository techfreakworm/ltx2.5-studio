"""Audio conditioning that agrees with the sampler.

Upstream's condition pipeline applies its single ``noise_scale`` to the user's
audio latents AND to the video's initial noise, while the schedule still starts
at sigma 1.0: the denoiser is told "pure noise" and handed half-variance
latents. Here the audio stream gets its own noise level and its own sigma
schedule, and the video is sampled exactly as in every other mode
(``noise_scale`` unset, so it starts at the schedule's first sigma).

* **frozen** (level 0, ``audio_to_video``): the audio latents are the encoded
  upload, unchanged, and the transformer is told at every step that they are
  clean. The picture is generated to the sound.
* **variation at level s** (``audio_to_audio``): the upload is noised to ``s``
  and the audio sigma at each step is ``min(video sigma, s)``: it holds at ``s``
  (zero-length steps) until the video schedule comes down to ``s``, then
  follows the same distilled tail. ``s = 1.0`` is the plain schedule, which
  ignores the upload.

Imports nothing heavy at module level; the diffusers subclasses are built on
first use.
"""

from __future__ import annotations

from .constraints import DISTILLED_SIGMA_VALUES

__all__ = [
    "DEFAULT_AUDIO_LEVEL",
    "DEFAULT_VIDEO_LEVEL",
    "FROZEN",
    "VARIATION_LEVELS",
    "VIDEO_VARIATION_LEVELS",
    "audio_pipeline",
    "audio_sigmas",
    "is_condition_pipeline",
    "level_note",
    "sigma_tail",
    "snap_level",
    "snap_video_level",
    "snapped_level_note",
    "video_level_note",
]

#: The audio is held exactly as supplied.
FROZEN = 0.0

#: The levels a variation may start from: the model's own distilled sigmas below
#: 1.0, and 1.0 itself (fresh audio).
VARIATION_LEVELS = (*sorted(s for s in DISTILLED_SIGMA_VALUES if s < 0.95), 1.0)

#: A clip variation's levels (``video_to_video``): 1.0 would be plain text to video.
VIDEO_VARIATION_LEVELS = VARIATION_LEVELS[:-1]
#: The defaults when the request names no level. A clip is changed a lot by
#: default: the only level at which the prompt visibly acts (measured, F66).
DEFAULT_AUDIO_LEVEL = VARIATION_LEVELS[0]
DEFAULT_VIDEO_LEVEL = VIDEO_VARIATION_LEVELS[-1]

_VIDEO_WORDS = dict(zip(VIDEO_VARIATION_LEVELS, ("a little", "moderately", "a lot"), strict=True))

_LEVEL_WORDS = {
    VARIATION_LEVELS[0]: "a little",
    VARIATION_LEVELS[1]: "moderately",
    VARIATION_LEVELS[2]: "a lot",
}


def snap_level(noise_scale) -> float:
    """The supported variation level nearest to ``noise_scale`` (ties go higher)."""
    value = float(noise_scale)
    return min(VARIATION_LEVELS, key=lambda level: (abs(level - value), -level))


def snap_video_level(noise_scale) -> float:
    """The supported clip-variation level nearest to ``noise_scale`` (ties go higher)."""
    value = float(noise_scale)
    return min(VIDEO_VARIATION_LEVELS, key=lambda level: (abs(level - value), -level))


def sigma_tail(level: float) -> list[float]:
    """The distilled schedule from ``level`` on: what a variation at ``level`` denoises."""
    schedule = list(DISTILLED_SIGMA_VALUES)
    return schedule[schedule.index(float(level)) :]


def video_level_note(level: float) -> str:
    """What happened to the user's clip, in the UI's words: the level that was used."""
    return f"your clip was changed {_VIDEO_WORDS[float(level)]}."


def snapped_level_note(level: float, requested) -> str | None:
    """The request, said only when it was snapped to a different level.

    A request that reads as the level used (0.42 for 0.421875) is not a snap a
    person would recognise, so nothing is said for it.
    """
    if requested is None or _two(requested) == _two(level):
        return None
    return (
        f"you asked for {_two(requested)}, which is not one of the levels; the nearest, "
        f"{_two(level)}, was used."
    )


def audio_sigmas(video_sigmas, level: float) -> list[float]:
    """The audio's sigma at each step: the video's, held at ``level`` until it falls below."""
    return [min(float(sigma), float(level)) for sigma in video_sigmas]


def _two(value) -> str:
    """Two decimals, half up (0.725 -> 0.73), as a person would round."""
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def level_note(level: float) -> str:
    """What happened to the user's audio, in the UI's words: the level that was used."""
    level = float(level)
    if level == FROZEN:
        return "the sound in this file is your audio, unchanged."
    if level >= 1.0:
        return "your audio was not used: this sound was made fresh."
    return f"your audio was changed {_LEVEL_WORDS[level]}."


# ---------------------------------------------------------------------------
# the diffusers subclasses, built on first use
# ---------------------------------------------------------------------------

_CLASSES: dict[str, type] = {}


def _classes() -> tuple[type, type]:
    if _CLASSES:
        return _CLASSES["pipeline"], _CLASSES["scheduler"]
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ConditionPipeline
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteSchedulerOutput,
    )

    class AudioScheduler(FlowMatchEulerDiscreteScheduler):
        """The audio stream's own schedule: ``min(video sigma, audio_level)`` per step.

        ``step`` is called with the VIDEO timestep, so it indexes by its own
        counter. A step whose sigma does not move (every step when frozen, the
        hold steps of a variation) returns the sample untouched: with sigma 0
        upstream's x0 -> velocity conversion is 0/0, and that must never reach
        the latents.
        """

        audio_level: float = FROZEN

        def set_timesteps(
            self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None
        ):
            super().set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                timesteps=timesteps,
            )
            video = self.sigmas[:-1]
            audio = torch.clamp(video, max=float(self.audio_level))
            self.sigmas = torch.cat([audio, self.sigmas[-1:]])
            self.timesteps = audio * self.config.num_train_timesteps
            self._audio_step = 0

        def step(self, model_output, timestep, sample, *args, return_dict=True, **kwargs):
            index = self._audio_step
            self._audio_step += 1
            sigma, sigma_next = self.sigmas[index], self.sigmas[index + 1]
            if bool(sigma_next == sigma):
                prev_sample = sample
            else:
                prev_sample = sample.to(torch.float32) + (sigma_next - sigma) * model_output
                prev_sample = prev_sample.to(model_output.dtype)
            if not return_dict:
                return (prev_sample,)
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    class AudioConditionedPipeline(LTX2ConditionPipeline):
        """``LTX2ConditionPipeline`` whose supplied audio is noised to its own level.

        ``audio_noise_level`` is set per call by :func:`audio_pipeline`. The
        ``noise_scale`` upstream hands ``prepare_audio_latents`` is the video's
        initial level (its first sigma), and the audio never starts noisier than
        the video: stage 2 starts at its own first sigma.
        """

        audio_noise_level: float | None = None

        def prepare_audio_latents(
            self,
            batch_size: int = 1,
            num_channels_latents: int = 8,
            audio_latent_length: int = 1,
            num_mel_bins: int = 64,
            noise_scale: float = 0.0,
            dtype=None,
            device=None,
            generator=None,
            latents=None,
        ):
            if latents is not None and self.audio_noise_level is not None:
                noise_scale = min(float(self.audio_noise_level), float(noise_scale))
            return super().prepare_audio_latents(
                batch_size,
                num_channels_latents=num_channels_latents,
                audio_latent_length=audio_latent_length,
                num_mel_bins=num_mel_bins,
                noise_scale=noise_scale,
                dtype=dtype,
                device=device,
                generator=generator,
                latents=latents,
            )

    _CLASSES.update(pipeline=AudioConditionedPipeline, scheduler=AudioScheduler)
    return AudioConditionedPipeline, AudioScheduler


def is_condition_pipeline(pipeline) -> bool:
    """True for a real ``LTX2ConditionPipeline`` (a stand-in in the offline gate is not)."""
    try:
        from diffusers import LTX2ConditionPipeline
    except ImportError:
        return False
    return isinstance(pipeline, LTX2ConditionPipeline)


def audio_pipeline(base, level: float, *, derived=None):
    """``base`` as an :class:`AudioConditionedPipeline` at ``level``, sharing every module.

    ``derived`` is a previously derived pipeline to reuse; its modules are
    re-synced to ``base``'s on every call, because the local runner attaches and
    releases the transformer and text encoder on the base between calls. The
    audio scheduler is fresh per call (it carries a step counter).
    """
    from .pipelines import derive_pipeline

    pipeline_cls, scheduler_cls = _classes()
    if derived is None:
        derived = derive_pipeline(base, pipeline_cls)
    for name, module in base.components.items():
        if name != "audio_scheduler" and getattr(derived, name, None) is not module:
            setattr(derived, name, module)
    for name in ("transformer_spatial_patch_size", "transformer_temporal_patch_size"):
        if hasattr(base, name):
            setattr(derived, name, getattr(base, name))
    scheduler = scheduler_cls.from_config(base.scheduler.config)
    scheduler.audio_level = float(level)
    derived.audio_scheduler = scheduler
    derived.audio_noise_level = float(level)
    return derived
