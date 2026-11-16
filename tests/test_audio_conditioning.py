"""Spec for ``audio_conditioning`` — the audio stream's own noise level and schedule.

Upstream's condition pipeline used one ``noise_scale`` for the audio AND the
video's initial noise while the schedule started at sigma 1.0: the denoiser was
told "pure noise" and handed half-variance video latents (the grey veil in
``audio_to_video``). Now the video is sampled as in every other mode and the
audio has its own level:

* ``audio_to_video`` — frozen: every audio sigma is 0, the audio latents are the
  encoded upload, bit for bit, at every step and after the loop;
* ``audio_to_audio`` — variation at a snapped level ``s``: the audio is noised
  to ``s`` and its sigma per step is ``min(video sigma, s)``.

The real-pipeline half runs the tiny random checkpoint on the CPU: it verifies
mechanics, not sound or pictures.
"""

from __future__ import annotations

import pytest
from conftest import inputs_for

from ltx25_studio import audio_conditioning, errors, modes
from ltx25_studio import generate as generate_module
from ltx25_studio.constraints import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

# --------------------------------------------------------------------------
# levels and schedules (pure)
# --------------------------------------------------------------------------


def test_the_variation_levels_are_the_models_own_distilled_sigmas_and_one():
    assert audio_conditioning.VARIATION_LEVELS == (0.421875, 0.725, 0.909375, 1.0)


@pytest.mark.parametrize(
    "asked,level",
    [
        (0.0, 0.421875),
        (0.5, 0.421875),
        (0.6, 0.725),
        (0.8, 0.725),
        (0.85, 0.909375),
        (0.95, 0.909375),
        (0.96, 1.0),
        (1.0, 1.0),
    ],
)
def test_a_requested_level_snaps_to_the_nearest_supported_one(asked, level):
    assert audio_conditioning.snap_level(asked) == level


@pytest.mark.parametrize("level", audio_conditioning.VARIATION_LEVELS)
def test_the_audio_schedule_holds_at_its_level_then_follows_the_tail(level):
    sigmas = audio_conditioning.audio_sigmas(DISTILLED_SIGMA_VALUES, level)

    assert len(sigmas) == len(DISTILLED_SIGMA_VALUES)
    reached = next(i for i, sigma in enumerate(DISTILLED_SIGMA_VALUES) if sigma <= level)
    assert sigmas[:reached] == [level] * reached
    assert sigmas[reached:] == list(DISTILLED_SIGMA_VALUES[reached:])


def test_the_frozen_schedule_is_all_zero():
    assert audio_conditioning.audio_sigmas(DISTILLED_SIGMA_VALUES, 0.0) == [0.0] * 8


def test_the_level_note_says_what_happened_in_plain_words():
    assert "changed moderately (level 0.73)" in audio_conditioning.level_note(0.725)
    assert "lightly" in audio_conditioning.level_note(0.421875)
    assert "strongly" in audio_conditioning.level_note(0.909375)
    fresh = audio_conditioning.level_note(1.0)
    assert "not used" in fresh, fresh
    snapped = audio_conditioning.level_note(0.421875, requested=0.5)
    assert "0.50" in snapped and "0.42" in snapped, snapped


# --------------------------------------------------------------------------
# the registry: what each mode sends
# --------------------------------------------------------------------------


def test_audio_to_video_freezes_the_audio_and_sends_no_noise_scale(make_request):
    kwargs = modes.get_mode("audio_to_video").build_kwargs(
        make_request(mode="audio_to_video", audio=object(), audio_latents="latents")
    )

    assert kwargs["audio_noise_level"] == audio_conditioning.FROZEN
    assert "noise_scale" not in kwargs, (
        "noise_scale on the condition pipeline is also the video's initial noise level"
    )
    assert kwargs["audio_latents"] == "latents"


def test_audio_to_audio_varies_at_the_snapped_level_and_sends_no_noise_scale(make_request):
    kwargs = modes.get_mode("audio_to_audio").build_kwargs(
        make_request(mode="audio_to_audio", audio=object(), audio_latents="x", noise_scale=0.7)
    )

    assert kwargs["audio_noise_level"] == 0.725
    assert "noise_scale" not in kwargs


def test_video_to_audio_keeps_the_video_exactly(make_request):
    kwargs = modes.get_mode("video_to_audio").build_kwargs(
        make_request(mode="video_to_audio", video=object(), strength=0.5)
    )

    assert [c.strength for c in kwargs["conditions"]] == [1.0]


@pytest.mark.parametrize("asked,level", [(None, 0.909375), (0.5, 0.421875), (1.0, 0.909375)])
def test_video_to_video_reads_its_level_from_noise_scale(make_request, asked, level):
    kwargs = modes.get_mode("video_to_video").build_kwargs(
        make_request(mode="video_to_video", video=object(), noise_scale=asked, strength=0.6)
    )

    assert kwargs["video_variation"].level == level


@pytest.mark.parametrize(
    "mode_id,field,value",
    [("audio_to_video", "noise_scale", 0.9), ("video_to_audio", "strength", 0.3)],
)
def test_a_lever_the_mode_does_not_read_is_reset_by_scoping(mode_id, field, value):
    spec = modes.get_mode(mode_id)
    request = generate_module.GenerationRequest(mode=mode_id, **{field: value})

    scoped, _notes = generate_module.scope_request_to_mode(request, spec)

    assert getattr(scoped, field) == getattr(generate_module.GenerationRequest(mode=mode_id), field)


def test_no_user_visible_string_claims_the_model_suppresses_audio():
    for spec in modes.MODES.values():
        assert "suppress" not in (spec.cost_note or "").lower(), spec.id
    assert not hasattr(generate_module, "SUPPRESSION_EVIDENCE")
    for exports in (("audio",), ("audio", "video")):
        for muxed in (True, False):
            for note in generate_module._supplied_audio_note("m", exports, muxed=muxed):
                assert "suppress" not in note.lower(), note
        for note in generate_module._supplied_latents_note("m", exports):
            assert "suppress" not in note.lower(), note


# --------------------------------------------------------------------------
# through generate() with a fake pipeline: the level reaches the run, and its note
# --------------------------------------------------------------------------


class _Pipe:
    def __init__(self):
        from conftest import FakeAudioVae

        self.calls = []
        self.audio_vae = FakeAudioVae()

    def __call__(self, **kwargs):
        from test_exports_contract import FakeOutput

        self.calls.append(kwargs)
        return FakeOutput()


def _fake_run(mode_id, tmp_path, **overrides):
    from test_exports_contract import factory_returning, one_muxed_file_encoder

    pipe = _Pipe()
    result = generate_module.generate(
        mode=mode_id,
        seed=7,
        **{**inputs_for(mode_id), **overrides},
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    return result, pipe.calls[-1]


@pytest.mark.usefixtures("audio_input_ready")
def test_audio_to_audio_reports_the_level_it_used(tmp_path):
    result, sent = _fake_run("audio_to_audio", tmp_path, noise_scale=0.5)

    assert sent["audio_noise_level"] == 0.421875
    assert "noise_scale" not in sent
    assert any("lightly (level 0.42)" in note for note in result.warnings), result.warnings


@pytest.mark.usefixtures("audio_input_ready")
def test_audio_to_video_is_frozen_through_generate(tmp_path):
    result, sent = _fake_run("audio_to_video", tmp_path, noise_scale=0.3)

    assert sent["audio_noise_level"] == 0.0
    assert "noise_scale" not in sent
    assert any("held fixed" in note for note in result.warnings), result.warnings


# --------------------------------------------------------------------------
# the real tiny pipelines (CPU): mechanics
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from test_real_pipeline_tiny import _tiny_snapshot_cached, load_smoke

    if not _tiny_snapshot_cached():
        pytest.skip("the tiny checkpoint is not in the HF cache")
    return load_smoke().tiny_stack(device="cpu")


@pytest.fixture
def transformer_calls(tiny):
    """Per transformer call: video sigma, audio sigma, and the two hidden states."""
    seen: list[dict] = []

    def hook(_module, _args, kwargs):
        seen.append(
            {
                "sigma": float(kwargs["sigma"].flatten()[0]) / 1000.0,
                "audio_sigma": float(kwargs["audio_sigma"].flatten()[0]) / 1000.0,
                "audio_timestep": kwargs["audio_timestep"].clone(),
                "hidden_states": kwargs["hidden_states"].clone(),
                "audio_hidden_states": kwargs["audio_hidden_states"].clone(),
            }
        )

    handle = tiny["base"].transformer.register_forward_pre_hook(hook, with_kwargs=True)
    yield seen
    handle.remove()


@pytest.fixture
def loop_ends(monkeypatch):
    """The audio latents leaving each denoise loop (packed, normalised)."""
    from diffusers import LTX2ConditionPipeline

    ends = []
    original = LTX2ConditionPipeline._denormalize_audio_latents

    def capture(latents, *args, **kwargs):
        ends.append(latents.clone())
        return original(latents, *args, **kwargs)

    monkeypatch.setattr(LTX2ConditionPipeline, "_denormalize_audio_latents", staticmethod(capture))
    return ends


@pytest.fixture
def sent_audio(monkeypatch):
    """The ``audio_latents`` each condition-pipeline call was handed."""
    from diffusers import LTX2ConditionPipeline

    sent = []
    original = LTX2ConditionPipeline.__call__

    def spy(self, *args, **kwargs):
        sent.append((type(self).__name__, kwargs.get("audio_latents"), dict(kwargs)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LTX2ConditionPipeline, "__call__", spy)
    return sent


def _clean_packed(base, latents):
    packed = base._pack_audio_latents(latents)
    return base._normalize_audio_latents(
        packed, base.audio_vae.latents_mean, base.audio_vae.latents_std
    ).float()


def _run(tiny, mode_id, tmp_path, **overrides):
    from test_real_pipeline_tiny import run_tiny

    return run_tiny(tiny, mode_id, tmp_path, **overrides)


def test_audio_to_video_holds_the_audio_clean_at_every_step(
    tiny, tmp_path, transformer_calls, loop_ends, sent_audio
):
    torch = pytest.importorskip("torch")

    _run(tiny, "audio_to_video", tmp_path)

    ((pipeline_name, latents, _kwargs),) = sent_audio
    assert pipeline_name == "AudioConditionedPipeline"
    clean = _clean_packed(tiny["base"], latents)
    assert len(transformer_calls) == len(DISTILLED_SIGMA_VALUES)
    for step, call in enumerate(transformer_calls):
        assert call["audio_sigma"] == 0.0 and not call["audio_timestep"].any(), step
        assert torch.equal(call["audio_hidden_states"].float(), clean), step
    (end,) = loop_ends
    assert torch.isfinite(end).all()
    assert torch.equal(end, clean), "the audio latents changed during the loop"


def test_audio_to_video_starts_the_video_from_unit_variance_noise(
    tiny, tmp_path, transformer_calls
):
    _run(tiny, "audio_to_video", tmp_path, width=128, height=128, num_frames=17)

    first = transformer_calls[0]
    assert first["sigma"] == 1.0
    std = float(first["hidden_states"].float().std())
    assert 0.85 < std < 1.15, f"the video started at std {std:.3f}, not ~1 (was ~0.5)"


@pytest.mark.parametrize("level", audio_conditioning.VARIATION_LEVELS)
def test_audio_to_audio_follows_the_hold_then_tail_schedule(
    tiny, tmp_path, transformer_calls, monkeypatch, level
):
    torch = pytest.importorskip("torch")
    from diffusers import LTX2ConditionPipeline

    noised = []
    original = LTX2ConditionPipeline._create_noised_state

    def capture(latents, noise_scale, generator=None):
        noised.append(float(noise_scale))
        return original(latents, noise_scale, generator)

    monkeypatch.setattr(LTX2ConditionPipeline, "_create_noised_state", staticmethod(capture))

    _run(tiny, "audio_to_audio", tmp_path, noise_scale=level)

    expected = torch.tensor(audio_conditioning.audio_sigmas(DISTILLED_SIGMA_VALUES, level))
    audio = torch.tensor([call["audio_sigma"] for call in transformer_calls])
    video = torch.tensor([call["sigma"] for call in transformer_calls])
    assert torch.allclose(audio, expected.float(), atol=1e-6), audio.tolist()
    assert torch.allclose(video, torch.tensor(DISTILLED_SIGMA_VALUES).float(), atol=1e-6)
    assert noised == [level], f"the supplied audio was noised to {noised}, not {level}"


def test_audio_to_video_stays_frozen_through_stage_2(
    tiny, tmp_path, transformer_calls, loop_ends, sent_audio
):
    torch = pytest.importorskip("torch")

    _run(tiny, "audio_to_video", tmp_path, enable_stage_2=True)

    assert len(sent_audio) == 2
    stage_count = len(DISTILLED_SIGMA_VALUES)
    video = [call["sigma"] for call in transformer_calls]
    assert video[stage_count:] == pytest.approx(list(STAGE_2_DISTILLED_SIGMA_VALUES))
    assert all(call["audio_sigma"] == 0.0 for call in transformer_calls)
    for (_name, latents, _kwargs), end in zip(sent_audio, loop_ends, strict=True):
        assert torch.equal(end, _clean_packed(tiny["base"], latents))


@pytest.mark.parametrize("level", [0.421875, 1.0])
def test_a_variation_follows_its_own_tail_in_stage_2(tiny, transformer_calls, level):
    """``audio_to_audio`` skips stage 2 in the registry; the graph still composes."""
    torch = pytest.importorskip("torch")

    runner, base = tiny["condition"], tiny["base"]
    bins = base.audio_mel_bins // base.audio_vae_mel_compression_ratio
    audio = torch.randn(
        1, base.audio_latent_channels, 6, bins, generator=torch.Generator().manual_seed(1)
    )
    runner(
        prompt="a drum",
        width=64,
        height=64,
        num_frames=9,
        audio_latents=audio,
        audio_noise_level=level,
        generator=torch.Generator().manual_seed(2),
        enable_stage_2=True,
    )

    stage_2 = transformer_calls[len(DISTILLED_SIGMA_VALUES) :]
    expected = audio_conditioning.audio_sigmas(STAGE_2_DISTILLED_SIGMA_VALUES, level)
    assert [c["audio_sigma"] for c in stage_2] == pytest.approx(expected)


def test_video_to_audio_sends_a_full_strength_video_condition(tiny, tmp_path, sent_audio):
    _run(tiny, "video_to_audio", tmp_path)

    ((_name, _latents, kwargs),) = sent_audio
    assert [condition.strength for condition in kwargs["conditions"]] == [1.0]


# --------------------------------------------------------------------------
# video_to_video: a variation of the user's clip (F66)
#
# The conditioning mask was a switch (clip unchanged at 0.3-0.8, prompt only at
# 0.0). The clip is now the starting point: encoded to raw VAE latents, noised to
# a level, and only the tail of the distilled schedule from that level denoised,
# with no condition -- what stage 2 does to its upsampled latents.
# --------------------------------------------------------------------------


def test_the_clip_levels_are_the_three_distilled_sigmas():
    assert audio_conditioning.VIDEO_VARIATION_LEVELS == (0.421875, 0.725, 0.909375)
    assert audio_conditioning.DEFAULT_VIDEO_LEVEL == 0.909375
    assert audio_conditioning.sigma_tail(0.725) == [0.725, 0.421875]
    assert audio_conditioning.snap_video_level(1.0) == 0.909375


def test_the_clip_note_says_how_much_it_was_changed():
    assert "changed a lot (level 0.91)" in audio_conditioning.video_level_note(0.909375)
    assert "a little (level 0.42)" in audio_conditioning.video_level_note(0.421875)


def _clip(tmp_path, frames, size=64):
    from test_real_pipeline_tiny import load_smoke

    return load_smoke().make_video(tmp_path / f"clip_{frames}.mp4", size, frames)


def _v2v(tiny, tmp_path, **overrides):
    return _run(tiny, "video_to_video", tmp_path, prompt="a cabin in snow", **overrides)


@pytest.mark.parametrize("level", audio_conditioning.VIDEO_VARIATION_LEVELS)
def test_the_first_call_starts_from_the_clips_latents_noised_to_the_level(
    tiny, tmp_path, transformer_calls, sent_audio, level
):
    torch = pytest.importorskip("torch")
    from diffusers.utils.torch_utils import randn_tensor

    result = _v2v(tiny, tmp_path, video=_clip(tmp_path, 9), noise_scale=level)

    ((_name, _audio, kwargs),) = sent_audio
    assert not kwargs.get("conditions"), "a clip variation must not also condition on the clip"
    assert kwargs["noise_scale"] == level
    assert kwargs["sigmas"] == audio_conditioning.sigma_tail(level)
    base = tiny["base"]
    raw = kwargs["latents"]
    clean = base._pack_latents(
        base._normalize_latents(
            raw, base.vae.latents_mean, base.vae.latents_std, base.vae.config.scaling_factor
        ),
        base.transformer_spatial_patch_size,
        base.transformer_temporal_patch_size,
    )
    noise = randn_tensor(clean.shape, generator=torch.Generator().manual_seed(7), dtype=clean.dtype)
    scaled = (1.0 - clean.new_zeros(clean.shape[:2] + (1,))) * level
    expected = noise * scaled + clean * (1 - scaled)
    assert torch.equal(transformer_calls[0]["hidden_states"], expected)
    assert len(transformer_calls) == len(audio_conditioning.sigma_tail(level))
    assert [c["sigma"] for c in transformer_calls] == pytest.approx(
        audio_conditioning.sigma_tail(level)
    )
    assert result.record.video_level == level


def test_the_length_follows_the_clip_and_the_note_says_so(tiny, tmp_path):
    result = _v2v(tiny, tmp_path, video=_clip(tmp_path, 17), num_frames=9)

    assert result.record.frames == 17
    assert result.record.length_source == "video"
    assert any("length set from your clip" in note for note in result.warnings)
    assert any("changed a lot (level 0.91)" in note for note in result.warnings)


def test_stage_2_composes_on_top_of_a_clip_variation(tiny, tmp_path, transformer_calls):
    result = _v2v(tiny, tmp_path, video=_clip(tmp_path, 9), enable_stage_2=True)

    tail = audio_conditioning.sigma_tail(audio_conditioning.DEFAULT_VIDEO_LEVEL)
    sigmas = [c["sigma"] for c in transformer_calls]
    assert sigmas == pytest.approx(tail + list(STAGE_2_DISTILLED_SIGMA_VALUES))
    assert (result.record.width, result.record.height) == (128, 128)


def test_a_missing_clip_is_refused_before_dispatch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", lambda **kw: calls.append(kw))

    with pytest.raises(errors.ConstraintError) as caught:
        generate_module.generate(mode="video_to_video", video=str(tmp_path / "gone.mp4"))
    assert not calls
    assert errors.categorise(caught.value) == errors.Failure(errors.INPUT, False)


def test_a_clip_with_no_picture_is_refused_before_dispatch(tmp_path, monkeypatch):
    from conftest import audio_file

    calls = []
    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", lambda **kw: calls.append(kw))

    with pytest.raises(errors.ConstraintError) as caught:
        generate_module.generate(mode="video_to_video", video=audio_file())
    assert not calls
    assert "no picture" in str(caught.value)
