"""Spec for ``StageRunner`` and ``startup_report`` — PLAN §7.4, §5.6.

The last two names on the ungrepped list: `grep -rn StageRunner tests/` and
`grep -rn startup_report tests/` both returned **0**.

``StageRunner`` is where PLAN §7.4 — "the most consequential architectural
decision in the document" — actually executes, and every failure mode it guards
is a *silent* one:

* **The diffusion decoder.** Upstream's ``LTX25AutoBlocks`` says it "is the
  native default from LTX-2.5 on". Decoding a staged render with the
  convolutional VAE instead "would not error. It would just be worse, forever,
  and nobody would know why." A stage graph that ran both denoise passes and
  then decoded with the wrong thing is indistinguishable from a correct one at
  every level a test usually looks.
* **``denormalize=False`` is mandatory.** ``output_type="latent"`` returns
  *already-denormalized* latents, so the default ``True`` double-denormalizes
  and silently corrupts the output. PLAN says in as many words: "This is a test
  case." It was not one.
* **No silent single-stage fallback.** Stage 2 requested without the upsampler
  or decoder loaded must raise, not quietly render one stage — "which would look
  like it worked and be visibly worse".
* **``noise_scale``.** Re-noising upsampled latents to 1.0 discards stage 1
  entirely; the value must be the first stage-2 sigma. A wrong constant here
  throws away the pass the user paid extra for and reports success.

``startup_report`` is the other kind of silence: it is what gets read at a
distance when a metered Space fails behind a 72 GB build, and "there is no
opportunity to add a print statement afterwards". A report that omits the thing
that broke, or that raises while being built, costs whole round trips.

All seams are injected — ``pipeline``, ``upsampler``, ``decoder`` are
constructor arguments — so none of this needs diffusers, torch or a GPU.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import pathlib

import pytest

from ltx25_studio import pipelines
from ltx25_studio.errors import ModelLoadError


class Recorder:
    """A callable stage that records how it was called and returns latents."""

    def __init__(self, name, output=None):
        self.name = name
        self.calls: list[dict] = []
        self._output = output if output is not None else Latents(f"{name}-latents")

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self._output

    @property
    def kwargs(self):
        assert self.calls, f"{self.name} was never called"
        return self.calls[-1]


class Latents:
    """``output_type="latent"`` puts the latents on the ``frames`` field."""

    def __init__(self, tag):
        self.frames = tag


def make_runner(*, with_stage_2_parts=True):
    pipeline = Recorder("pipeline")
    upsampler = Recorder("upsampler") if with_stage_2_parts else None
    decoder = Recorder("decoder") if with_stage_2_parts else None
    runner = pipelines.StageRunner(
        pipeline, kind="text2video", upsampler=upsampler, decoder=decoder
    )
    return runner, pipeline, upsampler, decoder


# --------------------------------------------------------------------------
# single stage: the default, and what must not leak
# --------------------------------------------------------------------------


def test_a_default_call_runs_exactly_one_denoise():
    runner, pipeline, upsampler, decoder = make_runner()

    runner(prompt="a lighthouse", num_frames=121)

    assert len(pipeline.calls) == 1, f"the pipeline ran {len(pipeline.calls)} times, not once"
    assert not upsampler.calls, "the upsampler ran on a single-stage call"
    assert not decoder.calls, "the diffusion decoder ran on a single-stage call"


def test_the_stage_2_flag_is_consumed_and_never_reaches_diffusers():
    """``enable_stage_2`` is ours, not upstream's.

    Forwarding it would be a ``TypeError`` from inside diffusers on every single
    run — the kind of failure that is obvious in production and invisible to a
    fake that accepts ``**kwargs``.
    """
    runner, pipeline, _upsampler, _decoder = make_runner()

    runner(prompt="a lighthouse", enable_stage_2=False)

    assert "enable_stage_2" not in pipeline.kwargs, (
        f"enable_stage_2 was passed through to the pipeline: {sorted(pipeline.kwargs)}"
    )


def test_a_single_stage_call_does_not_ask_for_latents():
    """Single stage is decoded by the pipeline itself, so it wants frames back.

    Requesting ``output_type="latent"`` here would hand ``outputs.py`` latents
    where it expects pixels — which fails as a shape error at the writer, several
    layers from the cause.
    """
    runner, pipeline, _upsampler, _decoder = make_runner()

    runner(prompt="a lighthouse")

    assert pipeline.kwargs.get("output_type") != "latent"


# --------------------------------------------------------------------------
# two stages: the graph PLAN §7.4 argues for
# --------------------------------------------------------------------------


def test_the_two_stage_graph_runs_every_step_in_order():
    runner, pipeline, upsampler, decoder = make_runner()

    runner(prompt="a lighthouse", enable_stage_2=True)

    assert len(pipeline.calls) == 2, (
        f"a staged render must denoise twice; the pipeline ran {len(pipeline.calls)} times"
    )
    assert len(upsampler.calls) == 1, "the latent upsampler did not run"
    assert len(decoder.calls) == 1, "the diffusion decoder did not run"


def test_the_staged_render_is_decoded_by_the_diffusion_decoder():
    """PLAN §7.4: the single biggest 2.5 quality feature, silently forfeitable.

    A graph that ran both passes and then let the pipeline decode with the
    convolutional VAE produces a plausible video, no error, and permanently
    worse output. The only observable difference is *which callable produced
    the result*.
    """
    runner, _pipeline, _upsampler, decoder = make_runner()

    result = runner(prompt="a lighthouse", enable_stage_2=True)

    assert result is decoder._output, (
        "the staged render did not come from the diffusion decoder. Decoding with the "
        "convolutional VAE instead would not error — it would just be worse, forever."
    )


def test_the_diffusion_decode_does_not_denormalize_twice():
    """PLAN §7.4: "``denormalize=False`` is MANDATORY [...] This is a test case."

    ``output_type="latent"`` returns already-denormalized latents. Passing the
    default ``True`` double-denormalizes and silently corrupts the output.
    """
    runner, _pipeline, _upsampler, decoder = make_runner()

    runner(prompt="a lighthouse", enable_stage_2=True)

    assert decoder.kwargs.get("denormalize") is False, (
        f"the diffusion decode was called with denormalize="
        f"{decoder.kwargs.get('denormalize')!r}. Anything but False double-denormalizes "
        f"already-denormalized latents and silently corrupts the frames."
    )


def test_stage_1_emits_latents_rather_than_frames():
    runner, pipeline, _upsampler, _decoder = make_runner()

    runner(prompt="a lighthouse", enable_stage_2=True)

    assert pipeline.calls[0].get("output_type") == "latent", (
        "stage 1 must emit latents for the upsampler; decoding to frames here throws the "
        "refinement pass away before it starts"
    )


def test_stage_2_refines_the_upsampled_latents_rather_than_starting_over():
    """The upsampler's output must be what stage 2 is handed."""
    runner, pipeline, upsampler, _decoder = make_runner()

    runner(prompt="a lighthouse", enable_stage_2=True)

    assert pipeline.calls[1].get("latents") == upsampler._output.frames, (
        "stage 2 did not receive the upsampled latents, so the upsample pass was paid for "
        "and discarded"
    )


def test_stage_2_does_not_renoise_away_stage_1():
    """``noise_scale`` must be the first stage-2 sigma, not 1.0.

    The upsampled latents are already partly denoised; re-noising to 1.0 discards
    stage 1 entirely — the user pays for two passes and receives one.
    """
    runner, pipeline, _upsampler, _decoder = make_runner()

    runner(prompt="a lighthouse", enable_stage_2=True)

    noise_scale = pipeline.calls[1].get("noise_scale")
    assert noise_scale == pipelines.STAGE_2_DISTILLED_SIGMA_VALUES[0], (
        f"stage 2 re-noised to {noise_scale!r}; it must start from the first stage-2 sigma "
        f"({pipelines.STAGE_2_DISTILLED_SIGMA_VALUES[0]}) or stage 1 is thrown away."
    )
    assert noise_scale != 1.0, "noise_scale=1.0 discards the input latents entirely"


def test_the_two_denoise_passes_use_different_schedules():
    """Stage 2 is the short refinement schedule, not a second full render."""
    runner, pipeline, _upsampler, _decoder = make_runner()

    runner(prompt="a lighthouse", enable_stage_2=True)

    first, second = pipeline.calls[0].get("sigmas"), pipeline.calls[1].get("sigmas")
    assert first and second, "both stages must pass an explicit sigma schedule"
    assert first != second, (
        "both denoise passes used the same schedule; stage 2 is a three-step refinement, not "
        "a second full render"
    )


def test_the_users_prompt_survives_into_both_stages():
    """A refinement pass that forgets the prompt refines toward something else."""
    runner, pipeline, _upsampler, _decoder = make_runner()

    runner(prompt="a lighthouse in heavy fog", enable_stage_2=True)

    for index, call in enumerate(pipeline.calls):
        assert call.get("prompt") == "a lighthouse in heavy fog", (
            f"stage {index + 1} was called without the user's prompt: {sorted(call)}"
        )


# --------------------------------------------------------------------------
# it must refuse rather than quietly render one stage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["upsampler", "decoder", "both"])
def test_stage_2_without_its_parts_raises_rather_than_falling_back(missing):
    """ "Refusing to silently fall back to a single-stage render, which would look
    like it worked and be visibly worse."

    This is §5.3's doctrine applied to quality instead of to files: the failure
    mode is a success-shaped result that is quietly not what was asked for.
    """
    pipeline = Recorder("pipeline")
    runner = pipelines.StageRunner(
        pipeline,
        kind="text2video",
        upsampler=None if missing in ("upsampler", "both") else Recorder("upsampler"),
        decoder=None if missing in ("decoder", "both") else Recorder("decoder"),
    )

    with pytest.raises(ModelLoadError):
        runner(prompt="a lighthouse", enable_stage_2=True)


@pytest.mark.parametrize("missing", ["upsampler", "decoder", "both"])
def test_a_refused_stage_2_does_not_burn_a_render_first(missing):
    """Raise before spending the GPU, not after."""
    pipeline = Recorder("pipeline")
    runner = pipelines.StageRunner(
        pipeline,
        kind="text2video",
        upsampler=None if missing in ("upsampler", "both") else Recorder("upsampler"),
        decoder=None if missing in ("decoder", "both") else Recorder("decoder"),
    )

    with pytest.raises(ModelLoadError):
        runner(prompt="a lighthouse", enable_stage_2=True)

    assert not pipeline.calls, (
        "the stage graph rendered stage 1 and only then discovered it could not finish; the "
        "check is cheap and belongs before the spend"
    )


def test_a_stage_that_returns_no_latents_stops_the_graph():
    """ "Refusing to continue with nothing" — the never-empty rule, mid-graph.

    This is the exact shape of the 2.3 outage: a mid-graph step produced nothing
    and the layers below turned it into a falsy value rather than an error.
    """

    class Empty:
        frames = None
        latents = None

    pipeline = Recorder("pipeline", output=Empty())
    runner = pipelines.StageRunner(
        pipeline, kind="text2video", upsampler=Recorder("up"), decoder=Recorder("dec")
    )

    with pytest.raises(ModelLoadError):
        runner(prompt="a lighthouse", enable_stage_2=True)


# --------------------------------------------------------------------------
# the startup report — what gets read at a distance
# --------------------------------------------------------------------------


@pytest.fixture
def runtime_config():
    """The **real** config object, not a fake.

    The report reads it through ``describe()``, so a hand-rolled stand-in would
    pin the shape of the fake rather than the shape of the thing — and would go
    on passing after the real config grew a field the report needs.
    """
    from ltx25_studio.config import load_config

    return load_config()


def test_the_startup_report_is_a_non_empty_string(runtime_config):
    report = pipelines.startup_report(runtime_config)
    assert isinstance(report, str) and report.strip(), (
        "the startup report is empty. It is what gets read at a distance when a metered "
        "Space fails behind a 72 GB build, and there is no chance to add a print afterwards."
    )


@pytest.mark.parametrize("quantization", ["bf16", "int8"])
def test_the_startup_report_names_the_resolved_precision(monkeypatch, quantization):
    """ "So the running configuration is never inferred."

    A report that omits the precision leaves the single most expensive
    configuration question — what dtype are we actually running — to guesswork.
    It used to print a fixed ``bf16`` whatever was configured; the live int8
    Space said ``bf16 (fixed; no precision switch)``.
    """
    from ltx25_studio import config, device, zerogpu

    monkeypatch.setenv(config.ENV_QUANTIZATION, quantization)
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: True)
    runtime_config = config.load_config()
    resolution = device.resolve(None, runtime_config)

    report = pipelines.startup_report(runtime_config, missing_files=(), resolution=resolution)

    line = next((row for row in report.splitlines() if "precision" in row), "")
    assert line.split(":", 1)[1].strip().startswith(quantization), (
        f"the report does not state the resolved precision:\n\n{report}"
    )
    assert "no precision switch" not in report


def test_the_startup_report_names_the_files_that_are_missing(runtime_config):
    """The whole point of it: say what is absent, by name.

    A missing shard surfaces 70 GB into a download otherwise.
    """
    missing = ("transformer/diffusion_pytorch_model-00003-of-00004.safetensors",)
    report = pipelines.startup_report(runtime_config, missing_files=missing)

    assert missing[0] in report or "00003" in report, (
        f"a missing weight file was not named in the startup report:\n\n{report}"
    )


def test_the_startup_report_survives_being_given_nothing(runtime_config):
    """It runs at startup, so it must never be the thing that breaks startup.

    Every argument but the config is optional, and a diagnostic that raises while
    being built converts a diagnosable failure into an undiagnosable one.
    """
    report = pipelines.startup_report(
        runtime_config, cached_files=None, missing_files=(), resolution=None
    )
    assert report.strip()


def test_a_clean_startup_report_does_not_claim_files_are_missing(runtime_config):
    """The inverse: a report that always warns is a report nobody reads."""
    report = pipelines.startup_report(runtime_config, missing_files=())
    assert "00003" not in report


# --------------------------------------------------------------------------
# one loaded copy of the model, shared — PLAN §8, and the 150 GB ceiling
# --------------------------------------------------------------------------
#
# `from_pretrained` twice materialises two complete independent copies, ~67 GiB
# each. On ZeroGPU that is ~134 GiB packed to disk against a 150 GB cap, with the
# packing running after both builds; on a 96 GiB card it is an outright OOM.
#
# At the pinned revision the two pipeline classes take IDENTICAL weight-bearing
# components — the only difference is `audio_scheduler`, which carries no weights
# and defaults to None. So there is no version of this where two copies are what
# anyone wanted.
#
# These assert IDENTITY, not equality. The defect is invisible to any check that
# only asks whether both pipelines work: two copies work fine, right up until the
# disk or the card runs out.

WEIGHT_BEARING = ("transformer", "vae", "audio_vae", "text_encoder", "connectors", "vocoder")


class _Module:
    """Stands in for a loaded component. Identity is the whole point."""


class _FakeBase:
    def __init__(self):
        self._modules = {name: _Module() for name in WEIGHT_BEARING}
        self._modules["audio_scheduler"] = _Module()

    @property
    def components(self):
        return dict(self._modules)

    def __getattr__(self, name):
        try:
            return self.__dict__["_modules"][name]
        except KeyError:
            raise AttributeError(name) from None


class _FakeTarget:
    """A pipeline class WITHOUT `audio_scheduler` — the real asymmetry."""

    def __init__(self, transformer, vae, audio_vae, text_encoder, connectors, vocoder):
        self.transformer = transformer
        self.vae = vae
        self.audio_vae = audio_vae
        self.text_encoder = text_encoder
        self.connectors = connectors
        self.vocoder = vocoder


@pytest.mark.parametrize("component", WEIGHT_BEARING)
def test_the_derived_pipeline_shares_the_very_same_module(component):
    """`is`, not `==`. Equality would pass with two copies loaded."""
    base = _FakeBase()
    derived = pipelines.derive_pipeline(base, _FakeTarget)

    assert getattr(derived, component) is getattr(base, component), (
        f"the derived pipeline has its own {component!r}. Two independent copies of "
        f"the model are ~134 GiB together — over a 96 GiB card and over the 150 GB "
        f"ZeroGPU disk cap, with packing running after both builds."
    )


def test_deriving_never_passes_a_component_the_target_cannot_accept():
    """`audio_scheduler` is the one asymmetry, and it must not be forced across.

    `LTX2ConditionPipeline` takes it; `LTX2Pipeline` does not. Passing it anyway
    would TypeError at construction — loudly, at least — but the reverse is the
    dangerous direction, which is why the base is always the superset class.
    """
    derived = pipelines.derive_pipeline(_FakeBase(), _FakeTarget)
    assert not hasattr(derived, "audio_scheduler")


def test_the_base_workflow_is_the_superset_class():
    """Deriving must run superset -> subset, never the reverse.

    Loading `LTX2Pipeline` first and deriving the condition pipeline from it
    would silently drop `audio_scheduler` — and the condition path is the one
    that uses it, so the joint audio track would degrade in a way the video
    output never reveals. That is this build's recurring failure shape.
    """
    assert pipelines.BASE_KIND == "condition"


# --------------------------------------------------------------------------
# identity was necessary and NOT sufficient — dtype must survive too
# --------------------------------------------------------------------------
#
# `from_pipe(base)` with no dtype resolves `... or torch.float32`
# (pipeline_utils.py:2155) and then calls `new_pipeline.to(dtype=dtype)` (:2253).
#
# The crash it caused on a 96 GiB card is the LESSER half: an fp32 cast of bf16
# allocates twice the bytes, so the "derived" pipeline was a second ~130 GiB copy.
#
# The half that matters is that `nn.Module.to()` mutates IN PLACE and the modules
# are genuinely shared — so had it fit, **the base would silently have become
# fp32** and every generation after it would have run at the wrong precision with
# no error and plausible output.
#
# The identity tests above passed WITH this bug present: the modules really were
# shared. They were just also silently recast. Sharing and preserving are two
# properties and this build only asserted one.


class _CastableModule:
    def __init__(self, dtype="bfloat16"):
        self.dtype = dtype


class _CastingBase:
    """Models the real failure: `to(dtype=...)` mutates the shared modules."""

    def __init__(self):
        self.dtype = "bfloat16"
        self._modules = {name: _CastableModule() for name in WEIGHT_BEARING}

    @property
    def components(self):
        return dict(self._modules)

    def to(self, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            for module in self._modules.values():
                module.dtype = dtype
        return self

    def __getattr__(self, name):
        try:
            return self.__dict__["_modules"][name]
        except KeyError:
            raise AttributeError(name) from None


class _CastingTarget:
    """Stands in for diffusers: casts to float32 unless a dtype is supplied."""

    @classmethod
    def from_pipe(cls, base, **kwargs):
        base.to(dtype=kwargs.get("dtype") or "float32")
        return base


def test_deriving_does_not_recast_the_base():
    """The silent half. Mutate `derive_pipeline` back to `from_pipe(base)` and
    this fires; the identity tests above do not."""
    base = _CastingBase()
    pipelines.derive_pipeline(base, _CastingTarget)

    assert base.dtype == "bfloat16", (
        f"deriving cast the BASE pipeline to {base.dtype!r}. nn.Module.to() mutates in "
        f"place and the modules are shared, so every later generation would run at the "
        f"wrong precision — no error, no log line, plausible output."
    )


@pytest.mark.parametrize("component", WEIGHT_BEARING)
def test_the_shared_modules_keep_their_dtype(component):
    base = _CastingBase()
    derived = pipelines.derive_pipeline(base, _CastingTarget)

    assert getattr(derived, component).dtype == "bfloat16", (
        f"{component} was recast while deriving. On a card this OOMs at twice the bytes; "
        f"where it fits, it silently halves precision — and on ZeroGPU a cast also "
        f"materialises anonymous memory, so the post-pack prune frees nothing."
    )


# --------------------------------------------------------------------------
# two stages: the call shapes upstream actually requires
#
# Every test below fails against the graph as first written. That graph ran,
# returned a video, and passed the whole suite -- which is why these assert the
# shape of each call rather than that a call happened.
# --------------------------------------------------------------------------


class FakeTensor:
    """Enough of a tensor to be cast and compared."""

    def __init__(self, tag):
        self.tag = tag

    def to(self, dtype):
        return FakeTensor(f"{self.tag}@{dtype}")

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and other.tag == self.tag

    def __repr__(self):
        return f"FakeTensor({self.tag!r})"


class Staged:
    """A stage output carrying video latents and, optionally, audio latents.

    Field names mirror upstream's ``LTX2PipelineOutput`` because the stage-2
    finish rebuilds the output as ``type(stage_2)(frames=..., audio=...)``.
    """

    def __init__(self, frames, audio=None):
        self.frames = frames if isinstance(frames, FakeTensor) else FakeTensor(frames)
        self.audio = audio


class FakeAudioVae:
    dtype = "bfloat16"

    def __init__(self):
        self.decoded = []

    def decode(self, latents, return_dict=True):
        self.decoded.append(latents)
        return (FakeTensor(f"mel<-{latents.tag}"),)


class FakeVocoder:
    def __init__(self):
        self.mels = []

    def __call__(self, mel):
        self.mels.append(mel)
        return FakeTensor(f"waveform<-{mel.tag}")


class AudioPipeline(Recorder):
    """A pipeline that emits audio latents, as the six audio modes do."""

    def __init__(self, *, audio=True):
        super().__init__("pipeline")
        self.audio_vae = FakeAudioVae()
        self.vocoder = FakeVocoder()
        self._output = Staged("video-latents", audio=FakeTensor("audio-latents") if audio else None)


def two_stage(*, audio=True):
    pipeline = AudioPipeline(audio=audio)
    upsampler = Recorder("upsampler", output=Staged("upsampled"))
    decoder = Recorder("decoder", output=Staged("decoded"))
    runner = pipelines.StageRunner(
        pipeline, kind="text2video", upsampler=upsampler, decoder=decoder
    )
    return runner, pipeline, upsampler, decoder


def test_the_upsample_asks_for_latents_not_pixels():
    """The default is ``"pil"``, and taking it silently ruins the render.

    Without ``output_type="latent"`` the upsampler runs ``vae.decode`` and hands
    back images -- the convolutional decode stage 2 exists to replace -- and
    those images are then passed to stage 2 as its ``latents``.
    """
    runner, _pipeline, upsampler, _decoder = two_stage()

    runner(prompt="a lighthouse", enable_stage_2=True)

    assert upsampler.kwargs.get("output_type") == "latent", (
        "the upsampler was not asked for latents, so it VAE-decoded to images: "
        f"{sorted(upsampler.kwargs)}"
    )


def test_stage_2_is_not_given_a_size():
    """Stage 2 takes its size from the upsampled latents.

    Forwarding the request's stage-1 height/width asks for a render half the size
    of the latents it was just handed.
    """
    runner, pipeline, _upsampler, _decoder = two_stage()

    runner(prompt="a lighthouse", height=544, width=960, enable_stage_2=True)

    stage_1, stage_2 = pipeline.calls
    assert stage_1["height"] == 544, "stage 1 lost the requested size"
    leaked = sorted(k for k in ("height", "width") if k in stage_2)
    assert not leaked, f"stage 2 was given a size it must infer: {leaked}"


def test_the_audio_latents_reach_stage_2():
    """Stage 1 returns ``(video_latents, audio_latents)``; both must go forward.

    Keeping only ``frames`` means stage 2 starts the audio from scratch, so the
    track that ships is not the one the video was denoised alongside.
    """
    runner, pipeline, _upsampler, _decoder = two_stage()

    runner(prompt="a lighthouse", enable_stage_2=True)

    _stage_1, stage_2 = pipeline.calls
    assert stage_2.get("audio_latents") == FakeTensor("audio-latents"), (
        f"stage 1's audio latents did not reach stage 2: {sorted(stage_2)}"
    )


def test_the_diffusion_decode_is_seeded_with_the_run_generator():
    """The decoder is a diffusion model and draws its own noise.

    An unseeded decode makes two same-seed runs differ for a reason unrelated to
    what they were comparing -- which is what TEST_PLAN A.4.1 relies on.
    """
    runner, _pipeline, _upsampler, decoder = two_stage()

    runner(prompt="a lighthouse", generator="gen-42", enable_stage_2=True)

    assert decoder.kwargs.get("generator") == "gen-42", (
        f"the diffusion decode was left unseeded: {sorted(decoder.kwargs)}"
    )


def test_a_two_stage_audio_run_does_not_return_a_silent_video():
    """The decoder returns video only, and ``outputs`` reads audio with getattr.

    ``LTX2VideoDecodeOutput`` has no ``audio`` field, so returning it directly
    exports a silent file on every audio mode with nothing raised or logged.
    """
    runner, pipeline, _upsampler, _decoder = two_stage()

    result = runner(prompt="a lighthouse", enable_stage_2=True)

    audio = getattr(result, "audio", None)
    assert audio is not None, (
        "the two-stage result carries no audio, so every audio mode would export "
        "a silent file and report success"
    )
    assert pipeline.audio_vae.decoded, "the audio latents were never decoded to a mel"
    assert pipeline.vocoder.mels, "the mel never reached the vocoder"
    assert audio == FakeTensor("waveform<-mel<-audio-latents@bfloat16"), (
        f"the audio did not come from stage 2's latents via audio_vae + vocoder: {audio}"
    )
    assert getattr(result, "frames", None) == FakeTensor("decoded"), (
        "the frames are not the diffusion decoder's output"
    )


def test_a_video_only_two_stage_run_returns_the_decoder_output_unchanged():
    """No audio to reunite means nothing to wrap."""
    runner, _pipeline, _upsampler, decoder = two_stage(audio=False)

    result = runner(prompt="a lighthouse", enable_stage_2=True)

    assert result is decoder._output, "a video-only run was given an audio field it never had"


# --------------------------------------------------------------------------
# what `from_pretrained` is pointed at
#
# `require_cache_complete()` passing does NOT imply "no network needed to
# construct". Those are two claims and only the first was ever asserted: with a
# verified 37/37 cache, loading by repo id still resolved against the Hub and
# began fetching 48 files -- the 44.33 GB of dead shards `model_files` already
# documents as paid for nothing.
# --------------------------------------------------------------------------


class LoadRecorder:
    """Stands in for a diffusers class and records how it was asked to load."""

    def __init__(self, result=None):
        self.calls = []
        self._result = result

    def from_pretrained(self, target, **kwargs):
        self.calls.append({"target": target, **kwargs})
        return self._result if self._result is not None else FakeModel()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeModel()

    @property
    def target(self):
        assert self.calls, "from_pretrained was never called"
        return self.calls[-1]["target"]


class FakeModel:
    """Enough of a loaded component to survive construction."""

    vae = None
    scheduler = "scheduler"
    dtype = "bfloat16"

    def __init__(self):
        self.moved_to = None
        self.processor = None
        self.tiled = False

    def to(self, device):
        self.moved_to = device
        return self

    def set_attn_processor(self, processor):
        self.processor = processor

    def enable_tiling(self):
        self.tiled = True


def install_fake_stack(monkeypatch, snapshot, **classes):
    """Put a fake `torch` and `diffusers` tree on `sys.modules`."""
    import sys
    import types

    from ltx25_studio import preflight

    monkeypatch.setattr(preflight, "snapshot_dir", lambda *a, **k: snapshot)

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"
    monkeypatch.setitem(sys.modules, "torch", torch)

    made = {}
    for name in (
        # `transformers` is faked too, because the int8 path imports
        # `transformers.BitsAndBytesConfig` for the Gemma text encoder. On the
        # `quant_mapping` path each component's config must be the class ITS
        # OWN loader accepts -- diffusers' for the transformer, transformers'
        # for the text encoder -- so the two classes are separate recorders and
        # a test can tell which one reached which component.
        "transformers",
        "diffusers",
        "diffusers.models",
        "diffusers.models.autoencoders",
        "diffusers.models.autoencoders.ltx2_diffusion_decoder",
        "diffusers.pipelines",
        "diffusers.pipelines.ltx2",
        "diffusers.pipelines.ltx2.latent_upsampler",
        "diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode",
    ):
        module = types.ModuleType(name)
        made[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    defaults = {
        "transformers": ("BitsAndBytesConfig",),
        "diffusers": (
            # BOTH, as real diffusers exports both. If the fake omitted the
            # component-level class, a mutation using it would fail on an
            # ImportError rather than on the type assertion -- and a mutant that
            # breaks execution is not a mutant.
            "PipelineQuantizationConfig",
            "BitsAndBytesConfig",
            "LTX2ConditionPipeline",
            "LTX2Pipeline",
            "LTX2LatentUpsamplePipeline",
            "LTX2VideoDiffusionDecoderModel",
        ),
        "diffusers.models.autoencoders.ltx2_diffusion_decoder": (
            "LTX2VideoVaeNeighborhoodNattenProcessor",
        ),
        "diffusers.pipelines.ltx2.latent_upsampler": ("LTX2LatentUpsamplerModel",),
        "diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode": (
            "LTX2VideoDiffusionDecodePipeline",
        ),
    }
    recorders = {}
    for module_name, attrs in defaults.items():
        for attr in attrs:
            # Keyed BOTH ways. `BitsAndBytesConfig` exists in `diffusers` and in
            # `transformers` and they are NOT interchangeable -- the int8 path
            # must hand each component the class its own loader accepts -- so
            # the bare name alone cannot address them. The qualified key is what
            # a test asserting per-component classes uses; the bare key is kept
            # so the other callers here need no change, and for a collision it
            # resolves to the LAST module registered (`diffusers`), which is the
            # one this package means when it says `BitsAndBytesConfig`.
            qualified = f"{module_name}.{attr}"
            recorder = classes.get(qualified) or classes.get(attr) or LoadRecorder()
            recorders[qualified] = recorder
            recorders[attr] = recorder
            setattr(made[module_name], attr, recorder)

    pipelines.clear_cache()
    return recorders


def test_the_base_loads_from_the_local_snapshot_not_the_repo_id(tmp_path, monkeypatch):
    """Loading by repo id resolves against the Hub and refetches the whole repo."""
    from ltx25_studio import model_files

    snapshot = tmp_path / "snapshots" / model_files.REVISION
    snapshot.mkdir(parents=True)
    recorders = install_fake_stack(monkeypatch, snapshot)

    pipelines._base_pipeline()

    target = recorders["LTX2ConditionPipeline"].target
    assert target == snapshot, f"the base was not loaded from the local snapshot: {target}"
    assert model_files.REPO_ID not in str(target), (
        f"the base was loaded by repo id ({target}), which resolves against the Hub and "
        f"refetches 44.33 GB of shards the allow-list exists to exclude"
    )


def test_the_stage_2_parts_load_from_the_local_snapshot_too(tmp_path, monkeypatch):
    """The upsampler and decoder are separate loads with the same failure mode."""
    from ltx25_studio import model_files

    snapshot = tmp_path / "snapshots" / model_files.REVISION
    snapshot.mkdir(parents=True)
    recorders = install_fake_stack(monkeypatch, snapshot)

    pipelines._stage_2_parts(FakeModel())

    for name in ("LTX2LatentUpsamplerModel", "LTX2VideoDiffusionDecoderModel"):
        call = recorders[name].calls[-1]
        assert call["target"] == snapshot, f"{name} was not loaded from the snapshot"
        assert call["subfolder"], f"{name} was loaded without a subfolder"


def test_construction_refuses_when_there_is_no_local_snapshot(tmp_path, monkeypatch):
    """Better to name the missing snapshot than to quietly load by repo id."""
    from ltx25_studio import model_files

    missing = tmp_path / "snapshots" / model_files.REVISION
    install_fake_stack(monkeypatch, missing)

    with pytest.raises(ModelLoadError) as excinfo:
        pipelines._base_pipeline()
    assert "ensure_weights" in str(excinfo.value), (
        f"the error does not say how to get the weights: {excinfo.value}"
    )


def test_the_snapshot_the_loader_uses_is_the_one_the_completeness_check_checks():
    """One derivation, or the guarantee is about a different directory."""
    from ltx25_studio import preflight

    checked = preflight.snapshot_dir(hub_cache="/tmp/hub-cache-probe")
    assert str(checked).startswith("/tmp/hub-cache-probe"), checked
    assert checked.name == pipelines.model_files.REVISION, (
        f"the snapshot path does not end in the pinned revision: {checked}"
    )


# --------------------------------------------------------------------------
# stage 2 is a branch, not a startup cost
#
# Building the parts in `get_pipeline` turned a dependency of ONE branch into a
# hard requirement for all eleven modes: the diffusion decoder resolves a NATTEN
# kernel over the network, and when that failed nothing loaded at all -- eight
# modes that never touch stage 2 included.
# --------------------------------------------------------------------------


def test_getting_a_pipeline_does_not_build_the_stage_2_parts(tmp_path, monkeypatch):
    from ltx25_studio import model_files

    snapshot = tmp_path / "snapshots" / model_files.REVISION
    snapshot.mkdir(parents=True)
    recorders = install_fake_stack(monkeypatch, snapshot)

    pipelines.get_pipeline(device="cpu")

    for name in ("LTX2LatentUpsamplerModel", "LTX2VideoDiffusionDecoderModel"):
        assert not recorders[name].calls, (
            f"{name} was loaded just to hand back a pipeline, so every mode now pays "
            f"stage 2's weights and its network dependency"
        )


def test_a_single_stage_run_never_builds_the_stage_2_parts():
    """The default path must not touch the branch it does not use."""
    built = []
    runner = pipelines.StageRunner(
        Recorder("pipeline"),
        kind="text2video",
        stage_2_factory=lambda: (built.append(1), (Recorder("up"), Recorder("dec")))[1],
    )

    runner(prompt="a lighthouse")

    assert not built, "a single-stage run built the stage-2 parts"


def test_the_stage_2_parts_are_built_once_and_reused():
    """Two stage-2 runs, one build -- or every run pays for the weights again."""
    calls = []

    def factory():
        calls.append(1)
        return Recorder("upsampler", output=Staged("up")), Recorder("decoder", output=Staged("dec"))

    runner = pipelines.StageRunner(
        AudioPipeline(audio=False), kind="text2video", stage_2_factory=factory
    )

    runner(prompt="a", enable_stage_2=True)
    runner(prompt="b", enable_stage_2=True)

    assert len(calls) == 1, f"the stage-2 parts were built {len(calls)} times, not once"


def test_stage_2_still_refuses_when_the_parts_cannot_be_had():
    """Laziness must not become a silent fallback to a single-stage render."""
    runner = pipelines.StageRunner(Recorder("pipeline"), kind="text2video")

    with pytest.raises(ModelLoadError) as excinfo:
        runner(prompt="a lighthouse", enable_stage_2=True)
    assert "single-stage" in str(excinfo.value), (
        f"the refusal no longer explains what it is refusing to do: {excinfo.value}"
    )


# --------------------------------------------------------------------------
# the distilled recipe reaches every run
#
# It used to live on `stage_1_kwargs()`, which is applied ONLY when stage 2 is
# enabled -- so DISTILLED_SIGMA_VALUES was defined, documented as mandatory,
# quoted in the module docstring, and had never been sent to a pipeline until
# the first stage-2 run. Eleven of twelve host runs used upstream's linear
# fallback and the SFT guidance defaults.
# --------------------------------------------------------------------------

#: The distilled recipe as the model card states it, at revision 871165de….
#: *Recorded from the source rather than derived from our own constants, so a
#: change to ours cannot quietly redefine what "correct" means.*
MODEL_CARD_DISTILLED = {
    "guidance_scale": 1.0,
    "audio_guidance_scale": 1.0,
    "stg_scale": 0.0,
    "audio_stg_scale": 0.0,
    "modality_scale": 1.0,
    "audio_modality_scale": 1.0,
}


def test_a_single_stage_run_is_given_the_distilled_schedule():
    """The default path is the one that was missing it."""
    runner, pipeline, _up, _dec = two_stage()

    runner(prompt="a lighthouse")

    sent = pipeline.kwargs.get("sigmas")
    assert sent == list(pipelines.DISTILLED_SIGMA_VALUES), (
        f"a single-stage run sent sigmas={sent}. Without them upstream builds "
        f"linspace(1.0, 1/steps, steps) -- a floor of 0.125 against the distilled "
        f"schedule's 0.421875, which is a different denoising trajectory."
    )


@pytest.mark.parametrize("name,value", sorted(MODEL_CARD_DISTILLED.items()))
def test_a_single_stage_run_carries_every_guidance_value(name, value):
    """Omitting one does not leave it alone -- it selects the SFT default."""
    runner, pipeline, _up, _dec = two_stage()

    runner(prompt="a lighthouse")

    assert pipeline.kwargs.get(name) == value, (
        f"{name} was sent as {pipeline.kwargs.get(name)!r}, not {value!r}. The pipeline "
        f"defaults are the SFT values (stg 1.0, modality 3.0, audio_guidance 7.0), so an "
        f"omitted parameter silently selects a different model's settings."
    )


def test_num_inference_steps_is_not_sent():
    """Upstream builds the linear schedule from it; the card warns against it."""
    runner, pipeline, _up, _dec = two_stage()

    runner(prompt="a lighthouse")

    assert "num_inference_steps" not in pipeline.kwargs, (
        "num_inference_steps was sent alongside an explicit sigma schedule; upstream's "
        "own guidance is that passing it 'would hand the model a generic linear schedule "
        "and quietly cost quality'"
    )


def test_both_stages_share_one_guidance_set():
    """Two places to assemble the recipe is how it diverged the first time."""
    runner, pipeline, _up, _dec = two_stage()

    runner(prompt="a lighthouse", enable_stage_2=True)

    stage_1, stage_2 = pipeline.calls
    for name, value in MODEL_CARD_DISTILLED.items():
        assert stage_1.get(name) == value, f"stage 1 lost {name}"
        assert stage_2.get(name) == value, f"stage 2 lost {name}"
    assert stage_2["sigmas"] == list(pipelines.STAGE_2_DISTILLED_SIGMA_VALUES), (
        "stage 2 must use the three-step tail, not stage 1's schedule"
    )


def test_our_recipe_matches_the_model_card():
    """A recorded fact, not a restatement of the constant it checks."""
    assert dict(pipelines.DISTILLED_GUIDANCE) == MODEL_CARD_DISTILLED, (
        f"the guidance set has drifted from the published recipe: "
        f"{dict(pipelines.DISTILLED_GUIDANCE)} vs {MODEL_CARD_DISTILLED}"
    )


def test_no_advertised_lever_sets_guidance_or_steps():
    """A lever attached to nothing is a defect; one attached to harm is worse."""
    from ltx25_studio.ui.fields import FIELD_KEYS

    for key in ("guidance_scale", "num_inference_steps"):
        assert key not in FIELD_KEYS, (
            f"{key!r} is still a UI field, but the distilled recipe fixes it -- a user "
            f"changing it fights the schedule and gets worse output with no way to know why"
        )


# --------------------------------------------------------------------------


def test_the_schedule_is_not_truncated_for_a_partially_noised_run():
    """⏸️ Pins current behaviour, and records that it is an OPEN question.

    `noise_scale` and `sigmas[0]` are the same quantity to upstream, and ours
    disagree on the audio-input modes. It was diagnosed as the cause of a
    silent `audio_to_audio` and it was **not** — the silence was a
    spectrally-sparse fixture, and the mode is healthy at −10.4 dBFS carrying
    the full mismatch.

    This test exists so a future change is deliberate rather than incidental:
    truncating would take an eight-step run to **two** steps at 0.5 and **one**
    at 0.1, values that measured healthy on eight. If someone makes that change
    they should have evidence it improves something, and this failing is the
    prompt to go and get it.
    """
    runner, pipeline, _up, _dec = two_stage()

    runner(prompt="a lighthouse", noise_scale=0.5)

    assert pipeline.kwargs["sigmas"] == list(pipelines.DISTILLED_SIGMA_VALUES), (
        "the schedule is now truncated for a partially-noised run. That may be right, "
        "but it needs a measurement showing it improves output -- agreeing with the "
        "contract is not evidence, and it costs six of eight denoise steps at 0.5."
    )


# --------------------------------------------------------------------------
# the quantization lever is CONNECTED, not merely present
#
# A test that inspects the source of `_quantization_config` asserts the
# function looks right. It passes with the `load_kwargs["quantization_config"]`
# line deleted -- and then the Space deploys bf16 while the variable says int8,
# evicts because bf16 does not fit, and the conclusion drawn is "int8 does not
# fit" from a run in which int8 was never applied. A wrong conclusion from a
# real experiment, costing a 72 GB deploy cycle, looking exactly like a
# legitimate negative result.
#
# This is ladder rung 4 (PLAN 9.1.6): we can prove a function ran; this proves
# its result was used.
# --------------------------------------------------------------------------


def _loaded(monkeypatch, tmp_path, quantization):
    from ltx25_studio import config, model_files

    snapshot = tmp_path / "snapshots" / model_files.REVISION
    snapshot.mkdir(parents=True)
    monkeypatch.setenv(config.ENV_QUANTIZATION, quantization)
    recorders = install_fake_stack(monkeypatch, snapshot)

    pipelines._base_pipeline()

    return recorders["LTX2ConditionPipeline"].calls[-1], recorders


def test_int8_reaches_from_pretrained(monkeypatch, tmp_path):
    """The whole point of the deploy: the variable must change the load."""
    sent, _recorders = _loaded(monkeypatch, tmp_path, "int8")

    assert "quantization_config" in sent, (
        f"LTX25_QUANTIZATION=int8 and from_pretrained was called with {sorted(sent)}. "
        f"The Space would deploy bf16 while the variable says int8, evict because bf16 "
        f"does not fit, and the eviction would read as 'int8 does not fit'."
    )
    assert sent["quantization_config"] is not None


def test_bf16_sends_no_quantization_config(monkeypatch, tmp_path):
    """The default must be the baseline every measurement was taken against."""
    sent, _recorders = _loaded(monkeypatch, tmp_path, "bf16")

    assert "quantization_config" not in sent, (
        f"the bf16 default sent a quantization_config: {sorted(sent)}"
    )


def test_the_dtype_is_still_sent_when_quantising(monkeypatch, tmp_path):
    """The a14ebe8 lesson: an omitted dtype is not a neutral omission."""
    sent, _recorders = _loaded(monkeypatch, tmp_path, "int8")

    assert sent.get("dtype") is not None, (
        "quantising dropped the dtype; diffusers then resolves it to float32"
    )
    assert "torch_dtype" not in sent, "the pinned diffusers deprecates torch_dtype for dtype"


def test_the_load_still_comes_from_the_local_snapshot_when_quantising(monkeypatch, tmp_path):
    """Quantising must not reintroduce the 44.33 GB repo-id refetch."""
    from ltx25_studio import model_files

    sent, _recorders = _loaded(monkeypatch, tmp_path, "int8")

    assert model_files.REPO_ID not in str(sent["target"]), (
        f"the quantised load went to the repo id ({sent['target']})"
    )


def test_the_config_is_the_pipeline_level_class_built_granularly(monkeypatch, tmp_path):
    """The rung past "the argument arrived": **is it the type the callee takes?**

    A bare `BitsAndBytesConfig` reaches `from_pretrained` exactly as well as the
    right one — the kwargs test cannot tell them apart — and the pipeline then
    rejects it on the Space::

        ValueError: `quantization_config` must be an instance of
        `PipelineQuantizationConfig`.

    Component-level `from_pretrained` takes the former; the pipeline-level one
    takes the latter. That cost a deploy cycle.

    ⚠️ **And it must be built granularly.** This asserted
    `quant_backend == "bitsandbytes_8bit"` while that blanket form was shipping
    garbage frames — a green test on a broken deploy, because a backend name is
    not a configuration. `quant_backend` and `quant_mapping` are mutually
    exclusive upstream (`_validate_init_args` raises if both are given), so the
    absence of `quant_backend` is exactly the assertion that the per-component
    path is live.
    """
    _sent, recorders = _loaded(monkeypatch, tmp_path, "int8")

    built = recorders["PipelineQuantizationConfig"].calls
    assert built, (
        "the pipeline-level quantisation class was never constructed; a bare "
        "BitsAndBytesConfig reaches from_pretrained and is rejected there"
    )
    assert "quant_mapping" in built[-1], (
        f"the pipeline quantisation config was built with {sorted(built[-1])}; expected "
        f"a `quant_mapping`. The blanket `quant_backend` form quantises the "
        f"transformer's conditioning path and renders static."
    )
    assert "quant_backend" not in built[-1], (
        f"both `quant_mapping` and `quant_backend` were passed ({sorted(built[-1])}); "
        f"upstream's `_validate_init_args` raises on that combination"
    )


def test_the_quantised_components_are_exactly_the_declared_set(monkeypatch, tmp_path):
    """The value that reaches the pipeline, not merely that a value does.

    ⚠️ **A wrong name here would not raise.** `PipelineQuantizationConfig` does
    no validation — `pipe_quant_config.py:171` is a bare membership test — so a
    typo'd component silently loads at full precision. `_quantization_config`
    guards the names; this asserts the ones that actually arrive.

    **Widened from `["transformer"]` deliberately.** Transformer-only was right
    while there was no measured baseline: adding the text encoder would have
    confounded an int8-vs-bf16 comparison. The live Space then supplied one —
    55.89 GiB peak RSS, 60.0 GB steady — and a measured baseline turns that
    confound into a control. If this test goes red, someone has changed the
    deployed precision, and that needs to be on purpose.
    """
    _sent, recorders = _loaded(monkeypatch, tmp_path, "int8")

    built = recorders["PipelineQuantizationConfig"].calls
    assert built, "the pipeline-level class was not constructed at all"
    # The expected value is written out, NOT read from QUANTIZED_COMPONENTS. A
    # message that interpolates the constant under test tracks any mutation of
    # it and can never contradict it -- it printed "reached as ['transformer'],
    # not the declared ['transformer']" before this was fixed.
    #
    # On the `quant_mapping` path the MAPPING KEYS are what decides this;
    # `components_to_quantize` is not consulted at all when a mapping is given
    # (`_resolve_quant_config` takes the granular branch on `module_name in
    # quant_mapping` and returns before ever reading it).
    assert sorted(built[-1]["quant_mapping"]) == ["text_encoder", "transformer"], (
        f"the quant_mapping reached from_pretrained covering "
        f"{sorted(built[-1]['quant_mapping'])}, expected "
        f"['text_encoder', 'transformer']. Changing the deployed precision must "
        f"be deliberate."
    )


class _ConfigRecorder:
    """A config class that returns something identifiable rather than a shared fake.

    `LoadRecorder.__call__` returns a fresh `FakeModel`, which records the call
    but makes every constructed config indistinguishable from every other. That
    is not enough here: the invariant is *which* class built the object that
    landed under *which* mapping key, and `FakeModel() is not FakeModel()` can
    only ever say "different", never "from the right class".
    """

    def __init__(self, tag):
        self.tag = tag
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return f"{self.tag}#{len(self.calls)}"


def _int8_mapping(monkeypatch, tmp_path):
    """Build the int8 config with the two `BitsAndBytesConfig` classes distinguishable."""
    from ltx25_studio import config, model_files

    snapshot = tmp_path / "snapshots" / model_files.REVISION
    snapshot.mkdir(parents=True)
    monkeypatch.setenv(config.ENV_QUANTIZATION, "int8")
    overrides = {
        "diffusers.BitsAndBytesConfig": _ConfigRecorder("diffusers"),
        "transformers.BitsAndBytesConfig": _ConfigRecorder("transformers"),
    }
    recorders = install_fake_stack(monkeypatch, snapshot, **overrides)

    pipelines._base_pipeline()

    built = recorders["PipelineQuantizationConfig"].calls
    assert built, "the pipeline-level quantisation class was never constructed"
    return built[-1]["quant_mapping"], overrides


def test_each_component_gets_the_config_class_its_own_loader_accepts(monkeypatch, tmp_path):
    """diffusers' class for the transformer, transformers' class for the encoder.

    ⚠️ **The granular branch ignores `is_diffusers`.** `pipeline_loading_utils`
    calls `_resolve_quant_config(is_diffusers=is_diffusers_model, module_name=name)`,
    but when a `quant_mapping` is given that method returns `quant_mapping[module_name]`
    **verbatim** — it never looks at `is_diffusers`. So there is no layer left to
    correct a wrong class: whatever object sits under a key is handed straight to
    that component's own `from_pretrained`.

    `text_encoder` is a Gemma model loaded by transformers, so it needs
    `transformers.BitsAndBytesConfig`; the transformer is diffusers-owned and
    needs diffusers'. Upstream's docs say it outright — "You need to import and
    use `transformers.BitsAndBytesConfig` for components that come from
    Transformers." Swapping them is invisible to every kwargs-level assertion
    and fails only on the Space.
    """
    mapping, _overrides = _int8_mapping(monkeypatch, tmp_path)

    assert mapping["transformer"].startswith("diffusers#"), (
        f"the transformer's quantisation config was built by "
        f"{mapping['transformer']!r}; it is a diffusers-owned component and must get "
        f"`diffusers.BitsAndBytesConfig`"
    )
    assert mapping["text_encoder"].startswith("transformers#"), (
        f"the text encoder's quantisation config was built by "
        f"{mapping['text_encoder']!r}; Gemma is loaded by transformers and must get "
        f"`transformers.BitsAndBytesConfig`"
    )


def test_the_transformer_skips_the_conditioning_path_and_the_encoder_does_not(
    monkeypatch, tmp_path
):
    """The actual bug fix, asserted where it is configured.

    Every frame the Space ever produced was correctly-shaped, high-variance
    garbage because this list was absent and bnb replaced **every** `nn.Linear`
    in the transformer — the adaLN/modulation heads and the input/output
    projections included. Deterministic static at every resolution and in every
    mode is broken *conditioning*, not lost precision.

    **The text encoder must NOT get the list, and that is not an oversight.**
    It is a Gemma LLM, which is the exact case `LLM.int8()` was designed and
    validated for and is not implicated in this bug. The names here are
    transformer module names; on a Gemma they would match nothing, and
    `to_gate_logits` is precisely the kind of short leaf name that is a no-op on
    one tree and a catastrophe on another. Per-component configs are the whole
    reason this is a `quant_mapping`.

    This asserts the *wiring*. `test_int8_skip_modules.py` asserts the names
    themselves against the real checkpoint's module tree.
    """
    _mapping, overrides = _int8_mapping(monkeypatch, tmp_path)

    transformer_calls = overrides["diffusers.BitsAndBytesConfig"].calls
    encoder_calls = overrides["transformers.BitsAndBytesConfig"].calls
    assert len(transformer_calls) == 1, transformer_calls
    assert len(encoder_calls) == 1, encoder_calls

    assert transformer_calls[0].get("load_in_8bit") is True, transformer_calls[0]
    assert encoder_calls[0].get("load_in_8bit") is True, encoder_calls[0]

    skips = transformer_calls[0].get("llm_int8_skip_modules")
    assert skips, (
        "the transformer's 8-bit config named no `llm_int8_skip_modules`. That is the "
        "blanket config that rendered static at every resolution, in every mode, "
        "deterministically."
    )
    # A list, not the tuple constant: `BitsAndBytesConfig.post_init` raises
    # `TypeError: llm_int8_skip_modules must be a list of strings` on anything
    # else, and `bnb_quantizer` then `.extend()`s it in place -- so it must also
    # be a fresh copy rather than the module-level constant itself.
    assert isinstance(skips, list), f"llm_int8_skip_modules must be a list, got {type(skips)}"
    assert skips is not pipelines.TRANSFORMER_INT8_SKIP_MODULES, (
        "the module-level constant was passed through by identity; bnb_quantizer "
        "mutates this list in place, so repeated loads would grow it"
    )
    assert skips == list(pipelines.TRANSFORMER_INT8_SKIP_MODULES)

    assert "llm_int8_skip_modules" not in encoder_calls[0], (
        f"the Gemma text encoder was given transformer module names to skip "
        f"({encoder_calls[0].get('llm_int8_skip_modules')}). Blanket LLM.int8() is "
        f"exactly what that component wants; these names match nothing on it."
    )


def test_the_load_log_names_every_component_it_quantised(monkeypatch, tmp_path):
    """The line the RAM verdict will be read off, on both paths.

    It said `loaded transformer as int8` while TWO components were quantised.
    Not silent -- wrong in the direction that misdirects: a new peak-RSS number
    beside the word "transformer" invites attributing the whole change to the
    transformer, and that is the number deciding whether the RAM hypothesis
    survives. `expected_weights_gb` in miniature.

    The log DERIVES its list from `QUANTIZED_COMPONENTS`; this test writes the
    expectation OUT. That asymmetry is deliberate and is not the mistake the
    sibling test warns about -- a test must assert an independent expectation,
    while a log must report what was requested, so only one of the two may move
    on its own. Widening the constant updates the log and turns this red, which
    is the point: the format is an instrument, and instruments change on
    purpose.
    """
    with caplog_at_info() as records:
        _loaded(monkeypatch, tmp_path / "int8", "int8")
    int8_line = _rss_line(records)
    assert int8_line.startswith("quantised transformer, text_encoder as int8"), (
        f"the int8 load logged {int8_line!r}; expected it to open with "
        f"'quantised transformer, text_encoder as int8' so the peak RSS beside "
        f"it cannot be misread as the transformer's alone"
    )

    with caplog_at_info() as records:
        _loaded(monkeypatch, tmp_path / "bf16", "bf16")
    bf16_line = _rss_line(records)
    # The bf16 path quantises NOTHING. Saying "quantised ... as bf16" there
    # would be a false statement in the same log the verdict is read from.
    assert bf16_line.startswith("no components quantised; all at bf16"), (
        f"the bf16 load logged {bf16_line!r}; expected 'no components "
        f"quantised; all at bf16' -- bf16 is not a quantisation"
    )
    assert "quantised transformer" not in bf16_line, (
        f"the bf16 log claims a component was quantised: {bf16_line!r}"
    )


@contextlib.contextmanager
def caplog_at_info():
    """Capture records off the studio logger without depending on caplog's
    propagation settings, which `install_fake_stack` does not control."""
    records = []

    class _Sink(logging.Handler):
        def emit(self, record):
            records.append(record)

    sink = _Sink()
    logger = logging.getLogger("ltx25_studio")
    previous = logger.level
    # `setLevel`, NOT `logger.level = ...`. Assigning the attribute leaves
    # `Logger._cache` stale, so `isEnabledFor(INFO)` keeps returning whatever it
    # answered under the old level and the record is dropped before any handler
    # sees it. Only `setLevel` invalidates that cache. This cost a red that
    # looked like the log line was missing when it was being emitted fine.
    logger.setLevel(logging.INFO)
    logger.addHandler(sink)
    try:
        yield records
    finally:
        logger.removeHandler(sink)
        logger.setLevel(previous)


def _rss_line(records):
    hits = [r.getMessage() for r in records if "peak RSS" in r.getMessage()]
    assert len(hits) == 1, f"expected exactly one peak-RSS line, got {hits!r}"
    return hits[0]


def test_an_unknown_component_name_is_refused_rather_than_ignored(monkeypatch):
    """Upstream ignores unknown names silently; we must not.

    A component that is never matched loads at full precision, the deploy uses
    more memory than intended, and nothing says so — the same shape as the
    `LTX25_TEXT_ENCODER_PRECISION` variable that was wired to nothing.
    """
    monkeypatch.setattr(pipelines, "QUANTIZED_COMPONENTS", ("transformer", "text_encodr"))

    with pytest.raises(ModelLoadError) as excinfo:
        pipelines._quantization_config("int8")

    message = str(excinfo.value)
    assert "text_encodr" in message and "silently" in message, message


# --------------------------------------------------------------------------
# the disk line reports our footprint, not the host's free space
#
# `shutil.disk_usage(".")` was here. On the Space it printed "1969.8 GB free of
# 8589.9 GB" -- the HOST filesystem, 8.6 TB -- while the workload was evicted
# for exceeding 150 GB. It said that through two evictions and looked like a
# check the whole time. A report attached to nothing is a lever attached to
# nothing, with the added harm that a reader believes it.
# --------------------------------------------------------------------------


def test_the_disk_line_measures_against_the_cap(monkeypatch):
    monkeypatch.setattr(pipelines, "_directory_bytes", lambda path: 0)

    line = pipelines.disk_footprint()

    assert "150 GB cap" in line, f"the cap is not named: {line}"
    assert "free of" not in line, (
        f"the line still reports filesystem free space: {line}. That quantity is "
        f"~8.6 TB on the Space and cannot see the 150 GB the platform enforces."
    )


def test_it_says_OVER_when_our_footprint_exceeds_the_cap(monkeypatch):
    """The state it was silent through, twice."""
    monkeypatch.setattr(pipelines, "_directory_bytes", lambda path: 72_185_000_000)

    line = pipelines.disk_footprint()

    assert "[OVER]" in line, (
        f"snapshot + packed at 72.2 GB each plus the image is 154.4 GB against a "
        f"150 GB cap, and the line does not say so: {line}"
    )


def test_it_names_all_three_terms(monkeypatch):
    """Including the image, which is the term that pushes bf16 over.

    72.2 + 72.2 is 144.4 and fits; it is the image that does not.
    """
    monkeypatch.setattr(pipelines, "_directory_bytes", lambda path: 1_000_000_000)

    line = pipelines.disk_footprint()

    for term in ("snapshot", "packed", "image"):
        assert term in line, f"{term!r} is not broken out: {line}"


def test_the_startup_report_does_not_use_filesystem_free_space():
    """Pinned because the replacement looks like the thing it replaced.

    Both print a number followed by 'GB'. Only one of them can fail.
    """
    # AST, not a text search: the docstring of `disk_footprint` names
    # `shutil.disk_usage` to explain what was removed and why, and a substring
    # scan cannot tell an explanation from a call. Same lesson as a scan that
    # reported "duplicate definitions" that were source inside string literals.
    import ast

    tree = ast.parse(pathlib.Path(pipelines.__file__).read_text())
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "disk_usage" not in called, (
        "shutil.disk_usage is back in pipelines.py. It reports the host filesystem, "
        "which on the Space is 8.6 TB and has no relationship to the 150 GB quota "
        "the platform enforces by watching the workload."
    )


def test_the_footprint_says_it_is_a_floor(monkeypatch):
    """A number presented as complete when it is a floor is how the old line
    survived two evictions: it looked like a total."""
    monkeypatch.setattr(pipelines, "_directory_bytes", lambda path: 0)

    line = pipelines.disk_footprint()

    assert "FLOOR" in line, f"the line does not declare itself a floor: {line}"
    assert "NOT measurable" in line, (
        f"the line does not say the image is invisible from inside: {line}"
    )


def test_no_footprint_line_is_labelled_as_a_post_pack_measurement():
    """ZeroGPU packs at ``demo.launch()`` (spaces ``startup()`` -> ``pack()``),
    not inside ``.to(device)``: the old "after packing" line read an empty
    offload directory six seconds before the platform began packing, and
    printed ``packed 0.0`` under a label that said the opposite."""
    import re

    assert "after packing" not in pipelines.disk_footprint()
    phases = re.findall(r'disk_footprint\(phase="([^"]*)"\)', inspect.getsource(pipelines))
    assert phases, "the second footprint line is gone"
    assert all("after packing" not in phase and "before launch" in phase for phase in phases), (
        phases
    )
    assert "before launch" in pipelines.disk_footprint()


def test_an_empty_packed_copy_is_not_printed_as_a_measurement(monkeypatch):
    monkeypatch.setattr(pipelines, "_directory_bytes", lambda path: 0)

    line = pipelines.disk_footprint()

    assert "packed 0.0" not in line, line
    assert "not written yet" in line, line


def test_the_second_footprint_still_comes_after_the_device_move():
    """It answers "what does the disk hold once the base has moved" -- a real,
    pre-launch question -- and its label says exactly that."""
    source = inspect.getsource(pipelines._base_pipeline)
    move = source.index("built.to(device)")
    report = source.index("disk_footprint(phase=")

    assert move < report


def test_the_cap_is_one_constant_not_a_number_in_four_docstrings():
    """It was quoted in three docstrings before it was a constant, and making it
    a constant without pointing them at it made a fourth place to diverge."""
    import pathlib as _pathlib

    for module in ("pipelines.py", "config.py"):
        text = (_pathlib.Path(pipelines.__file__).parent / module).read_text()
        bare = [
            line
            for line in text.splitlines()
            if "150 GB" in line and "ZEROGPU_DISK_CAP_GB" not in line
        ]
        assert not bare, (
            f"{module} states the cap as a bare number: {bare}. It is "
            f"ZEROGPU_DISK_CAP_GB, and a number in two places is a number that "
            f"will diverge."
        )
