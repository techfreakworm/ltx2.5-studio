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
    """"Refusing to silently fall back to a single-stage render, which would look
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
    """"Refusing to continue with nothing" — the never-empty rule, mid-graph.

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


def test_the_startup_report_names_the_resolved_precision(runtime_config):
    """"So the running configuration is never inferred."

    A report that omits the precision leaves the single most expensive
    configuration question — what dtype are we actually running — to guesswork.
    """
    report = pipelines.startup_report(runtime_config)
    assert "bf16" in report, f"the report does not state the resolved precision:\n\n{report}"


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
        runtime_config, cached_files=None, missing_files=(), gpu_size=None
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
