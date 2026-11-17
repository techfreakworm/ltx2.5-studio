"""The diffusion decoder is built once, at start-up, with the other parts (#17).

On the Space the default decode is the diffusion decoder. It used to be built
lazily inside each GPU worker and lost with it, so every run loaded its 0.83 GB
and resolved its NATTEN kernel again. ``app.py`` now builds it at start-up, after
the pipeline and before ``demo.launch()``, so its ``.to("cuda")`` is registered
for packing with the rest; each worker's runner finds that instance through its
decoder factory. A failed start-up build logs one line and leaves today's per-run
build in place.

Two tiers: the CUDA path against the fake diffusers tree the stage-graph spec
uses (``.to("cuda")``, the NATTEN processor), and the worker path end to end on
the real tiny checkpoint, through ``generate()``.
"""

from __future__ import annotations

import ast
import logging

import pytest
from conftest import REPO_ROOT
from test_stage_graph_contract import FakeModel, LoadRecorder, install_fake_stack

from ltx25_studio import config, model_files, pipelines

DECODER = "LTX2VideoDiffusionDecoderModel"


@pytest.fixture(autouse=True)
def _no_cached_pipelines():
    pipelines.clear_cache()
    yield
    pipelines.clear_cache()


# --------------------------------------------------------------------------
# the CUDA path, on the fake diffusers tree
# --------------------------------------------------------------------------


class DecodePipelineRecorder:
    """``LTX2VideoDiffusionDecodePipeline``: records each build, returns a marked object."""

    def __init__(self):
        self.built = []

    def __call__(self, **kwargs):
        made = FakeModel()
        made.parts = kwargs
        self.built.append(made)
        return made


class NattenOnce:
    """The NATTEN processor: the first construction fails as a Hub outage would."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Cannot reach the Hub to resolve shi-labs/natten")
        return "natten-processor"


@pytest.fixture
def space(tmp_path, monkeypatch):
    """The ZeroGPU start-up's pieces on the fake tree: a cache holding the decoder at
    its own revision, the two runners ``app.py`` builds, and the recorders."""

    def build(**classes):
        monkeypatch.delenv(config.ENV_MODEL_SNAPSHOT_DIR, raising=False)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        repo = tmp_path / f"models--{model_files.REPO_ID.replace('/', '--')}" / "snapshots"
        snapshot = repo / model_files.REVISION
        snapshot.mkdir(parents=True)
        (repo / model_files.DIFFUSION_DECODER_REVISION / "diffusion_decoder").mkdir(parents=True)
        model = FakeModel()
        decode = DecodePipelineRecorder()
        base = FakeModel()
        base.components = {}  # what `derive_pipeline` shares into the text2video runner
        recorders = install_fake_stack(
            monkeypatch,
            snapshot,
            LTX2ConditionPipeline=LoadRecorder(result=base),
            LTX2VideoDiffusionDecoderModel=LoadRecorder(result=model),
            LTX2VideoDiffusionDecodePipeline=decode,
            **classes,
        )
        runners = {
            kind: pipelines.get_pipeline(pipeline=kind, device="cuda")
            for kind in ("condition", "text2video")
        }
        return model, decode, recorders, runners

    return build


def test_the_startup_build_loads_the_decoder_once_and_moves_it_to_cuda(space, caplog):
    model, decode, recorders, _runners = space()
    assert not recorders[DECODER].calls, "building the runners must still not load the decoder"

    with caplog.at_level(logging.INFO, logger="ltx25_studio"):
        built = pipelines.build_decoder_at_startup(device="cuda")

    assert built is True
    assert len(recorders[DECODER].calls) == 1
    assert model.moved_to == "cuda", "the .to('cuda') that ZeroGPU registers for packing"
    assert model.processor is not None, "the NATTEN processor is set at start-up too"
    assert model.tiled is True
    assert len(decode.built) == 1
    lines = [r.getMessage() for r in caplog.records if "diffusion decoder" in r.getMessage()]
    assert any("built at start-up" in line and "GiB" in line for line in lines), lines


def test_each_runner_finds_the_startup_decoder_instead_of_building_one(space):
    _model, decode, recorders, runners = space()
    pipelines.build_decoder_at_startup(device="cuda")
    startup_decoder = decode.built[0]

    for runner in runners.values():
        decoder, why_conv = runner._single_stage_decoder("auto", {})
        assert (decoder, why_conv) == ("diffusion", None)
        assert runner.decoder is startup_decoder

    assert len(recorders[DECODER].calls) == 1, "a run loaded the decoder again"
    assert len(decode.built) == 1


def test_a_second_startup_call_builds_nothing_more(space):
    _model, decode, recorders, _runners = space()

    assert pipelines.build_decoder_at_startup(device="cuda") is True
    assert pipelines.build_decoder_at_startup(device="cuda") is True

    assert len(recorders[DECODER].calls) == 1 and len(decode.built) == 1


def test_a_failed_startup_build_logs_one_line_and_the_run_builds_its_own(space, caplog):
    natten = NattenOnce()
    _model, decode, recorders, runners = space(
        LTX2VideoVaeNeighborhoodNattenProcessor=natten
    )

    with caplog.at_level(logging.INFO, logger="ltx25_studio"):
        built = pipelines.build_decoder_at_startup(device="cuda")

    assert built is False
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    message = warnings[0].getMessage()
    assert "NOT built at start-up" in message and "each run builds its own" in message, message
    assert "Cannot reach the Hub" in message and "\n" not in message, message
    assert not decode.built, "a half-built decoder was kept"

    decoder, why_conv = runners["text2video"]._single_stage_decoder("auto", {})

    assert (decoder, why_conv) == ("diffusion", None), "the per-run build is gone"
    assert len(recorders[DECODER].calls) == 2 and len(decode.built) == 1


def test_a_failed_startup_build_never_raises(space, monkeypatch):
    """Whatever goes wrong, start-up goes on: the Space must come up as before."""
    _model, _decode, _recorders, _runners = space()

    def broken(*args, **kwargs):
        raise ImportError("kernels is not installed")

    monkeypatch.setattr(pipelines, "_decoder_part", broken)

    assert pipelines.build_decoder_at_startup(device="cuda") is False


def test_the_app_builds_the_decoder_at_module_scope_after_the_pipeline():
    """Where ``app.py`` asks for it: in the CUDA branch, after both runners, at module
    scope -- inside ``@spaces.GPU`` it would never be packed."""
    tree = ast.parse((REPO_ROOT / "app.py").read_text())
    branch = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and "get_pipeline" in ast.unparse(node)
    )
    calls = [
        ast.unparse(statement.value)
        for statement in branch.body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    ]
    assert calls[-1] == "pipelines.build_decoder_at_startup(device='cuda')", calls
    assert sum("get_pipeline" in call for call in calls[:-1]) == 2, calls


def test_the_startup_report_names_the_decoder_beside_the_pipeline(monkeypatch):
    from ltx25_studio import device, zerogpu

    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: True)
    monkeypatch.setenv(config.ENV_DEVICE, "cuda")
    runtime_config = config.load_config()
    report = pipelines.startup_report(
        runtime_config, missing_files=(), resolution=device.resolve(None, runtime_config)
    )

    line = next(line for line in report.splitlines() if line.strip().startswith("runner"))
    assert "diffusion decoder" in line and "start" in line, line


# --------------------------------------------------------------------------
# the worker path, end to end, on the real tiny checkpoint
# --------------------------------------------------------------------------


def _tiny_composite(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from test_real_pipeline_tiny import _tiny_snapshot_cached, load_smoke

    if not _tiny_snapshot_cached():
        pytest.skip(f"{model_files.TINY_REPO_ID} is not in the HF cache")
    smoke = load_smoke()
    snapshot, missing = smoke.composite_snapshot(
        smoke.tiny_snapshot(fetch=False), tmp_path_factory.mktemp("tiny") / "snapshot", fetch=False
    )
    if missing:
        pytest.skip(missing)
    return snapshot


@pytest.fixture(scope="module")
def tiny_composite_dir(tmp_path_factory):
    return _tiny_composite(tmp_path_factory)


@pytest.fixture
def tiny_space(tiny_composite_dir, monkeypatch):
    """``app.py``'s start-up on the tiny checkpoint, on the CPU: both runners, then the
    decoder. Records every decoder load and every decode call."""
    from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode import (
        LTX2VideoDiffusionDecodePipeline,
    )

    monkeypatch.setenv(config.ENV_MODEL_SNAPSHOT_DIR, str(tiny_composite_dir))
    # The process's device, as LTX25_DEVICE sets it: the worker's own runner lookup
    # (`get_pipeline` with no device) resolves it.
    monkeypatch.setenv(config.ENV_DEVICE, "cpu")
    loads, decodes = [], []
    real_load = pipelines._load_part

    def counting_load(cls, component, **kwargs):
        if component == "diffusion_decoder":
            loads.append(component)
        return real_load(cls, component, **kwargs)

    monkeypatch.setattr(pipelines, "_load_part", counting_load)
    real_call = LTX2VideoDiffusionDecodePipeline.__call__

    def recording_call(self, *args, **kwargs):
        decodes.append(self)
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(LTX2VideoDiffusionDecodePipeline, "__call__", recording_call)
    for kind in ("condition", "text2video"):
        pipelines.get_pipeline(pipeline=kind, device="cpu")
    return loads, decodes


def _run(tmp_path, name):
    """One run as the GPU worker makes it: inside the worker context ``app.run_generation``
    sets (so no dispatch another test left registered can intercept it), through
    ``generate()`` and its default runner lookup (``pipelines.get_pipeline``), no factory
    injected. 128x128x9 is the tiny decoder's minimum and within the CPU decode bound."""
    from ltx25_studio import generate

    token = generate._IN_GPU_WORKER.set(True)
    try:
        return generate.generate(
            mode="text_to_video",
            prompt="a lighthouse at dusk",
            seed=7,
            width=128,
            height=128,
            num_frames=9,
            frame_rate=24.0,
            decoder="diffusion",
            output_root=tmp_path / name,
            retention_bytes=10**9,
        )
    finally:
        generate._IN_GPU_WORKER.reset(token)


def test_tiny_the_decoder_is_built_once_and_every_run_uses_that_instance(tiny_space, tmp_path):
    loads, decodes = tiny_space

    assert pipelines.build_decoder_at_startup(device="cpu") is True
    startup_decoder = pipelines._CACHE["__decoder__"]
    first = _run(tmp_path, "one")
    second = _run(tmp_path, "two")

    assert loads == ["diffusion_decoder"], f"the decoder was loaded {len(loads)} times"
    assert decodes == [startup_decoder, startup_decoder]
    assert first.output_path.is_file() and second.output_path.is_file()


def test_tiny_a_failed_startup_build_falls_back_to_the_per_run_build(
    tiny_space, tmp_path, monkeypatch, caplog
):
    loads, decodes = tiny_space
    real_part = pipelines._decoder_part
    failures = []

    def fails_at_startup(*args, **kwargs):
        if not failures:
            failures.append(1)
            raise OSError("the decoder's file could not be read")
        return real_part(*args, **kwargs)

    monkeypatch.setattr(pipelines, "_decoder_part", fails_at_startup)

    with caplog.at_level(logging.WARNING, logger="ltx25_studio"):
        assert pipelines.build_decoder_at_startup(device="cpu") is False
    result = _run(tmp_path, "one")

    fell_back = [
        r for r in caplog.records if r.getMessage().startswith("diffusion decoder NOT built")
    ]
    assert len(fell_back) == 1, [r.getMessage() for r in caplog.records]
    assert loads == ["diffusion_decoder"], "the run did not build its own decoder"
    assert len(decodes) == 1 and decodes[0] is pipelines._CACHE["__decoder__"]
    assert result.output_path.is_file()
