"""A memory refusal is said as one (F73, item 3).

``memory.MemoryPreflightError`` inherited the root's category, "bug", and
``gpu_used = True``, so the session's refused run read "Something went wrong in
this app — Any GPU time this run used is billed. Press Generate again." It has
its own category now, appended to ``errors.CATEGORIES`` the way "limit" and
"queue" were: ``"memory"``, this machine's memory check declined a step and
nothing was loaded for it, so no GPU time was used.

Pinned for a refusal raised at every step of a run by the real staged runner on
the tiny checkpoint, through ``generate()``, and across the dispatch boundary as
``test_errors_fork_boundary.py`` pins the other errors: the same class, category
and words on the far side of a pickle round trip, bare and in spaces 0.52's
wrapper.
"""

from __future__ import annotations

import html
import pickle

import pytest

from ltx25_studio import errors, memory
from ltx25_studio import generate as generate_module

SIZE = 64
FRAMES = 9
SEED = 1234
PROMPT = "a red ball bouncing on a wooden floor"

#: The category a card is keyed on (a string, so the UI needs no import), and no GPU time.
MEMORY_REFUSAL = errors.Failure("memory", False)

#: The session's refused run, in the words a user reads.
SESSION_WORDS = (
    "Not enough memory for 'load transformer': 67.3 GB is already in use and this step needs "
    "about 25.1 GB more, 92.4 GB in all, which is over the 90.0 GB limit by 2.4 GB. Nothing "
    "was loaded. Close other apps, pick a smaller size or fewer frames, or raise "
    "LTX25_MEMORY_LIMIT_GB if you have measured that this machine can take it."
)


def _session_refusal():
    return memory.MemoryPreflightError(
        memory.preflight("load transformer", need_gb=25.1, limit_gb=90.0, committed_gb=67.3)
    )


# --------------------------------------------------------------------------
# the category
# --------------------------------------------------------------------------


def test_memory_is_its_own_category_appended_after_the_others():
    assert errors.MEMORY == "memory"
    assert errors.CATEGORIES == (
        "quota",
        "timeout",
        "oom",
        "input",
        "adapter",
        "enhancer",
        "bug",
        "limit",
        "queue",
        "memory",
    )


@pytest.mark.parametrize(
    "error",
    [
        _session_refusal(),
        memory.MemoryPreflightError(reason="could not run vm_stat to measure memory: (stand-in)"),
    ],
    ids=["over the limit", "no reading"],
)
def test_a_memory_refusal_is_categorised_as_memory_with_no_gpu_time_used(error):
    assert (error.category, error.gpu_used) == ("memory", False)
    assert errors.categorise(error) == MEMORY_REFUSAL


def test_the_refusal_keeps_its_words_and_the_users_knob():
    assert str(_session_refusal()) == SESSION_WORDS


# --------------------------------------------------------------------------
# a refusal at each step of a run, from the real staged runner
# --------------------------------------------------------------------------


class ArmedGauge:
    """The gauge reads 200 GB while one step is judged and 30 GB otherwise.

    ``_LocalWeights._preflight`` is wrapped only to say which step is being
    judged; the judgement, any reading again and the refusal are the real ones.
    The clock moves only when the preflight sleeps.
    """

    def __init__(self, monkeypatch, step):
        from ltx25_studio import local_runtime

        self.step = step
        self.armed = False
        self.now = 0.0
        real = local_runtime._LocalWeights._preflight

        def judged(weights, phase, need_gb):
            self.armed = phase.startswith(step)
            try:
                return real(weights, phase, need_gb)
            finally:
                self.armed = False

        monkeypatch.setattr(local_runtime._LocalWeights, "_preflight", judged)
        monkeypatch.setattr(memory, "committed_gb", lambda **_: 200.0 if self.armed else 30.0)
        monkeypatch.setattr(memory, "_clock", lambda: self.now, raising=False)
        monkeypatch.setattr(memory, "_sleep", self.sleep, raising=False)
        monkeypatch.setenv("LTX25_MEMORY_LIMIT_GB", "90")
        monkeypatch.delenv("LTX25_MEMORY_SETTLE_SECONDS", raising=False)

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def tiny_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from ltx25_studio import local_runtime

    local_runtime.clear_local_cache()
    yield local_runtime
    local_runtime.clear_local_cache()


@pytest.fixture(scope="module")
def composite(tmp_path_factory):
    """The tiny checkpoint plus tiny stage-2 models, as one snapshot directory."""
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from local_fakes import tiny_snapshot
    from test_real_pipeline_tiny import load_smoke

    path, missing = load_smoke().composite_snapshot(
        tiny_snapshot(), tmp_path_factory.mktemp("composite") / "snapshot", fetch=False
    )
    if missing:
        pytest.skip(missing)
    return path


def _tiny_adapter(tmp_path):
    import torch
    from local_fakes import tiny_snapshot
    from test_real_pipeline_tiny import load_smoke

    from ltx25_studio import local_runtime

    adapter = tmp_path / "look.safetensors"
    donor = local_runtime.load_transformer(tiny_snapshot(), dtype=torch.float32, device="cpu")
    load_smoke().make_lora(adapter, donor)
    return {"source": str(adapter), "path": str(adapter), "weight": 1.0, "report": object()}


def _request(mode, tmp_path, **fields):
    from test_real_pipeline_tiny import load_smoke

    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    request = dict(mode=mode, seed=SEED, width=SIZE, height=SIZE, num_frames=FRAMES)
    request.update(load_smoke()._media_for(mode, media, SIZE, FRAMES))
    request.update(fields)
    return request


def _factory(local_runtime, snapshot):
    from local_fakes import install_tiny_audio_encode_shim

    def factory(*, pipeline, **_):
        runner = local_runtime.build_local_runner(
            kind=pipeline,
            device="cpu",
            dtype="float32",
            snapshot_dir=snapshot,
            residency="staged",
        )
        install_tiny_audio_encode_shim(runner.pipeline.audio_vae)
        return runner

    return factory


#: step -> (the phase it is judged as, mode, extra request fields)
STEPS = {
    "small set": ("load small set", "text_to_video", {}),
    "text encoder": ("load text encoder", "text_to_video", {}),
    "transformer": ("load transformer", "text_to_video", {}),
    "adapter attach": ("attach LoRA look.safetensors", "text_to_video", {"loras": "adapter"}),
    "clip encode": (f"encode clip {SIZE}x{SIZE}x{FRAMES}", "video_to_video", {}),
    "denoise": (f"denoise {SIZE}x{SIZE}x{FRAMES}", "text_to_video", {}),
    "stage-2 load": (
        "load latent_upsampler and diffusion_decoder",
        "image_to_video",
        {"enable_stage_2": True},
    ),
}


@pytest.mark.parametrize("step", list(STEPS))
def test_a_refusal_at_each_step_of_a_run_is_categorised_as_memory(
    step, tiny_runtime, tmp_path, monkeypatch, request
):
    from local_fakes import tiny_snapshot

    phase, mode, fields = STEPS[step]
    snapshot = tiny_snapshot()
    if step == "stage-2 load":
        snapshot = request.getfixturevalue("composite")
        # The stage-2 parts resolve through the snapshot directory.
        monkeypatch.setenv("LTX25_MODEL_SNAPSHOT_DIR", str(snapshot))
    if fields.get("loras") == "adapter":
        pytest.importorskip("peft")
        fields = {"loras": (_tiny_adapter(tmp_path),)}
    ArmedGauge(monkeypatch, phase)

    with pytest.raises(memory.MemoryPreflightError) as caught:
        generate_module.generate(
            **_request(mode, tmp_path, **fields),
            pipeline_factory=_factory(tiny_runtime, snapshot),
            output_root=tmp_path / "out",
            retention_bytes=10**12,
        )

    error = caught.value
    assert str(error).startswith(f"Not enough memory for '{phase}': 200.0 GB is already in use")
    assert "Nothing was loaded." in str(error) and "LTX25_MEMORY_LIMIT_GB" in str(error)
    assert errors.categorise(error) == MEMORY_REFUSAL


# --------------------------------------------------------------------------
# across the dispatch boundary
# --------------------------------------------------------------------------


class ZeroGPUTaskException(Exception):
    """spaces 0.52's wrapper: the worker's own error rides on ``.exception``."""

    def __init__(self, exception):
        super().__init__(type(exception).__name__)
        self.exception = exception


def _crossed(exc, *, wrapped):
    """What the parent raises: the worker's error after the result queue's pickle trip."""
    back = pickle.loads(pickle.dumps(exc))
    return ZeroGPUTaskException(back) if wrapped else back


@pytest.mark.parametrize("wrapped", [False, True], ids=["spaces 0.51", "spaces 0.52"])
def test_a_memory_refusal_crosses_the_dispatch_with_its_category_and_words(
    monkeypatch, tmp_path, wrapped
):
    raised = _session_refusal()

    def dispatch(**_kwargs):
        raise _crossed(raised, wrapped=wrapped)

    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
    with pytest.raises(memory.MemoryPreflightError) as caught:
        generate_module.generate(mode="text_to_video", prompt="p", output_root=tmp_path)

    arrived = caught.value
    assert type(arrived) is memory.MemoryPreflightError and arrived is not raised
    assert str(arrived) == SESSION_WORDS
    assert arrived.result == raised.result
    assert errors.categorise(arrived) == MEMORY_REFUSAL


def test_the_card_shows_the_refusal_word_for_word_on_both_sides_of_the_boundary():
    """The card's headline is the UI's; its detail must be the refusal, before and after."""
    from ltx25_studio.ui import render

    raised = _session_refusal()
    rendered = render.failure(raised)

    assert SESSION_WORDS in rendered or html.escape(SESSION_WORDS) in rendered
    assert render.failure(_crossed(raised, wrapped=False)) == rendered


def test_a_refusal_raised_by_the_local_runner_in_the_worker_reaches_the_caller_as_itself(
    tiny_runtime, tmp_path, monkeypatch
):
    from local_fakes import tiny_snapshot

    ArmedGauge(monkeypatch, "load transformer")
    real = generate_module.generate

    def dispatch(**kwargs):
        token = generate_module._IN_GPU_WORKER.set(True)
        try:
            return real(**kwargs)
        except BaseException as exc:
            raise _crossed(exc, wrapped=True) from None
        finally:
            generate_module._IN_GPU_WORKER.reset(token)

    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
    with pytest.raises(memory.MemoryPreflightError) as caught:
        generate_module.generate(
            **_request("text_to_video", tmp_path),
            pipeline_factory=_factory(tiny_runtime, tiny_snapshot()),
            output_root=tmp_path / "out",
            retention_bytes=10**12,
        )

    assert str(caught.value).startswith(
        "Not enough memory for 'load transformer': 200.0 GB is already in use"
    )
    assert errors.categorise(caught.value) == MEMORY_REFUSAL
