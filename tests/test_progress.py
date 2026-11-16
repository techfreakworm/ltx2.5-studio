"""Spec for sampling progress — the third of the operator's three original defects.

**There was none, and a 66 s render looked exactly like a hung one.** The only
signal was a status line written *before* the run started, so "working" and
"wedged" rendered identically for the whole of a metered call.

This file exists because the obvious ways to build this feature all fail
silently, and a progress bar that lies is worse than no progress bar — this
build has nine inert levers on record and will not ship a tenth. Three of those
silent failures are pinned here by name:

* **the handle is never injected.** Gradio finds a ``gr.Progress`` by walking
  the wired function's signature and stopping at the first non-positional
  parameter, so the natural spelling — ``(mode_id, *values, progress=...)`` —
  gets no handle at all and every call below becomes a no-op. Nothing raises.
* **the handle never crosses the fork.** ``callback_on_step_end`` fires inside
  the ``@spaces.GPU`` worker. A handle dropped by the dispatch reports the two
  phases that happen in the web process and then goes quiet for the entire
  render, which is the part the user is waiting on.
* **the callback is never wired.** ``StageRunner`` is the only layer that knows
  how many denoise passes a run has. If it does not pass a callback, the bar
  sits at "loading the model" for 60 s and then jumps to done.

Offline by construction: no GPU, no weights, no network. Gradio is a dev
dependency and is driven directly here, because "will Gradio inject this?" is
the question a comment cannot answer.
"""

from __future__ import annotations

import pytest
from test_exports_contract import MODE_IDS, FakeOutput, one_muxed_file_encoder

from ltx25_studio import generate as generate_module
from ltx25_studio import modes, pipelines

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


class Recorder:
    """A ``gr.Progress``-shaped handle that records every call."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, position, desc=None):
        self.calls.append((position, desc))

    @property
    def labels(self) -> list[str]:
        return [desc for _position, desc in self.calls]

    @property
    def fractions(self) -> list[float]:
        return [p for p, _ in self.calls if isinstance(p, (int, float))]


class SteppingPipe:
    """A denoise that actually calls the callback it is handed, like the real one.

    ``num_timesteps`` is the attribute upstream sets as ``self._num_timesteps =
    len(timesteps)`` before the loop, and it is deliberately NOT 8 — the declared
    schedule length — so a reporter that quietly counts to the constant instead
    of asking the pipeline is visible as a wrong denominator rather than as a
    coincidence.
    """

    num_timesteps = 5

    def __init__(self):
        from conftest import FakeAudioVae

        self.calls: list[dict] = []
        self.audio_vae = FakeAudioVae()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        if callback is not None:
            for step in range(self.num_timesteps):
                returned = callback(self, step, 0.5, {})
                assert isinstance(returned, dict), (
                    f"upstream does `callback_outputs.pop('latents', latents)` on this "
                    f"return value; {returned!r} is an AttributeError on every step"
                )
        return FakeOutput()


def run(mode_id, tmp_path, monkeypatch, *, progress=None, encoder=None, **overrides):
    """One real ``generate()`` through a real ``StageRunner`` over a fake denoise.

    The runner is REAL on purpose: the sampling callback is built by
    ``StageRunner``, not by ``generate()``, so a bare fake pipeline in its place
    would skip the entire mechanism this file is about and every assertion below
    would pass against nothing.
    """
    from conftest import STUB_SAMPLING_RATE, STUB_WAVEFORM, audio_file

    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )
    inner = SteppingPipe()
    runner = pipelines.StageRunner(inner, kind="condition")
    # A REAL mp4 for the video modes: `conditioning.decode_video_frames` decodes a
    # video condition before the pipeline sees it, so a path-shaped string there
    # is refused — the same fixture rule `test_ui_generate_seam` records. A REAL
    # wav for the audio modes, for the same reason: generate() opens it with av
    # before the dispatch.
    from test_ui_generate_seam import conditioning_video

    real = {"video": conditioning_video, "audio": audio_file}
    media = {
        kind: (real[kind]() if kind in real else f"conditioning-input.{kind}")
        for kind in modes.get_mode(mode_id).inputs
    }
    media.pop("text", None)
    result = generate_module.generate(
        mode=mode_id,
        seed=7,
        prompt="a lighthouse in fog",
        pipeline_factory=lambda **_kwargs: runner,
        encoder=encoder if encoder is not None else one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
        progress=progress,
        **media,
        **overrides,
    )
    return result, inner


# --------------------------------------------------------------------------
# the injection — the one that fails with no error at all
# --------------------------------------------------------------------------


def test_gradio_actually_injects_a_progress_handle_into_the_wired_handler():
    """The question a comment cannot answer, asked of Gradio itself.

    ``helpers.special_args`` collects positional parameters and ``break``s at the
    first one that is not — so ``*values`` ends the walk, and a ``progress``
    parameter declared after it is never found. The wrapper therefore takes the
    handle FIRST, which looks wrong and is the only spelling that works. If a
    future edit "tidies" the order, this fails; without it, the whole feature
    silently reverts to no progress at all.
    """
    from gradio.helpers import special_args

    from ltx25_studio.ui import blocks
    from ltx25_studio.ui.fields import FIELD_KEYS

    inputs = ["text_to_video", *range(len(FIELD_KEYS))]
    request = object()
    updated, progress_index, _event_index, _props = special_args(
        blocks._on_generate, inputs, request=request
    )

    assert progress_index is not None, (
        "Gradio found no gr.Progress in the wired handler's signature, so it will inject "
        "none and every progress call in generate() is a no-op that raises nothing"
    )
    assert isinstance(updated[progress_index], gr.Progress)
    # The request rides in the next slot, so the finish step can find this run.
    assert updated[progress_index + 1] is request
    assert updated[progress_index + 2] == "text_to_video"


def test_the_wrapper_still_forwards_the_mode_and_the_fields_in_order():
    """The handle takes slot 0, so the marshalling is one off-by-one from wrong."""
    from ltx25_studio.ui import blocks
    from ltx25_studio.ui.fields import FIELD_KEYS

    seen: dict = {}

    def fake_run(mode_id, *values, progress=None):
        seen.update(mode_id=mode_id, values=values, progress=progress)
        return blocks.handlers.RunOutcome(video=None, audio=None, status="ok", mode_id=mode_id)

    original = blocks.handlers.run
    blocks.handlers.run = fake_run
    try:
        handle = object()
        returned = blocks._on_generate(handle, None, "audio_to_video", *range(len(FIELD_KEYS)))
    finally:
        blocks.handlers.run = original

    assert seen["mode_id"] == "audio_to_video"
    assert seen["values"] == tuple(range(len(FIELD_KEYS)))
    assert seen["progress"] is handle
    assert returned == (None, None, "ok"), "the endpoint returns (video, audio, status)"


def test_the_published_endpoint_is_still_on_generate():
    """The wrapper renamed the wired function; ``api_name`` must undo that.

    Gradio names an endpoint after the function it wires, so wrapping
    ``handlers.on_generate`` would have published ``/_on_generate`` and broken
    every existing client — ``scripts/space_matrix.py`` first among them.
    """
    from ltx25_studio.ui import build_ui

    endpoints = build_ui().get_api_info()["named_endpoints"]

    assert "/on_generate" in endpoints, sorted(endpoints)
    assert "/_on_generate" not in endpoints, (
        "the wrapper's private name leaked into the published API"
    )
    assert [r["component"] for r in endpoints["/on_generate"]["returns"]] == [
        "Video",
        "Audio",
        "Markdown",
    ], "the frozen (video, audio, markdown) return contract changed"


def test_building_the_ui_leaves_existing_tqdm_bars_working():
    """F33: ``gr.Progress(track_tqdm=True)`` patched ``tqdm`` for the whole process.

    A default argument built inside ``build_ui()`` ran ``patch_tqdm()`` at every
    boot, and a bar that already existed -- the enhancer prefetch's download bar,
    started before the UI is built -- then raised ``AttributeError`` on its next
    ``update``. Driven here with a real tqdm created before the build.
    """
    import io

    from tqdm import tqdm

    from ltx25_studio.ui import build_ui

    bar = tqdm(total=3, file=io.StringIO())
    build_ui()
    bar.update(1)
    bar.update(1)
    bar.close()

    created_after = tqdm(total=2, file=io.StringIO())
    created_after.update(2)
    created_after.close()


def test_no_ui_event_turns_on_tqdm_tracking():
    """Every Progress default the UI declares leaves tqdm alone."""
    import inspect

    from ltx25_studio.ui import blocks

    for fn in (blocks._on_generate, blocks._on_validate_lora):
        for parameter in inspect.signature(fn).parameters.values():
            if isinstance(parameter.default, gr.Progress):
                assert not parameter.default.track_tqdm, f"{fn.__name__} tracks tqdm"


# --------------------------------------------------------------------------
# the handle actually travels
# --------------------------------------------------------------------------


def test_the_handler_passes_the_handle_down_without_importing_gradio():
    from ltx25_studio.ui import handlers

    seen: dict = {}
    original = generate_module.generate
    generate_module.generate = lambda **params: seen.update(params) or original(**params)
    try:
        from test_ui_generate_seam import values_for

        handlers.on_generate("text_to_video", *values_for("text_to_video"), progress="handle")
    finally:
        generate_module.generate = original

    assert seen.get("progress") == "handle"


def test_the_handle_crosses_the_gpu_dispatch(tmp_path, monkeypatch):
    """The phase that matters happens on the far side of the fork.

    Sampling runs inside the ``@spaces.GPU`` worker. A dispatch that drops the
    handle reports "loading the model", goes silent for the whole render and
    then reports "writing the file" — which is the shape of a hang, not of
    progress.
    """
    captured: list = []

    def dispatch(**kwargs):
        captured.append(kwargs)
        token = generate_module._IN_GPU_WORKER.set(True)
        try:
            return generate_module.generate(**kwargs)
        finally:
            generate_module._IN_GPU_WORKER.reset(token)

    recorder = Recorder()
    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
    run("text_to_video", tmp_path, monkeypatch, progress=recorder)

    assert captured and captured[0].get("progress") is recorder, (
        f"the dispatch payload was {sorted(captured[0]) if captured else 'never sent'}; "
        f"the handle never reached the worker"
    )
    assert any("sampling" in label for label in recorder.labels), (
        f"nothing reported from inside the worker: {recorder.labels}"
    )


# --------------------------------------------------------------------------
# what the user sees
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_every_mode_reports_its_phases_and_every_sampling_step(mode_id, tmp_path, monkeypatch):
    """Cover every mode, because a mode that reports nothing is the defect.

    All eleven run a denoise — the audio-only modes still render video, which is
    the whole of PLAN §3.1b — so there is no mode for which "no sampling
    progress" is the right answer.
    """
    recorder = Recorder()
    _result, inner = run(mode_id, tmp_path, monkeypatch, progress=recorder)

    assert "loading the model" in recorder.labels
    assert "preparing the inputs" in recorder.labels
    assert "writing the file" in recorder.labels
    assert inner.calls[-1].get("callback_on_step_end") is not None, (
        f"{mode_id!r} reached the denoise with no callback, so the long phase of the run "
        f"reports nothing: {sorted(inner.calls[-1])}"
    )
    assert recorder.labels.count("sampling") == inner.num_timesteps + 1, (
        f"{mode_id!r} reported {recorder.labels.count('sampling')} sampling updates for "
        f"{inner.num_timesteps} steps (one announcement plus one per step): {recorder.labels}"
    )


def test_the_step_total_is_read_off_the_pipeline_not_off_the_schedule(tmp_path, monkeypatch):
    """A denominator this app writes down is a denominator that can be wrong.

    The declared schedule is 8 steps; the fake reports 5. The final sampling
    update must land at the END of the sampling band, which it only can if the
    reporter divided by the pipeline's own ``num_timesteps``.
    """
    recorder = Recorder()
    run("text_to_video", tmp_path, monkeypatch, progress=recorder)

    sampling = [p for p, desc in recorder.calls if desc == "sampling"]
    _start, end = generate_module._PHASE_BANDS["sampling"]

    assert sampling[-1] == pytest.approx(end), (
        f"the last of {len(sampling) - 1} steps landed at {sampling[-1]} rather than {end}. "
        f"Dividing by the declared {len(generate_module.constraints.DISTILLED_SIGMA_VALUES)} "
        f"would leave the bar short of its own phase boundary."
    )


def test_progress_never_goes_backwards_across_a_whole_run(tmp_path, monkeypatch):
    """A bar that retreats reads as a restart, and a restart reads as a crash."""
    recorder = Recorder()
    run("text_to_video", tmp_path, monkeypatch, progress=recorder, enhance_prompt=False)

    fractions = recorder.fractions
    assert fractions == sorted(fractions), f"the bar went backwards: {recorder.calls}"
    assert fractions[-1] == pytest.approx(generate_module._PHASE_BANDS["writing the file"][0])


def test_the_enhancement_phase_is_announced_before_the_enhancer_runs(tmp_path, monkeypatch):
    """~21 s of dead air the user currently stares at, now labelled.

    Announced BEFORE the call, because a label that appears once the phase is
    over reports history rather than progress.
    """
    recorder = Recorder()
    order: list[str] = []

    def enhancer(prompt, **_kwargs):
        order.append("enhancer ran")
        return "a rewritten prompt"

    original = generate_module.ProgressReporter.phase

    def spy(self, label):
        order.append(label)
        return original(self, label)

    monkeypatch.setattr(generate_module.ProgressReporter, "phase", spy)
    run(
        "text_to_video",
        tmp_path,
        monkeypatch,
        progress=recorder,
        enhance_prompt=True,
        enhancer=enhancer,
    )

    assert "enhancing the prompt" in order, order
    assert order.index("enhancing the prompt") < order.index("enhancer ran"), order


def test_a_run_that_enhances_nothing_does_not_claim_to_be_enhancing(tmp_path, monkeypatch):
    """A phase that fires when it did not happen is how a bar stops being read."""
    recorder = Recorder()
    run("text_to_video", tmp_path, monkeypatch, progress=recorder)

    assert "enhancing the prompt" not in recorder.labels, recorder.labels


# --------------------------------------------------------------------------
# stage 2 — a second sampling pass, which is what makes a bar look like a crash
# --------------------------------------------------------------------------


def test_a_generation_that_produces_nothing_still_fails_loudly_with_progress_on(
    tmp_path, monkeypatch
):
    """Fail-loudly is untouchable, and a reporter that swallows is next to it.

    :class:`generate.ProgressReporter` swallows its OWN failures so a broken bar
    cannot kill a render. That is one exception's width away from swallowing the
    render's failure too, which is the §5.3 outage exactly: a run that produced
    nothing, reported as though it had.
    """
    from ltx25_studio.errors import EmptyOutputError

    recorder = Recorder()
    with pytest.raises(EmptyOutputError) as caught:
        run(
            "text_to_video",
            tmp_path,
            monkeypatch,
            progress=recorder,
            encoder=lambda *a, **k: None,
        )

    assert "text_to_video" in str(caught.value)
    assert recorder.labels, "the run failed and reported no progress at all either"


def test_the_two_denoise_passes_are_labelled_as_two_passes():
    """A bar that silently restarts at 0% reads as a crash.

    Stage 2 IS a second denoise over the same machinery and its steps genuinely
    start again at one. The label is the only thing that turns "it crashed and
    started over" into "it is on the second of two passes", so it is asserted
    rather than left to the reader of ``StageRunner``.
    """
    from test_stage_graph_contract import make_runner

    recorder = Recorder()
    reporter = generate_module.ProgressReporter(recorder)
    runner, pipeline, _upsampler, _decoder = make_runner()

    runner(prompt="a lighthouse", enable_stage_2=True, progress=reporter)

    assert "sampling — pass 1 of 2" in recorder.labels, recorder.labels
    assert "sampling — pass 2 of 2" in recorder.labels, recorder.labels
    assert "sampling" not in recorder.labels, (
        "a two-pass run also emitted the single-pass label, so the user cannot tell which "
        "pass a restarting bar belongs to"
    )
    assert len(pipeline.calls) == 2
    assert all("callback_on_step_end" in call for call in pipeline.calls), (
        f"a denoise pass ran with no callback: {[sorted(c) for c in pipeline.calls]}"
    )


def test_the_two_stage_steps_between_the_passes_are_named():
    """The upsample and the diffusion decode are not instantaneous.

    Leaving the user on a finished stage-1 bar for the length of both is the
    same dead air the whole feature is against.
    """
    from test_stage_graph_contract import make_runner

    recorder = Recorder()
    runner, _pipeline, _upsampler, _decoder = make_runner()

    runner(
        prompt="a lighthouse",
        enable_stage_2=True,
        progress=generate_module.ProgressReporter(recorder),
    )

    assert "upscaling the latents" in recorder.labels, recorder.labels
    assert "decoding the frames" in recorder.labels, recorder.labels
    assert recorder.labels.index("upscaling the latents") < recorder.labels.index(
        "sampling — pass 2 of 2"
    )


def test_a_two_stage_bar_never_retreats_even_though_its_steps_restart():
    """The steps restart; the bar must not. That is what the bands buy."""
    from test_stage_graph_contract import make_runner

    recorder = Recorder()
    runner, _pipeline, _upsampler, _decoder = make_runner()

    runner(
        prompt="a lighthouse",
        enable_stage_2=True,
        progress=generate_module.ProgressReporter(recorder),
    )

    fractions = recorder.fractions
    assert fractions == sorted(fractions), recorder.calls


def test_the_progress_kwarg_is_consumed_and_never_reaches_diffusers():
    """``progress`` is ours, exactly like ``enable_stage_2``.

    Forwarding it would be a TypeError from inside diffusers on every run with a
    UI attached — obvious in production, invisible to a fake that takes
    ``**kwargs``.
    """
    from test_stage_graph_contract import make_runner

    runner, pipeline, _upsampler, _decoder = make_runner()
    runner(prompt="a lighthouse", progress=generate_module.ProgressReporter(Recorder()))

    assert "progress" not in pipeline.kwargs, (
        f"progress was passed through to the pipeline: {sorted(pipeline.kwargs)}"
    )


def test_a_run_with_nobody_listening_hands_the_pipeline_exactly_what_it_used_to():
    """A no-op reporter must not add a ``callback_on_step_end=None`` either.

    The house rule is that a ``None`` in the kwargs is how an unset value becomes
    a confusing TypeError several layers away, and the offline gate should be
    handed the same call it was handed before this feature existed.
    """
    from test_stage_graph_contract import make_runner

    runner, pipeline, _upsampler, _decoder = make_runner()
    runner(prompt="a lighthouse", progress=generate_module.ProgressReporter(None))

    assert "callback_on_step_end" not in pipeline.kwargs, sorted(pipeline.kwargs)


# --------------------------------------------------------------------------
# it may never break a generation
# --------------------------------------------------------------------------


def test_a_progress_handle_that_raises_cannot_kill_the_generation(tmp_path, monkeypatch):
    """The ``_VramProbe`` rule: a diagnostic that can fail a metered run is worse
    than the problem it was added for.

    A ``gr.Progress`` call reaches the browser through a queue in another
    process; there are more ways for that to fail than for it to work, and none
    of them is worth a user's render.
    """

    class Exploding:
        def __init__(self):
            self.calls = 0

        def __call__(self, position, desc=None):
            self.calls += 1
            raise RuntimeError("the queue went away")

    handle = Exploding()
    result, _inner = run("text_to_video", tmp_path, monkeypatch, progress=handle)

    assert result.output_path.exists()
    assert handle.calls == 1, (
        f"the reporter kept calling a handle that had already failed ({handle.calls} times); "
        f"it must disable itself on the first failure like the VRAM probe does"
    )


def test_an_unknown_step_total_reports_the_step_and_claims_nothing_else():
    """No denominator anywhere is a real state, and it must not be filled in.

    Gradio renders ``(index, None)`` as a step count with an unknown total. A
    guessed total would render a lie, which is the one thing a progress bar
    cannot do and stay useful.
    """
    recorder = Recorder()
    reporter = generate_module.ProgressReporter(recorder)

    class Countless:
        num_timesteps = None

    # Cleared BEFORE the callback is built: the fallback is read once, when the
    # pass is announced, so a later edit would prove nothing.
    declared = generate_module._DECLARED_STEPS.copy()
    try:
        generate_module._DECLARED_STEPS.clear()
        callback = reporter.sampling()
        callback(Countless(), 3, 0.5, {})
    finally:
        generate_module._DECLARED_STEPS.update(declared)

    assert recorder.calls[-1] == ((4, None), "sampling"), recorder.calls
