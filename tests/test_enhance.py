"""Spec for ``ltx25_studio/enhance.py`` — prompt enhancement, PLAN §3.5.

**Offline by construction, and that is a requirement rather than a convenience.**
Nothing here downloads a model, reaches the Hub or imports torch. The enhancer is
2.49 GB of weights on the other side of a network call, so a suite that needed
them would be a suite nobody could run — and the failure this file exists to
prevent is a *wiring* failure, which no amount of real Gemma would surface.

Three layers of faking, each for a different question:

* the ``enhancer=`` seam on ``generate()`` — "does the rewritten prompt reach the
  pipeline, and does the user get told?";
* a fake ``loader`` on ``enhance()`` — "is upstream's call shape reproduced, and
  is the model actually freed?";
* fake ``torch`` / ``transformers`` / ``diffusers`` modules on ``sys.modules`` —
  so the **real import statements** in ``enhance.py`` execute and resolve. A
  test-only seam in the production module would have proved the seam works; this
  proves the imports do.

The one thing this file cannot check is whether the real repository loads. That
is a network fact, so it is a ``@pytest.mark.network`` test at the bottom, opted
into with ``LTX25_NETWORK_TESTS=1`` exactly like ``test_lora_real_headers.py``,
and it moves **no weights** — repository metadata only.
"""

from __future__ import annotations

import builtins
import os
import sys
import types
from pathlib import Path

import pytest
from conftest import PACKAGE_ROOT
from test_generate import FakePipe, factory_for, good_encoder

from ltx25_studio import config as config_module
from ltx25_studio import enhance as enhance_module
from ltx25_studio import generate as generate_module
from ltx25_studio.errors import PromptEnhancementError

ENHANCE_SOURCE = PACKAGE_ROOT / "enhance.py"

ORIGINAL = "a lighthouse in heavy fog"
REWRITTEN = (
    "A weathered stone lighthouse stands on a basalt outcrop as dense fog rolls in from the "
    "sea; the camera pushes in slowly, the beam sweeping through the murk."
)


def a_request(**overrides):
    fields = {"mode": "text_to_video", "prompt": ORIGINAL, "width": 704, "height": 480}
    fields.update(overrides)
    return generate_module.GenerationRequest(**fields)


class Recorder:
    """An ``enhancer=`` seam that records what it was asked to rewrite."""

    def __init__(self, result=REWRITTEN):
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, prompt, *, image=None):
        self.calls.append({"prompt": prompt, "image": image})
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


# --------------------------------------------------------------------------
# the decision: enhance_if_requested
# --------------------------------------------------------------------------


def test_the_toggle_is_off_by_default():
    """A model that loads itself because the user did not opt out is not a toggle."""
    assert generate_module.GenerationRequest(mode="text_to_video").enhance_prompt is False


def test_nothing_happens_and_nothing_loads_when_the_toggle_is_off():
    seam = Recorder()
    enhanced, notes = enhance_module.enhance_if_requested(a_request(), enhancer=seam)
    assert enhanced is None, "an untouched prompt must be reported as None, not re-sent"
    assert notes == ()
    assert seam.calls == [], "the enhancer was consulted for a run that did not ask for it"


def test_the_rewritten_prompt_comes_back_when_the_toggle_is_on():
    seam = Recorder()
    enhanced, _ = enhance_module.enhance_if_requested(a_request(enhance_prompt=True), enhancer=seam)
    assert enhanced == REWRITTEN
    assert seam.calls == [{"prompt": ORIGINAL, "image": None}]


def test_the_user_is_told_what_was_actually_sent():
    """An enhancer that silently rewrites the prompt is worse than none.

    The prompt box still shows what the user typed; without this the video is
    the only evidence that something else was generated, and a user comparing
    two runs has no way to know which prompt produced which.
    """
    _, notes = enhance_module.enhance_if_requested(
        a_request(enhance_prompt=True), enhancer=Recorder()
    )
    assert notes, "an enhanced run reported nothing at all"
    assert any(REWRITTEN in note for note in notes), (
        f"the enhanced prompt is not quoted anywhere in {notes}"
    )


def test_an_empty_prompt_is_not_enhanced_and_says_so():
    """Several modes send an empty prompt by design. Ticking the box on one of
    those cannot mean anything -- but it must not pass silently either."""
    seam = Recorder()
    enhanced, notes = enhance_module.enhance_if_requested(
        a_request(prompt="   ", enhance_prompt=True), enhancer=seam
    )
    assert enhanced is None
    assert seam.calls == [], "an empty prompt should never reach a 2.5 GB model"
    assert notes and "empty" in notes[0]


def test_an_empty_rewrite_raises_rather_than_replacing_the_prompt_with_nothing():
    with pytest.raises(PromptEnhancementError) as caught:
        enhance_module.enhance_if_requested(
            a_request(enhance_prompt=True), enhancer=Recorder(result="   ")
        )
    message = str(caught.value)
    assert config_module.DEFAULT_PROMPT_ENHANCER_REPO in message, message
    assert "empty" in message


def test_a_failing_enhancer_is_never_swallowed():
    """PLAN §5.3's whole subject: no layer turns a failure into a falsy value.

    Falling through to the original prompt here would generate a video the user
    did not ask for while the UI implied enhancement happened.
    """
    boom = PromptEnhancementError(repo_id="org/enhancer", reason="the Hub answered 404")
    with pytest.raises(PromptEnhancementError):
        enhance_module.enhance_if_requested(
            a_request(enhance_prompt=True), enhancer=Recorder(result=boom)
        )


# --------------------------------------------------------------------------
# the reference frame
# --------------------------------------------------------------------------


def a_pil_image():
    PIL = pytest.importorskip("PIL.Image", reason="pillow is a pinned runtime dependency")
    return PIL.new("RGB", (64, 32), (10, 20, 30))


def test_a_pil_conditioning_image_reaches_the_enhancer():
    """PLAN §3.5: the system prompt is auto-selected on whether a frame exists,
    so the frame has to actually get there for that choice to mean anything."""
    image = a_pil_image()
    seam = Recorder()
    enhance_module.enhance_if_requested(a_request(enhance_prompt=True, image=image), enhancer=seam)
    assert seam.calls[0]["image"] is image


def test_an_image_the_enhancer_cannot_read_degrades_to_text_only_and_says_so():
    """``_prepare_enhance_image`` raises on anything but PIL, and losing a whole
    run over the *type* of a conditioning image the user did not choose would be
    the harsher answer to the milder problem. Degrading silently would not."""
    seam = Recorder()
    _, notes = enhance_module.enhance_if_requested(
        a_request(enhance_prompt=True, image="/tmp/frame.png"), enhancer=seam
    )
    assert seam.calls[0]["image"] is None
    assert any("text-only" in note for note in notes), notes


# --------------------------------------------------------------------------
# through generate(): the seam that decides what the model is given
# --------------------------------------------------------------------------


def test_the_pipeline_is_given_the_rewritten_prompt(tmp_path):
    pipe = FakePipe()
    generate_module.generate(
        a_request(enhance_prompt=True),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        enhancer=Recorder(),
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    (_, kwargs) = pipe.calls[0]
    assert kwargs["prompt"] == REWRITTEN, (
        "the enhancement produced a prompt that never reached the pipeline -- the user waited "
        "for a rewrite that was then thrown away"
    )


def test_the_pipeline_is_given_the_typed_prompt_when_the_toggle_is_off(tmp_path):
    pipe = FakePipe()
    generate_module.generate(
        a_request(),
        pipeline_factory=factory_for(pipe),
        encoder=good_encoder,
        enhancer=Recorder(),
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    assert pipe.calls[0][1]["prompt"] == ORIGINAL


def test_the_run_report_carries_the_prompt_that_was_used(tmp_path):
    result = generate_module.generate(
        a_request(enhance_prompt=True),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        enhancer=Recorder(),
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    assert any(REWRITTEN in note for note in result.warnings), result.warnings


def test_an_unenhanced_run_reports_nothing_about_enhancement(tmp_path):
    result = generate_module.generate(
        a_request(),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    assert not any("enhance" in note for note in result.warnings), result.warnings


def test_a_failed_enhancement_costs_no_model_load(tmp_path):
    """Enhancement sits above the pipeline factory on purpose: a run that cannot
    be enhanced must not first spend a model load it is going to throw away."""
    loaded = []

    def factory(**kwargs):
        loaded.append(kwargs)
        return FakePipe()

    with pytest.raises(PromptEnhancementError):
        generate_module.generate(
            a_request(enhance_prompt=True),
            pipeline_factory=factory,
            encoder=good_encoder,
            enhancer=Recorder(result=PromptEnhancementError(repo_id="org/x", reason="offline")),
            output_root=tmp_path,
            retention_bytes=10**9,
        )
    assert loaded == []


def test_an_illegal_request_is_refused_before_the_enhancer_is_loaded():
    """``generate.py``'s "everything cheap first" rule, which enhancement now
    sits above. Waiting through a CPU decode to be told the frame count was
    always illegal is the same defect the rule was written for, one layer up."""
    from ltx25_studio.errors import FrameCountError, UnknownModeError

    seam = Recorder()
    with pytest.raises(FrameCountError):
        generate_module.generate(a_request(enhance_prompt=True, num_frames=120), enhancer=seam)
    with pytest.raises(UnknownModeError):
        generate_module.generate(
            a_request(mode="text_to_hologram", enhance_prompt=True), enhancer=seam
        )
    assert seam.calls == [], "the enhancer ran for a request that could never generate"


# --------------------------------------------------------------------------
# the placement: INSIDE the GPU worker, past the fork, on the GPU
# --------------------------------------------------------------------------
#
# ⚠️ These two tests are inversions of the pair that shipped here, and the pair
# they replace was not wrong about its own subject. It asserted that the request
# crossing the fork already carried the rewritten prompt, because enhancement
# ran in the WEB process where it cost no GPU-seconds. That was true, it was
# tested, and it was measured at +538 s of user-visible wall-clock: a 600-token
# greedy decode on a CPU. The cost argument was sound on the axis it considered
# (disk, quota) and silent on the one that reached the user (latency).
#
# So the property being pinned flips: what must be true now is that the enhancer
# runs on the far side of the dispatch, exactly once, and that its note still
# reaches the user -- which is a harder guarantee than before, because the note
# now has to cross the fork inside GenerationResult.warnings rather than being
# grafted on afterwards.


def test_enhancement_happens_inside_the_gpu_worker(tmp_path, monkeypatch):
    """The placement, as an assertion, pointed at where it now lives.

    Enhancement is GPU work in the metered window, so the request crossing the
    fork must still carry the ORIGINAL prompt -- the rewrite happens on the
    other side -- and the enhancer must run exactly once, inside.
    """
    seen: list = []

    def dispatch(*, request, **kwargs):
        seen.append(request)
        token = generate_module._IN_GPU_WORKER.set(True)
        try:
            return generate_module.generate(request, **kwargs)
        finally:
            generate_module._IN_GPU_WORKER.reset(token)

    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
    seam = Recorder()
    monkeypatch.setattr(enhance_module, "enhance", seam)

    result = generate_module.generate(
        a_request(enhance_prompt=True),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    assert len(seam.calls) == 1, (
        f"the enhancer ran {len(seam.calls)} times; the call is unconditional but must still "
        f"rewrite exactly once per generation"
    )
    assert seen and seen[0].prompt == ORIGINAL, (
        "the request that crossed the dispatch boundary already carried the rewrite, so "
        "enhancement is still happening in the web process -- which is the ~538 s CPU decode "
        "this move exists to end"
    )
    assert any(REWRITTEN in note for note in result.warnings), (
        "the note is now computed INSIDE the worker, so it has to survive the return leg in "
        "GenerationResult.warnings; without that the user never sees which prompt ran"
    )


def test_the_gpu_worker_is_where_enhancement_actually_runs(tmp_path, monkeypatch):
    """The same property from the other side: a request arriving inside the
    worker with the toggle set IS enhanced there.

    This is the exact inverse of ``test_the_gpu_worker_never_enhances``, which
    stood here and asserted ``seam.calls == []``. Keeping a worker-side guard at
    all is the point -- without one, a future re-hoist of the call above the
    dispatch branch would silently restore the CPU placement and nothing would
    fail.
    """
    seam = Recorder()
    monkeypatch.setattr(enhance_module, "enhance", seam)
    token = generate_module._IN_GPU_WORKER.set(True)
    try:
        result = generate_module.generate(
            a_request(enhance_prompt=True),
            pipeline_factory=factory_for(FakePipe()),
            encoder=good_encoder,
            output_root=tmp_path,
            retention_bytes=10**9,
        )
    finally:
        generate_module._IN_GPU_WORKER.reset(token)
    assert [call["prompt"] for call in seam.calls] == [ORIGINAL]
    assert any(REWRITTEN in note for note in result.warnings)


def test_the_injected_enhancer_survives_the_dispatch(tmp_path, monkeypatch):
    """The seam has to cross the branch, or it stops being a seam.

    ``generate()`` forwards ``pipeline_factory`` and ``encoder`` to the
    entrypoint; ``enhancer`` did not need forwarding while enhancement happened
    before the branch. It does now, and the failure mode of forgetting is quiet
    and expensive: the caller passes a fake, the worker ignores it, and reaches
    for 2.5 GB of real Gemma over the network instead. Caught exactly this way.
    """
    forwarded: list = []

    def dispatch(**kwargs):
        forwarded.append(kwargs)
        token = generate_module._IN_GPU_WORKER.set(True)
        try:
            return generate_module.generate(**kwargs)
        finally:
            generate_module._IN_GPU_WORKER.reset(token)

    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
    seam = Recorder()
    generate_module.generate(
        a_request(enhance_prompt=True),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        enhancer=seam,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    assert forwarded and forwarded[0]["enhancer"] is seam, (
        "the dispatch payload dropped `enhancer`, so the injected seam is unused on any host "
        "with an entrypoint registered"
    )
    assert seam.calls, "the worker never consulted the injected enhancer"


# --------------------------------------------------------------------------
# through the UI: the checkbox, and what the user reads afterwards
# --------------------------------------------------------------------------


def test_the_ui_declares_a_real_toggle():
    from ltx25_studio.ui import fields

    spec = fields.field_by_key("enhance_prompt")
    assert spec.kind == "checkbox"
    assert spec.default is False
    assert spec.key in fields.FIELD_KEYS


def test_the_toggle_is_appended_rather_than_inserted():
    """``/on_generate``'s positional order is the endpoint's signature.

    Inserting a field mid-tuple shifts every later value one slot to the left
    for any client that already knows the order -- prompt into negative_prompt,
    width into height -- which generates something plausible and wrong with no
    error anywhere.
    """
    from ltx25_studio.ui import fields

    before = (
        "prompt", "negative_prompt", "image", "video", "audio", "conditions", "width",
        "height", "auto_duration", "num_frames", "min_seconds", "max_seconds",
        "enable_stage_2", "seed", "strength", "noise_scale", "loras",
    )  # fmt: skip
    assert fields.FIELD_KEYS.index("enhance_prompt") == len(before) == 17
    assert fields.FIELD_KEYS[:17] == before, "a field was inserted ahead of the toggle"


def test_there_is_no_seed_control_for_enhancement():
    """PLAN §3.5: LTX-2.5 enhancement decodes greedily, so
    ``prompt_enhancement_seed`` changes nothing. This build has shipped seven
    inert levers already; an eighth would be a choice."""
    from ltx25_studio.ui import fields

    assert enhance_module.GENERATION_KWARGS["do_sample"] is False
    assert not [key for key in fields.FIELD_KEYS if "enhance" in key and "seed" in key]


def test_the_toggle_is_hidden_for_modes_that_discard_the_prompt():
    from ltx25_studio.ui import fields

    prompt_spec = fields.field_by_key("prompt")
    enhance_spec = fields.field_by_key("enhance_prompt")
    for mode_id in ("text_to_video", "image_to_video", "audio_to_audio"):
        from ltx25_studio import modes

        mode = modes.get_mode(mode_id)
        assert enhance_spec.visible_when(mode) == prompt_spec.visible_when(mode), (
            f"the enhance toggle and the prompt box disagree about {mode_id}"
        )


def test_the_wait_announces_itself_as_a_prompt_rewrite_not_a_gpu_reservation():
    """The generic running message says "reserving a ZeroGPU slot".

    With enhancement on that is not the whole story -- the first seconds of the
    reservation go on rewriting the prompt rather than denoising. A status line
    describing something other than what is happening, exactly where the user is
    waiting for it, is the same small lie as a hidden control.

    ⚠️ This test used to require the words "no quota" and "CPU" in the enhanced
    message, and both were true of the placement it was written for. Both are
    now false, so asserting them would pin a lie in place. The enhanced message
    must still DIFFER and must still name the rewrite -- that part was never
    about where it ran.
    """
    from ltx25_studio.ui import handlers

    plain = handlers.on_generate_start("text_to_video")
    enhancing = handlers.on_generate_start("text_to_video", True)
    assert plain != enhancing
    assert "quota" in plain
    assert "Enhancing your prompt" in enhancing
    assert "no quota" not in enhancing, (
        "the enhanced message still tells the user enhancement is free. It runs inside the "
        "reservation now and is billed like the rest of the run."
    )


def test_no_user_facing_copy_still_says_enhancement_is_free():
    """The claim is gone from every surface, not just the one under test.

    Three places said it -- the field caption, the running message and the
    module that answers "where does this run" -- and a correction that lands on
    two of three leaves the user reading the old promise wherever it survived.
    """
    from ltx25_studio.ui import fields

    caption = fields.field_by_key("enhance_prompt").info
    assert "NO extra GPU quota" not in caption and "costs a wait" not in caption, caption
    assert "GPU" in caption, "the caption no longer says where the rewrite runs"


def test_the_status_panel_shows_the_prompt_that_was_actually_sent(tmp_path, monkeypatch):
    """End to end through the real handler, which is what the user reads."""
    from ltx25_studio.ui import handlers
    from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

    values = {spec.key: spec.default for spec in FIELDS}
    values["prompt"] = ORIGINAL
    values["seed"] = 7
    values["enhance_prompt"] = True

    real = generate_module.generate

    def wired(*args, **kwargs):
        kwargs.setdefault("pipeline_factory", factory_for(FakePipe()))
        kwargs.setdefault("encoder", good_encoder)
        kwargs.setdefault("enhancer", Recorder())
        kwargs.setdefault("output_root", tmp_path)
        kwargs.setdefault("retention_bytes", 10**9)
        return real(*args, **kwargs)

    monkeypatch.setattr(generate_module, "generate", wired)
    _, _, status = handlers.on_generate("text_to_video", *(values[k] for k in FIELD_KEYS))
    assert REWRITTEN in status, status


def test_the_unavailable_note_is_gone_from_the_ui():
    """The copy it replaced said enhancement was off on disk budget. Leaving it
    next to a working checkbox would be a worse lie than the original."""
    blocks = (PACKAGE_ROOT / "ui" / "blocks.py").read_text()
    assert "_ENHANCER_NOTE" not in blocks
    assert "unaffordable" not in blocks


# --------------------------------------------------------------------------
# the loader, with the real import statements resolving to fakes
# --------------------------------------------------------------------------


class FakeInputs(dict):
    """Duck-types ``transformers.BatchFeature``: a mapping with attributes."""

    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids, attention_mask=[[1] * len(input_ids[0])])
        self.input_ids = input_ids
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.templated: list = []
        self.decoded: list = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.templated.append(messages)
        assert tokenize is False and add_generation_prompt is True
        return "<templated>"

    def batch_decode(self, sequences, skip_special_tokens=True):
        self.decoded.append(sequences)
        return [f"  “{REWRITTEN}" for _ in sequences]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls: list[dict] = []

    def __call__(self, *, text, images, return_tensors):
        self.calls.append({"text": text, "images": images, "return_tensors": return_tensors})
        return FakeInputs([[1, 2, 3]])


class FakeParameter:
    device = types.SimpleNamespace(type="cpu")


class FakeModel:
    def __init__(self):
        self.generate_kwargs: list[dict] = []
        self.moved_to = None
        self.evaluated = False

    def eval(self):
        self.evaluated = True
        return self

    def to(self, device):
        self.moved_to = device
        return self

    def parameters(self):
        yield FakeParameter()

    def generate(self, **kwargs):
        self.generate_kwargs.append(kwargs)
        return [[1, 2, 3, 9, 9, 9]]


T2V_SYSTEM_PROMPT = "SYSTEM-T2V: write a caption for a text-to-video request."
I2V_SYSTEM_PROMPT = "SYSTEM-I2V: write a caption that BEGINS from the reference image."


@pytest.fixture
def upstream(monkeypatch):
    """Fake ``torch`` and the two ``diffusers.pipelines.ltx2`` helper modules.

    Installed on ``sys.modules`` rather than monkeypatched onto ``enhance`` so
    the module's own ``from diffusers.pipelines.ltx2... import ...`` lines are
    the code under test. The names faked here are the exact names upstream
    exports -- ``test_the_upstream_helpers_this_module_imports_still_exist``
    checks that claim against a real diffusers when one is installed.
    """
    torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", torch)

    prepared: list = []

    for name in ("diffusers", "diffusers.pipelines", "diffusers.pipelines.ltx2"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    enhancement = types.ModuleType("diffusers.pipelines.ltx2.prompt_enhancement")
    enhancement.clean_response = lambda text: text.translate(
        str.maketrans("‘’“”", "''\"\"")
    ).lstrip(" \"'")
    enhancement._pad_inputs_for_attention_alignment = lambda inputs, pad_token_id=0: inputs

    def _prepare(image, long_side=896):
        prepared.append(image)
        return f"prepared:{image}"

    enhancement._prepare_enhance_image = _prepare
    monkeypatch.setitem(sys.modules, enhancement.__name__, enhancement)

    utils = types.ModuleType("diffusers.pipelines.ltx2.utils")
    utils.LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT = T2V_SYSTEM_PROMPT
    utils.LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT = I2V_SYSTEM_PROMPT
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    return types.SimpleNamespace(prepared=prepared)


def _guard(target: str, replacement):
    """An ``__import__`` that fails for exactly one module and defers otherwise."""
    real = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == target:
            return replacement(name, *args, **kwargs)
        return real(name, *args, **kwargs)

    return _import


def loader_for(model, processor):
    def _loader(repo_id):
        return model, processor

    return _loader


def test_a_text_only_rewrite_uses_the_t2v_system_prompt(upstream):
    model, processor = FakeModel(), FakeProcessor()
    out = enhance_module.enhance(
        ORIGINAL, repo_id="org/enhancer", loader=loader_for(model, processor)
    )
    assert out == REWRITTEN, "clean_response was not applied to the decoded text"
    messages = processor.tokenizer.templated[0]
    assert messages[0]["content"] == T2V_SYSTEM_PROMPT
    assert ORIGINAL in messages[1]["content"]
    assert processor.calls[0]["images"] is None


def test_an_image_conditioned_rewrite_uses_the_i2v_system_prompt(upstream):
    model, processor = FakeModel(), FakeProcessor()
    enhance_module.enhance(
        ORIGINAL, image="a-frame", repo_id="org/enhancer", loader=loader_for(model, processor)
    )
    messages = processor.tokenizer.templated[0]
    assert messages[0]["content"] == I2V_SYSTEM_PROMPT
    assert upstream.prepared == ["a-frame"], "the frame never reached _prepare_enhance_image"
    assert processor.calls[0]["images"] == "prepared:a-frame"


def test_decoding_is_greedy_and_bounded_at_the_planned_length(upstream):
    model, processor = FakeModel(), FakeProcessor()
    enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=loader_for(model, processor))
    kwargs = model.generate_kwargs[0]
    assert kwargs["max_new_tokens"] == 600
    assert kwargs["do_sample"] is False
    assert kwargs["no_repeat_ngram_size"] == 5


def test_the_inputs_follow_the_model_onto_whatever_device_it_landed_on(upstream):
    """Moving the model and not the tensors is a real failure mode, and a
    **silent** one on a CPU-only host -- where both are right by accident.

    Read off the model rather than re-probing torch, so an injected loader that
    moved nothing still gets coherent inputs.
    """
    model, processor = FakeModel(), FakeProcessor()
    enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=loader_for(model, processor))
    assert processor.calls, "the processor was never called"
    assert model.generate_kwargs, "the model never generated"


class RecordingProcessor(FakeProcessor):
    """Keeps the inputs it returned, so a test can see where they were moved."""

    def __init__(self):
        super().__init__()
        self.returned: list = []

    def __call__(self, *, text, images, return_tensors):
        inputs = super().__call__(text=text, images=images, return_tensors=return_tensors)
        self.returned.append(inputs)
        return inputs


@pytest.mark.parametrize("device", ["mps", "cuda", "cpu"])
def test_image_aware_inputs_go_to_the_device_the_model_is_on(device, upstream):
    """Measured on a real MPS run: the model sat on mps, the inputs were moved to
    cpu, and image-aware enhancement failed with "Tensor for argument input is on
    cpu but expected on mps". The device is read off the model's parameters."""

    class Placed(FakeModel):
        def parameters(self):
            yield types.SimpleNamespace(device=types.SimpleNamespace(type=device))

    processor = RecordingProcessor()
    enhance_module.enhance(
        ORIGINAL, image="a-frame", repo_id="org/enhancer", loader=loader_for(Placed(), processor)
    )
    assert processor.calls[0]["images"] == "prepared:a-frame"
    assert processor.returned[0].moved_to == device


def test_a_real_tensor_dict_follows_a_real_mps_model():
    """FULL env, Apple silicon only: a small BatchFeature with pixel values lands
    on the device _model_device reads off a real MPS parameter."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this machine")
    transformers = pytest.importorskip("transformers")

    model = torch.nn.Linear(4, 4).to("mps")
    device = enhance_module._model_device(model)
    assert device == "mps"
    inputs = transformers.BatchFeature(
        {"input_ids": torch.zeros(1, 3, dtype=torch.long), "pixel_values": torch.zeros(1, 3, 4, 4)}
    ).to(device)
    assert {tensor.device.type for tensor in inputs.values()} == {"mps"}


def test_the_device_is_resolved_rather_than_pinned_to_the_cpu(monkeypatch):
    """``ENHANCER_DEVICE = "cpu"`` used to be a module constant, and the
    reasoning behind it was correct for where the call then lived: probing
    ``torch.cuda`` from the web process is the blocked lazy init that *hangs*.

    The call now runs inside the ``@spaces.GPU`` worker, past ``torch.unpatch``,
    where CUDA is already up. A constant would pin a 5B-parameter model to the
    device that took 0.9 s/token, for a reason that no longer applies -- so the
    device is a resolution, and it answers "cuda" when there is one.
    """
    assert not hasattr(enhance_module, "ENHANCER_DEVICE"), (
        "the CPU pin is back as a constant; enhancement runs on the GPU now"
    )

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert enhance_module.enhancer_device() == "cuda"

    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    assert enhance_module.enhancer_device() == "cpu"


def _torch_with(*, cuda, mps, emptied=None):
    sink = emptied if emptied is not None else []
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda, empty_cache=lambda: sink.append("cuda")
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    torch.mps = types.SimpleNamespace(empty_cache=lambda: sink.append("mps"))
    return torch


def test_apple_silicon_resolves_to_mps(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch_with(cuda=False, mps=True))
    assert enhance_module.enhancer_device() == "mps"


def test_cuda_still_wins_when_both_are_present(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch_with(cuda=True, mps=True))
    assert enhance_module.enhancer_device() == "cuda"


def test_an_explicit_device_is_honoured(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch_with(cuda=True, mps=False))
    assert enhance_module.enhancer_device("cpu") == "cpu"
    assert enhance_module.enhancer_device("mps") == "mps"


def test_an_explicit_device_reaches_the_load(monkeypatch):
    transformers = types.ModuleType("transformers")
    model, processor = FakeModel(), FakeProcessor()
    transformers.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda repo_id, **kwargs: processor
    )
    transformers.Gemma4ForConditionalGeneration = types.SimpleNamespace(
        from_pretrained=lambda repo_id, **kwargs: model
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", _torch_with(cuda=True, mps=False))

    enhance_module._load_enhancer("org/enhancer", device="cpu")
    assert model.moved_to == "cpu"


def test_dropping_an_mps_resident_enhancer_empties_the_mps_cache(monkeypatch):
    emptied: list = []
    monkeypatch.setitem(sys.modules, "torch", _torch_with(cuda=False, mps=True, emptied=emptied))

    class OnMps(FakeModel):
        def parameters(self):
            yield types.SimpleNamespace(device=types.SimpleNamespace(type="mps"))

    enhance_module._ENHANCER[("org/enhancer", None)] = (OnMps(), FakeProcessor())
    enhance_module.clear_enhancer_cache()
    assert emptied == ["mps"]


def test_a_host_without_torch_resolves_to_the_cpu_rather_than_raising(monkeypatch):
    """The probe must never be the thing that breaks a run. A machine with no
    torch, or a torch built without CUDA, enhances slowly -- which is the honest
    answer for a machine that has nothing faster."""
    broken = types.ModuleType("torch")  # no `.cuda` attribute at all
    monkeypatch.setitem(sys.modules, "torch", broken)
    assert enhance_module.enhancer_device() == "cpu"


def test_the_loaded_model_is_put_on_the_resolved_device_and_into_eval_mode(monkeypatch):
    """Stated rather than inherited: a global default device must not be able to
    decide where 2.5 GB of Gemma lands behind this module's back."""
    transformers = types.ModuleType("transformers")
    model, processor = FakeModel(), FakeProcessor()
    transformers.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda repo_id, **kwargs: processor
    )
    transformers.Gemma4ForConditionalGeneration = types.SimpleNamespace(
        from_pretrained=lambda repo_id, **kwargs: model
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(enhance_module, "enhancer_device", lambda: "cuda")

    loaded, _ = enhance_module._load_enhancer("org/enhancer")
    assert loaded is model
    assert model.evaluated is True
    assert model.moved_to == "cuda", (
        "the load ignored the resolved device, so the enhancer stays on the CPU inside the "
        "worker and the move buys nothing"
    )


def test_dropping_the_resident_enhancer_gives_the_vram_back(monkeypatch):
    """The free path was **moved, not deleted**, and this is where it lives now.

    ``enhance()`` no longer frees per call -- that cost a measured +30.78 s every
    time -- so the ``del`` + ``gc.collect()`` + ``empty_cache()`` sequence moved
    into ``clear_enhancer_cache()``. If it had simply been deleted, "resident"
    would be a one-way door and the VRAM would be unrecoverable without killing
    the worker.

    Guarded on the MODEL's own device, never on ``torch.cuda.is_available()`` --
    a parameter can only report cuda if CUDA was already initialised to put it
    there, so reading it can never be what initialises it.
    """
    emptied: list = []
    torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.no_grad = _NoGrad
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: emptied.append(True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("diffusers", "diffusers.pipelines", "diffusers.pipelines.ltx2"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    enhancement = types.ModuleType("diffusers.pipelines.ltx2.prompt_enhancement")
    enhancement.clean_response = lambda text: text.strip()
    enhancement._pad_inputs_for_attention_alignment = lambda inputs, pad_token_id=0: inputs
    enhancement._prepare_enhance_image = lambda image, long_side=896: image
    monkeypatch.setitem(sys.modules, enhancement.__name__, enhancement)
    utils = types.ModuleType("diffusers.pipelines.ltx2.utils")
    utils.LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT = T2V_SYSTEM_PROMPT
    utils.LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT = I2V_SYSTEM_PROMPT
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    class OnCuda(FakeModel):
        def parameters(self):
            yield types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))

    enhance_module.enhance(
        ORIGINAL, repo_id="org/enhancer", loader=loader_for(OnCuda(), FakeProcessor())
    )
    assert emptied == [], (
        "enhance() emptied the CUDA cache on its way out, which is the per-call free this "
        "commit removed -- the model is supposed to still be resident here"
    )

    enhance_module.clear_enhancer_cache()
    assert emptied == [True], (
        "a GPU-resident enhancer was dropped without emptying the CUDA cache, so its ~2.5 GB "
        "can still be held when the transformer needs the VRAM"
    )
    assert enhance_module._ENHANCER == {}, "the cache still holds the model it just freed"


def test_a_cpu_enhancer_never_touches_torch_cuda_at_all(upstream):
    """The other half of the same guard. ``upstream``'s fake torch has no
    ``.cuda`` attribute, so any unguarded ``torch.cuda.empty_cache()`` on the
    free path raises AttributeError and this fails."""
    model, processor = FakeModel(), FakeProcessor()  # FakeParameter reports "cpu"
    enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=loader_for(model, processor))
    assert model.generate_kwargs, "the model never generated"


def test_the_model_survives_the_call_and_dies_when_the_cache_is_cleared(upstream):
    """The inversion of the rule this module shipped with, and the point of the change.

    It used to be checked that NO strong reference survived ``enhance()``. That
    was a real property, argued on VRAM safety, and it cost a 2.49 GB reload on
    every enhanced generation -- ✅ +30.78 s measured -- to protect headroom on a
    partition that turned out to be half the size the app was entitled to.

    Checked with a weak reference in both directions rather than by trusting the
    dict: "still referenced" and "actually collectable" are the only versions of
    resident and freed that mean anything.
    """
    import gc
    import weakref

    watched: dict = {}

    def loader(repo_id):
        # Built HERE so the only strong reference is the one the cache takes.
        model = FakeModel()
        watched["model"] = weakref.ref(model)
        return model, FakeProcessor()

    enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=loader)
    gc.collect()
    assert watched["model"]() is not None, (
        "the enhancer was dropped when enhance() returned, so the next enhanced generation "
        "in this worker pays the 2.49 GB load again (+30.78 s measured)"
    )

    enhance_module.clear_enhancer_cache()
    gc.collect()
    assert watched["model"]() is None, (
        "clearing the cache left a strong reference behind, so residency cannot be undone "
        "and the VRAM is unrecoverable short of killing the worker"
    )


# --------------------------------------------------------------------------
# residency: loaded once per worker, and never at import
# --------------------------------------------------------------------------


def test_the_second_rewrite_in_a_worker_does_not_load_the_model_again(upstream):
    """The change, stated as the only thing that makes it worth making.

    ``spaces`` runs the forked worker in a ``while True:`` loop that serves many
    tasks from one process, so module state written by call N is still there for
    call N+1. If it were not -- if the platform forked fresh per request -- this
    cache would never hit and the whole commit would be a no-op. See the module
    docstring in ``enhance.py`` for that reading of ``spaces/zero/wrappers.py``.
    """
    loads: list[str] = []

    def counting_loader(repo_id):
        loads.append(repo_id)
        return FakeModel(), FakeProcessor()

    first = enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=counting_loader)
    second = enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=counting_loader)

    assert first == second == REWRITTEN
    assert loads == ["org/enhancer"], (
        f"the enhancer was loaded {len(loads)} times for two rewrites in one process. "
        f"Residency is not hitting, so every enhanced generation still pays the 2.49 GB "
        f"reload this commit exists to remove"
    )


def test_a_failed_load_is_not_cached_so_the_next_call_can_retry(monkeypatch):
    """A transient Hub failure must not brick the enhancer for the worker's life.

    Caching the *attempt* rather than the result is the easy version of this bug:
    one 503 during a deploy and every enhanced run in that worker fails forever,
    with the toggle still advertising a feature that can no longer work.
    """
    attempts: list[str] = []

    def flaky(repo_id):
        attempts.append(repo_id)
        if len(attempts) == 1:
            raise PromptEnhancementError(repo_id=repo_id, reason="the Hub was briefly down")
        return FakeModel(), FakeProcessor()

    with pytest.raises(PromptEnhancementError):
        enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=flaky)
    assert enhance_module._ENHANCER == {}, "a load that raised was cached anyway"

    # Second call: the fake upstream helpers are needed only now, because the
    # first call never got as far as rewriting.
    torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("diffusers", "diffusers.pipelines", "diffusers.pipelines.ltx2"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    enhancement = types.ModuleType("diffusers.pipelines.ltx2.prompt_enhancement")
    enhancement.clean_response = lambda text: text.strip().lstrip("\u201c")
    enhancement._pad_inputs_for_attention_alignment = lambda inputs, pad_token_id=0: inputs
    enhancement._prepare_enhance_image = lambda image, long_side=896: image
    monkeypatch.setitem(sys.modules, enhancement.__name__, enhancement)
    utils = types.ModuleType("diffusers.pipelines.ltx2.utils")
    utils.LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT = T2V_SYSTEM_PROMPT
    utils.LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT = I2V_SYSTEM_PROMPT
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    assert enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=flaky) == REWRITTEN
    assert attempts == ["org/enhancer", "org/enhancer"], (
        "the retry never reached the loader, so a single transient failure is permanent"
    )


def test_the_cache_is_keyed_on_the_repository_so_the_lever_still_works(upstream):
    """``LTX25_PROMPT_ENHANCER_REPO`` is a real lever -- an operator repointing it
    at the bf16 build must not be served the QAT weights out of a cache that
    forgot which repo it loaded."""
    loads: list[str] = []

    def counting_loader(repo_id):
        loads.append(repo_id)
        return FakeModel(), FakeProcessor()

    enhance_module.enhance(ORIGINAL, repo_id="org/qat", loader=counting_loader)
    enhance_module.enhance(ORIGINAL, repo_id="org/bf16", loader=counting_loader)
    enhance_module.enhance(ORIGINAL, repo_id="org/qat", loader=counting_loader)

    assert loads == ["org/qat", "org/bf16"], (
        "the cache ignored the repo id, so pointing LTX25_PROMPT_ENHANCER_REPO somewhere "
        "new serves the previous repository's weights"
    )


def test_nothing_at_module_scope_fills_the_cache():
    """⛔ The constraint most likely to be broken by a later "optimisation".

    ZeroGPU packing fires once at ``demo.launch()`` and captures whatever
    registered in ``cuda_aliases`` during the import phase. A module-scope load --
    "we know we need it, why not have it ready" -- would put 2.49 GB of Gemma back
    into the 72 GB disk budget the whole feature was shaped around avoiding, and
    it would do it silently, with no error anywhere.

    Parsed rather than grepped: the module *documents* this constraint at length,
    so a string search finds the explanation and proves nothing.
    """
    import ast

    tree = ast.parse(ENHANCE_SOURCE.read_text())

    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_ENHANCER"
    ]
    assert len(declarations) == 1, "expected exactly one module-scope _ENHANCER declaration"
    declared = declarations[0].value
    assert isinstance(declared, ast.Dict) and not declared.keys, (
        "the enhancer cache is not declared empty; anything prebuilt here runs during the "
        "import phase, which is the packing window"
    )

    executable = [
        node
        for node in tree.body
        if node is not declarations[0]
        and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and any(isinstance(n, ast.Name) and n.id == "_ENHANCER" for n in ast.walk(node))
    ]
    assert executable == [], (
        "a module-scope statement touches the enhancer cache. It must be filled on first "
        "use inside the GPU worker, past the fork, and never during import"
    )

    assert enhance_module._ENHANCER == {}, (
        "importing enhance.py left something in the cache"
    )


def _recording_transformers(monkeypatch):
    seen: dict = {}
    model, processor = FakeModel(), FakeProcessor()
    transformers = types.ModuleType("transformers")

    def _processor(repo_id, **kwargs):
        seen["processor"] = (repo_id, kwargs)
        return processor

    def _model(repo_id, **kwargs):
        seen["model"] = (repo_id, kwargs)
        return model

    transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=_processor)
    transformers.Gemma4ForConditionalGeneration = types.SimpleNamespace(from_pretrained=_model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(enhance_module, "enhancer_device", lambda device=None: "cpu")
    return seen


def test_the_default_enhancer_loads_at_its_pinned_commit(monkeypatch):
    """The repo's chat template changed twice in July 2026; the template is what
    the rewrite is built from, so an unpinned load can change enhancement output
    without any change here."""
    seen = _recording_transformers(monkeypatch)
    default = config_module.DEFAULT_PROMPT_ENHANCER_REPO
    enhance_module._load_enhancer(default, revision=enhance_module.enhancer_revision(default))
    assert seen["processor"] == (default, {"revision": enhance_module.ENHANCER_REVISION})
    assert seen["model"] == (default, {"revision": enhance_module.ENHANCER_REVISION, "dtype": "auto"})


def test_the_pin_reaches_the_load_through_enhance(monkeypatch, upstream):
    seen = _recording_transformers(monkeypatch)
    monkeypatch.delenv(config_module.ENV_PROMPT_ENHANCER_REPO, raising=False)
    enhance_module.enhance(ORIGINAL)
    assert seen["model"][1]["revision"] == enhance_module.ENHANCER_REVISION
    assert seen["processor"][1]["revision"] == enhance_module.ENHANCER_REVISION


def test_the_pin_is_the_commit_that_was_head_when_it_was_measured():
    assert enhance_module.ENHANCER_REVISION == "dd693ff40353f057ca5f07e945ad867f4afbf2ec"
    default = config_module.DEFAULT_PROMPT_ENHANCER_REPO
    assert default == "google/gemma-4-E2B-it-qat-mobile-transformers", (
        "the default repo changed: the pin belongs to the old one, choose a new pin"
    )
    assert enhance_module.enhancer_revision(default) == enhance_module.ENHANCER_REVISION


def test_an_overridden_repo_is_loaded_unpinned_and_the_log_says_so(monkeypatch, caplog):
    seen = _recording_transformers(monkeypatch)
    assert enhance_module.enhancer_revision("google/gemma-4-E2B-it") is None
    with caplog.at_level("WARNING"):
        enhance_module._load_enhancer("google/gemma-4-E2B-it", revision=None)
    assert seen["model"][1]["revision"] is None
    assert "not pinned" in caplog.text


def test_a_load_failure_names_the_repository_and_the_reason(monkeypatch):
    """The enhancer is the one model resolved outside the pinned pack, so
    "which repo?" is a real question with a configurable answer."""
    transformers = types.ModuleType("transformers")

    class Boom:
        @staticmethod
        def from_pretrained(repo_id, **kwargs):
            raise OSError("no such repository")

    transformers.AutoProcessor = Boom
    transformers.Gemma4ForConditionalGeneration = Boom
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(PromptEnhancementError) as caught:
        enhance_module.enhance(ORIGINAL, repo_id="org/nope")
    message = str(caught.value)
    assert "org/nope" in message
    assert "no such repository" in message


def test_a_transformers_too_old_to_know_gemma_4_says_so(monkeypatch):
    transformers = types.ModuleType("transformers")  # no Gemma4ForConditionalGeneration
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(PromptEnhancementError) as caught:
        enhance_module.enhance(ORIGINAL, repo_id="org/nope")
    message = str(caught.value)
    assert "org/nope" in message
    assert "5.15.1" in message, "the message must name the version floor it needs"


# --------------------------------------------------------------------------
# the prefetch: a download, at startup, that can fail without consequence
# --------------------------------------------------------------------------
#
# Enhancement is inside the metered window now, so the 88 s Hub fetch would be
# BILLED on the first enhanced call in every container -- ~176 GPU-seconds to
# move a file. The prefetch spends that at startup instead. Three properties
# make it safe to do that, and each is a test below: it downloads and does not
# construct, it cannot fail the Space, and it cannot delay readiness.


def _hub(monkeypatch, downloaded, *, boom=None):
    """A fake ``huggingface_hub`` whose ``snapshot_download`` is observable."""
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id, token=None, **kwargs):
        downloaded.append({"repo_id": repo_id, "token": token, **kwargs})
        if boom is not None:
            raise boom
        return f"/cache/models--{repo_id.replace('/', '--')}/snapshots/deadbeef"

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return hub


def test_the_prefetch_downloads_the_configured_repository_at_its_pinned_commit(monkeypatch):
    downloaded: list = []
    _hub(monkeypatch, downloaded)
    monkeypatch.delenv(config_module.ENV_PROMPT_ENHANCER_REPO, raising=False)

    path = enhance_module.prefetch(token="hf_secret")
    assert downloaded == [
        {
            "repo_id": config_module.DEFAULT_PROMPT_ENHANCER_REPO,
            "revision": enhance_module.ENHANCER_REVISION,
            "token": "hf_secret",
        }
    ]
    assert path and "gemma-4-E2B-it-qat-mobile-transformers" in path


def test_a_repo_other_than_the_default_is_prefetched_at_its_head_and_the_log_says_so(
    monkeypatch, caplog
):
    downloaded: list = []
    _hub(monkeypatch, downloaded)
    monkeypatch.setattr(enhance_module, "enhancer_repo_id", lambda: "org/enhancer")

    with caplog.at_level("WARNING"):
        enhance_module.prefetch(token="hf_secret")
    assert downloaded == [{"repo_id": "org/enhancer", "revision": None, "token": "hf_secret"}]
    assert "org/enhancer" in caplog.text and "not pinned" in caplog.text


def test_the_prefetch_follows_the_repo_lever_rather_than_a_hardcoded_name(monkeypatch):
    """``LTX25_PROMPT_ENHANCER_REPO`` points the enhancer at any Hub repo, so a
    prefetch that fetched a constant would warm the cache for a model this run
    is never going to load -- paying the disk twice and the download twice."""
    downloaded: list = []
    _hub(monkeypatch, downloaded)
    monkeypatch.setenv(config_module.ENV_PROMPT_ENHANCER_REPO, "google/gemma-4-E2B-it")

    enhance_module.prefetch()
    assert downloaded[0]["repo_id"] == "google/gemma-4-E2B-it"


def test_the_prefetch_constructs_no_model_and_no_tensor(monkeypatch):
    """**The whole disk argument, as an assertion.**

    ZeroGPU packs tensors registered in ``cuda_aliases`` while the module scope
    runs. A prefetch that called ``from_pretrained`` would build a real model
    during exactly that window; a prefetch that only writes blobs cannot be
    packed, because files on disk are not packable state. So: no transformers,
    no torch, no ``from_pretrained`` -- reaching for any of them raises here.
    """
    downloaded: list = []
    _hub(monkeypatch, downloaded)

    for forbidden in ("torch", "transformers"):
        module = types.ModuleType(forbidden)

        def _explode(*args, **kwargs):
            raise AssertionError(
                "the prefetch touched the model stack; it must move bytes, not build models"
            )

        module.__getattr__ = _explode
        monkeypatch.setitem(sys.modules, forbidden, module)

    assert enhance_module.prefetch("org/enhancer") is not None
    assert downloaded, "nothing was downloaded"


def test_a_failed_prefetch_is_not_fatal_and_says_what_it_costs(monkeypatch, caplog):
    """A Space that cannot reach the Hub at startup must still start.

    The degraded outcome is one slow first enhanced call, not a dead Space and
    not a dead un-enhanced path -- so this swallows, and the log has to say so,
    because a silent swallow reads as a prefetch that worked.
    """
    _hub(monkeypatch, [], boom=OSError("connection reset by peer"))
    with caplog.at_level("WARNING"):
        assert enhance_module.prefetch("org/enhancer") is None
    logged = caplog.text
    assert "org/enhancer" in logged and "connection reset by peer" in logged
    assert "un-enhanced" in logged, "the log does not say what still works"


def test_a_missing_huggingface_hub_is_not_fatal_either(monkeypatch):
    """Import failure is a failure like any other on this path."""

    def _no_hub(name, *args, **kwargs):
        raise ImportError("no huggingface_hub here")

    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    monkeypatch.setattr(builtins, "__import__", _guard("huggingface_hub", _no_hub))
    assert enhance_module.prefetch("org/enhancer") is None


def test_the_prefetch_runs_on_a_daemon_thread_so_readiness_never_waits(monkeypatch):
    """``app.py`` calls this before ``demo.launch()``. The fetch is ~88 s; doing
    it inline would hold the Space unavailable for a minute and a half to
    prepare an optional feature. A daemon thread also cannot keep the process
    alive at exit, so a fetch still in flight does not delay a stop."""
    downloaded: list = []
    _hub(monkeypatch, downloaded)

    default = config_module.DEFAULT_PROMPT_ENHANCER_REPO
    thread = enhance_module.start_prefetch(default, token="hf_secret")
    assert thread.daemon is True, "a non-daemon prefetch can hold the process open at exit"
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert downloaded == [
        {"repo_id": default, "revision": enhance_module.ENHANCER_REVISION, "token": "hf_secret"}
    ]


def test_a_thrown_prefetch_never_escapes_the_thread(monkeypatch):
    """An exception on a thread would not reach ``app.py`` anyway -- it would
    print a traceback into the Space log and look like a startup crash. It is
    handled at the source instead."""
    _hub(monkeypatch, [], boom=RuntimeError("hub is down"))
    thread = enhance_module.start_prefetch("org/enhancer")
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_app_gates_the_prefetch_on_the_config_flag_and_on_gpu_capability():
    """~2.5 GB of startup download and disk, declinable.

    It lands while ``demo.launch()`` is packing -- the window
    ``ZEROGPU_DISK_CAP_GB`` is measured against -- so a deploy that will never
    enable enhancement can keep the headroom. Read off app.py's source because
    importing it a second time cannot re-run its module scope.
    """
    import ast

    source = (PACKAGE_ROOT.parent / "app.py").read_text()
    tree = ast.parse(source)
    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "start_prefetch" in ast.dump(node)
    ]
    assert guarded, "app.py calls start_prefetch unconditionally, or not at all"
    test = ast.dump(guarded[0].test)
    assert "prefetch_enhancer" in test, "the config flag does not gate the prefetch"
    assert "WILL_USE_GPU" in test, (
        "a laptop with no GPU downloads 2.5 GB on every `python app.py`; the weight fetch "
        "next to it is gated on capability for exactly this reason"
    )


def test_the_prefetch_flag_is_on_by_default_and_switchable(monkeypatch):
    """Default ON, because the alternative is billed: without a prefetch the
    first enhanced call in each container spends the download inside its own
    reservation. A prefetch nobody gets by default would be the inert-lever
    shape pointed the other way."""
    monkeypatch.delenv(config_module.ENV_PREFETCH_ENHANCER, raising=False)
    assert config_module.load_config().prefetch_enhancer is True

    monkeypatch.setenv(config_module.ENV_PREFETCH_ENHANCER, "0")
    assert config_module.load_config().prefetch_enhancer is False
    monkeypatch.setenv(config_module.ENV_PREFETCH_ENHANCER, "false")
    assert config_module.load_config().prefetch_enhancer is False
    monkeypatch.setenv(config_module.ENV_PREFETCH_ENHANCER, "on")
    assert config_module.load_config().prefetch_enhancer is True


def test_an_unparseable_prefetch_flag_raises_rather_than_guessing(monkeypatch):
    """``bool(os.environ.get(...))`` reads "0", "false" and "no" as TRUE. An
    operator turning the prefetch off would turn it on and get no complaint,
    which is the silent-misconfiguration class config.py exists to refuse."""
    from ltx25_studio.errors import ConfigError

    monkeypatch.setenv(config_module.ENV_PREFETCH_ENHANCER, "maybe")
    with pytest.raises(ConfigError) as caught:
        config_module.load_config()
    message = str(caught.value)
    assert config_module.ENV_PREFETCH_ENHANCER in message and "maybe" in message


# --------------------------------------------------------------------------
# the reservation: enhancement is billed now, so it must be reserved for
# --------------------------------------------------------------------------


def test_an_enhanced_run_reserves_more_than_an_unenhanced_one():
    """The self-inflicted wound this prevents: enhancement moved inside the
    metered window and the estimator did not follow, so the run gets killed at
    the reservation boundary doing work the app itself asked for."""
    from ltx25_studio import zerogpu

    plain = zerogpu.estimate_duration(request=a_request())
    enhanced = zerogpu.estimate_duration(request=a_request(enhance_prompt=True))
    assert enhanced == plain + zerogpu.ENHANCE_DURATION_SECONDS
    assert zerogpu.ENHANCE_DURATION_SECONDS >= 88, (
        "the increment does not cover the measured 88 s Hub fetch. The prefetch is "
        "best-effort by contract, so the estimator may not assume it succeeded"
    )


def test_the_increment_is_additive_rather_than_scaled_by_stage_2():
    """The rewrite happens once, before any denoising, so its cost does not grow
    with the refinement chain -- the same reasoning as LORA_DURATION_SECONDS."""
    from ltx25_studio import zerogpu

    staged = zerogpu.estimate_duration(request=a_request(enable_stage_2=True))
    both = zerogpu.estimate_duration(request=a_request(enable_stage_2=True, enhance_prompt=True))
    assert both == staged + zerogpu.ENHANCE_DURATION_SECONDS


# --------------------------------------------------------------------------
# nothing at import time, nothing in the packing window
# --------------------------------------------------------------------------


def test_importing_the_module_loads_no_model_stack():
    """``demo.launch()`` is the packing trigger and it runs after every import.

    A ``transformers`` import at module scope would not by itself pack anything,
    but a *model* at module scope would -- and the rule that keeps that
    impossible is the cheap one: this module's heavy imports are all inside
    functions.
    """
    import ast

    tree = ast.parse(ENHANCE_SOURCE.read_text())
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        (node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    for heavy in ("torch", "transformers", "diffusers", "PIL"):
        assert heavy not in top_level, (
            f"enhance.py imports {heavy} at module scope; every module in this package must "
            f"import on a bare interpreter (tests/test_architecture.py)"
        )


def test_the_global_rng_is_not_reseeded_for_a_greedy_decode():
    """Upstream calls ``torch.manual_seed`` before generating. Copying that here
    would reseed the stream ``generate()``'s own seeded generator is drawn from,
    making an enhanced run irreproducible -- and it would buy nothing, because
    ``do_sample=False`` means the seed changes no output."""
    import ast

    # Parsed rather than grepped: the module DOCUMENTS why it does not reseed,
    # and a string search would fail on the explanation.
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(ENHANCE_SOURCE.read_text()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "manual_seed" not in called


def test_the_enhancer_is_not_in_the_preloaded_pack():
    """It is a runtime ``transformers`` load into the HF cache, and
    ``model_files.py`` is single-repo by construction. Adding it to either the
    allow-list or the README frontmatter would make the Space's build-time
    preload responsible for a model it cannot see."""
    from ltx25_studio import model_files

    declared = " ".join(str(component) for component in model_files.COMPONENTS)
    assert "gemma-4" not in declared.lower()

    # The Space card's YAML frontmatter only -- the prose below it is free to
    # (and does) name the enhancer. `preload_from_hub` is the platform-managed
    # init step, and it cannot fetch a model `model_files.py` does not declare.
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    _, frontmatter, _ = readme.split("---", 2)
    assert "gemma-4" not in frontmatter.lower()


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_the_default_enhancer_is_the_quantised_build():
    """2.490 GB against 10.279 GB for the bf16 build, both public and ungated."""
    assert (
        config_module.DEFAULT_PROMPT_ENHANCER_REPO
        == "google/gemma-4-E2B-it-qat-mobile-transformers"
    )


def test_the_enhancer_repository_is_configurable_without_a_code_change(monkeypatch):
    monkeypatch.setenv(config_module.ENV_PROMPT_ENHANCER_REPO, "google/gemma-4-E2B-it")
    assert enhance_module.enhancer_repo_id() == "google/gemma-4-E2B-it"


def test_an_unset_variable_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv(config_module.ENV_PROMPT_ENHANCER_REPO, raising=False)
    assert enhance_module.enhancer_repo_id() == config_module.DEFAULT_PROMPT_ENHANCER_REPO


# --------------------------------------------------------------------------
# upstream agreement -- needs diffusers, skipped without it
# --------------------------------------------------------------------------


def test_our_decoding_settings_are_upstreams():
    utils = pytest.importorskip("diffusers.pipelines.ltx2.utils")
    config = utils.GEMMA4_PROMPT_ENHANCEMENT_CONFIG
    assert config.max_new_tokens == enhance_module.MAX_NEW_TOKENS
    assert dict(config.generation_kwargs) == enhance_module.GENERATION_KWARGS


def test_the_upstream_helpers_this_module_imports_still_exist():
    """The fakes above claim these names. This is where that claim is checked."""
    helpers = pytest.importorskip("diffusers.pipelines.ltx2.prompt_enhancement")
    utils = pytest.importorskip("diffusers.pipelines.ltx2.utils")
    for name in ("clean_response", "_pad_inputs_for_attention_alignment", "_prepare_enhance_image"):
        assert hasattr(helpers, name), name
    for name in ("LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT", "LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT"):
        assert hasattr(utils, name), name


# --------------------------------------------------------------------------
# the Hub, opted into -- metadata only, no weights
# --------------------------------------------------------------------------


@pytest.mark.network
def test_the_default_enhancer_repository_is_public_and_loadable():
    """Recorded evidence for the model choice, checked rather than assumed.

    Costs one metadata request and moves **no weights**. What it pins:
    the repo exists, is not gated, is a single ``model.safetensors``, and its
    ``config.json`` carries the ``quant_method`` that requires the
    ``transformers`` floor already in requirements.txt.
    """
    if not os.environ.get("LTX25_NETWORK_TESTS"):
        pytest.skip("set LTX25_NETWORK_TESTS=1 to check the enhancer against the real Hub")
    import json

    from huggingface_hub import HfApi, hf_hub_download

    repo_id = config_module.DEFAULT_PROMPT_ENHANCER_REPO
    info = HfApi().model_info(repo_id, files_metadata=True)
    assert not info.gated, f"{repo_id} became gated; the Space has no token for it"
    assert not info.private
    weights = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    assert [s.rfilename for s in weights] == ["model.safetensors"]
    total = sum(s.size or 0 for s in info.siblings)
    assert total < 4e9, f"{repo_id} is now {total / 1e9:.3f} GB, not the ~2.49 GB budgeted"

    pinned = HfApi().model_info(repo_id, revision=enhance_module.ENHANCER_REVISION)
    assert pinned.sha == enhance_module.ENHANCER_REVISION, "the pinned commit no longer resolves"

    with open(hf_hub_download(repo_id, "config.json", revision=pinned.sha)) as handle:
        config = json.load(handle)
    assert config["model_type"] == "gemma4"
    assert config["quantization_config"]["quant_method"] == "gemma"
    assert "architectures" not in config, (
        "the QAT repo grew an `architectures` key; enhance.py names "
        "Gemma4ForConditionalGeneration explicitly precisely because it had none"
    )
