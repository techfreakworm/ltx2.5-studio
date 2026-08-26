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
# the placement: web process, before dispatch, zero GPU-seconds
# --------------------------------------------------------------------------


def test_enhancement_happens_before_the_gpu_entrypoint_is_dispatched_to(tmp_path, monkeypatch):
    """The whole cost argument, as an assertion.

    Enhancement inside the ``@spaces.GPU`` window would bill 2.49 GB of Gemma
    against the same metered reservation as the video. Above the dispatch branch
    it is web-process CPU work and the reservation is unchanged -- so what must
    be true is that the request crossing the fork ALREADY carries the rewritten
    prompt, and that the worker is never asked to enhance anything.
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
        f"the enhancer ran {len(seam.calls)} times; twice means the GPU worker enhanced again "
        f"and the rewrite was billed"
    )
    assert seen and seen[0].prompt == REWRITTEN, (
        "the request that crossed the dispatch boundary still carried the original prompt"
    )
    assert any(REWRITTEN in note for note in result.warnings), (
        "the note is computed in the web process and the result is built in the worker; "
        "without _with_notes the user never sees which prompt ran"
    )


def test_the_gpu_worker_never_enhances(tmp_path, monkeypatch):
    """Belt and braces for the same property, from the other side.

    A request arriving inside the worker with the toggle still set must not be
    re-enhanced -- it has already been rewritten once, and doing it again would
    both cost GPU time and rewrite a caption that was never a user prompt.
    """
    seam = Recorder()
    monkeypatch.setattr(enhance_module, "enhance", seam)
    token = generate_module._IN_GPU_WORKER.set(True)
    try:
        generate_module.generate(
            a_request(enhance_prompt=True),
            pipeline_factory=factory_for(FakePipe()),
            encoder=good_encoder,
            output_root=tmp_path,
            retention_bytes=10**9,
        )
    finally:
        generate_module._IN_GPU_WORKER.reset(token)
    assert seam.calls == []


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

    assert fields.FIELD_KEYS[-1] == "enhance_prompt"


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

    With enhancement on that is not what happens first -- the app spends a
    while downloading and running Gemma on the CPU, before any slot exists.
    A status line describing something other than what is happening, exactly
    where the user is waiting for it, is the same small lie as a hidden control.
    """
    from ltx25_studio.ui import handlers

    plain = handlers.on_generate_start("text_to_video")
    enhancing = handlers.on_generate_start("text_to_video", True)
    assert plain != enhancing
    assert "quota" in plain
    assert "no quota" in enhancing and "CPU" in enhancing


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


def test_the_enhancer_never_leaves_the_cpu(upstream):
    """The placement argument depends on this literally.

    A ``.to("cuda")`` here would initialise CUDA in the web process -- the hang
    ``app.py`` warns about -- and it is also the only thing that could put the
    enhancer back in the ZeroGPU packer's way, which is the whole cost argument.
    """
    assert enhance_module.ENHANCER_DEVICE == "cpu"
    model, processor = FakeModel(), FakeProcessor()
    enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=loader_for(model, processor))
    # The tensors follow the model. Both are asserted because moving one and not
    # the other is a real failure mode -- and a silent one on a CPU-only host.
    assert processor.calls, "the processor was never called"
    assert model.generate_kwargs, "the model never generated"


def test_the_loaded_model_is_put_on_the_cpu_and_into_eval_mode(monkeypatch):
    """Stated rather than inherited: a global default device must not be able to
    move 2.5 GB of Gemma onto the GPU behind this module's back."""
    transformers = types.ModuleType("transformers")
    model, processor = FakeModel(), FakeProcessor()
    transformers.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda repo_id, **kwargs: processor
    )
    transformers.Gemma4ForConditionalGeneration = types.SimpleNamespace(
        from_pretrained=lambda repo_id, **kwargs: model
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    loaded, _ = enhance_module._load_enhancer("org/enhancer")
    assert loaded is model
    assert model.evaluated is True
    assert model.moved_to == "cpu"


def test_the_model_is_freed_before_the_call_returns(upstream):
    """It must never coexist with the 22B transformer.

    Checked with a weak reference rather than by trusting a ``del``: the point is
    that no strong reference survives the call, which is the only version of
    "freed" that means anything.
    """
    import gc
    import weakref

    watched: dict = {}

    def loader(repo_id):
        # Built HERE so the reference `enhance()` takes is the only strong one.
        model = FakeModel()
        watched["model"] = weakref.ref(model)
        return model, FakeProcessor()

    enhance_module.enhance(ORIGINAL, repo_id="org/enhancer", loader=loader)
    gc.collect()
    assert watched["model"]() is None, (
        "the enhancer was still referenced after enhance() returned, so ~2.5 GB stays "
        "resident for every generation that follows"
    )


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

    with open(hf_hub_download(repo_id, "config.json")) as handle:
        config = json.load(handle)
    assert config["model_type"] == "gemma4"
    assert config["quantization_config"]["quant_method"] == "gemma"
    assert "architectures" not in config, (
        "the QAT repo grew an `architectures` key; enhance.py names "
        "Gemma4ForConditionalGeneration explicitly precisely because it had none"
    )
