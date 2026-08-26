"""Prompt enhancement: the user's line, rewritten by Gemma, before any GPU work.

PLAN §3.5. That section said "unavailable by default; we ship no dead toggle",
and both halves of its reasoning have since been checked and found wrong:

* ⛔ **"the pack does not ship it, so `enable_prompt_enhancement=True` fails on
  our subset"** — the enhancer is not a pipeline component we have to resolve
  from the snapshot. Upstream lists it in ``_optional_components`` and types it
  ``Gemma4ForConditionalGeneration | None``: a plain **transformers** class.
  ``model_index.json`` ships ``prompt_enhancer: [None, None]`` precisely because
  it is expected to be *constructed*, not resolved. So it is an ordinary
  ``from_pretrained`` into the HF cache, invisible to ``model_files.py`` and
  ``preflight.py`` (both single-repo by construction) and absent from
  ``preload_from_hub`` on purpose.
* ⛔ **"ZeroGPU charges it twice at the packing peak"** — that was a *placement*
  artefact, not a property of the model. Packing happens exactly once per
  container, at ``demo.launch()``, over the tensors registered while the module
  scope ran. Nothing here runs at import, nothing here is on the GPU, and by the
  time a user presses Generate the packing window has closed. The enhancer is
  charged once, as a download: **2.490 GB** for the QAT default.

**Where this runs, and why that is the whole design.** In the *web* process,
before ``generate()`` dispatches to the ``@spaces.GPU`` entrypoint — so a
generation with enhancement on costs **exactly the same GPU quota** as one
without. The trade is real and is stated in the UI: it is CPU work, so it adds
wall-clock the user waits through. That is the right way round for a metered
Space, and it is the only placement that is free.

**On the CPU, and it stays there.** ``ENHANCER_DEVICE`` is not a lever. Touching
``torch.cuda`` from the web process is the failure ``app.py``'s docstring warns
about — a blocked lazy CUDA init that *hangs* with no error — and moving the
model to CUDA is also the only thing that could put it back in the packer's way.
The freeing path below therefore reads the model's *own* parameter device rather
than asking ``torch.cuda.is_available()``, so the CUDA module is never touched
on a run that never used it.

**Loaded per call and freed in the same call**, so the enhancer never coexists
with the 22B transformer. A cached singleton would be faster and would hold
~2.5 GB of RAM across every subsequent generation for the sake of the minority
of runs that enable it.

**Upstream's shape, not our invention.** ``diffusers.pipelines.ltx2`` already
owns this: :func:`clean_response`, :func:`_pad_inputs_for_attention_alignment`
and :func:`_prepare_enhance_image` are imported rather than reimplemented, and
:data:`MAX_NEW_TOKENS` / :data:`GENERATION_KWARGS` restate
``GEMMA4_PROMPT_ENHANCEMENT_CONFIG``. One deliberate difference from
``LTX2Pipeline.enhance_prompt``, and it is a fix rather than a divergence:
upstream calls ``torch.manual_seed(seed)`` before generating. Decoding here is
**greedy** (``do_sample=False``), so that seed changes nothing about the output
— it only reseeds the global RNG, which in this app is the stream the video's
own seeded generator is drawn from. Reseeding it would make an enhanced run
irreproducible for no benefit at all, which is why there is no seed control for
enhancement and no ``manual_seed`` call here (PLAN §3.5 banked the same finding).

Imports nothing heavy at module scope: ``torch``, ``transformers`` and
``diffusers`` are all resolved inside the functions that need them, so this
module imports on a bare interpreter like every other one in the package.
"""

from __future__ import annotations

import gc
import logging

from .errors import PromptEnhancementError

logger = logging.getLogger(__name__)

__all__ = [
    "ENHANCER_DEVICE",
    "GENERATION_KWARGS",
    "MAX_NEW_TOKENS",
    "enhance",
    "enhance_if_requested",
    "enhancer_repo_id",
]

#: ``GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens`` upstream, and the number
#: PLAN §3.5 banked in advance. Not a UI control: the caption length is what the
#: system prompt is written for, and a user shortening it gets a truncated
#: caption rather than a cheaper one.
MAX_NEW_TOKENS = 600

#: ``GEMMA4_PROMPT_ENHANCEMENT_CONFIG.generation_kwargs`` upstream. **Greedy** —
#: which is what makes ``prompt_enhancement_seed`` inert and why this app offers
#: no seed for it. See the module docstring.
GENERATION_KWARGS = {"do_sample": False, "no_repeat_ngram_size": 5}

#: Where the enhancer lives, start to finish. See the module docstring for why
#: this is a constant and not a configuration knob.
ENHANCER_DEVICE = "cpu"


def enhancer_repo_id() -> str:
    """The Hub repo the enhancer is loaded from, from the runtime config."""
    from .config import load_config

    return load_config().prompt_enhancer_repo


def enhance_if_requested(request, *, enhancer=None) -> tuple[str | None, tuple[str, ...]]:
    """The whole decision, as ``(prompt or None, notes)``.

    Called **unconditionally** by ``generate()`` on every run, and returns
    ``(None, ())`` immediately when the toggle is off. That shape is deliberate:
    a module reached only by a non-default toggle is a module nothing on the
    generation path executes, which is the exact state four of this build's
    blockers shipped in (``tests/test_module_reachability.py``).

    ``None`` means "leave ``request.prompt`` exactly as it is". The caller does
    the ``replace``, so this module never touches the request dataclass.

    ``enhancer`` is the injected seam, matching ``pipeline_factory`` and
    ``encoder`` in ``generate.py``: a callable taking ``(prompt, image=...)``
    and returning the rewritten prompt. Tests use it to exercise every branch
    here with no model, no download and no torch.
    """
    if not getattr(request, "enhance_prompt", False):
        return (None, ())

    original = (getattr(request, "prompt", "") or "").strip()
    if not original:
        # Not an error: several modes send an empty prompt by design, and a user
        # who ticks the box on one of those has asked for something that cannot
        # mean anything. Said out loud rather than swallowed, because the UI
        # would otherwise imply an enhancement that never happened.
        return (
            None,
            (
                "prompt enhancement was requested but the prompt was empty, so there was "
                "nothing to enhance and the run used the prompt exactly as it was.",
            ),
        )

    image, image_notes = _reference_frame(getattr(request, "image", None))
    run = enhancer if enhancer is not None else enhance
    enhanced = (run(original, image=image) or "").strip()
    if not enhanced:
        # A blank rewrite would replace a working prompt with nothing and render
        # as a successful enhanced run. Raise instead.
        raise PromptEnhancementError(
            repo_id=enhancer_repo_id(),
            reason=(
                "it returned an empty prompt, which would have replaced the prompt you typed "
                "with nothing"
            ),
        )

    logger.info("prompt enhanced: %d chars in, %d chars out", len(original), len(enhanced))
    return (enhanced, (*image_notes, _rewrite_note(enhanced)))


def enhance(prompt: str, *, image=None, repo_id: str | None = None, loader=None) -> str:
    """Load the enhancer, rewrite one prompt, free the enhancer. In that order.

    ``loader`` is the seam that keeps this function testable offline: it takes a
    repo id and returns ``(model, processor)``. The default hits the Hub.
    """
    repo_id = repo_id or enhancer_repo_id()
    load = loader if loader is not None else _load_enhancer
    model, processor = load(repo_id)
    try:
        return _rewrite(model, processor, prompt=prompt, image=image, repo_id=repo_id)
    finally:
        # Locals are dropped HERE, in the frame that owns the only references,
        # so the enhancer is gone before the caller returns and can never be
        # resident at the same time as the transformer.
        on_cuda = _is_on_cuda(model)
        del model, processor
        gc.collect()
        if on_cuda:
            # Unreachable while ENHANCER_DEVICE is "cpu", and guarded on the
            # model's OWN device rather than on `torch.cuda.is_available()`:
            # a parameter can only be on CUDA if CUDA was already initialised,
            # so this never initialises it. See the module docstring.
            import torch

            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _rewrite_note(enhanced: str) -> str:
    """What the user must be able to read after the run.

    An enhancer that silently rewrites the prompt is worse than none: the box
    still shows what they typed, the video shows something else, and there is
    nothing on screen that connects the two. ``ui/render.py`` prints every entry
    in ``GenerationResult.warnings``, so quoting the rewritten prompt here is
    what makes the substitution visible.
    """
    return (
        "prompt enhancement rewrote your prompt, so the model was NOT given what you typed. "
        f"The prompt actually sent was: {enhanced!r}"
    )


def _reference_frame(image):
    """The conditioning image, if it is one the enhancer can actually look at.

    Returns ``(image_or_None, notes)``. Upstream's ``_prepare_enhance_image``
    accepts ``PIL.Image.Image`` and raises on anything else, and the I2V system
    prompt is written around a reference frame that is genuinely present -- so a
    non-PIL image degrades to text-only enhancement **and says so**, rather than
    failing a run over a detail the user did not choose. ``gr.Image(type="pil")``
    means the UI always takes the first branch.
    """
    if image is None:
        return (None, ())
    try:
        import PIL.Image
    except Exception:  # noqa: BLE001 -- pillow absent is not worth failing a run
        return (None, ("prompt enhancement ran text-only: pillow is not installed.",))
    if isinstance(image, PIL.Image.Image):
        return (image, ())
    return (
        None,
        (
            f"prompt enhancement ran text-only: the conditioning image arrived as "
            f"{type(image).__name__}, and enhancing against a reference frame needs a PIL "
            f"image. The prompt was enhanced without looking at your image.",
        ),
    )


def _load_enhancer(repo_id: str):
    """``(model, processor)`` from the Hub, or one of ours naming the repo.

    Two failures are worth separating and both are named: ``transformers`` too
    old to know ``Gemma4ForConditionalGeneration`` at all, and the fetch/load
    itself. Neither ever falls through to the original prompt.
    """
    try:
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    except Exception as exc:
        raise PromptEnhancementError(
            repo_id=repo_id,
            reason=(
                f"transformers could not provide Gemma4ForConditionalGeneration ({exc!r}). "
                f"The enhancer needs transformers >= 5.15.1, which is the floor in "
                f"requirements.txt"
            ),
        ) from exc

    try:
        # The processor supplies BOTH halves upstream's `enhance_prompt` uses:
        # `.tokenizer` for the chat template and the decode, and the image prep
        # for the I2V route.
        processor = AutoProcessor.from_pretrained(repo_id)
        # Named explicitly rather than through `AutoModel*`: the QAT repo's
        # config.json has no `architectures` key, so auto-dispatch cannot
        # resolve it. `dtype="auto"` honours the checkpoint's own dtype and the
        # `quantization_config` block the QAT build carries.
        model = Gemma4ForConditionalGeneration.from_pretrained(repo_id, dtype="auto")
    except Exception as exc:
        raise PromptEnhancementError(repo_id=repo_id, reason=repr(exc)) from exc

    # `.eval()` and an explicit device, in that order. The device is stated
    # rather than inherited so a global default device cannot quietly move a
    # 2.5 GB model onto the GPU in the web process.
    return model.eval().to(ENHANCER_DEVICE), processor


def _system_prompts():
    """Upstream's two system prompts, imported rather than copied.

    Copying either would mean a 2,000-word prompt maintained in two places, and
    the one in this repo would be the stale one the moment upstream tuned it.
    """
    from diffusers.pipelines.ltx2.utils import (
        LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT,
        LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT,
    )

    return LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT, LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT


def _rewrite(model, processor, *, prompt: str, image, repo_id: str) -> str:
    """One greedy generation, in the shape ``LTX2Pipeline.enhance_prompt`` uses."""
    try:
        import torch
        from diffusers.pipelines.ltx2.prompt_enhancement import (
            _pad_inputs_for_attention_alignment,
            _prepare_enhance_image,
            clean_response,
        )

        t2v_system_prompt, i2v_system_prompt = _system_prompts()
    except Exception as exc:
        raise PromptEnhancementError(
            repo_id=repo_id,
            reason=f"the upstream enhancement helpers could not be imported ({exc!r})",
        ) from exc

    try:
        # PLAN §3.5: the system prompt is auto-selected on whether there is a
        # reference frame. Upstream's `__call__` only ever passes the T2V one
        # because it never forwards an image; the I2V prompt exists for exactly
        # this call and is what makes an image-conditioned enhancement describe a
        # video that BEGINS from the frame rather than one that merely matches it.
        if image is None:
            system_prompt = t2v_system_prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"user prompt: {prompt}"},
            ]
            enhance_image = None
        else:
            system_prompt = i2v_system_prompt
            enhance_image = _prepare_enhance_image(image)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                    ],
                },
            ]

        template = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = processor(text=template, images=enhance_image, return_tensors="pt").to(
            ENHANCER_DEVICE
        )
        pad_token_id = processor.tokenizer.pad_token_id
        model_inputs = _pad_inputs_for_attention_alignment(
            model_inputs, pad_token_id=pad_token_id if pad_token_id is not None else 0
        )

        # No `torch.manual_seed` here, deliberately -- see the module docstring.
        with torch.no_grad():
            sequences = model.generate(
                **model_inputs, max_new_tokens=MAX_NEW_TOKENS, **GENERATION_KWARGS
            )

        generated = [seq[len(model_inputs.input_ids[i]) :] for i, seq in enumerate(sequences)]
        decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
    except Exception as exc:
        raise PromptEnhancementError(
            repo_id=repo_id, reason=f"it failed while rewriting the prompt ({exc!r})"
        ) from exc

    if not decoded:
        raise PromptEnhancementError(repo_id=repo_id, reason="it decoded to no sequences at all")
    return clean_response(decoded[0])


def _is_on_cuda(model) -> bool:
    """Whether the model actually holds a CUDA tensor. Never initialises CUDA."""
    try:
        for parameter in model.parameters():
            return parameter.device.type == "cuda"
    except Exception:  # noqa: BLE001 -- a fake model in a test has no parameters
        return False
    return False
