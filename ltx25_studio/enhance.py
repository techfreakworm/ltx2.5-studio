"""Prompt enhancement: the user's line, rewritten by Gemma, before any GPU work.

PLAN §3.5.

**Where this runs, and why it moved.** Inside the ``@spaces.GPU`` worker, on the
GPU, immediately after validation and before the transformer is built. It did
not start there. It shipped in the *web* process, on the CPU, deliberately --
and that placement was correct about the thing it was reasoning about and wrong
about the thing it never measured.

⛔ **The web-process placement, and the measurement that ended it.** On the live
Space, ``text_to_video`` at 256x256x9 -- the cheapest configuration that exists
here::

    enhance OFF          15.38 s
    enhance ON, cold    654.15 s   (+638.8)
    enhance ON, warm    553.45 s   (+538.1)

The cold/warm split is what isolates the cause. The 2.46 GB Hub fetch takes
88 s and the weights load in under a second, so the **download is not the
cost**: ~538 s of the ~554 s warm figure is the CPU decode of 600 greedy tokens,
about **0.9 s/token**. The estimate that justified running it on the CPU was
"1-2 s" -- a *GPU* number applied to a *CPU* placement. 600 tokens in 2 s is
300 tok/s, which no CPU does on a 5B-parameter model. Nine minutes of wall-clock
were priced at two seconds because nobody put a stopwatch on the placement they
had chosen.

**The transferable lesson, which should outlive this feature: "where does it
run" has two axes -- disk and latency -- and only one of them was optimised.**
The disk argument was researched, written down, checked against the packer's
source and proved correct. The latency argument was never taken. A placement
decision that answers one axis and is silent on the other is not a decision, it
is half of one, and the half that is missing is the half that reaches the user.

✅ **The disk argument survives the move intact, and that is the crux.** Moving
to CUDA inside the worker does NOT put the enhancer back in the packer's way:

* ZeroGPU packing fires exactly **once per container**, via ``torch.pack()`` at
  ``demo.launch()`` (``spaces/zero/__init__.py:32``), and packs only tensors
  registered in ``cuda_aliases`` during the module-scope import phase.
* Inside a ``@spaces.GPU`` call the forked worker has already run
  ``torch.unpatch()`` (``wrappers.py:147``), so ``.to("cuda")`` there is a real
  H2D copy that is **never packed**.
* In-repo precedent already shipping: ``pipelines.py`` moves the stage-2
  upsampler and diffusion decoder to CUDA inside the worker for the same reason.

So the enhancer stays ~2.5 GB in the HF cache, never packed, never in the
snapshot, never in ``preload_from_hub`` -- and it gets CUDA speed. The cost is
~1-2 s of billed wall per enhanced generation, against ~85 s for a real
generation: under 5% overhead, where the CPU placement was **+3500%**.

**What it now costs, and who is told.** Enhancement is inside the metered
window, so ``zerogpu.estimate_duration`` adds
:data:`~ltx25_studio.zerogpu.ENHANCE_DURATION_SECONDS` to the reservation when
the toggle is set, and the UI caption says the run costs a little quota rather
than none. An enhanced run that got killed at the reservation boundary would be
a self-inflicted wound.

**Prefetch, so the download is never billed.** :func:`prefetch` pulls the
weights into the HF cache at startup, on the CPU, as a **download only** -- see
its docstring for why that cannot enter the packing window. Without it the first
enhanced call in each container spends 88 s of *billed GPU time* on a file
transfer. It is best-effort by construction: a Space whose prefetch failed still
starts and still serves un-enhanced generations, and its first enhanced call is
merely slow.

**Loaded per call and freed in the same call**, and the claim that buys is worth
stating precisely rather than generously. It is *not* "the enhancer never
coexists with the transformer": inside the worker the transformer has already
been restored to VRAM before this runs, so for the length of the rewrite they
are both resident. What is true, and what matters, is that the enhancer is
**gone before denoising starts** -- ``enhance()`` frees it in the same frame
that loaded it, and ``generate()`` calls it above the pipeline factory -- so the
memory peak, which is denoising activations, never also carries 2.5 GB of Gemma.
A cached singleton would hold that 2.5 GB across every subsequent generation for
the sake of the minority of runs that enable it.

The free path (``del`` + ``gc.collect()`` + ``torch.cuda.empty_cache()``) is
guarded on the model's **own** parameter device rather than on
``torch.cuda.is_available()``, so a run that never touched CUDA never
initialises it.

**Upstream's shape, not our invention.** ``diffusers.pipelines.ltx2`` already
owns this: :func:`clean_response`, :func:`_pad_inputs_for_attention_alignment`
and :func:`_prepare_enhance_image` are imported rather than reimplemented, and
:data:`MAX_NEW_TOKENS` / :data:`GENERATION_KWARGS` restate
``GEMMA4_PROMPT_ENHANCEMENT_CONFIG``. One deliberate difference from
``LTX2Pipeline.enhance_prompt``, and it is a fix rather than a divergence:
upstream calls ``torch.manual_seed(seed)`` before generating. Decoding here is
**greedy** (``do_sample=False``), so that seed changes nothing about the output
-- it only reseeds the global RNG, which in this app is the stream the video's
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
    "GENERATION_KWARGS",
    "MAX_NEW_TOKENS",
    "enhance",
    "enhance_if_requested",
    "enhancer_device",
    "enhancer_repo_id",
    "prefetch",
    "start_prefetch",
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

def enhancer_repo_id() -> str:
    """The Hub repo the enhancer is loaded from, from the runtime config."""
    from .config import load_config

    return load_config().prompt_enhancer_repo


def enhancer_device() -> str:
    """``"cuda"`` where there is a GPU to use, ``"cpu"`` where there is not.

    **Resolved per call rather than fixed at import**, because the answer
    differs by *process*, not by deployment. This used to be the constant
    ``ENHANCER_DEVICE = "cpu"``, and the reasoning behind that constant was
    sound for where the call then lived: probing ``torch.cuda`` from the web
    process is the blocked lazy CUDA init ``app.py``'s docstring warns about,
    which *hangs* with no error. The call now runs inside the ``@spaces.GPU``
    worker, past ``torch.unpatch()``, where CUDA is already initialised and the
    probe is free -- so the constant would today be pinning the enhancer to the
    slow device for a reason that no longer applies. See the module docstring
    for the measurement that moved it.

    Never raises. A host with no torch, or a torch built without CUDA, answers
    ``"cpu"`` and the enhancement still runs -- slowly, which is the correct
    outcome for a machine that has nothing faster.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 -- no torch, or no CUDA build; both mean cpu
        return "cpu"
    return "cpu"


def prefetch(repo_id: str | None = None, *, token=None) -> str | None:
    """Pull the enhancer's weights into the HF cache. **A download, nothing more.**

    Returns the snapshot directory, or ``None`` if the fetch did not succeed.
    **Never raises** -- see "best-effort" below.

    ⚠️ **Why this cannot enter the ZeroGPU packing window, since the next reader
    will worry about exactly that.** Packing captures tensors registered in
    ``cuda_aliases`` while the module scope runs -- i.e. things that called
    ``.to("cuda")`` before ``demo.launch()``. ``snapshot_download`` constructs
    **no torch tensors at all**: it resolves a file list over HTTP and writes
    blobs to ``~/.cache/huggingface/hub``. There is nothing for ``torch.pack()``
    to see, on CUDA or otherwise. Files on disk are not packable state; only
    alias registrations are. This is what lets the weights be present before the
    first request without the enhancer ever entering the disk budget the
    web-process placement was originally chosen to protect.

    **Why ``snapshot_download`` and not the two alternatives.**

    * Not ``from_pretrained`` at startup. That would do the job *and* build a
      live 2.5 GB model object during the module-scope import phase -- the exact
      window this build has been burned in -- and hold it in RAM for the
      container's life to save a sub-second load. The prefetch must move bytes,
      not construct models.
    * Not ``hf_hub_download`` per file, the ``preflight.ensure_weights`` idiom.
      That needs a filename allow-list, and this repo is **configurable**
      (``LTX25_PROMPT_ENHANCER_REPO``): any list we hardcoded would be wrong the
      moment an operator pointed it at the bf16 build, which has a different
      file layout. ``snapshot_download`` resolves the file list from the Hub
      itself, which is the property that makes it correct under a repo we do not
      control. It fetches the repo as-is -- no ``allow_patterns``, deliberately:
      a pattern that guessed wrong would skip the weights and leave a prefetch
      that "succeeded" and cached nothing, which is worse than fetching a little
      extra.

    **Best-effort, and that is a contract.** Startup must not fail, and
    readiness must not wait, because the enhancer could not be fetched -- an
    un-enhanced generation is unaffected by any of this. A failed prefetch
    degrades the first enhanced call to paying the download inside its
    reservation, which is what happened before the prefetch existed. Degraded,
    not broken. So every failure is logged and swallowed here.

    **What happens if a user presses Generate while this is still running**, the
    obvious question and the reassuring answer: ``huggingface_hub`` takes a
    cross-process ``WeakFileLock`` on a per-blob ``{etag}.lock`` before writing,
    so the worker's ``from_pretrained`` blocks on the same lock rather than
    racing it or corrupting the cache. The worker waits for the in-flight fetch
    and then loads the completed file. That wait is inside the reservation,
    which is the other reason ``ENHANCE_DURATION_SECONDS`` is sized for a whole
    download rather than for a decode.
    """
    repo_id = repo_id or enhancer_repo_id()
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, token=token)
    except Exception as exc:  # noqa: BLE001 -- see "best-effort" above
        logger.warning(
            "prompt-enhancer prefetch failed for %r (%r). The Space is unaffected: "
            "un-enhanced generations run normally, and the first enhanced run will pay "
            "the download inside its own GPU reservation.",
            repo_id,
            exc,
        )
        return None
    logger.info("prompt-enhancer weights cached at %s", path)
    return str(path)


def start_prefetch(repo_id: str | None = None, *, token=None):
    """Run :func:`prefetch` on a daemon thread. Returns it, already started.

    **A thread because readiness is the requirement.** The fetch is ~88 s for
    the QAT default and ``app.py`` calls this before ``demo.launch()``; doing it
    inline would hold the Space unavailable for a minute and a half to prepare
    an optional feature. A daemon thread cannot keep the process alive at exit
    either, so a prefetch still in flight when the Space stops does not delay
    the stop.

    The thread does pure network and disk I/O and touches no torch, so it adds
    nothing to the packing window it runs alongside -- see :func:`prefetch`.
    """
    import threading

    thread = threading.Thread(
        target=prefetch,
        args=(repo_id,),
        kwargs={"token": token},
        name="ltx25-enhancer-prefetch",
        daemon=True,
    )
    thread.start()
    return thread


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
    # Read off the MODEL rather than re-probing: the inputs must land wherever
    # the weights actually did, and an injected test loader that moved nothing
    # answers "cpu" without anyone having to tell it to.
    device = _model_device(model)
    try:
        return _rewrite(
            model, processor, prompt=prompt, image=image, repo_id=repo_id, device=device
        )
    finally:
        # Locals are dropped HERE, in the frame that owns the only references,
        # so the enhancer is gone before the caller returns -- and `generate()`
        # calls this above the pipeline factory, so it is gone before denoising
        # starts. That is the honest version of the claim: inside the worker the
        # transformer is already VRAM-resident while the rewrite runs, so the
        # two do overlap. What must not happen is 2.5 GB of Gemma still being
        # held at the memory peak, which is the activations.
        on_cuda = device == "cuda"
        del model, processor
        gc.collect()
        if on_cuda:
            # Guarded on the model's OWN device rather than on
            # `torch.cuda.is_available()`: a parameter can only be on CUDA if
            # CUDA was already initialised, so this never initialises it. See
            # the module docstring.
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

    # `.eval()` and an explicit device, in that order. The device is STATED
    # rather than inherited: a global default device deciding where 2.5 GB of
    # Gemma lands is exactly the kind of action-at-a-distance that made the
    # original CPU pin necessary. It is now resolved rather than hardcoded --
    # see `enhancer_device`.
    return model.eval().to(enhancer_device()), processor


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


def _rewrite(model, processor, *, prompt: str, image, repo_id: str, device: str) -> str:
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
        # The tensors follow the MODEL. Moving one and not the other is a real
        # failure mode and a silent one on a CPU-only host, where both happen to
        # be right by accident.
        model_inputs = processor(text=template, images=enhance_image, return_tensors="pt").to(
            device
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


def _model_device(model) -> str:
    """The device the model's parameters are actually on. Never initialises CUDA.

    Asked of the model rather than of ``torch``, which is what makes it safe: a
    parameter can only report ``cuda`` if CUDA was already initialised to put it
    there, so reading this can never be the thing that initialises it. A model
    with no parameters -- every fake in the test suite -- answers ``"cpu"``,
    which is also the honest answer for a load that never moved anything.
    """
    try:
        for parameter in model.parameters():
            return "cuda" if parameter.device.type == "cuda" else "cpu"
    except Exception:  # noqa: BLE001 -- a fake model in a test has no parameters
        return "cpu"
    return "cpu"
