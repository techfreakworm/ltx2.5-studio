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
snapshot, never in ``preload_from_hub`` -- and it gets CUDA speed. The decode
itself is ~1-2 s of billed wall against ~85 s for a real generation: under 5%
overhead, where the CPU placement was **+3500%**. What that figure did NOT
include, and what the next section is about, is the load that used to precede
every single decode.

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

**Loaded once per worker and KEPT**, which is the reverse of how it shipped.
The measurement that reversed it is not the one anyone expected.

⛔ **The rule it replaces, and why it was wrong.** The enhancer used to be loaded
to CUDA, used, and freed inside the same call, every call. The argument was VRAM
safety -- 2.5 GB of Gemma must not still be held at the denoise peak -- and that
argument was correct about the thing it reasoned about. What it never measured
was the reload. ✅ **It cost +30.78 s per enhanced generation** (46.53 s against
a 15.75 s baseline at 256x256x9): a 2.49 GB load, every single time, to protect
headroom on a partition whose size nobody in this build had ever read.

✅ **The partition had been read wrong, and that is the actual finding.** The
Space ran its entire life on ZeroGPU's ``large`` size -- 47.375 GiB -- because
``LTX25_ZEROGPU_SIZE=auto`` sends no ``gpuSize`` to the scheduler at all and
ZeroGPU's own xlarge promotion never reaches it either. The four-hop trace is in
``docs/MIG_ALLOCATOR_SIGHTINGS.md``. Set to ``xlarge``, the same worker reports::

    before   47.375 GiB total   headroom  1.19 GiB   -> residency impossible
    after    94.971 GiB total   headroom ~53.7 GiB   -> residency trivially affordable

✅ The largest peak this build has ever measured is stage 2 at 704x480x49:
**55.906 GiB reserved**. That, plus a permanently resident 2.49 GB, is ~58.4 GiB
against 94.971 GiB. The VRAM argument for freeing did not get weaker. It stopped
applying.

✅ **The forked worker persists, so the cache actually hits.** This is the one
way the change could silently do nothing, so it was established from ``spaces``'
own source *before* anything was built rather than assumed afterwards:

* ``spaces/zero/wrappers.py`` runs the forked child in a ``while True:`` loop
  blocking on ``arg_queue.get()``. One process serves **many** tasks; it does not
  exit when a task returns.
* ``worker_init`` -- the ``torch.unpatch()`` / ``torch.init()`` / ``torch.move()``
  that makes CUDA usable in the child -- is guarded by an ``initialized`` flag and
  runs **once per process**. That guard is only meaningful because the process
  outlives the call.
* The web process keeps the child in ``workers[nvidia_index]`` on a successful
  result and reuses it when ``worker.process.is_alive() and
  schedule_response.idle``.

So module state written during call N is still there for call N+1 in that worker.
Corroborated by this build's own record: the enhancer sighting in
``docs/MIG_ALLOCATOR_SIGHTINGS.md`` turns entirely on the observation that *"the
second call in the same worker succeeded"*.

⚠️ **It is a warm path with a cold fallback, not a guarantee**, and that is the
honest version of the claim. Reuse needs the scheduler to report the slot idle
and the child to still be alive. Three things send it back to cold: a new
container, a reaped worker, and -- worth naming, because this module can cause it
-- **any exception crossing the fork**, since the web process returns the worker
to ``workers`` only on an ``OkResult``. A :class:`PromptEnhancementError`
therefore costs the resident copy and the next enhanced call pays the load again.
That is correct rather than a regression: a reload is the cheap half of what a
failed enhancement already cost the user.

⛔ **Still nothing at module scope, and this is the constraint most likely to be
broken by a later "optimisation".** :data:`_ENHANCER` is a module-level dict that
is **empty at import** and filled on first use, inside the worker, past the fork.
Hoisting the load to module scope -- *"we know we need it, why not have it ready
at startup"* -- would put a 2.49 GB CUDA allocation inside ZeroGPU's packing
window, which fires exactly once at ``demo.launch()`` via ``torch.pack()`` and
captures whatever registered in ``cuda_aliases`` during the import phase. That
would put the enhancer straight back into the 72 GB disk budget this whole
feature was shaped around avoiding, and it would do it **silently**. Lazy first
use, then keep. Never at import.

The free path is not deleted, it is **moved**: :func:`clear_enhancer_cache` still
does ``del`` + ``gc.collect()`` + a ``torch.cuda.empty_cache()`` guarded on the
model's **own** parameter device rather than on ``torch.cuda.is_available()``, so
a process that never touched CUDA never initialises it. Nothing on the generation
path calls it -- it exists so that dropping the enhancer is an operation somebody
can actually perform rather than a comment describing one.

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
    "ENHANCER_REVISION",
    "GENERATION_KWARGS",
    "MAX_NEW_TOKENS",
    "clear_enhancer_cache",
    "enhance",
    "enhance_if_requested",
    "enhancer_device",
    "enhancer_repo_id",
    "enhancer_revision",
    "prefetch",
    "start_prefetch",
]

#: The commit ``config.DEFAULT_PROMPT_ENHANCER_REPO``
#: (``google/gemma-4-E2B-it-qat-mobile-transformers``) is loaded at: its head
#: on 2026-08-27 and still on 2026-11-16. The repo's chat template, which the
#: rewrite is built from, changed twice in July 2026. Applies to the default
#: repo only; a repo set through ``LTX25_PROMPT_ENHANCER_REPO`` loads at its head.
ENHANCER_REVISION = "dd693ff40353f057ca5f07e945ad867f4afbf2ec"

#: ``GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens`` upstream, and the number
#: PLAN §3.5 banked in advance. Not a UI control: the caption length is what the
#: system prompt is written for, and a user shortening it gets a truncated
#: caption rather than a cheaper one.
MAX_NEW_TOKENS = 600

#: ``GEMMA4_PROMPT_ENHANCEMENT_CONFIG.generation_kwargs`` upstream. **Greedy** —
#: which is what makes ``prompt_enhancement_seed`` inert and why this app offers
#: no seed for it. See the module docstring.
GENERATION_KWARGS = {"do_sample": False, "no_repeat_ngram_size": 5}

#: The loaded ``(model, processor)``, kept for the life of the worker process.
#:
#: ⛔ **EMPTY AT IMPORT, AND IT MUST STAY THAT WAY.** This dict is declared here
#: and populated on first use inside the ``@spaces.GPU`` worker -- never at module
#: scope. A ``.to("cuda")`` during the import phase registers in ZeroGPU's
#: ``cuda_aliases`` and is captured by the ``torch.pack()`` that fires once at
#: ``demo.launch()``, which would put 2.49 GB of Gemma back into the 72 GB disk
#: budget the whole feature was shaped around avoiding -- silently. Declaring the
#: container at import costs nothing; *filling* it at import costs the feature.
#:
#: Keyed by ``(repo id, revision)`` because ``LTX25_PROMPT_ENHANCER_REPO`` is a
#: real lever: an operator who repoints it must not be served the previous
#: repo's weights. Cleared by :func:`clear_enhancer_cache`.
_ENHANCER: dict[tuple, tuple] = {}


def enhancer_repo_id() -> str:
    """The Hub repo the enhancer is loaded from, from the runtime config."""
    from .config import load_config

    return load_config().prompt_enhancer_repo


def enhancer_revision(repo_id: str | None = None) -> str | None:
    """The pinned commit for the default repo; ``None`` (the Hub head) for any other."""
    from .config import DEFAULT_PROMPT_ENHANCER_REPO

    repo_id = repo_id or enhancer_repo_id()
    return ENHANCER_REVISION if repo_id == DEFAULT_PROMPT_ENHANCER_REPO else None


def _warn_unpinned(repo_id: str, action: str) -> None:
    logger.warning(
        "prompt enhancer %r is not the default repository, so it is not pinned: %s it at its "
        "current Hub head. Its chat template, and so the rewritten prompts, can change "
        "without notice.",
        repo_id,
        action,
    )


def enhancer_device(device: str | None = None) -> str:
    """``device`` when the caller names one; otherwise ``"cuda"``, then ``"mps"``, then ``"cpu"``.

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

    Never raises. A host with no torch, or a torch with neither CUDA nor MPS,
    answers ``"cpu"`` and the enhancement still runs -- slowly, which is the
    correct outcome for a machine that has nothing faster.
    """
    if device:
        return str(device)
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 -- no torch, or no CUDA build; both mean cpu
        return "cpu"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 -- a torch without the mps backend
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
    revision = enhancer_revision(repo_id)
    if revision is None:
        _warn_unpinned(repo_id, "prefetching")
    try:
        from huggingface_hub import snapshot_download

        # A commit-hash revision also lets later loads resolve cached files
        # with no network round-trip.
        path = snapshot_download(repo_id=repo_id, revision=revision, token=token)
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


def enhance(
    prompt: str, *, image=None, repo_id: str | None = None, loader=None, device: str | None = None
) -> str:
    """Rewrite one prompt with the enhancer, loading it on first use and KEEPING it.

    The load happens at most once per worker process. The reload it replaces cost
    a measured **+30.78 s on every enhanced generation**; see the module docstring
    for that number, for the partition measurement that made residency
    affordable, and for the evidence that the forked worker lives long enough for
    a module-level cache to be hit at all.

    ``loader`` is the seam that keeps this function testable offline: it takes a
    repo id and returns ``(model, processor)``. The default hits the Hub. It is
    cached exactly like the default -- an injected loader that was *not* cached
    would mean the tested lifetime and the shipped lifetime were different
    lifetimes, which is the shape of test theatre this suite exists to avoid.

    ``device`` overrides :func:`enhancer_device` for the default loader.
    """
    repo_id = repo_id or enhancer_repo_id()
    model, processor = _resident_enhancer(repo_id, loader, device=device)
    # Read off the MODEL rather than re-probing: the inputs must land wherever
    # the weights actually did, and an injected test loader that moved nothing
    # answers "cpu" without anyone having to tell it to.
    device = _model_device(model)
    # No `finally: del` any more, and that is the whole change. `generate()` still
    # calls this above the pipeline factory, so the ordering is untouched -- what
    # is gone is the free, which means the next enhanced call in this worker skips
    # the load entirely instead of paying for it again.
    return _rewrite(model, processor, prompt=prompt, image=image, repo_id=repo_id, device=device)


def _resident_enhancer(repo_id: str, loader=None, *, device: str | None = None) -> tuple:
    """The cached ``(model, processor)`` for ``repo_id``, loading it if need be.

    **Only a successful load is cached.** :func:`_load_enhancer` raises
    :class:`~ltx25_studio.errors.PromptEnhancementError` on a bad repo, an offline
    Hub or a too-old ``transformers``, and caching that would turn one transient
    network failure into an enhancer that stays broken for the life of the worker
    with no way to retry.
    """
    revision = enhancer_revision(repo_id)
    cached = _ENHANCER.get((repo_id, revision))
    if cached is not None:
        return cached
    if loader is not None:
        loaded = loader(repo_id)
    else:
        loaded = _load_enhancer(repo_id, revision=revision, device=device)
    _ENHANCER[(repo_id, revision)] = loaded
    return loaded


def clear_enhancer_cache() -> None:
    """Drop every resident enhancer and give its VRAM back.

    **Nothing on the generation path calls this.** It exists because "the
    enhancer is resident" should be a state something can leave, not a one-way
    door -- and because the test suite needs each spec to start from an empty
    cache rather than inheriting the previous spec's fake model.

    The free itself is the one the per-call path used to run, moved rather than
    deleted: drop the references, collect, and empty the CUDA cache **only** when
    the model was actually on CUDA (or MPS, the same way). Guarded on the
    model's OWN parameter device rather than on ``torch.cuda.is_available()`` --
    a parameter can only report ``cuda`` if CUDA was already initialised to put
    it there, so this can never be the call that initialises it.
    """
    while _ENHANCER:
        _key, (model, processor) = _ENHANCER.popitem()
        device = _model_device(model)
        # The dict held the last strong references until `popitem`; these locals
        # hold them now, and they die at the end of this iteration.
        del model, processor
        gc.collect()
        if device == "cuda":
            import torch

            torch.cuda.empty_cache()
        elif device == "mps":
            import torch

            torch.mps.empty_cache()


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


def _load_enhancer(repo_id: str, *, revision: str | None = None, device: str | None = None):
    """``(model, processor)`` from the Hub at ``revision``, or one of ours naming the repo.

    Two failures are worth separating and both are named: ``transformers`` too
    old to know ``Gemma4ForConditionalGeneration`` at all, and the fetch/load
    itself. Neither ever falls through to the original prompt.
    """
    if revision is None:
        _warn_unpinned(repo_id, "loading")
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
        processor = AutoProcessor.from_pretrained(repo_id, revision=revision)
        # Named explicitly rather than through `AutoModel*`: the QAT repo's
        # config.json has no `architectures` key, so auto-dispatch cannot
        # resolve it. `dtype="auto"` honours the checkpoint's own dtype and the
        # `quantization_config` block the QAT build carries.
        model = Gemma4ForConditionalGeneration.from_pretrained(
            repo_id, revision=revision, dtype="auto"
        )
    except Exception as exc:
        raise PromptEnhancementError(repo_id=repo_id, reason=repr(exc)) from exc

    # `.eval()` and an explicit device, in that order. The device is STATED
    # rather than inherited: a global default device deciding where 2.5 GB of
    # Gemma lands is exactly the kind of action-at-a-distance that made the
    # original CPU pin necessary. It is now resolved rather than hardcoded --
    # see `enhancer_device`.
    return model.eval().to(device or enhancer_device()), processor


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
    there, so reading this can never be the thing that initialises it. The
    processor's outputs are moved here, so it must be the real device type
    (``cuda``, ``mps``, ``cpu`` ...): answering ``cpu`` for an MPS model sent the
    image tensors to the CPU and failed image-aware enhancement. A model with no
    parameters answers ``"cpu"``.
    """
    try:
        for parameter in model.parameters():
            return str(parameter.device.type)
    except Exception:  # noqa: BLE001 -- a fake model in a test has no parameters
        return "cpu"
    return "cpu"
