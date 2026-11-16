# The mode matrix on the live Space: 11 of 11 execute, and the defect that took two of them out

> **How to read this document (2026-11-16).** §1–§9 are the August 2026 matrix on the Space,
> made with builds whose output was colour static (the block below); they are corrected in
> place where later measurements overtook them. **§10 is the current state:** what has been
> verified since the 2026-11-16 fixes, locally on real weights, and the place for the
> post-deploy Space run, which has not happened yet.

> # ⛔ INVALIDATED (2026-08-25) — every generation result in this document was produced by a BROKEN TRANSFORMER
>
> **Read this before any number below.** Every Space sha up to and including
> `ff142d9` ran **blanket bitsandbytes int8**: a `PipelineQuantizationConfig` that
> named no `llm_int8_skip_modules`, so bnb replaced **every** `nn.Linear` in the
> transformer — the eight top-level `LTX2AdaLayerNormSingle` modulation heads and
> the in/out projections included. Rounding the shift/scale/gate that all 48
> blocks are conditioned on does not *degrade* an image; it stops the denoiser
> being steered by the timestep and the prompt at all, and the output becomes a
> correctly-shaped sample from the wrong distribution. In practice: **colour
> static and flat fields, at every resolution, in every mode, deterministically.**
>
> ✅ **Fixed at `eb1aa9f`** — `llm_int8_skip_modules` via `quant_mapping`, **316 of
> 1660** Linears spared, the names read off the real checkpoint's 4091-entry
> safetensors weight map at revision `871165de` rather than inferred from another
> DiT. **Deployed as Space sha `e3b5333`**, and proven there: the same seed, the
> same prompt and the same geometry that had produced colour static now produce a
> **photographic** frame — a red hot-air balloon over a lake at dawn with its
> reflection, a treeline, a skyline and sunset cloud banding. Before and after,
> side by side:
> `~/.soma/leads/ltx25-studio/frames/COMPARE_before_after_skipfix.png` 📄.
>
> ### ⛔ A second mechanism was proposed and is REFUTED — recorded so nobody re-chases it
>
> The `LLM.int8()`-computes-in-fp16 hypothesis — bitsandbytes' own bf16→fp16 cast
> warning in the Space log, read at the time as the most incriminating line
> available — **does not explain the garbage.** `llm_int8_threshold` sits at its
> **6.0 default**, and that threshold is precisely the mechanism that routes
> fp16-overflowing columns through the bf16 outlier path; inside fp16's range a
> bf16→fp16 cast is exact. ✅ The skip list alone fixed it with
> `llm_int8_threshold` untouched — one variable at a time, which is what makes
> the attribution clean. The warning is real; it was not the mechanism.
>
> ### ✅ MECHANICAL CLAIMS SURVIVE — all of them, unchanged
>
> A broken transformer is still a transformer, and every code path this document
> exercised did execute. Nothing here about *execution* is retracted:
>
> * **All eleven modes execute on the live Space** and return well-formed
>   artifacts with the stream layout the registry declares (§2, §2.1, §3).
> * **Both fork-boundary directions carry their cargo** — outbound (§4, `6bb211a`,
>   proven live in [`SPACE_SESSION_2026-08-25.md`](SPACE_SESSION_2026-08-25.md)
>   §2) and inbound (`0be0014`, that file's §6).
> * **Determinism and byte-reproducibility hold** (§5, and that file's §3):
>   sha256-equal artifacts across containers, sessions, code shas and both LoRA
>   routes.
> * **Tiled decode enables 1216×704×121** — the geometry, the 121 frames and the
>   6,258,833 bytes are real, and `cdb3211` is the difference between the failing
>   and passing runs (that file's §4).
> * **Stage 2 runs and doubles geometry** — output geometry verified 1408×960
>   (that file's §5). *Corrected 2026-11-16: the "ceiling between 25 and 49
>   input frames" was the 47.375 GiB `large` partition; 704×480×49 passes on
>   `xlarge` with 55.906 GiB reserved
>   ([`MIG_ALLOCATOR_SIGHTINGS.md`](MIG_ALLOCATOR_SIGHTINGS.md) §2). The ceiling
>   on `xlarge` is not measured.*
> * **LoRA applies, detaches, and both routes agree** (that file's §6 and §3).
> * **The disk and RAM arithmetic**, the tier and the ledger (§7). *The ×2
>   quota multiplier for `xlarge` is Hugging Face's documented rate, never
>   measured here (no quota endpoint answers).*
>
> ### ⛔ OUTPUT CLAIMS DO NOT SURVIVE — not one of them
>
> **Every "content-verified" artifact in this document was verified non-empty and
> non-frozen. None was ever verified as an image.** The `content ✅` column of §2
> and §2.1 means distinct frames, byte counts, luma range and dBFS — and colour
> static has all of those in abundance. Named specifically, because each was
> reported as a positive result:
>
> * **§6 and §6.1's near-static and quiet-audio observations** are readings of
>   *static*, not of video. Those mean |Δ| figures are the motion energy of noise.
> * **`SPACE_SESSION_2026-08-25.md` §7 (R7b) — the conditioning-fidelity
>   conclusion.** It measured one garbage output's motion statistics tracking a
>   second clip's. That the conditioning *path* moves numbers is a mechanical
>   result and it stands; "conditioning demonstrably drives the output" was
>   measured on garbage and does not. Its reclassification of §6 falls with it.
> * **The LoRA inertness check** (that file's §6): the adapter changed the
>   output's bytes. It changed one garbage render into a *different* garbage
>   render. The injection path ran; nothing about the adapter's effect on an
>   image is established.
> * **The stage-2 pass** (that file's §5): 1408×960 of upscaled garbage. The
>   latent upsampler, the extra denoise and the diffusion decoder all executed —
>   what they produced was never looked at.
>
> They demonstrate that the code paths ran. They do not demonstrate that the
> output was correct.
>
> ### ⛔ `b78803cc` is byte-reproducible NOISE
>
> The determinism claim is intact and its artifact is worthless. Both halves of
> that sentence are load-bearing: §5's resolution — three containers, three code
> shas (corrected 2026-11-16: `SPACE_SESSION_2026-08-25.md` §3 names the bank's
> `8fb292d6`-era build, `8e9f7a1` and the tiled-decode build `ff142d9`), one sha256 — is a
> **correct** result about the sampler and the seed, and
> the thing it reproduced so faithfully was never a picture.
>
> ### ⛔ The method failure, and the policy that replaces it
>
> **A statistical gate cannot distinguish a render from mush.** Every
> verification this project ran was statistical — distinct frames, geometry, byte
> counts, luma range, dBFS — and colour static passes all of them. The first
> person to look at a frame was the operator, on 2026-08-25.
>
> **Standing policy from here (operator ruling, 2026-08-25):** statistical gates
> are a **pre-filter, never a verdict — they demote artifacts, they never promote
> one.** No artifact is a PASS until a human or model eye has seen a frame **at
> native resolution**, and every report claiming an output is good carries the
> frame.
>
> ⚠️ **This binds the new structure detector too.** `e54477c` added SR/HF
> measurement precisely because the old checks could not see mush — and it is not
> a promotion mechanism either. It **FAILS** the operator's flat grey (SR 0.5298,
> HF 3.1455 — the conjunction fires) and **PASSES colour static comfortably** (SR
> 0.9183, HF 0.2254 — 1.31× clear of the 0.70 gate and 4.4× clear of the 1.0
> gate). ✅ Re-measured on the exported mid-frames at the same 512×960 geometry:
> static SR 0.9228 / HF 0.2097, grey SR 0.5561 / HF 3.1544, and the **post-fix
> photograph SR 0.9961–0.9968 / HF 0.075–0.133**. Two consequences worth stating
> plainly:
>
> 1. **SR does not separate static from picture** — 0.918 against 0.997 at the
>    same geometry. It separates *flat* from *everything else*, which is a
>    narrower thing than the axis it was named for.
> 2. ⛔ **The detector's "real" population was calibrated on garbage.** All eight
>    banked artifacts in `tests/test_frame_structure.py`'s table are Space
>    artifacts from a broken build, so "worst real SR 0.9183" is the worst
>    *colour static*, and the 0.70 gate sits at the geometric middle between
>    static and grey rather than between picture and static. The real
>    photograph's HF (0.075–0.133) falls **below** the band that table calls real
>    (0.147–0.386). Recalibration against post-fix artifacts is open work —
>    §8 follow-up, and the detector's own conclusion is unaffected: it still
>    cannot promote.
>
> ### ⛔ The cheap-config lesson
>
> §1's 256×256×9 — the cheapest legal config that still proves a code path, used
> for most of the campaign — produced output **too small and too blobby for
> anyone to judge by eye**. That is how three weeks of confident nonsense got
> manufactured. **A config too small to look at is a config that cannot be
> verified.** §1's own ⏸️ ("a code-path probe, not a quality sample") was right
> and was not enough: it warned that *quality* was untested when the truth was
> that *correctness* was untested, and at that size the probe could not have told
> the difference.
>
> ### ⛔ The post-fix matrix was never written up
>
> A fresh eleven-mode matrix at 512×960×121 ran against `e3b5333` on
> 2026-08-25. It was never written up, and its video and audio files are gone;
> only single mid-frames survive, outside this repo. It cannot be
> reconstructed (corrected 2026-11-16). **§10** now holds what has been
> verified since, and the place for the post-deploy Space run. Every section
> below stays exactly as measured.

---

> ## ✅ All eleven modes execute on the live Space
>
> *(Corrected 2026-11-16. This heading used to say "generate … content-verified";
> every output here was colour static, see the block above.)*
>
> The matrix itself found **9 of 11**. The two failures were `video_to_video` and
> `video_to_audio` — **both video-INPUT modes, and only those.** The cause was a
> **missing video decoder**, not a decode failure (§3); it was recovered from the
> **Space run log**, which `spaces` prints even when it strips it from the client
> response (§4), and fixed at `d2eb6f2`.
>
> ✅ **Both now pass on the live Space**, re-run at sha `287ba9c`:
>
> ```
> video_to_audio   19.95 s wall   <= 39.9 GPU-s   63,438 B   48 kHz 2 ch  -19.5 dBFS
> video_to_video   13.67 s wall   <= 27.3 GPU-s   33,812 B   9 frames / 9 distinct  mean |Δ| 17.277
> ```
>
> ⏸️ **What this does not establish: that either mode FOLLOWS its conditioning
> clip.** The numbers show real motion, not fidelity to the input. §9.
>
> §4 stands unchanged and still matters: the one string that would have made this
> diagnosable in a single step was destroyed in transit, and recovering it took a
> Space-log extraction to learn what our own code had already written down.

**Every load-bearing number is marked:** ✅ *we measured it* · 📄 *read off a
pinned source or the platform* · ⚠️ *inferred* · ⏸️ *not established*.

> ### ⚠️ The rule this document is organised around
>
> **A wall-clock figure is not a billing figure.** There is no readable ZeroGPU
> quota endpoint 📄, so every cost here is `wall × 2` and bounds the spend from
> above. It is written `<=` everywhere it appears. A bound that loses its label
> becomes a measurement the moment someone quotes it, and this build has already
> spent a day undoing exactly that with the inherited RAM ceiling.

---

## 1. What ran

| | |
|---|---|
| Space | `techfreakworm/LTX2.5-Studio`, **private**, ZeroGPU, stage RUNNING 📄 |
| Space SHA | `8fb292d6` 📄 |
| Client | `gradio_client` 2.6.0 → `/on_generate` 📄 |
| Runner | `scripts/space_matrix.py` |
| Account | PRO 📄, 2400 GPU-s/day |

Every run used the same configuration — the cheapest legal one that still proves
a code path:

```
width, height   = 256, 256      (256 = modes.MIN_RESOLUTION, and 256 % 32 == 0)
num_frames      = 9             (k*8 + 1 at k=1)
enable_stage_2  = False
seed            = 1234          (pinned; seed=-1 draws fresh per call and any
                                 comparison at the default passes vacuously)
prompt          = 'a red balloon rising over a calm lake at dawn'
```

✅ Pre-validated offline against `constraints.validate_dimensions` and
`constraints.validate_num_frames` before any quota was spent.

⏸️ **This config is a code-path probe, not a quality sample.** Nothing here says
what the modes look like at a resolution anyone would ship.

---

## 2. Results

**As the matrix ran: 9 PASS ✅ · 2 FAIL ⛔.** The two failures were fixed and
re-run afterwards on a later sha — §2.1. This table is left as the record of
what the matrix itself measured, because that is what makes §3's localisation
readable: nine passes and two failures, split cleanly along the input kind.

| mode | verdict | wall ✅ | artifact ✅ | content ✅ |
|---|---|---|---|---|
| `text_to_video` | ✅ | 21.54 s | 39,146 B | 9 distinct frames, mean \|Δ\| 18.894 |
| `text_to_audio` | ✅ | 15.35 s | 63,438 B | 48 kHz, 2 ch, −7.6 dBFS |
| `text_to_audio_video` | ✅ | 14.23 s | 44,823 B | video stats identical to `text_to_video`; audio −8.3 dBFS |
| `image_to_video` | ✅ | 14.80 s | 19,792 B | mean \|Δ\| 10.336 |
| `image_text_to_video` | ✅ | ~15 s | — | — |
| `image_to_audio_video` | ✅ | ~15 s | — | — |
| `image_text_to_audio_video` | ✅ | ~15 s | 16,312 B | audio −11.1 dBFS; mean \|Δ\| 0.591 (§6) |
| `audio_to_audio` | ✅ | ~15 s | — | 48 kHz, 2 ch, −8.8 dBFS |
| `audio_to_video` | ✅ | 21.59 s | 7,619 B | mean \|Δ\| 0.905 (§6) |
| `video_to_video` | ⛔ → ✅ §2.1 | see below | none | — |
| `video_to_audio` | ⛔ → ✅ §2.1 | see below | none | — |

⏸️ Cells marked `—` were not recorded in the handoff; they are absent from this
record rather than estimated.

Every passing run was checked in **both** directions before being called a pass:
the container's stream layout against what the registry says the mode exports
(`tests/verify_streams.py`, resolved from `modes.py` on every call), and then the
decoded **content** — because a green stream table and a silent file are
indistinguishable, which is what H9 cost.

### The failures

✅ Both failing modes returned `'DecodeError'`. Three failing attempts were
timed at **3.53 s**, **8.12 s** and **9.13 s** of wall clock — against the
**14.23–21.59 s** every successful generation took.

> **The failures are far too fast to have generated anything.** They are not
> timeouts and not degraded runs; they stop before the work starts.

⏸️ Which of the three attempts belongs to which mode is not recorded here. One
of the three is the refutation run described in §3.

## 2.1 ✅ Both re-run after the fix, and both pass on the live Space

Re-run at Space sha `287ba9c`, carrying `d2eb6f2`. Same 256×256×9, same seed,
same prompt as every row above:

| mode | wall ✅ | cost | artifact ✅ | content ✅ |
|---|---|---|---|---|
| `video_to_audio` | 19.95 s | `<=` 39.9 GPU-s | 63,438 B | 48 kHz, 2 ch, **−19.5 dBFS** (§6.1) |
| `video_to_video` | 13.67 s | `<=` 27.3 GPU-s | 33,812 B | 9 frames, 9 distinct, mean \|Δ\| **17.277** |

**Both land inside the 13.67–21.59 s band every successful generation occupies**,
against the 3.53–9.13 s the failures took. The timing alone separates a run that
worked from one that aborted before starting.

### ✅ The fixture was rebuilt first, and that matters

The runner chains fixtures forward: each mode's output can become the next
mode's input. By the end of the matrix `video.mp4` had drifted to
`audio_to_video`'s output — **7,619 B at mean \|Δ\| 0.905**, one of the two
near-static clips in §6. Conditioning on that would have proved very little: a
near-static result from a near-static input is unreadable in either direction.

✅ So the fixture was rebuilt before the re-run, and **both modes conditioned on
run 1's 39,146 B clip at mean \|Δ\| 18.894** — the most active input the matrix
produced. `video_to_video` returned **17.277** from that 18.894 input.

> ⏸️ **That is motion, not fidelity.** 17.277 out of 18.894 is exactly as
> consistent with faithfully following the conditioning clip as with ignoring it
> and generating fresh video from the prompt. **Nothing here distinguishes those
> two**, and no test in this document was designed to. §9.

---

## 3. ✅ The defect is localised to VIDEO INPUT

This is the most useful thing in this document, and it is a measurement rather
than a reading of the error text.

| input kind | modes | result |
|---|---|---|
| text only | 3 | ✅ all pass |
| **image** | 4 | ✅ all pass |
| **audio** | 2 | ✅ all pass |
| **video** | 2 | ⛔ **all fail** |

**Both audio-input modes pass and both video-input modes fail.** The split is
clean and it is along the input kind, not along the pipeline: `image_to_video`,
`audio_to_video` and `video_to_video` all run through the same `condition`
workflow, and only the video one fails.

### ⛔ A hypothesis was raised and REFUTED — record it, it narrows the search

> **Refuted:** *"the video fixture's AAC audio track is what the decode chokes
> on."*
>
> ✅ Tested directly with a **video-only fixture** — h264, no audio stream,
> verified with `av`. **It failed identically.**

The audio track is not the cause. Whatever fails, fails on the video input
itself.

### ✅ The mechanism, now CONFIRMED from the worker's own traceback

The prediction below was made before the matrix ran, and it was right. It is
kept as written because a prediction on record is worth more than the same
sentence written afterwards — but it is no longer the reason to believe this.
**The reason is the traceback**, recovered from the Space run log:

```
File "diffusers/pipelines/ltx2/pipeline_ltx2_condition.py", line 899,
  in preprocess_conditions
    raise TypeError(f"Unsupported `frames` type for condition {i}: {type(condition.frames)}")
TypeError: Unsupported `frames` type for condition 0: <class 'str'>
```

✅ **A `str` reached `LTX2VideoCondition.frames`.** `preprocess_conditions`
dispatches that field over four accepted types — `PIL.Image.Image`,
`list[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor` 📄 (read at the pinned SHA
`ac56fa2f`) — and raises for anything else. **It is not a decode failure at
all**: nothing ever attempted a decode. `'DecodeError'` was our own wrapper
around upstream's `TypeError`, which is why the name was so misleading.

✅ **Reproduced offline** before any fix was written, and ✅ **fixed at
`d2eb6f2`** — `conditioning.decode_video_frames` decodes the upload to
`(F, H, W, 3)` `uint8` at the `build_conditions` seam, so a video condition
stops being a path before it can reach the pipeline. Offline gate 2140 passed.

✅ **Confirmed on the metered side.** Both modes were re-run at Space sha
`287ba9c` and both produced real, content-verified artifacts — §2.1. The
diagnosis is therefore closed in both directions: the traceback said a `str`
reached a field that dispatches on type, and supplying a decoded array made the
mode work on the same platform that had refused it.

✅ **The `uint8` choice is the part that was load-bearing here rather than
locally.** Upstream re-compresses a single-frame condition at the model CRF and
rejects a non-`uint8` array while doing it 📄. A float payload would have decoded
cleanly against the offline gate and failed **only** on the Space — the exact
shape of defect this build keeps paying for. It did not, because the decoder
emits `uint8` at the source.

The pre-run read, on record from before the matrix:

* `gr.Image(type="pil")` hands the handler a **PIL Image** 📄 — the same object
  `scripts/host_matrix.py` has always passed, on the one path where video
  conditioning is known to work. (That hinode script, like the others, was
  removed on 2026-11-16; `git show 2f356b5:scripts/host_matrix.py` restores it.)
* `gr.Video` hands the handler a **filepath string** 📄, and ✅ **nothing in
  `ltx25_studio/` decodes video** — there is no video decoder anywhere in the
  package. `host_matrix.py` carries its own `load_video_frames` helper precisely
  because the library has none.
* That string then travels **unconverted**: `modes._conditions` →
  `conditioning.normalise_condition` (which passes `source` through verbatim ✅)
  → `pipelines.py:933`, `LTX2VideoCondition(frames=spec.source, …)`.

So the video modes hand diffusers a `str` where every run that has ever worked
handed a list of decoded frames.

> ✅ **Established — and note what it cost to establish.** When this section was
> first written the reading was marked *consistent, not proven*, because
> `'DecodeError'` is equally what a genuinely unreadable file produces and **the
> string that distinguished them had been destroyed in transit** (§4). That
> caution was correct: the two hypotheses were indistinguishable from the
> client, and only the Space-log traceback separated them. §4 is still the
> cheaper fix to make first — it is what turns the *next* failure of this kind
> into one line instead of a log extraction.

---

## 4. ⛔ Fail-loudly is degraded across the ZeroGPU fork boundary

**Fail-loudly is an acceptance criterion for this whole build, and it is
silently weakened inside every `@spaces.GPU` call.**

### What was observed

✅ The Space printed:

```
### ❌ Generation failed — unexpected `Error`

> 'DecodeError'
```

That template is `ui/render.py:399` 📄, and it is the branch taken when the
exception is **not** an `LTXStudioError`:

```python
return (
    f"### ❌ Generation failed — unexpected `{exc.__class__.__name__}`\n\n"
    f"> {detail}\n\n"
    ...
)
```

### What that proves

✅ **`DecodeError` IS ours** — `ltx25_studio/errors.py:426`, a subclass of
`GenerationError` and therefore of `LTXStudioError`. Had our exception object
reached `render.failure()`, it would have taken the *other* branch and printed
its own class name and its own message.

✅ It did not. The class that arrived was named **`Error`**, and the whole
payload was the single-quoted string `'DecodeError'` — our class's *name*, not
its message.

> **So the exception was re-boxed between the GPU worker and `render.py`, and
> the `reason=` we attach was destroyed in transit.**

### ✅ Where it is re-boxed — the exact line, and it REPRODUCES

This was first recorded as ⚠️ inferred. It is now read off the `spaces` source
and **reproduced byte for byte offline**, which is why it is marked ✅ here:

1. `wrappers.py:227` / `:379` 📄 — `raise error("ZeroGPU worker error", res.error_cls)`
2. `wrappers.py:41` 📄 — `from .gradio import error`
3. `spaces/zero/gradio.py:171` 📄 — that helper returns
   **`gr.Error(message, title=…, print_exception=False)`**

✅ Both halves of the observed output follow from step 3, and neither is a
coincidence of formatting:

* **`unexpected \`Error\``** — the class really is `gradio.exceptions.Error`.
  `"ZeroGPU worker error"` is passed as `title=`, **not** as the class, so
  `exc.__class__.__name__` is the bare word `Error`.
* **`'DecodeError'`, with the single quotes** — they are `repr()` of the message
  inside gradio's `Error.__str__`, not punctuation of ours. **That quoting is a
  signature of this path**, and it is what makes the `'GPU task aborted'`
  corroboration below *exact* rather than merely similar: the same helper
  produces it at `wrappers.py:217`/`:370` with a literal string.

✅ Fed through our own `render.failure()` against the gradio this app pins
(**6.24.0**), it emits the Space's message character for character — the loss is
**by design, not by accident**: `render.py` truthfully reports "not one of ours",
because by the time it sees the object, it genuinely is not.

⚠️ **Read at `spaces` 0.48.3; the Space ran 0.51.1 then, and reports 0.51.3 on
2026-11-16** 📄. The marshalling source
is therefore near-but-not-identical to the deployed version. The *rendering* was
reproduced against the same gradio we actually ship, and it matches observed
0.51.1 output for **two different failures**.

> ### ✅ And one sentence of ours is right here and wrong elsewhere
>
> Our copy ends *"The full traceback is in the Space logs."* **In this failure
> mode that is true** — `wrappers.py:226`/`:378` does `print(res.traceback)`
> before raising, and that is exactly how §3's root cause was recovered. **In a
> `SIGKILL` it is false**: nothing is printed, and `SPACE_GENERATION_FINDING.md`
> §3 already flags it as false there. **The same sentence, correct in one
> failure mode and misleading in the other** — which is what makes it worth
> rewriting rather than deleting. Recorded against follow-up #1 of that file.

That `reason=` is not decoration. `generate.py` raises `DecodeError` at **four**
sites, and they say different things:

| site | `reason=` |
|---|---|
| `generate.py:413` | `"the pipeline returned no output object at all"` |
| `generate.py:416` | `"the pipeline returned an output with no frames"` |
| `generate.py:418` | `"the pipeline returned zero frames"` |
| **`generate.py:597`** | **`f"the pipeline raised {exc!r}"`** |

⛔ **The fourth one is the loss that matters.** `generate.py:597` wraps whatever
the pipeline itself threw and quotes its `repr` — that string would have named
the actual diffusers failure behind §3. **We cannot even tell which of the four
sites fired**, because the string that distinguishes them is exactly the string
that was lost.

### ✅ A mechanism, measured offline — this costs zero GPU

Separately from *where* the re-boxing happens, there is a concrete **measured**
reason our exceptions could not survive a fork even if they were eligible to
travel — and it is the half that makes the fix a pair rather than a single line:

```
>>> pickle.loads(pickle.dumps(DecodeError(mode="video_to_video", reason="…")))
TypeError: DecodeError.__init__() takes 1 positional argument but 2 were given
```

✅ **`DecodeError` is unpicklable.** Its `__init__` takes **keyword-only**
arguments (`*, mode, reason`) while `super().__init__()` stores a single
formatted message, so `self.args` is a 1-tuple that pickle then feeds back
positionally — and the constructor refuses it.

✅ **22 of our 27 error classes have this shape**, measured by walking
`ltx25_studio/errors.py`:

```
unpicklable (required keyword-only __init__ args):  22
    AudioVideoDesyncError, ConfigError, DecodeError, DimensionError,
    EmptyOutputError, EncodeError, FrameCountError, LoRAConventionError,
    LoRADimensionError, LoRAFormatError, LoRAHeaderError, LoRAInContextError,
    LoRAModuleNotFoundError, LoRANotAttachedError, LoRAPairingError,
    LoRATooLargeError, LoRATruncatedError, LoRAVersionError,
    MissingModelFileError, ModelLoadError, UnknownComponentError,
    UnknownModeError
round-trips fine:  5
    ConstraintError, GenerationError, LTXStudioError, LoRAError, ModelFileError
```

The five that survive are the abstract bases nobody raises with arguments. **Every
error class written to carry actionable detail is one that cannot cross the
fork.**

### ✅ Corroboration — and the asymmetry that turned out to be the mechanism

The earlier `'GPU task aborted'` had the **identical shape** ✅ — class `Error`,
payload a single-quoted string — and that one is definitely a `spaces`-runtime
message 📄. So `Error('…')` is how the platform surfaces things.

One asymmetry was recorded here as *sharpening the inference*: for the killed
worker the payload was a **message**, and for ours it was a **class name** —
"what you would expect from a marshaller that could not reconstruct our object
and fell back to naming its type."

✅ **That reading was right, and the source says why**: both come from the same
helper, `wrappers.py:217`/`:370` passing a literal string and `:227`/`:379`
passing `res.error_cls`. The asymmetry is not two behaviours; it is **one call
with two different arguments**, and which argument you get depends solely on
whether the worker was killed or raised.

> ✅ **SETTLED, from the source — and the answer is "no".** The caution above
> was right to be raised and is now closed by reading `spaces` rather than by
> spending a live run on it.
>
> `ExceptionResult` 📄 (`spaces/zero/types.py:28-31`) carries exactly three
> fields — `traceback: str`, `error_cls: str`, `gradio_error: Err | None` — and
> `exception_result()` (`wrappers.py:453`) populates `gradio_error` **only** when
> `isinstance(exc, gr.Error)`. **The original exception object is never
> transmitted otherwise.** So for a `DecodeError` nothing is ever pickled,
> `__reduce__` is never consulted, and the parent still runs
> `raise error("ZeroGPU worker error", res.error_cls)` — the class name alone.
>
> **Picklability alone changes nothing.** It becomes load-bearing only once the
> object is eligible to travel at all — see the fix below, which is a pair.

### ✅ The fix is a PAIR, and neither half works alone

An earlier version of this section said the fix was one method. **It is two, and
that correction came from reading `spaces` rather than from trying it.** Both
halves are needed and each is inert — or worse — without the other:

| change | what actually happens |
|---|---|
| `__reduce__` only | The exception never enters `ExceptionResult` in the first place, so picklability is never consulted. **No improvement whatsoever.** |
| `gr.Error` subclass only | `gradio_error` is populated, so the object must now cross a `multiprocessing.Queue` 📄 (`wrappers.py:14, 65, 83` — `ForkProcess`, `res_queue = Queue()`). Ours cannot pickle. **Strictly worse — see below.** |
| **both** | `gradio_error` is populated **and** survives the queue; the parent runs `raise res.gradio_error` (`wrappers.py:224/376`) and the real object arrives with its real message. **This is the fix.** |

✅ **The middle row is measured, not guessed.** `spaces` anticipates exactly this
and handles it at `wrappers.py:272-275`:

```python
try:
    res_queue.put(res)
except PicklingError as e:
    res_queue.put(exception_result(e))
```

The worker replaces the whole result with an `ExceptionResult` describing **the
`PicklingError` itself** — which is not a `gr.Error`, so `gradio_error` is
`None` and `error_cls` becomes `"PicklingError"`. Run through our own renderer,
that is what the user would see:

```
### ❌ Generation failed — unexpected `Error`

> 'PicklingError'
```

**A `gr.Error` subclass on its own would destroy the one piece of information
that currently survives** — today `'DecodeError'` at least names the failure;
after that change it would name a serialisation artefact instead. Half this fix
is a regression.

⚠️ **And `gr.Error` is the obvious half.** It is what the source points at, and
what anyone reading `exception_result` would reach for first — a one-line base
class change with a clear rationale. **On its own it would have failed, for
exactly the reason measured above.** That is what makes the picklability finding
load-bearing rather than a curiosity: it is not a second improvement stacked on
the first, it is the thing that makes the obvious change insufficient. Neither
half looks necessary until you are holding both, which is the argument for
recording the pair here rather than leaving the next person to discover it one
failed deploy at a time.

✅ The `__reduce__` half is verified offline by round-tripping a real
`DecodeError`, restoring class, message **and every attribute** for all 22
subclasses at once:

```
class : DecodeError      str : decoding failed for mode 'video_to_video': the pipeline raised …
mode  : video_to_video   reason : the pipeline raised ValueError('bad frames')
```

⏸️ **Neither half is applied.** `errors.py` is outside this task's file
ownership, the change touches 22 classes and their specs, and it has not been
authorised. This is a follow-up with a complete design attached, which is a
better thing to hand over than a half-diagnosed one.

> ✅ **LANDED (2026-08-25) at `6bb211a` — both halves, exactly as designed
> here.** `LTXStudioError` now inherits `gradio.exceptions.Error` **and**
> defines `__reduce__` (rebuild via `__new__` + `args` + `__dict__`, skipping
> `__init__`); all 22 keyword-only subclasses round-trip; `__str__` keeps
> BaseException semantics so the repr-quoting stays gradio's, not ours; and
> `print_exception` stays true so the parent keeps printing the worker traceback
> — the channel §3's root cause was recovered through. Offline gate green at the
> landing tree (2245 passed / 13 skipped). ⏸️ **Not yet exercised on the
> platform**: the live Space sha `287ba9c` predates the fix, so the pair is
> offline-verified against the `spaces` marshalling contract, not yet observed
> in a live failure.
>
> *Update, later on 2026-08-25 — exercised, deliberately and then for real: a
> provoked `ConstraintError` and a provoked `FrameCountError` both crossed the
> boundary intact on the deployed Space, and the day's two genuine failures
> arrived client-side with class and reason readable.
> `SPACE_SESSION_2026-08-25.md` §2; follow-up #2 below is closed.*

⚠️ **Version caveat, stated exactly.** Everything above is read off `spaces`
**0.48.3**, the locally readable copy; the Space ran **0.51.1** then and reports
**0.51.3** on 2026-11-16 📄. This is
read-off-source, **not** measured on the Space. What *was* reproduced against
the same gradio our app actually pins (6.24.0) is the client-side rendering, and
it matches the observed 0.51.1 output for both failures.

---

## 5. ✅ Reproducibility on the video path — now evidenced

Two independent signals, from this matrix:

**1. A byte-identical artifact across containers and sessions.** ✅
`text_to_video` returned **39,146 bytes** — byte-identical to the result banked
the previous day, on a **different container**, in a **different session**, at
the same seed.

> ⛔ **CORRECTED (2026-08-25): "byte-identical" was never verified, and cannot
> now be verified against that file.** No hash of the previous day's artifact was
> ever recorded. What was actually compared: **file size** (39,146 B on both
> days) and **every decoded statistic to three decimals** (pixel std 72.657,
> mean |Δ| 18.894, per-pair range 8.35 – 57.17). That is *size- and
> statistics-identical* — strong evidence, and **not the same claim as identical
> bytes**. The day-1 file is gone with its host, so byte identity against it is
> now ⏸️ permanently unverifiable.
>
> **The caveat below was right, and the error ran in the other direction:** this
> headline (and reports quoting it) upgraded past the doc's own "Do NOT upgrade
> this" instruction. The measurement that closes it is scheduled: a same-seed
> re-run **sha256-compared against the surviving 2026-08-22 artifact**
> (`sha256 b78803cc998645dde06007c70cfa13d30af0ed37dd87e31dcc1d068da4aa212d`),
> planned 2026-08-25, which will settle byte-level reproducibility across the
> sampled containers. Until that runs, the supportable claim is: **identical in
> size and in all decoded statistics; byte identity UNVERIFIED.**

> ✅ **RESOLVED (2026-08-25, later the same day) — the planned re-run ran, and
> byte identity is now VERIFIED in the happy direction.** A fresh seed-1234
> 256×256×9 run on a **third container and a different code sha** hashed
> **sha256-equal to the surviving 2026-08-22 artifact** (`b78803cc…212d`,
> 39,146 B) — and a second fresh run later that day, through the redesigned UI
> on the tiled-decode build, on a warm worker that had just carried a LoRA
> adapter, hashed equal **again**. The condition pipeline matched too
> (`video_to_video` reproduced the banked `ef8571e6…` byte-for-byte), and the
> LoRA path is deterministic across both routes. Full record:
> [`SPACE_SESSION_2026-08-25.md`](SPACE_SESSION_2026-08-25.md) §3. The
> correction block above stays as history: the claim was unverified when it was
> made, and the distinction between "matched on size and statistics" and
> "matched on hash" is exactly what this resolution needed to exist.

**2. Enabling audio did not perturb the video stream.** ✅
`text_to_audio_video` produced video statistics identical to `text_to_video` **to
three decimals** at the same seed:

```
pixel std        72.657
mean |Δ|         18.894
per-pair range   8.35 .. 57.17
```

> ⛔ **Do NOT upgrade this to "byte-identical seeding is proven for video."**
> **No two-run frame-hash comparison was made.** ⏸️ Identical file size and
> identical summary statistics are strong evidence and are not the same claim as
> identical bytes frame-for-frame. The measurement that would close it is cheap:
> hash the decoded frames of two same-seed runs.

The second signal carries a bonus worth recording separately: it suggests the
audio branch does not disturb the video branch at a fixed seed. ⚠️ Inferred from
n=1 at one configuration.

---

## 6. ⚠️ Two modes produce near-static video — an observation, not a defect

| mode | mean \|Δ\| ✅ |
|---|---|
| `text_to_video` | 18.894 |
| `image_to_video` | 10.336 |
| **`audio_to_video`** | **0.905** |
| **`image_text_to_audio_video`** | **0.591** |

Both pass the distinctness and motion checks — they are moving, and they are not
frozen or blank — but they are moving roughly **twenty to thirty times less**
than `text_to_video` at the same geometry and seed.

> ⏸️ **Filed as an observation.** At 9 frames (0.375 s) and 256×256 this may be
> entirely expected, and nothing here establishes otherwise. It is recorded
> because the number is far from its siblings and that is the kind of gap that
> turns out to matter — exactly how H9's −55 dBFS was first noticed.

## 6.1 ⚠️ `video_to_audio` is the quietest audio mode — also an observation

| mode | rms ✅ |
|---|---|
| `text_to_video` (audio track) | −7.6 dBFS |
| `text_to_audio_video` | −8.3 dBFS |
| `audio_to_audio` | −8.8 dBFS |
| `image_text_to_audio_video` | −11.1 dBFS |
| **`video_to_audio`** | **−19.5 dBFS** |

✅ Not silent, and it passes the level check — the floor for "silent" is −70 dBFS
and the "very quiet" flag is −40 dBFS, so it is comfortably clear of both. But it
sits **8 to 12 dB below every other audio mode** at the same geometry and seed.

> ⏸️ **Observation, not a defect, and deliberately not shrugged off.** It is the
> only audio mode conditioned on **video**, so a systematic difference is not
> surprising — but "not surprising" is not evidence, and nothing here establishes
> whether the gap is inherent to the conditioning path or a symptom.
>
> ⚠️ It is recorded at this length for one reason: **H9 was a −55 dBFS artifact
> that passed every structural check**, and the lesson taken from it was that a
> level far from its siblings is worth writing down before anyone has a theory
> about it. This is that, at a much milder magnitude, on the mode H9 was about.

---

## 7. Cost

```
matrix (9 modes + 3 failed attempts)  <= ~346   GPU-seconds UPPER BOUND
re-run after the fix (2 modes)        <=   67.3 GPU-seconds UPPER BOUND
                                      ------------
today, all of it                      <= ~413   GPU-seconds UPPER BOUND
budget   2400 GPU-seconds/day (PRO) 📄   ->  <= ~17.2% consumed
ceiling  1920 GPU-seconds (80%)          ->  never approached
```

> **[On "today"] (annotated 2026-08-25)** — written on the day, and the day has
> a date: the UTC quota day **2026-08-21** (the matrix ran late 2026-08-21 UTC;
> the §2.1 re-run landed the same UTC day, early 2026-08-22 IST — ZeroGPU quota
> days are per-UTC-date). That day closed at **≤ ~413 GPU-s** and the ceiling
> was never approached. This block is history, not a running total: as of
> 2026-08-25 the quota day is fresh and none of it is current consumption.

> ### ⚠️ The ledger says `67.252`, and that is not today's total
>
> `scripts/space_matrix.py`'s quota ledger — the cross-invocation one added at
> `62d612f` — reports **67.252 GPU-s** for this UTC day. **That is the re-run
> alone.** The ledger was installed *after* the matrix had already run, and it
> cannot see spend that predates its own first write.
>
> ⛔ **So the ledger's number is correct and incomplete at the same time**, and
> those are easy to confuse: it is an accurate record of everything it observed,
> and it observed only the last two runs of the day. Today's real bound is
> **~413**, of which the ledger accounts for 67.252.
>
> **This is worth stating in its own right, because it is a fresh instance of the
> class §8 is about.** A guard whose history begins mid-day reports a number that
> is true, quotable, and wrong by everything that happened before it existed. It
> is not inert — it works — but a figure read off it tomorrow without this
> caveat understates the day by a factor of six. ⏸️ Nothing in the ledger's own
> output says where its history starts.

⏸️ **Exact billed seconds remain unmeasurable** — there is no readable ZeroGPU
quota endpoint, and all three candidate URLs 404 📄. The figure above is
`wall × 2` (the `xlarge` size multiplier 📄) and is a **bound, not a
measurement**. Do not quote it as billed seconds. *Corrected 2026-11-16: these
runs were on the 47.375 GiB `large` partition, not `xlarge`
([`MIG_ALLOCATOR_SIGHTINGS.md`](MIG_ALLOCATOR_SIGHTINGS.md) §1); × 2 stays a
valid upper bound.*

⚠️ The bound is loose in the safe direction: wall clock includes queueing,
upload, download and client overhead, none of which is GPU time.

---

## 8. Follow-ups this matrix created

| | item | status |
|---|---|---|
| 1 | ✅ **CLOSED.** A filepath `str` reached `LTX2VideoCondition(frames=…)`; `preprocess_conditions` dispatches that field over four types and raises for anything else (§3). Confirmed from the worker traceback, reproduced offline, fixed at **`d2eb6f2`**, and **verified on the live Space at sha `287ba9c`** — both modes returned real content-verified artifacts (§2.1). 11 of 11 modes now generate. **→ ⛔ THE OUTPUT HALF IS RETRACTED (2026-08-25): "real content-verified artifacts" meant non-empty and non-frozen. Both modes executed and returned well-formed files; neither file was ever looked at, and both were colour static. The dispatch fix and the execution claim stand — see the invalidation block at the top of this file.** | ✅ closed (mechanically) · ⛔ output claim retracted |
| 2 | ⛔ **22 of 27 error classes cannot survive the ZeroGPU fork** (§4), measured. The fix is a **PAIR** — `__reduce__` on `LTXStudioError` **and** deriving it from `gr.Error` — and neither half works alone: `__reduce__` alone is never consulted, and `gr.Error` alone fails to pickle onto `res_queue` and surfaces as `'PicklingError'`, **destroying the class name that survives today**. Design complete, unapplied, `errors.py` unowned and unauthorised. **→ LANDED at `6bb211a` (2026-08-25), both halves, all 22 classes round-tripping; see the annotation in §4. ⏸️ Not yet exercised on the platform — the live Space sha `287ba9c` predates it.** **→ PROVEN LIVE later the same day, both classes: `ConstraintError` crossed with message + path + wrapped `av` errno intact, `FrameCountError` (keyword-only) with its full remedy sentence, and the day's two real failures (R3 decode, R6 dispatch) both arrived client-side legible — `SPACE_SESSION_2026-08-25.md` §2.** | ✅ closed, proven live (2026-08-25) |
| 3 | ✅ **Settled from source: picklability alone would NOT restore the message.** `ExceptionResult` carries only `traceback`, `error_cls` and `gradio_error`, and the exception object travels only when it is a `gr.Error` 📄 (`types.py:28-31`, `wrappers.py:453`). Read at `spaces` 0.48.3 against a Space on 0.51.1; read-off-source, not measured on-Space. **No live run is needed to know this** — one would still confirm it end to end. **→ Confirmed end to end on the live Space (2026-08-25): with the pair applied, real exception objects crossed the boundary with class and message intact, on the deployed 0.51.1 — the source read holds in production (`SPACE_SESSION_2026-08-25.md` §2).** | ✅ resolved, live-confirmed |
| 4 | ⏸️ **Reproducibility is evidenced, not proven** (§5). No two-run frame-hash comparison exists. Cheap to close. **→ Sharpened (2026-08-25): §5's "byte-identical" headline itself overstated the evidence — the cross-day match was size + decoded statistics, no day-1 hash was ever recorded, and the day-1 file is gone. The close is now a same-seed re-run sha256-compared against the surviving 2026-08-22 artifact (`b78803cc…`), planned 2026-08-25 — it settles byte-level reproducibility across the sampled containers.** **→ CLOSED later the same day: sha256-verified byte identity across three containers and three code shas for `text_to_video`, plus condition-pipeline (`ef8571e6…`) and LoRA-path (`20a3a84b…`) determinism — `SPACE_SESSION_2026-08-25.md` §3.** | ✅ closed (2026-08-25) |
| 5 | ⚠️ **`audio_to_video` and `image_text_to_audio_video` are near-static** (§6). May be expected at this geometry; nothing establishes it either way. **→ EXPLAINED (2026-08-25) by the conditioning discriminator: output motion statistics mirror the conditioning input's (a near-static clip at 0.905 produced 0.864 where the active clip produced 17.277), so a near-static output from gentle conditioning is the conditioning *working*, not a defect — `SPACE_SESSION_2026-08-25.md` §7. Motion mirroring is not semantic fidelity; that limit is stated there.** **→ ⛔ THE EXPLANATION IS WITHDRAWN (2026-08-25, later): both the 0.905 input and the 0.864 output were colour static from the broken int8 build. Those mean |Δ| figures are the motion energy of noise, so nothing was explained. Back to what it was — an unexplained observation, now to be re-measured against `e3b5333`.** | ⏸️ REOPENED (2026-08-25) |
| 6 | ⛔ **`audio.validate_audio_latents_against_video` has ZERO callers.** Found by inspection, not by this matrix. It is exported, documented as *"the check that stops a desynced file being emitted rather than refused"*, and nothing calls it — so supplied `audio_latents` set the audio length while `num_frames` sets the video length, with nothing comparing them. Both audio-input modes passed here **only because their fixture came from a same-geometry run**, which is a property of the runner's chaining and not a defect anyone would notice. **→ WIRED at `59cfa6e` (2026-08-25): `generate._prepared_audio`'s supplied-latents branch now calls it, a mismatch raises with both lengths named, `auto_duration` skips, and could-not-check shapes surface as warnings.** | ✅ wired `59cfa6e` |
| 7 | ⏸️ **Only one configuration has been exercised** — 256×256×9, stage 2 off, no LoRA, no `auto_duration`. Nothing here says what any mode does at a shippable resolution, and stage 2 has never run on the Space at all. **→ SUPERSEDED (2026-08-25): 704×480×49 and 1216×704×121 (via the tiled decode, `cdb3211`) both pass; stage 2 ran for the first time — PASS at 704×480×25 input (1408×960 out), FAIL at ×49 in the diffusion decoder, so its ceiling is between those; LoRA exercised in both directions with a passing inertness check. Still never exercised on the platform: `auto_duration`. `SPACE_SESSION_2026-08-25.md`.** **→ ⛔ Corrected 2026-11-16: the ×49 failure was the `large` partition, and ×49 passes on `xlarge` (§1 of [`MIG_ALLOCATOR_SIGHTINGS.md`](MIG_ALLOCATOR_SIGHTINGS.md)); the inertness check compared two broken renders and is withdrawn (the block at the top). `auto_duration` has since run on real weights locally (§10.2); on the platform it never has.** | ✅ superseded · ⏸️ auto_duration open on the Space |
| 8 | ⏸️ **No mode has been shown to FOLLOW its conditioning input.** Every check in this document is a property of the output alone — motion, level, distinctness. `video_to_video` returning mean \|Δ\| 17.277 from an 18.894 input is equally consistent with faithful conditioning and with ignoring the clip entirely (§2.1). Settling it needs a comparison the runner does not currently make: same seed, two different conditioning clips, and a difference in the output. **→ THE NAMED COMPARISON WAS MADE (2026-08-25): same seed, same prompt, two clips — outputs 17.277 vs 0.864, mirroring their inputs (18.894 vs 0.905), against a control pair whose noise floor is exactly zero (byte-identical). Conditioning drives the output; the prompt-only null hypothesis is dead. Resolved for motion-following; *semantic* fidelity — content, not motion energy — remains ⏸️ untested. `SPACE_SESSION_2026-08-25.md` §7.** **→ ⛔ THE RESOLUTION IS WITHDRAWN (2026-08-25, later). R7a and R7b both ran on the broken int8 build; the comparison was between two garbage renders. It shows the conditioning *path* carries a number across the fork — a mechanical result, and it stands — and it does not show that any mode follows its conditioning input. This item returns to ⏸️ open, exactly as originally written, and the clip-swap is the right experiment to re-run against `e3b5333`.** **→ Locally, on real weights (2026-11-16, §10.2): the image modes start on the supplied image (frame-0 SSIM 0.74–0.87 against it), and video to sound follows what is visible in the clip. On the Space: not re-run.** | ⏸️ open on the Space · ✅ locally (§10.2) |
| 9 | ⚠️ **`video_to_audio` is 8–12 dB quieter than every other audio mode** (§6.1) — −19.5 dBFS against −7.6 to −11.1. Passes the level checks, is nowhere near the −40 dBFS flag, and is the only audio mode conditioned on video. Recorded because H9 was a level anomaly that passed every structural check. **→ Annotation (2026-08-25): now CONSISTENT-WITH the conditioning finding — outputs mirror their conditioning input (`SPACE_SESSION_2026-08-25.md` §7), and a gentle scene plausibly yields quiet audio. Consistent-with is not established-by: nothing has probed audio conditioning directly, so this stays what it is — an observation.** **→ ⛔ The consistent-with reasoning is withdrawn (2026-08-25, later): it derived entirely from §7, which is itself withdrawn above. −19.5 dBFS was the level of audio generated by a broken transformer. Unexplained again, and cheap to re-measure.** **→ Locally, on real weights (2026-11-16, §10.2): video to sound follows what is visible; an eventless clip gives a quiet, eventless track. Not re-measured on the Space.** | ⏸️ open on the Space · ✅ locally (§10.2) |
| 10 | ⚠️ **The quota ledger's history begins mid-day** (§7). It reports 67.252 GPU-s for a day whose real bound is ~413, because it was installed after the matrix ran. The figure is accurate for what it observed and misleading if quoted as a daily total; nothing in its output says where its history starts. | ⏸️ open |
| 11 | ⛔ **RECALIBRATE THE STRUCTURE DETECTOR AGAINST POST-FIX ARTIFACTS** (2026-08-25). `e54477c`'s SR/HF thresholds were fitted to a "real" population of eight banked artifacts that were all colour static, and one flat-grey failure. It therefore separates *flat* from *everything else* and does not separate static from picture: SR 0.9183 (static) against 0.9961–0.9968 (the post-fix photograph), with the 0.70 gate below both. The photograph's HF (0.075–0.133) sits **outside** the band the table calls real (0.147–0.386). The detector's conclusion is unaffected — it never promotes — but its numbers now describe the wrong two populations. **→ Recalibrated on 2026-11-16 (`752b18e`) on a labelled set of real frames from the local runs, with three named conditions (flat, veiled, static), still demote-only. On five correct outputs it had not seen it gave no note, and it flagged the pre-fix veiled audio-to-video output as "veiled".** | ✅ recalibrated (local evidence) |
| 12 | ⏸️ **The post-fix eleven-mode matrix** — every mode re-run against `e3b5333` at a shippable config, with a frame from each looked at. §10 is where it goes. Until it lands, the only output this project has ever verified as an image is the single post-fix comparison frame. **→ ⛔ Corrected 2026-11-16: it ran on 2026-08-25 but was never written up and its files are gone. §10.2 records the local real-weights matrix of 2026-11-16; the Space matrix is to be run after the next deploy (§10.1).** | ⏸️ on the Space · ✅ locally, 10 of 11 (§10.2) |
| 13 | ⏸️ **The NVML allocator assert has now been seen THREE times, and the record held them apart** (2026-08-27). R3's VAE decode and R4b's diffusion decoder (both 2026-08-25, `SPACE_SESSION_2026-08-25.md` §4/§5) and the first enhanced generation in a fresh worker (2026-08-27) all died on `NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1165`. Collected side by side in [`MIG_ALLOCATOR_SIGHTINGS.md`](MIG_ALLOCATOR_SIGHTINGS.md) with the shared shape stated as a **hypothesis, explicitly not established** — this partition may mishandle a large allocation that follows another large allocation. **Deliberately not being chased**: sighting 3's premise disappears if the enhancer is made resident, and every question is blocked on a headroom number nobody has. `0a4ddbb` now reports peak VRAM (allocated / reserved / total, split by phase) on every generation, so one flagship run settles whether there was room. **→ ✅ IT DID, AND THE ANSWER WAS THE PARTITION (2026-08-27). The first run to print `total_memory` read 47.375 GiB — ZeroGPU's `large` — where the whole build assumed `xlarge`'s 94.971 GiB. `LTX25_ZEROGPU_SIZE=auto` sends no `gpuSize` at all, and the packer's own promotion never leaves `get_duration_seconds`. R4b is RETIRED by measurement: re-run on `xlarge` it PASSES at 53.08 s with peak 55.906 GiB reserved — 8.5 GiB more than the entire old partition, so it was arithmetically impossible and always would have been. ⏸️ The other two sightings are NOT retired: R3's untiled path is consistent-but-permanently-untestable (tiling is unconditional since `cdb3211`, and a tiling-off toggle was refused), and the enhancer sighting is pending the residency change. The single-cause account is SUPPORTED, NOT PROVEN.** **→ (2026-11-16) The residency change is `52a0f56`, deployed on 2026-08-26; no enhanced run has been recorded on the Space since, so sighting 3 is still pending.** | ✅ root cause found · ⏸️ 2 of 3 sightings unretired |

### ⛔ Item 6 was the THIRD instance of one defect class. There are now FOUR, and the fourth cost half a GPU

An advertised guard wired to nothing has now been found four times in this
build, by four unrelated routes:

| | the guard | how it was inert | how it was caught |
|---|---|---|---|
| 1 | `LTX25_TEXT_ENCODER_PRECISION` / `LTX25_TRANSFORMER_PRECISION` | not a `RuntimeConfig` field at all — never resolved, validated or logged | mutation test, after the value had already been written into a findings doc as a live lever 📄 |
| 2 | `--quota-ceiling` | per-invocation; five invocations each announced **FITS** knowing nothing of the other four | reading the runner's own output across runs |
| 3 | `audio.validate_audio_latents_against_video` | exported, documented, **zero callers** | inspection, not by this matrix |
| 4 ⛔ | `LTX25_ZEROGPU_SIZE=auto`, and the "ZeroGPU auto-promotes us" belief resting on it | the promotion **fires** and sets `_default_gpu_size`, which is read by one function (`get_duration_seconds`) and feeds one branch (the duration multiplier). It never reaches the `gpuSize` the scheduler allocates on. ✅ The Space ran on `large`/47.375 GiB for its entire life while four documents asserted `xlarge`/94.971 GiB | a peak-VRAM probe added for an unrelated reason (`0a4ddbb`) printing `total_memory` for the first time |

> **Status of the three, read back 2026-08-25.** Instance 3 is **closed** —
> wired at `59cfa6e` (follow-up #6). Instance 2 was closed at `62d612f` (the
> cross-invocation ledger, §7). Instance 1's cleanup is **half-done on the
> platform**: the live Space variables still carry `LTX25_TRANSFORMER_PRECISION=bf16`
> — inert as ever, and now advertising bf16 on a Space whose transformer loads
> in int8. Its deletion is being requested with the next deploy. See
> `SPACE_GENERATION_FINDING.md` §2's annotation; the row above stands as the
> record of how each was found. *(2026-11-16: done. The Space's variables,
> read back that day, no longer include `LTX25_TRANSFORMER_PRECISION`.)*

**The class is not "a bug that was missed". It is a mechanism that reports
safety it never checked** — which is strictly worse than its absence, because an
inert *lever* merely does nothing while an inert *report* actively tells you that
you are safe. `PLAN.md` states the rule for levers already: *"an advertised lever
attached to nothing is worse than no lever at all."* Four instances say the rule
needs a **test-shaped** counterpart, not another restatement.

> ### ⛔ Instance 4 is a different animal and it deserves its own sentence
>
> The first three were **our** guards, in **our** code, which a mutation test or
> a caller count can reach. The fourth was a claim about the **vendor's** code —
> that `startup()`'s promotion reaches the scheduler — believed by four documents
> in this repo, each citing the others, none of which had followed the value past
> the function that sets it. No test we could write against our own package would
> have caught it. What caught it was a number printed from inside the worker.
>
> ⏸️ **The generalisation this one implies, and nobody has run:** every remaining
> claim in this repo of the form *"the platform does X for us automatically"*,
> traced to the call that would actually **observe** X rather than the call that
> sets it. Reading a source and reading a source *to the end of the value's life*
> are different acts, and only one of them is evidence.

⏸️ **The cheap generalisation nobody has run:** every guard in this package that
is exported and documented, checked for callers. #3 was found by eye. Nothing
says it is the last one, and three of these four were invisible to a green gate.

---

## 9. What this document does not say

* It **does** say all eleven modes generate on the Space — and the evidence is
  exactly this: each produced a non-empty artifact whose **decoded content** was
  checked, not merely its container. For the two re-run modes that is 33,812 B at
  9 distinct frames and 63,438 B at −19.5 dBFS, at sha `287ba9c` (§2.1). That is
  the whole claim.
  — *⛔ Correction (2026-08-25): "decoded content was checked" is the sentence
  that did the damage. What was checked was distinctness, byte count, level and
  luma range — properties **noise has in abundance**. Every one of those
  artifacts was colour static. The surviving claim is narrower and still real:
  all eleven modes **execute** and return well-formed files. See the
  invalidation block at the top.*
* It does **not** say any mode **follows** its conditioning input. Every check
  here reads the output alone: motion, level, distinctness. A mode that ignored
  its conditioning clip entirely and generated from the prompt would pass every
  test in this document. §2.1, follow-up #8.
  — *✅ Still exactly true (2026-08-25). Follow-up #8 was marked resolved on
  2026-08-25 and is **REOPENED**: the clip-swap that resolved it ran on the
  broken int8 build and compared two colour statics. This bullet, as originally
  written, is the current state.*
* It does **not** say the modes are *correct*, only that they run and return
  something real. "Generates" is a much weaker claim than "generates what was
  asked for", and only the first is established.
* It does **not** say the fork-boundary fix has been applied or tried. §4's
  design is complete and read off `spaces` **0.48.3** while the Space ran
  **0.51.1** (0.51.3 on 2026-11-16); nothing in it has been exercised against
  the platform.
  — *Annotation (2026-08-25): the first half is stale — the pair **is** applied,
  at `6bb211a`, offline gate green. The second half stands unchanged: the live
  Space sha `287ba9c` predates the fix, so nothing has been exercised against
  the platform.* *Later the same day the second half fell too: exercised live,
  both classes — `SPACE_SESSION_2026-08-25.md` §2.*
* It does **not** say any figure here is billed GPU-seconds. All of them are
  wall-clock-derived bounds.
* It does **not** say the passing modes are *good*. They generate, and their
  artifacts are non-empty, non-silent and non-frozen. Quality at a real
  resolution is untested.
  — *✅ This bullet was right, and it was righter than it knew (annotated
  2026-08-25). "Non-empty, non-silent and non-frozen" is an exact and complete
  description of what was measured, and the gap it flags turned out not to be
  between "generates" and "generates well" but between "generates" and
  "generates **anything at all**". The caveat was written, published, and then
  quoted past by every report that used this document. **A caveat is not a
  control.***

---

## 10. Verification after the 2026-11-16 fixes

Nothing in §1–§9 describes the current code. This section does: 10.1 is the Space, 10.2 is
what has been verified on real weights so far, on a Mac.

### 10.1 On the Space — to be written from the post-deploy Space run

> ⏸️ **Not run yet.** The Space still runs the 2026-08-26 build (sha `f809472`, the tree of
> `445abcb`) and has generated nothing since; none of the 2026-11-16 fixes is deployed. This
> section is to be written from the post-deploy Space run. The August post-fix matrix
> (`e3b5333`, 512×960×121) cannot stand in for it: it was never written up and its files are
> gone.
>
> Only the Space can show the parts that need CUDA, and none of them has been verified since
> the fixes: Upscale ×2 (stage 2), the diffusion decoder (the default decode where it can run,
> `LTX25_DECODER=auto`), bitsandbytes int8, and ZeroGPU's progress, quota and timeout
> behaviour.
>
> What each row must carry, per the standing operator ruling (statistical checks are a
> pre-filter, never a verdict):
>
> 1. The config, stated as §1 states its own, and large enough to look at. 256×256×9 does not
>    qualify.
> 2. Per mode: wall, cost bound, artifact bytes, stream layout.
> 3. A frame per mode at native resolution and the record of a person or model having looked
>    at it; for sound, the level and a listening record. No row passes without them.
> 4. The picture check as a demotion signal only.
> 5. What it cost, against the daily GPU budget.

### 10.2 Locally, on real weights (Apple silicon), 2026-11-16

Recorded by the lead in the run log of the 2026-11-16 fix; each run's files (log, memory
trace, output, frames, spectrogram, measurements) are kept under its run id in that run's
evidence directory, outside this repo.

**How the runs were made.**

- An Apple-silicon Mac with 128 GB of unified memory; PyTorch 2.15.0 in a separate
  environment (2.11, the Space's pin, computes attention wrongly on MPS).
- The distilled transformer as the Q8_0 GGUF from `unsloth/LTX-2.5-GGUF` (revision
  `ac3c540c`); every other component from `Lightricks/LTX-2.5-Diffusers` at the app's pinned
  revision `871165de`. bf16 on MPS, staged residency: the text encoder and the transformer are
  never resident together.
- Through `scripts/local_generate.py`, which drives the app's own `generate()`. Single stage,
  conv-VAE decode (the diffusion decoder needs CUDA), 704×480×121 (5 s at 24 fps) unless a row
  says otherwise. **The app's own UI was not part of these runs.**
- The lead opened every output: six native-resolution frames per video; per soundtrack the
  loudness, a spectrogram and a transcript; for conditioned modes, frame 0 against the
  supplied image (SSIM, PSNR). Inputs were files prepared for the test (photos, synthetic
  clips, recorded speech, the operator's own recording); the video-to-sound clips were two
  outputs of earlier runs.

**Per mode.** Ten of eleven verified; mode 04 is pending.

| # | mode | run id(s) and code | what was seen and heard | measured | verdict |
|---|---|---|---|---|---|
| 01 | Text to video | `G3_704x480x121` at `75d0c53`; repeated as `G3r_rope64` at `5e4829a` | a red tram climbs a cobbled street and passes the camera, coherent over the 5 s | luma p1–p99 1–248, contrast 64.5; the repeat: same scene, statistics within 1 % | works |
| 02 | Image to video | `M02_i2v` at `75d0c53` | frame 0 is the supplied tram photo; the tram then rolls past the camera | frame 0 vs the photo: SSIM 0.75, PSNR 21.8 dB | works |
| 03 | Image + text to video | `M03_it2v` at `75d0c53` | starts on the supplied photo, slow push-in as prompted | frame 0 vs the photo: SSIM 0.74 | works |
| 04 | Video to video | `M04_v2v`, `M04b_v2v_s05` … `M04e_v2v_s00` at `75d0c53`/`5e4829a`; prototype `V2V_LEVEL_*` at `5e4829a` | with the earlier mechanism the source clip came back nearly unchanged, or the prompt replaced it entirely; nothing useful in between | SSIM to the source 0.72–0.83 at three settings, 0.50 and 0.27 at the two lowest | **pending**: the mechanism was replaced after these runs (`85405e9`), chosen from the prototype runs; not yet run on real weights through the app |
| 05 | Text to video with sound | `M05_t2av` at `75d0c53`; duration head `X1_duration_head` at `5e4829a` | a ship in a storm, a figure in yellow on deck, the prompt's line shouted; with the duration head, a lighthouse keeper climbs to the lamp | AAC 48 kHz stereo, 5.01 s, −11.5 LUFS, transcript = the quoted line word for word; the duration head "chose 121 frames (5.04 s)" of the 5–10 s allowed, transcript exact | works |
| 06 | Image to video with sound | `M06_i2av` at `75d0c53` | the supplied cabin photo comes alive (smoke, drifting aurora, slow camera move); a low ambient drone | frame 0 vs the photo: SSIM 0.85, PSNR 27.7 dB; −14.6 LUFS | works |
| 07 | Image + text to video with sound | `M07_it2av` at `75d0c53`, 480×704 portrait | the supplied portrait, upright; his mouth opens on the shouted words | frame 0 vs the photo: SSIM 0.87, PSNR 31.8 dB; transcript exact | works |
| 08 | Text to audio | `M08_t2a` at `75d0c53` | a male voice reads the announcement | wav only, 48 kHz stereo, 5.01 s, −15.1 LUFS; transcript exact | works |
| 09 | Video to audio | `M09c_v2a_captain_noprompt`, `M09d_v2a_captain_prompt`, `M09e_v2a_tram_prompt`, `M09f_v2a_tram_noprompt` at `b12415f` | on a clip of a man shouting (only the picture reaches the model), a voice in time with his mouth, in different words; on the tram clip, machine hum and a low rumble | loudness envelope vs the clip's own soundtrack: correlation 0.92 (lag −40 ms); waveform correlation 0.03, so generated rather than copied; −18.0 / −13.5 LUFS (man), −31.3 / −26.7 LUFS (tram) | works, with and without a prompt |
| 10 | Audio to audio | `M10_a2a_l042`, `M10_a2a_l073`, `M10_a2a_l091`, `M10_a2a_l073_noprompt` at `b12415f` | the ocean recording's two swells stay in place at levels 0.42 and 0.73 and are gone at 0.91, where the prompt's rain takes over | same events at the same times (mel-modulation correlation) 0.90 / 0.85 / 0.59; unrelated-sound controls 0.00 and 0.33; before the fix (`M10_a2a_before_ns05`): clipped and unrelated (−0.04) | works as designed. **Open:** every level came back about 9 dB quieter and duller than the input; not explained yet |
| 11 | Audio to video | `M11_a2v_after`, `M11b_a2v_firstframe`, `M11c_a2v_noprompt`, `X1b_a2v_operator_audio` at `b12415f` | a clear storm scene on a ship's deck (before the fix, `X4_a2v_before_ns05`: a grey veil); with a portrait as first frame, the man shouts the recorded line; with no prompt, a man talking | the soundtrack is the supplied audio (correlation 1.0000 at 0 ms); length from the audio (4.83 s → 113 frames; the operator's 8.16 s, 24 kHz mono recording → 193 frames, written as 48 kHz stereo, +0.06 dB); luma p1–p99 9–180, contrast 51.3 (before: 59–105, 13.2); first frame vs the portrait SSIM 0.87 | works. **Open:** with no first-frame image the picture was black and white in three of three runs |

**Other checks on real weights.**

- **LoRA.** Text to video at 704×480×49 with and without `SOLRICKS/LTX-2.5-BTS-Movie-Set` at
  0.8 (`X2a_t2v_nolora`, `X2b_t2v_bts08`): the adapter visibly changes the picture, to a film
  set with camera, boom and lighting truss. Image + text to video with and without
  `DigitalByte/LTX-2.5-Exploded-View-XPLDV` at 1.0 (`X2e_it2v_nolora`, `X2f_it2v_xpldv10`):
  the adapter's exploded view appears. In image to video (`X2c_i2v_xpldv05`, `X2d_i2v_nolora`)
  it changed the motion only: that mode has no prompt, so the adapter's trigger words could
  not be given. Code `5e4829a`.
- **Conditioning clips.** A 60 fps clip is read in real time (`X7a_v2v_60fps`), and a clip
  stored sideways with a rotation flag comes out upright (`X7b_v2v_rotated`, SSIM 0.90 against
  the upright reference). Code `5e4829a`.
- **Not checked yet on real weights:** the prompt enhancer after its transformers 5.23.0 fix
  (`2f356b5`; it failed to load before, `X3b_t2av_enhanced`); the duration head at the
  operator's failing size (512×960, 5–10 s), which does not fit this Mac's memory budget.

**Memory and time.** Committed memory is the whole machine's active + wired + compressed
pages, sampled through each run; the operator's limit for these runs was 90 GB.

| runs | peak committed | MPS driver | denoise + decode |
|---|---|---|---|
| `G0b`, `G1`, `G2`, `G3`: 256×256×9, 512×320×25, 704×480×49, 704×480×121 | 56.9 / 57.0 / 57.0 / 58.0 GB | 35.8 / 35.8 / 37.0 / 41.5 GB | 49 / 65 / 148 / 333 s |
| batch B1: ten runs up to 704×480×121 | 59.9–61.3 GB | ≤ 42.3 GB | |
| `X1b_a2v_operator_audio`: 704×480×193 (8 s) | 70.1 GB | | 563 s |
| runs made while other jobs' test suites ran on the same machine | up to 74.8 GB | | |

The lead records no swap growth for `G0b`, `G3` and batches B1–B4. `G3`'s 333 s was taken
while five test suites ran; repeated on a quieter machine (`G3r_rope64`) it took 142 s.
Loading adds about a minute (`G0`: small components 3.9 s, text encoder 12.0 s plus 4.2 s to
encode, transformer 32.5 s).

Not run locally, by design: Upscale ×2, the diffusion decoder and bitsandbytes int8 need CUDA.
Their state is in 10.1.
