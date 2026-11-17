# The mode matrix on the live Space: 11 of 11 execute, and the defect that took two of them out

> **How to read this document (2026-11-16).** §1–§9 are the August 2026 matrix on the Space,
> made with builds whose output was colour static (the block below); they are corrected in
> place where later measurements overtook them. **§10 is the current state:** what has been
> verified since the 2026-11-16 fixes, on the Space after deploy 1 (§10.1, and the size and
> length it finishes, §10.3) and on real weights on a Mac (§10.2). Deploy 2's own check on the
> Space is reported on the tracking issue, [#32](https://github.com/techfreakworm/ltx2.5-studio/issues/32).

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
> verified since: on the Space after deploy 1 (§10.1) and on a Mac (§10.2). Every section
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
| 7 | ⏸️ **Only one configuration has been exercised** — 256×256×9, stage 2 off, no LoRA, no `auto_duration`. Nothing here says what any mode does at a shippable resolution, and stage 2 has never run on the Space at all. **→ SUPERSEDED (2026-08-25): 704×480×49 and 1216×704×121 (via the tiled decode, `cdb3211`) both pass; stage 2 ran for the first time — PASS at 704×480×25 input (1408×960 out), FAIL at ×49 in the diffusion decoder, so its ceiling is between those; LoRA exercised in both directions with a passing inertness check. Still never exercised on the platform: `auto_duration`. `SPACE_SESSION_2026-08-25.md`.** **→ ⛔ Corrected 2026-11-16: the ×49 failure was the `large` partition, and ×49 passes on `xlarge` (§1 of [`MIG_ALLOCATOR_SIGHTINGS.md`](MIG_ALLOCATOR_SIGHTINGS.md)); the inertness check compared two broken renders and is withdrawn (the block at the top). `auto_duration` has since run on real weights locally (§10.2); on the platform it never has.** **→ On the Space after deploy 1 (2026-11-16): the model picked the length at 512×960 with 5–10 s allowed (§10.1, row 9).** | ✅ superseded · ✅ auto_duration on the Space (§10.1) |
| 8 | ⏸️ **No mode has been shown to FOLLOW its conditioning input.** Every check in this document is a property of the output alone — motion, level, distinctness. `video_to_video` returning mean \|Δ\| 17.277 from an 18.894 input is equally consistent with faithful conditioning and with ignoring the clip entirely (§2.1). Settling it needs a comparison the runner does not currently make: same seed, two different conditioning clips, and a difference in the output. **→ THE NAMED COMPARISON WAS MADE (2026-08-25): same seed, same prompt, two clips — outputs 17.277 vs 0.864, mirroring their inputs (18.894 vs 0.905), against a control pair whose noise floor is exactly zero (byte-identical). Conditioning drives the output; the prompt-only null hypothesis is dead. Resolved for motion-following; *semantic* fidelity — content, not motion energy — remains ⏸️ untested. `SPACE_SESSION_2026-08-25.md` §7.** **→ ⛔ THE RESOLUTION IS WITHDRAWN (2026-08-25, later). R7a and R7b both ran on the broken int8 build; the comparison was between two garbage renders. It shows the conditioning *path* carries a number across the fork — a mechanical result, and it stands — and it does not show that any mode follows its conditioning input. This item returns to ⏸️ open, exactly as originally written, and the clip-swap is the right experiment to re-run against `e3b5333`.** **→ Locally, on real weights (2026-11-16, §10.2): the image modes start on the supplied image (frame-0 SSIM 0.74–0.87 against it), and video to sound follows what is visible in the clip. On the Space: not re-run.** **→ On the Space after deploy 1 (2026-11-16, §10.1): the image modes start on the supplied image (frame-0 SSIM 0.741–0.872 against it), and audio to video's sound is the supplied recording (correlation 0.982–0.990).** | ✅ on the Space (§10.1) · ✅ locally (§10.2) |
| 9 | ⚠️ **`video_to_audio` is 8–12 dB quieter than every other audio mode** (§6.1) — −19.5 dBFS against −7.6 to −11.1. Passes the level checks, is nowhere near the −40 dBFS flag, and is the only audio mode conditioned on video. Recorded because H9 was a level anomaly that passed every structural check. **→ Annotation (2026-08-25): now CONSISTENT-WITH the conditioning finding — outputs mirror their conditioning input (`SPACE_SESSION_2026-08-25.md` §7), and a gentle scene plausibly yields quiet audio. Consistent-with is not established-by: nothing has probed audio conditioning directly, so this stays what it is — an observation.** **→ ⛔ The consistent-with reasoning is withdrawn (2026-08-25, later): it derived entirely from §7, which is itself withdrawn above. −19.5 dBFS was the level of audio generated by a broken transformer. Unexplained again, and cheap to re-measure.** **→ Locally, on real weights (2026-11-16, §10.2): video to sound follows what is visible; an eventless clip gives a quiet, eventless track. Not re-measured on the Space.** **→ Re-measured on the Space after deploy 1 (2026-11-16, §10.1): on the tram clip, −44.2 dBFS at seed 7 and −35.9 dBFS at seed 19.** | ✅ re-measured on the Space (§10.1) · ✅ locally (§10.2) |
| 10 | ⚠️ **The quota ledger's history begins mid-day** (§7). It reports 67.252 GPU-s for a day whose real bound is ~413, because it was installed after the matrix ran. The figure is accurate for what it observed and misleading if quoted as a daily total; nothing in its output says where its history starts. | ⏸️ open |
| 11 | ⛔ **RECALIBRATE THE STRUCTURE DETECTOR AGAINST POST-FIX ARTIFACTS** (2026-08-25). `e54477c`'s SR/HF thresholds were fitted to a "real" population of eight banked artifacts that were all colour static, and one flat-grey failure. It therefore separates *flat* from *everything else* and does not separate static from picture: SR 0.9183 (static) against 0.9961–0.9968 (the post-fix photograph), with the 0.70 gate below both. The photograph's HF (0.075–0.133) sits **outside** the band the table calls real (0.147–0.386). The detector's conclusion is unaffected — it never promotes — but its numbers now describe the wrong two populations. **→ Recalibrated on 2026-11-16 (`752b18e`) on a labelled set of real frames from the local runs, with three named conditions (flat, veiled, static), still demote-only. On five correct outputs it had not seen it gave no note, and it flagged the pre-fix veiled audio-to-video output as "veiled".** | ✅ recalibrated (local evidence) |
| 12 | ⏸️ **The post-fix eleven-mode matrix** — every mode re-run against `e3b5333` at a shippable config, with a frame from each looked at. §10 is where it goes. Until it lands, the only output this project has ever verified as an image is the single post-fix comparison frame. **→ ⛔ Corrected 2026-11-16: it ran on 2026-08-25 but was never written up and its files are gone. §10.2 records the local real-weights matrix of 2026-11-16; the Space matrix is to be run after the next deploy (§10.1).** **→ Run on deploy 1 (2026-11-16): 24 calls, all eleven modes, every output looked at (§10.1).** | ✅ on the Space, 11 of 11 (§10.1) · ✅ locally, 11 of 11 (§10.2) |
| 13 | ⏸️ **The NVML allocator assert has now been seen THREE times, and the record held them apart** (2026-08-27). R3's VAE decode and R4b's diffusion decoder (both 2026-08-25, `SPACE_SESSION_2026-08-25.md` §4/§5) and the first enhanced generation in a fresh worker (2026-08-27) all died on `NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1165`. Collected side by side in [`MIG_ALLOCATOR_SIGHTINGS.md`](MIG_ALLOCATOR_SIGHTINGS.md) with the shared shape stated as a **hypothesis, explicitly not established** — this partition may mishandle a large allocation that follows another large allocation. **Deliberately not being chased**: sighting 3's premise disappears if the enhancer is made resident, and every question is blocked on a headroom number nobody has. `0a4ddbb` now reports peak VRAM (allocated / reserved / total, split by phase) on every generation, so one flagship run settles whether there was room. **→ ✅ IT DID, AND THE ANSWER WAS THE PARTITION (2026-08-27). The first run to print `total_memory` read 47.375 GiB — ZeroGPU's `large` — where the whole build assumed `xlarge`'s 94.971 GiB. `LTX25_ZEROGPU_SIZE=auto` sends no `gpuSize` at all, and the packer's own promotion never leaves `get_duration_seconds`. R4b is RETIRED by measurement: re-run on `xlarge` it PASSES at 53.08 s with peak 55.906 GiB reserved — 8.5 GiB more than the entire old partition, so it was arithmetically impossible and always would have been. ⏸️ The other two sightings are NOT retired: R3's untiled path is consistent-but-permanently-untestable (tiling is unconditional since `cdb3211`, and a tiling-off toggle was refused), and the enhancer sighting is pending the residency change. The single-cause account is SUPPORTED, NOT PROVEN.** **→ (2026-11-16) The residency change is `52a0f56`, deployed on 2026-08-26; no enhanced run has been recorded on the Space since, so sighting 3 is still pending.** **→ (2026-11-16, deploy 1) An enhanced run on the Space loaded the enhancer and finished, peak 61.426 GiB reserved of 94.971 GiB (§10.1, row 10).** | ✅ root cause found · ⏸️ 2 of 3 sightings unretired |

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

Nothing in §1–§9 describes the code after the 2026-11-16 fixes. This section does: 10.1 is
deploy 1 on the Space, 10.2 is what has been verified on real weights on a Mac, and 10.3 is the
size and length the Space was measured to finish.

### 10.1 On the Space, deploy 1 (2026-11-16)

Recorded by the lead in the run log of the 2026-11-16 fix (the entries of 21:32, 21:41 and
21:57). Each call's files are kept under its label in that run's evidence directory, outside
this repo: `evidence/space/matrix/<label>/` holds what was sent (`request.json`), what came back
(`result.json` with the page's own report, and the files), the client's measurements
(`verify.json`, `timing.json`, the progress events in `events.jsonl`) and the lead's
inspection (`inspect/`: six frames at native resolution, a frame sheet, a spectrogram and
`report.json`); `evidence/space/deploy1/` holds the build log, the Space's start-up and run
logs, the endpoint check (`info.txt`) and the timings (`timings.md`). Every figure below is in
those files, under the label its row names.

**What ran.**

| | |
|---|---|
| Space | `techfreakworm/LTX2.5-Studio` on ZeroGPU; private during these calls, public since |
| Code | Space commit `0cd54f2`, whose tree is `1e9fd20` on `fix/studio-2026-11`: deploy 1 ([`RELEASE_NOTES.md`](RELEASE_NOTES.md)) |
| Packages | as pinned (the build log): transformers 5.23.0, tokenizers 0.23.3, bitsandbytes 0.50.2, kernels 0.16.2, accelerate 1.15.0, peft 0.20.0, huggingface_hub 1.33.0, safetensors 0.8.0, numpy 2.5.5, gradio 6.24.0, diffusers 0.40.0.dev0 at the pinned commit; the platform's `spaces` 0.51.3 |
| Cold start | 17 min 24 s (15:44:26 → 16:01:50 UTC), 14 min 55 s of it fetching 37 files, 72.19 GB, at the model's revision `871165de` and the decoder's `aba1b702` |
| Start-up report | "precision: int8 for transformer, text_encoder (bitsandbytes LLM.int8; the transformer keeps its conditioning, projection and gate Linears bf16 via 13 skip patterns); bf16 for every other component"; "spaces: 0.51.3"; "packed copy: not measured yet, ZeroGPU writes it at launch after this report"; "quantised transformer, text_encoder as int8 \| peak RSS 0.79 -> 85.67 GiB" |
| GPU | `xlarge`, requested by the app; every run's report reads 94.971 GiB total |
| Endpoint | all 19 inputs of `/on_generate` match the deployed commit's field spec |
| Client | `scripts/space_matrix.py` from a clean export of `1e9fd20`; 24 calls, one at a time, 16:02–16:27 UTC |
| Files | H.264 video at 24 fps, with AAC 48 kHz stereo sound in the modes that have sound; every mode that returns sound also returns it as a 48 kHz stereo `.wav` |

**The inputs.** The tram prompt: "A red vintage tram climbs a steep cobbled street at golden
hour, slow push-in. The tram bell rings twice and steel wheels grind on the rails." The captain
prompt: 'A weathered sea captain in a yellow raincoat stands on the deck of a ship in a storm
and shouts: "The storm is coming! All hands on deck!" Thunder rolls and waves crash against the
hull.' The close-up prompt: "Close-up of a weathered sea captain shouting on the deck of a ship
in a storm, rain on his face." Files prepared for the test: a tram picture and a cabin picture
(`tram_landscape.png`, `cabin_landscape.png`), the captain's portrait (`captain_portrait.png`),
a cabin clip and a tram clip (704×480, 121 frames), an ocean recording (5.0 s), the captain's
recorded line (4.83 s, 48 kHz stereo) and the operator's own recording (8.16 s, 24 kHz mono).

**Per call.** Seed 7, the Decoder at Auto, no adapter, no Upscale and no Enhance unless the row
says otherwise; the four sound-only calls render at 256×256, as the app does in every mode that
returns only sound. The window runs from the worker's start to its end (`timing.json`); GPU-s
is that window × 2, the `xlarge` multiplier: an upper bound, not billed seconds.

| # | label | sent | back | seen and measured | window · GPU-s ≤ |
|---|---|---|---|---|---|
| 1 | `m01_text_to_video` | Text to video, the tram prompt, 704×480, 121 frames | 704×480, 121 frames, silent, 1,104,842 B; "the frames were decoded by the model's diffusion decoder"; peak 53.090 GiB reserved | a tram on a cobbled street, sharp | 32.3 s · 64.6 |
| 2 | `m04_video_to_video` | Video to video, the cabin clip, "The same cabin in heavy snowfall at dusk, warm light glowing in the windows.", A lot | 704×480, 121 frames, 1,147,376 B; "Length set from your clip: 5.0 s." "Your clip was changed a lot." | the clip varied: SSIM 0.433 to it | 17.0 s · 34.0 |
| 3 | `m05_text_to_audio_video` | Text to video with sound, the captain prompt, 704×480, 121 frames | the mp4 with picture and sound, 1,135,358 B, and the `.wav`, 961,998 B | transcript exact ("The storm is coming! All hands on deck!"); −19.0 dBFS | 24.0 s · 47.9 |
| 4 | `m10_audio_to_audio` | Audio to audio, the ocean recording, "Heavy rain on a tin roof with distant rolling thunder.", Moderately | a `.wav`, 4.73 s, 908,238 B; "Your audio was trimmed from 5.00 s to 4.72 s to match the video." "Your audio was changed moderately." | +1.4 dB against the input (locally +1.5 dB, §10.2 row 10) | 22.2 s · 44.5 |
| 5 | `m11_audio_to_video_first_frame` | Audio to video, the captain's line, his portrait as the first frame, the close-up prompt, 480×704, length from the audio | 480×704, 113 frames, 1,047,762 B, and the `.wav`; "Length set from your audio: 4.7 s." "The sound in this file is your audio, unchanged." | in colour; frame 0 against the portrait SSIM 0.872 (locally 0.871, `M11h`); the sound against the recording: correlation 0.990, level +0.06 dB | 24.7 s · 49.3 |
| 6 | `x2_m05_text_to_audio_video` | row 3 at 49 frames with **Upscale ×2** | 1408×960, 49 frames, 1,492,822 B, and the `.wav`, 385,998 B; the progress read "sampling — pass 1 of 2", "upscaling ×2", "sampling — pass 2 of 2", "decoding the frames"; peak 55.912 GiB reserved | stage 2 works: added detail at native size (railings, rust, chain), not washed out; transcript "The storm is coming all hands on deck"; the sound reaches 0 dBFS, 30 samples clipped | 33.4 s · 66.7 (reserved 270 s) |
| 7 | `dec_m01_fast` | row 1 with the standard (conv) decoder, the page's "Fast" at deploy 1 | 704×480, 121 frames, 1,686,458 B; "the frames were decoded by the standard (convolutional) decoder, as chosen"; peak 42.711 GiB reserved | the same scene as row 1, SSIM 0.876 to it (the decoder A/B below) | 19.6 s · 39.2 |
| 8 | `x1_m05_49f_single_pass` | row 3 at 49 frames without Upscale: row 6's twin | 704×480, 49 frames, 524,955 B, and the `.wav`, 385,998 B | transcript exact; the sound reaches 0 dBFS, 78 samples clipped: as loud as row 6's, so Upscale is not what makes it loud | 37.4 s · 74.8 |
| 9 | `len_m05_512x960_model_picks` | row 3 at 512×960 with "Let the model pick the length", 5–10 s | "The model chose the length: 5.0 s."; 512×960, 121 frames, 2,202,425 B, and the `.wav`; the Space's log (`run_log_after_11_calls.log`): "Duration prediction clamped: raw 4.47s outside [5.00s, 10.00s], using 5.04s (121 frames) @ 24.00 fps" | transcript exact | 28.2 s · 56.3 |
| 10 | `enh_m05_text_to_audio_video` | Text to video with sound with **Enhance my prompt**: "A sea captain shouts a warning on a ship's deck as a storm hits." | 704×480, 121 frames, 1,582,284 B, and the `.wav`; the progress began "enhancing the prompt"; the Space's log (`run_log_after_11_calls.log`) has transformers' MISMATCH line for one of the enhancer's weights, then "restored model.language_model.embed_tokens_per_layer.embedding_scale [262144, 35] from model.safetensors" and "prompt enhanced: 64 chars in, 1030 chars out"; peak 61.426 GiB reserved | a storm-at-sea sequence ending on a captain shouting; transcript "Hold hold fast ye dogs!" | 51.6 s · 103.3 (reserved 360 s) |
| 11 | `op_m11_operator_audio` | Audio to video, the operator's recording, the close-up prompt, 512×960, length from the audio | 512×960, 193 frames (8.04 s), 2,559,888 B, and the `.wav`; "Your audio was converted from mono at 24 kHz to stereo at 48 kHz to match the video's soundtrack. Its pitch and length are unchanged." "Your audio was trimmed from 8.16 s to 8.04 s to match the video."; peak 63.301 GiB reserved | the sound is the recording: correlation 0.988 (the +3.0 dB the inspector prints compares the mono file with the stereo one) | 36.5 s · 73.0 |
| 12 | `lora_m01_bts_w100` | Text to video with the adapter `SOLRICKS/LTX-2.5-BTS-Movie-Set` from the Hub at 1.0, "btssetstyle, a film crew sets up lights around a red vintage tram on a cobbled street at golden hour.", a random seed | Validate on the Space: "Every check before a run passed: the file reads cleanly and every layer fits its target in LTX-2.5.", "Fits LTX-2.5: 1632 of 1632 layers", rank 16; the run: seed 2690990278, 704×480, 121 frames, 1,449,691 B | a film crew with camera rigs around the tram | 36.4 s · 72.9 (reserved 240 s) |
| 13 | `m02_image_to_video` | Image to video, the tram picture, no prompt | 704×480, 121 frames, 1,908,309 B | frame 0 against the picture SSIM 0.742 (locally 0.75, §10.2) | 21.9 s · 43.7 |
| 14 | `m03_image_text_to_video` | Image + text to video, the tram picture and the tram prompt | 704×480, 121 frames, 1,759,736 B | frame 0 SSIM 0.741 (locally 0.74) | 17.8 s · 35.7 |
| 15 | `m06_image_to_audio_video` | Image to video with sound, the cabin picture, no prompt | 704×480, 121 frames, 1,176,212 B, and the `.wav` | frame 0 SSIM 0.850 (locally 0.85); −24.3 dBFS | 23.7 s · 47.5 |
| 16 | `m07_image_text_to_audio_video` | Image + text to video with sound, the tram picture and the tram prompt (a bell and wheels; no speech) | 704×480, 121 frames, 1,842,167 B, and the `.wav` | frame 0 SSIM 0.741; no speech asked for and none made; a quiet metallic sound, −35.4 dBFS with peaks at −15.5 | 28.2 s · 56.5 |
| 17 | `m08_text_to_audio` | Text to audio, 'A calm male voice says: "Welcome aboard. Mind the gap between the train and the platform." A two-tone door chime follows.' | a `.wav`, 5.01 s, 961,998 B | the announcement, with two words of the instruction read aloud: "voice says welcome aboard mind the gap between the train and the platform"; −23.4 dBFS | 13.5 s · 27.0 |
| 18 | `m09_video_to_audio` | Video to audio, the tram clip, "A tram bell, steel wheels on rails, city street ambience." | a `.wav`, 5.01 s, 961,998 B | −44.2 dBFS, peaks at −31.5 | 22.4 s · 44.8 |
| 19 | `m11_audio_to_video` | Audio to video, the captain's line, the close-up prompt, 704×480, no picture | 704×480, 113 frames, 1,108,958 B, and the `.wav` | a **black-and-white** period-film picture of the captain (saturation 0.012); the sound is the recording (correlation 0.990, +0.06 dB) | 23.1 s · 46.2 |
| 20 | `x2_m11_audio_to_video` | row 19 at 49 frames with **Upscale ×2** | 1408×960, 49 frames, 1,603,601 B, and the `.wav`; "Your audio was trimmed from 4.83 s to 2.04 s to match the video."; peak 56.537 GiB reserved | in colour, a ship at night in rain, coherent: stage 2 runs in a mode it used to fail in; the sound against the recording: correlation 0.982 | 29.4 s · 58.8 (reserved 270 s) |
| 21 | `m11_audio_to_video_seed21` | row 19 at seed 21, the seed of the local run `M11g_a2v_same_prompt` | 704×480, 113 frames, 920,176 B, and the `.wav` | in colour (saturation 0.402; `M11g` 0.29, §10.2): a ship at sea | 21.7 s · 43.4 |
| 22 | `m09_video_to_audio_seed19` | row 18 at seed 19 | a `.wav`, 5.01 s, 961,998 B | −35.9 dBFS, peaks at −23.8 | 10.5 s · 21.1 |
| 23 | `m05_text_to_audio_video_seed12` | row 3 at seed 12, the seed of the local final pass's `F05_t2av` | 704×480, 121 frames, 1,039,616 B, and the `.wav` | transcript exact; −15.2 dBFS | 22.9 s · 45.9 |
| 24 | `lora_twin_no_adapter` | row 12 without the adapter, at row 12's seed | 704×480, 121 frames, 1,212,419 B | a different picture: SSIM 0.322 to row 12 (the adapter A/B below) | 22.2 s · 44.3 |

**What it cost:** at most 1241.3 GPU-s for UTC 2026-11-16, against the run's ceiling of 1920
(80 % of the PRO quota's 2400). No quota refusal, no queue timeout, no failed run. Every
window was 10.5–51.6 s, against reservations of 180–360 s: a plain run at 704×480 with 121
frames reserved 180 s, so it needed 360 s of quota free to start, and drew at most 34.0–64.6
GPU-s.

**How the outputs were judged.** The lead looked at every video's six frames at native
resolution and its frame sheet, and measured every sound track: its level (the RMS of the
`.wav`), a spectrogram and, where there is speech, a transcript. The app's picture check (in
`verify.json`) raised no condition on any of the 20 videos: it demotes and never promotes, and a
black-and-white picture (row 19) is not one of its conditions. At the time the client called
the 13 runs that return video with sound a MISMATCH (`result.json`, `outcome`), because it
expected a picture stream in the `.wav` the app returns beside the mp4. That was the client's
defect, not the app's, fixed at `a40a311` (its replay of row 3's files passes); the lead
judged those files with his own tools.

**What these calls settle**, the parts only a GPU can show:

- transformers 5.23.0 and bitsandbytes 0.50.2 load the int8 transformer and text encoder, and
  the start-up report says int8;
- on CUDA the default decode is the diffusion decoder, at its own revision (the Space's log:
  "diffusion_decoder: ignored the expected extra key(s) decoder.type_emb"), and its pictures
  are correct; the report of every run that returns video names the decoder that ran;
- Upscale ×2 runs in text to video with sound and in audio to video (rows 6 and 20): in row 6
  it adds detail rather than washing the picture out, and row 20 is a coherent picture in
  colour;
- progress crosses the ZeroGPU fork under `spaces` 0.51.3, in the app's own words, and every
  run is granted and measures the whole `xlarge` GPU;
- the prompt enhancer loads and rewrites on CUDA under transformers 5.23.0 (row 10);
- the model picks the length at 512×960 (row 9);
- an adapter from the Hub passes Validate on the Space and changes the picture (rows 12, 24);
- the image modes start on the supplied image as they do on the Mac (rows 5, 13–15), and audio
  to video's sound is the recording (rows 5, 11, 19, 20).

**Decoder A/B** (rows 1 and 7: the same seed, prompt and size; `ab_decoder/`: `full_0072.png`,
`crop3x_0072.png`, `measure.txt`). The same scene (SSIM 0.876). The diffusion decoder is a little
softer and steadier from frame to frame; the standard decoder is sharper frame by frame and
shimmers more. Both pictures are correct.

| | diffusion decoder (row 1) | standard decoder (row 7) |
|---|---|---|
| fine detail (std) | 10.30 | 11.65 |
| Laplacian variance | 421.6 | 591.6 |
| frame-to-frame change of the fine detail (shimmer) | 1.471 | 1.883 (28 % more) |
| from the last sampling step to "writing the file" (`events.jsonl`) | 4.4 s | 1.9 s |

**Adapter A/B** (rows 12 and 24: the same prompt and seed). With the adapter, a film crew with
camera rigs around the tram; without it, a street scene (SSIM 0.322; frame 72 of both side by
side: `ab_adapter_frame72.png`).

**Audio to video and colour** (rows 19, 21, 5). The same recording and prompt gave a
black-and-white period-film picture at seed 7 (saturation 0.012) and a colour picture at seed 21
(0.402); with a first-frame picture it came out in colour. On the Mac, before the
audio-encoding fix, all three runs with no picture and no colours named in the prompt came out
black and white (§10.2, row 11); now the look depends on the sample.

**Open at deploy 1, found by these calls:**

- the model's sound for a 2 s shouted line reaches full scale and is clipped when written: 30
  and 78 samples at 0 dBFS in rows 6 and 8;
- the Space's log is flooded by one bitsandbytes warning, one per int8 matmul ("MatMul8bitLt:
  inputs will be cast from torch.bfloat16 to float16 during quantization"): 55,400 of the
  55,534 lines of `run_log_after_11_calls.log`;
- when the model's own length is outside the range and is moved to its edge, the note says
  only "The model chose the length" (row 9);
- the diffusion decoder is loaded again inside the runs ("Some weights of the model checkpoint
  … were not used when initializing" appears in the run log at each load), not once at
  start-up;
- the endpoint returns the files, the notes and the diagnostics, but neither which adapter was
  attached nor the time the run took.

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
  says otherwise. **The app's own UI was not part of these runs**; the runs through the page
  are at the end of this section.
- The code is named per row. Most runs predate `c712e7e`, which made the local runner a
  `StageRunner`, one stage graph with the Space's (the local side supplies only the
  residency); the lip-sync runs below (`aa7ff84`, `482800d`) ran after it, and so did the
  final pass, which repeated every mode on `e59944e` (below).
- The lead opened every output: six native-resolution frames per video; per soundtrack the
  loudness, a spectrogram and a transcript; for conditioned modes, frame 0 against the
  supplied image (SSIM, PSNR). Inputs were files prepared for the test (generated images, synthetic
  clips, recorded speech, the operator's own recording); the video-to-sound clips were two
  outputs of earlier runs.

**Per mode.** All eleven verified, and all eleven again on the final code (the final pass,
below).

| # | mode | run id(s) and code | what was seen and heard | measured | verdict |
|---|---|---|---|---|---|
| 01 | Text to video | `G3_704x480x121` at `75d0c53`; repeated as `G3r_rope64` at `5e4829a` | a red tram climbs a cobbled street and passes the camera, coherent over the 5 s | luma p1–p99 1–248, contrast 64.5; the repeat: same scene, statistics within 1 % | works |
| 02 | Image to video | `M02_i2v` at `75d0c53` | frame 0 is the supplied tram image; the tram then rolls past the camera | frame 0 vs the image: SSIM 0.75, PSNR 21.8 dB | works |
| 03 | Image + text to video | `M03_it2v` at `75d0c53` | starts on the supplied image, slow push-in as prompted | frame 0 vs the image: SSIM 0.74 | works |
| 04 | Video to video | `V04_l042_summer`, `V04_l073_summer`, `V04_l091_car`, `V04_l091_noprompt`, `V04_rot_l042`, `V04_60fps_l042` at `85405e9`; `S04_summer_l091` at `7cf1547`. The earlier mechanism: `M04_v2v`, `M04b_v2v_s05` … `M04e_v2v_s00` at `75d0c53`/`5e4829a`; prototype `V2V_LEVEL_*` at `5e4829a` | at "a little" and "moderately" the clip comes back re-rendered (snow and aurora stay) and the summer prompt does not act; at "a lot" with "a red sports car on a desert highway" the cabin, trees and aurora stay in place, a road with a centre line now runs past the cabin and the snow is gone from the roof; at "a lot" with the summer prompt the cabin and the camera move stay, the snow goes and the aurora sky becomes a green hillside; without a prompt, the same scene re-imagined; the sideways-stored phone clip comes out upright, 480×704 | SSIM to the clip: 0.778 / 0.646 at 0.42 / 0.73; at 0.91 0.388 (car), 0.405 (no prompt), 0.362 (summer); the rotated clip 0.863 against the upright reference; the 60 fps clip at 0.42: 113 frames for its 4.7 s, SSIM 0.757 against the 24 fps original; length = the clip's (121 frames), with a note. The earlier mechanism (the clip as a condition with a strength) was a switch: SSIM 0.72–0.83 at three settings, 0.50 and 0.27 at the two lowest, where the prompt replaced the clip | works: graded by level, and the prompt acts at "a lot" |
| 05 | Text to video with sound | `M05_t2av` at `75d0c53`; duration head `X1_duration_head` at `5e4829a` | a ship in a storm, a figure in yellow on deck, the prompt's line shouted; with the duration head, a lighthouse keeper climbs to the lamp | AAC 48 kHz stereo, 5.01 s, −11.5 LUFS, transcript = the quoted line word for word; the duration head "chose 121 frames (5.04 s)" of the 5–10 s allowed, transcript exact | works |
| 06 | Image to video with sound | `M06_i2av` at `75d0c53` | the supplied cabin image comes alive (smoke, drifting aurora, slow camera move); a low ambient drone | frame 0 vs the image: SSIM 0.85, PSNR 27.7 dB; −14.6 LUFS | works |
| 07 | Image + text to video with sound | `M07_it2av` at `75d0c53`, 480×704 portrait | the supplied portrait, upright; his mouth opens on the shouted words | frame 0 vs the image: SSIM 0.87, PSNR 31.8 dB; transcript exact | works |
| 08 | Text to audio | `M08_t2a` at `75d0c53` | a male voice reads the announcement | wav only, 48 kHz stereo, 5.01 s, −15.1 LUFS; transcript exact | works |
| 09 | Video to audio | `M09c_v2a_captain_noprompt`, `M09d_v2a_captain_prompt`, `M09e_v2a_tram_prompt`, `M09f_v2a_tram_noprompt` at `b12415f`; `S09_captain_voice` at `7cf1547`, on a copy of the man's clip with no sound track at all | on a clip of a man shouting (only the picture reaches the model), a voice in time with his mouth, in different words; on the tram clip, machine hum and a low rumble | loudness envelope vs the clip's own soundtrack: correlation 0.92 (lag −40 ms); waveform correlation 0.03, so generated rather than copied; −18.0 / −13.5 LUFS (man), −31.3 / −26.7 LUFS (tram); `S09`: 0.926 against the track the clip originally had | works, with and without a prompt |
| 10 | Audio to audio | ocean `M10_a2a_l042`, `M10_a2a_l073`, `M10_a2a_l091`, `M10_a2a_l073_noprompt` and speech `M10s_speech_l042` / `_l073` / `_l091` at `b12415f`; after the audio-encoding fix, speech `M10f_speech_l042` / `_l073` / `_l091` and ocean `M10f_ocean_l073`, `M10f_ocean_l091_rain` at `7cf1547` | the ocean recording's two swells stay in place at levels 0.42 and 0.73 and are gone at 0.91, where the prompt's rain takes over; speech keeps its words at "a little" and nearly at "moderately" ("The store was withing, all hands on debt…"), and becomes other words at "a lot" | at `7cf1547`, level against the input: speech +0.6 / +1.1 / −6.8 dB at 0.42 / 0.73 / 0.91, ocean +1.5 dB at 0.73 and −1.4 dB at 0.91 with the rain prompt (at `b12415f`: −7.2 / −7.0 / −15.3, −8.9 and −11.6); same events at the same times (mel-modulation correlation): speech 0.97 / 0.95 / 0.71, transcript exact at 0.42; ocean 0.86 at 0.73 and 0.59 at 0.91 (at `b12415f`: 0.90 / 0.85 / 0.59); unrelated-sound controls 0.00 and 0.33; before the first fix (`M10_a2a_before_ns05`): clipped and unrelated (−0.04) | works: graded by level, at the input's own loudness. The 7–9 dB loss seen at `b12415f` was the app's mel transform (window 512, taken from the vocoder's config; F69), fixed at `7cf1547` (window 1024): the app's own encode → decode round trip, with no generation, now returns speech +0.2 dB and the ocean −0.3 dB (it was −8.9 / −9.7) |
| 11 | Audio to video | `M11_a2v_after`, `M11b_a2v_firstframe`, `M11c_a2v_noprompt`, `M11d_a2v_colour_prompt`, `M11e_a2v_female_studio`, `M11f_a2v_ocean`, `X1b_a2v_operator_audio` at `b12415f`; after the audio-encoding fix, `M11g_a2v_same_prompt`, `M11h_a2v_firstframe`, `M11i_a2v_female_studio` at `7cf1547` | a clear storm scene on a ship's deck (before the first fix, `X4_a2v_before_ns05`: a grey veil); with a portrait as first frame, the man shouts while the recording plays, not in time with its words (below); with no prompt, a man talking; a woman's voice with a studio prompt, and the ocean recording with a beach prompt, in colour. At `b12415f`, three runs with no first-frame image and a prompt naming no colours came out black and white; after the audio-encoding fix the same audio, prompt and seed give the same scene in colour (`M11g`): the model had been hearing the upload about 9 dB quiet and dull | the soundtrack is the supplied audio (correlation 1.0000 at 0 ms); length from the audio (4.83 s → 113 frames; the operator's 8.16 s, 24 kHz mono recording → 193 frames, written as 48 kHz stereo, +0.06 dB); luma p1–p99 9–180, contrast 51.3 (before: 59–105, 13.2); first frame vs the portrait SSIM 0.87 (`M11h`: 0.871, transcript exact); `M11g`: saturation 0.29 against 0.03 for `M11_a2v_after`, luma 8–170, contrast 49.4; `M11i`: in colour, transcript exact | works, in every variant tried: prompt only, no prompt, a first-frame image, non-speech audio, the operator's own recording. It does not lip-sync (below) |

**Speech and lips in audio to video.** Measured with the lead's lip-sync tool (MediaPipe face
landmarks give the mouth's opening in every frame; that curve is correlated with the speech
level of the sound track; the null is the same statistic over shifted audio). The reference
for "in sync" is speech the model writes itself.

| clip | mouth vs speech, lag 0 | best (lag) | 95 % of the shifted-audio null | opening, speech − pauses |
|---|---|---|---|---|
| image + text to video with sound, jointly generated (`M07_it2av`) — the reference | **0.84** | 0.84 (0) | 0.29 | +2.6 σ |
| audio to video, portrait + recorded speech (`M11h`) | **−0.38** | −0.24 (+250 ms) | 0.35 | −0.6 σ |
| audio to video, studio voice, prompt only (`M11i`) | 0.07 | 0.31 (−125 ms) | 0.34 | +0.3 σ |
| control: `M11h`'s picture against another recording | 0.02 | | 0.24 | |

The mouth in `M11h` is open and moving, but not in time with the words: below the
shifted-audio null.

**A prompt technique** (the operator's tip: ask for perfect lip sync, lip movement matching the
audio, and quote the spoken words), A/B on the app's own path at `aa7ff84` — the first
real-weights runs after the one-stage-graph change `c712e7e` — with the same 4.83 s recording,
113 frames, and the same seeds with and without it (`LS_*`). Mouth vs speech at lag 0:

| case | seed | base prompt | + "perfect lip sync" + the quoted transcript |
|---|---|---|---|
| image + audio, close-up portrait (face 250–310 px) | 21 | −0.38 | −0.04 |
| | 22 | −0.31 | −0.17 |
| image + audio, medium shot (face 83–94 px) | 21 | −0.36 | −0.24 |
| | 22 | +0.04 | −0.09 |
| audio only, close-up prompt | 21 | +0.04 | −0.18 |
| | 22 | −0.04 | −0.03 |

A third arm, word-by-word acting directions in the prompt (close-up portrait, seed 21), scored
+0.07. With the technique the mouth no longer opens mostly in the pauses, but in no case does it
follow the words: no run, with or without it, scores above what an unrelated sound track does
(up to 0.2–0.4). In the mouth strips, without it the mouth opens after about half a second and
stays open, one long shout through words and pauses; with it the mouth opens and closes more,
but not with the words. The A/B was 17 runs and the probe below 11, all exit 0.

**How the audio is held while sampling** — a probe outside the app (the app's own path with
one switch each; portrait + recording, seed 21, at `482800d`: `A2V_*`, `MG_*`, `LV_*`,
`MG2_*`). Mouth vs speech at lag 0:

| how the audio is held during sampling | cross-modal guidance off | video modality guidance 3.0 | 6.0 |
|---|---|---|---|
| frozen clean, as the app does (and as the vendor's ComfyUI template for 2.3 does) | −0.38 | −0.28 | |
| at a constant level 0.42 / 0.725 | −0.41 / −0.41 | −0.29 / — | |
| re-noised to the video's level each step | −0.31 | **+0.24** (+0.16 with the technique above) | **+0.26** |
| the same until sigma 0.5, then clean | −0.30 | | |

Only matched noise levels together with modality guidance turn the sign, to about a third of
the way to the reference, at twice the denoise time, and more guidance does not help further.
Neither is in the app. The distilled LTX-2.5 transformer, run unguided, makes speech in sync
when it writes the sound itself, and its sound follows a given picture (row 09: 0.92), but its
picture does not follow a given voice at the level of syllables. Lightricks' audio-driven
lip-sync recipes for LTX-2.3 use the non-distilled model: the ComfyUI template "LTX-2.3: Image
Audio to Video" runs the dev model with the distilled LoRA at half strength, and Lightricks'
lip-sync guide for 2.3 names `modality_scale` (about 3.0) and CFG about 4, which is guidance.
This app runs the distilled transformer unguided; the repository's dev transformer
(`transformer_full`, about 44 GB more) is downloaded neither by the Space nor for the Mac.

**What the app says.** Audio to video sets a picture to a recording (content, mood, length; the
recording is the sound track, exactly) and does not lip-sync; its audio caption says so and
points to image + text to video with sound, with the line in quotes, for a face that speaks in
time. Nothing is baked into prompts or placeholders. Not started, and a cost decision for the
operator: the dev transformer with the vendor's lip-sync settings, or an audio-conditioning
LoRA if Lightricks publishes one for 2.5.

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
  the upright reference). Code `5e4829a`, with the earlier video-to-video mechanism; the same
  two clips through the current one are in row 04.
- **Prompt enhancer.** On transformers 5.23.0 it failed to load before any generation
  (`X3b_t2av_enhanced`, code `75d0c53`). After the fix (`2f356b5`) both routes work:
  `X3c_t2av_enhanced_F67` turned "a dog runs on a beach" into a full shot description and the
  video shows it; `X3d_it2av_enhanced_image` wrote details that are only in the supplied image.
- **Not checked yet on real weights:** the duration head at the operator's failing size
  (512×960, 5–10 s), which does not fit this Mac's memory budget.

**The final pass, on the final code** (the lead's run log, 2026-11-16, 19:16–20:16). Fourteen
runs on `e59944e`, the commit at which the backend was declared finished: every mode once more,
the adapter, the enhancer and the model-chosen length, each with the inputs, prompt, size and
seed of the run it is compared with, one at a time under the 90 GB watchdog; each output looked
at as a frame sheet, measured, and compared with that run. Eleven ran there. The other three
were refused there by the memory check's early reading and ran on `ba2d582`, which fixed it
(below). Two new audio-to-video runs on `e59944e` were added for the page's examples.

| # | mode | run | compared with | result |
|---|---|---|---|---|
| 01 | Text to video | `F01_t2v` | `S01_tram_silent` (`7cf1547`) | identical |
| 02 | Image to video | `F02_i2v` | `M02_i2v` (`75d0c53`) | the same scene, the tram approaching, framed slightly differently; frame 0 against the image 0.7507, as before |
| 03 | Image + text to video | `F03_it2v` | `S03_tram_silent` (`7cf1547`) | identical |
| 04 | Video to video | `F04_v2v` | `S04_summer_l091` (`7cf1547`) | identical; level 0.91 without the flag, the command line's default since `419e0a5` |
| 05 | Text to video with sound | `F05_t2av` | `M05_t2av` (`75d0c53`) | the same scene, a ship at sea, framed slightly differently; transcript exact |
| 06 | Image to video with sound | `F06_i2av` | `M06_i2av` (`75d0c53`) | the same scene, the cabin under the aurora, with its sound |
| 07 | Image + text to video with sound | `F07_it2av` | `M07_it2av` (`75d0c53`) | the captain speaks; transcript exact; mouth vs speech +0.845 |
| 08 | Text to audio | `F08_t2a` (`ba2d582`) | its earlier run | the announcement, transcript exact; the same sound (level 0.0 dB, envelope 1.000) |
| 09 | Video to audio | `F09_v2a` | `S09_captain_voice` (`7cf1547`) | identical |
| 10 | Audio to audio | `F10_a2a` | `M10f_ocean_l073` (`7cf1547`) | identical |
| 11 | Audio to video | `F11_a2v` | `M11h_a2v_firstframe` (`7cf1547`) | identical; mouth vs speech −0.379 |
| 11 | Audio to video: the ocean recording, a beach prompt | `F11b_ocean` | (new) | waves rolling onto a beach at sunset with gulls, in full colour; the sound track is the recording itself (level +0.1 dB against it) |
| 11 | Audio to video: the same recording, the captain's portrait as first frame, a prompt in which nobody speaks | `F11c_rail` | (new) | his lips closed, gulls passing behind him; frame 0 against the portrait SSIM 0.86 |
| | LoRA: `SOLRICKS/LTX-2.5-BTS-Movie-Set` at 0.8 | `F12_lora` | `X2b_t2v_bts08` (`5e4829a`) | identical; the run record says "on 1632 layers" |
| | Prompt enhancer | `F13_enh` (`ba2d582`) | `X3c_t2av_enhanced_F67` (`2f356b5`) | identical |
| | Model-chosen length | `F14_auto` (`ba2d582`, up to 8 s) | `X1_duration_head` (`5e4829a`, up to 10 s) | identical; the model chose 5.0 s (121 frames) both times; "Light the lamp!" |

Identical means SSIM 1.000 and, with sound, sound correlation 1.000: nothing that landed after
those runs changed an output. The four compared with runs made one commit before the float64
RoPE fix (`5e4829a`) show the same scenes framed slightly differently (SSIM 0.78–0.94), the same
words and sound (levels within 0.2 dB) and the same first-frame fidelity. The lip-sync figures
on these outputs are the ones in the tables above: +0.845 where the model writes the speech
itself, −0.379 in audio to video. `F11b_ocean` is mode 11's shipped sample, made from its first
example ("Waves, pictured"), and `F11c_rail` ran its second ("A portrait over the sea"); no one
speaks in either.

**The three runs the early memory reading refused** on `e59944e`, while test suites and a
browser session shared the machine:

| run | refused on `e59944e` | on `ba2d582` |
|---|---|---|
| `F08_t2a`, text to audio | `69.0 + 25.1 = 94.1` and `65.2 + 25.1 = 90.3 GB` > 90, at the transformer load | "memory settled in 0.16 s: 51.1 → 47.7 GB", then `47.7 + 25.1 = 72.8` OK |
| `F13_enh`, the enhancer | `70.8 + 25.1 = 95.9`, at the transformer load | settled in 0.36 s, 67.0 → 54.0 GB; `54.0 + 25.1 = 79.0` OK |
| `F14_auto`, the model-chosen length | `64.0 + 27.6 = 91.6` at the denoise check, which budgets for the longest clip the model may return: 233 frames for up to 10 s | settled in 0.26 s, 55.6 → 52.7 GB; with up to 8 s the denoise check is `65.1 + 20.0 = 85.1` OK |

The settled readings were 3 to 13 GB below the first ones, reached in 0.16–0.36 s.

**Memory and time.** Committed memory is the whole machine's active + wired + compressed
pages, sampled through each run; the operator's limit for these runs was 90 GB.

| runs | peak committed | MPS driver | denoise + decode |
|---|---|---|---|
| `G0b`, `G1`, `G2`, `G3`: 256×256×9, 512×320×25, 704×480×49, 704×480×121 | 56.9 / 57.0 / 57.0 / 58.0 GB | 35.8 / 35.8 / 37.0 / 41.5 GB | 49 / 65 / 148 / 333 s |
| batch B1: ten runs up to 704×480×121 | 59.9–61.3 GB | ≤ 42.3 GB | |
| `X1b_a2v_operator_audio`: 704×480×193 (8 s) | 70.1 GB | | 563 s |
| runs made while other jobs' test suites ran on the same machine | up to 74.8 GB | | |
| the final pass on `e59944e`, 13 runs, other jobs' test suites coming and going | 66.5–78.2 GB | | |
| `F08_t2a`, `F13_enh`, `F14_auto` on `ba2d582` | 77.0 / 77.6 / 84.5 GB; `F14_auto` (model-chosen length, up to 8 s) at a base load of about 46 GB | | |

The lead records no swap growth for `G0b`, `G3` and batches B1–B4. `G3`'s 333 s was taken
while five test suites ran; repeated on a quieter machine (`G3r_rope64`) it took 142 s.
Loading adds about a minute (`G0`: small components 3.9 s, text encoder 12.0 s plus 4.2 s to
encode, transformer 32.5 s).

**Through the page, in a real browser.** The app started as a user starts it (`python app.py`,
staged residency, the Q8_0 GGUF transformer), Chrome driven by a script, and the files the
browser downloaded inspected like every other run's. A first attempt (code `16bbbd3`) ran text
to video with sound through the page (208 s of denoise and decode) and found that a result
written to an output folder outside the working directory and the temp dir could not be
served, and that the page then stayed on "Generating…" for good; both are fixed in deploy 1.
Then two sessions:

| session, code | run | result |
|---|---|---|
| 1, `b27082c` | text to video with sound, its first example | refused after 14 s by the memory check's early reading: `committed 67.3 + need 25.1 = 92.4 GB > 90.0 GB`, read just after the text encoder's release; eight seconds later the gauge read 38.8 GB. The fix is the one above (`ba2d582`) |
| 1, `b27082c` | audio to video, its shipped example "A portrait set to a recording" (480×704, seed 21) | Done in 137 s; the downloaded file byte-identical to the run folder's; the portrait in colour, breaking into a shout; its sound against the example's recording: level +0.1 dB, envelope correlation 0.999 |
| 1, `b27082c` | text to video with `SOLRICKS/LTX-2.5-BTS-Movie-Set` uploaded on the page at 0.8 (480×704, 2 s) | the adapter panel: "Fits LTX-2.5: 1632 of 1632 layers"; the report: "LoRA: BTSMovieSet_LTX25_v1 at 0.8, on 1632 layers"; 84 s; the adapter's film-set look. No page errors in the browser console |
| 2, `8cfe2f2` | text to video with sound, its first example, the results written to a folder outside the app's directory and the temp dir | "memory settled in 0.22 s: 62.2 -> 53.4 GB", then `53.4 + 25.1 = 78.4` OK; Done in 215 s; the file the browser downloaded is byte-identical to the run folder's and identical to the command-line run `F05_t2av` (SSIM 1.000, transcript exact) |
| 2, `8cfe2f2` | audio to video, its shipped example | refused by the memory check after it had waited and read again: `65.5 + 27.5 = 93.0 GB > 90`, while the machine's other load stood at 42–53 GB |
| 2, `8cfe2f2` | text to video with the adapter | passed the memory check (`61.3 + 25.1 = 86.3`); while the 22.8 GB transformer file was read, the machine's committed memory rose to 93.3 GB and the test machine's watchdog stopped the session |

The second session's audio to video and adapter runs were not repeated through the page that
day.

Not run locally, by design: Upscale ×2, the diffusion decoder and bitsandbytes int8 need CUDA.
Their state is in 10.1.

### 10.3 Size and length on the Space (measured on deploy 1, 2026-11-17)

Recorded by the lead in the run log of the deploy-2 work (the entry "ANSWER to the operator's
question", 2026-11-17). Each call's files are kept under its label in that run's evidence
directory, outside this repo, `evidence/space/d2_measure/<label>/`, beside the Space's run logs
and the summary `MEASUREMENTS.md`, where every figure below can be found.

Six calls on deploy 1 (Space `0cd54f2`, int8, `xlarge`), 02:33–02:50 UTC, one at a time: text
to video with sound, the captain prompt, seed 7, the Decoder at Auto (the diffusion decoder).
The window is the Space log's schedule → release; GPU-s is the window × 2, an upper bound.
Tokens N = (W / 32) × (H / 32) × ((F − 1) / 8 + 1), and the Upscale pass runs at 2W × 2H.
Before = schedule → first denoise step; after = last denoise step → release (the decode, the
mux, the write).

| label | asked for | returned | N, pass 1 / 2 | s per step, pass 1 / 2 | before · after | window | GPU-s ≤ | peak GiB, allocated / reserved |
|---|---|---|---|---|---|---|---|---|
| `M1_x2_m05_704x480_121f` | 704×480×121, Upscale ×2 | 1408×960×121 | 5,280 / 21,120 | 1.49 / 5.17 | 6 · 19 s | 52 s | 103.6 | 56.46 / 67.14 |
| `M2_m05_704x480_481f_20s` | 704×480×481 (20 s) | 704×480×481 | 20,130 | 4.83 | 5 · 22 s | 65 s | 130.3 | 50.27 / 59.49 |
| `M3_m05_1408x960_121f` | 1408×960×121 | 1408×960×121 | 21,120 | 5.14 | 8 · 20 s | 69 s | 135.9 | 55.52 / 66.20 |
| `M5_x2_m05_704x480_241f` | 704×480×241, Upscale ×2 | 1408×960×241 | 10,230 / 40,920 | about 2.6 / 11.12 | about 4 · 42 s | 100 s | 200.2 | 59.84 / 78.31 |
| `M4_m05_1280x704_241f` | 1280×704×241 (10 s) | 1280×704×241 | 27,280 | 6.80 | 6 · 28 s | 88 s | 175.9 | 56.37 / 71.98 |
| `M6_x2_m05_960x544_241f` | 960×544×241, Upscale ×2 | 1920×1088×241 | 15,810 / 63,240 | 3.79 / 19.96 | 1 · 62 s | 154 s | 307.9 | 62.75 / 74.33 |

Every output was probed (size, frame count, 48 kHz stereo AAC) and looked at (frames for M2, M4
and M6): coherent pictures at native size over the whole clip. M5's first-pass step time is
approximate because its progress bar was pushed out of the log. Six and seven samples were
clipped in M3 and M4 (the clipping found by §10.1). At most 1053.9 GPU-s for the six, no quota
refusal, no failed run.

**The fit**, least squares through 5,280, 21,120 and 40,920 tokens and within 0.2 s of every
other point measured, deploy 1's included: seconds per denoise step t(N) = 0.46 + 1.81 × 10⁻⁴ N
+ 1.94 × 10⁻⁹ N²; after the last step, about 1.0 s + 1.25 × 10⁻⁷ s per returned pixel-frame
(1.29 × 10⁻⁷ at 241 frames). Checked against M4 (89 s predicted, 88 measured) and M6 (about 162
predicted, 154 measured). The app's reservation is this fit (`zerogpu.estimate_duration`): 25 s
before the first step, eight steps of t(N1), then 1.0 s + 1.3 × 10⁻⁷ s per returned pixel-frame;
Upscale ×2 adds 3 s, three steps of t(N2) and its larger output; each part is reserved at
1.3 × its time, rounded up to 5 s. Every one of these six runs and deploy 1's 24 is reserved at
least 1.3 × its window (`tests/test_reservation.py`).

**The operator's question: the largest size that completes a 10 s clip.** 1920 × 1088 at 10 s
completes, through Upscale ×2 from 960 × 544 (M6), in 154 s of GPU time (at most 307.9 GPU-s,
about 5 minutes of `xlarge` quota). In one pass, 1280 × 704 at 10 s took 88 s (M4); through
Upscale ×2 from 704 × 480, 1408 × 960 at 10 s took 100 s (M5). 1920 × 1088 in one pass was not
among these six calls. On the Space the app now refuses, before any GPU time, a run whose pass would hold more than
64,000 tokens (M6's Upscale pass held 63,240) or whose returned video would pass 520 million
pixel-frames (M6 returned 503,439,360): `constraints.validate_envelope`.
