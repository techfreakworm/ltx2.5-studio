# The mode matrix on the live Space: 9 of 11, and one defect localised to video input

> ## ✅ Nine of eleven modes generate on the live Space, content-verified.
>
> `video_to_video` and `video_to_audio` fail — **both video-INPUT modes, and only
> those.** Every other mode passes, including both audio-input modes.
>
> ✅ **Both failures are now diagnosed, and the cause is fixed offline.** They
> returned `'DecodeError'`; the real exception was recovered from the **Space run
> log**, which `spaces` prints even when it strips it from the client response
> (§4). The cause is a **missing video decoder**, not a decode failure — §3.
> Fixed at `d2eb6f2`.
>
> ⏸️ **Not yet re-run on the Space.** "Fixed" here means the offline gate agrees,
> not that the Space has served these two modes. Only a live run closes that.
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

**9 PASS ✅ · 2 FAIL ⛔**

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
| `video_to_video` | ⛔ | see below | none | — |
| `video_to_audio` | ⛔ | see below | none | — |

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

⏸️ **The Space has not re-run these two modes.** The fix is verified against the
offline gate and against upstream's own dispatch, not against the platform.

The pre-run read, on record from before the matrix:

* `gr.Image(type="pil")` hands the handler a **PIL Image** 📄 — the same object
  `scripts/host_matrix.py` has always passed, on the one path where video
  conditioning is known to work.
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

⚠️ **Read at `spaces` 0.48.3; the Space runs 0.51.1** 📄. The marshalling source
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

⚠️ **Version caveat, stated exactly.** Everything above is read off `spaces`
**0.48.3**, the locally readable copy; the Space runs **0.51.1** 📄. This is
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

---

## 7. Cost

```
matrix to date  <= ~346 GPU-seconds UPPER BOUND
budget          2400 GPU-seconds/day (PRO) 📄  ->  <= 14.4% consumed
ceiling         1920 GPU-seconds (80%)     ->  never approached
```

⏸️ **Exact billed seconds remain unmeasurable** — there is no readable ZeroGPU
quota endpoint, and all three candidate URLs 404 📄. The figure above is
`wall × 2` (the `xlarge` size multiplier 📄) and is a **bound, not a
measurement**. Do not quote it as billed seconds.

⚠️ The bound is loose in the safe direction: wall clock includes queueing,
upload, download and client overhead, none of which is GPU time.

---

## 8. Follow-ups this matrix created

| | item | status |
|---|---|---|
| 1 | ✅ **Cause found and fixed offline.** A filepath `str` reached `LTX2VideoCondition(frames=…)`; `preprocess_conditions` dispatches that field over four types and raises for anything else (§3). Confirmed from the worker traceback, reproduced offline, fixed at **`d2eb6f2`**. ⏸️ **The Space has not re-run these two modes** — that is what remains open. | ⏸️ open: re-run on Space |
| 2 | ⛔ **22 of 27 error classes cannot survive the ZeroGPU fork** (§4), measured. The fix is a **PAIR** — `__reduce__` on `LTXStudioError` **and** deriving it from `gr.Error` — and neither half works alone: `__reduce__` alone is never consulted, and `gr.Error` alone fails to pickle onto `res_queue` and surfaces as `'PicklingError'`, **destroying the class name that survives today**. Design complete, unapplied, `errors.py` unowned and unauthorised. | ⏸️ open, cheap, offline |
| 3 | ✅ **Settled from source: picklability alone would NOT restore the message.** `ExceptionResult` carries only `traceback`, `error_cls` and `gradio_error`, and the exception object travels only when it is a `gr.Error` 📄 (`types.py:28-31`, `wrappers.py:453`). Read at `spaces` 0.48.3 against a Space on 0.51.1; read-off-source, not measured on-Space. **No live run is needed to know this** — one would still confirm it end to end. | ✅ resolved |
| 4 | ⏸️ **Reproducibility is evidenced, not proven** (§5). No two-run frame-hash comparison exists. Cheap to close. | ⏸️ open |
| 5 | ⚠️ **`audio_to_video` and `image_text_to_audio_video` are near-static** (§6). May be expected at this geometry; nothing establishes it either way. | ⏸️ open, observation |
| 6 | ⛔ **`audio.validate_audio_latents_against_video` has ZERO callers.** Found by inspection, not by this matrix. It is exported, documented as *"the check that stops a desynced file being emitted rather than refused"*, and nothing calls it — so supplied `audio_latents` set the audio length while `num_frames` sets the video length, with nothing comparing them. Both audio-input modes passed here **only because their fixture came from a same-geometry run**, which is a property of the runner's chaining and not a defect anyone would notice. | ⏸️ open |
| 7 | ⏸️ **Only one configuration has been exercised** — 256×256×9, stage 2 off, no LoRA, no `auto_duration`. Nothing here says what any mode does at a shippable resolution, and stage 2 has never run on the Space at all. | ⏸️ open |

### ⛔ Item 6 is the THIRD instance of one defect class, and that is the finding

An advertised guard wired to nothing has now been found three times in this
build, by three unrelated routes:

| | the guard | how it was inert | how it was caught |
|---|---|---|---|
| 1 | `LTX25_TEXT_ENCODER_PRECISION` / `LTX25_TRANSFORMER_PRECISION` | not a `RuntimeConfig` field at all — never resolved, validated or logged | mutation test, after the value had already been written into a findings doc as a live lever 📄 |
| 2 | `--quota-ceiling` | per-invocation; five invocations each announced **FITS** knowing nothing of the other four | reading the runner's own output across runs |
| 3 | `audio.validate_audio_latents_against_video` | exported, documented, **zero callers** | inspection, not by this matrix |

**The class is not "a bug that was missed". It is a mechanism that reports
safety it never checked** — which is strictly worse than its absence, because an
inert *lever* merely does nothing while an inert *report* actively tells you that
you are safe. `PLAN.md` states the rule for levers already: *"an advertised lever
attached to nothing is worse than no lever at all."* Three instances say the rule
needs a **test-shaped** counterpart, not another restatement.

⏸️ **The cheap generalisation nobody has run:** every guard in this package that
is exported and documented, checked for callers. #3 was found by eye. Nothing
says it is the last one, and two of these three were invisible to a green gate.

---

## 9. What this document does not say

* It does **not** say the video modes now work on the Space. §3 gives the cause,
  confirmed from the worker traceback, and a fix at `d2eb6f2` that the **offline
  gate** accepts. ⏸️ Neither mode has been re-run on the platform, and this
  document's own standard is that a green local gate is not a Space result.
* It does **not** say the fork-boundary fix has been applied or tried. §4's
  design is complete and read off `spaces` **0.48.3** while the Space runs
  **0.51.1**; nothing in it has been exercised against the platform.
* It does **not** say any figure here is billed GPU-seconds. All of them are
  wall-clock-derived bounds.
* It does **not** say the passing modes are *good*. They generate, and their
  artifacts are non-empty, non-silent and non-frozen. Quality at a real
  resolution is untested.
