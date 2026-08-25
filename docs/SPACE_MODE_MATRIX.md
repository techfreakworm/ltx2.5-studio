# The mode matrix on the live Space: 11 of 11, and the defect that took two of them out

> ## ✅ All eleven modes generate on the live Space, content-verified.
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
measurement**. Do not quote it as billed seconds.

⚠️ The bound is loose in the safe direction: wall clock includes queueing,
upload, download and client overhead, none of which is GPU time.

---

## 8. Follow-ups this matrix created

| | item | status |
|---|---|---|
| 1 | ✅ **CLOSED.** A filepath `str` reached `LTX2VideoCondition(frames=…)`; `preprocess_conditions` dispatches that field over four types and raises for anything else (§3). Confirmed from the worker traceback, reproduced offline, fixed at **`d2eb6f2`**, and **verified on the live Space at sha `287ba9c`** — both modes returned real content-verified artifacts (§2.1). 11 of 11 modes now generate. | ✅ closed |
| 2 | ⛔ **22 of 27 error classes cannot survive the ZeroGPU fork** (§4), measured. The fix is a **PAIR** — `__reduce__` on `LTXStudioError` **and** deriving it from `gr.Error` — and neither half works alone: `__reduce__` alone is never consulted, and `gr.Error` alone fails to pickle onto `res_queue` and surfaces as `'PicklingError'`, **destroying the class name that survives today**. Design complete, unapplied, `errors.py` unowned and unauthorised. **→ LANDED at `6bb211a` (2026-08-25), both halves, all 22 classes round-tripping; see the annotation in §4. ⏸️ Not yet exercised on the platform — the live Space sha `287ba9c` predates it.** | ✅ landed `6bb211a` · ⏸️ on-Space unexercised |
| 3 | ✅ **Settled from source: picklability alone would NOT restore the message.** `ExceptionResult` carries only `traceback`, `error_cls` and `gradio_error`, and the exception object travels only when it is a `gr.Error` 📄 (`types.py:28-31`, `wrappers.py:453`). Read at `spaces` 0.48.3 against a Space on 0.51.1; read-off-source, not measured on-Space. **No live run is needed to know this** — one would still confirm it end to end. | ✅ resolved |
| 4 | ⏸️ **Reproducibility is evidenced, not proven** (§5). No two-run frame-hash comparison exists. Cheap to close. **→ Sharpened (2026-08-25): §5's "byte-identical" headline itself overstated the evidence — the cross-day match was size + decoded statistics, no day-1 hash was ever recorded, and the day-1 file is gone. The close is now a same-seed re-run sha256-compared against the surviving 2026-08-22 artifact (`b78803cc…`), planned 2026-08-25 — it settles byte-level reproducibility across the sampled containers.** | ⏸️ open, re-run planned |
| 5 | ⚠️ **`audio_to_video` and `image_text_to_audio_video` are near-static** (§6). May be expected at this geometry; nothing establishes it either way. | ⏸️ open, observation |
| 6 | ⛔ **`audio.validate_audio_latents_against_video` has ZERO callers.** Found by inspection, not by this matrix. It is exported, documented as *"the check that stops a desynced file being emitted rather than refused"*, and nothing calls it — so supplied `audio_latents` set the audio length while `num_frames` sets the video length, with nothing comparing them. Both audio-input modes passed here **only because their fixture came from a same-geometry run**, which is a property of the runner's chaining and not a defect anyone would notice. **→ WIRED at `59cfa6e` (2026-08-25): `generate._prepared_audio`'s supplied-latents branch now calls it, a mismatch raises with both lengths named, `auto_duration` skips, and could-not-check shapes surface as warnings.** | ✅ wired `59cfa6e` |
| 7 | ⏸️ **Only one configuration has been exercised** — 256×256×9, stage 2 off, no LoRA, no `auto_duration`. Nothing here says what any mode does at a shippable resolution, and stage 2 has never run on the Space at all. | ⏸️ open |
| 8 | ⏸️ **No mode has been shown to FOLLOW its conditioning input.** Every check in this document is a property of the output alone — motion, level, distinctness. `video_to_video` returning mean \|Δ\| 17.277 from an 18.894 input is equally consistent with faithful conditioning and with ignoring the clip entirely (§2.1). Settling it needs a comparison the runner does not currently make: same seed, two different conditioning clips, and a difference in the output. | ⏸️ open |
| 9 | ⚠️ **`video_to_audio` is 8–12 dB quieter than every other audio mode** (§6.1) — −19.5 dBFS against −7.6 to −11.1. Passes the level checks, is nowhere near the −40 dBFS flag, and is the only audio mode conditioned on video. Recorded because H9 was a level anomaly that passed every structural check. | ⏸️ open, observation |
| 10 | ⚠️ **The quota ledger's history begins mid-day** (§7). It reports 67.252 GPU-s for a day whose real bound is ~413, because it was installed after the matrix ran. The figure is accurate for what it observed and misleading if quoted as a daily total; nothing in its output says where its history starts. | ⏸️ open |

### ⛔ Item 6 is the THIRD instance of one defect class, and that is the finding

An advertised guard wired to nothing has now been found three times in this
build, by three unrelated routes:

| | the guard | how it was inert | how it was caught |
|---|---|---|---|
| 1 | `LTX25_TEXT_ENCODER_PRECISION` / `LTX25_TRANSFORMER_PRECISION` | not a `RuntimeConfig` field at all — never resolved, validated or logged | mutation test, after the value had already been written into a findings doc as a live lever 📄 |
| 2 | `--quota-ceiling` | per-invocation; five invocations each announced **FITS** knowing nothing of the other four | reading the runner's own output across runs |
| 3 | `audio.validate_audio_latents_against_video` | exported, documented, **zero callers** | inspection, not by this matrix |

> **Status of the three, read back 2026-08-25.** Instance 3 is **closed** —
> wired at `59cfa6e` (follow-up #6). Instance 2 was closed at `62d612f` (the
> cross-invocation ledger, §7). Instance 1's cleanup is **half-done on the
> platform**: the live Space variables still carry `LTX25_TRANSFORMER_PRECISION=bf16`
> — inert as ever, and now advertising bf16 on a Space whose transformer loads
> in int8. Its deletion is being requested with the next deploy. See
> `SPACE_GENERATION_FINDING.md` §2's annotation; the row above stands as the
> record of how each was found.

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

* It **does** say all eleven modes generate on the Space — and the evidence is
  exactly this: each produced a non-empty artifact whose **decoded content** was
  checked, not merely its container. For the two re-run modes that is 33,812 B at
  9 distinct frames and 63,438 B at −19.5 dBFS, at sha `287ba9c` (§2.1). That is
  the whole claim.
* It does **not** say any mode **follows** its conditioning input. Every check
  here reads the output alone: motion, level, distinctness. A mode that ignored
  its conditioning clip entirely and generated from the prompt would pass every
  test in this document. §2.1, follow-up #8.
* It does **not** say the modes are *correct*, only that they run and return
  something real. "Generates" is a much weaker claim than "generates what was
  asked for", and only the first is established.
* It does **not** say the fork-boundary fix has been applied or tried. §4's
  design is complete and read off `spaces` **0.48.3** while the Space runs
  **0.51.1**; nothing in it has been exercised against the platform.
  — *Annotation (2026-08-25): the first half is stale — the pair **is** applied,
  at `6bb211a`, offline gate green. The second half stands unchanged: the live
  Space sha `287ba9c` predates the fix, so nothing has been exercised against
  the platform.*
* It does **not** say any figure here is billed GPU-seconds. All of them are
  wall-clock-derived bounds.
* It does **not** say the passing modes are *good*. They generate, and their
  artifacts are non-empty, non-silent and non-frozen. Quality at a real
  resolution is untested.
