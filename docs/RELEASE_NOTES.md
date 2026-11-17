# Release notes

What each deploy of LTX-2.5 Studio to its Hugging Face Space contains, what was verified on it,
and what was known to be open when it shipped. Newest first. ZeroGPU's billed seconds cannot be
read back, so every GPU figure here is an upper bound, the run's wall clock × 2, never billed
seconds.

## Deploy 2 — 2026-11-17 (in preparation)

**Not deployed yet.** The code is the branch `fix/studio-2026-11`, at `7cef5fb` with both UI
packages of round 4, plus these docs. Every commit named below is on it. What the Space shows
after the deploy goes under "On the Space", at the end of this section.

### Landed after deploy 1's push, before the deploy-2 work (`1e9fd20..a40a311`)

- **#2** A run never inherits another run's adapter: before anything is attached, every run
  removes any adapter left on the model, and a run whose leftover cannot be removed is refused
  as an adapter failure and never sampled. `d9da392`
- **#3, #4** On short desktop windows the cost figure stays beside Generate, what a click changes
  comes into view, and Generate never leaves the rail. `3a32028`
- **#7** An adapter whose weight pairs are named `lora_down`/`lora_up` is refused at Validate,
  naming both namings, instead of passing every check and failing inside the run. `daa617a`,
  `7214030`
- **#8** Validate's shape check is never skipped silently: an adapter validated without a shape
  table or a transformer has its report say so, and the run checks it against the live
  transformer before attaching it. `d2ed8cb`
- **#9** A condition passed to `conditioning.build_conditions` as a mapping keeps its kind, so a
  direct caller's video is built as a video condition. `88b163b`
- **#13** The Space harness judges what a mode promises over the files a call returns, so a
  video-with-sound call with its `.wav` beside the mp4 passes (deploy 1's false MISMATCH).
  `a40a311`
- **#20** The last UI round's follow-ons and polish: a clip matches its picture, the result plays
  by itself, an unshown run is listed with Show as its step, Longest stops at 20 s above Shortest;
  no Gradio "Runs" link on narrow screens, locked decoder chips drawn as locked, sizes and seeds
  that never break. `f0ff7e3`, `58847c6`

### The deploy-2 work

- **#10** A bad MPS watermark pair stops an MPS start in the app's own words before the numerics
  self-test runs, so it is never reported as a PyTorch that computes wrongly. `e664ad9`
- **#11** `scripts/local_generate.py` applies the app's MPS numerics gate and exits refused before
  loading anything on a PyTorch that fails it. `444bc2b`
- **#12** The progress bar names "loading the model" while a staged local run loads, and
  "decoding the frames" after the last step of a pass that decodes inside the pipeline call (the
  standard decoder). `aeab028`
- **#14** Sound the model makes over full scale is turned down as a whole to a −0.1 dBFS peak
  instead of being clipped, the mp4 and the `.wav` carry the same samples, and the diagnostics
  say by how many dB. `ed91409`
- **#15** bitsandbytes' int8 cast warning reaches the log once per process, not once per matrix
  product. `18bcc7a`
- **#16** When the model's own length lies outside the range allowed, the note says what it chose
  and what the range made of it. `9b280fb`
- **#17** The diffusion decoder is built once at start-up, beside the pipeline, so ZeroGPU packs
  it and every run uses that instance; a start-up build that fails logs one line and each run
  builds its own. `ed13532`, `d229cdc`, `5e06a43`
- **#18** An API caller's diagnostics name each attached adapter and give the run's time split
  into the progress bar's phases. `0f7e71d`, `17782c8`
- **#21** The reservation follows each run's size and length, fitted to the Space's measured runs
  with a 1.3 margin: the default run reserves 60 s and needs 120 s of quota free to start (it was
  180 s and 360 s); on the Space a run past the measured envelope (64,000 tokens in a pass, 520
  million returned pixel-frames) is refused before any GPU time; `LTX25_GPU_DURATION_SECONDS` is
  ignored and `LTX25_GPU_DURATION_FLOOR_SECONDS` sets a minimum. `ac68de5`, `1b0022d`
- **#23** The weights come from the image: `preload_from_hub` fetches the 35 model files the
  Space loads (71.92 GB; the temporal upsampler is no longer fetched) and the prompt enhancer's 8
  (2.49 GB), and start-up downloads only what is missing, refusing a download the disk cannot
  hold. `41dcf05`, `ff62c55`, `72b989d`
- **#20** UI round 4, its first package: the monitor's player, and every video player on the page,
  works by keyboard and touch; the jump to a mode's sample also shows on desktop while the monitor
  holds a run of another mode; the cost travels with a stuck Generate; after a run the report's
  head and the monitor fit in the rail together; the idle rail no longer scrolls over nothing;
  Generate in flight is drawn busy; 44 px targets on a phone. `93a7f52`, `f8393c1`, `3b92225`,
  `5a6543d`, `0cf63ed`, `81032ed`, `700b8bc`, `07beaa9`, `9e59de0`, `fe49110`
- **#20** UI round 4, its second package: after a run every mode's sample can still be seen, the
  button under the purpose line reading "See an example of" the selected mode and then "Back to
  your run"; the line beside Generate names what the run returns; the Done card says "chosen by
  the model" only strictly inside the range allowed, and no "legal" is left on the page; a run
  of under half a second took "under 1 s"; a LoRA rejection gives one next step for its kind;
  Restore settings says what it did; "Use this prompt" turns Enhance off and says so; arrow keys on
  a LoRA strength reach the run; off CUDA a parked Upscale promises no size; a cost lamp's state is
  not in colour alone; a failure card no longer outlives Show, a Restore or its own remedy; the
  sample captions name their example and their real lengths and promise no repeat. `78036d2`,
  `05737f6`, `a8bf7a4`, `74d0925`, `1583c93`, `6150e0e`, `8183336`, `d468ab6`, `44c216d`,
  `44edefb`, `7896071`, `9ab0812`, `b46dfc2`, with `e6f95d8`, `e48ebc3`, `67478d3` (UI_DESIGN),
  `f16d35d` (layout only) and `e4fc218` (a test)
- **#5** A request Gradio refuses before any handler runs no longer leaves the page on
  "Generating…": typed number fields declare no range, and a last failure listener brings
  Generate back with a card. `bfd58d1`
- **#6** A video with sound offers its sound track as "Download the sound (WAV)" in the Done
  card's action row. `99f710e`
- **#27** The decoder choices are named by what they do: "Auto", "Steadier (diffusion decoder)"
  and "Sharper (standard decoder)", in the choice, its line, the Upscale lock line, the Done card
  and the backend's refusals; the values sent stay `auto`, `diffusion` and `conv`. `1c8a58d`
- **#28** Mode 11's first-frame line: "With a speech recording and no first-frame picture, some
  seeds come out as a black-and-white period film: add “in colour” to the prompt, or give a
  first-frame picture." `9dcc63c`, `9887158`
- **#21** UI_DESIGN gives the figure line and the Upscale line as the reservation now sizes
  them. `7cef5fb`

### The docs

README, `docs/SPACE_MODE_MATRIX.md`, `docs/UI_DESIGN.md`, `docs/PLAN.md`,
`docs/OPERATOR_HANDOFF.md`, `docs/ZEROGPU_DISK.md` and these notes, by issue: **#25** deploy 1's
Space record (§10.1) and this file `0646770`, `c8c5965`; **#29** `da5f47d`, `0eb8e93`, `70ddbf2`;
**#24** `7dbfae6`; **#28** `d527b6d`, `8d0b074`, `1cda206`; **#27** `360edcc`; **#21** the
reservation and §10.3 `ca1f2a8`; **#23** `38e36e9`; **#17** `d8bbbdb`; **#20** `4d60faf`,
`e7caf99`; **#14** `3498ce9`; **#15** `a0b76da`; **#16** `c511137`, `58732a9`; **#18** `f9bde99`;
**#10** `ed8912d`; **#11** `71e35bc`; **#12** `974496c`; **#32** this section, `85c4433`,
`8f43118` and the commit that names these hashes.

### Not in deploy 2

- **#22** Audio to video does not lip-sync: deferred, left open.
- **#24** Gradio 6.24.0's un-awaited `get_current_user` warning: documented (README, Known
  issues); Gradio is not bumped.

### On the Space

*Not deployed yet.*

## Deploy 1 — 2026-11-16

**What it is.** Commit `1e9fd20` of the branch `fix/studio-2026-11`, pushed to GitHub on
2026-11-16. The Space, `techfreakworm/LTX2.5-Studio`, runs it as commit `0cd54f2`, whose tree is
that of `1e9fd20`; `main` points at `1e9fd20` too. The Space was private while deploy 1 was
verified on it, and is public since.

**This section is PR #1's corrected text.** PR #1 (`soma/ltx25-studio` at `445abcb`, the code
the Space ran from 2026-08-26 until deploy 1) is merged, and its body and its comment of
2026-08-25 still carry claims that were withdrawn later. They stay as they are on GitHub; this
section corrects them and brings them up to deploy 1. A withdrawn claim is struck through and
followed by its reason.

### Read this first

- Until `eb1aa9f` (2026-08-25, deployed as Space `e3b5333`) every Space output was colour
  static: the int8 build quantised the transformer's conditioning path. After the fix one
  text-to-video frame was verified by eye. The eleven-mode re-run of that evening was never
  written up and its files are lost.
- At `445abcb` the offline suite passes only in a venv without torch (2570 passed, 20 skipped);
  with `requirements.txt` installed it is 34 failed and 20 errors (measured 2026-11-16).
- GPU spend in August: the persistent quota ledger for UTC 2026-08-25 reads ≤ 1528.672 GPU-s
  over 20 runner calls; with the ≤ ~93.3 GPU-s of manual calls bounded by hand, that day's
  bound is ~1622 GPU-s. Nothing was ledgered after 2026-08-25 14:09 UTC until deploy 1.
- The fixes of 2026-11-16 are in deploy 1. With them all eleven modes are verified on real
  weights on an Apple-silicon Mac (Q8_0 GGUF transformer, single stage, conv-VAE decode), each
  output opened and checked by eye and ear, with measurements
  ([`SPACE_MODE_MATRIX.md`](SPACE_MODE_MATRIX.md) §10.2):
  - video to video is a variation of the clip at three levels (SSIM to the clip 0.78 / 0.65 /
    0.36–0.40), and the prompt acts at the highest; the length defect found that day in
    Matroska/WebM clips is fixed and verified;
  - the audio-encoding defect found that day is fixed (every upload had reached the model about
    9 dB too quiet and duller, from the app's own mel transform): audio to audio returns sound
    at the input's level, graded by the chosen level (on speech, the same events at the same
    times 0.97 / 0.95 / 0.71, the words exact at the lowest level), and audio to video's
    picture, black and white whenever the prompt named no colours, came out in colour with the
    same prompt and seed;
  - video to audio follows what is visible: from a silent clip of a man shouting, a voice in
    time with his mouth (loudness envelope correlation 0.92 with the clip's original track);
  - audio to video carries the user's recording exactly and takes its length from it, but does
    not lip-sync: the mouth does not follow the words (mouth opening against speech level
    between −0.38 and +0.07, against +0.84 for speech the model writes itself in image + text to
    video with sound); asking for lip sync and quoting the words in the prompt did not change
    that in any of the cases tried, and the page and the README say so;
  - the prompt enhancer works on the pinned transformers 5.23.0, on both its routes;
  - the local runner and the Space runner are one stage graph;
  - a final pass repeated every mode, an adapter, the enhancer and the model-chosen length on
    `e59944e`, where the backend was declared finished (three of those runs on `ba2d582`, after
    the memory-check fix below): every output compared with a run made after the RoPE fix was
    identical to it (SSIM 1.000, sound correlation 1.000), and the four compared with runs from
    one commit before that fix show the same scenes framed slightly differently, with the same
    words and sound. Lip sync on these outputs: +0.845 where the model writes the speech itself,
    −0.379 in audio to video;
  - the page ships an example and a sample output for every mode (`examples/`, media through
    Git LFS), each sample a re-encode of that final pass's output. Audio to video's showcase does
    not show a speaking face: its sample pictures an ocean recording as waves at sunset, its
    examples are that recording with a beach prompt and the captain's portrait over the sea with
    his lips closed. A test refuses an audio-to-video example that asks for speech.
- The offline gate of `1e9fd20`, from a clean export: 4735 passed / 38 skipped with the full
  requirements, 4516 passed / 233 skipped without torch, ruff clean; all 145 commits by the
  operator, no binary file changes in the history; the examples tested as real media and as
  Git LFS pointers.

### What the Space showed after the fixes

Deploy 1 ran on the Space on 2026-11-16, 16:02–16:27 UTC: 24 generations covering all eleven
modes, Upscale ×2, both decoders, an adapter from the Hub, prompt enhancement, the model-chosen
length and the operator's own recording, each output looked at and measured. At most 1241.3
GPU-s; no quota refusal, no queue timeout, no failed run. Call by call:
[`SPACE_MODE_MATRIX.md`](SPACE_MODE_MATRIX.md) §10.1.

- The pinned packages installed exactly, and the transformer and the text encoder load as int8
  on transformers 5.23.0 and bitsandbytes 0.50.2. The start-up report says int8 (it used to say
  bf16), names `spaces` 0.51.3 (it used to say None) and says the packed copy is not measured
  yet (it used to print 0.0). The cold start took 17 min 24 s, 14 min 55 s of it the 72.19 GB
  download.
- On the Space's GPU (CUDA) the default decode is the model's diffusion decoder, and every run
  that returns video says which decoder ran. The same run through both decoders gives the same
  scene: the diffusion decoder a little softer and steadier from frame to frame, the standard
  decoder sharper frame by frame, with more shimmer.
- Upscale ×2 works: 704×480 with 49 frames came back at 1408×960, in text to video with sound
  with added detail and not washed out, and in audio to video, a mode it used to fail in, as a
  coherent picture in colour.
- Every run was granted the whole `xlarge` GPU (94.971 GiB); the largest peak was 63.301 GiB
  reserved. Progress reaches the page in the app's own words on `spaces` 0.51.3.
- The prompt enhancer loads and rewrites on the GPU under transformers 5.23.0.
- The model picked the length at 512×960 with 5–10 s allowed: 5.0 s.
- An adapter from the Hub, `SOLRICKS/LTX-2.5-BTS-Movie-Set`, passed Validate on the Space
  ("Fits LTX-2.5: 1632 of 1632 layers") and changed the picture: with the same prompt and seed
  and no adapter the picture is a different one.
- The image modes start on the supplied image (the first frame against the picture: SSIM
  0.741–0.872), and audio to video's sound track is the recording itself.
- Audio to video without a picture can come out black and white: the same recording and prompt
  gave a period-film picture at one seed and colour at another.
- GPU time: every run's window was 10.5–51.6 s against reservations of 180–360 s. A plain run at
  704×480 with 121 frames reserved 180 s, so it needed 360 s of quota free to start, and drew
  at most 34–65 GPU-s.

### Through the page, on real weights, on a Mac

Two sessions ran in a real browser, with the app started as a user starts it. In the first,
audio to video and text to video with an adapter uploaded on the page finished; text to video
with sound was refused by the app's own memory check. Five defects were found that evening,
and deploy 1 has a fix for each:

- with the output folder outside the working directory and the temp dir, Gradio refused to
  serve the finished result; that folder is now passed to launch as the one allowed path;
- the page then stayed on "Generating…" for good; such a run now ends with Generate back and a
  card saying the run finished but its result could not be shown;
- the local memory check read the gauge before freed memory had left it (the kernel returns the
  text encoder's 24 GB over 0.1–0.35 s; the readings were 3.5–23 GB too high) and refused runs
  that fitted. It now judges the gauge once it has stopped falling and reads a failing check
  again before refusing, and a refusal gets its own card, "This machine does not have enough
  free memory for this run", not an app error. On real weights the three runs of the final pass
  it had refused ran, their settled readings 3–13 GB below the first ones;
- "Match my clip" with a recording shorter than nine frames ran a one-frame video; it is now
  refused before anything runs, in the words a set length gets;
- on a local machine the page spoke of reserving GPU time, quota and billing; off ZeroGPU the
  cost panel now says where the run happens and that nothing is metered, and no caption, card
  or reason speaks of quota, reserving, billing or paying. On the Space nothing changes.

A second session ran on a build with the memory-check fix. Text to video with sound, refused in
the first session, finished; its results were written to a folder outside the app's directory
and the temp dir, and the file the browser downloaded from there is byte-identical to the run
folder's and identical to the command-line run's (SSIM 1.000, transcript exact). That session's
audio to video was refused by the memory check (`65.5 + 27.5 = 93.0 GB` against the 90 GB
limit, with other heavy jobs on the machine), and its third run was stopped from outside the
app, by the test machine's memory watchdog, while the transformer loaded. Those two have not
been run through the page on the fixed build.

### The UI at deploy 1

The last design review (round 3) returned REVISE, 7.5/10 (round 1: 4, round 2: 6). The design is
final and has no blocker; four majors stood between it and acceptance. Two are fixed in deploy
1: audio to video's showcase was speaking faces, the one thing that mode cannot do (replaced,
above), and a failure card outlived its run. Two are not in deploy 1: on short laptop windows
the cost figures slide under Generate, and on desktop a failure card or an earlier run can land
out of view.

### Known open at deploy 1

None of these stops a mode from working.

- After an adapter fails to come off the model, the next run without an adapter is not checked
  for it and can be sampled through it.
- An adapter whose weight pairs are named `lora_down`/`lora_up` passes Validate and then fails
  inside the run.
- With the local snapshot override, Validate's shape check is skipped without saying so.
- A condition passed straight to `conditioning.build_conditions` as a mapping with
  `"kind": "video"` is taken for an image (direct callers only; `generate()` and the page are
  not affected).
- Two guards for local runs on a Mac are unfinished: the app does not check the MPS watermark
  pair before its start-up numerics self-test, and `scripts/local_generate.py` does not run that
  self-test.
- The progress bar says "sampling" while a local run loads the model, and while the standard
  decoder decodes (on the Space too).
- The page can stay on "Generating…" when Gradio itself refuses an input before any handler
  runs, and the sound of a video-with-sound run is not offered as a download on the page (the
  endpoint returns it).
- The findings of the Space run: the sound of a short shouted line reaches full scale and is
  clipped when written; the Space's log is flooded by one bitsandbytes warning per int8 matmul;
  when the model's own length is outside the range and moved to its edge, the note says only
  that the model chose the length; the diffusion decoder is loaded inside the runs instead of
  once at start-up; the endpoint returns neither which adapter was attached nor the time the run
  took.
- Gradio 6.24.0 logs "coroutine 'App.create_app.<locals>.get_current_user' was never awaited"
  on page loads; it is Gradio's own.

### What this is

LTX-2.5 Studio — a Gradio app wrapping Lightricks LTX-2.5, covering **all eleven modes** plus
user-uploadable LoRA applied at inference (Hub repo-id or file upload).

### Host-proven — eleven of eleven modes, on a host that no longer exists

Verified on an unpartitioned RTX PRO 6000 Blackwell (97887 MiB, 124 GB RAM), through
`get_pipeline` **as `app.py` calls it** — no bypass, no harness accommodation. **bf16,
unquantised, on hinode, which has since been retired: not the Space configuration (int8).**

| mode | artifact | notes |
|---|---|---|
| `text_to_video` | 791,782 B | 704×480×121, h264 @24 |
| `image_to_video` | 307,583 B | |
| `image_text_to_video` | 775,213 B | |
| `video_to_video` | 708,885 B | strength 0.7 |
| `text_to_audio_video` | 792,803 B | + aac 2ch 48 kHz |
| `image_to_audio_video` | 374,184 B | + aac 2ch |
| `image_text_to_audio_video` | 995,843 B | + aac 2ch |
| `text_to_audio` | 961,998 B | pcm_s16le 2ch, no video |
| `video_to_audio` | 961,998 B | pcm_s16le 2ch, no video |
| `audio_to_audio` | 961,998 B | −10.4 dBFS |
| `audio_to_video` | 193,825 B | video only |

These runs were unseeded (the seed reached the model only from `4ebb8b9`), so the sizes are
not reproducible references.

`verify_streams` exit 0 on every one. Also shown on that host: **stage 2** (704×480 in →
**1408×960** out, a size only the spatial upsampler produces), **duration head** (97 frames
chosen, not the 121 baseline), **flagship 1216×704** with stereo in 21.0 s, **LoRA both
directions** as validator verdicts and an attach count (accepted with 576 modules; rejected at
stage 6.5 in both key conventions) — the adapter's *effect* was not established there — and the
**mel round-trip** recovering 4/4 tones under a strict criterion.

~~**Operator-verified by eye and ear** for the video modes and five of six audio artifacts.~~
*Withdrawn: the operator's only sign-off ("visuals are good, audios are good") was on artifacts
served from his own Mac (local MPS, bf16, made with tooling outside this repo) before hinode was
proposed, and before the distilled-recipe fix. Nobody has signed off the artifacts in this
table.*

### On the Space before the fixes (ZeroGPU)

~~**The Space has evicted twice and has never run.**~~ *Stale since 2026-08-20: the Space
starts, and all eleven modes execute. Until `eb1aa9f` (Space `e3b5333`) every output was colour
static; after the fix one text-to-video frame was verified by eye. The eleven-mode matrix after
the 2026-11-16 fixes is written: [`SPACE_MODE_MATRIX.md`](SPACE_MODE_MATRIX.md) §10.1.*

~~hinode cannot answer: … the ~68 GB RAM ceiling … or `size=` promotion.~~ *Both answered: the
RAM ceiling is above 93.56 GB (measured; the inherited ~68 GB is falsified), and `auto` never
promotes — the Space ran on the 47.375 GiB `large` partition until `LTX25_ZEROGPU_SIZE=xlarge`
was set as a Space variable on 2026-08-26 (94.971 GiB measured since). The app now asks for
`xlarge` itself: `LTX25_ZEROGPU_SIZE` defaults to it.*

~~**int8 is no longer parked** — it is built (`components_to_quantize=["transformer"]` via
`PipelineQuantizationConfig`, bitsandbytes 8-bit) and a deploy is in flight.~~ *int8 is deployed:
`PipelineQuantizationConfig(quant_mapping=…)` on the transformer, with its 316 conditioning
Linears kept in bf16, and on the text encoder.*

~~The derivation puts it at a 133.4–137.4 GB peak against the 150 GB cap~~ *That was the
transformer-only bound. For the deployed config the derivation gives 121.4–125.4 GB; the app has
never measured it, because its disk reading runs before ZeroGPU packs. Deploy 1's start-up
report says so instead of printing 0.0.* The bf16 bound of 152.4–156.4 GB, and the fact that the
derivation retrodicts both evictions, stand.

~~**RAM is the unanswered question that decides it**~~ *Answered: int8 loads peak at
84.67–87.13 GiB RSS and are not killed (deploy 1: 85.67 GiB); the ceiling is above 93.56 GB, its
true value unknown.*

### Corrections to the PR comment of 2026-08-25, item by item

That comment was posted about five hours before the same day's finding that every Space output
up to `ff142d9` was colour static: the int8 build had quantised the transformer's conditioning
path (fixed at `eb1aa9f`, deployed as Space `e3b5333`). Its mechanical results stand; its output
claims do not.

**Header.** ~~Branch `soma/ltx25-studio` at `76c408d` … gate **2344 passed / 13 skipped**. Space …
RUNNING at `ff142d9`.~~ PR #1's branch is at `445abcb`, which the Space ran (as `f809472`) from
2026-08-26 until deploy 1. That gate ran in a venv without torch. At `445abcb`, with
`requirements.txt` installed, the suite is 34 failed and 20 errors (without torch: 2570 passed,
20 skipped; both measured 2026-11-16).

**~~Proven on the metered side~~ → executed on the metered side** (every output then was colour
static):

- **Fork boundary carries its cargo both directions** — stands.
- **Byte reproducibility, `b78803cc` across three containers and three code shas** — stands as a
  result about the sampler and the seed; the artifact itself was colour static, so determinism
  holds and content did not.
- **1216×704×121 in 66.5 s through tiled VAE decode** — the tiled decode works (mechanical); the
  content was colour static, and the run was on the 47.375 GiB `large` partition.
- ~~**Stage 2 … PASS at 704×480×25 … Ceiling at 49 frames** (diffusion decoder, no tiling
  API…)~~ **Withdrawn.** The 49-frame failure was the partition: the Space ran on `large` until
  `LTX25_ZEROGPU_SIZE=xlarge` was set on 2026-08-26, and re-run on `xlarge` 704×480×49 passes
  (53.08 s, 55.906 GiB reserved). On deploy 1, stage 2 at 704×480×49 returned 1408×960 in
  two modes, in text to video with sound with added detail and not washed out; the ceiling on
  `xlarge` is not measured.
- ~~**LoRA … inertness check passed** (adapter `20a3a84b` ≠ base `b78803cc`)~~ **Withdrawn
  (output half):** it compared two broken renders. Injection, detach, the size-bounded fetch and
  route equivalence stand. On deploy 1 an adapter visibly changed the picture against the same
  prompt and seed without it.
- ~~**Conditioning demonstrably drives output** — motion mirrors the conditioning clip~~
  **Withdrawn:** both clips and both outputs were colour static. The conditioning path carries a
  value across the fork (mechanical). On deploy 1 the image modes start on the supplied image
  (first-frame SSIM 0.741–0.872).

**GPU spend.** ~~**≤675.4 of 2400 s (28%)**; the 1920 s stop was never approached.~~ The
persistent quota ledger for UTC 2026-08-25 reads **≤ 1528.672 GPU-s over 20 runner calls**
(wall × 2 upper bounds, not billed seconds). Adding the ≤ ~93.3 GPU-s of manual calls bounded
by hand in [`SPACE_SESSION_2026-08-25.md`](SPACE_SESSION_2026-08-25.md) §8 puts the day's bound
at **~1622 GPU-s, about 84 % of the 1920 stop line**. The evening's manual calls, and all of
2026-08-26/27, were never recorded.

**Open follow-ups, as they stand at deploy 1:**

1. ~~Stage-2 frame ceiling … honest-bounded at ≤25 frames @ 704×480 input in README + UI.~~
   The premise fell (above), and there was never a stage-2 frame bound in the UI; the README's
   25/49-frame ceiling is corrected.
2. Slim-projection dispatch payload — still open; not implemented.
3. ~~`video_to_audio` at −19.5 dBFS — consistent with the conditioning-mirrors-input
   finding~~ **Withdrawn:** −19.5 dBFS was audio from the broken transformer. Re-measured on
   deploy 1: −44.2 and −35.9 dBFS on the tram clip at two seeds.
4. `auto_duration` ran on the platform on deploy 1 (the model picked the length at 512×960), and
   so did the two checks that were open there again: an adapter's effect and conditioning
   fidelity (above). The August audio-input check, two levels of change on one recording, has
   not been run on the Space; deploy 1 ran audio to audio at one level, Moderately, and it
   returned sound at the input's level (+1.4 dB).
5. Ledger — deploy 1's 24 calls are in the run's quota ledger: at most 1241.3 GPU-s for UTC
   2026-11-16.
6. First container's SIGKILL — closed-unexplained by operator ruling; stands.
7. ~~`space/main` is one docs-only commit behind origin~~ Stale: the Space runs `0cd54f2`, the
   tree of `1e9fd20`.

### Reproducibility — resolved, and verified

`_resolve_seed` originally reached sidecar metadata only; there was no generator anywhere in
the package, so `seed` recorded rather than determined. A generator is now threaded to the
pipeline and to `audio_vae.encode`.

**Verified byte-identical.** Two runs at an identical seed, mode, fixture and `noise_scale`,
back to back:

```
run-a1d35aaaab06/output.wav   md5 2988f01159a5322afb28abb6e9a73d26
run-b9dde07ed8bb/output.wav   md5 2988f01159a5322afb28abb6e9a73d26
```

~~Whether it generalises to the video modes is untested.~~ *Video-path determinism was later
sha256-verified on the Space (`b78803cc…` across three containers and three code shas, plus the
condition and LoRA paths). That artifact was pre-fix colour static: determinism holds, content
did not.*

The before/after comparison table in [`OPERATOR_HANDOFF.md`](OPERATOR_HANDOFF.md) §2.5 was
measured *before* seeding worked, so its pixel differences cannot attribute change to the
guidance fix. ~~A seeded re-run is in flight and will be pushed to this PR.~~ *Withdrawn: the
seeded re-run was dropped with the host, so that comparison stays permanently unattributable.*
The audio evidence carries that fix on its own: −55.0 → −17.8 dBFS and peaks 1.7811 → 1.0050
(hinode, bf16).

### The audio-input schedule (superseded in deploy 1)

Upstream infers `noise_scale = sigmas[0]` when custom sigmas are given; the audio-input path
here disagreed. Applying the alignment naively cut the eight-step distilled schedule to two
steps at the default `noise_scale` 0.5 and to one at 0.1, so it was reverted and pinned by a
test. *On 2026-11-16 the supplied audio got its own noise level and sigma schedule, and the
video is sampled on the full schedule; measured on real weights locally, the washed-out
audio-to-video picture is gone.*

### Known limitation

`audio_vae.encode` is never called by the pipeline, so our encode is the only producer of
those latents and **nothing validates their distribution**. Unusual input — a test tone, a
sine sweep — yields a structurally perfect, inaudible file with no error. Not reliably
detectable at that boundary; recorded rather than papered over.

### The lesson worth carrying

Our reachability guard climbs a ladder: *imported → reached by the working path → this
function ran → **the arguments were the vendor's***.

**We can prove a function ran. We cannot prove it ran with the right arguments.**
`DISTILLED_SIGMA_VALUES` was defined, documented as mandatory, quoted in a module docstring —
and reached the pipeline only on the stage-2 branch. Eleven of twelve runs used a generic
linear schedule and SFT guidance, costing **3× the compute as well as the quality**. The
constant was referenced, so nothing flagged it. The only instrument that works on that rung is
recording the vendor's spec independently and diffing against it.

#### And a second, about instruments

A startup report printed **`disk: 1969.8 GB free of 8589.9 GB`** while the workload was being
evicted for exceeding 150 GB. It did that twice, silently, through the two failures it existed
to warn about. `shutil.disk_usage()` was reading the **host** filesystem.

**No better path existed.** The cap is not a filesystem limit — the platform enforces it by
watching the workload, which is why exceeding it arrives as an eviction rather than `ENOSPC`.
So the fix was never a better location to point `statvfs` at; it was **measuring a different
quantity**: our own footprint against a cap we already knew. *Caveat (2026-11-16): its "after
packing" sample ran before ZeroGPU packs, so the packed term read 0.0 and the packed copy was
never measured by it — the same inert-report shape. Deploy 1's report says "packed copy: not
measured yet, ZeroGPU writes it at launch after this report".*

> **An inert lever does nothing. An inert *report* actively misleads, because a reader believes
> it.** This one survived two evictions unexamined precisely because it looked like it was
> working.

And a coda: the first regression guard against the old call was a substring scan, which failed
immediately — the replacement function's own docstring *names* `shutil.disk_usage` to explain
what was removed. **A substring scan cannot tell an explanation from a call.** It is AST now.

Full detail in [`OPERATOR_HANDOFF.md`](OPERATOR_HANDOFF.md), [`PLAN.md`](PLAN.md) §9 and
[`TEST_PLAN.md`](TEST_PLAN.md) — each of which now opens with what has been superseded since —
and, for what has been verified after the fixes, [`SPACE_MODE_MATRIX.md`](SPACE_MODE_MATRIX.md)
§10.
