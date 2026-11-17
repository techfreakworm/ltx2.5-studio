# UI design: the machine room

What the LTX-2.5 Studio interface is, and why each part is shaped as it is. Code:
`ltx25_studio/ui/` (`chrome.py` holds tokens, stylesheet and identity bar; `blocks.py`
holds layout and wiring; `handlers.py` holds every decision; `render.py` holds every
string a user reads; `fields.py` holds the input spec).

## Job

Practitioners spend a metered daily GPU quota on a model that makes video and sound
together. The page lets them pick what they start from and what they want back, see
exactly what the run will cost and return, press Generate, and watch the result
arrive without moving their eyes.

## Layout

At 1024 px and wider the page has two columns under a one-line identity bar.

- **The commission column.** At the top is the mode matrix: rows are what you start
  from (Text, Image, Image + text, Video, Audio), and columns are what you get back
  (Video + sound, Video (silent), Sound only). Below it come the purpose line and up
  to three example buttons, then the selected mode's inputs: media first, each with
  its one lever, then the prompt with Enhance. After those come shape, length and
  Upscale, the LoRA row and Advanced.
- **The rail.** From the top it holds the monitor, its caption, the cost panel,
  Generate with its reason line, the "What you got" card with its actions, the run
  report and Recent runs. It is sticky at `top: 12px`. A small script sizes it: at page
  scroll 0 it ends 12 px above the bottom of the window at any desktop size, and once
  the page scrolls and the rail sticks it grows to the window less 24 px, so no dead
  band is left under it. Its content never wraps: Gradio gives every Row child
  `flex-wrap: wrap`, and a rail that wrapped sent its overflow into a second column off
  the right edge, so the rail sets `nowrap` and its children never shrink. When the
  content is taller than the rail, the rail scrolls inside itself with a real
  scrollbar, drawn in a gutter outside the aligned right edge. A 4 px padding keeps
  focus rings unclipped; negative margins keep the content edges level with the
  identity bar.
- **Generate never leaves the rail.** It is sticky at both of the rail's edges, flush
  with them, so nothing shows past it. A row that passes under its top edge (the cost
  panel, on a short window) or its bottom edge (the report, with Generate stuck at the
  top) runs into a 12 px ink fade instead of being cut.
- **The cost figure stays beside Generate on a short window.** The script fits the
  monitor, the caption, the cost panel and Generate into the rail at rail scroll 0.
  When they are taller than the rail, the monitor gives up height first, down to
  180 px, letterboxed. If the panel still runs under Generate, the panel's figure line
  stands above the button: "704 × 480, 5.0 s, needs 360 s quota" (off ZeroGPU, "on the
  CPU"). It names what the run returns, as the panel's Returns row does: "1408 × 960
  upscaled, 5.0 s, …" with Upscale ×2, and "Sound, 5.0 s, …" (or "Sound, the length of
  your audio, …") in a sound-only mode, never the 256 × 256 picture it throws away.
  When there is room to spare, the caption keeps four lines less the panel's notes, so
  Generate stays where it is from mode to mode.
- **What a click changes comes into view.** Generate and Show scroll the rail as little
  as it takes to show the monitor; when a run ends, the head of its report (the Done
  card's head, or a failure's headline and next step) is brought into view clear of
  Generate.

Below 1024 px the page is one column. The inputs come first, then the cost panel,
monitor and report. A fixed bottom bar holds one short line (the figure line above, or
why Generate is off: "Needs a prompt", "Won't run: 2 problems (width, frames)") and the
same Generate button, restyled. There is never a second Generate in the DOM. Pressing
Generate or Show on a narrow screen scrolls the monitor into view; Restore settings
brings the purpose line, which names the restored mode, into view; and while the
monitor shows a sample a "See an example output" button under the purpose line jumps
to it. The same button, at every width, offers the selected mode's sample over a run
of another mode (see the monitor, below).

## The matrix

The matrix is the existing mode `gr.Radio`, which stays parameter 0 of `/on_generate`.
CSS grid rules generated from the registry place each choice in its cell:

- the row comes from the inputs the mode requires;
- the column comes from the tracks it exports;
- the small corner number comes from registry order. It is an ID for bug reports, not
  a label, and it is generated content with empty alternative text, so it is not read
  as part of the cell's name.

A `gr.HTML` frame behind the radio draws the axis headers, the four inert empty cells
and the two column notes. The headers and empty cells are hidden from assistive tech
(each cell's own name already says what it starts from and returns); the column notes
are not, and each cell of a noted column is described by its note
(`aria-describedby`). The cost marker is a tungsten underline on "Sound only", the
column that pays for a thrown-away video.

The eleven radios share one `name`, so the matrix is one Tab stop. A small script
makes the arrow keys move by the grid's geometry. A mode has one name everywhere,
the registry's `label`: the cell, the purpose line, the cost panel's title, the
monitor caption, the status line and Recent runs.

## What you see is what runs

One predicate decides what each mode shows and what it requires:

- a field is shown exactly when `modes.accepts(spec, kind)`;
- a field is marked required exactly when `modes.requires(spec, kind)`.

This is the predicate `generate.scope_request_to_mode` scopes with.

**Media order.** The media inputs are built in one order derived from the registry
(`handlers.media_order`): in every mode the inputs it requires come before the ones it
may take, so the first media input a mode shows is one it needs (Audio to video asks
for its audio, then offers an optional first-frame image). DOM order is visual order
and focus order; no CSS reorders them. A registry no single order could satisfy fails
at build time.

**Mode 11's help.** Under Audio to video's audio, the line says speech is not
lip-synced and where a speaking face is made. Under its optional first-frame image,
above the prompt, the line says an old-sounding recording can come out as a period
film, and the two ways to get colour: "in colour" in the prompt, or a first-frame
picture. No other mode says either.

**Parking.** When the mode changes, the value of every field the new mode does not use
is parked: it moves into a session `gr.State`, and the widget goes back to its default.
The value comes back when a mode that uses it is selected (`handlers.mode_switch`).
This covers the prompt, Enhance, media, the change level, the decoder, and the size
and length fields a mode hides (width and height in the sound-only modes, the shortest
and longest seconds where an input sets the length). So a UI run never submits a
hidden value, and the backend never has to say "an image was supplied … ignored" about
something the user cannot see. API callers still get the backend's scoping and its
notes.

**Restore settings** runs the same switch against the mode actually left, then writes
the restored run's own settings, in one step (a second step chained after it would race
the radio's own change event). What the restored mode does not take is parked like any
other switch. Its media are not restored.

**Length.** "Match my clip" writes a clip's frame count into the frame field for the
run; a mode without a timed input gets the user's own length back, never the clip's.
In Advanced the frame count shows while the length is set, and the shortest and
longest seconds while the model picks it; never both. The length slider steps by
0.1 s.

**Levers.** Every visible control changes what reaches the pipeline (a test flips each
one, in every mode, with Upscale on and off).

- Video to video is a variation of the user's clip: it shows "How much to change your
  clip" (A little / Moderately / A lot, default A lot), writing the backend's own
  levels to `noise_scale`; the prompt steers the result only at A lot. Its length is
  always its clip's, so it offers no length choice.
- Audio to sound shows "How much to change your sound" (four choices, the same field).
- The clip-strength slider shows only where a video condition reads it; with today's
  registry that is no mode, and its API slot stays.
- While Upscale ×2 is on, the decoder choice cannot be changed: stage 2 always finishes
  with the diffusion decoder, and the line under the choice says so.

Some fields keep their API slot but show nothing:

- the negative prompt (guidance is fixed at 1.0);
- the shot-list `conditions` (a row with text is refused before every run);
- `noise_scale` itself, which the change choices write.

Some modes pin their values. Their pinned values are never written into the user's
fields. The sound-only modes show a read-only line, "Renders at 256 × 256 with no
upscale", in place of the shape controls. A size no preset matches checks no Shape
chip.

## The rail

**The monitor** is always dark and 16:9. It shows one player, chosen by what the run
exported, so joint modes never show one file twice. Its contents depend on the state:

- **Idle:** the selected mode's sample output from the manifest, captioned "Example
  output, not your run". If there is no sample, a designed idle frame names the mode.
- **During a run:** the phase-labelled progress (`show_progress_on`).
- **After a run:** that run's result, captioned with its own mode (the one that ran,
  even if another was selected while it ran) and seed. It stays on screen when the
  mode changes.
- **Shown from Recent runs:** an earlier run's own file, captioned "Earlier run".
- **A sample over a run:** while the monitor holds a run (yours or an earlier one) of
  another mode than the selected one, and that mode has a sample, the button under the
  purpose line reads "See an example of Audio to audio" at every width (class
  `mr-over-run`). It puts that sample on the monitor with its usual caption, and later
  switches follow with their own samples; the run is kept, and the same button reads
  "Back to your run" and puts it back, while its file is still on disk. Below 1024 px
  the button also scrolls to the monitor.

**The cost panel** is the signature element. It is a labelled grid with these rows:

- Renders
- Returns
- Reserves
- Needs (quota free to start)
- Billed
- LoRA

Above the grid are cost lamps that light only when they are true; a request that would
be refused lights none. The figures come from these sources:

- **Reserves:** `zerogpu.estimate_duration` for the same request, with each extra
  named with its seconds.
- **Needs:** the reservation times the quota multiplier.
- **Renders and Returns:** the size and length come from the request that
  `generate.prepare_request` scopes. A clip mode with no clip loaded yet says "the
  length of your clip".

A request that would be refused shows "Won't run", lists every size and length problem
with its nearest legal values, and shows no figure. The lamp words are:

- "Upscale +N s"
- "Enhance +N s"
- "LoRA +N s"
- "Video thrown away"
- "Audio encode"

The cost words at the point of choice come from the same estimator: the Upscale line
says how many seconds it adds, and the Enhance line says what it reserves and that a
cold start can take about 2 minutes (it mentions an image only where the mode takes
one).

**Off ZeroGPU there is no meter, and the page names none.** The platform is resolved as
`app.py` resolves it (`device.current()`, read by `handlers.run_platform`). Where the app
is not on ZeroGPU (a Mac, the CPU, a CUDA box of one's own):

- the Reserves, Needs and Billed rows give way to one row, "Runs: on the CPU; nothing is
  metered" (or "on this Mac's GPU", "on the GPU": the Done card's device words), and the
  lamps carry no seconds;
- "How quota works" is not shown; the column notes, the Enhance, Upscale and LoRA lines,
  the phone bar's line ("704 × 480, 5.0 s, on the CPU") and the refused panel say what
  happens without a meter;
- a failure card has no quota line: a request refused before running says "Nothing
  ran.", any other card only its step; the cards about the platform's quota and its
  per-run limit, which a local run cannot raise, fall back to the generic card; the lost
  run and the empty-file cards have no billing sentence.
- Upscale ×2 where this machine cannot run it for the request (the backend's own gate,
  `generate._checked_stage_2_size`: off CUDA, above a small decoder size) is parked:
  unticked, disabled, and its line says "Upscale ×2 needs a CUDA GPU; this machine runs on
  the CPU." It frees itself at a size the backend accepts.

On ZeroGPU every one of these sentences is as above.

**Generate** is disabled whenever the form would be refused. The line under it uses
the backend's own refusal words, from `generate.prepare_request`. Generate is also
disabled while a run is in flight. The click runs three steps:

1. A start step disables Generate and clears the monitor.
2. `/on_generate` runs, with its inputs and outputs frozen.
3. A final step re-enables Generate and puts the result on the monitor.

Gradio 6.24's page runs a step's `.then` listeners only when the step completes. When
the server fails it (the function raised, or Gradio could not process what it returned,
such as a result file it will not serve) or the page cannot apply its outputs, the page
runs the `.failure` listeners alone. So the final step is `.then`, and a failure step of
its own is wired as `.failure`, after `/on_generate` and after the final step too. The
failure step sends no file, so it cannot fail the way the step before it did. It brings
Generate back, puts the monitor back to its idle frame and shows a card: for a run that
finished, "The run finished, but its result could not be shown", what the run used (on
ZeroGPU, that it is billed), Press Generate again, and the server's own message as its
Detail. Gradio tells the page only "Error", so the app keeps that message per session
from the step that failed. A request that would be refused never reaches `generate()`,
so its red panel can truthfully say "Nothing ran. No GPU time was used."

The failure step reads the run's own inputs, so when Gradio could not prepare them (an
upload its cache clean-up removed after six hours) it fails the same way. A last
failure listener after it reads no input at all and sends no file: it brings Generate
back, puts the monitor back to its idle frame and shows "Your inputs could not be read"
("Nothing ran", the upload to add again, Gradio's message as its Detail). Typed number
fields declare no range, because Gradio refuses a value outside one before any handler
runs; the app's own gate refuses a seed below −1 in the page's words. Sliders keep
their ranges.

**The report** is the `/on_generate` Markdown, in `#run-status`. That region is in the
page from the first paint (empty, visually hidden), with `role="status"`, so its first
message is announced; it turns assertive on failure. On success it carries a "Done"
line (read, not shown), the backend's notes as plain text (the UI shows them as they
come and does not reword them) and a collapsed Diagnostics block with
`record.diagnostics` (VRAM, partition, run folder) and the mode id. A "What you got"
card above it is built from the run record:

- mode
- size and upscale
- length, and where it came from
- the change level, named with its choice (Video to video, Audio to sound)
- seed
- LoRA strengths
- the decoder that ran, by its choice's name (Steadier (diffusion decoder), Sharper
  (standard decoder))
- time taken, labelled by the device that ran it; on ZeroGPU, with the quota it drew

Reuse seed and Use this prompt close the card, and so does a quiet "Download the sound
(WAV)" when the run returned its picture and its sound as two files (a video with
sound): the sound track alone, for a voice or ambience to use elsewhere. A sound-only
or silent run, or one file carrying both tracks, offers none; the player's own
download is the video, with its sound.

A failure has a plain category headline, a quota line, one next step and the verbatim
detail in a `<details>`. Categories arrive as strings from `errors.categorise`; each
has its own card (not enough quota, a run longer than the platform allows, no GPU free
in time, out of GPU time, out of memory, an input problem, an adapter that does not
fit, the enhancer unavailable, a bug), and a category the UI has never seen gets a
generic one. A queue timeout reads as "No GPU was free in time", never as running out
of GPU time.

**Recent runs** keeps the last six runs of the session, each with "Show" (while its
file is still on disk), "Restore settings" and "Reuse seed".

## LoRA

The LoRA row is an accordion whose label is the live state: "LoRA: none. Add an
adapter" or "LoRA: 2 adapters at 0.8 and 1.0". Inside it are two routes:

- a Hub repo-id box, which accepts Hub ids only, with "Check and add";
- an upload, which is copied into an app-owned folder before it is checked, because
  `delete_cache` prunes Gradio's upload cache.

Adapters are staged with `lora.staged_entry` at strength 1.0 and de-duplicated by
content key. The bay pre-builds three rows, each with the adapter's name, a strength
slider and Remove. A row's slider takes its range from `lora.staging` (more than 0, at
most its `MAX_WEIGHT`) when the row is shown, so the LoRA package stays lazy until an
adapter is checked. An accepted adapter gets a roll-edged card. A rejected one gets a
cut card with a plain headline and the validator's message, with a server path to the
file shown as its name. At 560 px and below a row stacks: name, strength, Remove.

## Visual system

The tokens are dark under both OS schemes. Gradio's theme variables are pinned to
them with `!important` from the identity bar's `head=`, because Gradio's own sheet
loads after ours. Where a Gradio rule out-ranks a class (its `h1`–`h6` reset is
`(0,1,1)`), our selectors carry an ID.

| token | hex | role |
|---|---|---|
| ink | `#12161B` | page ground |
| slate | `#1C232B` | the commission panel, the cost panel, the report |
| slate-2 | `#242D37` | raised: selected cell, secondary buttons |
| well | `#0E1216` | recessed: inputs, drop zones, the monitor |
| line | `#39434F` | hairlines |
| line-strong | `#66727F` | control edges: input borders (3.8:1 on the well), a slider's unfilled track (3.2:1 on slate) |
| paper | `#E9ECEF` | text, the focus ring, neutral accents, a slider's fill and thumb |
| dust | `#9AA6B2` | secondary and help text |
| tungsten | `#F2A33C` | cost only: lamps, the reserve and quota figures, the Generate fill |
| cut | `#E5484D` / text `#FF8A8E` | failure only |
| roll | `#3FBF7F` | accepted only: staged adapters, a file in a drop zone, a finished run |

**Type:**

- Chakra Petch 600 for the wordmark (20), matrix headers (14), panel titles (16),
  Generate (16) and lamps (12).
- Spline Sans for prose (14/21; help text 13/19) and for figures inside sentences, with
  tabular figures. Help lines are capped at `70ch`, which at 13 px is about 89
  characters a line, not 70 (review round 3, P7; not changed this round).
- Spline Sans Mono 500, with tabular figures, for number inputs and the two tungsten
  figures.

**Components:**

- **One button family.** Primary is Generate only. Secondary is a slate-2 fill with a
  line border. Quiet buttons are underlined text. All of them use the small size.
- **Native controls** are drawn from the tokens: the slider's track is paper against
  line-strong (4.1:1) with a paper thumb in an ink ring; a checked box carries an ink
  tick, a radio an ink dot; accordions use the same chevron as "How quota works".
- **One focus ring:** 2 px paper with a 2 px offset.
- **Radii:** panels 10, controls 6, lamps 3.
- **Motion:** state changes only, and none under `prefers-reduced-motion`.
- **Footer:** Gradio's hidden footer controls stay hidden ("Stop Recording" is never
  shown), and its run-history link, which competes with Recent runs, is hidden.

## Showcase

`LTX25_EXAMPLES_DIR` (default `examples/`) may hold a `manifest.json` (schema in
`ui/showcase.py`). Each mode can have examples and one sample output:

- An example has a label, a prompt, media paths and settings (shape or exact size,
  seconds, upscale, enhance, `auto_duration`, seed, and `noise` naming a change
  choice). "Try an example" fills the form and runs nothing.
- A sample output has its file, caption, settings and seed. Its settings may be free
  text, or a mapping (size, seconds, upscale, enhance, `auto_duration`) that the
  caption says in words.

Entries that name a missing file, or a path outside the folder, are skipped. With no
manifest, there is no example row and the monitor shows its idle frame. A manifest
with entries for all eleven modes needs no code change.

## Contracts

- **`/on_generate`:** parameter 0 is the mode radio, then `FIELD_KEYS` in order, which
  is append-only with unchanged component classes. It returns `[Video, Audio,
  Markdown]`.
- **`/on_validate_lora`:** its inputs and outputs are unchanged.
- **Other events:** every other event is `api_visibility="private"`.
- **Progress:** no event builds `gr.Progress(track_tqdm=True)`, which patches tqdm for
  the whole process.
