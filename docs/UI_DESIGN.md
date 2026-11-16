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
- **The rail.** It is sticky at `top: 12px` and scrolls inside itself if it outgrows
  the viewport. From the top it holds the monitor, the status line, the cost panel,
  Generate with its reason line, the run report and Recent runs.

Below 1024 px the page is one column. The inputs come first, then the cost panel,
monitor and report. A fixed bottom bar holds one line (the cost, or why Generate is
off) and the same Generate button, restyled. There is never a second Generate in the
DOM. Pressing Generate on a narrow screen scrolls the monitor into view.

## The matrix

The matrix is the existing mode `gr.Radio`, which stays parameter 0 of `/on_generate`.
CSS grid rules generated from the registry place each choice in its cell:

- the row comes from the inputs the mode requires;
- the column comes from the tracks it exports;
- the small corner number comes from registry order. It is an ID for bug reports, not
  a label.

A `gr.HTML` frame behind the radio draws the axis headers, the four inert empty cells
and the two column notes. The cost marker is a tungsten underline on "Sound only", the
column that pays for a thrown-away video.

The eleven radios share one `name`, so the matrix is one Tab stop. A small script
makes the arrow keys move by the grid's geometry. Each radio's accessible name is the
mode's full registry label.

## What you see is what runs

One predicate decides what each mode shows and what it requires:

- a field is shown exactly when `modes.accepts(spec, kind)`;
- a field is marked required exactly when `modes.requires(spec, kind)`.

This is the predicate `generate.scope_request_to_mode` scopes with.

When the mode changes, the value of every field the new mode does not use is
**parked**: it moves into a session `gr.State`, and the widget goes back to its
default. The value comes back when a mode that uses it is selected (`handlers.mode_switch`).
So a UI run never submits a hidden value, and the backend never has to say "an image
was supplied … ignored" about something the user cannot see. API callers still get
the backend's scoping and its notes.

Some modes pin their values. Their pinned values are never written into the user's
fields. The sound-only modes show a read-only line, "Renders at 256 × 256 with no
upscale", in place of the shape controls.

Some fields keep their API slot but show nothing:

- the negative prompt (guidance is fixed at 1.0);
- the shot-list `conditions` (a row with text is refused before every run);
- `noise_scale`, which is written by the four "How much to change your sound" choices
  in Audio → sound.

Strength is shown only for Video → video. "Let the model pick the length" is shown
only where no input has a length. Where an input does have a length, "Match my clip"
writes the matched frame count, or for audio inputs turns on the backend's "match my
audio".

## The rail

**The monitor** is always dark and 16:9. It shows one player, chosen by what the run
exported, so joint modes never show one file twice. Its contents depend on the state:

- **Idle:** the selected mode's sample output from the manifest, captioned "Example
  output, not your run". If there is no sample, a designed idle frame is shown
  instead.
- **During a run:** the phase-labelled progress (`show_progress_on`).
- **After a run:** that run's result, captioned with its mode and seed. It stays on
  screen when the mode changes.

**The cost panel** is the signature element. It is a labelled grid with these rows:

- Renders
- Returns
- Reserves
- Needs (quota free to start)
- Billed
- LoRA

Above the grid are cost lamps that light only when they are true. The figures come
from these sources:

- **Reserves:** `zerogpu.estimate_duration` for the same request, with each extra
  named with its seconds.
- **Needs:** the reservation times the quota multiplier.
- **Renders and Returns:** the size and length come from the request that
  `generate.prepare_request` scopes.

A request that would be refused shows "Won't run", lists every size and length problem
with its nearest legal values, and shows no figure. The lamp words are:

- "Upscale +N s"
- "Enhance +N s"
- "LoRA +N s"
- "Video thrown away"
- "Audio encode"

**Generate** is disabled whenever the form would be refused. The line under it uses
the backend's own refusal words, from `generate.prepare_request`. Generate is also
disabled while a run is in flight. The click runs three steps:

1. A start step disables Generate and clears the monitor.
2. `/on_generate` runs, with its inputs and outputs frozen.
3. A final step re-enables Generate.

The final step is wired twice, as `.then` and as `.failure`, because Gradio 6.24 runs
`.then` listeners only when the step completes. A request that would be refused never
reaches `generate()`, so its red panel can truthfully say "Nothing ran. No GPU time
was used."

**The report** is the `/on_generate` Markdown. On success it shows the seed, the
backend's notes as plain text and a collapsed Diagnostics block, with no cost
reminder. A "What you got" card above it is built from the request and the wall-clock
time:

- mode
- size and upscale
- length
- seed
- LoRA strengths
- time taken

A failure has a plain category headline, a quota line, one next step and the verbatim
detail in a `<details>`. `#run-status` is a live region, and it becomes assertive on
failure.

**Recent runs** keeps the last six runs of the session, each with "Restore settings"
and "Reuse seed".

## LoRA

The LoRA row is an accordion whose label is the live state: "LoRA: none. Add an
adapter" or "LoRA: 2 adapters at 0.8 and 1.0". Inside it are two routes:

- a Hub repo-id box, which accepts Hub ids only, with "Check and add";
- an upload, which is copied into an app-owned folder before it is checked, because
  `delete_cache` prunes Gradio's upload cache.

Adapters are staged with `lora.staged_entry` at strength 1.0 and de-duplicated by
content key. The bay pre-builds three rows, each with the adapter's name, a strength
slider (0.05 to 1.5) and Remove. An accepted adapter gets a roll-edged card. A
rejected one gets a cut card with a plain headline and the validator's message
verbatim.

## Visual system

The tokens are dark under both OS schemes. Gradio's theme variables are pinned to
them with `!important` from the identity bar's `head=`, because Gradio's own sheet
loads later.

| token | hex | role |
|---|---|---|
| ink | `#12161B` | page ground |
| slate | `#1C232B` | the commission panel, the cost panel, the report |
| slate-2 | `#242D37` | raised: selected cell, secondary buttons |
| well | `#0E1216` | recessed: inputs, drop zones, the monitor |
| line | `#39434F` | hairlines |
| paper | `#E9ECEF` | text, the focus ring, neutral accents |
| dust | `#9AA6B2` | secondary and help text |
| tungsten | `#F2A33C` | cost only: lamps, the reserve and quota figures, the Generate fill |
| cut | `#E5484D` / text `#FF8A8E` | failure only |
| roll | `#3FBF7F` | accepted only: staged adapters, a file in a drop zone, a finished run |

**Type:**

- Chakra Petch 600 for the wordmark (20), matrix headers (14), panel titles (16),
  Generate (16) and lamps (12).
- Spline Sans for prose (14/21; help text 13/19).
- Spline Sans Mono 500, with tabular figures, for numerals only.

**Components:**

- **One button family.** Primary is Generate only. Secondary is a slate-2 fill with a
  line border. Quiet buttons are underlined text.
- **One focus ring:** 2 px paper with a 2 px offset.
- **Radii:** panels 10, controls 6, lamps 3.
- **Motion:** state changes only, and none under `prefers-reduced-motion`.

## Showcase

`LTX25_EXAMPLES_DIR` (default `examples/`) may hold a `manifest.json` (schema in
`ui/showcase.py`). Each mode can have examples and one sample output:

- An example has a label, a prompt, media paths and settings. "Try an example" fills
  the form and runs nothing.
- A sample output has its file, caption, settings and seed.

Entries that name a missing file, or a path outside the folder, are skipped. With no
manifest, there is no example row and the monitor shows its idle frame.

## Contracts

- **`/on_generate`:** parameter 0 is the mode radio, then `FIELD_KEYS` in order, which
  is append-only with unchanged component classes. It returns `[Video, Audio,
  Markdown]`.
- **`/on_validate_lora`:** its inputs and outputs are unchanged.
- **Other events:** every other event is `api_visibility="private"`.
- **Progress:** no event builds `gr.Progress(track_tqdm=True)`, which patches tqdm for
  the whole process.
