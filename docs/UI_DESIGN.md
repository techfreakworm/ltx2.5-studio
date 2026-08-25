# UI Design — "The Machine Room"

The design direction for LTX-2.5 Studio's interface, every deliberate choice and its
reason, and what was rejected. Written so the design can be reviewed without running
the app. Implementation lives in `ltx25_studio/ui/chrome.py` (tokens, stylesheet,
masthead) and `ltx25_studio/ui/blocks.py` (layout and the `elem_id`/`elem_classes`
hooks the stylesheet targets).

## The brief, restated

Eleven modes over one jointly audio-visual model, on metered ZeroGPU time. The
audience is practitioners spending quota. The page has one job: **let a user
commission one run knowing exactly what it will cost and what it will return** —
before the money is spent, not after. The previous UI was never designed, only
accumulated: a wall of eleven radio rows, banners of paragraph soup, the headline
LoRA feature in an accordion, and two cost truths (stage-2 doubling, audio-only
full-video-compute) stated either after the run or never.

## Direction

**The machine room** — the visual language of broadcast and production hardware:
slate console panels, engraved equipment lettering, annunciator lamps, mono data
readouts. Chosen because it is *this* subject's own world (the app already speaks in
shots, frames, slates) and because this project's defining ethos — honesty about
cost, loud failure — has a native idiom there: the annunciator panel, whose entire
job is to light a lamp when something expensive or dangerous is true.

The interface is dark. That is a subject decision, not a default: footage-viewing
environments are dark so the footage reads true, and the output pane holds video.
The specificity comes from *which* dark and *what signals*: warm blue-gray slates
rather than neutral black, and a three-signal semantic system rather than a single
decorative accent.

## Tokens

Palette (defined once in `chrome.py`, prefixed `--mr-`):

| token | value | role |
|---|---|---|
| ink | `#12161B` | identity ground: masthead, annunciator |
| slate | `#1C232B` | console panel: mode board, briefing |
| slate-2 | `#242D37` | raised panel / hover / selected |
| line | `#39434F` | gasket hairlines |
| paper | `#E9ECEF` | primary text on slate |
| dust | `#9AA6B2` | secondary text on slate |
| tungsten | `#F2A33C` | **cost** signal — lamps, warned-mode edges, Generate |
| cut | `#E5484D` | **failure** signal — the run-status panel only |
| roll | `#3FBF7F` | reserved for accepted/complete accents |

The three signal colors are semantic, never decorative: tungsten always means "this
costs", cut always means "this failed". Generate is tungsten on purpose — pressing
it is the act of spending, and the button wears the spend color.

Type, three roles, loaded from Google Fonts with real fallback stacks:

* **Chakra Petch** (display) — equipment lettering: the wordmark, mode-card names,
  panel headers, lamp labels. Squared terminals read as engraved hardware type.
* **Spline Sans** (body) — labels, info strings, briefing prose.
* **Spline Sans Mono** (data) — every number a user reads: the annunciator, the
  constraint hints, seeds, dimensions. A number in the UI is an instrument reading
  and is set like one.

Chakra Petch + Spline Sans (+ its companion mono) were chosen over the default-y
pairs (Inter, Space Grotesk, IBM Plex) precisely because they are *not* the faces a
generic pass produces, while still being sober enough for a working tool.

## Layout

Two-column console, unchanged in skeleton from the predecessor (it was the one
sound structural decision): the **commission column** (mode → prompt/inputs → shape
→ sampling → LoRA bay → annunciator → Generate) and the **returns column** (status,
then players). The order in the commission column is the order of the decision, top
to bottom, ending at the point of commitment.

### Masthead

A slate identity strip: wordmark (`LTX-2.5` with the model number in tungsten),
one-line tagline, then the **fact rail** — the two standing truths as
lamp-labelled rows (`JOINT A/V`, `METERED`). These replace two stacked Markdown
banner paragraphs; every fact from the old banners survives (both tracks always
rendered; full video compute; daily quota at 2× xlarge; failed runs cost quota; no
auto-retry and why), but as scannable labelled rows instead of prose soup. The copy
lives in `render.STANDING_FACTS`; `chrome.masthead_html` only gives it form.

### The mode board

Eleven modes is the interface's hardest problem, and a vertical radio list answered
it with a wall. The selector is now a **board**: the same `gr.Radio` (the component
class is part of the published API surface — `parameters[0]["component"] == "Radio"`
in `test_space_matrix`), styled by CSS grid into slate cards. Each card carries its
registry number (`01`–`11`, zero-padded — the same numbers PLAN §3.2 and the mode
matrix docs use, so a mode can be named in a bug report without ambiguity) and its
label. The four modes carrying a registry `cost_note` get a **tungsten left edge**
plus the ⚠️ marker.

The tungsten-edge selector is *generated from the registry* at build time
(`chrome._cost_marked_selector` walks `modes.MODES` and emits
`label:has(input[value="<mode id>"])` per warned mode) — the same derivation rule as
the ⚠️ marker: a new warned mode is painted by adding a registry entry, never by
editing CSS. Card selection state uses Gradio's own `selected` class; the native
radio dot is visually hidden with focus preserved
(`label:has(input:focus-visible)` gets a visible outline).

### The briefing

Directly under the board, one slate card describing the selected mode: takes /
returns, the registry `cost_note` **verbatim** in a tungsten-edged blockquote, and
the mitigations (min resolution, stage-2 skip) in plain words. Unchanged in
substance from the old `render.mode_briefing`; restyled to read as the board's
detail panel.

### The annunciator — the signature element

One strip above Generate, mono type on ink, always present, kept live by
`handlers.on_cost_change` on every change to the mode or any cost-relevant control:

```
text_to_video · renders 704×480 · 121 frames ≈ 5.0 s @ 24 fps · reserves ≈60 GPU-seconds
STAGE 2 ×2 — returns 1408×960, not the 704×480 requested: it renders at the size
above, then upsamples and refines. GPU reservation ×1.5, included in the figure above.
```

Lamps light only when true:

* **FULL VIDEO COMPUTE** — audio-only modes: video rendered at the pinned minimum
  and discarded, mitigations named. (Restates the briefing's cost note at the point
  of commitment, for the user who scrolled past it.)
* **AUDIO ENCODE ON TOP** — `audio_to_video`: the encode is paid on top of a full
  render.
* **STAGE 2 ×2** — the returned size stated next to the requested size *before the
  run*: the operator requirement that nothing previously satisfied. The doubling
  factor is `pipelines.STAGE_2_SPATIAL_SCALE` — the same constant the post-run size
  note uses — and the reservation figure calls the real
  `zerogpu.estimate_duration`, the function the platform actually bills on. No
  number in the strip is a second copy of anything.

This is where the design spends its one aesthetic risk: a black instrument strip
with amber lamps is an unusual thing to put in a Gradio form, and it is the one
element the page should be remembered by — because it *is* the product's thesis
(honest cost, before commitment) made visible.

### The LoRA bay

The headline feature was in an accordion; it is now a **bay**: always open, framed
(`#lora-bay`), with a display-type header band and a one-line promise
("validated before any weights are touched … rejected in milliseconds, with the
exact reason"). Both acquisition routes stay side by side; the verdict surface and
the staged-adapters line remain first-class. The empty state of the staged list was
deduplicated: it previously rendered the identical idle paragraph twice in a row
(verdict + summary), which read as a broken template; the summary's empty state is
now one distinct line ("**Staged for the next run:** none — base LTX-2.5
weights."), still stated positively, never as blankness.

### Failure, loud

The run status (`#run-status`) is quiet by default. Every failure shape
`render.py` produces — `### ❌ Generation failed`, `### 🛑 Internal error` — opens
with an `<h3>`, and success/running never do, so the stylesheet keys the loud
treatment on that structure: `#run-status:has(h3)` gets a cut-red border, red
left edge and red tint, with the error class name and the verbatim actionable
message inside. The state lives in the *content*, which is what the frozen
`(video, audio, markdown)` handler contract allows; no second channel, no CSS
class the handler would have to set.

Verified live (faked pipeline): an illegal width renders the red panel with
`DimensionError — width=700 is not divisible by 32. … Use 672 or 704 instead.`

### The run report

`render.success` now surfaces `result.warnings` — previously those notes (the
stage-2 "this output is 1408×960, not the 704×480 requested" note, the mono-widen
note) were computed, attached to the result, and *never shown*; they reached the
sidecar file and the Space log only. The seed line now says what the seed is for:
"seed `N` — rerun with it and the same settings to reproduce".

## Copy decisions

* **The shot-list contradiction is fixed.** The old UI told users, in two places,
  that the anchor column takes a *latent index* (`_MULTISHOT_NOTE`, and the
  Dataframe label "anchor (latent index)") while the ruled convention — executable
  in `test_shot_anchor_convention.py` — is that the user types a **frame** and
  `conditioning.py` converts exactly once. A user following the old note got a
  mis-anchored shot every time. Both places now say *frame*, and the note explains
  the 8:1 snap and points at the readout.
* **Prompt enhancement stays documented-as-unavailable, no dead toggle** (PLAN
  §3.5): a caption under the prompt fields names the missing component
  (`google/gemma-4-E2B-it`), says why no toggle is offered, and tells the user what
  to do instead (write the full shot; camera/lighting/motion language works).
* **`noise_scale=1.0` trap named**: the slider info now says that at 1.0 the upload
  contributes *nothing* and the run is a full-price generation of fresh audio.
* **Seed reproducibility stated** on the field and in the run report.
* The empty status is an invitation with a guarantee: "Nothing is spent until the
  button is pressed."

## Mechanics (Gradio 6, pinned 6.24.0)

* `css=`/`theme=` moved to `launch()` in Gradio 6, and `app.py` (not this layer's
  file) calls a bare `demo.launch()`. The stylesheet and font links ride on
  **`gr.HTML(head=...)`** — the documented Gradio 6 injection point — carried by
  the masthead component, so shell and styles arrive together. No deprecated
  constructor kwargs, no `app.py` change.
* The CSS is assembled in Python so the warned-mode selector derives from the
  registry (see above). Identity surfaces are self-colored — the same slate in both
  Gradio themes, like hardware on a desk — so every text node inside them carries
  an explicit color (Gradio's theme stylesheets load after ours and win specificity
  ties; inheritance painted the wordmark dark-on-dark in light mode until this was
  made explicit). Form internals deliberately keep the native theme so light-mode
  users keep a light form.
* Reduced motion respected (the only transitions are 120 ms card hovers, disabled
  under `prefers-reduced-motion`); focus-visible outlines on cards and Generate;
  dust-on-slate and tungsten-on-ink both clear ~6:1 contrast.
* Captions under media inputs (Gradio 6 media components have no `info=`) are now
  registered per-field and **toggled with their field** by the mode selector; the
  old shape built them write-only, so a hidden image input left its caption
  floating on screen for modes with no image at all.

## Contracts honoured

* `/on_generate` endpoint: same fn, no `api_name`, positional order `mode_id` +
  `FIELDS` order, returns `(video, audio, markdown)`. Verified headlessly:
  `parameters[0] == Radio`, 17 parameters, returns `[Video, Audio, Markdown]`.
* `fields.FIELD_KEYS` order unchanged; component classes per field unchanged.
* `handlers.on_mode_change` / `on_shots_change` signatures unchanged (tests call
  them positionally). The new `on_cost_change` is additive.
* No behavioural or seam test was touched. **Zero presentational test updates were
  required**: no test pinned the old layout or copy except through self-referencing
  `render.*` comparisons, which move with the renderer.

## Rejected

* **Tabs or a dropdown for the eleven modes** — a dropdown hides ten of eleven
  answers to "what can this do?", tabs would either duplicate widget sets (the
  exact predecessor disease `fields.py` exists to prevent) or fake tab-ness over
  one form; and the Radio component class is part of the published API surface.
  The board keeps all eleven visible, scannable, and warning-marked at the moment
  of choice.
* **A gradio Theme subclass** — `theme=` belongs to `launch()` in Gradio 6, which
  is `app.py`'s line; and a theme cannot express the registry-derived cost edges or
  the annunciator anyway. CSS-over-default-theme delivers the identity without
  touching the frozen entrypoint.
* **Hiding the resolution controls on audio-only modes** (rather than greying) —
  rejected long before this redesign and kept rejected: a disabled control showing
  the forced 256 answers "what ran?"; absence answers nothing.
* **Restoring width/height when leaving an audio-only mode** — visiting mode
  08–10 forces 256 into the boxes; returning to a video mode leaves 256 in place
  (the handler is stateless and cannot distinguish "forced 256" from "typed 256").
  A restore-to-default would clobber values users chose; invisible state edits are
  the exact class of magic this project distrusts. The annunciator makes the
  current truth visible at all times, and the sticky value errs cheap, never
  expensive.
* **Duplicating the full `cost_note` in the annunciator** — the briefing already
  quotes it verbatim at selection; the strip restates the fact in instrument
  wording at commitment. Twice verbatim would train users to skip both.
* **Motion beyond hover** (pulsing running lamp, scroll reveals) — this is a tool;
  a page-load sequence on a quota console is noise. The one place motion was
  considered (a pulsing ⏳ during runs) was cut because encoding run-state into
  content-keyed CSS would have meant styling on emoji text-matching or widening the
  handler contract.
* **The three AI-default looks** (cream + serif + terracotta; neutral black + acid
  accent; broadsheet hairlines) — the first and third fight the subject (footage
  wants dark; a console is not a newspaper); the second is what "dark tool" becomes
  without a decision, and the warm slate + semantic three-signal system is that
  decision.

## Verification

Everything below ran with **no GPU, no weights, no network**, via the test suite's
faked-pipeline seams (`pipeline_factory`, `encoder` — the same pattern as
`tests/test_ui_generate_seam.py`; the throwaway launcher lives in
`outputs/.ui_preview.py`, gitignored):

* Headless build + API contract check (twice-buildable, endpoint shape).
* Full offline gate green (see commit / report for counts).
* Live in a browser: mode board selection and per-mode visibility; forced 256 on
  mode 08 with the FULL VIDEO COMPUTE lamp; STAGE 2 lamp showing the doubled
  return size and the ×1.5 reservation from the real estimator; a complete run
  rendering the report with seed, run dir and the stage-2 size warning; an illegal
  width rendering the loud red `DimensionError` panel; light theme coherent after
  the explicit-color hardening.
