"""The visual shell: tokens, stylesheet, identity bar and the matrix frame.

"The machine room", dark under both OS colour schemes. Three signals, never
decorative: tungsten means this costs (cost lamps, the quota figure, the
Generate fill), cut means this failed, roll means accepted, legal or complete.
Chakra Petch for equipment lettering, Spline Sans for prose, Spline Sans Mono
for numerals only. The full rationale is in ``docs/UI_DESIGN.md``.

**Derived, not written.** The matrix places each mode's cell from the
registry -- its row from the inputs it requires, its column from the tracks it
exports -- and its corner number from registry order. A new mode gets a cell by
adding a registry entry.

**Why the stylesheet rides on ``gr.HTML(head=...)``.** Gradio 6 moved ``css=``
and ``theme=`` to ``launch()``, which ``app.py`` owns. The identity bar carries
the stylesheet, the fonts and a small script, so the shell arrives in one piece.
Gradio's own theme sheet loads after ours, so theme variables are set with
``!important``.

Imports the registry and ``render`` only: no Gradio. Every function returns a
string.
"""

from __future__ import annotations

import html

from .. import modes
from . import render

__all__ = ["MATRIX_ID", "head_html", "identity_html", "matrix_cells", "matrix_frame_html"]

MATRIX_ID = "mode-matrix"

_FONTS_HREF = (
    "https://fonts.googleapis.com/css2"
    "?family=Chakra+Petch:wght@500;600"
    "&family=Spline+Sans:wght@400;500;600"
    "&family=Spline+Sans+Mono:wght@500"
    "&display=swap"
)


# ---------------------------------------------------------------------------
# the matrix, derived from the registry
# ---------------------------------------------------------------------------


def matrix_cells() -> dict[str, tuple[int, int, int]]:
    """``mode id -> (row, column, number)``; rows and columns count from 1."""
    rows = [key for key, _ in render.ROWS]
    columns = [key for key, _ in render.COLUMNS]
    cells = {}
    for number, (mode_id, spec) in enumerate(modes.MODES.items(), start=1):
        row = rows.index(frozenset(spec.inputs)) + 1
        column = columns.index(frozenset(spec.exports)) + 1
        cells[mode_id] = (row, column, number)
    return cells


def _empty_cells() -> list[tuple[int, int]]:
    taken = {(r, c) for r, c, _ in matrix_cells().values()}
    return [
        (r, c)
        for r in range(1, len(render.ROWS) + 1)
        for c in range(1, len(render.COLUMNS) + 1)
        if (r, c) not in taken
    ]


def matrix_frame_html() -> str:
    """Axis headers, the inert empty cells and the column notes, behind the radio."""
    parts = ['<div class="mx-frame" aria-hidden="true">']
    for c, (_key, label) in enumerate(render.COLUMNS, start=1):
        cost = (
            " is-cost"
            if label in render.COLUMN_NOTES and "pay" in render.COLUMN_NOTES[label]
            else ""
        )
        parts.append(
            f'<div class="mx-col{cost}" style="grid-column:{c + 1};grid-row:1">'
            f"{html.escape(label)}</div>"
        )
    for r, (_key, label) in enumerate(render.ROWS, start=1):
        parts.append(
            f'<div class="mx-row" style="grid-row:{r + 1};grid-column:1">{html.escape(label)}</div>'
        )
    for r, c in _empty_cells():
        parts.append(f'<div class="mx-empty" style="grid-row:{r + 1};grid-column:{c + 1}"></div>')
    notes_row = len(render.ROWS) + 2
    for c, (_key, label) in enumerate(render.COLUMNS, start=1):
        note = render.COLUMN_NOTES.get(label)
        if note:
            parts.append(
                f'<p class="mx-note" style="grid-row:{notes_row};grid-column:{c + 1}">'
                f'<span class="mx-note-col">{html.escape(label)}:</span> {html.escape(note)}</p>'
            )
    parts.append("</div>")
    return "".join(parts)


def _matrix_css() -> str:
    rules = []
    for mode_id, (row, column, number) in matrix_cells().items():
        sel = f'#{MATRIX_ID} label:has(input[value="{mode_id}"])'
        rules.append(f"{sel} {{ grid-row: {row + 1}; grid-column: {column + 1}; }}")
        rules.append(f'{sel}::after {{ content: "{number:02d}"; }}')
    return "\n".join(rules)


# ---------------------------------------------------------------------------
# the stylesheet
# ---------------------------------------------------------------------------

_TOKENS = """
:root {
  --mr-ink: #12161b;
  --mr-slate: #1c232b;
  --mr-slate-2: #242d37;
  --mr-well: #0e1216;
  --mr-line: #39434f;
  --mr-paper: #e9ecef;
  --mr-dust: #9aa6b2;
  --mr-tungsten: #f2a33c;
  --mr-on-tungsten: #16130b;
  --mr-cut: #e5484d;
  --mr-cut-text: #ff8a8e;
  --mr-roll: #3fbf7f;
  --mr-display: "Chakra Petch", "Avenir Next Condensed", "Arial Narrow", sans-serif;
  --mr-body: "Spline Sans", "Helvetica Neue", Arial, sans-serif;
  --mr-mono: "Spline Sans Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --mr-gutter: 16px;
  --mr-bar-h: 104px;
}
"""

#: Gradio's theme variables, pinned to the tokens in BOTH schemes. Gradio's own
#: sheet loads after this one, hence ``!important`` on every declaration.
_THEME_VARS = {
    "body-background-fill": "var(--mr-ink)",
    "background-fill-primary": "var(--mr-slate)",
    "background-fill-secondary": "var(--mr-slate)",
    "body-text-color": "var(--mr-paper)",
    "body-text-color-subdued": "var(--mr-dust)",
    "block-background-fill": "transparent",
    "block-border-color": "transparent",
    "block-border-width": "0px",
    "block-shadow": "none",
    "block-label-background-fill": "transparent",
    "block-label-text-color": "var(--mr-paper)",
    "block-label-border-color": "transparent",
    "block-title-text-color": "var(--mr-paper)",
    "block-title-background-fill": "transparent",
    "block-info-text-color": "var(--mr-dust)",
    "block-info-text-size": "13px",
    "block-radius": "10px",
    "panel-background-fill": "var(--mr-slate)",
    "panel-border-color": "var(--mr-line)",
    "border-color-primary": "var(--mr-line)",
    "border-color-accent": "var(--mr-paper)",
    "border-color-accent-subdued": "var(--mr-line)",
    "color-accent": "var(--mr-paper)",
    "color-accent-soft": "var(--mr-slate-2)",
    "input-background-fill": "var(--mr-well)",
    "input-background-fill-focus": "var(--mr-well)",
    "input-background-fill-hover": "var(--mr-well)",
    "input-border-color": "var(--mr-line)",
    "input-border-color-focus": "var(--mr-paper)",
    "input-border-color-hover": "var(--mr-line)",
    "input-border-width": "1px",
    "input-placeholder-color": "#8b97a3",
    "input-radius": "6px",
    "input-shadow": "none",
    "input-shadow-focus": "none",
    "input-text-size": "14px",
    "checkbox-background-color": "var(--mr-well)",
    "checkbox-background-color-hover": "var(--mr-well)",
    "checkbox-background-color-focus": "var(--mr-well)",
    "checkbox-background-color-selected": "var(--mr-paper)",
    "checkbox-border-color": "var(--mr-dust)",
    "checkbox-border-color-hover": "var(--mr-paper)",
    "checkbox-border-color-focus": "var(--mr-paper)",
    "checkbox-border-color-selected": "var(--mr-paper)",
    "checkbox-border-radius": "3px",
    "checkbox-label-background-fill": "var(--mr-well)",
    "checkbox-label-background-fill-hover": "var(--mr-slate-2)",
    "checkbox-label-background-fill-selected": "var(--mr-slate-2)",
    "checkbox-label-border-color": "var(--mr-line)",
    "checkbox-label-border-color-hover": "var(--mr-dust)",
    "checkbox-label-border-color-selected": "var(--mr-paper)",
    "checkbox-label-text-color": "var(--mr-paper)",
    "checkbox-label-text-color-selected": "var(--mr-paper)",
    "checkbox-label-shadow": "none",
    "slider-color": "var(--mr-paper)",
    "loader-color": "var(--mr-paper)",
    "button-primary-background-fill": "var(--mr-tungsten)",
    "button-primary-background-fill-hover": "#f6b45c",
    "button-primary-text-color": "var(--mr-on-tungsten)",
    "button-primary-text-color-hover": "var(--mr-on-tungsten)",
    "button-primary-border-color": "var(--mr-tungsten)",
    "button-primary-border-color-hover": "#f6b45c",
    "button-secondary-background-fill": "var(--mr-slate-2)",
    "button-secondary-background-fill-hover": "#2c3742",
    "button-secondary-text-color": "var(--mr-paper)",
    "button-secondary-text-color-hover": "var(--mr-paper)",
    "button-secondary-border-color": "var(--mr-line)",
    "button-secondary-border-color-hover": "var(--mr-dust)",
    "button-cancel-background-fill": "var(--mr-slate-2)",
    "button-cancel-text-color": "var(--mr-paper)",
    "button-border-width": "1px",
    "button-shadow": "none",
    "button-shadow-hover": "none",
    "button-shadow-active": "none",
    "button-large-radius": "6px",
    "button-medium-radius": "6px",
    "button-small-radius": "6px",
    "link-text-color": "var(--mr-paper)",
    "link-text-color-hover": "var(--mr-paper)",
    "link-text-color-visited": "var(--mr-paper)",
    "link-text-color-active": "var(--mr-paper)",
    "code-background-fill": "var(--mr-well)",
    "table-border-color": "var(--mr-line)",
    "table-even-background-fill": "var(--mr-slate)",
    "table-odd-background-fill": "var(--mr-slate)",
    "table-text-color": "var(--mr-paper)",
    "error-background-fill": "color-mix(in srgb, var(--mr-cut) 12%, var(--mr-slate))",
    "error-border-color": "var(--mr-cut)",
    "error-text-color": "var(--mr-cut-text)",
    "error-icon-color": "var(--mr-cut-text)",
    "accordion-text-color": "var(--mr-paper)",
    "stat-background-fill": "var(--mr-paper)",
    "font": "var(--mr-body)",
    "font-mono": "var(--mr-mono)",
    "layout-gap": "12px",
    "form-gap-width": "0px",
    "shadow-drop": "none",
    "shadow-drop-lg": "none",
    "shadow-inset": "none",
}
#: Gradio's primary scale drives sliders, loaders and focus rings. Neutralised
#: so its orange appears nowhere.
_PRIMARY_SCALE = {
    "50": "#f4f6f8",
    "100": "#e9ecef",
    "200": "#d6dbe0",
    "300": "#c0c8d0",
    "400": "#aab4be",
    "500": "#e9ecef",
    "600": "#d6dbe0",
    "700": "#9aa6b2",
    "800": "#39434f",
    "900": "#242d37",
    "950": "#1c232b",
}


def _theme_css() -> str:
    lines = [f"  --{name}: {value} !important;" for name, value in _THEME_VARS.items()]
    lines += [f"  --primary-{step}: {value} !important;" for step, value in _PRIMARY_SCALE.items()]
    body = "\n".join(lines)
    return f":root, :root .dark, .dark, body, gradio-app {{\n{body}\n}}\n"


_SHELL = """
html, body, gradio-app, .gradio-container, .main, .app { background: var(--mr-ink) !important; }
body { color: var(--mr-paper); color-scheme: dark; }
.gradio-container { font-family: var(--mr-body) !important; font-size: 14px; line-height: 21px;
  max-width: none !important; padding: 0 var(--mr-gutter) !important; overflow: visible !important; }
.gradio-container > .main, .gradio-container .main.fillable { padding-left: 0 !important;
  padding-right: 0 !important; overflow: visible !important; }
@media (min-width: 1024px) { :root { --mr-gutter: 32px; } }
.gradio-container .contain { max-width: 1520px; margin: 0 auto; }
footer { color: var(--mr-dust) !important; }
footer a, footer button { color: var(--mr-dust) !important; }
footer img, footer svg { filter: grayscale(1) brightness(1.5); }
footer { flex-wrap: wrap !important; justify-content: center; gap: 4px 14px !important; }
footer a, footer button { white-space: nowrap !important; }
.mr-num { font-family: var(--mr-mono); font-weight: 500; font-variant-numeric: tabular-nums; }

/* one focus ring, everywhere */
:is(button, a, input, textarea, select, summary, [tabindex]):focus-visible {
  outline: 2px solid var(--mr-paper) !important; outline-offset: 2px !important;
  box-shadow: none !important;
}

/* == identity bar ======================================================== */
#identity { margin: 0 !important; padding: 0 !important; }
:is(#identity, #matrix-frame, #cost, #monitor-idle, #run-summary) .html-container { padding: 0 !important; }
.mr-id { display: flex; align-items: center; gap: 16px; min-height: 56px; max-height: 64px;
  padding: 0; position: relative; }
.mr-wordmark { font-family: var(--mr-display); font-weight: 600; font-size: 20px;
  letter-spacing: 0.02em; color: var(--mr-paper); margin: 0; white-space: nowrap; }
.mr-tagline { color: var(--mr-dust); margin: 0; font-size: 14px; flex: 1 1 auto;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.mr-quota { position: static; flex: 0 0 auto; }
.mr-quota > summary { list-style: none; cursor: pointer; color: var(--mr-paper);
  font-size: 14px; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--mr-line);
  background: var(--mr-slate); white-space: nowrap; min-height: 44px; box-sizing: border-box;
  display: flex; align-items: center; gap: 8px; }
.mr-quota > summary::-webkit-details-marker { display: none; }
.mr-quota > summary::after { content: ""; width: 7px; height: 7px; border-right: 1.5px solid
  var(--mr-dust); border-bottom: 1.5px solid var(--mr-dust); transform: rotate(45deg)
  translateY(-2px); }
.mr-quota[open] > summary::after { transform: rotate(-135deg) translateY(-2px); }
.mr-quota-panel { position: absolute; right: 0; top: calc(100% + 6px); z-index: 50;
  width: min(560px, calc(100vw - 2 * var(--mr-gutter))); background: var(--mr-slate);
  border: 1px solid var(--mr-line); border-radius: 10px; padding: 14px 16px; color: var(--mr-paper);
  font-size: 14px; line-height: 21px; }
.mr-quota-panel p { margin: 0; max-width: 70ch; }

/* == the two columns ===================================================== */
#shell { gap: 20px !important; align-items: flex-start !important; flex-wrap: nowrap !important; }
#commission { flex: 1 1 auto !important; min-width: 0 !important; gap: 0 !important;
  background: var(--mr-slate); border: 1px solid var(--mr-line); border-radius: 10px;
  padding: 16px !important; }
#rail { flex: 0 0 clamp(400px, 38vw, 560px) !important; min-width: 0 !important;
  position: sticky; top: 12px; align-self: flex-start; max-height: calc(100vh - 24px);
  overflow-y: auto; overscroll-behavior: contain; gap: 12px !important;
  scrollbar-width: thin; scrollbar-color: var(--mr-line) transparent; padding: 0 2px 4px; }
.mr-divide { border-top: 1px solid var(--mr-line) !important; margin-top: 14px !important;
  padding-top: 14px !important; }

/* blocks inside the commission panel sit flat on the slate */
#commission .block, #rail .block { background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important; }
#commission > .form, #commission .form { background: transparent !important; border: none !important;
  gap: 12px !important; }
#commission .column, #rail .column { gap: 10px !important; }
#commission label > span[data-testid="block-info"], #commission [data-testid="block-info"],
#rail [data-testid="block-info"] { font-family: var(--mr-body); font-weight: 500; font-size: 14px;
  color: var(--mr-paper) !important; margin-bottom: 6px; }
#commission [data-testid="block-label"], #rail [data-testid="block-label"] {
  font-family: var(--mr-body); font-size: 13px; color: var(--mr-dust) !important;
  background: transparent !important; border: none !important; }
#commission .info, #commission [data-testid="block-info"] + .info { color: var(--mr-dust) !important; }
#commission textarea, #commission input[type="text"], #commission input[type="number"] {
  background: var(--mr-well) !important; color: var(--mr-paper) !important;
  border: 1px solid var(--mr-line) !important; border-radius: 6px !important; }
#commission textarea::placeholder, #commission input::placeholder { color: #8b97a3 !important; }
#commission textarea:focus, #commission input:focus { border-color: var(--mr-paper) !important; }
input[type="number"] { font-family: var(--mr-mono) !important; font-variant-numeric: tabular-nums; }
input:disabled, textarea:disabled { background: var(--mr-slate) !important; color: var(--mr-dust) !important;
  border-style: dashed !important; }
input[type="range"] { accent-color: var(--mr-paper); }

/* == the matrix ========================================================== */
#matrix { display: grid !important; grid-template-areas: "stack"; gap: 0 !important; }
#matrix > * { grid-area: stack; min-width: 0; }
#matrix-frame, #mode-matrix { --mx-head: 112px; --mx-colhead: 40px; --mx-cell: 46px; }
.mx-frame, #mode-matrix > div.wrap:not([data-testid]) { display: grid !important;
  grid-template-columns: var(--mx-head) repeat(3, minmax(0, 1fr));
  grid-template-rows: var(--mx-colhead) repeat(5, var(--mx-cell)) auto;
  column-gap: 8px; row-gap: 6px; }
.mx-col { align-self: end; font-family: var(--mr-display); font-weight: 600; font-size: 14px;
  color: var(--mr-paper); padding: 0 2px 8px; border-bottom: 1px solid var(--mr-line); line-height: 18px; }
.mx-col.is-cost { border-bottom: 2px solid var(--mr-tungsten); }
.mx-row { align-self: center; font-family: var(--mr-display); font-weight: 600; font-size: 14px;
  color: var(--mr-paper); line-height: 18px; }
.mx-empty { border-radius: 6px; border: 1px dashed #2c3540;
  background: repeating-linear-gradient(135deg, transparent 0 7px, rgba(57,67,79,0.35) 7px 8px); }
.mx-note { margin: 8px 0 0; font-size: 13px; line-height: 19px; color: var(--mr-dust); }
.mx-note-col { display: none; }
#mode-matrix { padding: 0 !important; pointer-events: none; }
#mode-matrix > [data-testid="block-info"], #mode-matrix .block-title,
#mode-matrix > span:first-child { position: absolute !important; width: 1px; height: 1px;
  overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
#mode-matrix > div.wrap:not([data-testid]) { pointer-events: none; }
#mode-matrix label { pointer-events: auto; position: relative; margin: 0 !important;
  display: flex !important; align-items: center; justify-content: center; min-height: 44px;
  border-radius: 6px; border: 1px solid var(--mr-line) !important; background: var(--mr-well) !important;
  cursor: pointer; transition: background-color 120ms ease, border-color 120ms ease;
  box-shadow: none !important; padding: 0 !important; }
#mode-matrix label > span { position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; margin: 0 !important; }
#mode-matrix label input { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0;
  margin: 0; cursor: pointer; }
#mode-matrix label::before { content: ""; width: 12px; height: 12px; border-radius: 50%;
  border: 1.5px solid var(--mr-dust); background: transparent; transition: background-color 120ms ease; }
#mode-matrix label::after { position: absolute; top: 4px; right: 6px; font-family: var(--mr-mono);
  font-size: 11px; line-height: 12px; color: var(--mr-dust); }
#mode-matrix label:hover { background: var(--mr-slate-2) !important; border-color: var(--mr-dust) !important; }
#mode-matrix label.selected, #mode-matrix label:has(input:checked) {
  background: var(--mr-slate-2) !important; border-color: var(--mr-paper) !important; }
#mode-matrix label.selected::before, #mode-matrix label:has(input:checked)::before {
  background: var(--mr-paper); border-color: var(--mr-paper);
  box-shadow: 0 0 0 4px rgba(233,236,239,0.14); }
#mode-matrix label.selected::after, #mode-matrix label:has(input:checked)::after { color: var(--mr-paper); }
#mode-matrix label:has(input:focus-visible) { outline: 2px solid var(--mr-paper); outline-offset: 2px; }
#purpose { margin-top: 12px !important; }
#purpose p { margin: 0; font-size: 14px; color: var(--mr-dust); max-width: 70ch; }
#purpose strong { color: var(--mr-paper); font-weight: 600; }

/* == examples ============================================================ */
#examples { gap: 8px !important; align-items: center !important; flex-wrap: wrap !important;
  margin-top: 10px !important; }
#examples > button { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
#examples > .mr-example-note { flex: 1 1 240px !important; min-width: 200px !important; }
#examples .mr-example-note p { margin: 0; font-size: 13px; color: var(--mr-dust); }

/* == buttons: one family ================================================= */
#commission button.secondary, #rail button.secondary, .mr-secondary {
  background: var(--mr-slate-2) !important; border: 1px solid var(--mr-line) !important;
  color: var(--mr-paper) !important; font-family: var(--mr-body) !important; font-weight: 500 !important;
  border-radius: 6px !important; min-height: 40px; box-shadow: none !important; }
#commission button.secondary:hover, #rail button.secondary:hover { border-color: var(--mr-dust) !important; }
.mr-quiet, button.mr-quiet, #commission button.mr-quiet, #rail button.mr-quiet {
  background: transparent !important; border: 1px solid transparent !important;
  color: var(--mr-paper) !important; text-decoration: underline; text-underline-offset: 3px;
  text-decoration-color: var(--mr-dust); box-shadow: none !important; font-weight: 500 !important;
  min-height: 40px; padding: 0 8px !important; }
.mr-quiet:hover, #rail button.mr-quiet:hover, #commission button.mr-quiet:hover {
  text-decoration-color: var(--mr-paper); border-color: transparent !important; }
button:disabled { cursor: not-allowed !important; }

/* == media drop zones: recessed, dashed; a roll edge once a file is in ===== */
#commission :is(#in-image, #in-video, #in-audio).block, #commission .mr-drop.block {
  background: var(--mr-well) !important; border: 1px dashed var(--mr-line) !important;
  border-radius: 10px !important; padding: 6px !important; overflow: hidden; }
#commission #in-image.block:has(img), #commission #in-video.block:has(video),
#commission #in-audio.block:has(audio, canvas) {
  border-style: solid !important; border-color: var(--mr-roll) !important; }
#commission :is(#in-image, #in-video, #in-audio) [data-testid="block-label"] {
  background: transparent !important; color: var(--mr-dust) !important; }
#commission :is(#in-image, #in-video, #in-audio) :is(button, .upload-container, .wrap) {
  color: var(--mr-dust) !important; }
#in-audio .upload-container, #in-audio [data-testid="upload"] { min-height: 120px !important; }
.mr-caption p { margin: 4px 0 0 !important; font-size: 13px; line-height: 19px; color: var(--mr-dust); }
#shape label:has(input[value="Match input"]) { font-family: var(--mr-body); font-size: 14px; }

/* == chips (presets, choices) ============================================ */
.mr-chips > div.wrap:not([data-testid]) { gap: 6px !important; }
.mr-chips label { min-height: 40px; padding: 0 12px !important; border-radius: 6px !important;
  border: 1px solid var(--mr-line) !important; background: var(--mr-well) !important;
  color: var(--mr-paper) !important; font-family: var(--mr-mono); font-size: 13px; box-shadow: none !important; }
.mr-chips.mr-words label { font-family: var(--mr-body); font-size: 14px; }
.mr-chips label.selected, .mr-chips label:has(input:checked) { border-color: var(--mr-paper) !important;
  background: var(--mr-slate-2) !important; }
.mr-chips label input[type="radio"] { width: 12px; height: 12px; margin-right: 8px; }
.mr-line p { margin: 0; font-size: 13px; color: var(--mr-dust); }
.mr-line.is-pinned p { color: var(--mr-paper); padding: 10px 12px; border: 1px dashed var(--mr-line);
  border-radius: 6px; background: transparent; }
.mr-legal p::before { content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: var(--mr-roll); margin-right: 8px; vertical-align: middle; }

/* == accordions: the LoRA row and Advanced =============================== */
#commission .mr-acc { border: 1px solid var(--mr-line) !important; border-radius: 10px !important;
  background: transparent !important; padding: 0 !important; margin-top: 4px; }
#commission .mr-acc > button, #commission .mr-acc .label-wrap { padding: 12px 14px !important;
  min-height: 48px; font-family: var(--mr-body) !important; font-weight: 500; font-size: 14px;
  color: var(--mr-paper) !important; }
#commission .mr-acc .label-wrap span { font-family: var(--mr-body) !important; font-size: 14px;
  letter-spacing: 0; text-transform: none; }
#commission .mr-acc > div:not(.label-wrap) { padding: 0 14px 14px !important; }
.mr-bay-head p { font-size: 13px; line-height: 19px; color: var(--mr-dust); margin: 0 0 10px; max-width: 70ch; }
.mr-lora-row { align-items: center !important; gap: 10px !important; }
.mr-lora-name p { margin: 0; font-size: 14px; color: var(--mr-paper); overflow-wrap: anywhere; }
#commission .mr-drop.block .wrap, #commission .mr-drop.block button { min-height: 0 !important; }
.mr-lora-name p::before { content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: var(--mr-roll); margin-right: 8px; vertical-align: middle; }
#lora-verdict h3 { font-family: var(--mr-display); font-size: 16px; color: var(--mr-cut-text) !important; margin: 0 0 6px; }
#lora-verdict h4 { font-family: var(--mr-display); font-size: 16px; color: var(--mr-paper) !important; margin: 0 0 6px; }
#commission #lora-verdict.block:has(h3) { border-left: 4px solid var(--mr-cut) !important; padding: 10px 12px !important;
  background: color-mix(in srgb, var(--mr-cut) 10%, var(--mr-slate)) !important; border-radius: 6px; }
#commission #lora-verdict.block:has(h4) { border-left: 4px solid var(--mr-roll) !important; padding: 10px 12px !important;
  background: color-mix(in srgb, var(--mr-roll) 8%, var(--mr-slate)) !important; border-radius: 6px; }
#lora-verdict p, #lora-verdict td, #lora-verdict li { font-size: 13px; color: var(--mr-paper); }
#lora-verdict blockquote { border-left: 2px solid var(--mr-line); margin: 6px 0 0; padding: 2px 10px;
  color: var(--mr-paper); }
#lora-verdict :is(table, thead, tbody, tr, th, td) { border: none !important;
  background: transparent !important; box-shadow: none !important; }
#lora-verdict table { margin: 6px 0 0; width: auto; }
#lora-verdict td { padding: 2px 16px 2px 0 !important; }
#lora-verdict td:first-child { color: var(--mr-dust); }
#lora-verdict thead { display: none; }
.mr-hint p { margin: 2px 0 0; font-size: 13px; line-height: 19px; color: var(--mr-dust); }
.mr-slot p { margin: 0; font-size: 13px; color: var(--mr-dust); border: 1px dashed var(--mr-line);
  border-radius: 6px; padding: 8px 10px; }

/* == the rail: monitor ==================================================== */
#monitor { position: relative; background: var(--mr-well); border: 1px solid var(--mr-line);
  border-radius: 10px; aspect-ratio: 16 / 9; overflow: hidden; gap: 0 !important; padding: 0 !important; }
#monitor > * { position: absolute !important; inset: 0; margin: 0 !important; }
#monitor .block { height: 100% !important; }
#monitor video { width: 100%; height: 100%; object-fit: contain; background: #000; }
#monitor [data-testid="video"], #monitor .video-container, #monitor .wrap { height: 100%; }
#monitor #audio-out { display: flex !important; align-items: center; padding: 0 24px !important;
  background: var(--mr-well) !important; }
#audio-out > div { width: 100%; }
.mr-monitor-idle { position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 6px; text-align: center; }
.mr-idle-mode { margin: 0; font-family: var(--mr-display); font-weight: 600; font-size: 16px; color: var(--mr-paper); }
.mr-idle-hint { margin: 0; font-size: 13px; color: var(--mr-dust); }
.mr-crop { position: absolute; width: 22px; height: 22px; border-color: var(--mr-line); border-style: solid; }
.mr-crop.tl { top: 14px; left: 14px; border-width: 1.5px 0 0 1.5px; }
.mr-crop.tr { top: 14px; right: 14px; border-width: 1.5px 1.5px 0 0; }
.mr-crop.bl { bottom: 14px; left: 14px; border-width: 0 0 1.5px 1.5px; }
.mr-crop.br { bottom: 14px; right: 14px; border-width: 0 1.5px 1.5px 0; }
#monitor .progress-text, #monitor .meta-text, #monitor .progress-level-inner { color: var(--mr-paper) !important;
  font-family: var(--mr-mono) !important; font-size: 13px !important; }
#monitor .progress-bar { background: var(--mr-paper) !important; }
#monitor .wrap.default, #monitor .wrap.center { background: rgba(14,18,22,0.86) !important; }
#monitor [aria-label="Empty value"] .icon { visibility: hidden; }
#monitor-caption p { margin: 0; font-size: 13px; line-height: 19px; color: var(--mr-dust); }
#monitor-caption strong { color: var(--mr-paper); font-weight: 600; }

/* == the rail: cost panel (the annunciator) ================================ */
.mr-cost { background: var(--mr-slate); border: 1px solid var(--mr-line); border-radius: 10px;
  padding: 14px 16px; }
.mr-cost-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: 8px 12px; margin-bottom: 10px; }
.mr-cost-title { font-family: var(--mr-display); font-weight: 600; font-size: 16px; margin: 0;
  color: var(--mr-paper); }
.mr-lamps { display: flex; flex-wrap: wrap; gap: 6px; }
.mr-lamp { font-family: var(--mr-display); font-weight: 600; font-size: 12px; line-height: 16px;
  letter-spacing: 0.02em; padding: 3px 7px; border-radius: 3px;
  border: 1px solid var(--mr-line); color: var(--mr-dust); white-space: nowrap;
  transition: background-color 160ms ease, color 160ms ease; }
.mr-lamp-cost.is-lit { background: var(--mr-tungsten); border-color: var(--mr-tungsten); color: var(--mr-on-tungsten); }
.mr-lamp-cut.is-lit { background: var(--mr-cut); border-color: var(--mr-cut); color: var(--mr-on-tungsten); }
.mr-lamp-roll.is-lit { background: var(--mr-roll); border-color: var(--mr-roll); color: var(--mr-on-tungsten); }
.mr-cost-grid, .mr-done-grid { display: grid; grid-template-columns: 76px minmax(0, 1fr);
  gap: 6px 12px; margin: 0; }
.mr-cost-grid dt, .mr-done-grid dt { color: var(--mr-dust); font-size: 13px; line-height: 21px; }
.mr-cost-grid dd, .mr-done-grid dd { margin: 0; color: var(--mr-paper); font-size: 14px; line-height: 21px; }
.mr-cost-grid .mr-num { color: var(--mr-paper); }
.mr-cost-grid dt:nth-of-type(3) + dd .mr-num, .mr-cost-grid dt:nth-of-type(4) + dd .mr-num { color: var(--mr-tungsten); }
.mr-sub { color: var(--mr-dust); font-size: 13px; }
.mr-cost-note { margin: 10px 0 0; font-size: 13px; line-height: 19px; color: var(--mr-dust); }
.mr-cost-pending { display: none; margin: 10px 0 0; font-size: 13px; line-height: 19px; color: var(--mr-paper);
  border-top: 1px solid var(--mr-line); padding-top: 8px; }
.mr-wont-run { border-left: 3px solid var(--mr-cut); padding: 4px 0 2px 12px; }
.mr-wont-run-head { margin: 0 0 6px; font-size: 13px; color: var(--mr-dust); }
.mr-wont-run-head .mr-lamp { margin-right: 6px; }
.mr-wont-run ul { margin: 0; padding-left: 18px; }
.mr-wont-run li { color: var(--mr-paper); font-size: 14px; line-height: 21px; }

/* == Generate ============================================================= */
#commit { gap: 6px !important; background: transparent; }
#commit-line { display: none; }
#generate-btn { background: var(--mr-tungsten) !important; color: var(--mr-on-tungsten) !important;
  border: 1px solid var(--mr-tungsten) !important; font-family: var(--mr-display) !important;
  font-weight: 600 !important; font-size: 16px !important; letter-spacing: 0.02em; min-height: 52px;
  border-radius: 6px !important; box-shadow: none !important; }
#generate-btn:hover:not(:disabled) { background: #f6b45c !important; }
#generate-btn:disabled { background: var(--mr-slate-2) !important; color: var(--mr-dust) !important;
  border: 1px dashed var(--mr-line) !important; opacity: 1 !important; }
#generate-btn.is-running:disabled, #generate-btn:disabled[data-running] { border-style: solid !important; }
#generate-reason p { margin: 0; font-size: 13px; line-height: 19px; color: var(--mr-paper); }
#generate-reason:empty, #generate-reason:not(:has(p)) { display: none; }

/* == report =============================================================== */
.mr-done { border: 1px solid var(--mr-line); border-left: 4px solid var(--mr-roll); border-radius: 10px;
  padding: 12px 14px; background: var(--mr-slate); }
.mr-done-head { display: block; margin: 0 0 10px !important; line-height: 22px; }
.mr-prompt-sent { margin-top: 10px; }
.mr-prompt-sent blockquote { margin: 4px 0 0; padding: 8px 10px; border-left: 2px solid var(--mr-line);
  background: var(--mr-well); color: var(--mr-paper); font-size: 13px; line-height: 19px;
  user-select: text; white-space: pre-wrap; }
#report-actions { gap: 8px !important; }
#rail #run-status.block { border: 1px solid var(--mr-line) !important; border-radius: 10px !important;
  padding: 12px 14px !important; background: var(--mr-slate) !important; }
#run-status p { font-size: 14px; line-height: 21px; color: var(--mr-paper); margin: 0 0 6px; max-width: 70ch; }
#run-status .mr-report-head { display: none; }
#run-status .mr-notes p { color: var(--mr-paper); }
#run-status details { margin-top: 6px; }
#run-status summary { cursor: pointer; color: var(--mr-dust); font-size: 13px; }
#run-status details p { color: var(--mr-dust); font-size: 13px; font-family: var(--mr-mono); word-break: break-word; }
#rail #run-status.block:has(h3) { border-color: var(--mr-cut) !important; border-left-width: 4px !important;
  background: color-mix(in srgb, var(--mr-cut) 10%, var(--mr-slate)) !important; }
#run-status h3 { font-family: var(--mr-display); font-weight: 600; font-size: 16px; margin: 0 0 6px;
  color: var(--mr-cut-text) !important; }
#run-status:has(h3) details p { color: var(--mr-paper); }

/* == recent runs ========================================================== */
#recent .mr-recent-head p { margin: 0; font-family: var(--mr-display); font-weight: 600; font-size: 16px; }
.mr-recent-row { align-items: center !important; gap: 6px !important; flex-wrap: wrap !important; }
.mr-recent-row:has(.mr-recent-text:not(.hidden)) { border-top: 1px solid var(--mr-line); padding-top: 6px; }
.mr-recent-row:not(:has(button:not(.hidden))), .mr-lora-row:not(:has(button:not(.hidden))) { display: none !important; }
.mr-recent-row .mr-recent-text p { margin: 0; font-size: 13px; color: var(--mr-paper); }

/* == the matrix cells name themselves wherever there is room ================ */
@media (min-width: 561px) {
  #mode-matrix label { justify-content: flex-start !important; gap: 10px;
    padding: 4px 30px 4px 12px !important; text-align: left; }
  #mode-matrix label::before { flex: 0 0 12px; }
  #mode-matrix label > span { position: static !important; width: auto; height: auto; overflow: visible;
    clip: auto; white-space: normal; font-family: var(--mr-body); font-size: 12px; line-height: 15px;
    color: var(--mr-dust); }
  #mode-matrix label:hover > span, #mode-matrix label.selected > span,
  #mode-matrix label:has(input:checked) > span { color: var(--mr-paper); }
}

/* == below 1024 px: one column and a commit bar ============================ */
@media (max-width: 1023px) {
  #shell { flex-direction: column !important; }
  #commission, #rail { flex: 1 1 auto !important; width: 100% !important; }
  #rail { position: static; max-height: none; overflow: visible; }
  #monitor { order: 2; } #monitor-caption { order: 3; } #cost { order: 1; }
  #run-summary { order: 4; } #report-actions { order: 5; } #run-status { order: 6; } #recent { order: 7; }
  #commit { position: fixed; left: 0; right: 0; bottom: 0; z-index: 40; margin: 0 !important;
    padding: 10px var(--mr-gutter) calc(10px + env(safe-area-inset-bottom)) !important;
    background: var(--mr-ink); border-top: 1px solid var(--mr-line); }
  #commit-line { display: block; }
  #commit-line p { margin: 0; font-size: 13px; line-height: 18px; color: var(--mr-paper);
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  #generate-reason { display: none !important; }
  .mr-cost-pending { display: block; }
  #generate-btn { min-height: 48px; }
  .gradio-container { padding-bottom: calc(var(--mr-bar-h) + env(safe-area-inset-bottom) + 16px) !important; }
  .mr-tagline { display: none; }
  .mr-id { justify-content: space-between; }
  #commission button, #rail button, #commission button.secondary, #rail button.secondary,
  #commission .mr-secondary, footer button, footer a { min-height: 44px !important; }
  #commission button.reset-button, #commission .reset-button { min-width: 44px !important; }
  #monitor button, #monitor a[download] { min-width: 44px !important; min-height: 44px !important;
    display: inline-flex; align-items: center; justify-content: center; }
  footer button, footer a { display: inline-flex !important; align-items: center; }
  .mr-chips label { min-height: 44px; }
}
@media (max-width: 560px) {
  #matrix-frame, #mode-matrix { --mx-head: 82px; --mx-cell: 46px; --mx-colhead: 44px; }
  .mx-col, .mx-row { font-size: 13px; line-height: 16px; }
  .mx-frame { grid-template-rows: var(--mx-colhead) repeat(5, var(--mx-cell)) auto auto; }
  .mx-note { grid-column: 1 / -1 !important; margin: 6px 0 0; }
  .mx-note + .mx-note { grid-row: 8 !important; }
  .mx-note-col { display: inline; color: var(--mr-paper); }
  #commission { padding: 12px !important; }
  .mr-cost-grid, .mr-done-grid { grid-template-columns: 70px minmax(0, 1fr); }
}

/* == restraint ============================================================ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
"""

_SCRIPT = """
(() => {
  const ready = (fn) => {
    const run = () => { try { fn(); } catch (e) { console.warn(e); } };
    const obs = new MutationObserver(run);
    obs.observe(document.body, { childList: true, subtree: true });
    run();
  };
  ready(() => {
    const status = document.getElementById('run-status');
    if (status && !status.dataset.mrLive) {
      status.dataset.mrLive = '1';
      status.setAttribute('role', 'status');
      status.setAttribute('aria-live', 'polite');
      const loud = () => status.setAttribute('aria-live', status.querySelector('h3') ? 'assertive' : 'polite');
      new MutationObserver(loud).observe(status, { childList: true, subtree: true });
      loud();
    }
    const gen = document.getElementById('generate-btn');
    if (gen && !gen.getAttribute('aria-describedby')) {
      gen.setAttribute('aria-describedby', 'cost generate-reason');
    }
    const group = document.querySelector('#mode-matrix > div.wrap:not([data-testid])');
    if (group && group.getAttribute('role') !== 'radiogroup') {
      group.setAttribute('role', 'radiogroup');
      group.setAttribute('aria-label', 'What you start from, and what you get back');
    }
    // One group, one name: a single Tab stop. Arrow keys move by the grid's
    // geometry (up/down/left/right), not by registry order.
    document.querySelectorAll('#mode-matrix input[type="radio"]').forEach((input) => {
      if (input.name !== 'mode-matrix') input.name = 'mode-matrix';
      if (input.dataset.mrKeys) return;
      input.dataset.mrKeys = '1';
      input.addEventListener('keydown', (event) => {
        const step = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[event.key];
        if (!step) return;
        event.preventDefault();
        const centre = (el) => { const r = el.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; };
        const [x, y] = centre(input.closest('label'));
        let best = null, bestScore = Infinity;
        document.querySelectorAll('#mode-matrix input[type="radio"]').forEach((other) => {
          if (other === input) return;
          const [ox, oy] = centre(other.closest('label'));
          const dx = ox - x, dy = oy - y;
          const along = step[0] ? dx * step[0] : dy * step[1];
          const across = step[0] ? Math.abs(dy) : Math.abs(dx);
          if (along <= 4) return;
          const score = along + across * 4;
          if (score < bestScore) { bestScore = score; best = other; }
        });
        if (best) { best.click(); best.focus(); }
      });
    });
  });
})();
"""


def _css() -> str:
    return _TOKENS + _theme_css() + _SHELL + _matrix_css()


def head_html() -> str:
    """Font links, the stylesheet and the accessibility script, for ``gr.HTML(head=...)``."""
    return (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link rel="stylesheet" href="{_FONTS_HREF}">\n'
        f"<style>{_css()}</style>\n"
        f"<script>{_SCRIPT}</script>"
    )


def identity_html(*, quota_text: str) -> str:
    """The one-line identity bar with the "How quota works" disclosure."""
    return (
        '<header class="mr-id">'
        '<h1 class="mr-wordmark">LTX-2.5 Studio</h1>'
        f'<p class="mr-tagline">{html.escape(render.TAGLINE)}</p>'
        '<details class="mr-quota"><summary>How quota works</summary>'
        f'<div class="mr-quota-panel"><p>{html.escape(quota_text)}</p></div></details>'
        "</header>"
    )
