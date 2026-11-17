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


def matrix_frame_html(*, on_zerogpu: bool = True) -> str:
    """Axis headers, the inert empty cells and the column notes, behind the radio.

    The headers and empty cells are drawing (each cell's own name already says what it
    starts from and returns); the column notes are words a screen reader must reach, so
    they stay in the accessibility tree and describe their column's cells (the script
    points each cell's ``aria-describedby`` at its note). The notes are in the platform's
    words (``render.column_notes``).
    """
    cells = matrix_cells()
    notes = render.column_notes(on_zerogpu=on_zerogpu)
    parts = ['<div class="mx-frame">']
    for c, (_key, label) in enumerate(render.COLUMNS, start=1):
        cost = " is-cost" if label == render.COST_COLUMN else ""
        parts.append(
            f'<div class="mx-col{cost}" style="grid-column:{c + 1};grid-row:1" aria-hidden="true">'
            f"{html.escape(label)}</div>"
        )
    for r, (_key, label) in enumerate(render.ROWS, start=1):
        parts.append(
            f'<div class="mx-row" style="grid-row:{r + 1};grid-column:1" aria-hidden="true">'
            f"{html.escape(label)}</div>"
        )
    for r, c in _empty_cells():
        parts.append(
            f'<div class="mx-empty" style="grid-row:{r + 1};grid-column:{c + 1}" '
            'aria-hidden="true"></div>'
        )
    notes_row = len(render.ROWS) + 2
    for c, (_key, label) in enumerate(render.COLUMNS, start=1):
        note = notes.get(label)
        if note:
            column_modes = " ".join(m for m, (_r, col, _n) in cells.items() if col == c)
            parts.append(
                f'<p class="mx-note" id="mx-note-{c}" data-modes="{column_modes}" '
                f'style="grid-row:{notes_row};grid-column:{c + 1}">'
                f'<span class="mx-note-col">{html.escape(label)}:</span> {html.escape(note)}</p>'
            )
    parts.append("</div>")
    return "".join(parts)


def _matrix_css() -> str:
    rules = []
    for mode_id, (row, column, number) in matrix_cells().items():
        sel = f'#{MATRIX_ID} label:has(input[value="{mode_id}"])'
        rules.append(f"{sel} {{ grid-row: {row + 1}; grid-column: {column + 1}; }}")
        # the alternative text "" keeps the number out of the cell's accessible name
        rules.append(f'{sel}::after {{ content: "{number:02d}"; content: "{number:02d}" / ""; }}')
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
  /* control edges: 3.8:1 on the well, 3.2:1 on slate */
  --mr-line-strong: #66727f;
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
  /* the rail's top at scroll 0 plus its bottom margin; the script measures it */
  --mr-rail-offset: 92px;
  --mr-rail-gutter: 10px;
}
"""

#: Gradio paints the checkbox tick and the radio dot white; on the paper fill
#: that is 1.1:1. These are the same glyphs in ink.
_INK_TICK = (
    "url(\"data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='%2312161b' "
    "xmlns='http://www.w3.org/2000/svg'%3e%3cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 "
    "0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3e"
    '%3c/svg%3e")'
)
_INK_DOT = (
    "url(\"data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='%2312161b' "
    "xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='8' cy='8' r='3'/%3e%3c/svg%3e\")"
)

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
    "input-border-color": "var(--mr-line-strong)",
    "input-border-color-focus": "var(--mr-paper)",
    "input-border-color-hover": "var(--mr-dust)",
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
    "checkbox-check": _INK_TICK,
    "radio-circle": _INK_DOT,
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
footer :is(a, button):not(.hidden) { color: var(--mr-dust) !important; white-space: nowrap !important; }
footer img, footer svg { filter: grayscale(1) brightness(1.5); }
footer { flex-wrap: wrap !important; justify-content: center; gap: 4px 14px !important; }
/* Gradio's own run history competes with Recent runs */
footer .run-history, footer .run-history + .divider { display: none !important; }
/* inline figures: the body face with tabular figures; Mono stays for inputs */
.mr-num { font-family: var(--mr-body); font-weight: 500; font-variant-numeric: tabular-nums; }

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
/* Gradio's ".gradio-container-6-24-0 h1" reset (0,1,1) flattens a bare class */
#identity h1.mr-wordmark { font-family: var(--mr-display); font-weight: 600; font-size: 20px;
  line-height: 28px; letter-spacing: 0.02em; color: var(--mr-paper); margin: 0; white-space: nowrap; }
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
/* Gradio gives every Row child flex-wrap: wrap; a column that wraps sends its
   overflow into an off-screen second column instead of scrolling. */
#rail { min-width: 0 !important; gap: 12px !important; flex-wrap: nowrap !important; }
#rail > * { flex-shrink: 0 !important; }
@media (min-width: 1024px) {
  /* Sized to fit the viewport from where it starts (the script measures it, and grows it
     to the viewport less 24 px once the page scrolls and it sticks); scrolls inside itself
     with a real scrollbar, in a gutter outside the aligned edge. The 4 px padding keeps
     focus rings unclipped; the negative margins keep the content edges aligned. */
  #rail { flex: 0 0 clamp(400px, 38vw, 560px) !important; position: sticky; top: 12px;
    align-self: flex-start; max-height: calc(100vh - var(--mr-rail-offset));
    overflow-x: hidden; overflow-y: auto; overscroll-behavior: contain;
    scrollbar-gutter: stable; padding: 4px !important;
    margin: -4px calc(-4px - var(--mr-rail-gutter)) -4px -4px !important; }
  #rail::-webkit-scrollbar { width: var(--mr-rail-gutter); }
  #rail::-webkit-scrollbar-track { background: transparent; }
  #rail::-webkit-scrollbar-thumb { background: var(--mr-line-strong); border-radius: 6px;
    border: 2px solid var(--mr-ink); }
  /* the caption keeps the lines the script reserves (two by default), so Generate does not
     move from mode to mode */
  #rail > #monitor-caption { min-height: var(--mr-caption-min, 38px); }
  /* a short window: the monitor gives up height first (180 px at least), letterboxed */
  #rail.mr-monitor-fit > #monitor { aspect-ratio: auto; height: var(--mr-monitor-h); }
  /* Generate never leaves the rail's visible area: sticky at both edges, flush with the
     rail's own edges, so nothing shows past it */
  #rail > #commit { position: sticky; top: -4px; bottom: -4px; z-index: 3;
    background: var(--mr-ink); padding: 8px 0 8px !important; margin-top: -8px !important; }
  /* a row passing under Generate fades into it, never cut */
  #rail > #commit::before, #rail > #commit::after { content: ""; position: absolute; left: 0;
    right: 0; height: 12px; pointer-events: none; display: none; }
  #rail > #commit::before { top: -12px;
    background: linear-gradient(to bottom, rgba(18, 22, 27, 0), var(--mr-ink)); }
  #rail > #commit::after { bottom: -12px;
    background: linear-gradient(to top, rgba(18, 22, 27, 0), var(--mr-ink)); }
  #rail.mr-under-commit > #commit::before, #rail.mr-over-commit > #commit::after {
    display: block; }
  /* the cost panel runs under Generate: its figure stands above the button */
  #rail.mr-figure-under #commit-line { display: block; }
  #commit-line p { margin: 0; font-size: 13px; line-height: 19px; color: var(--mr-paper); }
  #commit-line .mr-commit-phone { display: none; }
  /* a run's report, which follows Generate in the rail, comes into view clear of it even
     when Generate is stuck at the rail's top */
  #run-summary .mr-done-head, #run-status h3, #run-status h3 + p {
    scroll-margin-top: calc(var(--mr-commit-h, 80px) + 12px); scroll-margin-bottom: 12px; }
}
@supports not selector(::-webkit-scrollbar) {
  #rail { scrollbar-width: thin; scrollbar-color: var(--mr-line-strong) transparent; }
}
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
  border: 1px solid var(--mr-line-strong) !important; border-radius: 6px !important; }
/* the slider's number box sits in a bordered strip of its own */
#commission .tab-like-container { border-color: var(--mr-line-strong) !important; }
#commission .tab-like-container input[type="number"] { border: none !important; border-radius: 0 !important; }
#commission textarea::placeholder, #commission input::placeholder { color: #8b97a3 !important; }
#commission textarea:focus, #commission input:focus { border-color: var(--mr-paper) !important; }
input[type="number"] { font-family: var(--mr-mono) !important; font-variant-numeric: tabular-nums; }
input:disabled, textarea:disabled { background: var(--mr-slate) !important; color: var(--mr-dust) !important;
  border-style: dashed !important; }
input[type="range"] { accent-color: var(--mr-paper); }
/* The slider: Gradio paints a near-white bar (#e4e4e7 track, #fff thumb). Filled
   paper against an unfilled line-strong track is 4.1:1; the track reads 3.2:1
   on slate. */
#commission input[type="range"]::-webkit-slider-runnable-track { height: 4px !important;
  border-radius: 2px !important; background: linear-gradient(to right, var(--mr-paper)
  var(--range_progress, 0%), var(--mr-line-strong) var(--range_progress, 0%)) !important; }
#commission input[type="range"]::-webkit-slider-thumb { width: 16px !important; height: 16px !important;
  margin-top: -6px !important; background-color: var(--mr-paper) !important;
  box-shadow: 0 0 0 2px var(--mr-ink) !important; }
#commission input[type="range"]::-moz-range-track { height: 4px !important; border-radius: 2px;
  background: var(--mr-line-strong) !important; }
#commission input[type="range"]::-moz-range-progress { height: 4px !important;
  background-color: var(--mr-paper) !important; }
#commission input[type="range"]::-moz-range-thumb { background-color: var(--mr-paper) !important;
  box-shadow: 0 0 0 2px var(--mr-ink) !important; }
#commission input[type="range"][disabled]::-webkit-slider-thumb { background-color: var(--mr-dust) !important; }
/* help text at a readable measure */
#commission .info-text, #commission .mr-caption p, #commission .mr-hint p, #commission .mr-line p,
#commission .mr-bay-head p, #commission #lora-verdict p { max-width: 70ch; }

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
/* the jump to the sample output: where the monitor sits far below (one column); and at every
   width while the monitor holds a run from another mode (.mr-over-run), where it is the only
   way to that mode's sample. It stays a quiet text button in the form column, beside the rail. */
#sample-jump { align-self: flex-start; width: auto !important; flex: 0 0 auto !important; }
@media (min-width: 1024px) { #sample-jump { display: none !important; } }
@media (min-width: 1024px) { #sample-jump.mr-over-run { display: flex !important; } }

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
/* Gradio's upload prompt is Title Case ("Drop Image Here - or - Click to Upload"):
   its words are hidden (and so dropped from the accessibility tree) and ours shown */
#commission :is(#in-image, #in-video, #in-audio) .wrap:has(> .or) { font-size: 0 !important;
  visibility: hidden; }
#commission :is(#in-image, #in-video, #in-audio) .wrap:has(> .or) > :first-child { visibility: visible; }
#commission :is(#in-image, #in-video, #in-audio) .wrap:has(> .or)::after { visibility: visible;
  font-size: 14px; line-height: 21px; color: var(--mr-dust); text-align: center; max-width: 32ch;
  text-wrap: balance; }
/* the empty audio zone is as tall as the image and video ones (200 px with its source bar):
   Gradio gives its upload area a 240 px minimum */
#commission #in-audio .audio-container > button { height: 148px !important; min-height: 0 !important; }
#commission #in-audio .audio-container > button .wrap { min-height: 0 !important; }
#commission #in-image .wrap:has(> .or)::after { content: "Drop an image here, or click to choose one"; }
#commission #in-video .wrap:has(> .or)::after { content: "Drop a video here, or click to choose one"; }
#commission #in-audio .wrap:has(> .or)::after {
  content: "Drop an audio file here, or click to choose one"; }
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
/* a choice that cannot be changed now (the decoder under Upscale) is drawn locked */
.mr-chips label:has(input:disabled) { background: transparent !important; border-style: dashed !important;
  color: var(--mr-dust) !important; cursor: not-allowed; }
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
  color: var(--mr-paper) !important; text-align: left !important; align-items: center; gap: 12px; }
#commission .mr-acc .label-wrap span { font-family: var(--mr-body) !important; font-size: 14px;
  letter-spacing: 0; text-transform: none; }
/* Gradio's filled triangle becomes the same chevron as "How quota works" */
#commission .mr-acc .label-wrap .icon { font-size: 0 !important; flex: 0 0 16px; width: 16px; height: 16px;
  display: inline-flex; align-items: center; justify-content: center; transform: none !important; }
#commission .mr-acc .label-wrap .icon::after { content: ""; width: 7px; height: 7px;
  border-right: 1.5px solid var(--mr-dust); border-bottom: 1.5px solid var(--mr-dust);
  transform: rotate(45deg) translateY(-2px); }
#commission .mr-acc .label-wrap.open .icon::after { transform: rotate(-135deg) translateY(-2px); }
#commission .mr-acc > div:not(.label-wrap) { padding: 0 14px 14px !important; }
.mr-bay-head p { font-size: 13px; line-height: 19px; color: var(--mr-dust); margin: 0 0 10px; max-width: 70ch; }
/* the Hub box and its button share a baseline: the button sits level with the input */
.mr-hub-row { align-items: flex-end !important; }
.mr-lora-row { align-items: center !important; gap: 10px !important; }
.mr-lora-name p { margin: 0; font-size: 14px; color: var(--mr-paper); overflow-wrap: break-word; }
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
#monitor video { width: 100%; height: 100%; object-fit: contain; background: var(--mr-well); }
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
/* the player's toolbar is a quiet button in the corner, not a bordered strip */
#monitor :is(.icon-button-wrapper, .icon-buttons) { background: transparent !important;
  border: none !important; box-shadow: none !important; }
#monitor :is(.icon-button-wrapper, .icon-buttons) :is(button, a) { border: none !important;
  background: rgba(14, 18, 22, 0.72) !important; border-radius: 6px !important;
  color: var(--mr-paper) !important; box-shadow: none !important; }
/* the download corner sits inside the clipped edge, so its focus ring is whole */
#video-out .icon-button-wrapper { top: 8px !important; right: 8px !important; }
/* the player's bar: Gradio shows it to a hovering pointer only. It also shows to keyboard
   focus inside the player, while the video is paused (the script sets .mr-paused from the
   video's own events) and on a touch screen; in the well colour, not Gradio's grey */
#video-out .wrap > .controls { background: rgba(14, 18, 22, 0.86) !important; }
#video-out .wrap:has(:focus-visible) > .controls, #video-out .wrap.mr-paused > .controls {
  opacity: 1 !important; }
@media (hover: none) {
  #video-out .wrap > .controls { opacity: 1 !important; }
}
#monitor-caption p { margin: 0; font-size: 13px; line-height: 19px; color: var(--mr-dust); }
#monitor-caption strong { color: var(--mr-paper); font-weight: 600; }

/* == the rail: cost panel (the annunciator) ================================ */
.mr-cost { background: var(--mr-slate); border: 1px solid var(--mr-line); border-radius: 10px;
  padding: 14px 16px; }
.mr-cost-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: 8px 12px; margin-bottom: 10px; }
#cost h2.mr-cost-title { font-family: var(--mr-display); font-weight: 600; font-size: 16px;
  line-height: 22px; margin: 0; color: var(--mr-paper); }
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
/* the reservation and the quota needed, on ZeroGPU: the only figures in tungsten */
.mr-cost-grid .mr-num.mr-cost-figure { color: var(--mr-tungsten); font-family: var(--mr-mono); }
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
/* Reuse seed and Use this prompt belong to the Done card: the actions row closes it */
#report-actions { gap: 8px !important; }
#run-summary:has(+ #report-actions button:not(.hidden)) .mr-done {
  border-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0; padding-bottom: 4px; }
/* an empty actions row or an empty Recent runs takes no room: each was a 12 px gap in the
   rail, and on a short window a scrollbar over nothing */
#rail > :is(#report-actions, #recent):not(:has(button:not(.hidden))) { display: none !important; }
#report-actions:has(button:not(.hidden)) { margin-top: -12px !important; padding: 2px 8px 8px !important;
  border: 1px solid var(--mr-line); border-top: 0; border-left: 4px solid var(--mr-roll);
  border-radius: 0 0 10px 10px; background: var(--mr-slate); }
#rail #run-status.block { border: 1px solid var(--mr-line) !important; border-radius: 10px !important;
  padding: 12px 14px !important; background: var(--mr-slate) !important; }
#run-status p { font-size: 14px; line-height: 21px; color: var(--mr-paper); margin: 0 0 6px; max-width: 70ch; }
/* read by assistive tech (the live region's "Done" line), not shown: the Done card is */
#run-status .mr-report-head, #rail #run-status.block:not(:has(p, h3, details, li)) {
  position: absolute !important; width: 1px !important; height: 1px !important;
  padding: 0 !important; margin: -1px !important; overflow: hidden !important;
  clip: rect(0 0 0 0) !important; clip-path: inset(50%) !important; white-space: nowrap !important;
  border: 0 !important; }
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
  #monitor { order: 2; } #monitor-caption { order: 3; } #cost { order: 1; }
  #run-summary { order: 4; } #report-actions { order: 5; } #run-status { order: 6; } #recent { order: 7; }
  #commit { position: fixed; left: 0; right: 0; bottom: 0; z-index: 40; margin: 0 !important;
    padding: 10px var(--mr-gutter) calc(10px + env(safe-area-inset-bottom)) !important;
    background: var(--mr-ink); border-top: 1px solid var(--mr-line); }
  #commit-line { display: block; }
  #commit-line p { margin: 0; font-size: 13px; line-height: 18px; color: var(--mr-paper);
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  #commit-line .mr-commit-desk { display: none; }
  #generate-reason { display: none !important; }
  .mr-cost-pending { display: block; }
  .gradio-container { padding-bottom: calc(var(--mr-bar-h) + env(safe-area-inset-bottom) + 16px) !important; }
  .mr-id { justify-content: space-between; }
  #commission button, #rail button, #commission button.secondary, #rail button.secondary,
  #commission .mr-secondary { min-height: 44px !important; }
  #rail #generate-btn { min-height: 48px !important; }
  /* Gradio's slider number strip is 24 px tall with overflow hidden, which clipped a
     44 px reset button; the strip grows instead (46 = 44 + its two borders). */
  #commission .tab-like-container { height: 46px !important; }
  #commission button.reset-button { min-width: 44px !important; min-height: 44px !important;
    height: 44px !important; }
  #monitor button, #monitor a[download], #monitor .controls [role="button"] {
    min-width: 44px !important; min-height: 44px !important;
    display: inline-flex; align-items: center; justify-content: center; }
  /* the player's glyphs keep their size inside those targets */
  #monitor .controls :is(button, [role="button"]) svg { width: 20px !important; height: 20px !important; }
  /* 44 px targets for the footer's links; Gradio's run history stays hidden here too */
  footer :is(button, a):not(.hidden):not(.run-history) { display: inline-flex !important;
    align-items: center; min-height: 44px !important; }
  footer .divider { display: none !important; }
  .mr-chips label { min-height: 44px; }
  /* Gradio drops a slider's minimum and maximum labels at 480 px and below */
  #commission .min_value, #commission .max_value { display: inline-block !important; }
}
@media (max-width: 760px) { .mr-tagline { display: none; } }
@media (max-width: 560px) {
  #matrix-frame, #mode-matrix { --mx-head: 82px; --mx-cell: 46px; --mx-colhead: 44px; }
  .mx-col, .mx-row { font-size: 13px; line-height: 16px; }
  .mx-frame { grid-template-rows: var(--mx-colhead) repeat(5, var(--mx-cell)) auto auto; }
  .mx-note { grid-column: 1 / -1 !important; margin: 6px 0 0; }
  .mx-note + .mx-note { grid-row: 8 !important; }
  .mx-note-col { display: inline; color: var(--mr-paper); }
  #commission { padding: 12px !important; }
  .mr-cost-grid, .mr-done-grid { grid-template-columns: 70px minmax(0, 1fr); }
  /* a staged adapter stacks: its name on one line, the strength under it, then Remove */
  .mr-lora-row { flex-wrap: wrap !important; row-gap: 4px !important; }
  .mr-lora-row > * { flex: 1 1 100% !important; min-width: 0 !important; }
  .mr-lora-row > button { flex: 0 0 auto !important; }
  /* a recent run's line takes the full width, its buttons wrap under it */
  .mr-recent-row > .mr-recent-text { flex: 1 1 100% !important; min-width: 0 !important; }
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
  const byId = (id) => document.getElementById(id);
  const wide = () => window.matchMedia('(min-width: 1024px)').matches;
  const still = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const px = (value) => parseFloat(value) || 0;

  // == the desktop rail ====================================================
  // Its home is where the columns start: at page scroll 0 it ends 12 px above the
  // viewport's bottom; once the page scrolls and it sticks at 12 px it grows to the
  // viewport less 24 px, so no dead band is left under it. At the page's end it stops at
  // its row's bottom (the footer is below), so the row never pushes its top out of view.
  const home = () => {
    const column = byId('commission');
    return column ? Math.round(column.getBoundingClientRect().top + window.scrollY) - 4 : null;
  };
  const placeRail = () => {
    const top = home(), shell = byId('shell');
    if (top === null) return;
    const now = Math.max(12, top - window.scrollY);
    const floor = window.innerHeight - 12;
    const end = shell ? Math.min(floor, Math.floor(shell.getBoundingClientRect().bottom)) : floor;
    document.documentElement.style.setProperty(
      '--mr-rail-offset', (now + window.innerHeight - end) + 'px');
  };
  // At rail scroll 0 the rail holds the monitor, its caption, the cost panel and Generate.
  // Taller than the rail (a short window): the monitor gives up height first, down to
  // 180 px, letterboxed; whatever of the panel still runs under Generate fades into it, and
  // the figure line (figureLine, below) stands above the button if that hides the figure.
  // With room to spare, the caption keeps four lines less the panel's notes, so Generate
  // stays put from mode to mode.
  const MONITOR_FLOOR = 180;
  const CAPTION_LINES = 4;
  const fit = () => {
    const rail = byId('rail'), monitor = byId('monitor'), caption = byId('monitor-caption');
    const cost = byId('cost'), commit = byId('commit'), line = byId('commit-line');
    if (!rail || !monitor || !caption || !cost || !commit) return;
    if (!wide()) {
      rail.classList.remove('mr-monitor-fit', 'mr-figure-under');
      rail.style.removeProperty('--mr-caption-min');
      return;
    }
    const top = home();
    if (top === null) return;
    const rs = getComputedStyle(rail), ks = getComputedStyle(commit);
    const gap = px(rs.rowGap) || 12;
    const avail = window.innerHeight - 12 - top - px(rs.paddingTop) - px(rs.paddingBottom);
    const natural = monitor.offsetWidth * 9 / 16;
    const lines = [...caption.querySelectorAll('p')].filter((p) => p.offsetHeight > 0);
    const captionH = lines.length
      ? lines[lines.length - 1].getBoundingClientRect().bottom - lines[0].getBoundingClientRect().top
      : 0;
    const lineH = lines.length ? px(getComputedStyle(lines[0]).lineHeight) || 19 : 19;
    const notesH = [...cost.querySelectorAll('.mr-cost-note')]
      .reduce((sum, note) => sum + note.offsetHeight + px(getComputedStyle(note).marginTop), 0);
    const lineShown = rail.classList.contains('mr-figure-under') && !!line && line.offsetHeight > 0;
    const commitH = commit.offsetHeight - (lineShown ? line.offsetHeight + px(ks.rowGap) : 0);
    const above = (m, c) => m + gap + c + gap + cost.offsetHeight + gap + px(ks.marginTop) + commitH;
    let monitorH = natural, captionMin = captionH;
    const over = above(natural, captionH) - avail;
    if (over > 0) monitorH = Math.max(MONITOR_FLOOR, natural - over);
    else captionMin = captionH + Math.min(-over, Math.max(0, CAPTION_LINES * lineH - notesH - captionH));
    rail.classList.toggle('mr-monitor-fit', monitorH < natural - 0.5);
    rail.style.setProperty('--mr-monitor-h', Math.round(monitorH) + 'px');
    rail.style.setProperty('--mr-caption-min', Math.round(captionMin) + 'px');
    rail.style.setProperty('--mr-commit-h', commit.offsetHeight + 'px');
  };
  // The figure line stands above Generate whenever the panel's figure row (Needs on ZeroGPU,
  // Runs elsewhere: render.cost_panel's labels) is not wholly in view clear of Generate:
  // scrolled above the rail's visible top, under Generate, or below the rail. Judged from
  // Generate's box without the line, and a shown line goes only once the row is clear by a few
  // px, so neither the line's own height nor a small scroll makes it flicker. shade() runs it
  // on every rail and page scroll and after every fit().
  const FIGURE_ROWS = ['Needs', 'Runs'];
  const FIGURE_SLACK = 6;
  const figureLine = (rail, commit, cost) => {
    const line = byId('commit-line');
    const shown = rail.classList.contains('mr-figure-under');
    const figure = !!line && (line.querySelector('.mr-commit-desk')?.textContent || '').trim() !== '';
    const label = [...cost.querySelectorAll('.mr-cost-grid dt')]
      .find((dt) => FIGURE_ROWS.includes(dt.textContent.trim()));
    if (!figure || !label) { rail.classList.remove('mr-figure-under'); return; }
    const value = label.nextElementSibling || label;
    const a = label.getBoundingClientRect(), b = value.getBoundingClientRect();
    const top = Math.min(a.top, b.top), bottom = Math.max(a.bottom, b.bottom);
    const r = rail.getBoundingClientRect(), k = commit.getBoundingClientRect();
    const lineH = shown && line.offsetHeight > 0 ? line.offsetHeight + px(getComputedStyle(commit).rowGap) : 0;
    // stuck at the rail's foot, Generate grows upward by the line; anywhere else, downward
    const footed = k.bottom >= r.bottom - 6;
    const kTop = footed ? k.top + lineH : k.top, kBottom = footed ? k.bottom : k.bottom - lineH;
    const slack = shown ? FIGURE_SLACK : 0;
    const clear = top >= r.top + slack && bottom <= r.bottom - slack
      && (bottom <= kTop - slack || top >= kBottom + slack);
    const want = !clear;
    if (want === shown) return;
    const button = byId('generate-btn');
    const before = button ? button.getBoundingClientRect().top : 0;
    rail.classList.toggle('mr-figure-under', want);
    // Off the rail's foot an arriving line grows Generate downward, pushing what follows it: the
    // rail scrolls by as much, so the button and the report under it stay where they were and a
    // rail scrolled to its end stays at its end. (A leaving line gets no scroll: that would cut
    // short a smooth scroll back up to the panel or the monitor.)
    if (want && !footed && button) rail.scrollTop += button.getBoundingClientRect().top - before;
  };
  // A row under Generate's top edge (the panel, at a short window) or under its bottom
  // edge (the report, with Generate stuck at the top) fades into it.
  const shade = () => {
    const rail = byId('rail'), commit = byId('commit'), cost = byId('cost');
    if (!rail || !commit || !cost) return;
    if (!wide()) {
      rail.classList.remove('mr-under-commit', 'mr-over-commit');
      return;
    }
    figureLine(rail, commit, cost);
    const edge = commit.getBoundingClientRect();
    rail.classList.toggle('mr-under-commit', cost.getBoundingClientRect().bottom > edge.top + 0.5);
    // the next row in the rail's flow (not the visually hidden live region)
    let next = commit.nextElementSibling;
    while (next && (next.offsetHeight < 2 || getComputedStyle(next).position === 'absolute')) {
      next = next.nextElementSibling;
    }
    rail.classList.toggle('mr-over-commit', !!next && next.getBoundingClientRect().top < edge.bottom - 0.5);
  };
  // When a run ends on desktop, the head of its report comes into view just clear of
  // Generate: the Done card's head, or a failure's headline and next step.
  let lastReport = '';
  const follow = () => {
    const summary = byId('run-summary'), status = byId('run-status');
    const head = status ? status.querySelector('h3') : null;
    let key = '', target = null;
    if (summary && summary.offsetHeight > 0 && summary.textContent.trim()) {
      key = 'done:' + summary.textContent.trim().slice(0, 300);
      target = summary.querySelector('.mr-done-head') || summary;
    } else if (head) {
      key = 'failed:' + status.textContent.trim().slice(0, 300);
      const step = head.nextElementSibling;
      target = step && step.tagName === 'P' ? step : head;
    }
    if (key === lastReport) return;
    lastReport = key;
    if (target && wide()) target.scrollIntoView({ block: 'nearest', behavior: still() ? 'auto' : 'smooth' });
  };

  // == the monitor's player ================================================
  // Gradio's bar shows to a hovering pointer only, names Play "play-pause-replay-button" and
  // toggles playback on any key pressed on it, Tab included. The stylesheet also shows the bar
  // while the video is paused (.mr-paused, set here from the video's own events); here the
  // controls get their names, and only Enter and Space act on them.
  const playerState = (wrap, video) => {
    const paused = video.paused || video.ended;
    wrap.classList.toggle('mr-paused', paused);
    const play = wrap.querySelector(':scope > .controls .inner > [role="button"]:first-child');
    if (play) {
      play.dataset.mrControl = 'play';
      const name = paused ? 'Play' : 'Pause';
      if (play.getAttribute('aria-label') !== name) play.setAttribute('aria-label', name);
    }
    wrap.querySelectorAll(':scope > .controls [aria-label="full-screen"]').forEach((full) => {
      full.dataset.mrControl = 'full-screen';
      full.setAttribute('aria-label', 'Full screen');
    });
  };
  // A new file on the monitor is a new player: each video is wired once.
  const players = () => {
    document.querySelectorAll('#video-out video').forEach((video) => {
      const wrap = video.closest('.wrap');
      if (!wrap) return;
      if (!video.dataset.mrPlayer) {
        video.dataset.mrPlayer = '1';
        ['play', 'pause', 'ended', 'emptied', 'loadedmetadata'].forEach((type) =>
          video.addEventListener(type, () => playerState(wrap, video)));
      }
      playerState(wrap, video);
    });
  };
  // Runs before Gradio's handlers (its Play's delegated keydown, its full screen's keypress):
  // any other key never reaches them, and Space never scrolls the rail or the page. A cancelled
  // keydown sends no keypress, so Space presses full screen here.
  const playerKeys = (event) => {
    const control = event.target instanceof Element
      ? event.target.closest('#video-out [data-mr-control]') : null;
    if (!control) return;
    const acts = (event.key === 'Enter' || event.key === ' ') && !event.repeat;
    if (!acts) {
      event.stopPropagation();
      if (event.key === ' ') event.preventDefault();
    } else if (event.key === ' ' && event.type === 'keydown') {
      event.preventDefault();
      if (control.dataset.mrControl === 'full-screen') control.click();
    }
  };
  document.addEventListener('keydown', playerKeys, true);
  document.addEventListener('keypress', playerKeys, true);

  // Restore settings rewrites the form, and so the cost: on desktop the cost panel comes into
  // view in the rail (the button's own script brings in the purpose line, already in view
  // there). The button is found by the words blocks.py gives it.
  const RESTORE = 'Restore settings';
  document.addEventListener('click', (event) => {
    const button = event.target instanceof Element ? event.target.closest('#recent button') : null;
    if (!button || !wide() || button.textContent.trim() !== RESTORE) return;
    byId('cost')?.scrollIntoView({ block: 'nearest', behavior: still() ? 'auto' : 'smooth' });
  });

  let queued = false;
  const settle = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      try { placeRail(); fit(); shade(); follow(); } catch (e) { console.warn(e); }
    });
  };
  window.addEventListener('resize', settle);
  window.addEventListener('scroll', () => { placeRail(); shade(); }, { passive: true });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(settle);
  ready(() => {
    settle();
    players();
    const rail = byId('rail');
    if (rail && !rail.dataset.mrFit) {
      rail.dataset.mrFit = '1';
      rail.addEventListener('scroll', shade, { passive: true });
      if (window.ResizeObserver) {
        const watch = new ResizeObserver(settle);
        ['rail', 'monitor', 'monitor-caption', 'cost', 'commit'].forEach((id) => byId(id) && watch.observe(byId(id)));
      }
    }
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
    document.querySelectorAll('.mx-note[data-modes]').forEach((note) => {
      note.dataset.modes.split(' ').forEach((mode) => {
        const input = document.querySelector(`#mode-matrix input[value="${mode}"]`);
        if (input && input.getAttribute('aria-describedby') !== note.id) {
          input.setAttribute('aria-describedby', note.id);
        }
      });
    });
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


def _control_marks_css() -> str:
    """The ink tick and dot, set where they are painted: Gradio's preflight sheet draws
    its white ones directly at (0,3,0), past the theme variable its component reads."""
    return (
        f'#commission input[type="checkbox"]:checked {{ background-image: {_INK_TICK} !important; }}\n'
        f'#commission input[type="radio"]:checked {{ background-image: {_INK_DOT} !important; }}\n'
    )


def _css() -> str:
    return _TOKENS + _theme_css() + _SHELL + _matrix_css() + _control_marks_css()


def head_html() -> str:
    """Font links, the stylesheet and the accessibility script, for ``gr.HTML(head=...)``."""
    return (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link rel="stylesheet" href="{_FONTS_HREF}">\n'
        f"<style>{_css()}</style>\n"
        f"<script>{_SCRIPT}</script>"
    )


def identity_html(*, quota_text: str | None) -> str:
    """The one-line identity bar, with the "How quota works" disclosure on ZeroGPU only
    (``quota_text`` is None anywhere else: there is no quota to explain)."""
    disclosure = (
        '<details class="mr-quota"><summary>How quota works</summary>'
        f'<div class="mr-quota-panel"><p>{html.escape(quota_text)}</p></div></details>'
        if quota_text
        else ""
    )
    return (
        '<header class="mr-id">'
        '<h1 class="mr-wordmark">LTX-2.5 Studio</h1>'
        f'<p class="mr-tagline">{html.escape(render.TAGLINE)}</p>'
        f"{disclosure}</header>"
    )
