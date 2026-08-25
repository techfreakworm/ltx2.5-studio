"""The visual shell: design tokens, stylesheet and masthead for the Studio.

This module is the design system for the UI — "the machine room". The direction,
every deliberate choice and everything rejected is written up in
``docs/UI_DESIGN.md``; the short version:

* **Dark slate surfaces** because this is a footage-viewing tool, with a
  blue-gray material palette rather than neutral black;
* **three semantic signals** — tungsten amber for *cost*, cut red for
  *failure*, roll green for *accepted* — never used decoratively;
* **Chakra Petch** for equipment lettering, **Spline Sans** for prose,
  **Spline Sans Mono** for every number a user reads.

**Why CSS is assembled here in Python.** The set of modes that carry a cost
warning is registry data (``modes.cost_note``), and the mode board paints those
cards with a tungsten edge. Writing that selector list by hand in a static
stylesheet would be a second copy of the registry that drifts the first time a
mode is added — so the selector is generated from ``modes.MODES`` at build time,
the same way ``blocks._mode_choices`` derives its ⚠ markers. A new warned mode
gets a tungsten edge by adding a registry entry, never by editing CSS.

**Why the stylesheet rides on ``gr.HTML(head=...)``.** Gradio 6 moved ``css=``
and ``theme=`` from the ``Blocks`` constructor to ``launch()`` (passing them to
``Blocks`` warns as a deprecated 5.x idiom), and ``app.py`` — which owns the
bare ``demo.launch()`` — is not this layer's file to edit. ``gr.HTML`` documents
``head=`` as "a raw HTML string to inject into the document ``<head>``", which
is exactly what a stylesheet and two font links are. The masthead component
carries it, so the shell and the styles arrive together or not at all.

Imports nothing but the registry: no Gradio, no torch. Every function returns a
string, so the whole shell can be asserted on from a bare interpreter.
"""

from __future__ import annotations

import html
from collections.abc import Iterable

from .. import modes

__all__ = ["head_html", "masthead_html"]


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------

#: Three roles, loaded once. Every family gets a real fallback stack in the CSS,
#: so an offline or font-blocked browser degrades to system faces, not to blank.
_FONTS_HREF = (
    "https://fonts.googleapis.com/css2"
    "?family=Chakra+Petch:wght@500;600;700"
    "&family=Spline+Sans:wght@400;500;600"
    "&family=Spline+Sans+Mono:wght@400;500"
    "&display=swap"
)


# ---------------------------------------------------------------------------
# the registry-derived selector
# ---------------------------------------------------------------------------


def _cost_marked_selector() -> str:
    """One CSS selector matching every mode card whose mode carries a cost note.

    Gradio renders each Radio choice as a ``<label>`` wrapping an
    ``<input type="radio" value="<mode id>">``, so ``:has(input[value=...])``
    addresses a card by the same id the registry uses. Derived, not written:
    see the module docstring.
    """
    warned = [mode_id for mode_id, spec in modes.MODES.items() if spec.cost_note]
    return ", ".join(f'#mode-board label:has(input[value="{mode_id}"])' for mode_id in warned)


# ---------------------------------------------------------------------------
# the stylesheet
# ---------------------------------------------------------------------------


def _css() -> str:
    """The machine-room stylesheet. Tokens first, then one block per surface."""
    cost_cards = _cost_marked_selector()
    return f"""
/* == tokens ============================================================== */
:root {{
  --mr-ink: #12161b;        /* page-identity ground */
  --mr-slate: #1c232b;      /* console panel */
  --mr-slate-2: #242d37;    /* raised panel / hover */
  --mr-line: #39434f;       /* gasket hairline */
  --mr-paper: #e9ecef;      /* primary text on slate */
  --mr-dust: #9aa6b2;       /* secondary text on slate */
  --mr-tungsten: #f2a33c;   /* cost signal */
  --mr-cut: #e5484d;        /* failure signal */
  --mr-roll: #3fbf7f;       /* accepted signal */
  --mr-display: "Chakra Petch", "Avenir Next Condensed", "Arial Narrow", sans-serif;
  --mr-body: "Spline Sans", "Helvetica Neue", Arial, sans-serif;
  --mr-mono: "Spline Sans Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}

/* == base type =========================================================== */
.gradio-container {{ font-family: var(--mr-body); }}

/* == masthead ============================================================ */
/* The identity surfaces are self-colored -- the same slate in both Gradio
   themes, like hardware on a desk -- so every text node inside them carries an
   EXPLICIT color. Gradio's theme stylesheets load after this one and win ties,
   and in light mode their inherited text color is near-black: relying on
   inheritance here painted the wordmark dark-on-dark. */
.mr-masthead {{
  background: var(--mr-ink);
  border: 1px solid var(--mr-line);
  border-radius: 10px;
  padding: 18px 22px 12px;
  color: var(--mr-paper);
  font-family: var(--mr-body);
}}
.mr-wordmark {{
  font-family: var(--mr-display);
  font-weight: 700;
  font-size: 1.5rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin: 0;
  color: var(--mr-paper) !important;
}}
.mr-wordmark .mr-model {{ color: var(--mr-tungsten) !important; }}
.mr-tagline {{
  color: var(--mr-dust) !important;
  font-size: 0.85rem;
  margin: 4px 0 12px;
}}
.mr-fact {{
  display: flex;
  align-items: baseline;
  gap: 12px;
  border-top: 1px dashed var(--mr-line);
  padding: 7px 0;
  font-size: 0.82rem;
  line-height: 1.45;
}}
.mr-fact > span:last-child {{ color: var(--mr-dust) !important; }}
.mr-fact-lamp {{
  flex: 0 0 auto;
  font-family: var(--mr-display);
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--mr-tungsten) !important;
  border: 1px solid var(--mr-tungsten);
  border-radius: 3px;
  padding: 2px 8px;
  white-space: nowrap;
}}

/* == column eyebrows ===================================================== */
.mr-eyebrow, .mr-eyebrow p {{
  font-family: var(--mr-display) !important;
  font-size: 0.72rem !important;
  font-weight: 600;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--mr-dust) !important;
  margin: 2px 0 !important;
}}

/* == mode board ========================================================== */
#mode-board {{
  background: var(--mr-slate);
  border: 1px solid var(--mr-line);
  border-radius: 10px;
  padding: 12px;
}}
/* The component's own label and info line sit on the slate too. */
#mode-board [data-testid="block-info"] {{ color: var(--mr-paper) !important; }}
#mode-board .info-text {{ color: var(--mr-dust) !important; }}
#mode-board .wrap {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(158px, 1fr));
  gap: 8px;
}}
#mode-board label {{
  display: flex;
  align-items: center;
  margin: 0;
  background: var(--mr-slate);
  border: 1px solid var(--mr-line);
  border-radius: 7px;
  padding: 10px 12px;
  font-family: var(--mr-display);
  font-size: 0.82rem;
  font-weight: 500;
  line-height: 1.25;
  color: var(--mr-dust);
  cursor: pointer;
  transition: border-color 120ms ease, background 120ms ease, color 120ms ease;
}}
#mode-board label:hover {{ background: var(--mr-slate-2); color: var(--mr-paper); }}
#mode-board label.selected {{
  background: var(--mr-slate-2);
  border-color: var(--mr-paper);
  color: var(--mr-paper);
}}
#mode-board input[type="radio"] {{
  position: absolute;
  opacity: 0;
  width: 1px;
  height: 1px;
}}
#mode-board label:has(input:focus-visible) {{
  outline: 2px solid var(--mr-paper);
  outline-offset: 2px;
}}
/* Cards for modes that carry a registry cost note: a tungsten edge, generated
   from modes.MODES — see _cost_marked_selector(). */
{cost_cards} {{
  border-left: 3px solid var(--mr-tungsten);
}}

/* == mode briefing ======================================================= */
.mr-briefing {{
  background: var(--mr-slate);
  border: 1px solid var(--mr-line);
  border-radius: 10px;
  padding: 12px 16px;
  color: var(--mr-paper);
}}
.mr-briefing h3 {{
  font-family: var(--mr-display);
  font-size: 1rem;
  letter-spacing: 0.04em;
  margin: 0 0 4px;
  color: var(--mr-paper) !important;
}}
.mr-briefing :is(p, li) {{ color: var(--mr-dust) !important; font-size: 0.85rem; }}
.mr-briefing strong {{ color: var(--mr-paper) !important; }}
.mr-briefing blockquote {{
  border-left: 3px solid var(--mr-tungsten);
  margin: 8px 0;
  padding: 4px 12px;
  background: color-mix(in srgb, var(--mr-tungsten) 9%, transparent);
}}
.mr-briefing blockquote :is(p, strong) {{ color: var(--mr-paper) !important; }}

/* == data readouts (hints, shot readout) ================================= */
.mr-hint, .mr-hint p, .mr-hint li, .mr-hint td, .mr-hint th {{
  font-family: var(--mr-mono);
  font-size: 0.78rem;
  line-height: 1.5;
}}
.mr-hint table {{ margin: 6px 0; }}

/* == field captions under media inputs =================================== */
.mr-caption p {{ font-size: 0.75rem; opacity: 0.75; margin: 0 0 6px; padding: 2px 10px; }}

/* == the annunciator strip =============================================== */
#annunciator {{
  background: var(--mr-ink);
  border: 1px solid var(--mr-line);
  border-radius: 8px;
  padding: 10px 14px;
}}
#annunciator p {{
  font-family: var(--mr-mono);
  font-size: 0.78rem;
  line-height: 1.6;
  color: var(--mr-dust) !important;
  margin: 2px 0;
}}
#annunciator strong {{ color: var(--mr-tungsten) !important; font-weight: 500; }}
#annunciator code {{
  font-family: var(--mr-mono);
  color: var(--mr-paper) !important;
  background: transparent;
  border: none;
  padding: 0;
}}

/* == the generate control ================================================ */
#generate-btn {{
  background: var(--mr-tungsten);
  color: #16130b;
  border: none;
  font-family: var(--mr-display);
  font-weight: 600;
  font-size: 1rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}}
#generate-btn:hover {{ filter: brightness(1.08); }}
#generate-btn:focus-visible {{ outline: 2px solid var(--mr-paper); outline-offset: 2px; }}

/* == run status: quiet by default, loud on failure ======================= */
#run-status {{
  border: 1px solid var(--mr-line);
  border-radius: 8px;
  padding: 10px 14px;
}}
#run-status p {{ font-size: 0.85rem; }}
/* Every failure shape render.py produces opens with an <h3> ("### ❌ …",
   "### 🛑 …"); success and running never do. Keying the loud treatment on that
   structure keeps the state in the *content*, where the frozen
   (video, audio, markdown) contract puts it. */
#run-status:has(h3) {{
  border-color: var(--mr-cut);
  border-left-width: 4px;
  background: color-mix(in srgb, var(--mr-cut) 10%, transparent);
}}
#run-status h3 {{
  color: var(--mr-cut) !important;
  font-family: var(--mr-display);
  font-size: 1rem;
  letter-spacing: 0.03em;
  margin: 0 0 6px;
}}

/* == LoRA bay ============================================================ */
#lora-bay {{
  border: 1px solid var(--mr-line);
  border-radius: 10px;
  padding: 14px 16px;
}}
.mr-bay-head h3 {{
  font-family: var(--mr-display);
  font-size: 0.95rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin: 0 0 2px;
}}
.mr-bay-head p {{ font-size: 0.8rem; opacity: 0.8; margin: 0 0 8px; }}

/* == accordions as console panels ======================================== */
.mr-panel {{
  border: 1px solid var(--mr-line) !important;
  border-radius: 10px !important;
}}
.mr-panel > button span,
.mr-panel .label-wrap span {{
  font-family: var(--mr-display);
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}}

/* == restraint =========================================================== */
@media (prefers-reduced-motion: reduce) {{
  #mode-board label {{ transition: none; }}
}}
"""


def head_html() -> str:
    """Font links plus the stylesheet, ready for ``gr.HTML(head=...)``."""
    return (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link rel="stylesheet" href="{_FONTS_HREF}">\n'
        f"<style>{_css()}</style>"
    )


# ---------------------------------------------------------------------------
# the masthead
# ---------------------------------------------------------------------------


def masthead_html(facts: Iterable[tuple[str, str]]) -> str:
    """The identity strip: wordmark, tagline, and the standing cost facts.

    ``facts`` is ``render.STANDING_FACTS`` — ``(lamp label, sentence)`` pairs.
    The copy lives in ``render.py`` with every other string a user reads; this
    only gives it the fact-rail form. Escaped because that is free, not because
    the facts are user input.
    """
    rows = "\n".join(
        f'<div class="mr-fact"><span class="mr-fact-lamp">{html.escape(lamp)}</span>'
        f"<span>{html.escape(sentence)}</span></div>"
        for lamp, sentence in facts
    )
    return (
        '<header class="mr-masthead">\n'
        '<h1 class="mr-wordmark">LTX<span class="mr-model">-2.5</span> Studio</h1>\n'
        '<p class="mr-tagline">Eleven modes over one jointly audio-visual model · '
        "user-loadable LoRA at inference time</p>\n"
        f"{rows}\n"
        "</header>"
    )
