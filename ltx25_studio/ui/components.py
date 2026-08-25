"""Reusable widgets: the condition list, the LoRA picker, constraint hints.

PLAN §3.1(a) -- one condition list serves both image and video conditioning,
each entry carrying a source, an ``index`` and a ``strength``; §3.3 builds
multishot on the same list. §5.5 -- dimensions snap to /32 and the snapped value
is shown, never silently substituted.

This is the only module that maps a :class:`~ltx25_studio.ui.fields.Field` onto a
Gradio component. ``fields.py`` stays Gradio-free so the spec can be read
without a UI toolkit installed; the knowledge of *which class* and *which
constructor arguments* lives here and nowhere else.

**Gradio 6, not 5.** Gradio 5 is end of life and several of its idioms are now
either removed or deprecated on exactly the components this app uses. The ones
that bite here, verified against the installed 6.24.0 source rather than
recalled:

* ``gr.Video``: the ``(video_path, subtitle_path)`` tuple return is **gone**;
  subtitles are a constructor/update argument. ``mirror_webcam`` and
  ``webcam_constraints`` are now ``webcam_options=gr.WebcamOptions(...)``.
* ``gr.Video`` / ``gr.Audio`` / ``gr.Image``: ``show_download_button`` and
  ``show_share_button`` are now ``buttons=["download", "share"]``.
* ``gr.Audio``: ``WaveformOptions.show_controls`` is now
  ``show_recording_waveform``.
* ``gr.Video`` / ``gr.Audio``: ``min_length`` / ``max_length`` are gone; bounds
  are event-listener validators.
* ``gr.Dataframe``: ``row_count=(0, "dynamic")`` / ``col_count=(3, "fixed")``
  are the 5.x spelling and now raise ``DeprecationWarning``; 6.x wants a plain
  ``row_count`` / ``column_count`` int.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from .fields import FIELD_KEYS, MEDIA_KINDS, Field, field_by_key

__all__ = [
    "assert_pool_complete",
    "build_field",
    "lora_panel",
    "render_fields",
]

#: Constructor arguments that only some components accept. ``gr.Image``,
#: ``gr.Video``, ``gr.Audio`` and ``gr.Dataframe`` have no ``info=`` in Gradio 6,
#: so their guidance is rendered as a caption underneath instead of being
#: dropped on the floor.
_ACCEPTS_INFO = frozenset({"text", "number", "slider", "checkbox"})


def _build(spec: Field) -> gr.components.Component:
    """Construct the Gradio component for one field spec."""
    options: dict[str, Any] = dict(spec.options)
    common: dict[str, Any] = {"label": spec.label}
    if spec.kind in _ACCEPTS_INFO and spec.info:
        common["info"] = spec.info

    if spec.kind == "text":
        return gr.Textbox(value=spec.default, **common, **options)
    if spec.kind == "number":
        return gr.Number(value=spec.default, **common, **options)
    if spec.kind == "slider":
        return gr.Slider(value=spec.default, **common, **options)
    if spec.kind == "checkbox":
        return gr.Checkbox(value=spec.default, **common, **options)
    if spec.kind == "image":
        return gr.Image(type="pil", buttons=["download"], **common, **options)
    if spec.kind == "video":
        return gr.Video(buttons=["download"], **common, **options)
    if spec.kind == "audio":
        return gr.Audio(type="filepath", buttons=["download"], **common, **options)
    if spec.kind == "dataframe":
        return gr.Dataframe(value=spec.default, interactive=True, **common, **options)
    if spec.kind == "state":
        # gr.State takes no label and has no visibility -- see fields.Field.
        return gr.State(value=list(spec.default or []))
    raise ValueError(f"unknown field kind {spec.kind!r} for {spec.key!r}")


def build_field(spec: Field) -> gr.components.Component:
    """Build one field where the caller currently is in the layout."""
    return _build(spec)


def render_fields(
    pool: dict[str, Any], *keys: str, captions: dict[str, Any] | None = None
) -> list[Any]:
    """Build several fields in order and register them in ``pool``.

    Components attach to whichever Blocks context is open when they are
    constructed, so layout decides *where* a field is built. The pool exists so
    that ordering for *marshalling* stays governed by ``FIELD_KEYS`` alone --
    layout order and marshalling order are deliberately independent.

    A media field's ``info`` becomes a caption below it, because Gradio 6's
    media components have no ``info=``. Captions are registered in ``captions``
    -- a separate dict, never the pool, which stays exactly the set of
    marshalled inputs -- so that ``blocks.py`` can show and hide a caption with
    its field. The previous shape built captions as write-only components, and
    a hidden image input left its caption floating on screen for modes with no
    image at all.
    """
    built = []
    for key in keys:
        if key in pool:
            raise RuntimeError(f"field {key!r} was rendered twice")
        spec = field_by_key(key)
        component = build_field(spec)
        pool[key] = component
        built.append(component)
        if spec.kind in MEDIA_KINDS and spec.info and captions is not None:
            # Styled by the stylesheet in `chrome.py` rather than with an inline
            # <sub> hack, so caption size and tone are one decision made once.
            captions[key] = gr.Markdown(spec.info, elem_classes=["mr-caption"])
    return built


def assert_pool_complete(pool: dict[str, Any]) -> None:
    """Fail loudly if a declared field was never rendered, or vice versa.

    This is the check that replaces the predecessor's parallel ``if``/``elif``
    chains. There, a field added to one chain and forgotten in another produced
    a silent misalignment; here it is an exception at build time, before the app
    can serve a single request.
    """
    declared, rendered = set(FIELD_KEYS), set(pool)
    missing, extra = sorted(declared - rendered), sorted(rendered - declared)
    if missing or extra:
        raise RuntimeError(
            "the UI field pool does not match the field spec. "
            f"Declared but never rendered: {missing or 'none'}. "
            f"Rendered but not declared: {extra or 'none'}."
        )


# ---------------------------------------------------------------------------
# the LoRA panel -- PLAN §4, the headline feature
# ---------------------------------------------------------------------------


def lora_panel() -> dict[str, Any]:
    """Both acquisition routes, plus the verdict surface.

    PLAN §4.3 accepts *either* a Hub repo id *or* a direct upload, so both are
    offered side by side rather than behind a mode switch. The verdict area is a
    first-class part of the panel and starts populated with "no adapter
    applied": PLAN §4.2's silent-garbage failure looks exactly like a blank
    panel, so blankness is never used to mean "nothing is loaded".

    This is the headline feature, and the layout says so: an always-open bay
    with its own header band (``blocks.py`` supplies the ``#lora-bay`` frame),
    never an accordion a first-time user has to discover.
    """
    from . import render

    widgets: dict[str, Any] = {}
    gr.Markdown(
        "### LoRA adapters\n"
        "Load an adapter from the Hub or upload one directly. It is validated **before any "
        "weights are touched** — only the safetensors header is read, so a multi-gigabyte "
        "file for the wrong architecture is rejected in milliseconds, with the exact reason.",
        elem_classes=["mr-bay-head"],
    )
    with gr.Row():
        widgets["repo_id"] = gr.Textbox(
            label="Hub repo id",
            placeholder="joyfox/LTX-2.3-Transition-LORA",
            info="A repo id, or a repo id plus a filename.",
            scale=3,
        )
        widgets["upload"] = gr.File(
            label="…or upload an adapter (.safetensors)",
            file_types=[".safetensors"],
            type="filepath",
            scale=2,
        )
    with gr.Row():
        widgets["validate"] = gr.Button("Validate & apply", variant="secondary")
        widgets["clear"] = gr.Button("Clear applied adapters")
    widgets["verdict"] = gr.Markdown(render.lora_idle())
    widgets["applied_summary"] = gr.Markdown(render.applied_loras([]))
    return widgets
