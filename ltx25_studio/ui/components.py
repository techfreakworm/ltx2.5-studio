"""Gradio components for the field spec, and the LoRA bay.

The only module that maps a :class:`~ltx25_studio.ui.fields.Field` onto a Gradio
class. ``fields.py`` stays Gradio-free; which class and which constructor
arguments live here.

Gradio 6 spellings used here (6.24.0): ``buttons=[...]`` replaces the 5.x
download/share flags; ``gr.Dataframe`` takes plain ``row_count`` /
``column_count`` ints; media components have no ``info=``, so their line of
guidance is a caption rendered underneath.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from . import render
from .fields import FIELD_KEYS, Field, field_by_key

__all__ = ["assert_pool_complete", "build_field", "lora_bay", "render_fields"]

_ACCEPTS_INFO = frozenset({"text", "number", "slider", "checkbox", "radio"})

#: Media inputs share one compact height so the inputs column stays short.
MEDIA_HEIGHT = 200


def _build(spec: Field, **overrides: Any) -> gr.components.Component:
    options: dict[str, Any] = dict(spec.options)
    common: dict[str, Any] = {"label": spec.label}
    if spec.kind in _ACCEPTS_INFO and spec.info:
        common["info"] = spec.info
    common.update(overrides)

    if spec.kind == "text":
        return gr.Textbox(value=spec.default, **common, **options)
    if spec.kind == "number":
        return gr.Number(value=spec.default, **common, **options)
    if spec.kind == "slider":
        return gr.Slider(value=spec.default, **common, **options)
    if spec.kind == "checkbox":
        return gr.Checkbox(value=spec.default, **common, **options)
    if spec.kind == "radio":
        return gr.Radio(value=spec.default, **common, **options)
    if spec.kind == "image":
        return gr.Image(
            type="pil",
            sources=["upload", "clipboard"],
            buttons=["download"],
            height=MEDIA_HEIGHT,
            **common,
            **options,
        )
    if spec.kind == "video":
        return gr.Video(
            sources=["upload"], buttons=["download"], height=MEDIA_HEIGHT, **common, **options
        )
    if spec.kind == "audio":
        return gr.Audio(
            type="filepath",
            sources=["upload", "microphone"],
            buttons=["download"],
            **common,
            **options,
        )
    if spec.kind == "dataframe":
        return gr.Dataframe(value=spec.default, interactive=True, **common, **options)
    if spec.kind == "state":
        return gr.State(value=list(spec.default or []))
    raise ValueError(f"unknown field kind {spec.kind!r} for {spec.key!r}")


def build_field(spec: Field, **overrides: Any) -> gr.components.Component:
    """Build one field where the caller currently is in the layout."""
    return _build(spec, **overrides)


def render_fields(
    pool: dict[str, Any],
    *keys: str,
    captions: dict[str, Any] | None = None,
    **overrides: Any,
) -> list[Any]:
    """Build several fields in order and register them in ``pool``.

    A media field's guidance becomes a caption below it; captions go in their
    own dict, never the pool, which stays exactly the marshalled inputs.
    """
    built = []
    for key in keys:
        if key in pool:
            raise RuntimeError(f"field {key!r} was rendered twice")
        spec = field_by_key(key)
        component = build_field(spec, **overrides)
        pool[key] = component
        built.append(component)
        if spec.kind in ("image", "video", "audio") and captions is not None:
            captions[key] = gr.Markdown(spec.info, elem_classes=["mr-caption"])
    return built


def assert_pool_complete(pool: dict[str, Any]) -> None:
    """Fail at build time if a declared field was never rendered, or vice versa."""
    declared, rendered = set(FIELD_KEYS), set(pool)
    missing, extra = sorted(declared - rendered), sorted(rendered - declared)
    if missing or extra:
        raise RuntimeError(
            "the UI field pool does not match the field spec. "
            f"Declared but never rendered: {missing or 'none'}. "
            f"Rendered but not declared: {extra or 'none'}."
        )


# ---------------------------------------------------------------------------
# the LoRA bay
# ---------------------------------------------------------------------------


def lora_bay(*, rows: int, cap: str, step: float, on_zerogpu: bool = True) -> dict[str, Any]:
    """Both routes (Hub id, upload), the verdict, and ``rows`` staged-adapter rows.

    Each row is a name, a strength slider and a Remove button; rows show and
    hide with the staged list. A row is hidden until an adapter is staged, and
    showing it sets the slider's range from the LoRA module (never a copy kept
    here), so the range built here is only a placeholder nobody sees.
    """
    widgets: dict[str, Any] = {}
    gr.Markdown(render.lora_heading(cap, on_zerogpu=on_zerogpu), elem_classes=["mr-bay-head"])
    with gr.Row(elem_classes=["mr-hub-row"]):
        widgets["repo_id"] = gr.Textbox(
            label="Hub repo id",
            placeholder="org/name, or org/name/file.safetensors",
            scale=4,
            min_width=200,
        )
        widgets["validate"] = gr.Button(
            "Check and add",
            variant="secondary",
            elem_classes=["mr-secondary"],
            size="sm",
            scale=1,
            min_width=140,
        )
    widgets["upload"] = gr.File(
        label="Or upload a .safetensors file",
        file_types=[".safetensors"],
        type="filepath",
        height=120,
        elem_classes=["mr-drop"],
    )
    widgets["verdict"] = gr.Markdown(render.lora_idle(), elem_id="lora-verdict")
    widgets["rows"] = []
    for index in range(rows):
        with gr.Row(elem_classes=["mr-lora-row"], equal_height=True):
            name = gr.Markdown(
                "", visible=False, elem_classes=["mr-lora-name"], min_width=200, scale=2
            )
            weight = gr.Slider(
                minimum=step,
                maximum=1.0,
                step=step,
                value=1.0,
                label=f"Strength of adapter {index + 1}",
                visible=False,
                scale=3,
                min_width=160,
                elem_id=f"lora-strength-{index}",
            )
            remove = gr.Button(
                "Remove",
                visible=False,
                elem_classes=["mr-quiet"],
                size="sm",
                scale=0,
                min_width=90,
            )
        widgets["rows"].append({"name": name, "weight": weight, "remove": remove})
    widgets["clear"] = gr.Button("Remove all", elem_classes=["mr-quiet"], size="sm", visible=False)
    widgets["applied_summary"] = gr.Markdown(render.applied_loras([]), visible=False)
    return widgets
