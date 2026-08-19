"""Gradio Blocks assembly.

PLAN §7.1. Also the layer that turns a typed exception into a string a human
reads (§7.2) -- and the layer that must render generation failures VISIBLY
rather than as a silent form reset (§5.3).

**Eleven modes, one set of widgets.** PLAN §3.2's registry is a presentation
layer over four real workflows, so this builds *one* control column and lets the
mode selector decide what is visible. The visibility decision, the marshalling
order and the component pool all derive from the single spec in ``fields.py``;
this module only places widgets and converts booleans into ``gr.update`` calls.

**No handler wired here may return a no-op.** Every event below has a concrete
value for every output it declares. That is the direct lesson of §5.3: the
predecessor's outage was a handler returning ``gr.update()`` for an empty path,
which on screen is indistinguishable from a form resetting itself.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from .. import modes
from . import components, handlers, render
from .fields import FIELD_KEYS, TOGGLEABLE_FIELDS

__all__ = ["build_ui"]

_TITLE = "LTX-2.5 Studio"

_INTRO = """
# 🎬 LTX-2.5 Studio

Every mode LTX-2.5 supports, with **user-uploadable LoRA** at inference time.
"""

_MULTISHOT_NOTE = """
**Multishot is prompt-structure driven and unverified.** LTX-2.5 has no scene or
multishot API, and the 2.5 prompt enhancer is explicitly instructed *not* to describe scene
cuts — so a shot list works with the model's phrasing, not with a supported feature. It is
offered as a documented technique, not as a button that implies backing we do not have.

**The anchor is a LATENT index, not a pixel frame number.** The pipeline converts it with
`pixel ≈ max((latent − 1) × 8 + 1, 0)`, so anchor `2` lands near pixel frame 9 — not frame 2.
Typing a pixel frame number here is an off-by-8 every time. The conversion itself happens in
`conditioning.py`; this table takes the latent index that layer expects.
"""


def _mode_choices() -> list[tuple[str, str]]:
    """Radio choices, warning-marked straight from the registry.

    PLAN §3.2 requires the UI to say *where the user selects them* that modes
    8-11 still render video. Deriving the marker from ``cost_note`` means a new
    warned mode is marked by adding a registry entry, never by editing this
    file.
    """
    choices = []
    for number, mode_id in enumerate(modes.MODES, start=1):
        mode = modes.get_mode(mode_id)
        marker = " ⚠️" if getattr(mode, "cost_note", None) else ""
        choices.append((f"{number}. {mode.label}{marker}", mode_id))
    return choices


def build_ui() -> gr.Blocks:
    """Assemble the Studio. Returns an unlaunched :class:`gradio.Blocks`."""
    pool: dict[str, Any] = {}
    default_mode = next(iter(modes.MODES))

    with gr.Blocks(title=_TITLE, fill_width=True) as demo:
        gr.Markdown(_INTRO)
        gr.Markdown(render.ZEROGPU_BANNER)

        with gr.Row():
            # ---------------------------------------------------------------
            # controls
            # ---------------------------------------------------------------
            with gr.Column(scale=5):
                gr.Markdown(render.AUDIO_VISUAL_BANNER)
                mode_selector = gr.Radio(
                    choices=_mode_choices(),
                    value=default_mode,
                    label="Mode",
                    info="⚠️ marks a mode that costs more than its name suggests — see below.",
                )
                briefing = gr.Markdown(render.mode_briefing(modes.get_mode(default_mode)))

                with gr.Group():
                    components.render_fields(pool, "prompt", "negative_prompt")
                    components.render_fields(pool, "image", "video", "audio")

                with gr.Accordion("Shot list / multishot", open=False):
                    gr.Markdown(_MULTISHOT_NOTE)
                    components.render_fields(pool, "conditions")
                    shot_readout = gr.Markdown()

                with gr.Accordion("Size and length", open=True):
                    with gr.Row():
                        components.render_fields(pool, "width", "height")
                    components.render_fields(pool, "auto_duration")
                    with gr.Row():
                        components.render_fields(pool, "num_frames")
                        components.render_fields(pool, "min_seconds", "max_seconds")
                    snap_button = gr.Button("Snap to legal values", size="sm")
                    dimension_hint = gr.Markdown()
                    frame_hint = gr.Markdown()

                with gr.Accordion("Sampling", open=False):
                    # No guidance or step controls: the distilled recipe fixes
                    # both, and `pipelines.distilled_kwargs()` is the one place
                    # that sets them. See the note in `ui/fields.py`.
                    components.render_fields(pool, "seed")
                    components.render_fields(pool, "strength", "noise_scale")
                    components.render_fields(pool, "enable_stage_2")

                with gr.Accordion("LoRA", open=True):
                    lora = components.lora_panel()
                    components.render_fields(pool, "loras")

                generate_button = gr.Button("Generate", variant="primary", size="lg")

            # ---------------------------------------------------------------
            # outputs
            # ---------------------------------------------------------------
            with gr.Column(scale=4):
                status = gr.Markdown("Ready.")
                video_out = gr.Video(
                    label="Video",
                    interactive=False,
                    buttons=["download"],
                    autoplay=True,
                )
                audio_out = gr.Audio(
                    label="Audio",
                    interactive=False,
                    buttons=["download"],
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                )

        components.assert_pool_complete(pool)

        # -------------------------------------------------------------------
        # wiring
        # -------------------------------------------------------------------
        toggleable = [pool[spec.key] for spec in TOGGLEABLE_FIELDS]
        shape_inputs = [pool["width"], pool["height"], pool["num_frames"]]
        hint_outputs = [dimension_hint, frame_hint]

        def _on_mode_change(mode_id: str) -> list[Any]:
            """Booleans in, ``gr.update``s out. The decision itself is in
            ``handlers`` so it can be asserted on without a Blocks context."""
            text, updates = handlers.on_mode_change(mode_id)
            shows_video, shows_audio = handlers.export_flags(mode_id)
            return [
                text,
                *(
                    gr.update(visible=u.visible, interactive=u.interactive)
                    if u.value is None
                    else gr.update(visible=u.visible, interactive=u.interactive, value=u.value)
                    for u in updates
                ),
                gr.update(visible=shows_video),
                gr.update(visible=shows_audio),
            ]

        mode_selector.change(
            fn=_on_mode_change,
            inputs=[mode_selector],
            outputs=[briefing, *toggleable, video_out, audio_out],
        )

        for control in shape_inputs:
            control.change(fn=handlers.on_shape_change, inputs=shape_inputs, outputs=hint_outputs)

        shot_inputs = [pool["conditions"], pool["num_frames"]]
        for control in shot_inputs:
            control.change(fn=handlers.on_shots_change, inputs=shot_inputs, outputs=[shot_readout])

        snap_button.click(
            fn=handlers.on_snap, inputs=shape_inputs, outputs=[*shape_inputs, *hint_outputs]
        )

        pool["auto_duration"].change(
            fn=lambda chosen: [
                gr.update(visible=flag) for flag in handlers.on_auto_duration_toggle(chosen)
            ],
            inputs=[pool["auto_duration"]],
            outputs=[pool["num_frames"], pool["min_seconds"], pool["max_seconds"]],
        )

        lora_outputs = [lora["verdict"], pool["loras"], lora["applied_summary"]]
        lora["validate"].click(
            fn=handlers.on_validate_lora,
            inputs=[lora["repo_id"], lora["upload"], pool["loras"]],
            outputs=lora_outputs,
        )
        lora["clear"].click(fn=handlers.on_clear_loras, inputs=None, outputs=lora_outputs)

        # `FIELD_KEYS` orders this list AND orders the unpacking inside
        # `handlers.on_generate`, which zips with `strict=True`. One tuple, both
        # jobs -- the two cannot drift, and if they ever did it would raise
        # rather than silently pass arguments to the wrong parameters.
        generate_button.click(
            fn=handlers.on_generate_start, inputs=[mode_selector], outputs=[status]
        ).then(
            fn=handlers.on_generate,
            inputs=[mode_selector, *(pool[key] for key in FIELD_KEYS)],
            outputs=[video_out, audio_out, status],
        )

        # Put the form in the selected mode's state before the first interaction,
        # and show the constraint guidance immediately rather than only after a
        # value changes.
        demo.load(
            fn=_on_mode_change,
            inputs=[mode_selector],
            outputs=[briefing, *toggleable, video_out, audio_out],
        )
        demo.load(fn=handlers.on_shape_change, inputs=shape_inputs, outputs=hint_outputs)
        demo.load(fn=handlers.on_shots_change, inputs=shot_inputs, outputs=[shot_readout])
        demo.load(
            fn=lambda chosen: [
                gr.update(visible=flag) for flag in handlers.on_auto_duration_toggle(chosen)
            ],
            inputs=[pool["auto_duration"]],
            outputs=[pool["num_frames"], pool["min_seconds"], pool["max_seconds"]],
        )

    return demo
