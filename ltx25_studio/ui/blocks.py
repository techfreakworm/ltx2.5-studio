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

**The look is "the machine room"** -- slate console panels, a mode board,
tungsten cost lamps and a live annunciator strip above Generate. The design
system (tokens, stylesheet, masthead) lives in ``chrome.py``; the full design
rationale in ``docs/UI_DESIGN.md``. This module only assigns the ``elem_id`` /
``elem_classes`` hooks the stylesheet targets.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from .. import modes
from . import chrome, components, handlers, render
from .fields import FIELD_KEYS, FIELDS, TOGGLEABLE_FIELDS

__all__ = ["build_ui"]

_TITLE = "LTX-2.5 Studio"

_MULTISHOT_NOTE = """
**Multishot is prompt-structure driven and unverified.** LTX-2.5 has no scene or
multishot API — a shot list works with the model's phrasing, not with a supported
feature. It is offered as a documented technique, not as a button that implies backing
we do not have.

**The anchor column takes a frame number.** It is converted to the model's latent grid
exactly once, in `conditioning.py`. That grid is 8:1, so anchors snap: frames 25–32 all
land on the same latent. The readout below shows where each anchor really lands, and
flags anchors past the end of the clip — upstream skips those with only a log line and
generates anyway.
"""

_READY = (
    "**Ready.** Pick a mode, set the run up, and press Generate — the output and the "
    "run report appear here. Nothing is spent until the button is pressed."
)


def _mode_choices() -> list[tuple[str, str]]:
    """Radio choices, warning-marked straight from the registry.

    PLAN §3.2 requires the UI to say *where the user selects them* that modes
    8-11 still render video. Deriving the marker from ``cost_note`` means a new
    warned mode is marked by adding a registry entry, never by editing this
    file. The zero-padded number is the same one PLAN §3.2 and the mode matrix
    docs use, so a mode can be named across a bug report without ambiguity.
    """
    choices = []
    for number, mode_id in enumerate(modes.MODES, start=1):
        mode = modes.get_mode(mode_id)
        marker = " ⚠️" if getattr(mode, "cost_note", None) else ""
        choices.append((f"{number:02d} · {mode.label}{marker}", mode_id))
    return choices


def _field_defaults() -> dict[str, Any]:
    """Every field's declared default, for computing first-paint readouts."""
    return {spec.key: spec.default for spec in FIELDS}


def build_ui() -> gr.Blocks:
    """Assemble the Studio. Returns an unlaunched :class:`gradio.Blocks`."""
    pool: dict[str, Any] = {}
    captions: dict[str, Any] = {}
    default_mode = next(iter(modes.MODES))
    defaults = _field_defaults()

    with gr.Blocks(title=_TITLE, fill_width=True) as demo:
        # The masthead carries the stylesheet and fonts in its `head=` -- the
        # Gradio 6 injection point that keeps `app.py`'s bare `launch()` intact.
        gr.HTML(
            value=chrome.masthead_html(render.STANDING_FACTS),
            head=chrome.head_html(),
            container=False,
            padding=False,
        )

        with gr.Row():
            # ---------------------------------------------------------------
            # the commission column: mode board -> inputs -> shape -> LoRA
            # ---------------------------------------------------------------
            with gr.Column(scale=5):
                mode_selector = gr.Radio(
                    choices=_mode_choices(),
                    value=default_mode,
                    label="Mode",
                    info="⚠️ marks a mode that costs more than its name suggests — the briefing below says exactly what.",
                    elem_id="mode-board",
                )
                briefing = gr.Markdown(
                    render.mode_briefing(modes.get_mode(default_mode)),
                    elem_classes=["mr-briefing"],
                )

                with gr.Group():
                    components.render_fields(pool, "prompt", "negative_prompt", captions=captions)
                    # Laid out here, declared last in `fields.py`. The two
                    # orderings are independent by design (see `components.py`):
                    # the endpoint's signature is append-only, the screen is not.
                    components.render_fields(pool, "enhance_prompt", captions=captions)
                    components.render_fields(pool, "image", "video", "audio", captions=captions)

                with gr.Accordion("Shot list / multishot", open=False, elem_classes=["mr-panel"]):
                    gr.Markdown(_MULTISHOT_NOTE)
                    components.render_fields(pool, "conditions", captions=captions)
                    shot_readout = gr.Markdown(elem_classes=["mr-hint"])

                with gr.Accordion("Size & length", open=True, elem_classes=["mr-panel"]):
                    with gr.Row():
                        components.render_fields(pool, "width", "height", captions=captions)
                    components.render_fields(pool, "auto_duration", captions=captions)
                    with gr.Row():
                        components.render_fields(pool, "num_frames", captions=captions)
                        components.render_fields(
                            pool, "min_seconds", "max_seconds", captions=captions
                        )
                    snap_button = gr.Button("Snap to legal values", size="sm")
                    dimension_hint = gr.Markdown(elem_classes=["mr-hint"])
                    frame_hint = gr.Markdown(elem_classes=["mr-hint"])

                with gr.Accordion("Sampling", open=True, elem_classes=["mr-panel"]):
                    # No guidance or step controls: the distilled recipe fixes
                    # both, and `pipelines.distilled_kwargs()` is the one place
                    # that sets them. See the note in `ui/fields.py`.
                    components.render_fields(pool, "seed", captions=captions)
                    components.render_fields(pool, "strength", "noise_scale", captions=captions)
                    components.render_fields(pool, "enable_stage_2", captions=captions)

                # The headline feature gets a bay, not an accordion: always
                # open, framed, first-class. PLAN §4.
                with gr.Group(elem_id="lora-bay"):
                    lora = components.lora_panel()
                    components.render_fields(pool, "loras")

                # The annunciator: the run's true size, length and cost, kept
                # live at the point of commitment. See handlers.on_cost_change.
                annunciator = gr.Markdown(
                    handlers.on_cost_change(
                        default_mode,
                        defaults["width"],
                        defaults["height"],
                        defaults["num_frames"],
                        defaults["auto_duration"],
                        defaults["min_seconds"],
                        defaults["max_seconds"],
                        defaults["enable_stage_2"],
                    ),
                    elem_id="annunciator",
                )
                generate_button = gr.Button(
                    "Generate", variant="primary", size="lg", elem_id="generate-btn"
                )

            # ---------------------------------------------------------------
            # the returns column: status first, then the players
            # ---------------------------------------------------------------
            with gr.Column(scale=4):
                gr.Markdown("Output", elem_classes=["mr-eyebrow"])
                status = gr.Markdown(_READY, elem_id="run-status")
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
        # Captions follow their field's visibility. `captions` is keyed by field
        # key; each pairs with the index of that field's update in the tuple
        # `handlers.on_mode_change` returns, so a caption can never outlive its
        # field on screen. Media fields always carry `visible_when`, hence are
        # always toggleable -- `strict` guards that assumption.
        caption_bindings = [
            (captions[spec.key], index)
            for index, spec in enumerate(TOGGLEABLE_FIELDS)
            if spec.key in captions
        ]
        assert len(caption_bindings) == len(captions), (
            "a caption belongs to a field the mode selector never toggles; "
            "it would linger on screen for modes without that field"
        )
        caption_components = [component for component, _ in caption_bindings]

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
                *(gr.update(visible=updates[index].visible) for _, index in caption_bindings),
                gr.update(visible=shows_video),
                gr.update(visible=shows_audio),
            ]

        mode_change_outputs = [
            briefing,
            *toggleable,
            *caption_components,
            video_out,
            audio_out,
        ]
        mode_selector.change(
            fn=_on_mode_change, inputs=[mode_selector], outputs=mode_change_outputs
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

        # Everything that moves a cost fact refreshes the annunciator: the mode
        # (audio-only pinning), the shape (stage 2 doubles it), and the stage-2
        # box itself (the ×1.5 reservation).
        cost_inputs = [
            mode_selector,
            pool["width"],
            pool["height"],
            pool["num_frames"],
            pool["auto_duration"],
            pool["min_seconds"],
            pool["max_seconds"],
            pool["enable_stage_2"],
        ]
        for control in cost_inputs:
            control.change(fn=handlers.on_cost_change, inputs=cost_inputs, outputs=[annunciator])

        lora_outputs = [lora["verdict"], pool["loras"], lora["applied_summary"]]

        def _on_validate_lora(
            repo_id: str | None,
            upload: Any,
            applied: list,
            progress: Any = gr.Progress(track_tqdm=True),  # noqa: B008 -- gradio's injection contract
        ):
            """The handler, with the blocks-level progress channel attached.

            The Hub route downloads in the web process during validation, so the
            press can take tens of seconds. ``gr.Progress`` must appear in the
            *wired* function's signature, and ``handlers.py`` deliberately does
            not import gradio -- so the channel is attached here, at the one
            layer allowed to know about it. ``track_tqdm=True`` mirrors
            ``hf_hub_download``'s own tqdm bars onto the verdict surface, so a
            first fetch shows the real download progress rather than a spinner.
            (The patch it installs is contextvar-gated: outside an active event,
            tqdm behaves exactly as before.)
            """
            return handlers.on_validate_lora(repo_id, upload, applied)

        # Two steps, exactly like Generate: an instant state message, then the
        # work. `api_name` pins the published endpoint to the name it has always
        # had, so the wrapper's leading underscore never leaks into the API.
        lora["validate"].click(
            fn=handlers.on_validate_start,
            inputs=[lora["repo_id"], lora["upload"]],
            outputs=[lora["verdict"]],
        ).then(
            fn=_on_validate_lora,
            inputs=[lora["repo_id"], lora["upload"], pool["loras"]],
            outputs=lora_outputs,
            api_name="on_validate_lora",
            show_progress_on=lora["verdict"],
        )
        lora["clear"].click(fn=handlers.on_clear_loras, inputs=None, outputs=lora_outputs)

        # `FIELD_KEYS` orders this list AND orders the unpacking inside
        # `handlers.on_generate`, which zips with `strict=True`. One tuple, both
        # jobs -- the two cannot drift, and if they ever did it would raise
        # rather than silently pass arguments to the wrong parameters.
        generate_button.click(
            fn=handlers.on_generate_start,
            inputs=[mode_selector, pool["enhance_prompt"]],
            outputs=[status],
        ).then(
            fn=handlers.on_generate,
            inputs=[mode_selector, *(pool[key] for key in FIELD_KEYS)],
            outputs=[video_out, audio_out, status],
        )

        # Put the form in the selected mode's state before the first interaction,
        # and show the constraint guidance immediately rather than only after a
        # value changes.
        demo.load(fn=_on_mode_change, inputs=[mode_selector], outputs=mode_change_outputs)
        demo.load(fn=handlers.on_shape_change, inputs=shape_inputs, outputs=hint_outputs)
        demo.load(fn=handlers.on_shots_change, inputs=shot_inputs, outputs=[shot_readout])
        demo.load(fn=handlers.on_cost_change, inputs=cost_inputs, outputs=[annunciator])
        demo.load(
            fn=lambda chosen: [
                gr.update(visible=flag) for flag in handlers.on_auto_duration_toggle(chosen)
            ],
            inputs=[pool["auto_duration"]],
            outputs=[pool["num_frames"], pool["min_seconds"], pool["max_seconds"]],
        )

    return demo
