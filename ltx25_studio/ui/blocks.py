"""Gradio Blocks assembly: layout and wiring.

Two columns at 1024 px and wider -- the commission column (the input × output
matrix, then what the selected mode takes) and a sticky rail (monitor, cost
panel, Generate, report, recent runs). Below 1024 px, one column and a fixed
commit bar holding the same Generate button. Styles are in ``chrome.py``;
every decision is in ``handlers.py``; every string in ``render.py``.

Contracts kept here:

* ``/on_generate``: parameter 0 is the mode radio, then ``FIELD_KEYS`` in
  order; returns ``[Video, Audio, Markdown]``. ``/on_validate_lora`` keeps its
  inputs and outputs. Every other event is private.
* No handler wired here returns a no-op for an output it declares.
* Container visibility and labels are changed in their own step, never in the
  same response as their children's values.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import gradio as gr

from .. import modes
from . import chrome, components, handlers, render, showcase
from .fields import FIELD_KEYS, PARKED_FIELDS

__all__ = ["DEFAULT_MODE", "build_ui"]

_TITLE = "LTX-2.5 Studio"

#: Text → video + sound: the model's native output, and every run renders and
#: pays for the sound anyway.
DEFAULT_MODE = "text_to_audio_video"

#: Recent runs kept per session.
RECENT_LIMIT = 6

#: Example buttons pre-built per mode.
EXAMPLE_SLOTS = 3

#: Prune Gradio's upload cache every hour, removing files older than six hours.
#: Staged LoRA uploads are copied out of it first (``uploads.own_copy``).
DELETE_CACHE = (3600, 6 * 3600)

#: Upper bound for the LoRA strength slider; never above the LoRA module's own.
LORA_SLIDER_MAX = 1.5

_PRIVATE = {"api_visibility": "private", "show_progress": "hidden"}

# ---------------------------------------------------------------------------
# the run outcome, handed from /on_generate to the chained finish step
# ---------------------------------------------------------------------------

_OUTCOMES: OrderedDict[str, handlers.RunOutcome] = OrderedDict()
_OUTCOMES_LOCK = threading.Lock()
_OUTCOMES_KEPT = 64


def _session(request: Any) -> str | None:
    return getattr(request, "session_hash", None) if request is not None else None


def _remember(request: Any, outcome: handlers.RunOutcome) -> None:
    key = _session(request)
    if key is None:
        return
    with _OUTCOMES_LOCK:
        _OUTCOMES[key] = outcome
        _OUTCOMES.move_to_end(key)
        while len(_OUTCOMES) > _OUTCOMES_KEPT:
            _OUTCOMES.popitem(last=False)


def _recall(request: Any) -> handlers.RunOutcome | None:
    key = _session(request)
    if key is None:
        return None
    with _OUTCOMES_LOCK:
        return _OUTCOMES.pop(key, None)


def _on_generate(progress: Any = gr.Progress(), request: gr.Request = None, *values: Any):  # noqa: B008
    """``/on_generate``: ``handlers.run`` with Gradio's progress channel attached.

    The parameter order is load-bearing. Gradio finds special arguments by
    walking the positional parameters and stopping at the first other kind
    (``helpers.special_args``), so ``progress`` and ``request`` must come
    before ``*values``; ``tests/test_progress.py`` drives ``special_args``
    against this function. ``track_tqdm`` stays off: it patches ``tqdm``
    process-wide.
    """
    mode_id, *field_values = values
    outcome = handlers.run(mode_id, *field_values, progress=progress)
    _remember(request, outcome)
    return outcome.as_tuple()


def _on_validate_lora(repo_id: str | None, upload: Any, applied: list):
    """``/on_validate_lora``. No ``gr.Progress(track_tqdm=True)`` here: building
    one patches ``tqdm`` for the whole process and breaks bars that already
    exist (the enhancer prefetch)."""
    return handlers.on_validate_lora(repo_id, upload, applied)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _hide() -> dict:
    return gr.update(visible=False)


def _exists(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return Path(value).exists()
    return True


def _values(values: tuple) -> dict[str, Any]:
    return dict(zip(FIELD_KEYS, values, strict=True))


def _noise_value(choice: Any) -> float:
    return dict(render.NOISE_CHOICES).get(choice, dict(render.NOISE_CHOICES)["Moderately"])


def _noise_choice(value: Any) -> str:
    for name, level in render.NOISE_CHOICES:
        if abs(level - float(value or 0)) < 1e-6:
            return name
    return "Moderately"


def build_ui() -> gr.Blocks:
    """Assemble the Studio. Returns an unlaunched :class:`gradio.Blocks`."""
    show = showcase.load_showcase(known_modes=tuple(modes.MODES))
    if show.root is not None:
        gr.set_static_paths([show.root])

    from .. import zerogpu
    from ..config import load_config

    # The LoRA package stays lazy (it loads with the first check or run); the
    # slider's ceiling is held under the package's own maximum by a test.
    max_weight = LORA_SLIDER_MAX
    cap_gb = load_config().lora_max_bytes / 1e9
    multiplier = handlers.quota_multiplier()
    view0 = handlers.mode_view(DEFAULT_MODE)
    mode0 = modes.get_mode(DEFAULT_MODE)

    pool: dict[str, Any] = {}
    captions: dict[str, Any] = {}

    with gr.Blocks(title=_TITLE, fill_width=True, delete_cache=DELETE_CACHE) as demo:
        gr.HTML(
            value=chrome.identity_html(
                quota_text=render.quota_explainer(
                    multiplier=multiplier, example_reserve=zerogpu.estimate_duration()
                )
            ),
            head=chrome.head_html(),
            elem_id="identity",
            apply_default_css=False,
            container=False,
            padding=False,
        )

        with gr.Row(elem_id="shell", equal_height=False):
            # =============================================================
            # the commission column
            # =============================================================
            with gr.Column(elem_id="commission", scale=3):
                with gr.Column(elem_id="matrix"):
                    gr.HTML(
                        chrome.matrix_frame_html(),
                        elem_id="matrix-frame",
                        apply_default_css=False,
                        container=False,
                        padding=False,
                    )
                    mode_selector = gr.Radio(
                        choices=[(spec.label, mode_id) for mode_id, spec in modes.MODES.items()],
                        value=DEFAULT_MODE,
                        label="What you start from, and what you get back",
                        elem_id=chrome.MATRIX_ID,
                        container=False,
                    )
                purpose = gr.Markdown(view0.purpose, elem_id="purpose")

                example_buttons: list[Any] = []
                example_note = None
                if show.has_examples:
                    with gr.Row(elem_id="examples"):
                        for slot in range(EXAMPLE_SLOTS):
                            examples_now = show.for_mode(DEFAULT_MODE)
                            label = examples_now[slot].label if slot < len(examples_now) else ""
                            example_buttons.append(
                                gr.Button(
                                    label or "Example",
                                    visible=slot < len(examples_now),
                                    elem_classes=["mr-secondary", "mr-example"],
                                    size="sm",
                                    scale=0,
                                    min_width=120,
                                )
                            )
                        example_note = gr.Markdown(
                            render.example_label(),
                            visible=bool(show.for_mode(DEFAULT_MODE)),
                            elem_classes=["mr-example-note"],
                        )

                # -- what you give ------------------------------------------
                with gr.Column(elem_classes=["mr-divide"]):
                    components.render_fields(
                        pool,
                        "image",
                        captions=captions,
                        visible=view0.active["image"],
                        elem_id="in-image",
                    )
                    components.render_fields(
                        pool,
                        "video",
                        captions=captions,
                        visible=view0.active["video"],
                        elem_id="in-video",
                    )
                    components.render_fields(
                        pool, "strength", visible=view0.active["strength"], elem_id="lever-strength"
                    )
                    components.render_fields(
                        pool,
                        "audio",
                        captions=captions,
                        visible=view0.active["audio"],
                        elem_id="in-audio",
                    )
                    noise_choice = gr.Radio(
                        choices=[name for name, _ in render.NOISE_CHOICES],
                        value="Moderately",
                        label=render.FIELD_COPY["noise_scale"][0],
                        info=render.FIELD_COPY["noise_scale"][1],
                        visible=view0.active["noise_scale"],
                        elem_id="lever-noise",
                        elem_classes=["mr-chips", "mr-words"],
                    )
                    components.render_fields(pool, "noise_scale", visible=False)
                    components.render_fields(
                        pool,
                        "prompt",
                        visible=view0.active["prompt"],
                        elem_id="prompt",
                        label=view0.prompt_label,
                        placeholder=view0.prompt_placeholder,
                    )
                    components.render_fields(
                        pool,
                        "enhance_prompt",
                        visible=view0.active["enhance_prompt"],
                        elem_id="enhance",
                    )
                    components.render_fields(pool, "negative_prompt", "conditions", visible=False)
                    for key, caption in captions.items():
                        caption.visible = view0.active[key]

                # -- shape and length ---------------------------------------
                with gr.Column(elem_classes=["mr-divide"]):
                    shape_preset = gr.Radio(
                        choices=_shape_choices(view0),
                        value="3:2",
                        label="Shape",
                        visible=view0.sizes_picture,
                        elem_id="shape",
                        elem_classes=["mr-chips"],
                    )
                    match_line = gr.Markdown("", visible=False, elem_classes=["mr-line"])
                    pinned_line = gr.Markdown(
                        view0.pinned_line or "",
                        visible=view0.pinned_line is not None,
                        elem_classes=["mr-line", "is-pinned"],
                    )
                    components.render_fields(
                        pool,
                        "enable_stage_2",
                        visible=view0.upscale,
                        elem_id="upscale",
                        label=_upscale_label(704, 480),
                    )
                    length_source = gr.Radio(
                        choices=[render.LENGTH_MATCH, render.LENGTH_SET],
                        value=render.LENGTH_MATCH,
                        label="Length",
                        visible=view0.timed_input,
                        elem_id="length-source",
                        elem_classes=["mr-chips", "mr-words"],
                    )
                    seconds = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=5.0,
                        label="Length (seconds)",
                        info="Written as a legal frame count; the exact frames are in Advanced.",
                        elem_id="seconds",
                        visible=not view0.timed_input,
                    )
                    components.render_fields(
                        pool,
                        "auto_duration",
                        visible=view0.active["auto_duration"],
                        elem_id="auto-length",
                    )
                    length_line = gr.Markdown(
                        "", visible=view0.timed_input, elem_classes=["mr-line"]
                    )

                # -- LoRA -----------------------------------------------------
                with gr.Accordion(
                    render.lora_summary_label([]),
                    open=False,
                    elem_id="lora",
                    elem_classes=["mr-acc"],
                ) as lora_accordion:
                    bay = components.lora_bay(
                        rows=handlers.MAX_STAGED_LORAS,
                        cap=f"{cap_gb:.0f} GB",
                        max_weight=max_weight,
                    )
                    components.render_fields(pool, "loras")
                lora_note = gr.Markdown(
                    render.lora_no_trigger(),
                    visible=not view0.active["prompt"],
                    elem_classes=["mr-hint", "mr-lora-note"],
                )

                # -- Advanced -------------------------------------------------
                with gr.Accordion(
                    "Advanced: exact size and frames, seed, decoder",
                    open=False,
                    elem_id="advanced",
                    elem_classes=["mr-acc"],
                ):
                    with gr.Row():
                        components.render_fields(
                            pool, "width", "height", visible=view0.sizes_picture, min_width=120
                        )
                    dims_hint = gr.Markdown(elem_classes=["mr-hint"], visible=view0.sizes_picture)
                    with gr.Row():
                        components.render_fields(pool, "num_frames", min_width=120)
                        components.render_fields(
                            pool,
                            "min_seconds",
                            "max_seconds",
                            visible=view0.active["min_seconds"],
                            min_width=120,
                        )
                    frames_hint = gr.Markdown(elem_classes=["mr-hint"])
                    snap_button = gr.Button(
                        "Snap to legal values", elem_classes=["mr-secondary"], size="sm"
                    )
                    components.render_fields(pool, "seed")
                    components.render_fields(
                        pool,
                        "decoder",
                        visible=view0.active["decoder"],
                        elem_id="decoder",
                        elem_classes=["mr-chips", "mr-words"],
                    )
                    decoder_line = gr.Markdown(
                        render.decoder_line("auto"),
                        visible=view0.active["decoder"],
                        elem_classes=["mr-hint"],
                    )

            # =============================================================
            # the rail
            # =============================================================
            with gr.Column(elem_id="rail", scale=2):
                sample0 = show.sample_for(DEFAULT_MODE)
                with gr.Column(elem_id="monitor"):
                    monitor_idle = gr.HTML(
                        render.monitor_idle(mode0),
                        visible=sample0 is None,
                        apply_default_css=False,
                        container=False,
                        padding=False,
                        elem_id="monitor-idle",
                    )
                    video_out = gr.Video(
                        value=str(sample0.path) if sample0 and sample0.is_video else None,
                        label="Result",
                        show_label=False,
                        interactive=False,
                        buttons=["download"],
                        autoplay=False,
                        loop=False,
                        elem_id="video-out",
                        visible=bool(sample0 and sample0.is_video),
                    )
                    audio_out = gr.Audio(
                        value=str(sample0.path) if sample0 and not sample0.is_video else None,
                        label="Result",
                        show_label=False,
                        interactive=False,
                        buttons=["download"],
                        waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                        elem_id="audio-out",
                        visible=bool(sample0 and not sample0.is_video),
                    )
                monitor_caption = gr.Markdown(
                    _sample_caption(mode0, sample0), elem_id="monitor-caption"
                )
                cost = gr.HTML(
                    "", elem_id="cost", apply_default_css=False, container=False, padding=False
                )
                with gr.Column(elem_id="commit"):
                    commit_line = gr.Markdown("", elem_id="commit-line")
                    generate_button = gr.Button(
                        "Generate", variant="primary", size="lg", elem_id="generate-btn"
                    )
                    reason = gr.Markdown("", elem_id="generate-reason")
                run_summary = gr.HTML(
                    "",
                    visible=False,
                    elem_id="run-summary",
                    apply_default_css=False,
                    container=False,
                    padding=False,
                )
                with gr.Row(elem_id="report-actions"):
                    reuse_seed = gr.Button(
                        "Reuse seed",
                        visible=False,
                        elem_classes=["mr-quiet"],
                        size="sm",
                        scale=0,
                        min_width=120,
                    )
                    use_prompt = gr.Button(
                        "Use this prompt",
                        visible=False,
                        elem_classes=["mr-quiet"],
                        size="sm",
                        scale=0,
                        min_width=140,
                    )
                status = gr.Markdown(render.READY, elem_id="run-status", visible=False)
                with gr.Column(elem_id="recent"):
                    recent_head = gr.Markdown(
                        "Recent runs", visible=False, elem_classes=["mr-recent-head"]
                    )
                    recent_rows = []
                    for _ in range(RECENT_LIMIT):
                        with gr.Row(elem_classes=["mr-recent-row"], equal_height=True):
                            text = gr.Markdown("", visible=False, elem_classes=["mr-recent-text"])
                            restore = gr.Button(
                                "Restore settings",
                                visible=False,
                                elem_classes=["mr-quiet"],
                                size="sm",
                                scale=0,
                                min_width=130,
                            )
                            reuse = gr.Button(
                                "Reuse seed",
                                visible=False,
                                elem_classes=["mr-quiet"],
                                size="sm",
                                scale=0,
                                min_width=100,
                            )
                        recent_rows.append((text, restore, reuse))

        components.assert_pool_complete(pool)

        # -- session state ----------------------------------------------------
        current_mode = gr.State(DEFAULT_MODE)
        parked = gr.State({})
        running = gr.State(False)
        monitor_state = gr.State({"kind": "sample" if sample0 else "idle", "mode": DEFAULT_MODE})
        recent = gr.State([])
        last_seed = gr.State(None)
        last_prompt = gr.State(None)

        field_inputs = [pool[key] for key in FIELD_KEYS]
        parked_keys = [spec.key for spec in PARKED_FIELDS]
        parked_components = [pool[key] for key in parked_keys]

        # =================================================================
        # readiness and cost: one review, fired by anything that moves it
        # =================================================================
        review_outputs = [cost, commit_line, generate_button, reason]

        def _review(mode_id: str, is_running: bool, *values: Any) -> list[Any]:
            try:
                facts = handlers.review(mode_id, _values(values))
            except Exception as exc:  # noqa: BLE001 -- the panel must still say something
                facts = render.CostFacts(
                    mode_name=render.mode_name(modes.get_mode(mode_id)),
                    state="refused",
                    reason=str(exc) or exc.__class__.__name__,
                )
            ready = facts.state == "ready" and not is_running
            text = render.readiness_reason(facts, running=bool(is_running))
            return [
                render.cost_panel(facts),
                render.commit_line(facts),
                gr.update(interactive=ready),
                text,
            ]

        review_inputs = [mode_selector, running, *field_inputs]

        def _wire_review(trigger: Any) -> None:
            trigger(fn=_review, inputs=review_inputs, outputs=review_outputs, **_PRIVATE)

        for component in [mode_selector, *field_inputs]:
            if isinstance(component, gr.State):
                continue
            _wire_review(component.change)

        # =================================================================
        # mode switch: park hidden values, show the mode's controls
        # =================================================================
        example_outputs = [*example_buttons, *([example_note] if example_note else [])]
        switch_outputs = [
            current_mode,
            parked,
            purpose,
            *parked_components,
            *[captions[key] for key in ("image", "video", "audio")],
            noise_choice,
            shape_preset,
            match_line,
            pinned_line,
            pool["enable_stage_2"],
            length_source,
            seconds,
            length_line,
            pool["num_frames"],
            pool["width"],
            pool["height"],
            dims_hint,
            pool["min_seconds"],
            pool["max_seconds"],
            lora_note,
            decoder_line,
            *example_outputs,
            monitor_idle,
            video_out,
            audio_out,
            monitor_caption,
            monitor_state,
        ]

        def _switch(
            previous: str,
            mode_id: str,
            parked_now: dict,
            noise: str,
            source: str,
            preset: str,
            monitor: dict,
            num_frames: Any,
            width: Any,
            height: Any,
            *parked_values: Any,
        ) -> list[Any]:
            view = handlers.mode_view(mode_id)
            mode = modes.get_mode(mode_id)
            before = dict(zip(parked_keys, parked_values, strict=True))
            if previous == mode_id:
                after, parked_next = before, dict(parked_now or {})
            else:
                after, parked_next = handlers.mode_switch(previous, mode_id, before, parked_now)
            for key in ("image", "video", "audio"):
                if not _exists(after.get(key)) and isinstance(after.get(key), str):
                    after[key] = None
            if view.active["noise_scale"]:
                after["noise_scale"] = _noise_value(noise)
            length = _length_state(view, source, after, num_frames)
            if view.timed_by_audio:
                after["auto_duration"] = source == render.LENGTH_MATCH

            updates = []
            for key in parked_keys:
                changed = after[key] is not before[key] and after[key] != before.get(key)
                kwargs: dict[str, Any] = {}
                if key in (
                    "prompt",
                    "enhance_prompt",
                    "image",
                    "video",
                    "audio",
                    "strength",
                    "auto_duration",
                    "decoder",
                ):
                    kwargs["visible"] = view.active[key]
                if key == "prompt":
                    kwargs["label"] = view.prompt_label
                    kwargs["placeholder"] = view.prompt_placeholder
                if changed or (key == "noise_scale") or (key == "auto_duration"):
                    kwargs["value"] = after[key]
                updates.append(gr.update(**kwargs))
            caption_updates = [
                gr.update(value=view.captions.get(k, ""), visible=view.active[k])
                for k in ("image", "video", "audio")
            ]
            match = _match_size_line(view, preset, after)
            shape_choices = _shape_choices(view)
            preset_value = preset if preset in shape_choices else "3:2"
            seconds_visible = (not view.timed_input and not after["auto_duration"]) or (
                view.timed_input and source == render.LENGTH_SET
            )
            frames_value = length["frames"] if length["frames"] is not None else num_frames
            example_updates = _example_updates(show, mode_id, len(example_buttons), example_note)
            monitor_updates, monitor_next = _monitor_for_mode(show, mode, monitor)
            return [
                mode_id,
                parked_next,
                view.purpose,
                *updates,
                *caption_updates,
                gr.update(
                    visible=view.active["noise_scale"],
                    value=_noise_choice(after["noise_scale"])
                    if view.active["noise_scale"]
                    else noise,
                ),
                gr.update(visible=view.sizes_picture, choices=shape_choices, value=preset_value),
                gr.update(visible=bool(match["line"]) and view.sizes_picture, value=match["line"]),
                gr.update(visible=view.pinned_line is not None, value=view.pinned_line or ""),
                gr.update(visible=view.upscale),
                gr.update(visible=view.timed_input),
                gr.update(
                    visible=seconds_visible,
                    value=handlers.seconds_for_frames(frames_value) if frames_value else 5.0,
                ),
                gr.update(visible=view.timed_input, value=length["line"]),
                gr.update(value=frames_value),
                gr.update(
                    visible=view.sizes_picture,
                    **({"value": match["size"][0]} if match["size"] else {}),
                ),
                gr.update(
                    visible=view.sizes_picture,
                    **({"value": match["size"][1]} if match["size"] else {}),
                ),
                gr.update(visible=view.sizes_picture),
                gr.update(visible=view.active["min_seconds"]),
                gr.update(visible=view.active["max_seconds"]),
                gr.update(visible=not view.active["prompt"]),
                gr.update(visible=view.active["decoder"]),
                *example_updates,
                *monitor_updates,
                monitor_next,
            ]

        switch_inputs = [
            current_mode,
            mode_selector,
            parked,
            noise_choice,
            length_source,
            shape_preset,
            monitor_state,
            pool["num_frames"],
            pool["width"],
            pool["height"],
            *parked_components,
        ]
        mode_selector.change(fn=_switch, inputs=switch_inputs, outputs=switch_outputs, **_PRIVATE)

        # =================================================================
        # shape, length and the exact fields in Advanced
        # =================================================================
        shape_inputs = [pool["width"], pool["height"], pool["num_frames"]]

        def _hints(width: Any, height: Any, frames: Any) -> list[Any]:
            dims, frame_text = handlers.on_shape_change(width, height, frames)
            w, h = handlers._as_int(width), handlers._as_int(height)
            label = _upscale_label(w, h) if w and h else render.FIELD_COPY["enable_stage_2"][0]
            return [dims, frame_text, gr.update(label=label)]

        for control in shape_inputs:
            control.change(
                fn=_hints,
                inputs=shape_inputs,
                outputs=[dims_hint, frames_hint, pool["enable_stage_2"]],
                **_PRIVATE,
            )

        def _on_preset(preset: str, image: Any, video: Any, mode_id: str) -> list[Any]:
            view = handlers.mode_view(mode_id)
            match = _match_size_line(view, preset, {"image": image, "video": video})
            if preset == render.MATCH_INPUT:
                size = match["size"]
                return [
                    gr.update(value=size[0]) if size else gr.update(),
                    gr.update(value=size[1]) if size else gr.update(),
                    gr.update(visible=True, value=match["line"]),
                ]
            size = handlers.preset_size(preset) or (704, 480)
            return [size[0], size[1], gr.update(visible=False, value="")]

        shape_preset.input(
            fn=_on_preset,
            inputs=[shape_preset, pool["image"], pool["video"], mode_selector],
            outputs=[pool["width"], pool["height"], match_line],
            **_PRIVATE,
        )

        def _on_exact_size(width: Any, height: Any) -> Any:
            return gr.update(value=handlers.preset_for(width, height))

        for control in (pool["width"], pool["height"]):
            control.input(
                fn=_on_exact_size,
                inputs=[pool["width"], pool["height"]],
                outputs=[shape_preset],
                **_PRIVATE,
            )

        def _on_media(
            mode_id: str, preset: str, source: str, image: Any, video: Any, audio: Any, frames: Any
        ) -> list[Any]:
            view = handlers.mode_view(mode_id)
            values = {"image": image, "video": video, "audio": audio}
            match = _match_size_line(view, preset, values)
            length = _length_state(view, source, values, frames)
            size_updates = (
                [gr.update(value=match["size"][0]), gr.update(value=match["size"][1])]
                if match["size"] and preset == render.MATCH_INPUT
                else [gr.update(), gr.update()]
            )
            frames_value = length["frames"] if length["frames"] is not None else frames
            return [
                *size_updates,
                gr.update(visible=bool(match["line"]) and view.sizes_picture, value=match["line"]),
                gr.update(value=length["line"]),
                gr.update(value=frames_value),
                gr.update(value=handlers.seconds_for_frames(frames_value)),
            ]

        media_inputs = [
            mode_selector,
            shape_preset,
            length_source,
            pool["image"],
            pool["video"],
            pool["audio"],
            pool["num_frames"],
        ]
        media_outputs = [
            pool["width"],
            pool["height"],
            match_line,
            length_line,
            pool["num_frames"],
            seconds,
        ]
        for key in ("image", "video", "audio"):
            pool[key].change(fn=_on_media, inputs=media_inputs, outputs=media_outputs, **_PRIVATE)

        def _on_length_source(
            mode_id: str, source: str, video: Any, audio: Any, frames: Any, auto: Any
        ) -> list[Any]:
            view = handlers.mode_view(mode_id)
            length = _length_state(view, source, {"video": video, "audio": audio}, frames)
            frames_value = length["frames"] if length["frames"] is not None else frames
            auto_value = (source == render.LENGTH_MATCH) if view.timed_by_audio else bool(auto)
            return [
                gr.update(value=auto_value),
                gr.update(
                    visible=source == render.LENGTH_SET,
                    value=handlers.seconds_for_frames(frames_value),
                ),
                gr.update(value=length["line"]),
                gr.update(value=frames_value),
            ]

        length_source.input(
            fn=_on_length_source,
            inputs=[
                mode_selector,
                length_source,
                pool["video"],
                pool["audio"],
                pool["num_frames"],
                pool["auto_duration"],
            ],
            outputs=[pool["auto_duration"], seconds, length_line, pool["num_frames"]],
            **_PRIVATE,
        )

        seconds.input(
            fn=lambda value: handlers.frames_for_seconds(value),
            inputs=[seconds],
            outputs=[pool["num_frames"]],
            **_PRIVATE,
        )
        pool["num_frames"].input(
            fn=lambda frames: handlers.seconds_for_frames(frames),
            inputs=[pool["num_frames"]],
            outputs=[seconds],
            **_PRIVATE,
        )

        def _on_auto(auto: Any, mode_id: str, source: str) -> Any:
            view = handlers.mode_view(mode_id)
            if view.timed_input:
                return gr.update(visible=source == render.LENGTH_SET)
            return gr.update(visible=not bool(auto))

        pool["auto_duration"].change(
            fn=_on_auto,
            inputs=[pool["auto_duration"], mode_selector, length_source],
            outputs=[seconds],
            **_PRIVATE,
        )

        pool["decoder"].change(
            fn=render.decoder_line, inputs=[pool["decoder"]], outputs=[decoder_line], **_PRIVATE
        )
        noise_choice.input(
            fn=_noise_value, inputs=[noise_choice], outputs=[pool["noise_scale"]], **_PRIVATE
        )

        def _snap(width: Any, height: Any, frames: Any) -> list[Any]:
            w, h, f, dims, frame_text = handlers.on_snap(width, height, frames)
            return [
                w,
                h,
                f,
                dims,
                frame_text,
                handlers.preset_for(w, h),
                handlers.seconds_for_frames(f),
            ]

        snap_button.click(
            fn=_snap,
            inputs=shape_inputs,
            outputs=[*shape_inputs, dims_hint, frames_hint, shape_preset, seconds],
            **_PRIVATE,
        )

        # =================================================================
        # LoRA bay
        # =================================================================
        row_outputs = [bay["clear"]]
        for row in bay["rows"]:
            row_outputs += [row["name"], row["weight"], row["remove"]]

        def _lora_rows(staged: list) -> list[Any]:
            entries = list(staged or [])
            out = [gr.update(visible=bool(entries))]
            for index in range(handlers.MAX_STAGED_LORAS):
                if index < len(entries):
                    entry = entries[index]
                    out += [
                        gr.update(
                            visible=True, value=render.shown_name(str(entry.get("source", "")))
                        ),
                        gr.update(visible=True, value=float(entry.get("weight", 1.0))),
                        gr.update(visible=True),
                    ]
                else:
                    out += [
                        gr.update(visible=False, value=""),
                        gr.update(visible=False),
                        gr.update(visible=False),
                    ]
            return out

        def _lora_label(staged: list) -> Any:
            return gr.update(label=render.lora_summary_label(list(staged or [])))

        def _after_lora_change(dependency: Any) -> None:
            (
                dependency.then(
                    fn=_lora_rows, inputs=[pool["loras"]], outputs=row_outputs, **_PRIVATE
                )
                .then(fn=_lora_label, inputs=[pool["loras"]], outputs=[lora_accordion], **_PRIVATE)
                .then(fn=_review, inputs=review_inputs, outputs=review_outputs, **_PRIVATE)
            )

        lora_outputs = [bay["verdict"], pool["loras"], bay["applied_summary"]]
        checked = (
            bay["validate"]
            .click(
                fn=handlers.on_validate_start,
                inputs=[bay["repo_id"], bay["upload"]],
                outputs=[bay["verdict"]],
                **_PRIVATE,
            )
            .then(
                fn=_on_validate_lora,
                inputs=[bay["repo_id"], bay["upload"], pool["loras"]],
                outputs=lora_outputs,
                api_name="on_validate_lora",
                show_progress_on=[bay["verdict"]],
            )
        )
        _after_lora_change(checked)

        uploaded = (
            bay["upload"]
            .upload(
                fn=lambda upload: handlers.on_validate_start(None, upload),
                inputs=[bay["upload"]],
                outputs=[bay["verdict"]],
                **_PRIVATE,
            )
            .then(
                fn=lambda upload, staged: handlers.on_validate_lora(None, upload, staged),
                inputs=[bay["upload"], pool["loras"]],
                outputs=lora_outputs,
                api_visibility="private",
                show_progress_on=[bay["verdict"]],
            )
        )
        uploaded = uploaded.then(fn=lambda: None, inputs=None, outputs=[bay["upload"]], **_PRIVATE)
        _after_lora_change(uploaded)

        cleared = bay["clear"].click(
            fn=handlers.on_clear_loras, inputs=None, outputs=lora_outputs, **_PRIVATE
        )
        _after_lora_change(cleared)

        for index, row in enumerate(bay["rows"]):
            removed = row["remove"].click(
                fn=lambda staged, i=index: handlers.on_remove_lora(staged, i),
                inputs=[pool["loras"]],
                outputs=[pool["loras"]],
                **_PRIVATE,
            )
            _after_lora_change(removed)
            weighted = row["weight"].release(
                fn=lambda staged, weight, i=index: handlers.on_lora_weight(staged, i, weight),
                inputs=[pool["loras"], row["weight"]],
                outputs=[pool["loras"]],
                **_PRIVATE,
            )
            weighted.then(
                fn=_lora_label, inputs=[pool["loras"]], outputs=[lora_accordion], **_PRIVATE
            ).then(fn=_review, inputs=review_inputs, outputs=review_outputs, **_PRIVATE)

        # =================================================================
        # Generate: start, /on_generate, finish
        # =================================================================
        start_outputs = [
            generate_button,
            reason,
            running,
            monitor_idle,
            video_out,
            audio_out,
            monitor_caption,
            status,
            run_summary,
            reuse_seed,
            use_prompt,
            monitor_state,
        ]

        def _start(mode_id: str, enhance: Any, prompt: Any) -> list[Any]:
            mode = modes.get_mode(mode_id)
            return [
                gr.update(interactive=False, value="Generating…"),
                render.readiness_reason(
                    render.CostFacts(mode_name="", state="ready"), running=True
                ),
                True,
                gr.update(visible=False),
                gr.update(value=None, visible=True),
                gr.update(value=None, visible=False),
                render.monitor_caption(kind="running", mode=mode),
                gr.update(value=handlers.on_generate_start(mode_id, enhance, prompt), visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                {"kind": "running", "mode": mode_id},
            ]

        finish_outputs = [
            generate_button,
            reason,
            running,
            cost,
            commit_line,
            monitor_idle,
            video_out,
            audio_out,
            monitor_caption,
            run_summary,
            reuse_seed,
            last_seed,
            use_prompt,
            last_prompt,
            monitor_state,
            recent,
            recent_head,
            *[c for row in recent_rows for c in row],
            status,
        ]

        def _finish(request: gr.Request, mode_id: str, recent_now: list, *values: Any) -> list:
            outcome = _recall(request)
            # The step before this one returned nothing this page can use: Gradio
            # itself failed it (a file it would not serve, a lost connection).
            # Say so loudly rather than leave "Running" on screen.
            status_update = (
                gr.update(value=render.lost_run(), visible=True) if outcome is None else gr.update()
            )
            mode = modes.get_mode(mode_id)
            review_out = _review(mode_id, False, *values)
            entries = list(recent_now or [])
            ok = (
                outcome is not None
                and outcome.result is not None
                and outcome.error is None
                and (outcome.video is not None or outcome.audio is not None)
            )
            if ok:
                result = outcome.result
                params = outcome.params
                facts = handlers.run_facts(outcome)
                entries.insert(0, _recent_entry(mode_id, result, params, facts, outcome))
                entries = entries[:RECENT_LIMIT]
                has_video = outcome.video is not None
                monitor = [
                    gr.update(visible=False),
                    gr.update(value=outcome.video, visible=has_video, autoplay=True),
                    gr.update(
                        value=outcome.audio if not has_video else None, visible=not has_video
                    ),
                    render.monitor_caption(kind="run", mode=mode, seed=result.seed),
                ]
                summary = gr.update(value=render.run_summary(facts), visible=True)
                seed_button = gr.update(value="Reuse seed", visible=True)
                seed_value = int(result.seed)
                sent = facts.prompt_sent
                differs = bool(sent) and sent.strip() != str(params.get("prompt") or "").strip()
                prompt_button = gr.update(visible=differs)
                prompt_value = sent if differs else None
                monitor_next = {"kind": "run", "mode": mode_id}
            else:
                monitor = [
                    gr.update(value=render.monitor_idle(mode), visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    render.monitor_caption(kind="failed"),
                ]
                summary = gr.update(visible=False)
                seed_button = gr.update(visible=False)
                seed_value = None
                prompt_button = gr.update(visible=False)
                prompt_value = None
                monitor_next = {"kind": "failed", "mode": mode_id}
            return [
                gr.update(value="Generate", interactive=review_out[2]["interactive"]),
                review_out[3],
                False,
                review_out[0],
                review_out[1],
                *monitor,
                summary,
                seed_button,
                seed_value,
                prompt_button,
                prompt_value,
                monitor_next,
                entries,
                gr.update(visible=bool(entries)),
                *_recent_rows(entries),
                status_update,
            ]

        generation = generate_button.click(
            fn=_start,
            inputs=[mode_selector, pool["enhance_prompt"], pool["prompt"]],
            outputs=start_outputs,
            js=_SCROLL_TO_MONITOR,
            **_PRIVATE,
        ).then(
            fn=_on_generate,
            inputs=[mode_selector, *field_inputs],
            outputs=[video_out, audio_out, status],
            api_name="on_generate",
            show_progress_on=[video_out],
        )
        # Gradio 6.24 fires `.then` listeners only when the step completes and
        # `.failure` listeners only when Gradio itself fails it, so the step that
        # brings Generate back is wired to both: exactly one of them runs.
        finish = {
            "fn": _finish,
            "inputs": [mode_selector, recent, *field_inputs],
            "outputs": finish_outputs,
            **_PRIVATE,
        }
        generation.then(**finish)
        generation.failure(**finish)

        use_prompt.click(
            fn=lambda sent: sent if sent else gr.update(),
            inputs=[last_prompt],
            outputs=[pool["prompt"]],
            **_PRIVATE,
        )
        reuse_seed.click(
            fn=lambda seed: seed if seed is not None else -1,
            inputs=[last_seed],
            outputs=[pool["seed"]],
            **_PRIVATE,
        )

        # =================================================================
        # recent runs: restore settings, reuse seed
        # =================================================================
        restore_outputs = [
            current_mode,
            mode_selector,
            *[pool[k] for k in _RESTORED],
            shape_preset,
            seconds,
            noise_choice,
            length_source,
        ]

        for index, (_text, restore, reuse) in enumerate(recent_rows):
            restore.click(
                fn=lambda entries, i=index: _restore(entries, i),
                inputs=[recent],
                outputs=restore_outputs,
                **_PRIVATE,
            )
            reuse.click(
                fn=lambda entries, i=index: (
                    entries[i]["seed"] if i < len(entries or []) else gr.update()
                ),
                inputs=[recent],
                outputs=[pool["seed"]],
                **_PRIVATE,
            )

        # =================================================================
        # examples: fill the form, run nothing
        # =================================================================
        if example_buttons:
            example_outputs_fill = [
                *[pool[k] for k in _EXAMPLE_FIELDS],
                shape_preset,
                seconds,
                noise_choice,
                length_source,
            ]
            for index, button in enumerate(example_buttons):
                button.click(
                    fn=lambda mode_id, i=index: _fill_example(show, mode_id, i),
                    inputs=[mode_selector],
                    outputs=example_outputs_fill,
                    **_PRIVATE,
                )

        # =================================================================
        # first paint
        # =================================================================
        demo.load(fn=_review, inputs=review_inputs, outputs=review_outputs, **_PRIVATE)
        demo.load(
            fn=_hints,
            inputs=shape_inputs,
            outputs=[dims_hint, frames_hint, pool["enable_stage_2"]],
            **_PRIVATE,
        )

    return demo


# ---------------------------------------------------------------------------
# helpers for build_ui
# ---------------------------------------------------------------------------

_SCROLL_TO_MONITOR = """
(...args) => {
  if (window.innerWidth < 1024) {
    document.getElementById('monitor')?.scrollIntoView({behavior: 'instant', block: 'start'});
  }
  return args;
}
"""

#: Fields a "Restore settings" writes back (media stays as it is).
_RESTORED = (
    "prompt",
    "enhance_prompt",
    "width",
    "height",
    "auto_duration",
    "num_frames",
    "min_seconds",
    "max_seconds",
    "enable_stage_2",
    "seed",
    "strength",
    "noise_scale",
    "decoder",
)

#: Fields an example may write.
_EXAMPLE_FIELDS = (
    "prompt",
    "image",
    "video",
    "audio",
    "enhance_prompt",
    "width",
    "height",
    "num_frames",
    "auto_duration",
    "enable_stage_2",
    "seed",
    "strength",
    "noise_scale",
)


def _upscale_label(width: Any, height: Any) -> str:
    base = render.FIELD_COPY["enable_stage_2"][0]
    try:
        return f"{base}: returns {int(width) * 2} × {int(height) * 2}"
    except (TypeError, ValueError):
        return base


def _shape_choices(view: handlers.ModeView) -> list[str]:
    names = [name for name, _w, _h in render.SHAPE_PRESETS]
    return names + ([render.MATCH_INPUT] if view.match_input else [])


def _match_size_line(view: handlers.ModeView, preset: str, values: dict) -> dict:
    if preset != render.MATCH_INPUT or not view.match_input:
        return {"size": None, "line": ""}
    image = values.get("image") if view.active.get("image") else None
    video = values.get("video") if view.active.get("video") else None
    source = handlers.media_size(image=image, video=video)
    if source is None:
        what = (
            "image or clip"
            if view.active.get("image") and view.active.get("video")
            else ("image" if view.active.get("image") else "clip")
        )
        return {"size": None, "line": f"Add your {what} to match its shape."}
    size = handlers.match_size(*source)
    what = "image" if image is not None else "clip"
    return {
        "size": size,
        "line": render.match_input_line(
            width=size[0],
            height=size[1],
            source_width=source[0],
            source_height=source[1],
            what=what,
        ),
    }


def _length_state(view: handlers.ModeView, source: str, values: dict, frames: Any) -> dict:
    """The length line and, for a clip, the matched frame count."""
    from .. import constraints

    if not view.timed_input:
        return {"frames": None, "line": ""}
    if source != render.LENGTH_MATCH:
        return {"frames": None, "line": "Set with the slider below; Advanced has the exact frames."}
    audio = view.timed_by_audio
    path = values.get("audio") if audio else values.get("video")
    clip = handlers.clip_seconds(path, kind="audio" if audio else "video") if path else None
    matched = handlers.frames_matching_clip(clip)
    line = render.match_clip_line(
        clip_seconds=clip,
        frames=matched,
        fps=handlers.FRAME_RATE,
        audio=audio,
        cap_seconds=constraints.MAX_SECONDS,
    )
    return {"frames": None if audio else matched, "line": line}


def _sample_caption(mode: Any, sample: showcase.Sample | None) -> str:
    if sample is None:
        return render.READY
    return render.monitor_caption(
        kind="sample",
        mode=mode,
        seed=sample.seed,
        settings=sample.settings,
        caption=sample.caption,
    )


def _monitor_for_mode(show: showcase.Showcase, mode: Any, monitor: dict) -> tuple[list, dict]:
    """Swap the idle monitor to the new mode's sample; never hide a finished run."""
    kind = (monitor or {}).get("kind")
    if kind in ("run", "running"):
        return [gr.update(), gr.update(), gr.update(), gr.update()], monitor
    sample = show.sample_for(mode.id)
    if sample is None:
        caption = render.READY if kind != "failed" else render.monitor_caption(kind="failed")
        return (
            [
                gr.update(value=render.monitor_idle(mode), visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                caption,
            ],
            {"kind": "idle" if kind != "failed" else "failed", "mode": mode.id},
        )
    return (
        [
            gr.update(value=render.monitor_idle(mode), visible=False),
            gr.update(
                value=str(sample.path) if sample.is_video else None,
                visible=sample.is_video,
                autoplay=False,
            ),
            gr.update(
                value=None if sample.is_video else str(sample.path), visible=not sample.is_video
            ),
            _sample_caption(mode, sample),
        ],
        {"kind": "sample", "mode": mode.id},
    )


def _example_updates(show: showcase.Showcase, mode_id: str, slots: int, note: Any) -> list:
    if not slots:
        return []
    examples = show.for_mode(mode_id)
    out = []
    for slot in range(slots):
        if slot < len(examples):
            out.append(gr.update(value=examples[slot].label, visible=True))
        else:
            out.append(gr.update(visible=False))
    if note is not None:
        out.append(gr.update(visible=bool(examples)))
    return out


def _fill_example(show: showcase.Showcase, mode_id: str, index: int) -> list:
    """Write the example's own values; leave every other control untouched."""
    examples = show.for_mode(mode_id)
    out = handlers.example_values(examples[index], mode_id) if index < len(examples) else {}
    keys = [*_EXAMPLE_FIELDS, "preset", "seconds", "noise_choice", "length_source"]
    return [gr.update(value=out[k]) if k in out else gr.update() for k in keys]


def _recent_entry(
    mode_id: str, result: Any, params: dict, facts: render.RunFacts, outcome: handlers.RunOutcome
) -> dict:
    settings = {k: params.get(k) for k in _RESTORED}
    return {
        "mode": mode_id,
        "seed": int(result.seed),
        "size": facts.returned_size,
        "seconds": facts.seconds,
        "at": time.strftime("%H:%M"),
        "settings": settings,
        "video": outcome.video,
        "audio": outcome.audio,
    }


def _recent_rows(entries: list) -> list:
    out = []
    for index in range(RECENT_LIMIT):
        if index < len(entries):
            entry = entries[index]
            mode = modes.get_mode(entry["mode"])
            label = render.recent_run_label(
                mode=mode,
                seed=entry["seed"],
                size=entry["size"],
                seconds=entry["seconds"],
                at=entry["at"],
            )
            out += [
                gr.update(value=label, visible=True),
                gr.update(visible=True),
                gr.update(value="Reuse seed", visible=True),
            ]
        else:
            out += [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
    return out


def _restore(entries: list, index: int) -> list:
    entries = list(entries or [])
    count = 2 + len(_RESTORED) + 4
    if index >= len(entries):
        return [gr.update() for _ in range(count)]
    entry = entries[index]
    settings = entry["settings"]
    mode_id = entry["mode"]
    view = handlers.mode_view(mode_id)
    preset = handlers.preset_for(settings.get("width"), settings.get("height"))
    frames = settings.get("num_frames")
    noise = _noise_choice(settings.get("noise_scale"))
    return [
        mode_id,
        mode_id,
        *[gr.update(value=settings.get(k)) for k in _RESTORED],
        gr.update(value=preset),
        gr.update(value=handlers.seconds_for_frames(frames)) if frames else gr.update(),
        gr.update(value=noise) if view.active["noise_scale"] else gr.update(),
        gr.update(value=render.LENGTH_MATCH if settings.get("auto_duration") else render.LENGTH_SET)
        if view.timed_by_audio
        else gr.update(),
    ]
