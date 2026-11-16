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

import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import gradio as gr

from .. import modes
from . import chrome, components, handlers, render, showcase
from .fields import FIELD_KEYS, PARKED_FIELDS, field_by_key

__all__ = ["DEFAULT_MODE", "build_ui", "served_outputs"]

_log = logging.getLogger("ltx25_studio")

#: The media kinds, in the order the switch's caption outputs list them.
_MEDIA = ("image", "video", "audio")

#: Parked fields whose visibility is the mode's own (``ModeView.active``).
_SHOWN_BY_MODE = (
    "prompt",
    "enhance_prompt",
    "image",
    "video",
    "audio",
    "strength",
    "auto_duration",
    "decoder",
)

_DEFAULT_FRAMES = field_by_key("num_frames").default


def _enhance_seconds() -> int:
    from .. import zerogpu

    return int(zerogpu.ENHANCE_DURATION_SECONDS)


_ENHANCE_SECONDS = _enhance_seconds()

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

#: The LoRA strength slider's step. Its range is the LoRA module's own (more than 0,
#: at most ``staging.MAX_WEIGHT``), so its lowest stop is one step above 0. The range
#: is set from the module when a row is shown (``lora_strength_range``): the package
#: stays lazy until an adapter is checked.
LORA_STEP = 0.05


def lora_strength_range() -> tuple[float, float, float]:
    """``(lowest, highest, default)`` strength, from ``lora.staging`` itself."""
    from ..lora import staging

    return (LORA_STEP, float(staging.MAX_WEIGHT), float(staging.DEFAULT_WEIGHT))


def served_outputs(output_root: str | Path) -> dict[str, list[str]]:
    """The ``launch`` arguments Gradio needs to serve the results written to ``output_root``.

    Gradio serves a file a step returns only from the working directory, the system temp
    dir, its own cache or ``allowed_paths``. The default output folder is under the
    working directory and needs nothing; a folder outside both is passed, and only it.
    """
    root = Path(output_root).resolve()
    for served in (os.getcwd(), tempfile.gettempdir()):
        if root.is_relative_to(Path(served).resolve()):
            return {}
    _log.info(
        "results are written to %s, outside the working directory and the temp dir, so it "
        "is passed to launch as allowed_paths",
        root,
    )
    return {"allowed_paths": [str(root)]}


_PRIVATE = {"api_visibility": "private", "show_progress": "hidden"}

# ---------------------------------------------------------------------------
# the run outcome, handed from /on_generate to the chained finish step, and
# the message of an exception Gradio raised running either of them
# ---------------------------------------------------------------------------

_OUTCOMES: OrderedDict[str, handlers.RunOutcome] = OrderedDict()
_FAILURES: OrderedDict[str, str] = OrderedDict()
_OUTCOMES_LOCK = threading.Lock()
_OUTCOMES_KEPT = 64


def _session(request: Any) -> str | None:
    return getattr(request, "session_hash", None) if request is not None else None


def _keep(store: OrderedDict, key: str | None, value: Any) -> None:
    if key is None:
        return
    with _OUTCOMES_LOCK:
        store[key] = value
        store.move_to_end(key)
        while len(store) > _OUTCOMES_KEPT:
            store.popitem(last=False)


def _take(store: OrderedDict, request: Any) -> Any:
    key = _session(request)
    if key is None:
        return None
    with _OUTCOMES_LOCK:
        return store.pop(key, None)


def _remember(request: Any, outcome: handlers.RunOutcome) -> None:
    _keep(_OUTCOMES, _session(request), outcome)


def _recall(request: Any) -> handlers.RunOutcome | None:
    return _take(_OUTCOMES, request)


def _note_failure(session_hash: str | None, exc: BaseException) -> None:
    _keep(_FAILURES, session_hash, str(exc).strip() or type(exc).__name__)


def _recall_failure(request: Any) -> str | None:
    return _take(_FAILURES, request)


def _forget(request: Any) -> None:
    """A new run starts: nothing left from an earlier one is read as its own."""
    _recall(request)
    _recall_failure(request)


def _keep_failures(demo: gr.Blocks, steps: tuple) -> None:
    """Keep, per session, the message of an exception Gradio raises running one of
    ``steps`` (the dependencies an event listener returned): preparing its inputs, in
    its function, or processing what it returned.

    Gradio prints that exception and sends the page only "Error" (``show_error`` is
    off), and the ``.failure`` listener it then fires is given no detail. The report
    that listener shows quotes the message kept here.
    """
    watched = {step["id"] for step in steps}
    process_api = demo.process_api

    async def process_api_keeping_failures(*args: Any, **kwargs: Any) -> Any:
        try:
            return await process_api(*args, **kwargs)
        except Exception as exc:
            block_fn = kwargs.get("block_fn", args[0] if args else None)
            fn_id = block_fn if isinstance(block_fn, int) else getattr(block_fn, "_id", None)
            if fn_id in watched:
                _note_failure(kwargs.get("session_hash"), exc)
            raise

    demo.process_api = process_api_keeping_failures


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


def _clip_value(choice: Any) -> float:
    levels = dict(render.CLIP_CHOICES)
    return levels.get(choice, levels[render.CLIP_DEFAULT])


def _clip_choice(value: Any) -> str:
    for name, level in render.CLIP_CHOICES:
        if abs(level - float(value or 0)) < 1e-6:
            return name
    return render.CLIP_DEFAULT


def build_ui() -> gr.Blocks:
    """Assemble the Studio. Returns an unlaunched :class:`gradio.Blocks`."""
    show = showcase.load_showcase(known_modes=tuple(modes.MODES))
    if show.root is not None:
        gr.set_static_paths([show.root])

    from .. import zerogpu
    from ..config import load_config

    cap_gb = load_config().lora_max_bytes / 1e9
    multiplier = handlers.quota_multiplier()
    view0 = handlers.mode_view(DEFAULT_MODE)
    mode0 = modes.get_mode(DEFAULT_MODE)
    # Where runs happen. Only ZeroGPU has a quota, a reservation and a bill: anywhere else
    # the page says where a run happens and nothing about a meter.
    _device, on_zerogpu = handlers.run_platform()
    upscale_seconds = zerogpu.estimate_duration(enable_stage_2=True) - zerogpu.estimate_duration()

    pool: dict[str, Any] = {}
    captions: dict[str, Any] = {}

    with gr.Blocks(title=_TITLE, fill_width=True, delete_cache=DELETE_CACHE) as demo:
        gr.HTML(
            value=chrome.identity_html(
                quota_text=render.quota_explainer(
                    multiplier=multiplier, example_reserve=zerogpu.estimate_duration()
                )
                if on_zerogpu
                else None
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
                        chrome.matrix_frame_html(on_zerogpu=on_zerogpu),
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
                sample_jump = gr.Button(
                    render.SAMPLE_JUMP,
                    visible=show.sample_for(DEFAULT_MODE) is not None,
                    elem_id="sample-jump",
                    elem_classes=["mr-quiet"],
                    size="sm",
                )

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

                # -- what you give: in every mode the inputs it requires come
                # first and the ones it may take after (DOM order = focus order)
                with gr.Column(elem_classes=["mr-divide"]):
                    noise_choice = clip_choice = None
                    for kind in handlers.media_order():
                        components.render_fields(
                            pool,
                            kind,
                            captions=captions,
                            visible=view0.active[kind],
                            elem_id=f"in-{kind}",
                            label=handlers.media_label(mode0, kind),
                        )
                        if kind == "video":
                            components.render_fields(
                                pool,
                                "strength",
                                visible=view0.active["strength"],
                                elem_id="lever-strength",
                            )
                            clip_choice = gr.Radio(
                                choices=[name for name, _ in render.CLIP_CHOICES],
                                value=render.CLIP_DEFAULT,
                                label=render.CLIP_LEVER[0],
                                info=render.CLIP_LEVER[1],
                                visible=view0.varies_clip,
                                elem_id="lever-clip",
                                elem_classes=["mr-chips", "mr-words"],
                            )
                        if kind == "audio":
                            noise_choice = gr.Radio(
                                choices=[name for name, _ in render.NOISE_CHOICES],
                                value="Moderately",
                                label=render.FIELD_COPY["noise_scale"][0],
                                info=render.FIELD_COPY["noise_scale"][1],
                                visible=view0.varies_sound,
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
                        info=render.enhance_info(mode0, _ENHANCE_SECONDS, on_zerogpu=on_zerogpu),
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
                        info=render.upscale_info(
                            upscale_seconds=upscale_seconds, on_zerogpu=on_zerogpu
                        ),
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
                        step=0.1,
                        value=handlers.seconds_for_frames(_DEFAULT_FRAMES),
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
                        step=LORA_STEP,
                        on_zerogpu=on_zerogpu,
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
                    # The frame count while the length is set; the shortest and longest
                    # seconds while the model picks it. Never both.
                    lengths0 = handlers.length_fields_view(DEFAULT_MODE, False)
                    with gr.Row():
                        components.render_fields(
                            pool, "num_frames", visible=lengths0["num_frames"], min_width=120
                        )
                        components.render_fields(
                            pool,
                            "min_seconds",
                            "max_seconds",
                            visible=lengths0["min_seconds"],
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
                        elem_id="decoder-line",
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
                status = gr.Markdown("", elem_id="run-status")
                with gr.Column(elem_id="recent"):
                    recent_head = gr.Markdown(
                        "Recent runs", visible=False, elem_classes=["mr-recent-head"]
                    )
                    recent_rows = []
                    for _ in range(RECENT_LIMIT):
                        with gr.Row(elem_classes=["mr-recent-row"], equal_height=True):
                            text = gr.Markdown("", visible=False, elem_classes=["mr-recent-text"])
                            show_run = gr.Button(
                                "Show",
                                visible=False,
                                elem_classes=["mr-quiet"],
                                size="sm",
                                scale=0,
                                min_width=70,
                            )
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
                        recent_rows.append((text, show_run, restore, reuse))

        components.assert_pool_complete(pool)

        # -- session state ----------------------------------------------------
        current_mode = gr.State(DEFAULT_MODE)
        parked = gr.State({})
        running = gr.State(False)
        monitor_state = gr.State({"kind": "sample" if sample0 else "idle", "mode": DEFAULT_MODE})
        recent = gr.State([])
        last_seed = gr.State(None)
        last_prompt = gr.State(None)
        # The frame count the user set. "Match my clip" writes a clip's count into
        # the shared frames field; a mode without a timed input gets this back.
        user_frames = gr.State(_DEFAULT_FRAMES)
        # The failure card on screen, as {"mode", "at", "headline"}; None when the status
        # holds anything else. A switch to another mode collapses the card to one line.
        last_failure = gr.State(None)

        field_inputs = [pool[key] for key in FIELD_KEYS]
        parked_keys = [spec.key for spec in PARKED_FIELDS]
        parked_components = [pool[key] for key in parked_keys]

        # =================================================================
        # readiness and cost: one review, fired by anything that moves it
        # =================================================================
        review_outputs = [cost, commit_line, generate_button, reason, pool["enable_stage_2"]]

        def _review(mode_id: str, is_running: bool, source: str, *values: Any) -> list[Any]:
            try:
                facts = handlers.review(mode_id, _values(values), length_source=source)
            except Exception as exc:  # noqa: BLE001 -- the panel must still say something
                facts = render.CostFacts(
                    mode_name=modes.get_mode(mode_id).label,
                    state="refused",
                    reason=str(exc) or exc.__class__.__name__,
                    on_zerogpu=on_zerogpu,
                )
            ready = facts.state == "ready" and not is_running
            text = render.readiness_reason(facts, running=bool(is_running))
            # Upscale where this machine cannot run it for this request: parked, unticked,
            # with the backend's reason, so Generate is never pressed into that refusal.
            info = render.upscale_info(
                upscale_seconds=upscale_seconds,
                on_zerogpu=on_zerogpu,
                refused=facts.upscale_refused,
            )
            upscale = (
                gr.update(interactive=False, value=False, info=info)
                if facts.upscale_refused
                else gr.update(interactive=True, info=info)
            )
            return [
                render.cost_panel(facts),
                render.commit_markup(facts),
                gr.update(interactive=ready),
                text,
                upscale,
            ]

        review_inputs = [mode_selector, running, length_source, *field_inputs]

        def _wire_review(trigger: Any) -> None:
            trigger(fn=_review, inputs=review_inputs, outputs=review_outputs, **_PRIVATE)

        for component in [mode_selector, length_source, *field_inputs]:
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
            *[captions[key] for key in _MEDIA],
            noise_choice,
            clip_choice,
            shape_preset,
            match_line,
            pinned_line,
            pool["enable_stage_2"],
            length_source,
            seconds,
            length_line,
            pool["num_frames"],
            dims_hint,
            lora_note,
            decoder_line,
            user_frames,
            *example_outputs,
            monitor_idle,
            video_out,
            audio_out,
            monitor_caption,
            monitor_state,
            sample_jump,
            status,
            last_failure,
        ]
        #: What a switch reads besides the mode it goes to; the first is the mode it leaves.
        switch_state = [
            current_mode,
            parked,
            noise_choice,
            clip_choice,
            length_source,
            shape_preset,
            monitor_state,
            pool["num_frames"],
            pool["enable_stage_2"],
            user_frames,
            last_failure,
            *parked_components,
        ]

        def _mode_change(
            mode_id: str,
            previous: str,
            parked_now: dict,
            noise: str,
            clip: str,
            source: str,
            preset: str,
            monitor: dict,
            num_frames: Any,
            upscale: Any,
            frames_user: Any,
            failure: dict | None,
            *parked_values: Any,
            restored: dict | None = None,
        ) -> list[Any]:
            """Every control's state after moving to ``mode_id``.

            The values the new mode does not use are parked by
            ``handlers.mode_switch`` against the mode actually left. With
            ``restored`` (a recent run's settings), that run's own fields are
            written over the result; its media are not, so what the restored
            mode does not take stays parked. A failure card from another mode's
            run becomes one line naming that run.
            """
            view = handlers.mode_view(mode_id)
            mode = modes.get_mode(mode_id)
            before = dict(zip(parked_keys, parked_values, strict=True))
            if previous == mode_id:
                after, parked_next = dict(before), dict(parked_now or {})
            else:
                after, parked_next = handlers.mode_switch(previous, mode_id, before, parked_now)
            frames_user = handlers._as_int(frames_user) or _DEFAULT_FRAMES
            upscale = bool(upscale)
            if restored is not None:
                for key in parked_keys:
                    if key in restored and key not in _MEDIA and view.active.get(key):
                        after[key] = restored[key]
                if restored.get("num_frames"):
                    frames_user = int(restored["num_frames"])
                if view.upscale:
                    upscale = bool(restored.get("enable_stage_2"))
                if view.varies_sound:
                    noise = _noise_choice(restored.get("noise_scale"))
                if view.varies_clip:
                    clip = _clip_choice(restored.get("noise_scale"))
                if view.timed_by_audio:
                    matched = restored.get("auto_duration")
                    source = render.LENGTH_MATCH if matched else render.LENGTH_SET
                elif view.timed_input:
                    source = render.LENGTH_SET
                if view.sizes_picture:
                    preset = handlers.preset_for(after["width"], after["height"])
            for key in _MEDIA:
                if not _exists(after.get(key)) and isinstance(after.get(key), str):
                    after[key] = None
            if view.varies_sound:
                after["noise_scale"] = _noise_value(noise)
            if view.varies_clip:
                after["noise_scale"] = _clip_value(clip)
            if view.timed_by_audio:
                after["auto_duration"] = source == render.LENGTH_MATCH
            length = _length_state(view, source, after, num_frames)
            frames_value = length["frames"] if length["frames"] is not None else frames_user
            lengths = handlers.length_fields_view(mode_id, after["auto_duration"])
            match = _match_size_line(view, preset, after)
            if match["size"]:
                after["width"], after["height"] = match["size"]
            shape_choices = _shape_choices(view)
            preset_value = (
                preset
                if preset in shape_choices
                else handlers.preset_for(after["width"], after["height"])
            )
            seconds_visible = (
                view.timed_input and source == render.LENGTH_SET and not view.length_from_clip
            ) or (not view.timed_input and not after["auto_duration"])
            _shown, decoder_free, decoder_text = handlers.decoder_view(
                mode_id, upscale, after["decoder"]
            )

            updates = []
            for key in parked_keys:
                changed = after[key] is not before[key] and after[key] != before.get(key)
                kwargs: dict[str, Any] = {}
                if key in _SHOWN_BY_MODE:
                    kwargs["visible"] = view.active[key]
                if key in ("width", "height"):
                    kwargs["visible"] = view.sizes_picture
                if key in ("min_seconds", "max_seconds"):
                    kwargs["visible"] = lengths[key]
                if key == "prompt":
                    kwargs["label"] = view.prompt_label
                    kwargs["placeholder"] = view.prompt_placeholder
                if key == "enhance_prompt" and view.active[key]:
                    kwargs["info"] = render.enhance_info(
                        mode, _ENHANCE_SECONDS, on_zerogpu=on_zerogpu
                    )
                if key in _MEDIA and view.active[key]:
                    kwargs["label"] = handlers.media_label(mode, key)
                if key == "decoder":
                    kwargs["interactive"] = decoder_free
                if changed or key in ("noise_scale", "auto_duration"):
                    kwargs["value"] = after[key]
                updates.append(gr.update(**kwargs))
            example_updates = _example_updates(show, mode_id, len(example_buttons), example_note)
            monitor_updates, monitor_next = _monitor_for_mode(show, mode, monitor)
            if failure and failure.get("mode") != mode_id:
                status_update = render.last_run_line(
                    mode=modes.get_mode(failure["mode"]),
                    at=failure["at"],
                    headline=failure["headline"],
                )
                failure = None
            else:
                status_update = gr.update()
            return [
                mode_id,
                parked_next,
                view.purpose,
                *updates,
                *[
                    gr.update(value=view.captions.get(k, ""), visible=view.active[k])
                    for k in _MEDIA
                ],
                gr.update(
                    visible=view.varies_sound,
                    value=_noise_choice(after["noise_scale"]) if view.varies_sound else noise,
                ),
                gr.update(
                    visible=view.varies_clip,
                    value=_clip_choice(after["noise_scale"]) if view.varies_clip else clip,
                ),
                gr.update(visible=view.sizes_picture, choices=shape_choices, value=preset_value),
                gr.update(visible=bool(match["line"]) and view.sizes_picture, value=match["line"]),
                gr.update(visible=view.pinned_line is not None, value=view.pinned_line or ""),
                gr.update(visible=view.upscale, value=upscale),
                gr.update(visible=view.timed_input and not view.length_from_clip, value=source),
                gr.update(visible=seconds_visible, value=handlers.seconds_for_frames(frames_value)),
                gr.update(visible=view.timed_input, value=length["line"]),
                gr.update(value=frames_value, visible=lengths["num_frames"]),
                gr.update(visible=view.sizes_picture),
                gr.update(visible=not view.active["prompt"]),
                gr.update(visible=view.active["decoder"], value=decoder_text),
                frames_user,
                *example_updates,
                *monitor_updates,
                monitor_next,
                gr.update(visible=monitor_next.get("kind") == "sample"),
                status_update,
                failure,
            ]

        def _switch(mode_id: str, *state: Any) -> list[Any]:
            return _mode_change(mode_id, *state)

        mode_selector.change(
            fn=_switch, inputs=[mode_selector, *switch_state], outputs=switch_outputs, **_PRIVATE
        )

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

        def _set_seconds(value: Any) -> list[Any]:
            frames = handlers.frames_for_seconds(value)
            return [frames, frames]

        def _set_frames(frames: Any) -> list[Any]:
            return [
                handlers.seconds_for_frames(frames),
                handlers._as_int(frames) or _DEFAULT_FRAMES,
            ]

        seconds.input(
            fn=_set_seconds,
            inputs=[seconds],
            outputs=[pool["num_frames"], user_frames],
            **_PRIVATE,
        )
        pool["num_frames"].input(
            fn=_set_frames,
            inputs=[pool["num_frames"]],
            outputs=[seconds, user_frames],
            **_PRIVATE,
        )

        def _on_auto(auto: Any, mode_id: str, source: str) -> list[Any]:
            view = handlers.mode_view(mode_id)
            lengths = handlers.length_fields_view(mode_id, auto)
            seconds_visible = (
                (source == render.LENGTH_SET and not view.length_from_clip)
                if view.timed_input
                else not bool(auto)
            )
            return [
                gr.update(visible=seconds_visible),
                gr.update(visible=lengths["num_frames"]),
                gr.update(visible=lengths["min_seconds"]),
                gr.update(visible=lengths["max_seconds"]),
            ]

        # The model-chosen length: Longest stops at the model's longest and Shortest stays
        # below Longest; the field not just edited moves (handlers.ordered_seconds).
        seconds_range = [pool["min_seconds"], pool["max_seconds"]]
        for edited in ("shortest", "longest"):
            field = pool["min_seconds"] if edited == "shortest" else pool["max_seconds"]
            field.input(
                fn=lambda low, high, e=edited: list(handlers.ordered_seconds(low, high, edited=e)),
                inputs=seconds_range,
                outputs=seconds_range,
                **_PRIVATE,
            )

        pool["auto_duration"].change(
            fn=_on_auto,
            inputs=[pool["auto_duration"], mode_selector, length_source],
            outputs=[seconds, pool["num_frames"], pool["min_seconds"], pool["max_seconds"]],
            **_PRIVATE,
        )

        decoder_inputs = [mode_selector, pool["enable_stage_2"], pool["decoder"]]

        def _decoder_state(mode_id: str, upscale: Any, choice: Any) -> list[Any]:
            _shown, free, text = handlers.decoder_view(mode_id, upscale, choice)
            return [gr.update(interactive=free), text]

        pool["enable_stage_2"].change(
            fn=_decoder_state,
            inputs=decoder_inputs,
            outputs=[pool["decoder"], decoder_line],
            **_PRIVATE,
        )
        pool["decoder"].change(
            fn=lambda mode_id, upscale, choice: handlers.decoder_view(mode_id, upscale, choice)[2],
            inputs=decoder_inputs,
            outputs=[decoder_line],
            **_PRIVATE,
        )
        noise_choice.input(
            fn=_noise_value, inputs=[noise_choice], outputs=[pool["noise_scale"]], **_PRIVATE
        )
        clip_choice.input(
            fn=_clip_value, inputs=[clip_choice], outputs=[pool["noise_scale"]], **_PRIVATE
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
                f,
            ]

        snap_button.click(
            fn=_snap,
            inputs=shape_inputs,
            outputs=[*shape_inputs, dims_hint, frames_hint, shape_preset, seconds, user_frames],
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
            # Only with an adapter staged, so the LoRA package is already loaded.
            lowest, highest, _default = lora_strength_range() if entries else (0, 0, 0)
            for index in range(handlers.MAX_STAGED_LORAS):
                if index < len(entries):
                    entry = entries[index]
                    out += [
                        gr.update(
                            visible=True, value=render.shown_name(str(entry.get("source", "")))
                        ),
                        gr.update(
                            visible=True,
                            minimum=lowest,
                            maximum=highest,
                            value=float(entry.get("weight", 1.0)),
                        ),
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
            sample_jump,
            last_failure,
        ]

        def _start(
            mode_id: str, enhance: Any, prompt: Any, request: gr.Request = None
        ) -> list[Any]:
            _forget(request)
            mode = modes.get_mode(mode_id)
            device, on_zerogpu = handlers.run_platform()
            starting = handlers.on_generate_start(
                mode_id, enhance, prompt, device=device, on_zerogpu=on_zerogpu
            )
            return [
                gr.update(interactive=False, value="Generating…"),
                render.readiness_reason(
                    render.CostFacts(mode_name="", state="ready"), running=True
                ),
                True,
                gr.update(visible=False),
                # Told to play before the file arrives: the player starts on loading only
                # when autoplay is already set (setting it with the file came too late, now
                # and then).
                gr.update(value=None, visible=True, autoplay=True),
                gr.update(value=None, visible=False),
                render.monitor_caption(kind="running", mode=mode),
                gr.update(value=starting, visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                {"kind": "running", "mode": mode_id},
                gr.update(visible=False),
                None,
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
            sample_jump,
            last_failure,
        ]

        def _ended(
            mode_id: str,
            recent_now: list,
            source: str,
            values: tuple,
            shown: handlers.RunOutcome | None,
            status_update: Any,
            caption: str | None = None,
            report: str | None = None,
            listed: handlers.RunOutcome | None = None,
        ) -> list:
            """Every finish output once a run has ended, with Generate back. ``shown`` is a
            run whose result goes on the monitor; without one the monitor is idle and
            holds no file, and ``report`` is the card the status holds. ``listed`` is a
            finished run that is not shown: it goes into Recent runs, where Show can serve
            its file."""
            mode = modes.get_mode(mode_id)
            review_out = _review(mode_id, False, source, *values)
            entries = list(recent_now or [])
            if listed is not None:
                entries.insert(
                    0,
                    _recent_entry(
                        mode_id, listed.result, listed.params, handlers.run_facts(listed), listed
                    ),
                )
                entries = entries[:RECENT_LIMIT]
            if shown is not None:
                result = shown.result
                params = shown.params
                facts = handlers.run_facts(shown)
                entries.insert(0, _recent_entry(mode_id, result, params, facts, shown))
                entries = entries[:RECENT_LIMIT]
                has_video = shown.video is not None
                monitor = [
                    gr.update(visible=False),
                    gr.update(value=shown.video, visible=has_video, autoplay=True),
                    gr.update(value=shown.audio if not has_video else None, visible=not has_video),
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
                failure = None
            else:
                monitor = [
                    gr.update(value=render.monitor_idle(mode), visible=True),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    caption or render.monitor_caption(kind="failed"),
                ]
                summary = gr.update(visible=False)
                seed_button = gr.update(visible=False)
                seed_value = None
                prompt_button = gr.update(visible=False)
                prompt_value = None
                monitor_next = {"kind": "failed", "mode": mode_id}
                headline = render.card_headline(report)
                failure = (
                    {"mode": mode_id, "at": time.strftime("%H:%M"), "headline": headline}
                    if headline
                    else None
                )
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
                gr.update(visible=False),
                failure,
            ]

        def _finish(
            request: gr.Request, mode_id: str, recent_now: list, source: str, *values: Any
        ) -> list:
            """After ``/on_generate`` completed: its result on the monitor, or the idle
            monitor under the report it returned."""
            outcome = _recall(request)
            # Nothing was remembered for this page: say so rather than leave "Running".
            lost = render.lost_run() if outcome is None else None
            status_update = gr.update(value=lost, visible=True) if lost else gr.update()
            shown = outcome if handlers.has_result(outcome) else None
            caption = render.monitor_caption(
                kind="refused" if handlers.ran_nothing(outcome) else "failed"
            )
            report = lost if outcome is None else outcome.status
            return _ended(
                mode_id, recent_now, source, values, shown, status_update, caption, report
            )

        def _finish_failed(
            request: gr.Request, mode_id: str, recent_now: list, source: str, *values: Any
        ) -> list:
            """After Gradio failed the run's step (or the finish step): the handler raised,
            or Gradio could not process what it returned (a result file it will not serve),
            or the page could not apply it. Generate comes back, the monitor goes idle and
            the report says what is known, with the server's message as its Detail. It
            sends no file, so it cannot fail the way the step before it did."""
            outcome, message = _recall(request), _recall_failure(request)
            try:
                report = handlers.unshown_report(outcome, message)
            except Exception as exc:  # noqa: BLE001 -- Generate comes back whatever the report
                report = render.lost_run(message or str(exc))
            if handlers.has_result(outcome):
                kind = "unshown"
            else:
                kind = "refused" if handlers.ran_nothing(outcome) else "failed"
            caption = render.monitor_caption(kind=kind)
            status_update = gr.update(value=report, visible=True)
            # A run that finished with its file kept is listed, so Show can serve it
            # without a paid rerun (the card's step points there).
            listed = outcome if kind == "unshown" and handlers.file_kept(outcome) else None
            return _ended(
                mode_id, recent_now, source, values, None, status_update, caption, report, listed
            )

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
        # Gradio 6.24's page runs a step's `.then` listeners only when the step
        # completes; when the server fails it (the function raised, or processing
        # its outputs did) or the page cannot apply its outputs, it runs the
        # `.failure` listeners alone. So the step that brings Generate back after a
        # run is wired to both, and so is the one after the finish step itself.
        finish_io = {
            "inputs": [mode_selector, recent, length_source, *field_inputs],
            "outputs": finish_outputs,
            **_PRIVATE,
        }
        finished = generation.then(fn=_finish, **finish_io)
        generation.failure(fn=_finish_failed, **finish_io)
        finished.failure(fn=_finish_failed, **finish_io)
        _keep_failures(demo, (generation, finished))

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
        # Restore runs the real mode switch against the mode actually left (so
        # what the restored mode does not use is parked, never submitted), then
        # writes the restored run's own settings, in ONE step: a chained step
        # would race the radio's own change event. Writing the radio leaves its
        # change handler nothing to do (previous == mode).
        restore_outputs = [*switch_outputs, mode_selector, pool["seed"]]

        def _restore(index: int, entries: list, *state: Any) -> list[Any]:
            entries = list(entries or [])
            if index >= len(entries):
                return [*_mode_change(state[0], *state), gr.update(), gr.update()]
            entry = entries[index]
            settings = dict(entry.get("settings") or {})
            out = _mode_change(entry["mode"], *state, restored=settings)
            seed = gr.update(value=settings["seed"]) if "seed" in settings else gr.update()
            return [*out, entry["mode"], seed]

        for index, (_text, _show, restore, reuse) in enumerate(recent_rows):
            restore.click(
                fn=lambda entries, *state, i=index: _restore(i, entries, *state),
                inputs=[recent, *switch_state],
                outputs=restore_outputs,
                js=_SCROLL_TO_PURPOSE,
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

        show_outputs = [monitor_idle, video_out, audio_out, monitor_caption, monitor_state]

        def _show(index: int, entries: list, monitor: dict) -> list[Any]:
            entries = list(entries or [])
            entry = entries[index] if index < len(entries) else None
            path = entry and (entry.get("video") or entry.get("audio"))
            if not path or not Path(path).exists():
                # The run folder has been evicted: say nothing new, show the row's state.
                return [
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    monitor,
                    gr.update(visible=False),
                    gr.update(),
                ]
            mode = modes.get_mode(entry["mode"])
            has_video = entry.get("video") is not None
            return [
                gr.update(visible=False),
                gr.update(value=entry.get("video"), visible=has_video, autoplay=False),
                gr.update(value=None if has_video else entry.get("audio"), visible=not has_video),
                render.monitor_caption(kind="earlier", mode=mode, seed=entry["seed"]),
                {"kind": "run", "mode": entry["mode"]},
                gr.update(),
                gr.update(visible=False),
            ]

        for index, row in enumerate(recent_rows):
            show_run = row[1]
            show_run.click(
                fn=lambda entries, monitor, i=index: _show(i, entries, monitor),
                inputs=[recent, monitor_state],
                outputs=[*show_outputs, show_run, sample_jump],
                js=_SCROLL_TO_MONITOR,
                **_PRIVATE,
            )

        sample_jump.click(fn=None, js=_SCROLL_TO_SAMPLE, **_PRIVATE)

        # =================================================================
        # examples: fill the form, run nothing
        # =================================================================
        if example_buttons:
            example_outputs_fill = [
                *[pool[k] for k in _EXAMPLE_FIELDS],
                shape_preset,
                seconds,
                noise_choice,
                clip_choice,
                length_source,
                user_frames,
            ]
            for index, button in enumerate(example_buttons):
                button.click(
                    fn=lambda mode_id, frames_user, i=index: _fill_example(
                        show, mode_id, i, frames_user
                    ),
                    inputs=[mode_selector, user_frames],
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

_SCROLL_TO_SAMPLE = """
() => {
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.getElementById('monitor')?.scrollIntoView({behavior: still ? 'auto' : 'smooth', block: 'center'});
}
"""

#: Generate and Show change the monitor: it comes into view at once, before the step's
#: outputs reflow the rail. On desktop that scrolls the rail as little as it takes; below
#: 1024 px, the page, to the monitor's top.
_SCROLL_TO_MONITOR = """
(...args) => {
  const wide = window.innerWidth >= 1024;
  document.getElementById('monitor')?.scrollIntoView(
    {behavior: 'instant', block: wide ? 'nearest' : 'start'});
  return args;
}
"""

#: Restore settings rewrites the matrix and the form, far above its row on a phone: the
#: purpose line, which names the restored mode, comes into view.
_SCROLL_TO_PURPOSE = """
(...args) => {
  const purpose = document.getElementById('purpose');
  if (purpose) {
    const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const wide = window.innerWidth >= 1024;
    purpose.scrollIntoView({behavior: still ? 'auto' : 'smooth', block: wide ? 'nearest' : 'center'});
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
    if view.length_from_clip:
        source = render.LENGTH_MATCH
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
    """Swap the idle monitor to the new mode's sample; never hide a finished run. A failed
    run's caption stays only in its own mode: elsewhere its card is one line naming it."""
    kind = (monitor or {}).get("kind")
    if kind in ("run", "running") or (kind == "failed" and monitor.get("mode") == mode.id):
        return [gr.update(), gr.update(), gr.update(), gr.update()], monitor
    sample = show.sample_for(mode.id)
    if sample is None:
        return (
            [
                gr.update(value=render.monitor_idle(mode), visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                render.READY,
            ],
            {"kind": "idle", "mode": mode.id},
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


def _fill_example(show: showcase.Showcase, mode_id: str, index: int, frames_user: Any) -> list:
    """Write the example's own values; leave every other control untouched."""
    examples = show.for_mode(mode_id)
    out = handlers.example_values(examples[index], mode_id) if index < len(examples) else {}
    keys = [*_EXAMPLE_FIELDS, "preset", "seconds", "noise_choice", "clip_choice", "length_source"]
    filled = [gr.update(value=out[k]) if k in out else gr.update() for k in keys]
    return [*filled, out.get("num_frames", frames_user)]


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
            kept = entry.get("video") or entry.get("audio")
            out += [
                gr.update(value=label, visible=True),
                gr.update(visible=bool(kept) and Path(kept).exists()),
                gr.update(visible=True),
                gr.update(value="Reuse seed", visible=True),
            ]
        else:
            out += [gr.update(visible=False) for _ in range(4)]
    return out
