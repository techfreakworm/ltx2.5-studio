"""The INBOUND fork boundary: everything the GPU dispatch pickles must pickle.

The pair to ``test_errors_fork_boundary.py``, and the direction that file
could not cover: on ZeroGPU, ``@spaces.GPU`` pickles the decorated function's
ARGUMENTS onto the worker's arg queue. There is no ``gr.Error`` escape hatch
on this side — the arguments must simply pickle, or the run dies at dispatch,
before any GPU work is done.

That is not hypothetical either. The first-ever staged-LoRA generation on the
Space died in 2.05 s with ``'PicklingError: cannot pickle mappingproxy
object'``: validation PASSED, and the accepted report — riding into the
worker inside ``GenerationRequest.loras`` — carried ``LoRAInspection``'s two
``MappingProxyType`` fields, which refuse to pickle by design. The 11-mode
matrix never saw it because no run of the matrix ever staged an adapter.

So this file pins, offline, that every argument shape the live dispatch
sends actually crosses:

* the staged-adapter dict from a REAL validation (fixture header), with the
  proxy invariant restored on the far side rather than traded away;
* a per-field guard walked programmatically off the report dataclasses, so a
  future field that cannot pickle fails HERE, naming itself, instead of on
  the metered side;
* the full dispatch payload as ``generate()`` actually shapes it — a
  ``GenerationRequest`` carrying every kind of UI input at once — and the
  ``GenerationResult`` that crosses back.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from types import MappingProxyType

import pytest
from conftest import (
    NATIVE_PREFIX,
    OFFICIAL_METADATA,
    TRANSFORMER_MODULE_PATHS,
    build_safetensors_bytes,
    lora_tensors,
)

from ltx25_studio import generate as generate_module
from ltx25_studio.lora import validate_lora


def _real_report(tmp_path):
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(
        build_safetensors_bytes(
            lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS), OFFICIAL_METADATA
        )
    )
    return str(adapter), validate_lora(str(adapter))


def _staged(tmp_path):
    source, report = _real_report(tmp_path)
    # The exact dict handlers.on_validate_lora stages.
    return {"source": source, "path": str(report.path), "report": report}


# --------------------------------------------------------------------------
# the staged adapter — the argument that actually died on the Space
# --------------------------------------------------------------------------


def test_a_staged_adapter_from_a_real_validation_round_trips(tmp_path):
    staged = _staged(tmp_path)
    back = pickle.loads(pickle.dumps(staged))
    assert back["source"] == staged["source"]
    assert back["path"] == staged["path"]
    assert back["report"] == staged["report"], (
        "the frozen dataclasses define equality; a round trip that changes the "
        "report is a different report"
    )


def test_the_proxy_invariant_is_restored_not_traded_away(tmp_path):
    """The mappingproxies exist so an accepted report cannot be edited after
    the fact. The fix must not quietly downgrade them to plain dicts on the
    worker side — the read-only invariant holds identically in both
    processes."""
    staged = _staged(tmp_path)
    inspection = pickle.loads(pickle.dumps(staged))["report"].inspection
    assert type(inspection.metadata) is MappingProxyType
    assert type(inspection.shapes) is MappingProxyType
    assert inspection.version_signals == staged["report"].inspection.version_signals


def test_every_report_field_pickles_as_transported_naming_itself_when_not(tmp_path):
    """The guard for the NEXT field. Pickle transports ``__getstate__()``, not
    raw attributes — that indirection is the fix itself, since the two proxy
    fields cross as their unwrapped dicts — so the guard walks the state each
    object actually ships. A field added tomorrow lands in that state the
    moment it exists and is covered without being hand-listed here, and a
    failure names the field instead of drowning it in a whole-object
    traceback."""
    _, report = _real_report(tmp_path)
    offenders = []
    for owner in (report, report.inspection):
        state = owner.__getstate__()
        declared = {field.name for field in dataclasses.fields(owner)}
        assert declared <= set(state), (
            f"{type(owner).__name__} ships a state missing declared fields "
            f"{sorted(declared - set(state))} — the guard would go blind to them"
        )
        for name, value in state.items():
            try:
                pickle.dumps(value)
            except Exception as exc:  # noqa: BLE001 -- collecting, not judging
                offenders.append(f"{type(owner).__name__}.{name}: {exc}")
    assert not offenders, (
        f"these report fields cannot cross the arg queue — the run would die at "
        f"dispatch, before any GPU work: {offenders}"
    )


# --------------------------------------------------------------------------
# the whole dispatch payload, shaped as the live path ships it
# --------------------------------------------------------------------------


def test_the_full_dispatch_payload_pickles_with_every_input_kind_at_once(tmp_path):
    """``generate()`` routes to the entrypoint as ``dispatch(request=...,
    pipeline_factory=None, encoder=None, output_root=None,
    retention_bytes=None)`` and spaces pickles exactly those kwargs. One
    request carrying every kind of UI input at once — media paths, a PIL
    image, raw condition rows, a staged adapter — is the sweep that beats
    three more ~4 GPU-second discoveries."""
    from PIL import Image

    request = generate_module.GenerationRequest(
        mode="image_text_to_audio_video",
        prompt="a lighthouse in heavy fog",
        negative_prompt="blurry",
        width=704,
        height=480,
        num_frames=121,
        seed=7,
        image=Image.new("RGB", (8, 8)),
        video=str(tmp_path / "conditioning.mp4"),
        audio=str(tmp_path / "conditioning.wav"),
        conditions=(["a wide establishing shot", 8, 0.8],),
        loras=(_staged(tmp_path),),
        enable_stage_2=True,
        extras={"note": "programmatic"},
    )
    payload = {
        "request": request,
        "pipeline_factory": None,
        "encoder": None,
        # Joined the payload when prompt enhancement moved into the worker: the
        # seam has to reach the side that now uses it. None on the live path,
        # which is what keeps this dict picklable at all -- a real callable here
        # would be a local function and would die at dispatch.
        "enhancer": None,
        "output_root": None,
        "retention_bytes": None,
    }
    back = pickle.loads(pickle.dumps(payload))
    assert back["request"].loras[0]["report"] == request.loras[0]["report"]
    assert back["request"].prompt == request.prompt


def test_the_dispatch_payload_matches_what_generate_actually_sends(tmp_path, monkeypatch):
    """The list above is a claim about the live call. Check it against the call.

    A hand-written payload drifts silently: ``enhancer`` was added to the real
    dispatch and this file would have gone on pinning the old five-key shape,
    proving that a dict nobody sends still pickles. So the keys are read off an
    actual ``generate()`` dispatch rather than retyped.
    """
    captured: list = []

    def dispatch(**kwargs):
        captured.append(kwargs)
        token = generate_module._IN_GPU_WORKER.set(True)
        try:
            return generate_module.generate(**kwargs)
        finally:
            generate_module._IN_GPU_WORKER.reset(token)

    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", dispatch)
    from test_generate import FakePipe, factory_for, good_encoder

    generate_module.generate(
        generate_module.GenerationRequest(mode="text_to_video", prompt="a lighthouse", seed=7),
        pipeline_factory=factory_for(FakePipe()),
        encoder=good_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    assert captured, "generate() never dispatched"
    sent = set(captured[0])
    assert sent == {
        "request",
        "pipeline_factory",
        "encoder",
        "enhancer",
        "output_root",
        "retention_bytes",
        "progress",
    }, (
        f"the live dispatch payload is {sorted(sent)}; the pickle test above covers a "
        f"different shape and is no longer testing the real call"
    )


def test_the_progress_handle_pickles_into_the_worker():
    """The one argument that can kill a run before any GPU work is done.

    ``arg_queue.put`` raises ``PicklingError`` on an argument that will not
    cross, and the dispatch dies at that point — 2.05 s in, quota spent on
    nothing, which is exactly how the staged-LoRA ``mappingproxy`` defect
    presented. ``gr.Progress`` now rides in the payload, so it is checked here
    rather than discovered on the Space.

    It is a small object by construction and this pins that it stays one:
    ``__call__`` builds its ``TrackedIterable`` inline and never appends, so with
    ``track_tqdm`` off ``iterables`` is empty and the whole payload is a bool and
    a list. ``spaces`` also carries a ``Progress`` inside ``GradioPartialContext``
    on the same ``put``, so this is the class it already transports, not a new
    one.
    """
    gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

    handle = gr.Progress()
    back = pickle.loads(pickle.dumps(handle))

    assert isinstance(back, gr.Progress)
    assert back.track_tqdm is handle.track_tqdm
    assert list(back.iterables) == []


def test_the_enhanced_prompt_crosses_back_inside_the_run_record(tmp_path):
    """The return leg for a prompt that is now built on the far side.

    Enhancement runs INSIDE the worker, so the only way the user ever reads which
    prompt was actually sent is if it survives the res queue. It travels as the
    run record's ``prompt``, which the UI shows (UI review 2, B-1: it is no
    longer repeated as a note).

    The record is builtins only, so this is cheap -- but the LoRA report was
    "obviously fine" too, right up until ``MappingProxyType`` killed the first
    staged generation on the Space at 2.05 s. The cost of checking is one assert.
    """
    from conftest import inputs_for
    from test_exports_contract import FakePipe, factory_returning, one_muxed_file_encoder

    rewritten = "A weathered stone lighthouse stands on a basalt outcrop as dense fog rolls in."
    result = generate_module.generate(
        mode="text_to_video",
        seed=7,
        enhance_prompt=True,
        **inputs_for("text_to_video"),
        pipeline_factory=factory_returning(FakePipe()),
        encoder=one_muxed_file_encoder,
        enhancer=lambda prompt, image=None: rewritten,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    back = pickle.loads(pickle.dumps(result))
    assert back == result
    assert back.record.prompt == rewritten, (
        "the record crossed the fork without the rewritten prompt in it, so the user is never "
        "told which prompt was sent"
    )


def test_the_result_crossing_back_pickles(tmp_path):
    """The OK direction of the return leg: a GenerationResult is what the
    worker puts on the res queue for a successful run."""
    result = generate_module.GenerationResult(
        output_path=tmp_path / "run-abc" / "video.mp4",
        run_id="abc123",
        run_dir=tmp_path / "run-abc",
        mode="text_to_video",
        seed=7,
        video_path=tmp_path / "run-abc" / "video.mp4",
        audio_path=None,
        warnings=("your audio was trimmed from 6.00 s to 4.84 s to match the video.",),
    )
    back = pickle.loads(pickle.dumps(result))
    assert back == result


def test_the_peak_vram_notes_cross_back_inside_the_result():
    """The return leg for numbers that only exist on the far side.

    The probe is opened inside the ``@spaces.GPU`` worker -- it has to be, since
    the web process has no CUDA to measure. So the only way a headroom figure
    ever reaches a human is if it survives the res queue in
    ``GenerationResult.warnings``.

    That is why they are plain ``str`` and not a dataclass, a namedtuple or a
    dict of ints. This file exists because "obviously fine" was wrong once
    already: ``LoRAInspection``'s ``MappingProxyType`` fields killed the first
    staged generation on the Space at 2.05 s. A measurement that dies at the
    fork measures nothing, and it would die on the *flagship* run -- the one
    expensive enough to be worth instrumenting.
    """
    notes = (
        "peak VRAM, prompt enhancement: 57.000 GiB allocated / 59.000 GiB reserved.",
        "peak VRAM, generation (model load through encode): 71.011 GiB allocated / "
        "73.484 GiB reserved.",
        "peak VRAM, whole call: 71.011 GiB allocated / 73.484 GiB reserved of 79.139 GiB "
        "total, with 54.912 GiB already resident on entry. Reserved is what bounds the "
        "next allocation; allocated is what the run itself needed.",
    )
    result = generate_module.GenerationResult(
        output_path=Path("/tmp/run-abc/video.mp4"),
        run_id="abc123",
        run_dir=Path("/tmp/run-abc"),
        mode="text_to_video",
        seed=7,
        warnings=notes,
    )
    back = pickle.loads(pickle.dumps(result))
    assert back == result
    assert back.warnings == notes, (
        "the peak-VRAM notes did not survive the fork, so the one run big enough to "
        "give this build a headroom number reports it to nobody"
    )


def test_the_probe_the_worker_builds_is_never_itself_put_on_the_res_queue():
    """Only its output crosses. The probe holds a live ``torch.cuda`` module
    reference, which is exactly the kind of object that does not pickle -- and
    the fix is that it never leaves the worker, not that it is made to."""
    assert "probe" not in {f.name for f in dataclasses.fields(generate_module.GenerationResult)}, (
        "GenerationResult grew a field holding the probe; it carries a module reference "
        "and would die at the fork on every run"
    )
