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
from types import MappingProxyType

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
        "output_root": None,
        "retention_bytes": None,
    }
    back = pickle.loads(pickle.dumps(payload))
    assert back["request"].loras[0]["report"] == request.loras[0]["report"]
    assert back["request"].prompt == request.prompt


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
        warnings=("audio input trimmed from 6.00s to 4.84s to match 121 frames",),
    )
    back = pickle.loads(pickle.dumps(result))
    assert back == result
