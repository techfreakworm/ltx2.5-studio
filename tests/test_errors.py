"""Spec for ``ltx25_studio/errors.py`` — PLAN §5.2, §5.3, §4.3, §7.2.

The contract, stated once so the implementer does not have to reverse-engineer it
from assertions:

* Every error in the package descends from a single root, ``LTXStudioError``.
* Every concrete error takes **keyword arguments naming the offending thing** and
  renders a ``str()`` that contains them. "Fail loudly" is not a slogan here — an
  error whose message does not name the file, the value, the URL or the prefixes
  it actually found is a test failure.
* Four abstract bases group the hierarchy so ``ui/`` can catch by family:
  ``ConstraintError``, ``ModelFileError``, ``LoRAError``, ``GenerationError``.
* Adding a new error class without adding it to ``ERROR_SPECS`` below fails
  ``test_every_error_class_is_covered_by_this_spec``. That is deliberate: a new
  error must come with a stated, checked message contract.
"""

from __future__ import annotations

import inspect

import pytest

from ltx25_studio import errors

# --------------------------------------------------------------------------
# The contract table: class name -> (kwargs, substrings the message must contain)
# --------------------------------------------------------------------------

ERROR_SPECS = {
    # -- constraints (PLAN §5.5) -----------------------------------------
    "FrameCountError": (
        {"num_frames": 120, "temporal_compression_ratio": 8, "nearest": (113, 121)},
        ["120", "113", "121"],
    ),
    "DimensionError": (
        {"name": "width", "value": 700, "multiple": 32},
        ["width", "700", "32"],
    ),
    "AudioVideoDesyncError": (
        {"video_seconds": 5.0, "audio_seconds": 4.25},
        ["5.0", "4.25"],
    ),
    # -- model files (PLAN §5.2) -----------------------------------------
    "MissingModelFileError": (
        {
            "filename": "diffusion_pytorch_model-00001-of-00004.safetensors",
            "url": (
                "https://huggingface.co/Lightricks/LTX-2.5-Diffusers/blob/main/"
                "transformer/diffusion_pytorch_model-00001-of-00004.safetensors"
            ),
        },
        ["diffusion_pytorch_model-00001-of-00004.safetensors", "https://huggingface.co/"],
    ),
    "UnknownComponentError": (
        {"name": "prompt_enhancer", "known": ("transformer", "vae", "vocoder")},
        ["prompt_enhancer", "transformer"],
    ),
    # -- LoRA (PLAN §4.3) ------------------------------------------------
    "LoRATooLargeError": (
        {"source": "/tmp/huge.safetensors", "size_bytes": 12_000_000_000, "max_bytes": 10_000_000_000},
        ["huge.safetensors", "12", "10"],
    ),
    "LoRAFormatError": (
        {"source": "/tmp/notes.txt", "reason": "not a safetensors container"},
        ["notes.txt", "safetensors"],
    ),
    "LoRAHeaderError": (
        {"source": "/tmp/adapter.safetensors", "declared_header_bytes": 1152921504606846976},
        ["adapter.safetensors", "1152921504606846976"],
    ),
    "LoRATruncatedError": (
        {"source": "/tmp/adapter.safetensors", "declared_bytes": 4096, "available_bytes": 512},
        ["adapter.safetensors", "4096", "512"],
    ),
    "LoRAVersionError": (
        {"source": "/tmp/adapter.safetensors", "found": "2.3.0", "expected": "2.5"},
        ["adapter.safetensors", "2.3.0", "2.5"],
    ),
    "LoRAConventionError": (
        {
            "source": "/tmp/adapter.safetensors",
            "found_prefixes": ("unet.", "controlnet."),
            "known_prefixes": ("diffusion_model.", "transformer."),
        },
        ["adapter.safetensors", "unet.", "controlnet.", "diffusion_model."],
    ),
    "LoRAPairingError": (
        {
            "source": "/tmp/adapter.safetensors",
            "unpaired_keys": ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",),
        },
        ["adapter.safetensors", "transformer_blocks.0.attn1.to_q", "lora_A"],
    ),
    "LoRANotAttachedError": (
        {
            "source": "/tmp/adapter.safetensors",
            "adapter_name": "user_lora",
            "components": ("transformer", "connectors"),
        },
        ["adapter.safetensors", "user_lora", "transformer", "connectors"],
    ),
    # -- generation (PLAN §5.3) ------------------------------------------
    "ModelLoadError": (
        {"component": "transformer", "reason": "connection reset by peer"},
        ["transformer", "connection reset by peer"],
    ),
    "DecodeError": (
        {"mode": "text_to_video", "reason": "pipeline returned no frames"},
        ["text_to_video", "no frames"],
    ),
    "EncodeError": (
        {"path": "/tmp/run-abc/video.mp4", "reason": "ffmpeg exited 1"},
        ["video.mp4", "ffmpeg"],
    ),
    "EmptyOutputError": (
        {"mode": "text_to_video", "expected_path": "/tmp/run-abc/video.mp4"},
        ["text_to_video", "video.mp4"],
    ),
    # -- modes (PLAN §3.2) -----------------------------------------------
    "UnknownModeError": (
        {"mode": "text_to_hologram", "known": ("text_to_video", "image_to_video")},
        ["text_to_hologram", "text_to_video"],
    ),
}

#: Grouping bases. They exist so callers can catch a family; they are not
#: required to accept the keyword contract above.
BASE_ERROR_NAMES = (
    "LTXStudioError",
    "ConstraintError",
    "ModelFileError",
    "LoRAError",
    "GenerationError",
)

BASE_OF = {
    "FrameCountError": "ConstraintError",
    "DimensionError": "ConstraintError",
    "AudioVideoDesyncError": "ConstraintError",
    "MissingModelFileError": "ModelFileError",
    "UnknownComponentError": "ModelFileError",
    "LoRATooLargeError": "LoRAError",
    "LoRAFormatError": "LoRAError",
    "LoRAHeaderError": "LoRAError",
    "LoRATruncatedError": "LoRAError",
    "LoRAVersionError": "LoRAError",
    "LoRAConventionError": "LoRAError",
    "LoRAPairingError": "LoRAError",
    "LoRANotAttachedError": "LoRAError",
    "ModelLoadError": "GenerationError",
    "DecodeError": "GenerationError",
    "EncodeError": "GenerationError",
    "EmptyOutputError": "GenerationError",
}


def _get(name):
    cls = getattr(errors, name, None)
    assert cls is not None, f"ltx25_studio.errors.{name} does not exist (PLAN §5.2/§4.3/§5.3)"
    return cls


# --------------------------------------------------------------------------
# hierarchy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", BASE_ERROR_NAMES)
def test_base_error_class_exists(name):
    assert isinstance(_get(name), type)


def test_root_error_is_an_exception_not_a_baseexception():
    root = _get("LTXStudioError")
    assert issubclass(root, Exception)


@pytest.mark.parametrize("name", sorted(BASE_ERROR_NAMES[1:]))
def test_grouping_bases_descend_from_the_root(name):
    assert issubclass(_get(name), _get("LTXStudioError"))


@pytest.mark.parametrize("name", sorted(ERROR_SPECS))
def test_error_class_exists(name):
    assert isinstance(_get(name), type)


@pytest.mark.parametrize("name", sorted(ERROR_SPECS))
def test_every_error_descends_from_the_single_root(name):
    assert issubclass(_get(name), _get("LTXStudioError"))


@pytest.mark.parametrize("name,base", sorted(BASE_OF.items()))
def test_error_is_in_the_right_family(name, base):
    assert issubclass(_get(name), _get(base))


# --------------------------------------------------------------------------
# actionable messages
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ERROR_SPECS))
def test_error_constructs_from_its_documented_keyword_arguments(name):
    kwargs, _ = ERROR_SPECS[name]
    exc = _get(name)(**kwargs)
    assert isinstance(exc, Exception)


@pytest.mark.parametrize("name", sorted(ERROR_SPECS))
def test_error_message_is_not_empty(name):
    kwargs, _ = ERROR_SPECS[name]
    message = str(_get(name)(**kwargs))
    assert message.strip(), f"{name} renders an empty message — PLAN §5.3 forbids silent failure"


@pytest.mark.parametrize("name", sorted(ERROR_SPECS))
def test_error_message_is_not_merely_the_class_name(name):
    kwargs, _ = ERROR_SPECS[name]
    message = str(_get(name)(**kwargs)).strip()
    assert message != name
    assert len(message) > 20, f"{name}: {message!r} is too terse to be actionable"


@pytest.mark.parametrize("name", sorted(ERROR_SPECS))
def test_error_message_names_the_offending_detail(name):
    kwargs, expected = ERROR_SPECS[name]
    message = str(_get(name)(**kwargs))
    missing = [needle for needle in expected if needle not in message]
    assert not missing, f"{name} message omits {missing}; got: {message!r}"


@pytest.mark.parametrize("name", sorted(ERROR_SPECS))
def test_error_message_reads_as_prose_not_a_repr_dump(name):
    """An actionable message is a sentence, not ``{'source': ..., 'reason': ...}``."""
    kwargs, _ = ERROR_SPECS[name]
    message = str(_get(name)(**kwargs))
    assert not message.startswith("{"), f"{name} renders a dict repr: {message!r}"
    assert not message.startswith("("), f"{name} renders a tuple repr: {message!r}"
    assert " " in message.strip(), f"{name} renders a single token: {message!r}"


@pytest.mark.parametrize("name", sorted(ERROR_SPECS))
def test_error_can_be_raised_and_caught_as_the_root(name):
    kwargs, _ = ERROR_SPECS[name]
    cls = _get(name)
    with pytest.raises(_get("LTXStudioError")) as excinfo:
        raise cls(**kwargs)
    assert isinstance(excinfo.value, cls)


# --------------------------------------------------------------------------
# no error may escape the spec
# --------------------------------------------------------------------------


def test_every_error_class_is_covered_by_this_spec():
    root = _get("LTXStudioError")
    defined = {
        name
        for name, obj in inspect.getmembers(errors, inspect.isclass)
        if getattr(obj, "__module__", None) == errors.__name__
        and issubclass(obj, root)
        and not name.startswith("_")
    }
    uncovered = defined - set(ERROR_SPECS) - set(BASE_ERROR_NAMES)
    assert not uncovered, (
        f"errors.py defines {sorted(uncovered)} with no message contract in ERROR_SPECS. "
        "Every error owes the user an actionable message (PLAN §5.3) — add it to the spec."
    )
