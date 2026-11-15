"""Which model module each adapter key will land on, resolved the way diffusers does it.

``LTX2LoraLoaderMixin.lora_state_dict`` (diffusers @ac56fa2f) decides per file:

* any key starting ``diffusion_model.`` -> the whole state dict goes through
  ``_convert_non_diffusers_ltx2_lora_to_diffusers``, which keeps ONLY those
  keys, renames native modules (``adaln_single.`` -> ``time_embed.``,
  ``patchify_proj`` -> ``proj_in`` …) and re-prefixes them ``transformer.``;
* any key starting ``text_embedding_projection.`` -> converted the same way
  into ``connectors.`` and added;
* otherwise the keys are used as they are.

peft then strips ``transformer.`` / ``connectors.`` and targets the module
named by what precedes ``.lora_A`` / ``.lora_B``. This module reproduces that
using diffusers' own converter when it can be imported, and an identical
table-driven rename when it cannot (the torch-less environment). A FULL-env
test keeps the two in agreement.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = ["CONNECTOR_RENAMES", "NATIVE_RENAMES", "KeyResolution", "resolve_keys", "split_suffix"]

NATIVE_PREFIX = "diffusion_model"
NATIVE_CONNECTOR_PREFIX = "text_embedding_projection"

#: diffusers' ``rename_dict`` for native transformer keys, in its order. Each
#: pair is applied with ``str.replace`` to the whole key, exactly as upstream.
NATIVE_RENAMES = (
    ("patchify_proj", "proj_in"),
    ("audio_patchify_proj", "audio_proj_in"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
)

#: ... and for native connector keys.
CONNECTOR_RENAMES = (("aggregate_embed", "text_proj_in"),)

#: Applied after the renames, to keys that START with the old name only.
_LEADING_RENAMES = (("adaln_single.", "time_embed."), ("audio_adaln_single.", "audio_time_embed."))

#: ``(down, up)`` factor tokens; side 0 is ``[rank, in]``, side 1 is ``[out, rank]``.
_SUFFIX_PAIRS = (("lora_A", "lora_B"), ("lora_down", "lora_up"))


@dataclass(frozen=True)
class KeyResolution:
    """Adapter keys grouped by the module they will be applied to.

    ``transformer`` / ``connectors`` map a module name to ``{side: original key}``.
    ``dropped`` lists keys diffusers will ignore (a ``transformer.`` key in a
    file that also has native keys).
    """

    transformer: dict = field(default_factory=dict)
    connectors: dict = field(default_factory=dict)
    dropped: tuple[str, ...] = ()
    converter: str = "table"


def split_suffix(key: str) -> tuple[str, int] | None:
    """``(module, side)`` for a low-rank factor key, or ``None``."""
    for pair in _SUFFIX_PAIRS:
        for side, token in enumerate(pair):
            marker = f".{token}."
            if marker in key:
                return key.split(marker)[0], side
            if key.endswith(f".{token}"):
                return key[: -len(token) - 1], side
    return None


def _table_convert(mapping: dict, non_diffusers_prefix: str) -> dict:
    """A line-for-line transcription of diffusers' converter, on ``{key: value}``."""
    kept = {
        key.removeprefix(f"{non_diffusers_prefix}."): value
        for key, value in mapping.items()
        if key.startswith(f"{non_diffusers_prefix}.")
    }
    renames = NATIVE_RENAMES if non_diffusers_prefix == NATIVE_PREFIX else CONNECTOR_RENAMES
    renamed = {}
    for key, value in kept.items():
        new_key = key
        for old, new in renames:
            new_key = new_key.replace(old, new)
        renamed[new_key] = value
    final = {}
    for key, value in renamed.items():
        for old, new in _LEADING_RENAMES:
            if key.startswith(old):
                key = key.replace(old, new)
                break
        final[key] = value
    prefix = "transformer" if non_diffusers_prefix == NATIVE_PREFIX else "connectors"
    return {f"{prefix}.{key}": value for key, value in final.items()}


def _diffusers_converter():
    try:
        from diffusers.loaders.lora_conversion_utils import (
            _convert_non_diffusers_ltx2_lora_to_diffusers,
        )
    except Exception:  # noqa: BLE001 -- no diffusers (torch-less env) means the table
        return None
    return _convert_non_diffusers_ltx2_lora_to_diffusers


def resolve_keys(keys: Iterable[str], *, converter: str | None = None) -> KeyResolution:
    """Group ``keys`` by the module diffusers + peft will apply them to.

    ``converter``: ``"diffusers"``, ``"table"`` or ``None`` (diffusers when it
    can be imported, else the table).
    """
    keys = tuple(keys)
    convert = None
    if converter in (None, "diffusers"):
        convert = _diffusers_converter()
        if convert is None and converter == "diffusers":
            raise ImportError("diffusers' LTX2 LoRA key converter is not importable here")
    used = "diffusers" if convert is not None else "table"
    convert = convert or _table_convert

    original = {key: key for key in keys if "dora_scale" not in key}
    converted = original
    if any(key.startswith(f"{NATIVE_PREFIX}.") for key in original):
        converted = dict(convert(dict(original), NATIVE_PREFIX))
    if any(key.startswith(f"{NATIVE_CONNECTOR_PREFIX}.") for key in original):
        converted = dict(converted)
        converted.update(convert(dict(original), NATIVE_CONNECTOR_PREFIX))

    transformer: dict[str, dict[int, str]] = {}
    connectors: dict[str, dict[int, str]] = {}
    reached: set[str] = set()
    for new_key, source_key in converted.items():
        for prefix, target in (("transformer.", transformer), ("connectors.", connectors)):
            if not new_key.startswith(prefix):
                continue
            split = split_suffix(new_key[len(prefix) :])
            if split is not None:
                module, side = split
                sides = target.setdefault(module, {})
                # A `lora_B.bias` must not displace the 2-D `lora_B.weight`.
                if side not in sides or source_key.endswith(".weight"):
                    sides[side] = source_key
                reached.add(source_key)
            break

    dropped = tuple(key for key in original if key not in reached)
    return KeyResolution(
        transformer=transformer, connectors=connectors, dropped=dropped, converter=used
    )
