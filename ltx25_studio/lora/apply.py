"""Stage 7: load the adapter, then prove it actually attached.

PLAN §4.3 stage 7, §9.3.1.

Stages 1-6.5 catch broken files. **This catches a valid file that is simply for
a different model** -- the case where diffusers loads it happily, attaches it to
nothing, and the user gets base-model output for the rest of the session with no
error anywhere.

Upstream's entire validation is a substring test::

    "Invalid LoRA checkpoint. Make sure all LoRA param names contain 'lora' substring."

An adapter for a completely different architecture passes that -- its keys do
contain ``lora`` -- and then attaches to zero modules. Generation proceeds.

**Counting modules is the only honest check.** peft records the adapter in
``peft_config`` and ``get_list_adapters()`` reports it *whether or not anything
attached*; both say "loaded". A module exposes a ``lora_A`` mapping containing
the adapter name only when a layer was really wrapped, so that is what we count.

Stage 7 is necessary but **not sufficient**: an adapter with the right module
names and the wrong dimensions gets as far as the load, where real peft raises
torch's size-mismatch one line per tensor. Stage 6.5 in ``validate.py`` refuses
it earlier; here the load failure is at least named for what it is.

Load failures are mapped to the error that describes them, with the original
exception kept as ``__cause__``: a size mismatch is a dimension error (first
mismatch only), peft finding nothing to target is the not-attached error, and
a name already in use is a duplicate-adapter error.

This is the only module in ``lora/`` that touches a real model. Even so it
imports nothing heavy -- everything here is duck-typed against the pipeline, so
the suite runs with no diffusers, no peft and no GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import LoRADimensionError, LoRAError, LoRAFormatError, LoRANotAttachedError
from .fetch import mark_used
from .staging import DEFAULT_WEIGHT, check_weight

__all__ = [
    "DEFAULT_ADAPTER_NAME",
    "LoRAAttachReport",
    "LoRADuplicateAdapterError",
    "apply_lora",
    "unfuse_lora",
]

DEFAULT_ADAPTER_NAME = "default"

#: Fallback when a pipeline does not advertise its own. Upstream's
#: ``LTX2LoraLoaderMixin`` sets exactly these two.
FALLBACK_LOADABLE_MODULES = ("transformer", "connectors")

#: torch's ``load_state_dict`` wording, as peft surfaces it through diffusers.
_SIZE_MISMATCH = re.compile(
    r"size mismatch for (?P<key>\S+?): copying a param with shape "
    r"torch\.Size\(\[(?P<given>[\d, ]*)\]\) from checkpoint, the shape in current model is "
    r"torch\.Size\(\[(?P<wanted>[\d, ]*)\]\)"
)
#: peft >= 0.20 and older peft, when no adapter key names a module of the model.
_NOTHING_TARGETED = ("No modules were targeted for adaptation", "not found in the base model")
_NAME_IN_USE = "already in use"
_MAX_REASON = 300


class LoRADuplicateAdapterError(LoRAError):
    """An adapter of this name is already loaded on the pipeline."""

    def __init__(self, *, source, adapter_name):
        self.source = source
        self.adapter_name = adapter_name
        super().__init__(
            f"LoRA {source} could not be attached as {adapter_name!r}: an adapter with that "
            f"name is already loaded on this worker, left over from an earlier run. Run the "
            f"generation again; if it repeats, restart the Space."
        )


@dataclass(frozen=True)
class LoRAAttachReport:
    """Proof of what actually happened, not what was requested."""

    source: Path
    adapter_name: str
    attached_modules: tuple[str, ...]
    per_component: dict[str, int]
    components: tuple[str, ...]
    fused: bool = False
    weight: float = DEFAULT_WEIGHT
    #: How the adapter is named in the run report; the file name by default.
    label: str = ""

    @property
    def attached(self) -> int:
        return len(self.attached_modules)

    @property
    def summary(self) -> str:
        """One run-report line: ``name attached to N modules at weight W``."""
        weight = f"{self.weight:g}"
        if "." not in weight and "e" not in weight:
            weight += ".0"
        name = self.label or self.source.stem
        return f"{name} attached to {self.attached} modules at weight {weight}"


def _loadable_components(pipeline) -> tuple[str, ...]:
    declared = getattr(pipeline, "_lora_loadable_modules", None) or FALLBACK_LOADABLE_MODULES
    return tuple(name for name in declared if getattr(pipeline, name, None) is not None)


def _attached_modules(component, adapter_name: str) -> list[str]:
    """Modules that really got a LoRA layer for this adapter.

    Not ``peft_config`` -- see the module docstring. ``lora_A`` mirrors peft's
    ``LoraLayer.lora_A: nn.ModuleDict``, which only exists once a layer was
    genuinely wrapped.
    """
    found = []
    for name, module in component.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None and adapter_name in lora_a:
            found.append(name)
    return found


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    line = text[0] if text else type(exc).__name__
    return line if len(line) <= _MAX_REASON else line[: _MAX_REASON - 1] + "…"


def _dimension_error(source: str, adapter_name: str, message: str):
    """The first size mismatch in torch's message, as a dimension error, or None."""
    match = _SIZE_MISMATCH.search(message)
    if match is None:
        return None
    key = match["key"]
    given = [int(x) for x in match["given"].replace(" ", "").split(",") if x]
    wanted = [int(x) for x in match["wanted"].replace(" ", "").split(",") if x]
    module = re.split(r"\.lora_(?:A|B|down|up|magnitude_vector)\b", key)[0]
    axis = next((i for i, (g, w) in enumerate(zip(given, wanted, strict=False)) if g != w), None)
    if axis is None:
        return None
    return LoRADimensionError(
        source=source, module=module, expected=wanted[axis], actual=given[axis], key=key
    )


def _mapped_load_error(path: Path, adapter_name: str, components, exc: Exception) -> LoRAError:
    """The error that says what a load failure actually was (PLAN §7.2)."""
    message = str(exc)
    if "size mismatch for" in message:
        mapped = _dimension_error(str(path), adapter_name, message)
        if mapped is not None:
            return mapped
    if any(marker in message for marker in _NOTHING_TARGETED):
        return LoRANotAttachedError(
            source=str(path), adapter_name=adapter_name, components=components
        )
    if _NAME_IN_USE in message and adapter_name in message:
        return LoRADuplicateAdapterError(source=str(path), adapter_name=adapter_name)
    if isinstance(exc, ValueError):
        # Upstream's substring rejection arrives as a bare ValueError.
        return LoRAFormatError(source=str(path), reason=_first_line(exc))
    return LoRAFormatError(
        source=str(path), reason=f"the adapter could not be loaded ({_first_line(exc)})"
    )


def apply_lora(
    pipeline,
    source,
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    weight: float = DEFAULT_WEIGHT,
    fuse: bool = False,
    label: str = "",
) -> LoRAAttachReport:
    """Load an adapter and assert it attached to more than zero modules.

    ``weight`` must be in (0, 2] and is checked before anything loads. ``label``
    names the adapter in :attr:`LoRAAttachReport.summary` (default: file name).

    Note there is deliberately **no flag that skips the assertion**. An escape
    hatch is how this check quietly stops running six months from now, and it
    must run on every load, built-in adapters included.
    """
    path = Path(source)
    weight = check_weight(weight, source=label or str(path))
    components = _loadable_components(pipeline)

    # So the LoRA cache cannot evict the file between here and the open.
    mark_used(path)
    try:
        pipeline.load_lora_weights(path, adapter_name=adapter_name)
    except LoRAError:
        raise
    except Exception as exc:
        raise _mapped_load_error(path, adapter_name, components, exc) from exc

    per_component: dict[str, int] = {}
    attached_modules: list[str] = []
    for name in components:
        found = _attached_modules(getattr(pipeline, name), adapter_name)
        per_component[name] = len(found)
        attached_modules.extend(f"{name}.{module}" for module in found)

    # The anti-silent-garbage check. Connector-only adapters are legitimate, so
    # this counts across every loadable component -- counting the transformer
    # alone would reject a perfectly good text-conditioning LoRA.
    if not attached_modules:
        raise LoRANotAttachedError(
            source=str(path), adapter_name=adapter_name, components=components
        )

    # Correct for THIS function's single-adapter contract only: diffusers'
    # `set_adapters` REPLACES the active set, so a caller attaching several
    # adapters through repeated apply_lora calls ends with only the last one
    # active while every report claims success. The collective activation for
    # multi-adapter staging lives in `generate._apply_loras`, the only place
    # the full name/weight lists exist.
    set_adapters = getattr(pipeline, "set_adapters", None)
    if callable(set_adapters):
        set_adapters(adapter_name, weight)

    if fuse:
        # fuse_lora defaults to components=["transformer"] only, which leaves
        # connector layers unfused -- the adapter half-applied, and nothing says
        # so. Always name both.
        pipeline.fuse_lora(components=list(components))

    return LoRAAttachReport(
        source=path,
        adapter_name=adapter_name,
        attached_modules=tuple(attached_modules),
        per_component=per_component,
        components=components,
        fused=fuse,
        weight=weight,
        label=label,
    )


def unfuse_lora(pipeline, *, components: tuple[str, ...] | None = None) -> None:
    """Unfuse across every loadable component, for the same reason as fusing."""
    names = tuple(components) if components else _loadable_components(pipeline)
    pipeline.unfuse_lora(components=list(names))
