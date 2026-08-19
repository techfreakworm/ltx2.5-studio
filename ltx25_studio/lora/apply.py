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
names and the wrong dimensions really does attach. Catching that is stage 6.5's
job in ``validate.py``, which is why deleting it as redundant would be a
mistake.

This is the only module in ``lora/`` that touches a real model. Even so it
imports nothing heavy -- everything here is duck-typed against the pipeline, so
the suite runs with no diffusers, no peft and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import LoRAError, LoRAFormatError, LoRANotAttachedError

__all__ = ["DEFAULT_ADAPTER_NAME", "LoRAAttachReport", "apply_lora", "unfuse_lora"]

DEFAULT_ADAPTER_NAME = "default"

#: Fallback when a pipeline does not advertise its own. Upstream's
#: ``LTX2LoraLoaderMixin`` sets exactly these two.
FALLBACK_LOADABLE_MODULES = ("transformer", "connectors")


@dataclass(frozen=True)
class LoRAAttachReport:
    """Proof of what actually happened, not what was requested."""

    source: Path
    adapter_name: str
    attached_modules: tuple[str, ...]
    per_component: dict[str, int]
    components: tuple[str, ...]
    fused: bool = False

    @property
    def attached(self) -> int:
        return len(self.attached_modules)


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


def apply_lora(
    pipeline,
    source,
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    weight: float = 1.0,
    fuse: bool = False,
) -> LoRAAttachReport:
    """Load an adapter and assert it attached to more than zero modules.

    Note there is deliberately **no flag that skips the assertion**. An escape
    hatch is how this check quietly stops running six months from now, and it
    must run on every load, built-in adapters included.
    """
    path = Path(source)
    components = _loadable_components(pipeline)

    try:
        pipeline.load_lora_weights(path, adapter_name=adapter_name)
    except LoRAError:
        raise
    except ValueError as exc:
        # Upstream's substring rejection arrives as a bare ValueError. Errors
        # cross layers as our typed exceptions (PLAN §7.2).
        raise LoRAFormatError(source=str(path), reason=str(exc)) from exc
    except Exception as exc:
        raise LoRAFormatError(
            source=str(path), reason=f"the adapter could not be loaded ({exc})"
        ) from exc

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
    )


def unfuse_lora(pipeline, *, components: tuple[str, ...] | None = None) -> None:
    """Unfuse across every loadable component, for the same reason as fusing."""
    names = tuple(components) if components else _loadable_components(pipeline)
    pipeline.unfuse_lora(components=list(names))
