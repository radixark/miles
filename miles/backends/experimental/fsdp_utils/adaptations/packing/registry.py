"""Registry that unifies packed-sequence layout handling across FSDP-backend architectures.

Stateful archs (GatedDeltaNet, Mamba2 hybrid, ...) each reset per-document state under THD packing but
hook different methods/kernels. Each registers a ``PackingPatch`` and the actor dispatches once per
lifetime; archs that need nothing (e.g. ``glm4_moe_lite``) just don't register. All patches share the
boundary derivation in ``boundaries.py``. Two lifetimes:
  * ``"config"``    — patch the transformers classes before model construction; ``apply()`` takes no model.
  * ``"post_load"`` — patch the instantiated model after ``from_pretrained``; ``apply(model)`` takes it.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class PackingPatch:
    name: str
    applies_to: Callable  # (hf_config) -> bool
    lifetime: str  # "config" | "post_load"
    apply: Callable  # config: apply() ; post_load: apply(model) ; both return truthy when applied


_PACKING_PATCHES: list[PackingPatch] = []


def register_packing_patch(patch: PackingPatch) -> None:
    _PACKING_PATCHES.append(patch)


def get_packing_patches(hf_config, lifetime: str) -> list[PackingPatch]:
    return [p for p in _PACKING_PATCHES if p.lifetime == lifetime and p.applies_to(hf_config)]


def apply_packing(target, hf_config, lifetime: str) -> list[str]:
    """Apply every registered packing patch matching this config + lifetime (idempotent). ``target`` is the
    model for ``post_load``, ignored for ``config``. Returns the names that fired (empty when no arch matches)."""
    fired = []
    for p in get_packing_patches(hf_config, lifetime):
        applied = p.apply(target) if lifetime == "post_load" else p.apply()
        if applied or applied is None:
            fired.append(p.name)
    return fired
