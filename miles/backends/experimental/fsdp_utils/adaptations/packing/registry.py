"""Registry unifying packed-sequence layout handling across FSDP-backend architectures."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
    """Apply every registered packing patch matching this config + lifetime (idempotent); return names fired."""
    fired = []
    for p in get_packing_patches(hf_config, lifetime):
        applied = p.apply(target) if lifetime == "post_load" else p.apply()
        if applied or applied is None:
            fired.append(p.name)
        else:
            logger.warning(
                "[fsdp packing] patch %r matched model_type=%r but did not apply; "
                "packed-document state resets are OFF for this arch",
                p.name,
                getattr(hf_config, "model_type", None),
            )
    return fired
