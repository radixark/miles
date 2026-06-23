"""Registry that unifies packed-sequence layout handling across FSDP-backend architectures.

Stateful archs (GatedDeltaNet, Mamba2 hybrid, ...) each need to reset their per-document state
under THD packing, but they hook different methods/kernels. Rather than a hardcoded
``if "nemotron_h" in model_type: ...`` in the actor plus a separate ModelPatchHook for GDN, every
packing arch registers a ``PackingPatch`` here and the actor dispatches once per lifetime. Archs
that pack register a patch; archs that need nothing (e.g. ``glm4_moe_lite``, whose MLA already does
native HF varlen via FlashAttentionKwargs) simply don't — ``apply_packing`` is a no-op for them.
All patches share the one boundary derivation in ``boundaries.py``.

This is the third FSDP-backend registry, alongside ``weight_bridge`` (train->rollout param
contract) and ``class_patches.ModelPatchHook`` (config-time HF-compat patches); a new arch
plugs into a registry instead of editing a sync/patch/dispatch loop.

Two lifetimes, because the archs patch at different times:
  * ``"config"``    — patch the transformers *classes* before model construction (GatedDeltaNet
                      patches ``DecoderLayer``/``GatedDeltaNet`` class forwards); ``apply()`` takes
                      no model.
  * ``"post_load"`` — patch the *instantiated* model after ``from_pretrained`` (NemotronH's remote
                      ``trust_remote_code`` modeling needs the live module tree to find the Mamba2
                      mixer + attention classes); ``apply(model)`` takes the model.
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
    """Apply every registered packing patch matching this config + lifetime. Idempotent.

    ``target`` is the instantiated model for ``lifetime == "post_load"`` and ignored for
    ``"config"``. Returns the names of the patches that fired (for logging/tests). Empty when no
    arch matches — e.g. dense Qwen3, qwen3_moe, or glm4_moe_lite.
    """
    fired = []
    for p in get_packing_patches(hf_config, lifetime):
        applied = p.apply(target) if lifetime == "post_load" else p.apply()
        if applied or applied is None:
            fired.append(p.name)
    return fired
