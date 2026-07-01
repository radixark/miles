"""HuggingFace-version compatibility patches for the experimental FSDP backend."""

import logging

logger = logging.getLogger(__name__)


class ModelPatchHook:
    """A config-time patch: an applies_to(hf_config) predicate + an apply(hf_config, args) action."""

    def __init__(self, name, applies_to, apply):
        self.name = name
        self.applies_to = applies_to
        self.apply = apply


_MODEL_PATCH_HOOKS: list[ModelPatchHook] = []


def register_model_patch(hook: ModelPatchHook) -> None:
    _MODEL_PATCH_HOOKS.append(hook)


def _always(hf_config) -> bool:
    return True


def _has_config(hf_config) -> bool:
    return hf_config is not None


# Per-arch model patches register in their spec (adaptations/specs/); this module keeps only generic ones.


def apply_class_patches(hf_config=None, args=None) -> None:
    """Apply all registered ModelPatchHooks. Safe to call once at actor init."""
    for hook in _MODEL_PATCH_HOOKS:
        if hook.applies_to(hf_config):
            hook.apply(hf_config, args)
