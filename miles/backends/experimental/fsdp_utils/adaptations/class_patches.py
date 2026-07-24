"""HuggingFace-version compatibility patches for the experimental FSDP backend.

The FSDP backend trains the stock HF model, so it's sensitive to transformers-version drift; these
idempotent patches keep the training forward runnable and warn when it diverges from the SGLang rollout.
"""

import logging

logger = logging.getLogger(__name__)


def check_train_infer_consistency(hf_config) -> None:
    """Warn when an arch's training forward diverges structurally from the rollout (e.g. DeepSeek DSA: the
    sparse-attention indexer is absent from HF training, so train is dense while the rollout is sparse)."""
    model_type = str(getattr(hf_config, "model_type", "") or "")
    is_dsa = (
        "deepseek_v3" in model_type
        or bool(getattr(hf_config, "index_topk", None))
        or getattr(hf_config, "attn_module_list_cfg", None) is not None
    )
    if is_dsa:
        logger.warning(
            "[fsdp class_patches] DeepSeek sparse-attention (DSA) detected (model_type=%s): the HF "
            "training forward has no indexer, so it is dropped and train attention is DENSE while "
            "the rollout is SPARSE. RL on DSA via FSDP is not currently consistent.",
            model_type,
        )


def check_fp8_checkpoint(hf_config) -> None:
    """Fail fast on native-fp8 checkpoints (the actor has no inline dequant)."""
    qc = getattr(hf_config, "quantization_config", None)
    if not qc:
        return
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    if str(method or "").lower() == "fp8":
        raise ValueError(
            "FSDP backend cannot train from an fp8-quantized checkpoint "
            "(quantization_config.quant_method='fp8'). Convert to bf16 first:\n"
            "  python tools/fp8_cast_bf16.py --input-fp8-hf-path <src> --output-bf16-hf-path <dst>\n"
            "then copy config/tokenizer (dropping quantization_config) into <dst> and point "
            "--hf-checkpoint at it."
        )


class ModelPatchHook:
    """A config-time patch: an ``applies_to(hf_config)`` predicate + an ``apply(hf_config, args)`` action
    (``args`` is the actor Namespace). New archs register a hook instead of editing ``apply_class_patches``."""

    def __init__(self, name, applies_to, apply):
        self.name = name
        self.applies_to = applies_to
        self.apply = apply


_MODEL_PATCH_HOOKS: list[ModelPatchHook] = []


def register_model_patch(hook: ModelPatchHook) -> None:
    _MODEL_PATCH_HOOKS.append(hook)


class ModelInstancePatchHook:
    """A post-construction patch applied only to matching model instances."""

    def __init__(self, name, applies_to, apply):
        self.name = name
        self.applies_to = applies_to
        self.apply = apply


_MODEL_INSTANCE_PATCH_HOOKS: list[ModelInstancePatchHook] = []


def register_model_instance_patch(hook: ModelInstancePatchHook) -> None:
    _MODEL_INSTANCE_PATCH_HOOKS.append(hook)


def _has_config(hf_config) -> bool:
    return hf_config is not None


register_model_patch(ModelPatchHook("fp8_checkpoint_guard", _has_config, lambda cfg, args: check_fp8_checkpoint(cfg)))
register_model_patch(
    ModelPatchHook("dsa_train_infer_warn", _has_config, lambda cfg, args: check_train_infer_consistency(cfg))
)
# Per-arch model patches register in their spec (adaptations/specs/); this module keeps only generic ones.


def apply_class_patches(hf_config=None, args=None) -> None:
    """Apply all registered ModelPatchHooks. Safe to call once at actor init."""
    for hook in _MODEL_PATCH_HOOKS:
        if hook.applies_to(hf_config):
            hook.apply(hf_config, args)


def apply_model_instance_patches(model, hf_config=None, args=None) -> None:
    """Apply matching instance-local patches after model construction."""
    for hook in _MODEL_INSTANCE_PATCH_HOOKS:
        if hook.applies_to(hf_config, args):
            hook.apply(model)
