"""Precision policy for the FSDP backend.

Resolves, per model, the FSDP ``MixedPrecisionPolicy`` dtypes and whether to keep an fp32 master
copy of the weights. Most archs train and reshard in bf16 directly; ``glm4_moe_lite`` keeps an
fp32 master because the FSDP2 bf16 reshard perturbs stored weights ~1 bf16 ULP vs a clean disk
load, which surfaces as a train<->rollout logprob gap.

This is the precision half of the per-arch adaptation layer (alongside weight_bridge,
class_patches, packing, post_load_fixups); a new arch that needs an fp32 master calls
``register_fp32_master_type`` in its spec (adaptations/specs/) instead of editing the actor.
"""

from dataclasses import dataclass

import torch


@dataclass
class PrecisionPolicy:
    param_dtype: torch.dtype  # FSDP MixedPrecisionPolicy compute dtype
    reduce_dtype: torch.dtype  # gradient all-reduce dtype
    keep_fp32_master: bool = False  # keep an fp32 master copy; downcast to on-disk dtype at weight sync


# model_types (matched as a substring of model_type) whose FSDP2 bf16 reshard needs an fp32 master.
# Archs register themselves in their spec (adaptations/specs/) so this module stays pure mechanism.
_FP32_MASTER_TYPES: set[str] = set()


def register_fp32_master_type(model_type: str) -> None:
    _FP32_MASTER_TYPES.add(model_type)


def resolve_precision_policy(hf_config, args) -> PrecisionPolicy:
    """Resolve the precision policy for this model. param_dtype follows args.fp16; reduce stays fp32."""
    param_dtype = torch.float16 if getattr(args, "fp16", False) else torch.bfloat16
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    keep_fp32_master = any(t in model_type for t in _FP32_MASTER_TYPES)
    return PrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, keep_fp32_master=keep_fp32_master)


def apply_fp32_master(model):
    """Convert ``model`` to an fp32 master in place, recording each param's on-disk dtype first.

    The fp32 master is bit-exact across the FSDP2 reshard; the weight sync downcasts it back to each
    param's on-disk dtype so sglang receives exactly what a clean disk load produces (compute still
    runs bf16 via MixedPrecisionPolicy). Dtypes are recorded BEFORE the float32 cast so fp32-on-disk
    params (e.g. glm's ``e_score_correction_bias``) stay fp32 -- casting those to bf16 would flip MoE
    routing. update_weight_utils reads ``model._fsdp_sync_orig_dtypes``.
    """
    orig_dtypes = {name: p.dtype for name, p in model.state_dict().items()}
    model = model.to(torch.float32)
    model._fsdp_sync_orig_dtypes = orig_dtypes
    return model
