"""Precision policy for the FSDP backend.

Resolves, per model, the FSDP ``MixedPrecisionPolicy`` dtypes and whether to keep an fp32 master copy.
Most archs train and reshard in bf16; ``glm4_moe_lite`` keeps an fp32 master because the FSDP2 bf16
reshard perturbs weights ~1 bf16 ULP vs a clean disk load, surfacing as a train/rollout logprob gap.
A new arch needing an fp32 master calls ``register_fp32_master_type`` in its spec.
"""

from dataclasses import dataclass

import torch


@dataclass
class PrecisionPolicy:
    param_dtype: torch.dtype  # FSDP MixedPrecisionPolicy compute dtype
    reduce_dtype: torch.dtype  # gradient all-reduce dtype
    keep_fp32_master: bool = False  # keep an fp32 master copy; downcast to on-disk dtype at weight sync


# model_types (substring-matched) whose FSDP2 bf16 reshard needs an fp32 master; registered in specs.
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

    The weight sync downcasts the master back to each param's on-disk dtype (compute still runs bf16 via
    MixedPrecisionPolicy). Dtypes are recorded BEFORE the cast so fp32-on-disk params (e.g. glm's
    ``e_score_correction_bias``) stay fp32 -- casting those to bf16 would flip MoE routing.
    """
    orig_dtypes = {name: p.dtype for name, p in model.state_dict().items()}
    model = model.to(torch.float32)
    model._fsdp_sync_orig_dtypes = orig_dtypes
    return model
