"""Precision policy for the FSDP backend.

Resolves the FSDP ``MixedPrecisionPolicy`` dtypes and whether to keep an fp32 master copy. The master copy is enabled by default for bit-exact weight sync and can be disabled for memory-constrained runs that only require forward accuracy.
"""

from dataclasses import dataclass

import torch


@dataclass
class PrecisionPolicy:
    param_dtype: torch.dtype  # FSDP MixedPrecisionPolicy compute dtype
    reduce_dtype: torch.dtype  # gradient all-reduce dtype
    keep_fp32_master: bool = True  # keep an fp32 master copy; downcast to on-disk dtype at weight sync


def resolve_precision_policy(hf_config, args) -> PrecisionPolicy:
    """Resolve compute, reduction, and master-weight precision from FSDP arguments."""
    param_dtype = torch.float16 if getattr(args, "fp16", False) else torch.bfloat16
    return PrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        keep_fp32_master=args.enable_fp32_master,
    )


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
