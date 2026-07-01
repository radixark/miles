"""Precision policy for the FSDP backend: resolve MixedPrecisionPolicy dtypes and whether to keep an fp32 master."""

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
