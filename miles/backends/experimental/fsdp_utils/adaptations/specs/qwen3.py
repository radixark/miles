"""Qwen3 dense adaptations for the formal true-on-policy precision contract."""

from dataclasses import replace

import torch

from miles.true_on_policy.contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1

from ..class_patches import ModelInstancePatchHook, register_model_instance_patch
from ..precision import PrecisionPolicyHook, register_precision_policy


def _is_qwen3(hf_config) -> bool:
    return str(getattr(hf_config, "model_type", "") or "") == "qwen3"


def _uses_formal_contract(hf_config, args) -> bool:
    return (
        _is_qwen3(hf_config)
        and getattr(args, "true_on_policy_mode", False)
        and getattr(args, "sglang_true_on_policy_contract", None) == QWEN3_DENSE_TRUE_ON_POLICY_V1.name
    )


def _resolve_precision(base_policy, hf_config, args):
    if getattr(args, "fp16", False):
        raise ValueError(f"{QWEN3_DENSE_TRUE_ON_POLICY_V1.name} requires bf16 training")
    if not base_policy.keep_fp32_master:
        raise ValueError(f"{QWEN3_DENSE_TRUE_ON_POLICY_V1.name} requires fp32 master weights")
    return replace(
        base_policy,
        param_dtype=torch.float32,
        autocast_dtype=torch.bfloat16,
        sync_dtype_resolver=_resolve_sync_dtype,
    )


def _resolve_sync_dtype(name, checkpoint_dtype):
    from ...models.qwen3 import resolve_qwen3_dense_sync_dtype

    return resolve_qwen3_dense_sync_dtype(name, checkpoint_dtype)


def _instance_patch_applies(hf_config, args) -> bool:
    return _uses_formal_contract(hf_config, args) and not getattr(args, "fp16", False)


def _apply_model_patch(model) -> None:
    from ...models.qwen3 import apply_qwen3_dense_true_on_policy_patch

    apply_qwen3_dense_true_on_policy_patch(model)


register_precision_policy(PrecisionPolicyHook("qwen3_dense_true_on_policy", _uses_formal_contract, _resolve_precision))
register_model_instance_patch(
    ModelInstancePatchHook("qwen3_dense_true_on_policy", _instance_patch_applies, _apply_model_patch)
)
