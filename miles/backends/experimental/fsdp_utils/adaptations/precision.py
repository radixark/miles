"""Precision policy for the FSDP backend.

Resolves the FSDP ``MixedPrecisionPolicy`` dtypes and whether to keep an fp32 master copy. The master copy is enabled by default for bit-exact weight sync and can be disabled for memory-constrained runs that only require forward accuracy.
"""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass
class PrecisionPolicy:
    param_dtype: torch.dtype  # FSDP MixedPrecisionPolicy compute dtype
    reduce_dtype: torch.dtype  # gradient all-reduce dtype
    keep_fp32_master: bool = True
    autocast_dtype: torch.dtype | None = None
    sync_dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None


@dataclass
class PrecisionPolicyHook:
    name: str
    applies_to: Callable  # (hf_config, args) -> bool
    resolve: Callable  # (base_policy, hf_config, args) -> PrecisionPolicy


_PRECISION_POLICY_HOOKS: list[PrecisionPolicyHook] = []


def register_precision_policy(hook: PrecisionPolicyHook) -> None:
    _PRECISION_POLICY_HOOKS.append(hook)


def resolve_precision_policy(hf_config, args) -> PrecisionPolicy:
    """Resolve compute, reduction, master-weight, and forward-autocast precision."""
    policy = PrecisionPolicy(
        param_dtype=torch.float16 if getattr(args, "fp16", False) else torch.bfloat16,
        reduce_dtype=torch.float32,
        keep_fp32_master=args.keep_fp32_master,
    )
    for hook in _PRECISION_POLICY_HOOKS:
        if hook.applies_to(hf_config, args):
            policy = hook.resolve(policy, hf_config, args)
    return policy


def precision_forward_context(policy: PrecisionPolicy):
    if policy.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=policy.autocast_dtype)


def apply_fp32_master(
    model,
    sync_dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None,
):
    """Convert ``model`` to an fp32 master and record each parameter's outbound sync dtype.

    The checkpoint dtype is the default. A model-specific precision policy may override it when the
    rollout contract stores selected parameters at a different dtype.
    """
    sync_dtypes = {}
    for name, param in model.state_dict().items():
        checkpoint_dtype = param.dtype
        sync_dtypes[name] = (
            sync_dtype_resolver(name, checkpoint_dtype) if sync_dtype_resolver is not None else checkpoint_dtype
        )
    model = model.to(torch.float32)
    model._fsdp_sync_dtypes = sync_dtypes
    return model
