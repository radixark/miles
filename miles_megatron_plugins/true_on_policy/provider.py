from __future__ import annotations

import warnings

import torch

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from .attention_fa3 import SGLangCoreAttention
from .linear import SGLangColumnParallelLinear, SGLangRowParallelLinear
from .norm import SGLangNorm, SGLangQKRMSNorm


class SGLangSpecProvider(BackendSpecProvider):
    """Backend provider for the correctness-first SGLang-compatible Megatron surface."""

    def column_parallel_linear(self) -> type:
        return SGLangColumnParallelLinear

    def row_parallel_linear(self) -> type:
        return SGLangRowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        return False

    def column_parallel_layer_norm_linear(self) -> type | None:
        return None

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False):
        if for_qk:
            return SGLangQKRMSNorm if rms_norm else SGLangNorm
        return ModuleSpec(
            module=SGLangNorm,
            params={
                "cast_x_before_out_mul": True,
                "override_orig_dtype": torch.float32,
                "keep_weight_fp32": True,
            },
        )

    def core_attention(self) -> type:
        return SGLangCoreAttention

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> tuple[type, MLPSubmodules | None]:
        del moe_use_legacy_grouped_gemm

        if moe_use_grouped_gemm:
            warnings.warn(
                "SGLang backend falls back to Megatron's existing GroupedMLP surface "
                "until the deterministic MoE backend is introduced.",
                stacklevel=2,
            )
            return GroupedMLP, None

        return SequentialMLP, MLPSubmodules(linear_fc1=SGLangColumnParallelLinear, linear_fc2=SGLangRowParallelLinear)

    def activation_func(self) -> type:
        return None
