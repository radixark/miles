"""Kimi K3 specs: custom KDA/MLA attention modules and latent MoE experts under ``block_sparse_moe``."""

from __future__ import annotations

import torch.nn as nn

from miles_plugins.lora.modules.kimi_k3 import KimiK3AttentionAdapter, KimiK3ExpertsAdapter, KimiK3MLPAdapter
from miles_plugins.lora.spec.base import AttachContext, AttentionFamily

_EXPERTS_BLOCK = "block_sparse_moe.experts."


class KimiK3AttentionSpec:
    name = "kimi_k3"
    family = AttentionFamily.MLA
    hf_block = "self_attn"
    supported_targets = frozenset({"o_proj", "q_a_proj", "kv_a_proj_with_mqa"})

    def validate(self, config, *, tp_size: int) -> None:
        del config, tp_size

    def serving_fused_families(self) -> list[frozenset[str]]:
        return []

    def attach(self, attention: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        attention.lora_adapter = KimiK3AttentionAdapter(hf_prefix=hf_prefix, attention=attention, context=context)
        return 1


class KimiK3MLPSpec:
    name = "kimi_k3_mlp"
    supported_targets = frozenset({"gate_proj", "up_proj", "down_proj"})

    def serving_fused_families(self) -> list[frozenset[str]]:
        return []

    def attach(self, mlp: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        mlp.lora_adapter = KimiK3MLPAdapter(hf_prefix=hf_prefix, mlp=mlp, context=context)
        return 1


class KimiK3ExpertsSpec:
    def attach(self, mlp: nn.Module, hf_layer_prefix: str, context: AttachContext) -> int:
        if not hasattr(mlp, "experts"):
            return 0
        assert (mlp.config.expert_tensor_parallel_size or 1) == 1, "Kimi K3 native expert LoRA requires ETP=1"
        assert mlp.shared_experts is not None, f"Kimi K3 MoE layer {hf_layer_prefix!r} is missing shared experts"
        prefix = f"{hf_layer_prefix}{_EXPERTS_BLOCK}"
        mlp.experts.lora_adapter = KimiK3ExpertsAdapter(
            hf_prefix=prefix, moe=mlp, context=context, include_fc2=context.selects(f"{prefix}0.w2")
        )
        return 1


def shared_outer_serving_targets(targets: list[str]) -> list[str]:
    """Shared-outer export stores the expert dimension in each tensor, not in its name."""
    return [target.replace(".experts.*.", ".experts.") for target in targets]
