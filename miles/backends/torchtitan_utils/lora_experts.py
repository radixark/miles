"""Shared-outer routed-expert LoRA for the torchtitan backend: the layout megatron's ``GroupedExpertsSpec`` trains
and SGLang serves with ``--experts-shared-outer-loras``.

gate/up (``num_linears`` 2): one ``lora_a`` [2, r, D] for all experts, per-expert ``lora_b`` [E, 2, F, r].
down (``num_linears`` 1): per-expert ``lora_a`` [E, r, F], one ``lora_b`` [1, D, r] for all experts.
The per-expert factor is sharded like the base experts; the shared one is replicated."""

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import GroupedLinear


class _LoRAFactor(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        shape: tuple[int, ...]
        zero: bool

    def __init__(self, config: Config):
        super().__init__()
        self.zero = config.zero
        self.weight = nn.Parameter(torch.empty(config.shape))

    def reset_parameters(self) -> None:
        if self.zero:
            nn.init.zeros_(self.weight)
        else:
            # fan-in is the input width, as torchtitan's LoRA initializes lora_A
            nn.init.kaiming_uniform_(self.weight.flatten(0, -2), a=math.sqrt(5))


class SharedOuterLoRAGroupedLinear(GroupedLinear):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        rank: int
        alpha: float

    def __init__(self, config: Config):
        super().__init__(config)
        self.weight.requires_grad_(False)
        self.rank = config.rank
        self._lora_scaling = config.alpha / config.rank
        experts, outputs = config.group_size, config.out_features
        per_expert, shared = _factor_shardings(config.sharding_config)
        if config.num_linears > 1:
            a_shape, a_sharding = (config.num_linears, config.rank, config.in_features), shared
            b_shape, b_sharding = (experts, config.num_linears, outputs, config.rank), per_expert
        else:
            a_shape, a_sharding = (experts, config.rank, config.in_features), per_expert
            b_shape, b_sharding = (1, outputs, config.rank), shared
        self.lora_a = _LoRAFactor.Config(shape=a_shape, zero=False, sharding_config=a_sharding).build()
        self.lora_b = _LoRAFactor.Config(shape=b_shape, zero=True, sharding_config=b_sharding).build()

    def _grouped_mm(
        self, *, input_RI: torch.Tensor, weight_EOI: torch.Tensor, offsets_E: torch.Tensor
    ) -> torch.Tensor:
        base_RO = super()._grouped_mm(input_RI=input_RI, weight_EOI=weight_EOI, offsets_E=offsets_E)
        x_RI = input_RI.bfloat16()
        lora_a, lora_b = self.lora_a.weight.bfloat16(), self.lora_b.weight.bfloat16()
        if self.num_linears > 1:
            hidden_RL = x_RI @ lora_a.flatten(0, 1).t()
            lora_RO = torch.cat(
                [
                    torch._grouped_mm(
                        hidden_RL[:, i * self.rank : (i + 1) * self.rank].contiguous(),
                        lora_b[:, i].transpose(-2, -1),
                        offs=offsets_E,
                    )
                    for i in range(self.num_linears)
                ],
                dim=-1,
            )
        else:
            inner_RL = torch._grouped_mm(x_RI, lora_a.transpose(-2, -1), offs=offsets_E)
            lora_RO = inner_RL @ lora_b[0].t()
        return base_RO + self._lora_scaling * lora_RO


def _factor_shardings(base: ShardingConfig | None) -> tuple[ShardingConfig | None, ShardingConfig | None]:
    """(per-expert, shared): the per-expert factor takes the base experts' placement, the shared one is replicated."""
    base_weight = base.state_shardings.get("weight") if base is not None else None
    if base_weight is None:
        return None, None
    return (
        ShardingConfig(state_shardings={"weight": base_weight}),
        ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)}),
    )
