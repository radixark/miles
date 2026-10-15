from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.token_dispatcher import update_ep_token_dispatcher_config
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

from miles.backends.torchtitan_utils.models.step5.layers import Step5Attention, Step5FeedForward, Step5RMSNorm
from miles.backends.torchtitan_utils.models.step5.packed_sequence import (
    ContextParallelLayout,
    PackedSequence,
    build_packed_sequence,
)

_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


class Step5Block(Module):
    """A pre-norm block over the FP32 residual stream: each sublayer reads a BF16 norm of it and adds
    its BF16 output back in FP32."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention_norm: Step5RMSNorm.Config
        ffn_norm: Step5RMSNorm.Config
        attention: Step5Attention.Config
        feed_forward: Step5FeedForward.Config | None = None
        moe: MoE.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        assert (config.feed_forward is None) != (config.moe is None)
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attention = config.attention.build()
        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            self.moe = config.moe.build()
        else:
            self.feed_forward = config.feed_forward.build()

    def forward(self, h_BLD: torch.Tensor, sequence: PackedSequence, positions: torch.Tensor) -> torch.Tensor:
        h_BLD = h_BLD + self.attention(self.attention_norm(h_BLD), sequence, positions)
        ffn = self.moe if self.moe_enabled else self.feed_forward
        return h_BLD + ffn(self.ffn_norm(h_BLD))


class Step5Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        max_position_embeddings: int
        sliding_window: int

        @property
        def max_seq_len(self) -> int:
            return self.max_position_embeddings

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError("Step5 on torchtitan does not support tensor parallelism yet")
            if config.training.seq_len > self.max_position_embeddings:
                raise ValueError(
                    f"training.seq_len {config.training.seq_len} exceeds max_position_embeddings "
                    f"{self.max_position_embeddings}"
                )

            update_ep_token_dispatcher_config(self, config)
            for layer_cfg in self.layers:
                if layer_cfg.moe is None:
                    continue
                layer_cfg.moe.router._debug_force_load_balance = config.debug.moe_force_load_balance
                set_moe_sharding_config(
                    layer_cfg.moe,
                    enable_ep=parallelism.expert_parallel_degree > 1,
                    enable_sp=False,
                    expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
                )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            attention = self.first_attention
            return get_moe_model_nparams_and_flops(self, model, attention.n_heads, 2 * attention.head_dim, seq_len)

    def __init__(self, config: Config):
        super().__init__(config)
        self.sliding_window = config.sliding_window
        self._cp_mesh: DeviceMesh | None = None
        self._cp_load_balancer: str | None = None
        self._cp_layouts: dict[int, ContextParallelLayout] = {}

    def enable_context_parallel(self, mesh: DeviceMesh, *, load_balancer: str) -> None:
        self._cp_mesh = mesh
        self._cp_load_balancer = load_balancer
        self._cp_layouts = {}

    def _cp_layout(self, local_len: int, device) -> ContextParallelLayout | None:
        if self._cp_mesh is None:
            return None
        if local_len not in self._cp_layouts:
            self._cp_layouts[local_len] = ContextParallelLayout.build(
                self._cp_mesh,
                load_balancer=self._cp_load_balancer,
                seq_len=local_len * self._cp_mesh.size(),
                device=device,
            )
        return self._cp_layouts[local_len]

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks=None,
    ) -> torch.Tensor:
        assert positions is not None and positions.shape[0] == 1, "Step5 trains on one packed sequence"
        sequence = build_packed_sequence(
            positions, self._cp_layout(positions.shape[1], positions.device), window=self.sliding_window
        )
        # a pipeline stage after the first receives the FP32 residual stream instead of tokens
        h_BLD = tokens if self.tok_embeddings is None else self.tok_embeddings(tokens).float()
        for layer in self.layers.values():
            h_BLD = layer(h_BLD, sequence, positions)
        if self.norm is None:
            return h_BLD
        h_BLD = self.norm(h_BLD)
        if self._skip_lm_head or self.lm_head is None:
            return h_BLD
        return self.lm_head(h_BLD)
