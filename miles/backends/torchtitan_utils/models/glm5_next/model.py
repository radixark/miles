from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn
from torchtitan.models.common.attention import VarlenMetadata, create_varlen_metadata_for_document
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.token_dispatcher import update_ep_token_dispatcher_config
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

from miles.backends.torchtitan_utils.models.glm5_next.layers import (
    ClampedFeedForward,
    DSAAttention,
    Glm5NextRMSNorm,
    HyperConnection,
    KimiDeltaAttention,
    hc_post,
)

_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


class Glm5NextBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_attn: HyperConnection.Config
        hc_ffn: HyperConnection.Config
        attention_norm: Glm5NextRMSNorm.Config
        ffn_norm: Glm5NextRMSNorm.Config
        kda: KimiDeltaAttention.Config | None = None
        attention: DSAAttention.Config | None = None
        feed_forward: ClampedFeedForward.Config | None = None
        moe: MoE.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        assert (config.kda is None) != (config.attention is None)
        assert (config.feed_forward is None) != (config.moe is None)
        self.hc_attn = config.hc_attn.build()
        self.hc_ffn = config.hc_ffn.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attn = (config.kda or config.attention).build()
        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            self.moe = config.moe.build()
        else:
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x_BLND: torch.Tensor,
        attention_masks: VarlenMetadata,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        aggregated, h_post, h_res = self.hc_attn(x_BLND)
        out = self.attn(self.attention_norm(aggregated), attention_masks)
        x_BLND = hc_post(out, x_BLND, h_post, h_res)

        aggregated, h_post, h_res = self.hc_ffn(x_BLND)
        ffn = self.moe if self.moe_enabled else self.feed_forward
        out = ffn(self.ffn_norm(aggregated))
        return hc_post(out, x_BLND, h_post, h_res)


class Glm5NextModel(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        num_streams: int
        max_position_embeddings: int

        @property
        def max_seq_len(self) -> int:
            return self.max_position_embeddings

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            unsupported = {
                "tensor": parallelism.tensor_parallel_degree,
                "context": parallelism.context_parallel_degree,
                "pipeline": parallelism.pipeline_parallel_degree,
            }
            for kind, degree in unsupported.items():
                if degree > 1:
                    raise NotImplementedError(
                        f"GLM-5.3-Flash on torchtitan supports FSDP + EP only, not {kind} parallel"
                    )
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
            return get_moe_model_nparams_and_flops(
                self,
                model,
                attention.n_heads,
                attention.qk_head_dim + attention.v_head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_streams = config.num_streams

    def get_attention_masks(self, positions: torch.Tensor) -> VarlenMetadata:
        return create_varlen_metadata_for_document(positions, include_host_offsets=True)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: VarlenMetadata | None = None,
    ) -> torch.Tensor:
        h_BLD = self.tok_embeddings(tokens)
        x_BLND = h_BLD.unsqueeze(-2).expand(*h_BLD.shape[:-1], self.num_streams, h_BLD.shape[-1]).contiguous()
        for layer in self.layers.values():
            x_BLND = layer(x_BLND, attention_masks, positions)
        h_BLD = self.norm(x_BLND.mean(dim=-2))
        if self._skip_lm_head:
            return h_BLD
        return self.lm_head(h_BLD)
