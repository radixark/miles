from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy, resolve_fsdp_mesh
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import replicated_state_sharding_config
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.flops import get_nparams_and_active_nparams, quadratic_attention_flops_per_token
from torchtitan.models.common.moe import MoE

from miles.backends.torchtitan_utils.models.glm5_next.layers import (
    DSAAttention,
    Glm5NextRMSNorm,
    HyperConnection,
    KimiDeltaAttention,
    update_residual_streams,
)
from miles.backends.torchtitan_utils.models.glm5_next.packed_sequence import PackedSequence, build_packed_sequence
from miles.backends.torchtitan_utils.models.glm5_next.state_dict_adapter import Glm5NextStateDictAdapter


class Glm5NextBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_attn: HyperConnection.Config
        hc_ffn: HyperConnection.Config
        attention_norm: Glm5NextRMSNorm.Config
        ffn_norm: Glm5NextRMSNorm.Config
        feed_forward: DenseFeedForward.Config | MoE.Config
        kda: KimiDeltaAttention.Config | None = None
        attention: DSAAttention.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        assert (config.kda is None) != (config.attention is None)
        self.hc_attn = config.hc_attn.build()
        self.hc_ffn = config.hc_ffn.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attn = (config.kda or config.attention).build()
        self.feed_forward = config.feed_forward.build()

    def forward(self, x_TND: torch.Tensor, sequence: PackedSequence) -> torch.Tensor:
        aggregated, h_post, h_res = self.hc_attn(x_TND)
        out = self.attn(self.attention_norm(aggregated), sequence)
        x_TND = update_residual_streams(out, x_TND, h_post, h_res)

        aggregated, h_post, h_res = self.hc_ffn(x_TND)
        out = self.feed_forward(self.ffn_norm(aggregated))
        return update_residual_streams(out, x_TND, h_post, h_res)


class Glm5NextModel(Decoder):
    state_dict_adapter_cls = Glm5NextStateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        num_streams: int

        @property
        def first_full_attention(self) -> None:
            return None

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_flops = sum(
                quadratic_attention_flops_per_token(
                    num_heads=layer.attention.n_heads,
                    qk_head_dim=layer.attention.qk_head_dim,
                    v_head_dim=layer.attention.v_head_dim,
                    seq_len=seq_len,
                )
                for layer in self.layers
                if layer.attention is not None
            )
            return nparams, 6 * active_nparams + attention_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_streams = config.num_streams
        self.kda_conv_kernel_size = next(
            (layer.kda.q_conv1d.kernel_size for layer in config.layers if layer.kda), None
        )

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        for module in self.modules():
            if not isinstance(module, Module) or module._sharding_config is not None:
                continue
            states = [name for name, _ in module.named_parameters(recurse=False)]
            states += [name for name, _ in module.named_buffers(recurse=False)]
            if states:
                module._sharding_config = replicated_state_sharding_config(*states)
        super()._parallelize(parallelism_context)

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        fp32_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False
        )
        reshard_after_forward = get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward, parallelism_context.pp_enabled
        )
        offload_policy = CPUOffloadPolicy() if training.enable_cpu_offload else OffloadPolicy()
        for block in self.layers.values():
            modules = [block.hc_attn, block.hc_ffn]
            if isinstance(block.attn, KimiDeltaAttention):
                modules.append(block.attn.gate)
            for module in modules:
                fully_shard(
                    module,
                    mesh=dp_mesh,
                    dp_mesh_dims=dp_mesh_dims,
                    mp_policy=fp32_policy,
                    offload_policy=offload_policy,
                    reshard_after_forward=reshard_after_forward,
                )
        super()._apply_fsdp(parallelism_context=parallelism_context, training=training, parallelism=parallelism)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata=None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert positions is not None and positions.ndim == 1, "GLM-5.3-Flash trains on one packed sequence"
        sequence = build_packed_sequence(
            positions, spmd_mesh_group(MeshAxisName.CP), conv_kernel_size=self.kda_conv_kernel_size
        )
        if self.tok_embeddings is None:
            x_TND = tokens
        else:
            h_TD = self.tok_embeddings(tokens)
            x_TND = h_TD.unsqueeze(-2).expand(*h_TD.shape[:-1], self.num_streams, h_TD.shape[-1]).contiguous()
        for layer in self.layers.values():
            x_TND = layer(x_TND, sequence)
        if self.norm is None:
            return x_TND
        h_TD = self.norm(x_TND.mean(dim=-2))
        if self._skip_lm_head or self.lm_head is None:
            return h_TD
        return self.lm_head(h_TD)
