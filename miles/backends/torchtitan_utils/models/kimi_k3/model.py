"""Kimi K3 model: KDA/NoPE MLA, latent MoE and attention residuals."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial
from torchtitan.models.common import Embedding
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict

from miles.backends.torchtitan_utils.models.kimi_k3.attention import DeltaAttention, MLAAttention
from miles.backends.torchtitan_utils.models.kimi_k3.context_parallel import ContextParallelLayout
from miles.backends.torchtitan_utils.models.kimi_k3.ops import (
    DenseMLP,
    KimiTokenDispatcher,
    RMSNorm,
    SituExperts,
    attention_residual,
    linear,
    packed_metadata,
)
from miles_plugins.models.cp_utils import build_fla_cp_context


class LatentMoE(Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts, self.top_k = config.num_experts, config.top_k
        self.route_scale = config.route_scale
        self.gate = linear(config.dim, config.num_experts)
        self.gate.register_buffer("e_score_correction_bias", torch.zeros(config.num_experts, dtype=torch.float32))
        self.routed_expert_down_proj = linear(config.dim, config.latent_dim)
        self.routed_expert_up_proj = linear(config.latent_dim, config.dim)
        self.routed_expert_norm = RMSNorm(config.latent_dim, eps=config.norm_eps)
        self.shared_experts = DenseMLP(config.dim, config.moe_intermediate * config.num_shared_experts)
        self.routed_experts = RoutedExperts.Config(
            inner_experts=SituExperts.Config(
                dim=config.latent_dim, hidden_dim=config.moe_intermediate, num_experts=config.num_experts
            ),
            token_dispatcher=KimiTokenDispatcher.Config(num_experts=config.num_experts, top_k=config.top_k),
        ).build()

    def _init_self_buffers(self, *, buffer_device=None):
        self.gate.e_score_correction_bias.zero_()

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        choice = scores + self.gate.e_score_correction_bias.float()
        indices = choice.topk(self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(-1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        local_indices = indices.to_local() if isinstance(indices, DTensor) else indices
        counts = torch.bincount(local_indices.flatten(), minlength=self.num_experts)
        if isinstance(indices, DTensor):
            counts = DTensor.from_local(counts, indices.device_mesh, [Partial()], run_check=False)
        latent = self.routed_expert_down_proj(x)
        routed = self.routed_experts(latent, weights, indices, counts)
        return self.routed_expert_up_proj(self.routed_expert_norm(routed)) + self.shared_experts(x)


class KimiBlock(Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx, self.block_size = layer_idx, config.attn_res_block_size
        self.self_attn = MLAAttention(config) if layer_idx + 1 in config.full_attn_layers else DeltaAttention(config)
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.self_attention_res_norm, self.mlp_res_norm = RMSNorm(config.dim, eps=config.norm_eps), RMSNorm(
            config.dim, eps=config.norm_eps
        )
        self.self_attention_res_proj, self.mlp_res_proj = linear(config.dim, 1), linear(config.dim, 1)
        self.moe_enabled = layer_idx >= config.dense_layers
        if self.moe_enabled:
            self.moe = LatentMoE(config)
        else:
            self.mlp = DenseMLP(config.dim, config.dense_intermediate)

    def forward(self, x, bank, cu_seqlens, mask, cp_layout=None, cp_context=None):
        prefix = x
        if bank.shape[-2]:
            x = attention_residual(prefix, bank, self.self_attention_res_norm, self.self_attention_res_proj)
        if self.layer_idx % self.block_size == 0:
            bank = torch.cat((bank, prefix.unsqueeze(-2)), dim=-2)
            prefix = None
        x = self.self_attn(self.input_layernorm(x), cu_seqlens, mask, cp_layout, cp_context)
        prefix = x if prefix is None else prefix + x
        x = attention_residual(prefix, bank, self.mlp_res_norm, self.mlp_res_proj)
        x = self.post_attention_layernorm(x)
        return prefix + (self.moe(x) if self.moe_enabled else self.mlp(x)), bank


class KimiK3Model(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 7168
        vocab_size: int = 163840
        n_layers: int = 4
        n_heads: int = 96
        head_dim: int = 128
        mla_heads: int = 96
        full_attn_layers: tuple[int, ...] = (4,)
        q_lora_rank: int = 1536
        kv_lora_rank: int = 512
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        conv_kernel_size: int = 4
        gate_lower_bound: float = -5.0
        norm_eps: float = 1e-5
        num_experts: int = 64
        top_k: int = 16
        num_shared_experts: int = 2
        route_scale: float = 1.0
        latent_dim: int = 3584
        dense_layers: int = 1
        dense_intermediate: int = 33792
        moe_intermediate: int = 3072
        attn_res_block_size: int = 12
        enable_weight_tying: bool = False

        def update_from_config(self, *, config, **kwargs):
            p = config.parallelism
            if p.pipeline_parallel_degree != 1:
                raise ValueError("Kimi K3 currently requires PP=1")
            if p.context_parallel_degree > 1 and p.context_parallel_load_balancer != "headtail":
                raise ValueError("Kimi K3 context parallelism requires headtail load balancing")
            if (
                self.n_heads % p.tensor_parallel_degree
                or self.mla_heads % p.tensor_parallel_degree
                or self.num_experts % p.expert_parallel_degree
            ):
                raise ValueError("K3 heads/experts must divide evenly across TP/EP")
            if p.tensor_parallel_degree > 1 and p.expert_parallel_degree != p.tensor_parallel_degree:
                raise ValueError("K3 tensor parallel training currently requires EP=TP")

        def get_nparams_and_flops(self, model, seq_len):
            total = sum(p.numel() for p in model.parameters())
            experts = sum(p.numel() for n, p in model.named_parameters() if "inner_experts" in n)
            active = total - experts + experts * self.top_k // self.num_experts
            qk_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            return total, 6 * active + 12 * len(self.full_attn_layers) * self.mla_heads * qk_dim * seq_len

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.enable_weight_tying = False
        self._skip_lm_head = False
        self.cp_mesh = None
        self.tok_embeddings = Embedding.Config(num_embeddings=config.vocab_size, embedding_dim=config.dim).build()
        self.layers = ModuleDict({str(i): KimiBlock(config, i) for i in range(config.n_layers)})
        self.output_attn_res_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output_attn_res_proj = linear(config.dim, 1)
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = linear(config.dim, config.vocab_size)

    def forward(self, tokens, *, positions=None, cu_seqlens=None, **kwargs):
        cp_layout, cp_context = None, None
        metadata_tokens = tokens
        if self.cp_mesh is not None:
            if tokens.shape[0] != 1:
                raise ValueError("Kimi K3 CP expects one flattened packed row")
            cp_layout = ContextParallelLayout.build(self.cp_mesh, tokens.shape[1], tokens.device)
            metadata_tokens = tokens.new_empty(1, tokens.numel() * self.cp_mesh.size())
            if cu_seqlens is None and positions is not None:
                positions = cp_layout.gather(positions)
        cu_seqlens, mask = packed_metadata(
            metadata_tokens,
            positions,
            cu_seqlens,
            query_indices=cp_layout.local_indices if cp_layout is not None else None,
        )
        if cp_layout is not None:
            cp_context = build_fla_cp_context(cu_seqlens, cp_layout.group, self.config.conv_kernel_size, tokens.device)
        x = self.tok_embeddings(tokens).reshape(1, -1, self.config.dim)
        bank = x.new_empty(1, x.shape[1], 0, x.shape[2])
        for layer in self.layers.values():
            x, bank = layer(x, bank, cu_seqlens, mask, cp_layout, cp_context)
        x = attention_residual(x, bank, self.output_attn_res_norm, self.output_attn_res_proj)
        x = self.norm(x).reshape(*tokens.shape, -1)
        return x if self._skip_lm_head else self.lm_head(x)
