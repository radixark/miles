from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fla.modules.convolution import causal_conv1d
from fla.modules.fused_norm_gate import rms_norm_gated
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import fused_kda_gate
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module

from miles.backends.torchtitan_utils.models.glm5_next.packed_sequence import PackedSequence, gather_tokens_no_grad
from miles.kernels.attention.dsa import sparse_attention
from miles.kernels.attention.dsa.kpool import build_pooled_keys, pool_boundaries
from miles_plugins.models.glm5_next.ops.kpool_indexer import kpool_select_topk

_SPARSE_MLA_TAIL_DIM = 64


class Glm5NextRMSNorm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.dim))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = F.rms_norm(x.float(), (x.shape[-1],), weight=self.weight.float(), eps=self.eps)
        return normed.to(x.dtype)


class Glm5NextLayerNorm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.dim))
        self.bias = nn.Parameter(torch.zeros(config.dim))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), (x.shape[-1],), self.weight.float(), self.bias.float(), self.eps)


def _clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


class ClampedFeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        swiglu_limit: float

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(_clamped_swiglu(self.w1(x), self.w3(x), self.swiglu_limit))


class ClampedGroupedExperts(GroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        swiglu_limit: float

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x_RD: torch.Tensor, num_tokens_per_expert_E: torch.Tensor) -> torch.Tensor:
        w1_EFD, w2_EDF, w3_EFD = (
            w.to_local() if isinstance(w, DTensor) else w for w in (self.w1_EFD, self.w2_EDF, self.w3_EFD)
        )
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        x_RD_bf16 = x_RD.bfloat16()
        gate_RF = self._grouped_mm(A=x_RD_bf16, B_t=w1_EFD.bfloat16().transpose(-2, -1), offs=offsets_E)
        up_RF = self._grouped_mm(A=x_RD_bf16, B_t=w3_EFD.bfloat16().transpose(-2, -1), offs=offsets_E)
        h_RF = _clamped_swiglu(gate_RF, up_RF, self.swiglu_limit)
        return self._grouped_mm(A=h_RF, B_t=w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E).type_as(x_RD)


def _sinkhorn(logits: torch.Tensor, num_iterations: int, eps: float) -> torch.Tensor:
    matrix = logits.softmax(dim=-1) + eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(num_iterations - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


class HyperConnection(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_streams: int
        sinkhorn_iterations: int
        eps: float
        norm_eps: float

    def __init__(self, config: Config):
        super().__init__()
        n = config.num_streams
        self.num_streams = n
        self.sinkhorn_iterations = config.sinkhorn_iterations
        self.eps = config.eps
        self.norm_eps = config.norm_eps
        self.fn = nn.Parameter(torch.empty(n * n + 2 * n, n * config.dim))
        self.base = nn.Parameter(torch.zeros(n * n + 2 * n))
        self.scale = nn.Parameter(torch.zeros(3))

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fn)
        nn.init.zeros_(self.base)
        nn.init.zeros_(self.scale)

    def forward(self, x_BLND: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.num_streams
        streams = x_BLND.float()
        flat = streams.flatten(-2)
        rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
        alpha = torch.cat(
            [self.scale[0].expand(n), self.scale[1].expand(n), self.scale[2].expand(n * n)],
        )
        logits = rms * (flat @ self.fn.float().t()) * alpha + self.base.float()
        h_pre = logits[..., :n].sigmoid() + self.eps
        h_post = logits[..., n : 2 * n].sigmoid() * 2
        h_res = _sinkhorn(logits[..., 2 * n :].unflatten(-1, (n, n)), self.sinkhorn_iterations, self.eps)
        aggregated = (streams * h_pre.unsqueeze(-1)).sum(dim=-2).to(x_BLND.dtype)
        return aggregated, h_post, h_res


def hc_post(
    x_BLD: torch.Tensor, residual_BLND: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor
) -> torch.Tensor:
    mixed = torch.einsum("blij,blic->bljc", h_res, residual_BLND.float())
    return (mixed + h_post.unsqueeze(-1) * x_BLD.float().unsqueeze(-2)).to(residual_BLND.dtype)


class ShortConv(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        channels: int
        kernel_size: int

    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.channels, 1, config.kernel_size))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, cu_seqlens_cpu, cp_context) -> torch.Tensor:
        out, _ = causal_conv1d(
            x=x,
            weight=self.weight.squeeze(1),
            activation="silu",
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
        )
        return out


class KDAGate(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        lower_bound: float

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.lower_bound = config.lower_bound
        self.A_log = nn.Parameter(torch.zeros(config.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(config.num_heads * config.head_dim))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.A_log)
        nn.init.zeros_(self.dt_bias)

    def forward(self, forget: torch.Tensor) -> torch.Tensor:
        return fused_kda_gate(
            forget.unflatten(-1, (self.num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            lower_bound=self.lower_bound,
        )


class GatedRMSNorm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.dim))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return rms_norm_gated(x, gate, self.weight, None, activation="sigmoid", eps=self.eps)


def _cu_seqlens(masks: VarlenMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    # fla caches varlen helpers by tensor identity; a fresh tensor keeps forward and AC recompute aligned
    cu_seqlens = masks.cu_seq_q.clone()
    cu_seqlens_cpu = torch.tensor(masks.cu_seq_q_host, dtype=cu_seqlens.dtype)
    return cu_seqlens, cu_seqlens_cpu


class KimiDeltaAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        q_conv1d: ShortConv.Config
        k_conv1d: ShortConv.Config
        v_conv1d: ShortConv.Config
        b_proj: Linear.Config
        f_a_proj: Linear.Config
        f_b_proj: Linear.Config
        g_a_proj: Linear.Config
        g_b_proj: Linear.Config
        gate: KDAGate.Config
        o_norm: GatedRMSNorm.Config
        o_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_conv1d",
            "k_conv1d",
            "v_conv1d",
            "b_proj",
            "f_a_proj",
            "f_b_proj",
            "g_a_proj",
            "g_b_proj",
            "gate",
            "o_norm",
            "o_proj",
        ):
            setattr(self, name, getattr(config, name).build())

    def forward(self, x_BLD: torch.Tensor, sequence: PackedSequence) -> torch.Tensor:
        if sequence.cp_layout is None:
            cu_seqlens, cu_seqlens_cpu = _cu_seqlens(sequence.masks)
            return self._forward_shard(x_BLD, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, cp_context=None)
        cp_context = sequence.kda_cp_context
        contiguous = sequence.to_contiguous(x_BLD.squeeze(0)).unsqueeze(0)
        out = self._forward_shard(
            contiguous, cu_seqlens=cp_context.cu_seqlens, cu_seqlens_cpu=None, cp_context=cp_context
        )
        return sequence.from_contiguous(out.squeeze(0)).unsqueeze(0)

    def _forward_shard(self, x_BLD: torch.Tensor, *, cu_seqlens, cu_seqlens_cpu, cp_context) -> torch.Tensor:
        heads = (self.num_heads, self.head_dim)
        conv_args = (cu_seqlens, cu_seqlens_cpu, cp_context)
        q = self.q_conv1d(self.q_proj(x_BLD), *conv_args).unflatten(-1, heads)
        k = self.k_conv1d(self.k_proj(x_BLD), *conv_args).unflatten(-1, heads)
        v = self.v_conv1d(self.v_proj(x_BLD), *conv_args).unflatten(-1, heads)
        beta = torch.sigmoid(self.b_proj(x_BLD).float())
        g = self.gate(self.f_b_proj(self.f_a_proj(x_BLD)))
        out, _ = chunk_kda(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
        )
        norm_gate = self.g_b_proj(self.g_a_proj(x_BLD))
        out = self.o_norm(out.reshape(-1, self.head_dim), norm_gate.reshape(-1, self.head_dim))
        return self.o_proj(out.reshape(*x_BLD.shape[:2], -1))


class KpoolIndexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        head_dim: int
        topk: int
        kpool: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: Glm5NextLayerNorm.Config
        weights_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.topk = config.topk
        self.kpool = config.kpool
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(config.head_dim, config.dim))
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(config.kpool, config.head_dim))
        self.head_weight_scale = (config.num_heads**-0.5) * (config.head_dim**-0.5)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.index_kpool_compress_gate)
        nn.init.zeros_(self.index_kpool_compress_ape)

    @torch.no_grad()
    def forward(self, x_TD: torch.Tensor, q_lora_TR: torch.Tensor, sequence: PackedSequence) -> torch.Tensor:
        cu_seqlens = sequence.masks.cu_seq_q
        x_TD = x_TD.detach()
        index_q = self.wq_b(q_lora_TR.detach()).unflatten(-1, (self.num_heads, self.head_dim))
        index_k = gather_tokens_no_grad(self.k_norm(self.wk(x_TD)).bfloat16(), sequence)
        gate_score = gather_tokens_no_grad(F.linear(x_TD, self.index_kpool_compress_gate.to(x_TD.dtype)), sequence)
        head_weights = F.linear(x_TD.float(), self.weights_proj.weight.float()) * self.head_weight_scale
        pooled_k = build_pooled_keys(index_k, gate_score, self.index_kpool_compress_ape, cu_seqlens, self.kpool)
        return kpool_select_topk(
            index_q=index_q,
            pooled_k=pooled_k,
            head_weights=head_weights,
            cu_seqlens=cu_seqlens,
            pool_cu_seqlens=pool_boundaries(cu_seqlens, self.kpool),
            index_topk=self.topk,
            kpool=self.kpool,
            query_token_ids=sequence.query_token_ids,
        )


class DSAAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        kv_lora_rank: int
        qk_head_dim: int
        v_head_dim: int
        wq_a: Linear.Config
        q_norm: Glm5NextRMSNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: Glm5NextRMSNorm.Config
        wkv_b: Linear.Config
        o_proj: Linear.Config
        indexer: KpoolIndexer.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = config.qk_head_dim**-0.5
        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.o_proj = config.o_proj.build()
        self.indexer = config.indexer.build()

    def forward(self, x_BLD: torch.Tensor, sequence: PackedSequence) -> torch.Tensor:
        x_TD = x_BLD.squeeze(0)
        q_lora = self.q_norm(self.wq_a(x_TD))
        q = self.wq_b(q_lora).unflatten(-1, (self.n_heads, self.qk_head_dim))
        latent_kv = sequence.gather_tokens(self.kv_norm(self.wkv_a(x_TD)))

        w_kc, w_vc = self.wkv_b.weight.unflatten(0, (self.n_heads, -1)).split(
            [self.qk_head_dim, self.v_head_dim], dim=1
        )
        query = torch.einsum("thd,hdm->thm", q, w_kc.to(q.dtype))

        topk_indices = self.indexer(x_TD, q_lora, sequence)
        out = sparse_attention(
            F.pad(query, (0, _SPARSE_MLA_TAIL_DIM)).unsqueeze(0),
            F.pad(latent_kv, (0, _SPARSE_MLA_TAIL_DIM)).unsqueeze(1).unsqueeze(0),
            topk_indices.unsqueeze(0),
            self.softmax_scale,
            d_v=self.kv_lora_rank,
        ).squeeze(0)
        out = torch.einsum("thm,hdm->thd", out, w_vc.to(out.dtype))
        return self.o_proj(out.flatten(-2)).unsqueeze(0)
