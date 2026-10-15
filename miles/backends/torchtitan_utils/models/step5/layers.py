"""Step5 blocks for torchtitan, numerically following the Megatron model in ``miles_plugins/models/step5``.

Norm weights are FP32 and round their output once to BF16; q and k rotate in FP32 with their attention
type's RoPE; each head's output is scaled by a sigmoid gate; every SwiGLU clamps to 7.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn.attention.varlen import varlen_attn
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module

from miles.backends.torchtitan_utils.models.step5.packed_sequence import PackedSequence
from miles.kernels.activation.swiglu_fp32 import swiglu_fp32
from miles.kernels.attention.region_sparse import sparse_region_attention
from miles_plugins.models.step5.indexer_ops import INDEXER_HEAD_DIM
from miles_plugins.models.step5.layer_math import RopeGeometry, zero_centered_rms_norm
from miles_plugins.models.step5.region_selector import IndexerWeights, select_regions


class Step5RMSNorm(Module):
    """Zero-centred RMSNorm over an FP32 weight, rounding once to BF16."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.zeros(config.dim))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return zero_centered_rms_norm(x, self.weight, self.eps).to(torch.bfloat16)


class Step5FinalRMSNorm(Module):
    """The final norm narrows the FP32 residual to BF16 before normalizing, as the reference does.

    The norm is a child so that FSDP can keep its weight in FP32 apart from the ``lm_head`` group.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rms: Step5RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.rms = config.rms.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms(x.to(torch.bfloat16))


def _rotate(x_TNH: torch.Tensor, positions_T: torch.Tensor, rope: RopeGeometry) -> torch.Tensor:
    """NeoX RoPE on the first ``rotary_percent`` of each head's channels, in FP32, rounded once."""
    head_dim = x_TNH.shape[-1]
    rotary_dim = round(head_dim * rope.rotary_percent)
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x_TNH.device) / rotary_dim
    angles = positions_T.float()[:, None] * (1.0 / rope.theta**exponents)
    cos, sin = (torch.cat([f(angles)] * 2, dim=-1)[:, None] for f in (torch.cos, torch.sin))
    rotated, passed = x_TNH.float().split([rotary_dim, head_dim - rotary_dim], dim=-1)
    first, second = rotated.chunk(2, dim=-1)
    rotated = rotated * cos + torch.cat([-second, first], dim=-1) * sin
    return torch.cat([rotated, passed], dim=-1).to(x_TNH.dtype)


class Step5Indexer(Module):
    """The frozen DSA indexer of one full-attention layer; ``ssmax_s`` rides along for the export."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        num_attention_heads: int
        eps: float

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        projections = config.num_heads * INDEXER_HEAD_DIM
        self.q_weight = _frozen(projections, config.dim)
        self.k_weight = _frozen(INDEXER_HEAD_DIM, config.dim)
        self.z_weight = _frozen(INDEXER_HEAD_DIM, config.dim)
        self.w_weight = _frozen(config.num_heads, config.dim)
        self.q_norm_weight = _frozen(INDEXER_HEAD_DIM)
        self.k_norm_weight = _frozen(INDEXER_HEAD_DIM)
        self.k_norm_bias = _frozen(INDEXER_HEAD_DIM)
        self.ssmax_s = _frozen(config.num_attention_heads)

    def reset_parameters(self) -> None:
        for param in self.parameters(recurse=False):
            nn.init.zeros_(param)

    @torch.no_grad()
    def forward(self, x_TD: torch.Tensor, sequence: PackedSequence) -> torch.Tensor:
        """int32 ``[T, groups, 513]`` region ids for the whole sequence, from this rank's attention input."""
        weights = IndexerWeights(
            q_weight=self.q_weight,
            k_weight=self.k_weight,
            z_weight=self.z_weight,
            w_weight=self.w_weight,
            q_norm_weight=self.q_norm_weight,
            k_norm_weight=self.k_norm_weight,
            k_norm_bias=self.k_norm_bias,
        )
        tokens = sequence.gather_tokens_detached(x_TD.detach())
        return select_regions(tokens, sequence.cu_seqlens, weights, eps=self.eps).region_ids


class Step5Attention(Module):
    """Gated GQA: per-head q/k norms, the layer's RoPE, and ``o_proj(attn * sigmoid(g_proj(x)))``.

    Subclasses differ only in which keys a query attends (``_attend``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        n_kv_heads: int
        head_dim: int
        rope: RopeGeometry
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        g_proj: Linear.Config
        o_proj: Linear.Config
        q_norm: Step5RMSNorm.Config
        k_norm: Step5RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rope = config.rope
        self.softmax_scale = config.head_dim**-0.5
        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.g_proj = config.g_proj.build()
        self.o_proj = config.o_proj.build()
        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

    def forward(self, x_BLD: torch.Tensor, sequence: PackedSequence, positions: torch.Tensor) -> torch.Tensor:
        x_TD = x_BLD.squeeze(0)
        positions_T = positions.squeeze(0)
        query = self.q_norm(self.q_proj(x_TD).unflatten(-1, (self.n_heads, self.head_dim)))
        key = self.k_norm(self.k_proj(x_TD).unflatten(-1, (self.n_kv_heads, self.head_dim)))
        value = self.v_proj(x_TD).unflatten(-1, (self.n_kv_heads, self.head_dim))
        query, key = _rotate(query, positions_T, self.rope), _rotate(key, positions_T, self.rope)
        out = self._attend(x_TD, query, key, value, sequence)
        # the reference rounds the gate to the activation dtype before the product
        out = out * torch.sigmoid(self.g_proj(x_TD).float())[..., None].to(out.dtype)
        return self.o_proj(out.flatten(-2)).unsqueeze(0)

    def _attend(self, x_TD, query, key, value, sequence: PackedSequence) -> torch.Tensor:
        raise NotImplementedError


class Step5SlidingWindowAttention(Step5Attention):
    """Each query attends the ``window`` keys ending at itself within its document."""

    @dataclass(kw_only=True, slots=True)
    class Config(Step5Attention.Config):
        window: int

    def __init__(self, config: Config):
        super().__init__(config)
        # torch's default flash kernel: the FA4 one torchtitan activates on Blackwell rejects 192-wide heads
        self.window = config.window

    def _attend(self, x_TD, query, key, value, sequence: PackedSequence) -> torch.Tensor:
        spans = sequence.window
        return varlen_attn(
            query,
            sequence.window_keys(key),
            sequence.window_keys(value),
            spans.cu_seq_q,
            spans.cu_seq_k,
            spans.max_q,
            spans.max_k,
            scale=self.softmax_scale,
            window_size=(self.window - 1, 0),
            enable_gqa=True,
        )


class Step5SparseAttention(Step5Attention):
    """Each query attends the regions the frozen indexer selects for its KV group."""

    @dataclass(kw_only=True, slots=True)
    class Config(Step5Attention.Config):
        indexer: Step5Indexer.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.indexer = config.indexer.build()

    def _attend(self, x_TD, query, key, value, sequence: PackedSequence) -> torch.Tensor:
        region_ids = self.indexer(x_TD, sequence)
        out = sparse_region_attention(
            sequence.gather_tokens(query),
            sequence.gather_tokens(key),
            sequence.gather_tokens(value),
            region_ids,
            sequence.cu_seqlens,
            self.softmax_scale,
        )
        return sequence.local_rows(out)


class Step5FeedForward(FeedForward):
    """``w2(min(silu(w1 x), limit) * clamp(w3 x, -limit, limit))`` in FP32, rounded once."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        swiglu_limit: float

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.w1(x), self.w3(x)], dim=-1)
        return self.w2(swiglu_fp32(gate_up, limit=self.swiglu_limit))


class Step5GroupedExperts(GroupedExperts):
    """The routed experts with Step5's clamped FP32 SwiGLU."""

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
        h_RF = swiglu_fp32(torch.cat([gate_RF, up_RF], dim=-1), limit=self.swiglu_limit)
        return self._grouped_mm(A=h_RF, B_t=w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E).type_as(x_RD)


def _frozen(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.zeros(shape), requires_grad=False)
