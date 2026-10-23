"""KDA and gated NoPE MLA, with document boundaries preserved during training."""

import torch
import torch.nn.functional as F
from fla.ops.kda import chunk_kda
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.protocols.module import Module

from miles.backends.torchtitan_utils.models.kimi_k3.ops import (
    GatedRMSNorm,
    RMSNorm,
    ShortConv,
    linear,
    local_tensor,
    wrap_like,
)


class KDACore(Module):
    def __init__(self, width, head_dim, lower_bound):
        super().__init__()
        self.tp_rank = 0
        # HF stores head_dim slots even with fewer active heads; preserve all slots on export.
        self.A_log = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(width, dtype=torch.float32))
        self.lower_bound = lower_bound

    def reset_parameters(self):
        nn.init.zeros_(self.A_log)
        nn.init.zeros_(self.dt_bias)

    def forward(self, q, k, v, gate, beta, cu_seqlens, cp_context=None):
        distributed_q = q if isinstance(q, DTensor) else None
        q, k, v, gate, beta = [local_tensor(value) for value in (q, k, v, gate, beta)]
        heads = q.shape[-2]
        first_head = self.tp_rank * heads
        output, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta.float().sigmoid(),
            A_log=local_tensor(self.A_log, partial_grad=distributed_q is not None)[first_head : first_head + heads],
            dt_bias=local_tensor(self.dt_bias),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=self.lower_bound,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )
        return wrap_like(output, distributed_q)


class DeltaAttention(Module):
    def __init__(self, config):
        super().__init__()
        dim, heads, head_dim = config.dim, config.n_heads, config.head_dim
        self.heads, self.head_dim = heads, head_dim
        width = heads * head_dim
        self.q_proj, self.k_proj, self.v_proj = [linear(dim, width) for _ in range(3)]
        self.q_conv1d, self.k_conv1d, self.v_conv1d = [
            ShortConv(hidden_size=width, kernel_size=config.conv_kernel_size, activation="silu") for _ in range(3)
        ]
        self.core = KDACore(width, head_dim, config.gate_lower_bound)
        self.f_a_proj, self.f_b_proj = linear(dim, head_dim), linear(head_dim, width)
        self.b_proj, self.g_proj = linear(dim, heads), linear(dim, width)
        self.o_norm, self.o_proj = GatedRMSNorm(head_dim, eps=config.norm_eps), linear(width, dim)

    def forward(self, x, cu_seqlens, mask, cp_layout=None, cp_context=None):
        if cp_layout is not None:
            x = cp_layout.to_contiguous(x)
            cu_seqlens = cp_context.cu_seqlens
        q, k, v = [
            conv(proj(x), cu_seqlens=cu_seqlens, output_final_state=False, cp_context=cp_context)[0].unflatten(
                -1, (self.heads, self.head_dim)
            )
            for proj, conv in (
                (self.q_proj, self.q_conv1d),
                (self.k_proj, self.k_conv1d),
                (self.v_proj, self.v_conv1d),
            )
        ]
        gate = self.f_b_proj(self.f_a_proj(x)).unflatten(-1, (self.heads, self.head_dim))
        output = self.core(q, k, v, gate, self.b_proj(x), cu_seqlens, cp_context)
        output = self.o_norm(output, self.g_proj(x).unflatten(-1, (self.heads, self.head_dim)))
        output = output.flatten(-2)
        if cp_layout is not None:
            output = cp_layout.from_contiguous(output)
        return self.o_proj(output)


class MLAAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.mla_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.nope_head_dim, self.v_head_dim = config.qk_nope_head_dim, config.v_head_dim
        self.q_a_proj = linear(config.dim, config.q_lora_rank)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.norm_eps)
        self.q_b_proj = linear(config.q_lora_rank, self.heads * self.qk_head_dim)
        self.kv_a_proj_with_mqa = linear(config.dim, config.kv_lora_rank + config.qk_rope_head_dim)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.norm_eps)
        self.kv_b_proj = linear(config.kv_lora_rank, self.heads * (self.nope_head_dim + self.v_head_dim))
        self.g_proj = linear(config.dim, self.heads * self.v_head_dim)
        self.o_proj = linear(self.heads * self.v_head_dim, config.dim)

    def forward(self, x, cu_seqlens, mask, cp_layout=None, cp_context=None):
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).unflatten(-1, (self.heads, self.qk_head_dim))
        latent, key_shared = self.kv_a_proj_with_mqa(x).split(
            (self.kv_lora_rank, self.qk_head_dim - self.nope_head_dim), dim=-1
        )
        key, value = (
            self.kv_b_proj(self.kv_a_layernorm(latent))
            .unflatten(-1, (self.heads, self.nope_head_dim + self.v_head_dim))
            .split((self.nope_head_dim, self.v_head_dim), dim=-1)
        )
        key = torch.cat((key, key_shared.unsqueeze(-2).expand(*key.shape[:-1], key_shared.shape[-1])), dim=-1)
        distributed_q = q if isinstance(q, DTensor) else None
        q, key, value = [local_tensor(tensor) for tensor in (q, key, value)]
        if cp_layout is not None:
            key, value = cp_layout.gather(key), cp_layout.gather(value)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
            scale=self.qk_head_dim**-0.5,
        ).transpose(1, 2)
        output = wrap_like(output, distributed_q).flatten(-2)
        return self.o_proj(output * self.g_proj(x).sigmoid())
