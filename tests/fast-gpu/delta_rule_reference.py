"""Replicated, HF-layout reference for the head-sharded KDA layer, and helpers to load their
weights into a :class:`LinearAttentionLayer`."""

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda
from megatron.core.process_groups_config import ProcessGroupCollection

from miles.backends.megatron_utils.megatron_to_hf.gdn_layout import qkv_flat_to_group_major
from miles.kernels.attention.delta_rule import DeltaRuleHeads, KimiDeltaRule
from miles_plugins.models.linear_attn import KimiDeltaAttention, LinearAttentionLayer

CONV = 4
EPS = 1e-6
KDA_LOWER_BOUND = -5.0


class ReplicatedKDA(nn.Module):
    """Pre-sharding KDA in the Kimi-K3 / GLM-5.3-flash HF layout."""

    def __init__(self, hidden: int, heads: DeltaRuleHeads, dtype):
        super().__init__()
        self.heads = heads
        size, d = heads.value_dim, heads.head_v_dim
        self.q_proj = nn.Linear(hidden, size, bias=False)
        self.k_proj = nn.Linear(hidden, size, bias=False)
        self.v_proj = nn.Linear(hidden, size, bias=False)
        self.q_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.f_a_proj = nn.Linear(hidden, d, bias=False)
        self.f_b_proj = nn.Linear(d, size, bias=False)
        self.b_proj = nn.Linear(hidden, heads.num_v_heads, bias=False)
        self.g_proj = nn.Linear(hidden, size, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads.num_v_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.rand(size))
        self.o_norm = FusedRMSNormGated(d, eps=EPS, activation="sigmoid")
        self.o_proj = nn.Linear(size, hidden, bias=False)
        self.to(dtype=dtype)
        self.A_log.data = self.A_log.data.float()
        self.dt_bias.data = self.dt_bias.data.float()

    def hf_conv(self):
        return torch.cat([self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight])

    def forward(self, x, cu_seqlens):
        bsz, seq_len, _ = x.shape
        h, d = self.heads.num_v_heads, self.heads.head_v_dim
        q, _ = self.q_conv1d(self.q_proj(x), cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(self.k_proj(x), cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(self.v_proj(x), cu_seqlens=cu_seqlens)
        out, _ = chunk_kda(
            q=q.view(bsz, seq_len, h, d),
            k=k.view(bsz, seq_len, h, d),
            v=v.view(bsz, seq_len, h, d),
            g=self.f_b_proj(self.f_a_proj(x)).view(bsz, seq_len, h, d),
            beta=self.b_proj(x).float().sigmoid(),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=KDA_LOWER_BOUND,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
        out = self.o_norm(out.reshape(-1, d), self.g_proj(x).reshape(-1, d))
        return self.o_proj(out.reshape(bsz, seq_len, -1))


def sharded_projections(ref, grad: bool = False) -> dict[str, torch.Tensor]:
    """Full (all-head) Megatron tensor of every head-sharded projection, from the reference's HF-layout
    weights or their gradients: what the bridges load."""

    def w(module):
        return module.weight.grad if grad else module.weight

    qkv = torch.cat([w(ref.q_proj), w(ref.k_proj), w(ref.v_proj)])
    return {
        "in_proj_qkv": qkv_flat_to_group_major(qkv, ref.heads),
        "g_proj": w(ref.g_proj),
        "b_proj": w(ref.b_proj),
        "f_b_proj": w(ref.f_b_proj),
    }


def conv_of(ref, grad: bool = False) -> torch.Tensor:
    w = torch.cat([c.weight.grad if grad else c.weight for c in (ref.q_conv1d, ref.k_conv1d, ref.v_conv1d)])
    return qkv_flat_to_group_major(w, ref.heads)


def shard(full: torch.Tensor, dim: int, group) -> torch.Tensor:
    return full.chunk(group.size(), dim=dim)[group.rank()].contiguous()


def gather(local: torch.Tensor, dim: int, group) -> torch.Tensor:
    parts = [torch.empty_like(local) for _ in range(group.size())]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def build_layer(ref, config, allgather_cp: bool = True) -> LinearAttentionLayer:
    """A head-sharded layer holding this TP rank's shard of ``ref``'s weights, identity input norm."""
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    tp = pg.tp
    core = KimiDeltaAttention(config, ref.heads, KimiDeltaRule(KDA_LOWER_BOUND), CONV, EPS, tp)
    with torch.no_grad():
        for name, full in sharded_projections(ref).items():
            getattr(core, name).weight.copy_(shard(full, 0, tp))
        core.conv1d.weight.copy_(shard(conv_of(ref), 0, tp))
        core.A_log.copy_(shard(ref.A_log, 0, tp))
        core.dt_bias.copy_(shard(ref.dt_bias, 0, tp))
        core.norm.weight.copy_(ref.o_norm.weight)
        core.out_proj.weight.copy_(shard(ref.o_proj.weight, 1, tp))
        core.f_a_proj.weight.copy_(ref.f_a_proj.weight)
    return LinearAttentionLayer(config, core, nn.Identity(), pg, allgather_cp=allgather_cp)


def packed(cu_seqlens):
    return SimpleNamespace(cu_seqlens_q=cu_seqlens)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()
