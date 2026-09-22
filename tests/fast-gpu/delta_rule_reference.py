"""Replicated, HF-layout references for the head-sharded delta-rule layers, and helpers to load their
weights into a :class:`LinearAttentionLayer`."""

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda
from megatron.core.process_groups_config import ProcessGroupCollection

from miles_plugins.models.layers.delta_rule_attention import (
    GatedDeltaRule,
    KimiDeltaAttention,
    KimiDeltaRule,
    LinearAttentionLayer,
)
from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads, qkv_flat_to_group_major
from miles_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet
from miles_plugins.models.qwen3_next import Qwen3NextGatedDeltaNet

CONV = 4
EPS = 1e-6
KDA_LOWER_BOUND = -5.0


class ReplicatedGDN(nn.Module):
    """Pre-sharding GDN: every rank holds all heads. ``family`` picks the HF projection layout."""

    def __init__(self, family: str, hidden: int, heads: DeltaRuleHeads, dtype):
        super().__init__()
        self.family, self.heads = family, heads
        h = heads
        if family == "qwen3_5":
            self.in_proj_qkv = nn.Linear(hidden, h.qkv_dim, bias=False)
            self.in_proj_z = nn.Linear(hidden, h.value_dim, bias=False)
            self.in_proj_b = nn.Linear(hidden, h.num_v_heads, bias=False)
            self.in_proj_a = nn.Linear(hidden, h.num_v_heads, bias=False)
        else:
            self.in_proj_qkvz = nn.Linear(hidden, h.qkv_dim + h.value_dim, bias=False)
            self.in_proj_ba = nn.Linear(hidden, 2 * h.num_v_heads, bias=False)
        self.conv1d = ShortConvolution(hidden_size=h.qkv_dim, kernel_size=CONV, bias=False)
        self.dt_bias = nn.Parameter(torch.rand(h.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(h.num_v_heads).uniform_(1, 16)))
        self.norm = FusedRMSNormGated(h.head_v_dim, eps=EPS, activation="silu")
        self.out_proj = nn.Linear(h.value_dim, hidden, bias=False)
        self.to(dtype=dtype)
        self.A_log.data = self.A_log.data.float()

    def _split(self, x):
        h = self.heads
        if self.family == "qwen3_5":
            q, k, v = self.in_proj_qkv(x).split([h.key_dim, h.key_dim, h.value_dim], dim=-1)
            return q, k, v, self.in_proj_z(x), self.in_proj_b(x), self.in_proj_a(x)
        r, hv = h.v_per_k, h.head_v_dim
        grouped = self.in_proj_qkvz(x).view(*x.shape[:-1], h.num_k_heads, 2 * h.head_k_dim + 2 * r * hv)
        q, k, v, z = grouped.split([h.head_k_dim, h.head_k_dim, r * hv, r * hv], dim=-1)
        b, a = self.in_proj_ba(x).view(*x.shape[:-1], h.num_k_heads, 2 * r).split([r, r], dim=-1)
        return q.flatten(-2), k.flatten(-2), v.flatten(-2), z.flatten(-2), b.flatten(-2), a.flatten(-2)

    def forward(self, x, cu_seqlens):
        h = self.heads
        bsz, seq_len, _ = x.shape
        q, k, v, z, b, a = self._split(x)
        mixed, _ = self.conv1d(x=torch.cat([q, k, v], dim=-1), cu_seqlens=cu_seqlens)
        q, k, v = mixed.split([h.key_dim, h.key_dim, h.value_dim], dim=-1)
        q = q.reshape(bsz, seq_len, -1, h.head_k_dim).repeat_interleave(h.v_per_k, dim=2)
        k = k.reshape(bsz, seq_len, -1, h.head_k_dim).repeat_interleave(h.v_per_k, dim=2)
        v = v.reshape(bsz, seq_len, -1, h.head_v_dim)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        out, _ = chunk_gated_delta_rule(
            q, k, v, g=g, beta=b.sigmoid(), use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens
        )
        out = self.norm(out.reshape(-1, h.head_v_dim), z.reshape(-1, h.head_v_dim))
        return self.out_proj(out.reshape(bsz, seq_len, -1))


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

    if isinstance(ref, ReplicatedKDA):
        qkv = torch.cat([w(ref.q_proj), w(ref.k_proj), w(ref.v_proj)])
        return {
            "in_proj_qkv": qkv_flat_to_group_major(qkv, ref.heads),
            "g_proj": w(ref.g_proj),
            "b_proj": w(ref.b_proj),
            "f_b_proj": w(ref.f_b_proj),
        }
    if ref.family == "qwen3_5":
        return {
            "in_proj_qkv": qkv_flat_to_group_major(w(ref.in_proj_qkv), ref.heads),
            "in_proj_z": w(ref.in_proj_z),
            "in_proj_b": w(ref.in_proj_b),
            "in_proj_a": w(ref.in_proj_a),
        }
    return {"in_proj_qkvz": w(ref.in_proj_qkvz), "in_proj_ba": w(ref.in_proj_ba)}


def conv_of(ref, grad: bool = False) -> torch.Tensor:
    if isinstance(ref, ReplicatedKDA):
        w = torch.cat([c.weight.grad if grad else c.weight for c in (ref.q_conv1d, ref.k_conv1d, ref.v_conv1d)])
    else:
        w = ref.conv1d.weight.grad if grad else ref.conv1d.weight
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
    if isinstance(ref, ReplicatedKDA):
        core = KimiDeltaAttention(config, ref.heads, KimiDeltaRule(KDA_LOWER_BOUND), CONV, EPS, tp)
    else:
        core_cls = Qwen3_5GatedDeltaNet if ref.family == "qwen3_5" else Qwen3NextGatedDeltaNet
        core = core_cls(config, ref.heads, GatedDeltaRule("fla", "silu"), CONV, EPS, tp)
    with torch.no_grad():
        for name, full in sharded_projections(ref).items():
            getattr(core, name).weight.copy_(shard(full, 0, tp))
        core.conv1d.weight.copy_(shard(conv_of(ref), 0, tp))
        core.A_log.copy_(shard(ref.A_log, 0, tp))
        core.dt_bias.copy_(shard(ref.dt_bias, 0, tp))
        norm = ref.o_norm if isinstance(ref, ReplicatedKDA) else ref.norm
        core.norm.weight.copy_(norm.weight)
        out_proj = ref.o_proj if isinstance(ref, ReplicatedKDA) else ref.out_proj
        core.out_proj.weight.copy_(shard(out_proj.weight, 1, tp))
        if isinstance(ref, ReplicatedKDA):
            core.f_a_proj.weight.copy_(ref.f_a_proj.weight)
    return LinearAttentionLayer(config, core, nn.Identity(), pg, allgather_cp=allgather_cp)


def packed(cu_seqlens):
    return SimpleNamespace(cu_seqlens_q=cu_seqlens)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()
