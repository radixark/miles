"""CPU stand-ins for the fla kernels GLM-5.3-Flash's KDA calls, and helpers that copy a replicated ``Glm5NextKDA``
into one TP rank's head-sharded ``Glm5NextKDA``.

The stand-ins are plain-torch, per-sequence (they honor ``cu_seqlens``) and per-head, so the replicated module and
the head-sharded one run the same math on CPU. They are not fla's numerics: the tests built on them check the
sharding (which rows, columns and gradients each rank holds), not the kernels.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_CONFIG = SimpleNamespace(
    hidden_size=32,
    linear_num_heads=4,
    linear_head_dim=8,
    linear_conv_kernel_dim=4,
    linear_lower_bound=-5.0,
    rms_norm_eps=1e-5,
)


def _segments(cu_seqlens: torch.Tensor):
    bounds = cu_seqlens.tolist()
    return list(zip(bounds[:-1], bounds[1:], strict=True))


class ShortConvolution(nn.Conv1d):
    """Depthwise causal conv + SiLU over ``[b, s, C]``, restarted at every packed-sequence boundary."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", **kwargs):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias, padding=kernel_size - 1)

    def forward(self, x, cu_seqlens=None, cp_context=None, **kwargs):
        assert cp_context is None
        cu_seqlens = torch.tensor([0, x.shape[1]]) if cu_seqlens is None else cu_seqlens
        outputs = []
        for start, end in _segments(cu_seqlens):
            segment = x[:, start:end].transpose(1, 2)
            y = F.conv1d(segment, self.weight, None, padding=self.kernel_size[0] - 1, groups=self.groups)
            outputs.append(F.silu(y[..., : end - start].transpose(1, 2)))
        return torch.cat(outputs, dim=1), None


def fused_kda_gate(g, A_log, dt_bias=None, lower_bound=None, output_dtype=torch.float32):
    heads, head_dim = g.shape[-2:]
    return (-A_log.float().exp()[:, None] * F.softplus(g.float() + dt_bias.view(heads, head_dim))).to(output_dtype)


def chunk_kda(q, k, v, g, beta, cu_seqlens=None, use_qk_l2norm_in_kernel=False, cp_context=None, **kwargs):
    """Per-head delta rule with per-channel decay. q/k/v/g ``[b, s, H, D]``, beta ``[b, s, H]``."""
    assert cp_context is None
    if use_qk_l2norm_in_kernel:
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    batch, seq_len, heads, head_dim = q.shape
    cu_seqlens = torch.tensor([0, seq_len]) if cu_seqlens is None else cu_seqlens
    out = []
    for start, end in _segments(cu_seqlens):
        head_outputs = []
        for h in range(heads):
            state = q.new_zeros(batch, head_dim, head_dim, dtype=torch.float32)
            steps = []
            for t in range(start, end):
                state = state * g[:, t, h].float().exp()[:, :, None]
                k_t, v_t, q_t = k[:, t, h].float(), v[:, t, h].float(), q[:, t, h].float()
                predicted = torch.einsum("bij,bi->bj", state, k_t)
                state = state + beta[:, t, h].float()[:, None, None] * torch.einsum("bi,bj->bij", k_t, v_t - predicted)
                steps.append(torch.einsum("bij,bi->bj", state, q_t))
            head_outputs.append(torch.stack(steps, dim=1))
        out.append(torch.stack(head_outputs, dim=2))
    return torch.cat(out, dim=1).to(v.dtype), None


def rms_norm_gated(x, g, weight, bias, activation="sigmoid", eps=1e-6, **kwargs):
    assert activation == "sigmoid"
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight
    return (y * torch.sigmoid(g.float())).to(x.dtype)


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, activation="sigmoid", **kwargs):
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    def forward(self, x, g):
        return rms_norm_gated(x, g, self.weight, self.bias, self.activation, eps=self.eps)


CPU_KERNELS = {
    "ShortConvolution": ShortConvolution,
    "FusedRMSNormGated": FusedRMSNormGated,
    "chunk_kda": chunk_kda,
    "fused_kda_gate": fused_kda_gate,
    "rms_norm_gated": rms_norm_gated,
}


def install_cpu_kernels(kda_module) -> None:
    """Point ``miles_plugins.models.glm5_next.kda`` at the CPU stand-ins, and the attention's HF config at
    ``TEXT_CONFIG`` (per process)."""
    from miles_plugins.models import hf_attention

    for name, kernel in CPU_KERNELS.items():
        setattr(kda_module, name, kernel)
    hf_attention.load_hf_config = lambda path: TEXT_CONFIG


def patch_cpu_kernels(monkeypatch, kda_module) -> None:
    """``install_cpu_kernels`` in this process, undone after the test."""
    from miles_plugins.models import hf_attention

    for name, kernel in CPU_KERNELS.items():
        monkeypatch.setattr(kda_module, name, kernel)
    monkeypatch.setattr(hf_attention, "load_hf_config", lambda path: TEXT_CONFIG)


def build_attention(kda_module, tp_group, *, sequence_parallel: bool = False, kda_tp_sharding: bool = True):
    """One TP rank's ``Glm5NextKDAAttention`` (CP size 1) on ``TEXT_CONFIG``."""
    return kda_module.Glm5NextKDAAttention(
        SimpleNamespace(hf_checkpoint="unused"),
        SimpleNamespace(sequence_parallel=sequence_parallel),
        layer_number=1,
        pg_collection=SimpleNamespace(tp=tp_group, cp=SimpleNamespace(size=lambda: 1)),
        kda_tp_sharding=kda_tp_sharding,
    )


def build_replicated(kda_module, *, scale: float = 0.3, tp_group=None):
    """A ``Glm5NextKDA`` (replicated unless ``tp_group`` has more than one rank) with random weights (scaled so the
    recurrence stays well conditioned)."""
    tc = TEXT_CONFIG
    full = kda_module.Glm5NextKDA(
        tc.hidden_size,
        tc.linear_num_heads,
        tc.linear_head_dim,
        tc.linear_conv_kernel_dim,
        tc.linear_lower_bound,
        tc.rms_norm_eps,
        tp_group=tp_group,
    )
    with torch.no_grad():
        for param in full.parameters():
            param.copy_(torch.randn_like(param) * scale)
    return full


def head_rows(sharded, tp_rank: int) -> slice:
    """This rank's rows of a ``[heads * head_dim, ...]`` replicated weight."""
    return slice(tp_rank * sharded.projection_size, (tp_rank + 1) * sharded.projection_size)


def head_slice(sharded, tp_rank: int) -> slice:
    """This rank's entries of a per-head ``[heads, ...]`` replicated weight."""
    return slice(tp_rank * sharded.num_heads, (tp_rank + 1) * sharded.num_heads)


def load_shard(full, sharded, tp_rank: int) -> None:
    """Copy this rank's part of the replicated module's weights into the head-sharded module."""
    rows, heads = head_rows(sharded, tp_rank), head_slice(sharded, tp_rank)
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "f_b_proj", "g_b_proj"):
            getattr(sharded, name).weight.copy_(getattr(full, name).weight[rows])
        sharded.b_proj.weight.copy_(full.b_proj.weight[heads])
        sharded.f_a_proj.weight.copy_(full.f_a_proj.weight)
        sharded.g_a_proj.weight.copy_(full.g_a_proj.weight)
        sharded.A_log.copy_(full.A_log[heads])
        sharded.dt_bias.copy_(full.dt_bias[rows])
        sharded.o_norm.weight.copy_(full.o_norm.weight)
        sharded.o_proj.weight.copy_(full.o_proj.weight[:, rows])
        for j, name in enumerate(("q_conv1d", "k_conv1d", "v_conv1d")):
            offset = j * full.projection_size
            getattr(sharded, name).weight.copy_(full.conv1d.weight[offset + rows.start : offset + rows.stop])
