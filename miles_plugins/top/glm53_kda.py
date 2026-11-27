"""GLM-5.3-Flash KDA layer under TOP (trainer TP1): every step delegated to the program serving runs.

Parameter names are those of ``miles_plugins.models.glm5_next.kda.Glm5NextKDAAttention`` (``kda.q_proj`` ...), so the
checkpoint bridge is unchanged. Serving (SGLang ``glm5_next.Glm5NextLinearAttention`` with BF16 weights) runs:

1. one merged projection over [q, k, v, b, f_a, g_a] (``fused_qkvbfg_a_proj``, BI GEMM);
2. one batched projection of [f_a, g_a] by [f_b, g_b] (``fused_fg_b_proj``, the BI bmm);
3. the short causal convolution over q|k|v (``causal_conv1d_fn``, SiLU, FP32 weights);
4. the KDA core (``kda_chunked``: the pinned chunk prefill; decode follows it);
5. the sigmoid-gated RMSNorm (``FusedRMSNormGated``) and the row projection ``o_proj``.

Each forward value is serving's kernel on the same inputs; each gradient is an ordinary PyTorch/FLA reference.
"""

import torch
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32

from miles_plugins.top.binding import canon_dump
from miles_plugins.top.kernels.delegated_linear import delegated_row_linear
from miles_plugins.top.kernels.kda import delegated_kda
from miles_plugins.top.kernels.surrogate import linear_surrogate, surrogate


def _merged_projection(x, weight):
    from sglang.srt.batch_invariant_ops import matmul_persistent

    return linear_surrogate(lambda x, w: matmul_persistent(x.contiguous(), w.t()), x, weight, name="kda_in_proj")


def _batched_projection(fg_a, weight, head_dim):
    """[f_b, g_b] of the stacked low-rank rows, as serving's ColumnParallelBatchedLinear runs it: bmm of
    fg_a.view(-1, 2, d).transpose(0, 1) [2, T, d] by weight [2, out, d] transposed."""
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_batch_invariant

    def visible(fg_a, w):
        return bmm_batch_invariant(fg_a.view(-1, 2, head_dim).transpose(0, 1), w.transpose(-1, -2))

    def reference(fg_a, w):
        return torch.bmm(fg_a.view(-1, 2, head_dim).transpose(0, 1), w.transpose(-1, -2))

    return surrogate(visible, reference, fg_a, weight, name="kda_fg_b_proj")


def _conv(x, weight, cu_seqlens, lengths):
    """Serving's packed causal conv (SiLU) over x [T, C] with weight [C, 4]; FLA's varlen conv as the gradient."""
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    def visible(x, w):
        docs = len(lengths)
        # The KDA pool keeps conv windows as [kernel - 1, dim]; serving passes the transposed view.
        state = x.new_zeros((docs, 3, x.shape[1])).transpose(-1, -2)
        return causal_conv1d_fn(
            x.transpose(0, 1),
            w.float(),
            None,
            activation="silu",
            conv_states=state,
            has_initial_state=torch.zeros(docs, dtype=torch.bool, device=x.device),
            cache_indices=torch.arange(docs, dtype=torch.int32, device=x.device),
            query_start_loc=cu_seqlens,
            seq_lens_cpu=list(lengths),
        ).transpose(0, 1)

    def reference(x, w):
        from fla.modules.conv.causal_conv1d import causal_conv1d

        return causal_conv1d(x=x.unsqueeze(0), weight=w, bias=None, activation="silu", cu_seqlens=cu_seqlens.long())[
            0
        ][0]

    return surrogate(visible, reference, x, weight, name="kda_conv")


def _gated_norm(x, gate, weight, eps):
    """Serving's FusedRMSNormGated (sigmoid): rmsnorm(x) * weight * sigmoid(gate), per head row."""
    from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated

    def visible(x, gate, w):
        # rms_norm_gated writes its output into x's buffer.
        return rms_norm_gated(x.clone(), gate, w, None, "sigmoid", eps=eps)

    def reference(x, gate, w):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        return (y * torch.sigmoid(gate.float())).to(x.dtype)

    return surrogate(visible, reference, x, gate, weight, name="kda_out_norm")


class _TopKDACore(torch.nn.Module):
    """Holds the native layer's parameters under its names (``kda.<name>``)."""

    def __init__(self, hf):
        super().__init__()
        lin = hf.linear_attn_config
        self.heads, self.head_dim = lin["num_heads"], lin["head_dim"]
        self.lower_bound = lin["gate_lower_bound"]
        self.eps = hf.rms_norm_eps
        hidden, proj = hf.hidden_size, self.heads * self.head_dim
        self.q_proj = torch.nn.Linear(hidden, proj, bias=False)
        self.k_proj = torch.nn.Linear(hidden, proj, bias=False)
        self.v_proj = torch.nn.Linear(hidden, proj, bias=False)
        self.conv1d = torch.nn.Conv1d(3 * proj, 3 * proj, lin["short_conv_kernel_size"], groups=3 * proj, bias=False)
        self.b_proj = torch.nn.Linear(hidden, self.heads, bias=False)
        self.f_a_proj = torch.nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = torch.nn.Linear(self.head_dim, proj, bias=False)
        self.g_a_proj = torch.nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = torch.nn.Linear(self.head_dim, proj, bias=False)
        # FP32 as the checkpoint and serving keep them: the BF16 model cast must not round them.
        self.A_log = mark_keep_in_fp32(torch.nn.Parameter(torch.zeros(self.heads, dtype=torch.float32)))
        self.dt_bias = mark_keep_in_fp32(torch.nn.Parameter(torch.zeros(proj, dtype=torch.float32)))
        self.o_norm = torch.nn.Module()
        self.o_norm.weight = torch.nn.Parameter(torch.ones(self.head_dim))
        self.o_proj = torch.nn.Linear(proj, hidden, bias=False)

    def forward(self, x, cu_seqlens, lengths):
        t, d = x.shape[0], self.head_dim
        proj = self.heads * d
        w_a = torch.cat(
            (
                self.q_proj.weight,
                self.k_proj.weight,
                self.v_proj.weight,
                self.b_proj.weight,
                self.f_a_proj.weight,
                self.g_a_proj.weight,
            )
        )
        y = _merged_projection(x, w_a)
        qkv, b, fg_a = y.split((3 * proj, self.heads, 2 * d), dim=-1)
        f, g = _batched_projection(fg_a, torch.stack((self.f_b_proj.weight, self.g_b_proj.weight)), d)
        qkv = _conv(qkv, self.conv1d.weight.reshape(3 * proj, -1), cu_seqlens, lengths)
        canon_dump("kda_core_input", qkv)
        core = delegated_kda(
            qkv,
            f,
            b,
            self.A_log,
            self.dt_bias,
            cu_seqlens=cu_seqlens,
            heads=self.heads,
            head_dim=d,
            lower_bound=self.lower_bound,
        )
        canon_dump("kda_core_output", core)
        out = _gated_norm(core, g.reshape(1, t, self.heads, d), self.o_norm.weight, self.eps)
        return delegated_row_linear(out.reshape(t, proj), self.o_proj.weight)


class TopGlm5NextKDAAttention(MegatronModule):
    """The ``self_attention`` of a GLM-5.3-Flash KDA layer under TOP (packed THD input, trainer TP1, no CP)."""

    def __init__(self, config, layer_number, hf_config, **kwargs):
        super().__init__(config=config)
        if config.tensor_model_parallel_size != 1 or config.sequence_parallel:
            raise NotImplementedError("[top] GLM-5.3-Flash KDA requires trainer TP1 without sequence parallelism")
        if config.context_parallel_size != 1:
            raise NotImplementedError("[top] GLM-5.3-Flash KDA does not support context parallelism")
        self.kda = _TopKDACore(hf_config)

    def forward(self, hidden_states, attention_mask=None, *, packed_seq_params=None, **kwargs):
        if packed_seq_params is None or hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("[top] GLM-5.3-Flash KDA requires packed THD input with microbatch dimension one")
        if packed_seq_params.qkv_format != "thd":
            raise ValueError("[top] GLM-5.3-Flash KDA requires THD packed sequence metadata")
        cu = packed_seq_params.cu_seqlens_q
        bounds = cu.tolist()
        lengths = tuple(b - a for a, b in zip(bounds, bounds[1:], strict=False))
        if not lengths or bounds[0] != 0 or bounds[-1] != hidden_states.shape[0] or min(lengths) <= 0:
            raise ValueError("[top] GLM-5.3-Flash KDA document boundaries must cover the packed tokens")
        y = self.kda(hidden_states[:, 0], cu.to(torch.int32), lengths)
        return y[:, None], None
