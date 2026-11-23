"""Qwen GDN block: serving's GDN forward (one chunk prefill per packed document), with reference backwards.

HF parameter names are preserved. A packed document starts from zero state, or under context parallelism from the
previous rank's state and convolution inputs; the trainer scope is TP1 (CP: gdn_cp).
"""

import torch
from torch import nn

from miles_plugins.top.kernels.delegated_linear import delegated_column_linear, delegated_row_linear
from miles_plugins.top.kernels.gdn import delegated_gdn
from miles_plugins.top.kernels.gdn_components import delegated_gated_norm
from miles_plugins.top.kernels.gdn_conv import delegated_conv


class TopQwenGDN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.eps = config.rms_norm_eps
        # Serving's RMSNormGated activation: Qwen3.5 SiLU ("swish"), Qwen3.8-Next sigmoid.
        self.gate_activation = (
            getattr(config, "output_gate_type", None) or "swish"
        )  # config-access-exempt: TransformerConfig versions and model families differ in optional output_gate_type
        if config.linear_conv_kernel_dim != 4 or config.hidden_act != "silu":
            raise ValueError("TOP Qwen GDN requires width-4 convolution and SiLU")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("GDN requires integral grouped value heads")
        self.in_proj_qkv = nn.Linear(config.hidden_size, 2 * self.key_dim + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.conv1d = nn.Conv1d(
            2 * self.key_dim + self.value_dim,
            2 * self.key_dim + self.value_dim,
            4,
            groups=2 * self.key_dim + self.value_dim,
            bias=False,
        )
        self.norm = nn.RMSNorm(self.head_v_dim, eps=self.eps)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype

        mark_param_dtype(self.A_log, torch.float32)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

    def _apply(self, fn):
        original = self.A_log.data
        super()._apply(fn)
        self.A_log.data = original.to(device=self.A_log.device, dtype=torch.float32)
        if self.A_log.grad is not None:
            self.A_log.grad.data = self.A_log.grad.data.float()
        return self

    @staticmethod
    def _project(x, layer):
        return delegated_column_linear(x, layer.weight)

    def forward(self, hidden_states, cu_seqlens, *, lengths_cpu=None, conv_init=None, cp=None):
        if hidden_states.ndim != 2:
            raise ValueError("GDN block expects packed [T,H] hidden states")
        t = hidden_states.shape[0]
        qkv = self._project(hidden_states, self.in_proj_qkv)
        z = self._project(hidden_states, self.in_proj_z).view(1, t, self.num_v_heads, self.head_v_dim)
        a = self._project(hidden_states, self.in_proj_a)
        b = self._project(hidden_states, self.in_proj_b)
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        lengths_cpu = lengths.tolist() if lengths_cpu is None else lengths_cpu
        halo = None
        if cp is not None:
            from miles_plugins.top.gdn_cp import exchange_halo

            halo = exchange_halo(qkv, cp, key=id(self))
        qkv = delegated_conv(qkv, self.conv1d.weight[:, 0], cu_seqlens, lengths_cpu, init_rows=conv_init, halo=halo)
        out = delegated_gdn(
            qkv,
            a,
            b,
            self.A_log,
            self.dt_bias,
            cu_seqlens=cu_seqlens,
            heads=(self.num_k_heads, self.head_k_dim, self.num_v_heads, self.head_v_dim),
            lengths=tuple(lengths_cpu),
            cp=cp,
            key=id(self),
        )
        out = delegated_gated_norm(out, z, self.norm.weight, self.eps, self.gate_activation).reshape(t, self.value_dim)
        return delegated_row_linear(out, self.out_proj.weight)
