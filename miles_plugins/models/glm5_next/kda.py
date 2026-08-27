"""GLM-5.3-Flash KDA (Kimi Delta Attention) linear-attention layer.

Wraps ``fla.ops.kda.chunk_kda`` behind the ``HuggingfaceAttention`` adapter.
Module field names follow the HF checkpoint 1:1; the only fused tensor is
``conv1d``, one packed depthwise conv over cat(q, k, v) whose weight
concatenates the checkpoint's ``q_conv1d``/``k_conv1d``/``v_conv1d``. The
block's own ``input_layernorm`` stays, so this wrapper adds none.
"""

import torch
import torch.nn as nn
from megatron.core.transformer.module import mark_keep_in_fp32

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda
except ImportError:
    FusedRMSNormGated = None
    ShortConvolution = None
    chunk_kda = None

from miles.backends.training_utils.cp_utils import build_gdn_cp_context
from miles_plugins.models.hf_attention import HuggingfaceAttention


def kda_gate(
    f: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    gate_lower_bound: float,
) -> torch.Tensor:
    """Per-token log-decay matching ``fla.ops.kda.fused_kda_gate``'s safe-gate
    branch: ``gate_lower_bound * sigmoid(exp(A_log) * (f + dt_bias))``, fully in
    fp32. ``f`` is ``[..., H, K]``, ``A_log`` is ``[H]``, ``dt_bias`` is ``[H * K]``.
    """
    num_heads = A_log.numel()
    a = A_log.float().exp().view(num_heads, 1)
    x = f.float() + dt_bias.float().view(num_heads, -1)
    return gate_lower_bound * torch.sigmoid(a * x)


def _get_text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def _linear_attn_fields(text_config) -> dict:
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if not isinstance(linear_attn_config, dict):
        linear_attn_config = {}

    def field(key, attr, default):
        value = linear_attn_config.get(key)
        if value is None:
            value = getattr(text_config, attr, default)
        return value

    gate_lower_bound = field("gate_lower_bound", "gate_lower_bound", None)
    if gate_lower_bound is None:
        gate_lower_bound = getattr(text_config, "linear_lower_bound", None)
    if gate_lower_bound is None:
        raise ValueError("GLM-5.3 KDA requires gate_lower_bound (safe gate) in the HF config.")
    return dict(
        num_heads=int(field("num_heads", "linear_num_heads", 64)),
        head_dim=int(field("head_dim", "linear_head_dim", 128)),
        conv_kernel_size=int(field("short_conv_kernel_size", "linear_conv_kernel_dim", 4)),
        gate_lower_bound=float(gate_lower_bound),
    )


class Glm5NextKDA(nn.Module):
    """GLM-5.3 KDA core with varlen support, calling fla's chunked autograd kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        if ShortConvolution is None or chunk_kda is None:
            raise ImportError("GLM-5.3 KDA requires flash-linear-attention >= 0.4.2 (fla.ops.kda).")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.conv1d = ShortConvolution(
            hidden_size=3 * self.projection_size,
            kernel_size=conv_kernel_size,
            bias=False,
            activation="silu",
        )
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.A_log = mark_keep_in_fp32(nn.Parameter(torch.zeros(num_heads, dtype=torch.float32)))
        self.dt_bias = mark_keep_in_fp32(nn.Parameter(torch.zeros(self.projection_size, dtype=torch.float32)))
        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor):
        cp_context = build_gdn_cp_context(self, cu_seqlens, hidden_states.device)

        mixed_qkv = torch.cat(
            (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)),
            dim=-1,
        )
        conv_cu_seqlens = cp_context.cu_seqlens if cp_context is not None else cu_seqlens
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=conv_cu_seqlens,
            cp_context=cp_context,
        )
        query, key, value = torch.split(mixed_qkv, [self.projection_size] * 3, dim=-1)
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        beta = torch.sigmoid(self.b_proj(hidden_states).float())
        forget = self.f_b_proj(self.f_a_proj(hidden_states))
        g = kda_gate(
            forget.unflatten(-1, (self.num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            self.gate_lower_bound,
        )

        if cp_context is not None:
            core_attn_out, _ = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cp_context.cu_seqlens,
                cp_context=cp_context,
            )
        else:
            core_attn_out, _ = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        norm_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        out_shape = core_attn_out.shape
        core_attn_out = self.o_norm(
            core_attn_out.reshape(-1, self.head_dim),
            norm_gate.reshape(-1, self.head_dim),
        )
        core_attn_out = core_attn_out.reshape(out_shape[0], out_shape[1], -1)
        return self.o_proj(core_attn_out)


class Glm5NextKDAAttention(HuggingfaceAttention):
    """HF-adapter wrapper placing the KDA core at ``self_attention.kda``.

    ``hybrid_cp=True``: fla's state-passing CP handles context parallelism
    natively via ``build_gdn_cp_context``, the same path as GDN.
    """

    hybrid_cp = True

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
    ):
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection, name=name)
        text_config = _get_text_config(self.hf_config)
        fields = _linear_attn_fields(text_config)
        self.kda = Glm5NextKDA(
            hidden_size=text_config.hidden_size,
            num_heads=fields["num_heads"],
            head_dim=fields["head_dim"],
            conv_kernel_size=fields["conv_kernel_size"],
            gate_lower_bound=fields["gate_lower_bound"],
            rms_norm_eps=text_config.rms_norm_eps,
        )

    def hf_forward(self, hidden_states, packed_seq_params):
        return self.kda(hidden_states, cu_seqlens=packed_seq_params.cu_seqlens_q)
