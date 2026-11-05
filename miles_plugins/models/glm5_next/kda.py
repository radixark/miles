import torch
import torch.nn as nn
from megatron.core.transformer.module import mark_keep_in_fp32

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    FusedRMSNormGated = None
    ShortConvolution = None
    chunk_kda = None
    fused_kda_gate = None

from miles_plugins.models.cp_utils import build_gdn_cp_context
from miles_plugins.models.hf_attention import HuggingfaceAttention
from miles_plugins.models.linear_attn import kda_kernel, miles_ext


def _get_text_config(hf_config):
    return (
        getattr(hf_config, "text_config", None) or hf_config
    )  # config-access-exempt: HF configs may wrap text_config for multimodal checkpoints


class Glm5NextKDA(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
        kda_backend: str = "fla",
        conv_backend: str = "fla",
    ):
        super().__init__()
        if ShortConvolution is None or chunk_kda is None:
            raise ImportError("GLM-5.3 KDA requires flash-linear-attention >= 0.4.2 (fla.ops.kda).")
        if kda_backend not in ("fla", "cake"):
            raise ValueError(f"Unsupported KDA backend: {kda_backend}")
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
        self.kda_backend = kda_backend
        self.conv_backend = conv_backend
        if conv_backend == "cake":
            conv = miles_ext("conv1d")
            conv.check_supported(3 * self.projection_size, conv_kernel_size, "silu")
            conv.prebuild(torch.bfloat16)
        if kda_backend == "cake":
            if head_dim != 128:
                raise ValueError(f"--kda-backend cake needs head dimension 128, got {head_dim}")
            self.cake_chunk_kda = kda_kernel("cake")
            from miles_ext_kernels.kda import prepare

            # the layer is not head-sharded: it raises above 64 heads, and on a GPU other than SM100a / SM103a / SM107a
            prepare(num_heads)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor):
        cp_context = build_gdn_cp_context(self, cu_seqlens, hidden_states.device)

        mixed_qkv = torch.cat(
            (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)),
            dim=-1,
        )
        if self.conv_backend == "cake":
            if cp_context is not None:
                raise NotImplementedError("--linear-attention-conv-backend cake does not support context parallelism")
            mixed_qkv = miles_ext("conv1d").causal_conv1d(mixed_qkv, self.conv1d.weight[:, 0], cu_seqlens)
        else:
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

        beta_logits = self.b_proj(hidden_states)
        forget = self.f_b_proj(self.f_a_proj(hidden_states)).unflatten(-1, (self.num_heads, self.head_dim))
        if self.kda_backend == "cake":
            core_attn_out = self._cake_kda(query, key, value, forget, beta_logits, cu_seqlens, cp_context)
        else:
            core_attn_out = self._fla_kda(query, key, value, forget, beta_logits, cu_seqlens, cp_context)

        norm_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        out_shape = core_attn_out.shape
        core_attn_out = self.o_norm(
            core_attn_out.reshape(-1, self.head_dim),
            norm_gate.reshape(-1, self.head_dim),
        )
        core_attn_out = core_attn_out.reshape(out_shape[0], out_shape[1], -1)
        return self.o_proj(core_attn_out)

    def _cake_kda(self, query, key, value, forget, beta_logits, cu_seqlens, cp_context):
        """The Cake kernels in raw-gate mode: they activate the forget gate themselves (no fused_kda_gate) and take
        a BF16 beta. q / k stay the strided split views of the conv output; v is made contiguous."""
        if cp_context is not None:
            raise NotImplementedError("--kda-backend cake does not support context parallelism yet.")
        assert query.shape[0] == 1, "--kda-backend cake takes one packed sequence (--qkv-format thd)"
        return self.cake_chunk_kda(
            query,
            key,
            value.contiguous(),
            g=forget,
            beta=beta_logits.float().sigmoid().to(torch.bfloat16),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
        )

    def _fla_kda(self, query, key, value, forget, beta_logits, cu_seqlens, cp_context):
        beta = torch.sigmoid(beta_logits.float())
        g = fused_kda_gate(forget, self.A_log, self.dt_bias, lower_bound=self.gate_lower_bound)

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
        return core_attn_out


class Glm5NextKDAAttention(HuggingfaceAttention):

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
        # the native config parses linear_attn_config into the linear_* fields
        if text_config.linear_lower_bound is None:
            raise ValueError("GLM-5.3 KDA requires gate_lower_bound (safe gate) in the HF config.")
        if args.linear_attention_conv_backend == "cake":
            miles_ext("conv1d", args)
        self.kda = Glm5NextKDA(
            hidden_size=text_config.hidden_size,
            num_heads=text_config.linear_num_heads,
            head_dim=text_config.linear_head_dim,
            conv_kernel_size=text_config.linear_conv_kernel_dim,
            gate_lower_bound=text_config.linear_lower_bound,
            rms_norm_eps=text_config.rms_norm_eps,
            kda_backend=args.kda_backend,
            conv_backend=args.linear_attention_conv_backend,
        )

    def hf_forward(self, hidden_states, packed_seq_params):
        return self.kda(hidden_states, cu_seqlens=packed_seq_params.cu_seqlens_q)
