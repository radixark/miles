"""Qwen3.8-Flash-Next attention bindings delegating shared TOP operations (trainer TP1/CP1).

Parameter names follow the Qwen3.8-Next Megatron layout that ``Qwen38NextBridge`` loads: GDN layers own
``linear_attn.*`` (the names of ``TopQwenGDN``, serving's chunk prefill per sample); full-attention layers own
``linear_qkv``/``linear_proj``/``q_layernorm``/``k_layernorm``/``indexer``. The full-attention program is
``TopQwen35FullAttention`` with QSA in place of FA4, so it takes no decode key split.
"""

import torch
from megatron.core.transformer.module import MegatronModule
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.utils.hf_transformers_utils import get_rope_config

from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import Qwen38NextQSAIndexer
from miles_plugins.top.binding import CaptureTap
from miles_plugins.top.gdn import TopQwenGDN
from miles_plugins.top.gdn_attention import (
    _packed_metadata,
    delegated_output_gate,
    fused_qk_norm_rope_gate,
    keep_fp32_cos_sin,
)
from miles_plugins.top.gdn_norm import TopGemmaRMSNorm
from miles_plugins.top.kernels.delegated_linear import delegated_column_linear, delegated_row_linear
from miles_plugins.top.qwen38_qsa import top_qsa_attention, top_qsa_selection


def megatron_qkv_to_serving_rows(weight, *, heads, kv_heads, head_dim):
    """Megatron [group][q | gate | k | v] rows -> serving [head][q, gate] | k | v rows."""
    per_group = heads // kv_heads
    w = weight.view(kv_heads, 2 * per_group + 2, head_dim, -1)
    q, gate = w[:, :per_group], w[:, per_group : 2 * per_group]
    qg = torch.stack((q, gate), dim=2).reshape(heads * 2 * head_dim, -1)
    k = w[:, 2 * per_group].reshape(kv_heads * head_dim, -1)
    v = w[:, 2 * per_group + 1].reshape(kv_heads * head_dim, -1)
    return torch.cat((qg, k, v))


class TopQwen38Attention(MegatronModule):
    def __init__(self, config, layer_number, hf_config, **kwargs):
        super().__init__(config=config)
        if config.tensor_model_parallel_size != 1 or config.context_parallel_size != 1:
            raise NotImplementedError("Qwen3.8-Next TOP attention requires trainer TP1/CP1")
        hf = hf_config
        self.linear = hf.layer_types[layer_number - 1] == "linear_attention"
        if self.linear:
            self.linear_attn = TopQwenGDN(hf)
            return
        self.heads, self.kv_heads, self.head_dim = hf.num_attention_heads, hf.num_key_value_heads, hf.head_dim
        qkv_rows = (2 * self.heads + 2 * self.kv_heads) * self.head_dim
        self.linear_qkv = torch.nn.Linear(hf.hidden_size, qkv_rows, bias=False)
        self.linear_proj = torch.nn.Linear(self.heads * self.head_dim, hf.hidden_size, bias=False)
        self.q_layernorm = TopGemmaRMSNorm(config, self.head_dim, hf.rms_norm_eps, role="q_layernorm")
        self.k_layernorm = TopGemmaRMSNorm(config, self.head_dim, hf.rms_norm_eps, role="k_layernorm")
        self.indexer = Qwen38NextQSAIndexer(config, layer_number=layer_number)
        # Capture point of the per-layer VJP (top_vjp): the packed key rows each row attends to ([T, W], -1 unused).
        self.top_index_selection = CaptureTap()
        self.softmax_scale = self.head_dim**-0.5
        theta, scaling = get_rope_config(hf)
        with torch.device("cuda"):
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position=hf.max_position_embeddings,
                base=theta,
                rope_scaling=scaling,
                partial_rotary_factor=hf.partial_rotary_factor,
                is_neox_style=True,
                dtype=torch.bfloat16,
            )
        keep_fp32_cos_sin(self.rotary_emb)

    def _full_attention(self, x, cu, positions, rope_positions):
        t = x.shape[0]
        w = megatron_qkv_to_serving_rows(
            self.linear_qkv.weight, heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim
        )
        qg, k, v = delegated_column_linear(x, w).split(
            (2 * self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim), -1
        )
        q, k, gate = fused_qk_norm_rope_gate(
            qg,
            k,
            q_norm=self.q_layernorm,
            k_norm=self.k_layernorm,
            positions=rope_positions,
            rope=self.rotary_emb,
            heads=self.heads,
            kv_heads=self.kv_heads,
        )
        with torch.no_grad():
            slots = self.top_index_selection(top_qsa_selection(self, x, positions, rope_positions, cu))
        out = top_qsa_attention(
            q.reshape(t, self.heads, self.head_dim),
            k.reshape(t, self.kv_heads, self.head_dim),
            v.reshape(t, self.kv_heads, self.head_dim),
            slots,
            self.softmax_scale,
        )
        return delegated_row_linear(delegated_output_gate(out.reshape(t, -1), gate), self.linear_proj.weight)

    def forward(self, hidden_states, attention_mask=None, *, packed_seq_params=None, **kwargs):
        if packed_seq_params is None or hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("Qwen3.8-Next TOP requires packed THD input with microbatch dimension one")
        if attention_mask is not None:
            raise ValueError("Qwen3.8-Next TOP accepts document-local causal attention only")
        m = _packed_metadata(packed_seq_params, hidden_states)
        if m.cp is not None or any(m.doc_starts):
            raise NotImplementedError("[top] Qwen3.8-Next scores whole samples on one rank (no context parallelism)")
        x = hidden_states[:, 0]
        if self.linear:
            y = self.linear_attn(x, m.cu, lengths_cpu=m.lengths)
        else:
            y = self._full_attention(x, m.cu, m.positions, m.rope_positions)
        return y[:, None], None
