"""Qwen3.5 hybrid layout bindings delegating shared TOP operations (trainer TP1; all-gather CP via gdn_cp)."""

from typing import NamedTuple

import torch
from megatron.core.transformer.module import MegatronModule
from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.true_on_policy.attention import flash_attn_varlen_func
from sglang.srt.utils.hf_transformers_utils import get_rope_config

from miles_plugins.top.gdn import TopQwenGDN
from miles_plugins.top.gdn_norm import TopGemmaRMSNorm
from miles_plugins.top.host_copy import device_tensor
from miles_plugins.top.kernels.delegated_linear import delegated_column_linear, delegated_row_linear
from miles_plugins.top.kernels.surrogate import surrogate


def _packed_metadata(packed, hidden_states):
    """Cache document metadata on its owner, once per packed layout/version."""
    if packed.qkv_format != "thd":
        raise ValueError("GDN TOP requires THD packed sequence metadata")
    source = packed.cu_seqlens_q
    cached = getattr(
        packed, "_top_gdn_metadata", None
    )  # config-access-exempt: TOP attaches _top_gdn_metadata to PackedSeqParams per batch; stock batches carry none
    kv_source = packed.cu_seqlens_kv
    try:
        key = (
            id(source),
            source._version,
            id(kv_source),
            kv_source._version,
            hidden_states.shape[0],
            hidden_states.device,
        )
    except RuntimeError:
        # Inference-mode tensors have no version counter; validate on each use.
        key = None
    if key is not None and cached is not None and cached[0] == key:
        return cached[1]
    bounds = source.tolist()
    lengths = tuple(b - a for a, b in zip(bounds, bounds[1:], strict=False))
    if not lengths or bounds[0] != 0 or bounds[-1] != hidden_states.shape[0] or min(lengths) <= 0:
        raise ValueError("GDN TOP document boundaries must cover the packed tokens")
    if packed.cu_seqlens_kv is not source and packed.cu_seqlens_kv.tolist() != bounds:
        raise ValueError("GDN TOP requires identical query and key document boundaries")
    cu = source.to(device=hidden_states.device, dtype=torch.int32)
    # A context-parallel piece starts at its absolute position in its sample; a whole sample at 0.
    doc_starts = tuple(getattr(packed, "doc_starts", ()) or ()) or (0,) * len(
        lengths
    )  # config-access-exempt: TOP attaches doc_starts to PackedSeqParams per batch; stock batches carry none
    offsets = device_tensor(doc_starts, device=hidden_states.device, dtype=torch.int32)
    starts = torch.repeat_interleave(cu[:-1] - offsets, cu[1:] - cu[:-1], output_size=hidden_states.shape[0])
    positions = torch.arange(hidden_states.shape[0], device=hidden_states.device) - starts
    rope_positions = positions
    mrope = getattr(
        packed, "mrope_positions", None
    )  # config-access-exempt: TOP attaches mrope_positions to PackedSeqParams per batch; stock batches carry none
    if mrope is not None:
        # [3, rows]: the temporal, height and width position of every row, as serving's processor computes them.
        if mrope.shape != (3, hidden_states.shape[0]):
            raise ValueError(f"GDN TOP mRoPE positions {tuple(mrope.shape)} do not cover the packed tokens")
        rope_positions = mrope.to(hidden_states.device)
    cp = getattr(
        packed, "gdn_cp", None
    )  # config-access-exempt: TOP attaches gdn_cp to PackedSeqParams per batch; stock batches carry none
    keys, conv_init = (cp.keys, cp.conv_init) if cp is not None else (None, None)
    split = _decode_split_plan(packed, bounds, doc_starts, keys, hidden_states.device)
    meta = _GDNPacked(cu, lengths, positions, rope_positions, doc_starts, keys, conv_init, cp, split)
    packed._top_gdn_metadata = (key, meta)
    return meta


class _GDNPacked(NamedTuple):
    cu: torch.Tensor
    lengths: tuple
    positions: torch.Tensor  # [T] each row's index in its sample
    rope_positions: torch.Tensor  # what attention rotates by: ``positions``, or the sample's [3, T] mRoPE positions
    doc_starts: tuple
    keys: tuple | None  # CP: (key rows in CP order of the all-gathered keys, cu_seqlens_k, max_seqlen_k)
    conv_init: tuple | None  # CP: per document, None or the 3 earlier conv inputs' rows in cat(halo, x)
    cp: object  # GDNCPLayout or None
    split: object  # decode_split.DecodeSplitPlan or None


def _decode_split_plan(packed, bounds, doc_starts, keys, device):
    """The attention decode split's plan for the rows it computes (decode_split), or None."""
    rows = getattr(
        packed, "decode_rows", ()
    )  # config-access-exempt: TOP attaches decode_rows to PackedSeqParams per batch; stock batches carry none
    if not rows:
        return None
    from miles_plugins.top.decode_split import bound_split, build_plan

    cu_k = keys[1].tolist() if keys is not None else bounds
    return build_plan(
        decode_rows=list(rows), cu=bounds, doc_starts=doc_starts, cu_k=cu_k, split=bound_split(), device=device
    )


# id(rotary embedding) -> (that rope, its FP32 cos/sin table), kept at construction; one entry per shared rope.
_FP32_COS_SIN: dict[int, tuple] = {}


def _kept_cos_sin(rope):
    entry = _FP32_COS_SIN.get(id(rope))
    return entry[1] if entry is not None and entry[0] is rope else None


def keep_fp32_cos_sin(rope):
    """Keep the FP32 cos/sin table serving's fused kernel reads. The module's own buffer does not stay FP32 in the
    trainer: the BF16 model cast and the eager mRoPE's dtype matching both narrow it."""
    table = _kept_cos_sin(rope)
    if table is None:
        if rope.cos_sin_cache.dtype != torch.float32:
            raise ValueError(f"[top] the rotary table must be built FP32, got {rope.cos_sin_cache.dtype}")
        table = rope.cos_sin_cache.clone()
        _FP32_COS_SIN[id(rope)] = (rope, table)
    return table


def _reference_cos_sin(positions, rope):
    """FP32 cos and sin [T, rotary_dim / 2] at 1-D positions [T], or at mRoPE positions [3, T] with each rotary
    lane read at its own axis' position, the selection serving's MRotaryEmbedding makes."""
    table = _kept_cos_sin(rope)
    cos, sin = (rope.cos_sin_cache if table is None else table)[positions].float().chunk(2, dim=-1)
    if positions.ndim == 1:
        return cos, sin
    if not rope.mrope_section:
        raise ValueError("[top] [3, T] positions need a rotary embedding with mRoPE sections")
    if rope.mrope_interleaved:
        from sglang.srt.layers.rotary_embedding.mrope import apply_interleaved_rope

        return apply_interleaved_rope(cos, rope.mrope_section), apply_interleaved_rope(sin, rope.mrope_section)
    return tuple(
        torch.cat([part[axis] for axis, part in enumerate(table.split(rope.mrope_section, dim=-1))], dim=-1)
        for table in (cos, sin)
    )


def rotary_reference(x, positions, rope):
    """Neox rotary on the first ``rope.rotary_dim`` dims of each head of x [T, heads*D], in FP32."""
    t, rd = x.shape[0], rope.rotary_dim
    x = x.float().reshape(t, -1, rope.head_size)
    cos, sin = _reference_cos_sin(positions, rope)
    cos, sin = cos[:, None], sin[:, None]
    x1, x2, rest = x[..., : rd // 2], x[..., rd // 2 : rd], x[..., rd:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1).reshape(t, -1)


def fused_qk_norm_rope_gate(qg, k, *, q_norm, k_norm, positions, rope, heads, kv_heads):
    """(q [T, heads*D], k [T, kv_heads*D], gate [T, heads*D]) from qg [T, heads*2D] (per head [q | gate]) and k: the
    value is serving's fused Q/K Gemma norm + partial RoPE + gate split kernel, the gradient the unfused chain's
    (serving's norm with TE's backward, then the FP32 rotation; the gate passes through)."""
    table = _kept_cos_sin(rope)
    if table is None:
        raise ValueError("[top] the fused Q/K norm + RoPE needs the FP32 table kept by keep_fp32_cos_sin")
    t, d = qg.shape[0], rope.head_size
    positions = positions.contiguous()
    axis_map = rope.axis_map if positions.ndim == 2 else None

    def visible(qg, k, wq, wk):
        q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(
            qg,
            k,
            wq,
            wk,
            table,
            positions,
            q_norm.eps,
            heads,
            kv_heads,
            d,
            rope.rotary_dim,
            has_gate=True,
            mrope_axis_map=axis_map,
        )
        return q, k, gate.view(t, -1)

    def reference(qg, k, wq, wk):
        q, gate = qg.reshape(t, heads, 2 * d).chunk(2, -1)
        q = q_norm.rows(q.reshape(-1, d), wq).reshape(t, -1)
        k = k_norm.rows(k.reshape(-1, d), wk).reshape(t, -1)
        return (
            rotary_reference(q, positions, rope).to(q.dtype),
            rotary_reference(k, positions, rope).to(k.dtype),
            gate.reshape(t, -1),
        )

    return surrogate(visible, reference, qg, k, q_norm.weight, k_norm.weight, name="qk_norm_rope_gate")


def delegated_output_gate(x, gate):
    """Serving's fused x * sigmoid(gate)."""

    def reference(x, gate):
        return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)

    return surrogate(lambda x, gate: fused_sigmoid_mul(x, gate, inplace=False), reference, x, gate, name="output_gate")


class TopQwen35FullAttention(torch.nn.Module):
    def __init__(self, hf, config):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = hf.num_attention_heads, hf.num_key_value_heads, hf.head_dim
        self.q_proj = torch.nn.Linear(hf.hidden_size, 2 * self.heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hf.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hf.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.heads * self.head_dim, hf.hidden_size, bias=False)
        self.q_norm = TopGemmaRMSNorm(config, self.head_dim, hf.rms_norm_eps, role="q_layernorm")
        self.k_norm = TopGemmaRMSNorm(config, self.head_dim, hf.rms_norm_eps, role="k_layernorm")
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

    def forward(self, x, cu, lengths, positions, keys=None, cp=None, split=None):
        t = x.shape[0]
        # Serving uses a merged QKV projection. Its output-row order is unchanged
        # by concatenating the HF-owned parameters here.
        w = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight))
        qg, k, v = delegated_column_linear(x, w).split(
            (2 * self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim),
            -1,
        )
        q, k, gate = fused_qk_norm_rope_gate(
            qg,
            k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            positions=positions,
            rope=self.rotary_emb,
            heads=self.heads,
            kv_heads=self.kv_heads,
        )
        k, v = k.reshape(t, self.kv_heads, self.head_dim), v.reshape(t, self.kv_heads, self.head_dim)
        cu_k, max_k = cu, max(lengths)
        if cp is not None:
            from miles_plugins.top.gdn_cp import all_gather_rows

            k, v = all_gather_rows(k, cp), all_gather_rows(v, cp)
        if keys is not None:
            index, cu_k, max_k = keys
            k, v = k.index_select(0, index), v.index_select(0, index)
        q = q.reshape(t, self.heads, self.head_dim)
        out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max_k,
            softmax_scale=self.head_dim**-0.5,
        )
        if isinstance(out, tuple):
            out = out[0]
        if split is not None:
            from miles_plugins.top.decode_split import apply

            out = apply(split, out, q=q, k=k, v=v, scale=self.head_dim**-0.5)
        return delegated_row_linear(delegated_output_gate(out.reshape(t, -1), gate), self.o_proj.weight)


class TopQwen35Attention(MegatronModule):
    def __init__(self, config, layer_number, hf_config, **kwargs):
        super().__init__(config=config)
        if config.tensor_model_parallel_size != 1:
            raise NotImplementedError("GDN TOP attention requires trainer TP1")
        self.linear = hf_config.layer_types[layer_number - 1] == "linear_attention"
        if self.linear:
            self.linear_attn = TopQwenGDN(hf_config)
        else:
            self.self_attn = TopQwen35FullAttention(hf_config, config)

    def forward(self, hidden_states, attention_mask=None, *, packed_seq_params=None, **kwargs):
        if packed_seq_params is None or hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("GDN TOP requires packed THD input with microbatch dimension one")
        if attention_mask is not None:
            raise ValueError("GDN TOP accepts document-local causal attention only")
        m = _packed_metadata(packed_seq_params, hidden_states)
        x = hidden_states[:, 0]
        if self.linear:
            y = self.linear_attn(x, m.cu, lengths_cpu=m.lengths, conv_init=m.conv_init, cp=m.cp)
        else:
            y = self.self_attn(x, m.cu, m.lengths, m.rope_positions, m.keys, m.cp, m.split)
        return y[:, None], None
