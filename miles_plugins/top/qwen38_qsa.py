"""Qwen3.8-Next QSA binding: rollout indexer selection and sparse attention.

Selection and attention delegate to SGLang's pinned program. Packed documents are
scored with zero history; compressed keys exist only for complete groups. The
attention backward differentiates an FP32 reference over the selected slots; the
selection is discrete, so the indexer takes no gradient.
"""

import json

import torch
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.true_on_policy import qsa as sgl_qsa

_ROPE = {}


def rope_params_from_hf(hf_checkpoint: str) -> dict:
    """``rope_base``/``rotary_dim`` as SGLang's attention RoPE resolves them for Qwen4-Exp."""
    with open(f"{hf_checkpoint}/config.json") as f:
        config = json.load(f)
    text = config.get("text_config", config)
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    base = rope.get("rope_theta", text.get("rope_theta"))
    factor = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))
    head_dim = text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])
    return {"rope_base": float(base), "rotary_dim": int(head_dim * factor)}


def _document_layout(positions, cu_seqlens, total, ratio):
    """Per token: document start and length; per compressed key: member rows."""
    device = positions.device
    starts = torch.arange(total, device=device) - positions
    if cu_seqlens is None:
        cu = torch.tensor([0, total], device=device, dtype=torch.int64)
    else:
        cu = cu_seqlens.to(device=device, dtype=torch.int64)
    lens = cu[1:] - cu[:-1]
    groups = torch.div(lens, ratio, rounding_mode="floor")
    group_base = torch.cumsum(groups, 0) - groups
    doc_of_token = torch.searchsorted(cu[1:], torch.arange(total, device=device), right=True)
    total_groups = int(groups.sum().item())
    doc_of_group = torch.repeat_interleave(
        torch.arange(groups.numel(), device=device), groups, output_size=total_groups
    )
    local = torch.arange(total_groups, device=device) - group_base.index_select(0, doc_of_group)
    first_rows = cu.index_select(0, doc_of_group) + local * ratio
    return {
        "token_start": starts,
        "token_len": lens.index_select(0, doc_of_token),
        "key_start": group_base.index_select(0, doc_of_token),
        "group_rows": first_rows.unsqueeze(1) + torch.arange(ratio, device=device),
    }


def top_qsa_selection(attention, hidden_states, positions, rope_positions, cu_seqlens):
    """[T, hidden] -> [T, budget + r - 1] int32 absolute key rows, ascending, -1 at the end. ``positions`` [T] index
    each row in its sample (what a key group and a block are counted by); ``rope_positions`` [T] or mRoPE [3, T]
    are what the indexer rotates by, which differ from the index at and after an image."""
    indexer = attention.indexer
    ratio = indexer.compress_ratio
    rope = _ROPE["params"]
    positions = positions.to(torch.int64)
    rope_rows = sgl_qsa.rope_positions(rope_positions)
    axes = sgl_qsa.lane_axes(attention.rotary_emb, hidden_states.device)
    total = hidden_states.shape[0]
    qk = sgl_qsa.index_projection(hidden_states, indexer.index_qk_proj.weight)
    split = indexer.n_heads * indexer.head_dim
    q = sgl_qsa.norm_rope(
        qk[:, :split].reshape(-1, indexer.n_heads, indexer.head_dim),
        rope_rows,
        weight=indexer.q_layernorm,
        eps=indexer.norm_eps,
        axes=axes,
        **rope,
    )
    token_k = qk[:, split:].reshape(-1, indexer.head_dim)

    layout = _document_layout(positions, cu_seqlens, total, ratio)
    keys = sgl_qsa.compress_norm_rope(
        token_k,
        layout["group_rows"],
        # A compressed key rotates at its group's first member.
        rope_rows.index_select(0, layout["group_rows"][:, 0]),
        weight=indexer.k_layernorm,
        eps=indexer.norm_eps,
        axes=axes,
        **rope,
    )

    counts = torch.div(positions + 1, ratio, rounding_mode="floor")
    blocks = sgl_qsa.select_contiguous(q, keys, layout["key_start"], counts, indexer.block_topk)
    logical = expand_qsa_block_indices(
        blocks,
        positions,
        layout["token_len"].to(torch.int32),
        compress_ratio=ratio,
        token_topk=indexer.token_topk,
    ).long()
    absolute = logical + layout["token_start"].unsqueeze(1)
    return torch.where(logical >= 0, absolute, torch.full_like(absolute, -1)).to(torch.int32)


# Query rows per backward chunk: a chunk gathers [rows, slots, HKV, D] FP32 keys and values.
_QSA_BACKWARD_ROWS = 256


def qsa_attention_reference(query, key, value, slots, scale):
    """FP32 attention of each query row over its selected key rows (``slots`` [T, W], -1 = unused): [T, HQ * D]."""
    T, HQ, D = query.shape
    HKV = key.shape[1]
    valid = slots >= 0
    idx = slots.clamp_min(0).long()
    k = key.float().index_select(0, idx.reshape(-1)).view(T, -1, HKV, D)
    v = value.float().index_select(0, idx.reshape(-1)).view(T, -1, HKV, D)
    q = query.float().view(T, HKV, HQ // HKV, D)
    s = torch.einsum("tkgd,twkd->tkgw", q, k) * scale
    s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("tkgw,twkd->tkgd", p, v).reshape(T, HQ * D).to(query.dtype)


def top_qsa_attention(query, key, value, slots, scale):
    """thd [T, Hq, D] query and [T, Hkv, D] key/value -> [T, Hq * D]: serving's sparse attention forward, the FP32
    reference's backward over the same selected slots (selection itself is discrete and takes no gradient)."""
    from miles_plugins.top.kernels.surrogate import row_surrogate

    key, value = key.contiguous(), value.contiguous()
    T = query.shape[0]

    def visible(q, k, v):
        out, _, _ = sgl_qsa.sparse_attention(q, k, v, slots, scale, aux=False)
        return out.reshape(T, -1)

    def chunk_reference(lo, hi, q, k, v):
        return qsa_attention_reference(q, k, v, slots[lo:hi], scale)

    return row_surrogate(
        visible, chunk_reference, query, key, value, chunk_rows=_QSA_BACKWARD_ROWS, name="qsa_attention"
    )


def install(*, rope_base: float, rotary_dim: int) -> str:
    """Pin the QSA RoPE parameters; ``TopQwen38Attention`` calls the program; returns the stamp."""
    _ROPE["params"] = {"base": rope_base, "rotary_dim": rotary_dim}
    return f"qwen38_qsa=delegate(sglang.true_on_policy.qsa) rope_base={rope_base} rotary_dim={rotary_dim}"
