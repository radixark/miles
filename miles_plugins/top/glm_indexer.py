"""Forward-only DSA index logits using the rollout's quantization and selection."""

import torch

_SGLANG_INDEXER_BLOCK_SIZE = 128
_SGLANG_INDEXER_SCALE_FMT = "ue8m0"
# One selection block's FP32 logits [rows, keys] stay within this many bytes, so long contexts select in smaller blocks.
_SELECT_LOGITS_BYTES = 1 << 30


def _sglang_fp8_indexer_logits(index_q, index_k, weights_2d, cu_seqlen_ks, cu_seqlen_ke):
    """Match rollout index logits using its quantization and packed key-cache format.

    Requires the unfused SGLang indexer so both sides apply the same Hadamard
    rotation. The caller supplies head weights scaled by num_heads**-0.5;
    the query quantization scale and head_dim**-0.5 are applied here.
    """
    from deep_gemm import fp8_mqa_logits
    from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import fused_store_index_k_cache
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation

    if index_q.shape[-1] != _SGLANG_INDEXER_BLOCK_SIZE:
        raise ValueError(
            f"[top] delegated indexer expects head_dim={_SGLANG_INDEXER_BLOCK_SIZE}, " f"got {index_q.shape[-1]}"
        )
    if index_k.dim() == 3:
        if index_k.shape[1] != 1:
            raise ValueError(f"[top] expected exactly one indexer KV head, got {tuple(index_k.shape)}")
        index_k = index_k.squeeze(1)

    q_fp8, q_scale = act_quant(
        rotate_activation(index_q.contiguous()),
        _SGLANG_INDEXER_BLOCK_SIZE,
        _SGLANG_INDEXER_SCALE_FMT,
    )

    # Use the cache writer's quantization and layout: each page stores 64 keys
    # followed by their FP32 scales. Plain act_quant does not reproduce this path.
    k_rotated = rotate_activation(index_k.contiguous())
    num_k = k_rotated.shape[0]
    page_size = 64
    num_pages = (num_k + page_size - 1) // page_size
    packed_k = torch.empty(
        (num_pages, page_size * (_SGLANG_INDEXER_BLOCK_SIZE + 4)),
        dtype=torch.uint8,
        device=k_rotated.device,
    )
    fused_store_index_k_cache(
        k_rotated,
        packed_k,
        torch.arange(num_k, dtype=torch.int64, device=k_rotated.device),
        page_size=page_size,
    )
    k_fp8 = (
        packed_k[:, : page_size * _SGLANG_INDEXER_BLOCK_SIZE]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(-1, _SGLANG_INDEXER_BLOCK_SIZE)[:num_k]
    )
    k_scale = (
        packed_k[:, page_size * _SGLANG_INDEXER_BLOCK_SIZE :].contiguous().view(torch.float32).reshape(-1)[:num_k]
    )

    # Preserve the rollout's multiplication order; folding these scales into
    # the head projection can change FP32 rounding.
    scaled_weights = (weights_2d.float() * q_scale.squeeze(-1).float() * (index_q.shape[-1] ** -0.5)).contiguous()

    logits = fp8_mqa_logits(
        q_fp8,
        (k_fp8, k_scale.squeeze(-1).contiguous()),
        scaled_weights,
        cu_seqlen_ks.to(torch.int32).contiguous(),
        cu_seqlen_ke.to(torch.int32).contiguous(),
        # Admission excludes forced prefix/local indices, so only the causal
        # range mask is needed.
        clean_logits=True,
    )
    # FlashInfer selection requires contiguous logits.
    return logits.contiguous()


def select_block_rows(num_keys, max_rows):
    """Rows per selection block: at most ``max_rows``, fewer when [rows, num_keys] FP32 logits would exceed the budget.
    Each row's logits and top-k depend only on that row, so the block size does not change the bytes."""
    rows = _SELECT_LOGITS_BYTES // (max(num_keys, 1) * 4) // 64 * 64
    return max(64, min(max_rows, rows))


def select_indices(index_q, index_k, weights, starts, ends, topk, *, fp8=False):
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend

    logits = _sglang_fp8_indexer_logits(index_q, index_k, weights, starts, ends)
    if fp8:
        from sglang.srt.true_on_policy.dsa.ordered_topk import ordered_dsa_topk

        relative = ordered_dsa_topk(logits, ends - starts, topk, starts)
    else:
        relative = DSATopKBackend.FLASHINFER.topk_func(
            logits,
            ends - starts,
            topk,
            row_starts=starts,
        )
    indices = torch.where(relative >= 0, relative + starts[:, None], relative)
    scores = logits.gather(-1, indices.clamp(min=0).long()).masked_fill(indices < 0, -torch.inf)
    return scores, indices.to(torch.int32)


class HeadWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        from sglang.srt.true_on_policy.dsa_head_gate import project_head_gate

        # Serving's op refuses operands that require grad; no gradient flows here (backward returns None).
        out = project_head_gate(flat.detach(), weight.detach())
        return out.view(*hidden_states.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        # The indexer is not trained (its selection is discrete and its scores are discarded).
        return None, None
