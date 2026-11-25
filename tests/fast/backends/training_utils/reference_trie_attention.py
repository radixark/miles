"""Plain-PyTorch trie attention, the reference the trie packing tests compare kernels and layouts against."""

import torch

from miles.backends.training_utils.data.trie_packing import TrieKV


def reference_trie_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv: TrieKV,
    cu_seqlens_q: torch.Tensor,
    *,
    window_left: int | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Trie attention over ``[T, heads, dim]`` tensors, one segment at a time.

    Each segment's queries attend to its gathered keys with the causal diagonal
    aligned bottom-right, which is what the varlen kernels do for
    ``seqlen_q < seqlen_k``.
    """
    heads, kv_heads, dim = query.shape[1], key.shape[1], query.shape[2]
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    gathered_k = key.index_select(0, kv.index)
    gathered_v = value.index_select(0, kv.index)
    outputs = []
    for segment in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = int(cu_seqlens_q[segment]), int(cu_seqlens_q[segment + 1])
        k_start, k_end = int(kv.cu_seqlens[segment]), int(kv.cu_seqlens[segment + 1])
        q = query[q_start:q_end].transpose(0, 1)
        k = gathered_k[k_start:k_end].transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
        v = gathered_v[k_start:k_end].transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
        n_q, n_k = q.shape[1], k.shape[1]
        row = torch.arange(n_q).unsqueeze(1) + (n_k - n_q)
        col = torch.arange(n_k).unsqueeze(0)
        allowed = col <= row
        if window_left is not None:
            allowed &= col >= row - window_left
        scores = (q @ k.transpose(1, 2)) * scale
        scores = scores.masked_fill(~allowed, float("-inf"))
        outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
    return torch.cat(outputs)
