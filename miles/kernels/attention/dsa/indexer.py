"""DSA lightning indexer: per-query scores over a KV range, top-k selection, and the gathered
top-k scores with a fused backward.

Every function takes the packed layout: q [T, H, D] bf16, k [T_kv, D] bf16, weights [T, H] fp32,
and per-query KV ranges cu_seqlen_ks / cu_seqlen_ke [T] int32 (query t scores keys in [ks, ke)).
A batched sbhd caller runs one packed problem per batch element with `indexer_logits_sbhd`.

Scores come from the Triton kernel (tensor-core MMA on Hopper and Blackwell) or the TileLang one, as an
IndexerConfig picks: measured rows live in _TUNED, keyed by (arch major, heads). Score rows are padded to
SCORE_ROW_ALIGN columns so the canonical top-k can read them with aligned vector loads.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from miles.kernels.attention.dsa.topk import SCORE_ROW_ALIGN
from miles.kernels.attention.dsa.triton.indexer_fwd import indexer_fwd as triton_indexer_fwd


_BWD_MIN_TOPK = 32


@dataclass(frozen=True)
class IndexerConfig:
    score_backend: Literal["triton", "tilelang"] = "tilelang"
    block_rows: int = 256
    block_n: int = 64
    num_warps: int = 4
    num_stages: int = 2


# Measured at 16k tokens (DeepSeek-V4 C4 indexer: 64 heads, ratio 4; V4.1 / GLM-5: 32 heads, ratio 1) on
# H200 (9) and GB300 (10). block_rows = queries per program x heads.
_TUNED = {
    (9, 32): IndexerConfig(score_backend="triton", block_rows=256, block_n=64, num_warps=8, num_stages=2),
    (9, 64): IndexerConfig(score_backend="triton", block_rows=256, block_n=64, num_warps=4, num_stages=3),
    (10, 32): IndexerConfig(score_backend="triton", block_rows=128, block_n=128, num_warps=4, num_stages=2),
    (10, 64): IndexerConfig(score_backend="triton", block_rows=128, block_n=128, num_warps=4, num_stages=2),
}


def causal_ranges(cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed causal ranges: query t attends keys [segment_start(t), t + 1)."""
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends


def causal_ranges_compressed(seq_len_q: int, compress_ratio: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal ranges over compressed KV: query p attends compressed groups [0, (p + 1) // ratio)."""
    positions = torch.arange(seq_len_q, device=device, dtype=torch.int32)
    ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    ke = ((positions + 1) // compress_ratio).to(torch.int32)
    return ks, ke


def _default_config(q: torch.Tensor) -> IndexerConfig:
    return _TUNED.get((torch.cuda.get_device_capability(q.device)[0], q.shape[-2]), IndexerConfig())


def _empty_scores(*leading: int, seq_len_kv: int, device) -> torch.Tensor:
    padded = -(-seq_len_kv // SCORE_ROW_ALIGN) * SCORE_ROW_ALIGN
    return torch.empty(*leading, padded, device=device, dtype=torch.float32)[..., :seq_len_kv]


def _write_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke, out, clean_logits, config):
    if config.score_backend == "triton":
        triton_indexer_fwd(
            q,
            k,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            out,
            clean_logits=clean_logits,
            block_rows=config.block_rows,
            block_n=config.block_n,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        return
    # tilelang is GPU-only; importing it lazily keeps the package importable on CPU
    from miles.kernels.attention.dsa.tilelang.indexer_fwd import indexer_fwd

    padded_kv = out.stride(0)
    k = torch.nn.functional.pad(k, (0, 0, 0, padded_kv - k.shape[0]))
    indexer_fwd(
        q,
        k,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out.as_strided((out.shape[0], padded_kv), (padded_kv, 1)),
        clean_logits=clean_logits,
    )


def indexer_logits(
    q, k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits: bool = True, config: IndexerConfig | None = None
) -> torch.Tensor:
    """Scores [T, T_kv] fp32 with SCORE_ROW_ALIGN-padded rows. Keys outside a query's range are -inf when
    clean_logits; otherwise they are undefined and only a range-aware top-k may read the result."""
    out = _empty_scores(q.shape[0], seq_len_kv=k.shape[0], device=q.device)
    _write_logits(
        q.contiguous(),
        k.contiguous(),
        weights.float().contiguous(),
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        clean_logits,
        config or _default_config(q),
    )
    return out


def indexer_logits_sbhd(
    q, k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits: bool = True, config: IndexerConfig | None = None
) -> torch.Tensor:
    """q [S, B, H, D], k [S_kv, B, D], weights [S, B, H]; ranges are shared across the batch.
    Returns [B, S, S_kv] with SCORE_ROW_ALIGN-padded rows."""
    seqlen, batch, _, _ = q.shape
    out = _empty_scores(batch, seqlen, seq_len_kv=k.shape[0], device=q.device)
    config = config or _default_config(q)
    for b in range(batch):
        _write_logits(
            q[:, b].contiguous(),
            k[:, b].contiguous(),
            weights[:, b].float().contiguous(),
            cu_seqlen_ks,
            cu_seqlen_ke,
            out[b],
            clean_logits,
            config,
        )
    return out


def gather_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    return torch.where(valid_mask, scores, float("-inf"))


def _pad_topk_pow2(topk_indices, grad_scores):
    topk = topk_indices.shape[-1]
    padded = max(_BWD_MIN_TOPK, 1 << (topk - 1).bit_length())
    if padded == topk:
        return topk_indices.contiguous(), grad_scores.contiguous()
    pad = padded - topk
    topk_indices = torch.nn.functional.pad(topk_indices, (0, pad), value=-1)
    grad_scores = torch.nn.functional.pad(grad_scores, (0, pad), value=0.0)
    return topk_indices.contiguous(), grad_scores.contiguous()


class _IndexerTopkScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, weights, logits, topk_indices):
        ctx.save_for_backward(q, k, weights, topk_indices)
        return gather_topk_scores(logits, topk_indices)

    @staticmethod
    def backward(ctx, grad_scores):
        q, k, weights, topk_indices = ctx.saved_tensors
        topk_indices, grad_scores = _pad_topk_pow2(topk_indices, grad_scores)
        from miles.kernels.attention.dsa.tilelang.indexer_bwd import indexer_bwd

        grad_q, grad_w, grad_k = indexer_bwd(q, weights, k, topk_indices, grad_scores)
        return grad_q, grad_k, grad_w, None, None


def indexer_topk_scores(q, k, weights, logits, topk_indices) -> torch.Tensor:
    """Scores [T, topk] gathered from `logits` at `topk_indices` (-1 = padding, scored -inf).
    The backward recomputes the selected scores from q / k / weights in one TileLang kernel."""
    return _IndexerTopkScores.apply(q, k, weights.float(), logits, topk_indices)


def lighting_indexer(q, k, weights, cu_seqlen_ks, cu_seqlen_ke, topk: int, topk_fn, clean_logits: bool = True):
    """Scores, top-k by `topk_fn(logits, topk, cu_seqlen_ks, cu_seqlen_ke)`, gathered scores. The caller owns
    the selection policy (e.g. routing replay), so the kernel holds no framework state; a range-aware topk_fn
    may skip clean_logits. Returns (scores [T, topk], topk_indices [T, topk] int32)."""
    logits = indexer_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits)
    topk_indices = topk_fn(logits, topk, cu_seqlen_ks, cu_seqlen_ke)
    return indexer_topk_scores(q, k, weights, logits, topk_indices), topk_indices
