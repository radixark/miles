"""MSA block scores for training: each query's maximum logit over every causal key block of its sequence."""

import torch
import triton
import triton.language as tl
from torch import Tensor

_BQ = 64


@triton.jit
def _block_max_scores_kernel(
    Q,
    K,
    TILES,
    OUT,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_oh,
    stride_ot,
    scale,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BLK: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    q_start = tl.load(TILES + tile * 3)
    seq_start = tl.load(TILES + tile * 3 + 1)
    seq_end = tl.load(TILES + tile * 3 + 2)

    offs_q = q_start + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q_in = offs_q < seq_end
    q = tl.load(Q + offs_q[:, None] * stride_qt + head * stride_qh + offs_d[None, :], mask=q_in[:, None], other=0.0)

    # a tile never straddles sequences, so every query shares the key blocks; the last query bounds them
    last_block = (tl.minimum(q_start + BQ, seq_end) - 1 - seq_start) // BLK
    for block in range(0, last_block + 1):
        offs_k = seq_start + block * BLK + tl.arange(0, BLK)
        k_in = offs_k < seq_end
        k = tl.load(K + offs_k[:, None] * stride_kt + offs_d[None, :], mask=k_in[:, None], other=0.0)
        logits = tl.dot(q, tl.trans(k))
        causal = (offs_k[None, :] <= offs_q[:, None]) & k_in[None, :]
        best = tl.max(tl.where(causal, logits, float("-inf")), axis=1) * scale
        tl.store(OUT + head * stride_oh + offs_q * stride_ot + block, best, mask=q_in)


def msa_block_max_scores(q: Tensor, k: Tensor, cu_seqlens: Tensor, block_size: int, scale: float) -> Tensor:
    """``q`` [T, H, D], ``k`` [T, D] -> ``[H, T, NB]`` fp32 block scores, ``-inf`` where no key is causal.

    ``NB`` is the block count of the longest sequence; blocks start at each sequence's first token.
    """
    T, H, D = q.shape
    bounds = cu_seqlens.tolist()
    tiles = [
        (q_start, start, end)
        for start, end in zip(bounds[:-1], bounds[1:], strict=True)
        for q_start in range(start, end, _BQ)
    ]
    num_blocks = max(-(-(end - start) // block_size) for start, end in zip(bounds[:-1], bounds[1:], strict=True))
    tiles = torch.tensor(tiles, dtype=torch.int32, device=q.device)
    out = torch.full((H, T, num_blocks), float("-inf"), dtype=torch.float32, device=q.device)
    qc, kc = q.contiguous(), k.contiguous()
    _block_max_scores_kernel[(tiles.shape[0], H)](
        qc,
        kc,
        tiles,
        out,
        qc.stride(0),
        qc.stride(1),
        kc.stride(0),
        out.stride(0),
        out.stride(1),
        scale,
        D=D,
        BQ=_BQ,
        BLK=block_size,
        num_warps=4,
        num_stages=2,
    )
    return out
