"""Step5's frozen DSA region selection in plain torch, shared by the Megatron and torchtitan indexers.

Each query keeps up to 512 earlier eight-token regions per KV group plus its own. `indexer_ops`
documents the remaining scoring-rounding differences from SGLang.
"""

import itertools
from typing import NamedTuple

import torch
import torch.nn.functional as F

from miles.kernels.linear.batch_invariant_linear import batch_invariant_linear
from miles_plugins.models.step5.indexer_ops import (
    HEADS_PER_GROUP,
    INDEXER_HEAD_DIM,
    WEIGHT_SCALE,
    Llama3Rope,
    compress_regions,
    norm_rope_key,
    norm_rope_query,
    rope_cos_sin,
    score_regions,
    to_e4m3,
    top_regions,
)
from miles_plugins.models.step5.layer_math import FULL_ATTENTION_ROPE
from miles_plugins.models.step5.region_selection import MAX_SELECTED_REGIONS, REGION_SIZE, TOP_REGIONS, RegionSelection

INDEXER_NUM_HEADS = 16
INDEXER_ROPE = Llama3Rope(
    theta=FULL_ATTENTION_ROPE.theta,
    factor=1.0,
    low_freq_factor=1.0,
    high_freq_factor=32.0,
    original_max_position_embeddings=1 << 20,
)

# Indexer parameter (Megatron and torchtitan name them alike) -> its checkpoint tensor under
# ``model.layers.N.self_attn``.
CHECKPOINT_NAMES = {
    "q_weight": "sparse_indexer_q.weight",
    "k_weight": "sparse_indexer_k.weight",
    "z_weight": "sparse_indexer_z.weight",
    "w_weight": "sparse_indexer_w.weight",
    "q_norm_weight": "sparse_indexer_q_norm.weight",
    "k_norm_weight": "sparse_indexer_k_norm.weight",
    "k_norm_bias": "sparse_indexer_k_norm.bias",
    "ssmax_s": "ssmax_s",
}

# Positions before this one have at most TOP_REGIONS earlier regions and keep them all, unscored.
_FIRST_SCORED_POSITION = (TOP_REGIONS + 1) * REGION_SIZE
# Default bound on the FP32 dot products alive at once, [query rows, local heads, regions]: 64 MiB.
SCORE_CHUNK_ELEMENTS = 1 << 24
# Queries are prepared this many rows at a time, then scored in memory-bounded chunks.
_QUERY_BLOCK_ROWS = 2048
# Below 16 rows cuBLAS runs the BF16 projections as a GEMV that rounds differently; SGLang's prefill
# projects whole prompts, so query projections always cover at least this many rows.
_MIN_GEMM_ROWS = 16


class IndexerWeights(NamedTuple):
    """One full-attention layer's indexer tensors, for the KV groups this rank selects for.

    ``q_weight`` and ``w_weight`` hold the rows of those groups' provider heads (group ``g`` reads
    heads ``4g .. 4g + 3``); the rest are whole. FP32 tensors round to BF16 where they are used.
    """

    q_weight: torch.Tensor  # [local heads * 256, hidden]
    k_weight: torch.Tensor  # [256, hidden]
    z_weight: torch.Tensor  # [256, hidden]
    w_weight: torch.Tensor  # [local heads, hidden]
    q_norm_weight: torch.Tensor  # [256]
    k_norm_weight: torch.Tensor  # [256]
    k_norm_bias: torch.Tensor  # [256]

    @property
    def num_groups(self) -> int:
        return self.w_weight.shape[0] // HEADS_PER_GROUP


@torch.no_grad()
def select_regions(
    tokens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    weights: IndexerWeights,
    *,
    eps: float,
    score_chunk_elements: int = SCORE_CHUNK_ELEMENTS,
) -> RegionSelection:
    """The regions SGLang selects for every token of a THD pack, per KV group of ``weights``.

    ``tokens`` is the attention input, the input norm's BF16 output ``[T, hidden]``; ``cu_seqlens``
    (int32) are the pack's sequence boundaries.
    """
    bounds = cu_seqlens.tolist()
    # Rows outside every sequence would stay unwritten, and the sparse kernel reads int32 boundaries.
    assert cu_seqlens.dtype == torch.int32 and bounds[0] == 0 and bounds[-1] == tokens.shape[0], (
        f"the pack's sequences must cover its {tokens.shape[0]} tokens: {cu_seqlens.dtype} cu_seqlens "
        f"from {bounds[0]} to {bounds[-1]}"
    )
    assert tokens.ndim == 2 and tokens.dtype == torch.bfloat16, "the indexer projections read BF16 [T, hidden]"
    region_ids = torch.empty(
        (tokens.shape[0], weights.num_groups, MAX_SELECTED_REGIONS), dtype=torch.int32, device=tokens.device
    )
    for start, end in itertools.pairwise(bounds):
        _select_sequence(
            tokens[start:end], weights, eps=eps, score_chunk_elements=score_chunk_elements, out=region_ids[start:end]
        )
    return RegionSelection(region_ids=region_ids, cu_seqlens=cu_seqlens)


def _select_sequence(
    tokens: torch.Tensor, weights: IndexerWeights, *, eps: float, score_chunk_elements: int, out: torch.Tensor
) -> None:
    """Fill ``out`` ``[L, groups, 513]`` with one sequence's selection; ``tokens`` is ``[L, hidden]``."""
    length = tokens.shape[0]
    out.copy_(_unscored_region_ids(length, device=tokens.device)[:, None])
    if length <= _FIRST_SCORED_POSITION:
        return
    cos, sin = rope_cos_sin(INDEXER_ROPE, length, tokens.device)
    region_keys = _region_keys(tokens, weights, cos, sin, eps=eps)
    rows_per_chunk = max(1, score_chunk_elements // (weights.w_weight.shape[0] * region_keys.shape[0]))
    for block in _row_slices(_FIRST_SCORED_POSITION, length, _QUERY_BLOCK_ROWS):
        query, head_weights = _queries(tokens, block, weights, cos[block], sin[block], eps=eps)
        _select_block(query, head_weights, region_keys, rows_per_chunk, first_row=block.start, out=out)


def _select_block(query, head_weights, region_keys, rows_per_chunk: int, *, first_row: int, out: torch.Tensor) -> None:
    """Score one block of queries against the regions before them, in memory-bounded row chunks."""
    for local in _row_slices(0, query.shape[0], rows_per_chunk):
        rows = slice(first_row + local.start, first_row + local.stop)
        scores = score_regions(query[local], head_weights[local], region_keys[: (rows.stop - 1) // REGION_SIZE])
        candidates = torch.arange(rows.start, rows.stop, device=query.device) // REGION_SIZE
        out[rows, :, :TOP_REGIONS] = top_regions(scores, candidates)


def _region_keys(
    tokens: torch.Tensor, weights: IndexerWeights, cos: torch.Tensor, sin: torch.Tensor, *, eps: float
) -> torch.Tensor:
    """E4M3 ``[complete regions, 256]``: each region's compressed index key."""
    complete = tokens.shape[0] // REGION_SIZE * REGION_SIZE
    tokens = tokens[:complete]
    key = norm_rope_key(
        F.linear(tokens, weights.k_weight),
        weights.k_norm_weight,
        weights.k_norm_bias,
        cos[:complete],
        sin[:complete],
        eps,
    )
    logits = F.linear(tokens, weights.z_weight)
    return compress_regions(
        key.view(-1, REGION_SIZE, INDEXER_HEAD_DIM), logits.view(-1, REGION_SIZE, INDEXER_HEAD_DIM)
    )


def _queries(tokens: torch.Tensor, rows: slice, weights: IndexerWeights, cos: torch.Tensor, sin: torch.Tensor, *, eps):
    """For ``tokens[rows]``: E4M3 queries ``[rows, groups, 4, 256]`` and FP32 head weights ``[rows, groups, 4]``."""
    first = max(0, min(rows.start, rows.stop - _MIN_GEMM_ROWS))
    query = F.linear(tokens[first : rows.stop], weights.q_weight)[rows.start - first :]
    count = query.shape[0]
    query = to_e4m3(norm_rope_query(query.view(count, -1, INDEXER_HEAD_DIM), weights.q_norm_weight, cos, sin, eps))
    head_weights = batch_invariant_linear(tokens[rows], weights.w_weight.to(torch.bfloat16))
    groups = (count, weights.num_groups, HEADS_PER_GROUP)
    return query.view(*groups, INDEXER_HEAD_DIM), head_weights.float().view(groups) * WEIGHT_SCALE


def _row_slices(start: int, stop: int, step: int) -> list[slice]:
    return [slice(row, min(row + step, stop)) for row in range(start, stop, step)]


def _unscored_region_ids(length: int, *, device: torch.device) -> torch.Tensor:
    """int32 ``[L, 513]``: regions ``0 .. min(c, 512) - 1``, then the token's own region ``c``, then -1.

    Complete for positions whose ``c = p // 8`` is at most TOP_REGIONS; later rows keep ``c`` in the
    last slot and get their scored regions written over the rest.
    """
    current = torch.arange(length, device=device)[:, None] // REGION_SIZE
    history = current.clamp(max=TOP_REGIONS)
    slot = torch.arange(MAX_SELECTED_REGIONS, device=device)
    return torch.where(slot < history, slot, torch.where(slot == history, current, -1)).to(torch.int32)
