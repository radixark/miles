"""Qwen3.8-Next QSA (Qwen Sparse Attention) indexer.

Runs on the full-attention layers (12 of 48) and picks, per query token, which
``indexer_budget`` key tokens the sparse attention will actually look at.

Reimplemented from sglang's ``QSAIndexer`` because Miles training does not depend
on SGLang internals.
"""

import math

import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


def _indexer_acc_dtype(x: Tensor) -> torch.dtype:
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32


def gemma_rmsnorm_last_dim(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm over the last dim with a Gemma-style ``1 + weight`` scale."""
    acc = _indexer_acc_dtype(x)
    xa = x.to(acc)
    var = xa.pow(2).mean(dim=-1, keepdim=True)
    return ((xa * torch.rsqrt(var + eps)) * (1.0 + weight.to(acc))).to(x.dtype)


def compress_keys_by_mean(token_k: Tensor, compress_ratio: int) -> Tensor:
    """``[T, head_dim] -> [ceil(T / r), head_dim]`` by averaging each run of ``r``."""
    tokens, dim = token_k.shape
    blocks = -(-tokens // compress_ratio)
    padded = blocks * compress_ratio
    if padded != tokens:
        pad = token_k.new_zeros(padded - tokens, dim)
        counts = token_k.new_ones(padded, 1)
        counts[tokens:] = 0
        summed = torch.cat([token_k, pad], dim=0).view(blocks, compress_ratio, dim).sum(1)
        denom = counts.view(blocks, compress_ratio, 1).sum(1).clamp_min(1)
        return summed / denom
    return token_k.view(blocks, compress_ratio, dim).mean(dim=1)


def block_causal_mask(query_positions: Tensor, num_blocks: int, compress_ratio: int) -> Tensor:
    """``[T, num_blocks]`` bool: which compressed blocks a query may attend to."""
    blocks = torch.arange(num_blocks, device=query_positions.device)
    first_invalid = (query_positions + 1) // compress_ratio
    return blocks.unsqueeze(0) < first_invalid.unsqueeze(1)


class PackedBlockLayout:
    """Per-sequence compressed-block grid for a packed (thd) batch.

    Every quantity the indexer needs is global-index-free: sglang scores one request
    at a time, so its block grid always starts at that request's token 0. A packed
    batch has to reproduce that per sequence, otherwise a sequence at pack offset
    ``s`` scores blocks that belong to whatever sits at the front of the buffer.
    """

    def __init__(self, cu_seqlens: Tensor, positions: Tensor, compress_ratio: int):
        device = positions.device
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        lengths = cu[1:] - cu[:-1]
        blocks_per_seq = -(-lengths // compress_ratio)
        self.num_blocks = int(blocks_per_seq.sum())

        seq_block_start = torch.cat([blocks_per_seq.new_zeros(1), blocks_per_seq.cumsum(0)[:-1]])
        seq_token_start = cu[:-1]

        total = positions.numel()
        seg = torch.zeros(total, dtype=torch.long, device=device)
        seg[seq_token_start[1:]] = 1
        seg = seg.cumsum(0)

        # per token
        self.token_seq = seg
        self.token_block = seq_block_start[seg] + positions // compress_ratio
        self.token_block_start = seq_block_start[seg]
        self.token_start = seq_token_start[seg]
        # per block
        self.block_seq = torch.repeat_interleave(torch.arange(blocks_per_seq.numel(), device=device), blocks_per_seq)
        self.block_local = torch.arange(self.num_blocks, device=device) - seq_block_start[self.block_seq]
        self.block_token_start = seq_token_start[self.block_seq] + self.block_local * compress_ratio


def compress_keys_by_mean_packed(token_k: Tensor, layout: PackedBlockLayout) -> Tensor:
    """``[T, head_dim] -> [layout.num_blocks, head_dim]``, averaging inside each sequence.

    Same result as ``compress_keys_by_mean`` per sequence, so no block ever mixes
    tokens from two sequences (which a global grid does whenever a sequence length
    is not a multiple of ``compress_ratio``).
    """
    acc = _indexer_acc_dtype(token_k)
    dim = token_k.shape[-1]
    summed = torch.zeros(layout.num_blocks, dim, dtype=acc, device=token_k.device)
    summed.index_add_(0, layout.token_block, token_k.to(acc))
    counts = torch.zeros(layout.num_blocks, 1, dtype=acc, device=token_k.device)
    counts.index_add_(0, layout.token_block, torch.ones(token_k.shape[0], 1, dtype=acc, device=token_k.device))
    return (summed / counts.clamp_min(1)).to(token_k.dtype)


def packed_block_causal_mask(query_positions: Tensor, layout: PackedBlockLayout, compress_ratio: int) -> Tensor:
    """``[T, num_blocks]`` bool, restricted to each query's own sequence."""
    blocks = torch.arange(layout.num_blocks, device=query_positions.device).unsqueeze(0)
    lo = layout.token_block_start.unsqueeze(1)
    first_invalid = lo + ((query_positions + 1) // compress_ratio).unsqueeze(1)
    return (blocks >= lo) & (blocks < first_invalid)


class Qwen38NextQSAIndexer(MegatronModule):
    """Selects the sparse-attention budget for one full-attention layer."""

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config)
        self.layer_number = layer_number
        self.n_heads = config.qwen3_8_next_indexer_n_heads
        self.kv_heads = config.qwen3_8_next_indexer_kv_heads
        self.head_dim = config.qwen3_8_next_indexer_head_dim
        self.token_topk = config.qwen3_8_next_indexer_budget
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio
        self.block_topk = self.token_topk // self.compress_ratio
        self.norm_eps = config.layernorm_epsilon
        self.index_qk_proj = TELinear(
            config.hidden_size,
            (self.n_heads + self.kv_heads) * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        dtype = config.params_dtype
        self.q_layernorm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        self.k_layernorm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        for p in (self.q_layernorm, self.k_layernorm):
            p.sequence_parallel = config.sequence_parallel

    def project_qk(
        self,
        hidden_states: Tensor,
        rotary_pos_emb: Tensor | None = None,
        layout: PackedBlockLayout | None = None,
    ):
        """``[T, hidden] -> (q [T, n_heads, head_dim], block_k [B, head_dim])``."""
        qk, _ = self.index_qk_proj(hidden_states)
        split = self.n_heads * self.head_dim
        q_raw, token_k = qk[..., :split], qk[..., split:]

        q = gemma_rmsnorm_last_dim(q_raw.reshape(-1, self.head_dim), self.q_layernorm, self.norm_eps).reshape(
            -1, self.n_heads, self.head_dim
        )

        token_k = token_k.reshape(-1, self.head_dim)
        if layout is None:
            block_k = compress_keys_by_mean(token_k, self.compress_ratio)
        else:
            block_k = compress_keys_by_mean_packed(token_k, layout)
        block_k = gemma_rmsnorm_last_dim(block_k, self.k_layernorm, self.norm_eps)

        if rotary_pos_emb is not None:
            q = self._apply_rope(q, rotary_pos_emb[: q.shape[0]])
            if layout is None:
                block_freqs = rotary_pos_emb[:: self.compress_ratio][: block_k.shape[0]]
            else:
                block_freqs = rotary_pos_emb[layout.block_token_start]
            block_k = self._apply_rope(block_k.unsqueeze(1), block_freqs).squeeze(1)
        return q, block_k

    def _apply_rope(self, x: Tensor, rotary_pos_emb: Tensor) -> Tensor:
        return apply_rotary_pos_emb(
            x.unsqueeze(1),
            rotary_pos_emb,
            config=self.config,
        ).squeeze(1)

    def score_blocks(
        self,
        q: Tensor,
        block_k: Tensor,
        query_positions: Tensor,
        layout: PackedBlockLayout | None = None,
    ) -> Tensor:
        """``[T, num_blocks]`` fp32 logits, invalid blocks at ``-inf``."""
        scores = torch.einsum("mhd,nd->mnh", q.float(), block_k.float())
        logits = torch.relu(scores).sum(dim=-1) / math.sqrt(self.head_dim)
        if layout is None:
            valid = block_causal_mask(query_positions, block_k.shape[0], self.compress_ratio)
        else:
            valid = packed_block_causal_mask(query_positions, layout, self.compress_ratio)
        return logits.masked_fill(~valid, float("-inf"))

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        cu_seqlens: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
    ) -> Tensor:
        """``[T, hidden] -> [T, token_topk]`` int32 token indices, ``-1`` where unused.

        ``positions`` restart at 0 per sequence; the returned indices are absolute in
        the packed buffer. With a single sequence both coordinate systems coincide,
        which is why the packed path has to be explicit: without ``cu_seqlens`` every
        sequence but the first scores the blocks sitting at the front of the buffer
        and then has them all clamped away, leaving it with no keys at all.
        """
        layout = None
        if cu_seqlens is not None and cu_seqlens.numel() > 2:
            layout = PackedBlockLayout(cu_seqlens, positions, self.compress_ratio)

        q, block_k = self.project_qk(hidden_states, rotary_pos_emb=rotary_pos_emb, layout=layout)
        logits = self.score_blocks(q, block_k, positions, layout=layout)

        k = min(self.block_topk, logits.shape[-1])
        block_scores, block_idx = torch.topk(logits, k, dim=-1)
        block_idx = block_idx.masked_fill(block_scores == float("-inf"), -1)

        offsets = torch.arange(self.compress_ratio, device=block_idx.device)
        if layout is None:
            tokens = block_idx.unsqueeze(-1) * self.compress_ratio + offsets
        else:
            block_local = block_idx - layout.token_block_start.unsqueeze(-1)
            tokens = layout.token_start.unsqueeze(-1).unsqueeze(-1) + (
                block_local.unsqueeze(-1) * self.compress_ratio + offsets
            )
        tokens = tokens.masked_fill(block_idx.unsqueeze(-1) < 0, -1).flatten(-2)
        limit = positions if layout is None else layout.token_start + positions
        tokens = tokens.masked_fill(tokens > limit.unsqueeze(-1), -1)
        if layout is not None:
            tokens = tokens.masked_fill(tokens < layout.token_start.unsqueeze(-1), -1)
        return tokens[..., : self.token_topk].to(torch.int32)
