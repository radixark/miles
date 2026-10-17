"""Qwen3.8-Next QSA (Qwen Sparse Attention) indexer.

Runs on the full-attention layers and picks, per query token, which
``indexer_budget`` key tokens the sparse attention will look at.
"""

import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles_plugins.models.qwen3_8_next.ops.qsa_selection import (
    PackedBlockLayout,
    project_indexer_qk,
    score_indexer_blocks,
    select_indexer_tokens,
)


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
        positions: Tensor,
        rotary_pos_emb: Tensor,
        layout: PackedBlockLayout | None = None,
    ):
        """``[T, hidden] -> (q [T, n_heads, head_dim], block_k [B, head_dim])``."""
        qk, _ = self.index_qk_proj(hidden_states)
        return project_indexer_qk(
            qk,
            self.q_layernorm,
            self.k_layernorm,
            positions,
            rotary_pos_emb,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
            eps=self.norm_eps,
            layout=layout,
        )

    def score_blocks(
        self,
        q: Tensor,
        block_k: Tensor,
        query_positions: Tensor,
        layout: PackedBlockLayout | None = None,
    ) -> Tensor:
        """``[T, num_blocks]`` fp32 logits, invalid blocks at ``-inf``."""
        return score_indexer_blocks(
            q, block_k, query_positions, head_dim=self.head_dim, compress_ratio=self.compress_ratio, layout=layout
        )

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        rotary_pos_emb: Tensor,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        """``[T, hidden] -> [T, token_topk]`` int32 token indices, ``-1`` where unused.

        ``positions`` restart at 0 per sequence; the returned indices are absolute in
        the packed buffer, so a packed batch must pass ``cu_seqlens``. ``rotary_pos_emb``
        is the attention's RoPE table, see ``apply_indexer_rope``.
        """
        layout = None
        if cu_seqlens is not None and cu_seqlens.numel() > 2:
            layout = PackedBlockLayout(cu_seqlens, positions, self.compress_ratio)

        q, block_k = self.project_qk(hidden_states, positions, rotary_pos_emb, layout=layout)
        logits = self.score_blocks(q, block_k, positions, layout=layout)

        return select_indexer_tokens(
            logits,
            positions,
            block_topk=self.block_topk,
            token_topk=self.token_topk,
            compress_ratio=self.compress_ratio,
            layout=layout,
        )
