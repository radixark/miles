"""MiniMax-M3 self-attention: GQA restricted to the MSA indexer's top-k key blocks.

Index head ``h`` scores each causal key block by its max logit and picks, with the
query's own block forced in, the blocks kv head ``h`` attends to.
"""

import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from miles.kernels.attention.qsa.block_sparse_attention import qsa_block_sparse_attention
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import PackedBlockLayout, packed_positions

# bounds the [heads, queries, keys] fp32 logits the block scores are reduced from
_QUERY_CHUNK = 2048
# topk priority: init blocks, then local blocks, then the best scores; future blocks never
_INIT_BLOCK_SCORE = 1e30
_LOCAL_BLOCK_SCORE = 1e29
_INVALID_BLOCK_SCORE = -1e30


def gemma_rmsnorm_fp32(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xf * (1.0 + weight.float())


def apply_neox_rope_fp32(x: Tensor, rotary_pos_emb: Tensor, positions: Tensor) -> Tensor:
    """Neox rope on the leading ``rotary_dim`` dims of fp32 ``x`` ``[T, heads, head_dim]``."""
    freqs = rotary_pos_emb.flatten(1)
    half = freqs.shape[-1] // 2
    freqs = freqs[:, :half].float().contiguous()
    cos = freqs.cos()[positions.long()].unsqueeze(-2)
    sin = freqs.sin()[positions.long()].unsqueeze(-2)
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 2 * half :]], dim=-1)


class MiniMaxM3HeadNorm(torch.nn.Module):
    """Per-head gemma RMSNorm that stays in fp32, so rope runs in fp32 and q/k round to bf16 once as in sglang."""

    def __init__(self, config, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size, dtype=config.params_dtype))
        self.weight.sequence_parallel = config.sequence_parallel
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return gemma_rmsnorm_fp32(x, self.weight, self.eps)


class MiniMaxM3Indexer(MegatronModule):
    """Scores key blocks per index head and returns each query's selected block set."""

    def __init__(self, config):
        super().__init__(config)
        self.n_heads = config.minimax_index_n_heads
        self.head_dim = config.minimax_index_head_dim
        self.topk_blocks = config.minimax_index_topk_blocks
        self.block_size = config.minimax_index_block_size
        self.local_blocks = config.minimax_index_local_blocks
        self.init_blocks = config.minimax_index_init_blocks
        self.eps = config.layernorm_epsilon
        self.index_q_proj = TELinear(
            config.hidden_size,
            self.n_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.index_k_proj = TELinear(
            config.hidden_size,
            self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.index_q_norm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=config.params_dtype))
        self.index_k_norm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=config.params_dtype))
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def project(self, hidden_states: Tensor, positions: Tensor, rotary_pos_emb: Tensor, heads: list[int]):
        """``[T, hidden] -> (q [T, len(heads), D], k [T, D])`` in bf16, normed then roped in fp32."""
        q, _ = self.index_q_proj(hidden_states)
        k, _ = self.index_k_proj(hidden_states)
        q = q.view(-1, self.n_heads, self.head_dim)[:, heads]
        q = gemma_rmsnorm_fp32(q, self.index_q_norm, self.eps)
        k = gemma_rmsnorm_fp32(k.view(-1, 1, self.head_dim), self.index_k_norm, self.eps)
        q = apply_neox_rope_fp32(q, rotary_pos_emb, positions).to(hidden_states.dtype)
        k = apply_neox_rope_fp32(k, rotary_pos_emb, positions).to(hidden_states.dtype)
        return q, k.squeeze(1)

    @torch.no_grad()
    def select(self, q: Tensor, k: Tensor, cu_seqlens: Tensor) -> Tensor:
        """Selected blocks as packed start positions ``[H, T, topk_blocks]`` int64, -1 padded."""
        T, H, D = q.shape
        bs = self.block_size
        scale = D**-0.5
        selected = torch.full((H, T, self.topk_blocks), -1, dtype=torch.int64, device=q.device)
        cu = cu_seqlens.tolist()
        for s, e in zip(cu[:-1], cu[1:], strict=True):
            length = e - s
            if length == 0:
                continue
            nb = -(-length // bs)
            kpad = torch.zeros(nb * bs, D, dtype=torch.float32, device=q.device)
            kpad[:length] = k[s:e].float()
            key_pos = torch.arange(nb * bs, device=q.device)
            block_ids = torch.arange(nb, device=q.device)
            for qs in range(0, length, _QUERY_CHUNK):
                qe = min(qs + _QUERY_CHUNK, length)
                logits = torch.einsum("qhd,kd->hqk", q[s + qs : s + qe].float(), kpad) * scale
                qpos = torch.arange(qs, qe, device=q.device)
                logits = logits.masked_fill(~(key_pos.unsqueeze(0) <= qpos.unsqueeze(1)).unsqueeze(0), float("-inf"))
                score = logits.view(H, qe - qs, nb, bs).amax(dim=-1)
                valid_blocks = qpos // bs + 1
                valid = block_ids.unsqueeze(0) < valid_blocks.unsqueeze(1)
                local = valid & (
                    block_ids.unsqueeze(0) >= (valid_blocks - self.local_blocks).clamp(min=0).unsqueeze(1)
                )
                init = valid & (block_ids.unsqueeze(0) < self.init_blocks)
                score = torch.where(valid.unsqueeze(0), score, torch.full_like(score, _INVALID_BLOCK_SCORE))
                score = torch.where(local.unsqueeze(0), torch.full_like(score, _LOCAL_BLOCK_SCORE), score)
                score = torch.where(init.unsqueeze(0), torch.full_like(score, _INIT_BLOCK_SCORE), score)
                kk = min(self.topk_blocks, nb)
                top_score, top_idx = torch.topk(score, kk, dim=-1)
                selected[:, s + qs : s + qe, :kk] = torch.where(top_score > _INVALID_BLOCK_SCORE, s + top_idx * bs, -1)
        return selected


def selection_to_flags(selected: Tensor, layout: PackedBlockLayout) -> Tensor:
    """``[H, T, K]`` packed block-start positions -> ``[H, T, NB]`` uint8 flags on the packed grid."""
    H, T, _ = selected.shape
    flags = torch.zeros(H, T, layout.num_blocks, dtype=torch.uint8, device=selected.device)
    keep = selected >= 0
    heads = torch.arange(H, device=selected.device).view(-1, 1, 1).expand_as(selected)
    rows = torch.arange(T, device=selected.device).view(1, -1, 1).expand_as(selected)
    flags[heads[keep], rows[keep], layout.token_block[selected[keep]]] = 1
    return flags


class MiniMaxM3SparseCoreAttention(MegatronModule):
    """Core attention restricted, per kv head, to the blocks its index head selected."""

    def __init__(self, config, owner):
        super().__init__(config)
        object.__setattr__(self, "_owner", owner)
        self.softmax_scale = config.kv_channels**-0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask=None, **kwargs):
        block_form = self._owner._msa_block_form
        assert block_form is not None, "MiniMax-M3 core attention ran without a published block selection"
        assert query.dim() == 3, f"MiniMax-M3 sparse attention expects thd inputs, got {tuple(query.shape)}"
        flags, lo, hi, blk_base, tok_base, block_first, block_last, block_size = block_form
        query, key = query.to(value.dtype), key.to(value.dtype)
        T, hq, d = query.shape
        hkv = key.shape[1]
        group = hq // hkv
        outs = []
        for h in range(hkv):
            outs.append(
                qsa_block_sparse_attention(
                    query[:, h * group : (h + 1) * group],
                    key[:, h : h + 1],
                    value[:, h : h + 1],
                    flags[h],
                    lo,
                    hi,
                    blk_base,
                    tok_base,
                    block_first,
                    block_last,
                    self.softmax_scale,
                    block_size,
                )
            )
        return torch.cat(outs, dim=1).reshape(T, hq * d)


class MiniMaxM3DenseCoreAttention(MegatronModule):
    """Megatron's dense core attention, fed the fp32 q/k that the head norm and rope leave."""

    def __init__(self, config, core_attention):
        super().__init__(config)
        self.inner = core_attention

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args, **kwargs):
        return self.inner(query.to(value.dtype), key.to(value.dtype), value, *args, **kwargs)


class MiniMaxM3Attention(SelfAttention):
    """Megatron GQA self-attention; on sparse layers restricted to the MiniMax (MSA) indexer's blocks."""

    def __init__(self, config, submodules, layer_number=1, *args, sparse: bool = True, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        if config.rotary_interleaved:
            raise NotImplementedError("MiniMax-M3 uses neox rope; rotary_interleaved is unsupported")
        self.sparse = sparse
        self._msa_block_form = None
        if not sparse:
            self.core_attention = MiniMaxM3DenseCoreAttention(config, self.core_attention)
            return
        if self.checkpoint_core_attention:
            raise NotImplementedError("MiniMax-M3 sparse attention does not support selective core_attn recompute")
        self.indexer = MiniMaxM3Indexer(config)
        self.core_attention = MiniMaxM3SparseCoreAttention(config, owner=self)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        if not self.sparse:
            return super().forward(hidden_states, *args, **kwargs)
        if parallel_state.get_context_parallel_world_size() > 1:
            raise NotImplementedError("MiniMax-M3 sparse attention does not support context parallelism yet")
        packed = kwargs.get("packed_seq_params")
        assert packed is not None and packed.cu_seqlens_q is not None, "MiniMax-M3 sparse attention needs thd inputs"
        rotary_pos_emb = kwargs.get("rotary_pos_emb")
        assert rotary_pos_emb is not None, "the MSA indexer reuses the attention rope table"
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb[0]

        with torch.no_grad():
            indexer_states = hidden_states
            if self.config.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
                indexer_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
            seq = indexer_states.shape[0]
            cu = packed.cu_seqlens_q
            positions = packed_positions(cu, seq)
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            kv_heads_per_rank = self.config.num_query_groups // parallel_state.get_tensor_model_parallel_world_size()
            local_heads = list(range(tp_rank * kv_heads_per_rank, (tp_rank + 1) * kv_heads_per_rank))
            q, k = self.indexer.project(indexer_states[:, 0], positions, rotary_pos_emb, local_heads)
            selected = self.indexer.select(q, k, cu)
            layout = PackedBlockLayout(cu, positions, self.indexer.block_size)
            token_start = layout.token_start.to(torch.int32)
            self._msa_block_form = (
                selection_to_flags(selected, layout),
                token_start,
                torch.arange(seq, dtype=torch.int32, device=positions.device),
                layout.token_block_start.to(torch.int32),
                token_start,
                layout.block_first_token.to(torch.int32),
                layout.block_last_token.to(torch.int32),
                self.indexer.block_size,
            )
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._msa_block_form = None
