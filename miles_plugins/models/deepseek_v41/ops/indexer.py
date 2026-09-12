import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.deepseek_v41.ops.kernel.tilelang_indexer_fwd import batched_indexer_fwd
from miles_plugins.models.deepseek_v41.ops.norm import RMSNorm
from miles_plugins.models.deepseek_v41.ops.quant import fake_quant_fp4
from miles_plugins.models.deepseek_v41.ops.rope import apply_rotary_emb
from miles_plugins.models.dsa_topk import get_dsa_topk_fn


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


INDEXER_QUERY_CHUNK = int(os.environ.get("MILES_DSV41_INDEXER_QUERY_CHUNK", "8192"))

try:
    import deep_select
except ImportError:  # the torch path stays for images without it
    deep_select = None
USE_DEEP_SELECT = deep_select is not None and os.environ.get("MILES_DSV41_DEEP_SELECT", "1") == "1"


def indexer_select(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_lens: torch.Tensor,
    candidates: torch.Tensor | None,
    *,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    topk: int,
    topk_fn,
    allow_deep_select: bool,
    query_chunk: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    bsz, seqlen = q.shape[:2]
    n_kv = index_k.size(1)
    k_t = index_k.transpose(0, 1).contiguous()
    use_deep_select = USE_DEEP_SELECT and allow_deep_select
    if use_deep_select:
        # deep_select requires aligned score rows; the padded columns sit past every query's length
        align = deep_select.get_stride_requirement()[0] // 4
        pad_kv = -(-n_kv // align) * align - n_kv
        if pad_kv:
            k_t = torch.nn.functional.pad(k_t, (0, 0, 0, 0, 0, pad_kv))
    chunk = seqlen if not query_chunk or query_chunk >= seqlen else query_chunk
    idx_parts: list[torch.Tensor] = []
    cand_parts: list[torch.Tensor] = []
    for s in range(0, seqlen, chunk):
        e = min(s + chunk, seqlen)
        lens = compress_lens[s:e]
        scores = batched_indexer_fwd(
            q[:, s:e].transpose(0, 1).contiguous(),
            k_t,
            weights[:, s:e].transpose(0, 1).float().contiguous(),
            torch.zeros(e - s, dtype=torch.int32, device=q.device),
            lens.to(torch.int32),
        )
        # batched_indexer_fwd already wrote -inf outside [0, lens) for every query
        if is_candidate_source:
            cand = select_candidate_blocks(
                scores[..., :n_kv], lens.unsqueeze(-1), candidate_topk_blocks, candidate_block_size
            )
            cand_parts.append(cand)
        elif uses_candidates:
            assert candidates is not None
            scores[..., :n_kv].masked_fill_(~candidates[:, s:e], -torch.inf)
        if use_deep_select:
            flat = scores.reshape(bsz * (e - s), scores.size(-1))
            row_end = lens.to(torch.int32).repeat(bsz)
            _, idx = deep_select.topk(
                flat,
                min(topk, n_kv),
                end=row_end,
                indices_type=torch.int64,
                return_value=False,
                idx_oob_fill_value=-1,
            )
            idx = idx.reshape(bsz, e - s, -1)
            if n_kv < topk:
                idx = torch.nn.functional.pad(idx, (0, topk - n_kv), value=-1)
            idx_parts.append(idx.sort(dim=-1).values)
            continue
        if n_kv < topk:
            scores = torch.nn.functional.pad(scores, (0, topk - n_kv), value=-torch.inf)
        idx = topk_fn(scores.reshape(bsz * (e - s), scores.size(-1)), topk)
        idx = idx.reshape(bsz, e - s, topk).sort(dim=-1).values.to(torch.int64)
        idx_parts.append(torch.where(idx < lens.unsqueeze(-1), idx, -1))
    idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=1)
    if is_candidate_source:
        candidates = cand_parts[0] if len(cand_parts) == 1 else torch.cat(cand_parts, dim=1)
    return idx, candidates


class DeepSeekV41Indexer(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_id: int,
        head_dim: int,
        compress_ratio: int,
        owns_k: bool,
        is_candidate_source: bool,
        uses_candidates: bool,
    ):
        super().__init__(config=config)
        self.layer_id = layer_id
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.is_candidate_source = is_candidate_source
        self.uses_candidates = uses_candidates
        self.candidate_topk_blocks = config.v41_candidate_topk_blocks
        self.candidate_block_size = config.v41_candidate_block_size
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.softmax_scale = self.index_head_dim**-0.5
        q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size

        self.linear_wq_b = TELinear(
            q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_weights_proj = TELinear(
            config.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        if owns_k:
            self.linear_wk = nn.Linear(head_dim, self.index_head_dim, bias=False, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.index_head_dim, config.layernorm_epsilon)
            mark_keep_in_fp32(self.k_norm.weight)
        indexer_replay_manager.register_to_module(
            self, "indexer_replay", stream_idx=sorted(config.v41_index_source_layer_ids).index(layer_id)
        )

    @torch.no_grad()
    def index_keys(self, latent: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        k = self.k_norm(F.linear(latent, self.linear_wk.weight)).clone()
        apply_rotary_emb(k[..., -self.rope_head_dim :], freqs_cis)
        return fake_quant_fp4(k)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        index_k: torch.Tensor,
        freqs_cis: torch.Tensor,
        compress_lens: torch.Tensor,
        candidates: torch.Tensor | None,
    ):
        bsz, seqlen, _ = x.shape
        rd = self.rope_head_dim
        q, _ = self.linear_wq_b(qr)
        q = q.view(bsz, seqlen, self.index_n_heads, self.index_head_dim).clone()
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = fake_quant_fp4(q)
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.softmax_scale * self.index_n_heads**-0.5)

        base_topk_fn = get_dsa_topk_fn("torch")
        topk_fn = indexer_replay_manager.get_topk_fn(base_topk_fn, return_probs=False)
        idx, candidates = indexer_select(
            q,
            index_k,
            weights,
            compress_lens,
            candidates,
            is_candidate_source=self.is_candidate_source,
            uses_candidates=self.uses_candidates,
            candidate_topk_blocks=self.candidate_topk_blocks,
            candidate_block_size=self.candidate_block_size,
            topk=self.index_topk,
            topk_fn=topk_fn,
            allow_deep_select=topk_fn is base_topk_fn,
            query_chunk=INDEXER_QUERY_CHUNK,
        )
        return idx, candidates
