from functools import lru_cache

import einops
import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.cp_row_balance import (
    RowBalancePlan,
    RowExchange,
    plan_causal_row_balance,
    send_rows_to_scorers,
)
from miles_plugins.models.deepseek_v4.ops.compressor import DeepSeekV4Compressor
from miles_plugins.models.deepseek_v4.ops.cp_utils import all_gather_cp, get_freqs_cis_for_cp
from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd import batched_indexer_fwd
from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat
from miles_plugins.models.deepseek_v4.ops.rope import apply_rotary_emb, wrapped_precompute_freqs_cis
from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout, compress_bounds_at_positions, get_q_positions_thd
from miles_plugins.models.deepseek_v4.ops.utils import rotate_activation
from miles_plugins.models.dsa_topk import get_dsa_topk_fn


class V4Indexer(MegatronModule):
    """DSA Indexer for DeepSeek-V4 C4 layers."""

    def __init__(self, config: TransformerConfig, pg_collection=None, layer_id: int = 0):
        super().__init__(config=config)

        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.topk_backend = config.miles_dsa_topk_backend
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.compress_ratio = 4
        self.use_fp8_qat = config.fp8 is not None

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        self.linear_wq_b = TELinear(
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_weights_proj = TELinear(
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.compressor = DeepSeekV4Compressor(
            config=config,
            head_dim=self.index_head_dim,
            compress_ratio=self.compress_ratio,
            rotate=True,
            cp_group=pg_collection.cp,
        )

        # RL rollout-routing-replay (R3) seam for the sparse-attention indexer topk: lets the miles
        # indexer_replay_manager record (on rollout) / replay (on train forward) the top-k KV picks,
        # mirroring the MoE routing-replay seam. No-op unless the manager is enabled.
        indexer_replay_manager.register_to_module(self, "indexer_replay", stream_idx=layer_id)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask=None,
        packed_seq_params=None,
        thd_layout: ThdLayout | None = None,
    ):
        """Forward pass.

        Args:
            x:  hidden states [seqlen, batch, hidden_size]
            qr: low-rank query [seqlen, batch, q_lora_rank]
            mask: unused (the kernel's per-query KV bounds are generated internally)
            packed_seq_params: unused
            thd_layout: packed-stream layout, or None when unpacked

        Returns:
            topk_indices: [batch, seqlen, min(index_topk, n_kv)] int32, or index_topk columns of -1
            when no segment has a compressed key
        """

        # =========================================
        # Gather inputs if SP is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        seqlen, bsz, _ = x.size()

        q, _ = self.linear_wq_b(qr)
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)

        rd = self.rope_head_dim
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_group = self.pg_collection.cp if hasattr(self.pg_collection, "cp") else None
        rope_base = self.config.csa_compress_rotary_base if self.compress_ratio else self.config.rotary_base
        freqs_cis = wrapped_precompute_freqs_cis(
            self.config, self.rope_head_dim, rope_base, False, seqlen * cp_size, x.device
        )
        if thd_layout is None:
            freqs_cis = get_freqs_cis_for_cp(freqs_cis, seqlen, cp_size, cp_group, stride=1)
        else:
            # Packed positions restart at every segment boundary and index the whole table, so
            # the rank's rows are selected here rather than by slicing it first.
            freqs_cis = freqs_cis.index_select(
                0, get_q_positions_thd(thd_layout.cu_seqlens, seqlen, thd_layout.global_start)
            )
        q = q.clone()
        q = einops.rearrange(q, "s b ... -> b s ...")
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = einops.rearrange(q, "b s ... -> s b ...")

        q = rotate_activation(q)
        if self.use_fp8_qat:
            q = fp8_simulate_qat(q, 128)

        weights, _ = self.linear_weights_proj(x)
        softmax_scale = self.index_head_dim**-0.5
        weights = (weights * (self.index_n_heads**-0.5) * softmax_scale).float()

        # Balance the causal scoring work over contiguous CP (miles_plugins.models.cp_row_balance).
        # Replay records and replays this rank's own rows, so it keeps the unbalanced path.
        exchange = None
        if cp_size > 1 and cp_group is not None and not indexer_replay_manager.enabled:
            # Sent before the compressor, which overlaps it; the key all-gather below shares the
            # CP communicator, so it starts once the exchange is through.
            exchange = start_row_exchange(q, weights, thd_layout, cp_group)
        if exchange is not None:
            q = weights = None  # the scorer reads the exchanged rows

        pre_grouped = thd_layout is not None and thd_layout.compressed_group_ids is not None
        k = self.compressor(thd_layout.hidden_compact if pre_grouped else x, thd_layout)
        if k is None:
            # Nothing to score when no segment reaches compress_ratio; -1 leaves each query
            # on its sliding window. Only without CP: under CP the compressor gets compacted groups.
            assert exchange is None, "the compressor returned no keys while rows were in flight"
            return torch.full((bsz, seqlen, self.index_topk), -1, dtype=torch.int32, device=x.device)

        if cp_size > 1 and cp_group is not None:
            k = all_gather_cp(k, dim=0, cp_group=cp_group)
            if thd_layout is not None and thd_layout.seq_to_rank_row is not None:
                # Per-row bounds are sequence-major, so reorder the rank-major gather first.
                k = k.index_select(0, thd_layout.seq_to_rank_row.clamp(min=0).long())

        # Route through the indexer replay manager (flattened to [n_tokens, n_kv], matching the
        # record/replay convention and the MoE seam) so RL replay can pin the rollout's top-k picks.
        # get_topk_fn is transparent when disabled, as it is whenever rows were exchanged.
        topk_fn = indexer_replay_manager.get_topk_fn(get_dsa_topk_fn(self.topk_backend), return_probs=False)
        return topk_for_local_rows(
            q,
            weights,
            k,
            exchange,
            cp_rank=cp_group.rank() if cp_size > 1 and cp_group is not None else 0,
            thd_layout=thd_layout,
            compress_ratio=self.compress_ratio,
            index_topk=self.index_topk,
            topk_fn=topk_fn,
        )


def start_row_exchange(q, weights, thd_layout, cp_group) -> RowExchange | None:
    """Start sending this rank's indexer rows to their scoring ranks; None keeps contiguous CP."""
    plan = _row_balance_plan(q.shape[0], thd_layout, cp_group, q.device)
    if plan is None:
        return None
    return send_rows_to_scorers([q.detach(), weights.detach()], plan, cp_group)


def topk_for_local_rows(q, weights, k, exchange, *, cp_rank, thd_layout, compress_ratio, index_topk, topk_fn):
    """The top-k picks for this rank's rows, in local order.

    Without an ``exchange`` the rows in ``q`` and ``weights`` are scored here. With one, this rank
    scores the rows it received instead (``q`` and ``weights`` are unused) and each row's picks go
    back to the rank that owns it.
    """
    if exchange is None:
        rows = q.shape[0]
        row_start = thd_layout.global_start if thd_layout is not None else cp_rank * rows
        positions = torch.arange(row_start, row_start + rows, device=k.device)
    else:
        q, weights = exchange.wait()
        positions = exchange.plan.scored_positions
    topk_indices = indexer_topk(
        q,
        k,
        weights,
        positions,
        thd_layout,
        compress_ratio=compress_ratio,
        index_topk=index_topk,
        topk_fn=topk_fn,
    )
    # [batch, rows, topk]: exchange along the row dim
    return topk_indices if exchange is None else exchange.return_to_owners(topk_indices, dim=1)


def indexer_topk(q, k, weights, positions, thd_layout, *, compress_ratio, index_topk, topk_fn):
    """Score the query rows at global stream ``positions`` against their visible compressed keys.

    Args:
        q: [rows, batch, heads, head_dim] index queries of those rows
        k: [n_kv, batch, head_dim] every compressed key of the stream, sequence-major under THD
        weights: [rows, batch, heads] fp32 head weights
        positions: [rows] global stream positions of the rows
        thd_layout: packed-stream layout, or None when unpacked

    Returns:
        [batch, rows, min(index_topk, n_kv)] int32 compressed-key indices
    """
    if thd_layout is None:
        cu_ks = torch.zeros_like(positions, dtype=torch.int32)
        cu_ke = ((positions + 1) // compress_ratio).int()
    else:
        cu_ks, cu_ke = compress_bounds_at_positions(
            thd_layout.cu_seqlens, thd_layout.cu_seqlens_compressed, positions, ratio=compress_ratio
        )
    index_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)
    bsz, rows, n_kv = index_scores.shape
    topk_count = min(index_topk, n_kv)
    topk_indices = topk_fn(index_scores.reshape(bsz * rows, n_kv), topk_count)
    return topk_indices.reshape(bsz, rows, topk_count)


def _row_balance_plan(seqlen_local, thd_layout, cp_group, device) -> RowBalancePlan | None:
    """This micro-batch's balanced exchange; every CP rank derives the same one."""
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    if thd_layout is None:
        return _unpacked_row_balance_plan(seqlen_local * cp_size, cp_rank, cp_size, str(device))
    # The first CSA layer of a micro-batch builds it, right after the host sync for the lengths.
    key = ("row_balance_plan", seqlen_local)
    cache = thd_layout.micro_batch_cache
    if key not in cache:
        seq_lens = thd_layout.host_seq_lens(seqlen_local * cp_size)
        cache[key] = plan_causal_row_balance(seq_lens, cp_rank=cp_rank, cp_size=cp_size, device=device)
    return cache[key]


@lru_cache(maxsize=8)
def _unpacked_row_balance_plan(seqlen, cp_rank, cp_size, device):
    """Each batch row of an unpacked sample is one sequence, so the plan depends only on the shape."""
    return plan_causal_row_balance((seqlen,), cp_rank=cp_rank, cp_size=cp_size, device=device)
