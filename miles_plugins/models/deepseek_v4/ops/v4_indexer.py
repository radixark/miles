import einops
import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.deepseek_v4.ops.compressor import DeepSeekV4Compressor
from miles_plugins.models.deepseek_v4.ops.cp_utils import all_gather_cp, get_freqs_cis_for_cp
from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd import (
    _make_causal_cu_seqlens,
    batched_indexer_fwd,
)
from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat
from miles_plugins.models.deepseek_v4.ops.rope import apply_rotary_emb, wrapped_precompute_freqs_cis
from miles_plugins.models.deepseek_v4.ops.utils import rotate_activation


class V4Indexer(MegatronModule):
    """DSA Indexer for DeepSeek-V4 C4 layers."""

    def __init__(self, config: TransformerConfig, pg_collection=None):
        super().__init__(config=config)

        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
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

    def forward(self, x: torch.Tensor, qr: torch.Tensor, mask=None, packed_seq_params=None):
        """Forward pass.

        Args:
            x:  hidden states [seqlen, batch, hidden_size]
            qr: low-rank query [seqlen, batch, q_lora_rank]
            mask: unused (causal mask generated internally via cu_seqlens)
            packed_seq_params: unused

        Returns:
            topk_indices: [batch, seqlen, index_topk] int64
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
        rope_base = self.config.dsv4_compress_rope_theta if self.compress_ratio else self.config.rotary_base
        freqs_cis = wrapped_precompute_freqs_cis(
            self.config, self.rope_head_dim, rope_base, False, seqlen * cp_size, x.device
        )
        freqs_cis = get_freqs_cis_for_cp(freqs_cis, seqlen, cp_size, cp_group, stride=1)
        q = q.clone()
        q = einops.rearrange(q, "s b ... -> b s ...")
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = einops.rearrange(q, "b s ... -> s b ...")

        q = rotate_activation(q)
        if self.use_fp8_qat:
            q = fp8_simulate_qat(q, 128)

        k = self.compressor(x)

        weights, _ = self.linear_weights_proj(x)
        softmax_scale = self.index_head_dim**-0.5
        weights = weights * (self.index_n_heads**-0.5) * softmax_scale

        if cp_size > 1 and cp_group is not None:
            k = all_gather_cp(k, dim=0, cp_group=cp_group)

        seqlen_global = seqlen * cp_size
        seqlen_kv = k.shape[0]
        cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_global, seqlen_kv, self.compress_ratio, q.device)
        # cu_seqlens are for global positions; slice to local query positions
        if cp_size > 1 and cp_group is not None:
            cp_rank = cp_group.rank()
            cu_ks = cu_ks[cp_rank * seqlen : (cp_rank + 1) * seqlen]
            cu_ke = cu_ke[cp_rank * seqlen : (cp_rank + 1) * seqlen]
        index_scores = batched_indexer_fwd(q, k, weights.float(), cu_ks, cu_ke)

        topk_count = min(self.index_topk, index_scores.size(-1))
        topk_indices = index_scores.topk(topk_count, dim=-1)[1]

        return topk_indices
