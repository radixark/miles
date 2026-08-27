"""GLM-5.3 DSA attention: GLM-5's absorbed sparse MLA minus rope, plus kpool.

Differences from ``DSAMLASelfAttention``:

* ``qk_rope_head_dim == 0`` -- no rope anywhere in the MLA path or the indexer;
  the softmax scale stays plain ``1/sqrt(qk_head_dim)`` (no yarn mscale).
* The lighting indexer's per-token top-k is replaced by the pooled-key (kpool)
  selection from ``ops/kpool_indexer.py``; the two compression parameters
  (``index_kpool_compress_gate`` bf16, ``index_kpool_compress_ape`` fp32) live on
  this module so the checkpoint round-trips, and stay frozen like the rest of
  the indexer.
* The tilelang SparseMLA kernels are compiled for dim 512 + tail 64, so q/kv are
  padded with a zero 64-wide tail -- numerically exact because a zero q-tail
  times a zero k-tail adds nothing to the logits and the softmax scale is passed
  explicitly.
* Indexer-replay streams are the DSA layers only (11 of 45), so the replay
  stream index is this layer's ordinal among ``full_attn_layers``, not
  ``layer_number - 1`` as on GLM-5 where every layer has an indexer.
"""

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction as WeightLinearFunction

from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.glm5.glm5 import DSAMLASelfAttention
from miles_plugins.models.glm5.ops.sparse_mla import SparseMLA
from miles_plugins.models.glm5_next.ops.kpool_indexer import build_pooled_keys, kpool_select_topk, pool_boundaries

_SPARSE_MLA_TAIL_DIM = 64


class Glm5NextDSAAttention(DSAMLASelfAttention):
    """No-rope absorbed sparse MLA with pooled-key indexer selection."""

    def __init__(
        self,
        config,
        submodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        topk_backend: str = "torch",
        is_mtp_layer: bool = False,
        cp_comm_type: str | None = None,
        model_comm_pgs=None,
        pg_collection=None,
        name: str | None = None,
    ):
        assert config.qk_pos_emb_head_dim == 0, "GLM-5.3 DSA skips rope; qk_pos_emb_head_dim must be 0"
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            topk_backend=topk_backend,
            is_mtp_layer=is_mtp_layer,
            cp_comm_type=cp_comm_type,
            model_comm_pgs=model_comm_pgs,
            pg_collection=pg_collection,
            name=name,
        )
        self.softmax_scale = self.q_head_dim**-0.5
        self.index_topk = int(getattr(config, "index_topk", 2048))
        self.index_kpool = int(getattr(config, "index_kpool", 4))

        self.index_kpool_compress_gate = torch.nn.Parameter(torch.zeros(config.index_head_dim, config.hidden_size))
        self.index_kpool_compress_ape = mark_keep_in_fp32(
            torch.nn.Parameter(torch.zeros(self.index_kpool, config.index_head_dim, dtype=torch.float32))
        )
        self.index_kpool_compress_gate.requires_grad_(False)
        self.index_kpool_compress_ape.requires_grad_(False)

        if indexer_replay_manager.enabled:
            full_attn_layers = list(config.glm5_next_full_attn_layers)
            self.indexer_replay.stream_idx = full_attn_layers.index(self.layer_number - 1)

    def get_absorb_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        assert packed_seq_params is not None

        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        q_compressed = q_compressed.squeeze(1)

        kv_compressed, _ = self.linear_kv_down_proj(hidden_states)
        if self.config.sequence_parallel:
            kv_compressed = gather_from_sequence_parallel_region(kv_compressed)
        kv_compressed = self.kv_layernorm(kv_compressed)

        q_compressed = self.q_layernorm(q_compressed)
        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

        w_kc, w_vc = self.linear_kv_up_proj.weight.unflatten(
            0,
            (-1, self.config.qk_head_dim + self.config.v_head_dim),
        ).split([self.config.qk_head_dim, self.config.v_head_dim], dim=1)

        query = torch.einsum("thd,hdm->thm", q, w_kc)

        kv_compressed = torch.nn.functional.rms_norm(
            kv_compressed.float(),
            normalized_shape=(kv_compressed.shape[-1],),
            weight=self.linear_kv_up_proj.layer_norm_weight.float(),
            eps=self.config.layernorm_epsilon,
        ).to(kv_compressed.dtype)
        kv_compressed = gather_from_sequence_parallel_region(
            kv_compressed, group=parallel_state.get_context_parallel_group()
        )

        query = query.contiguous()
        key = kv_compressed.contiguous()

        q_compressed = q_compressed.detach()
        hidden_states = hidden_states.detach()

        index_q, _ = self.wq_b(q_compressed)
        index_q = index_q.view(*index_q.size()[:-1], self.config.index_num_attention_heads, self.config.index_head_dim)
        if self.config.sequence_parallel:
            index_q = gather_from_sequence_parallel_region(index_q)

        index_k, _ = self.wk(hidden_states)
        index_k = self.k_norm(index_k.squeeze(1).float()).bfloat16()
        if self.config.sequence_parallel:
            index_k = gather_from_sequence_parallel_region(index_k)
        index_k = gather_from_sequence_parallel_region(index_k, group=parallel_state.get_context_parallel_group())

        gate_score = F.linear(hidden_states.squeeze(1), self.index_kpool_compress_gate)
        if self.config.sequence_parallel:
            gate_score = gather_from_sequence_parallel_region(gate_score)
        gate_score = gather_from_sequence_parallel_region(
            gate_score, group=parallel_state.get_context_parallel_group()
        )

        head_weights = WeightLinearFunction.apply(hidden_states, self.weights_proj.weight, None, torch.float32)
        head_weights = head_weights.squeeze(1) * (
            (self.config.index_num_attention_heads**-0.5) * (self.config.index_head_dim**-0.5)
        )
        if self.config.sequence_parallel:
            head_weights = gather_from_sequence_parallel_region(head_weights)

        return query, key, w_vc, index_q, index_k, head_weights, gate_score

    def _kpool_select(self, index_q, index_k, head_weights, gate_score, packed_seq_params):
        if parallel_state.get_context_parallel_world_size() > 1:
            raise NotImplementedError("GLM-5.3 kpool indexer selection does not support context parallelism yet.")
        cu_seqlens = packed_seq_params.cu_seqlens_kv
        pool_cu_seqlens = pool_boundaries(cu_seqlens, self.index_kpool)
        pooled_k = build_pooled_keys(
            index_k,
            gate_score,
            self.index_kpool_compress_ape,
            cu_seqlens,
            self.index_kpool,
        )
        return kpool_select_topk(
            index_q=index_q,
            pooled_k=pooled_k,
            head_weights=head_weights,
            cu_seqlens=cu_seqlens,
            pool_cu_seqlens=pool_cu_seqlens,
            index_topk=self.index_topk,
            kpool=self.index_kpool,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        router_token_masks=None,
        loss_mask=None,
    ):
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into GLM-5.3 DSA."
        assert attention_bias is None, "Attention bias should not be passed into GLM-5.3 DSA."

        query, key, w_vc, index_q, index_k, head_weights, gate_score = self.get_absorb_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        topk_indices = self._kpool_select(index_q, index_k, head_weights, gate_score, packed_seq_params)

        query = F.pad(query, (0, _SPARSE_MLA_TAIL_DIM)).contiguous()
        key = F.pad(key, (0, _SPARSE_MLA_TAIL_DIM)).contiguous()

        core_attn_out, _ = SparseMLA.apply(query, key, topk_indices, self.softmax_scale)
        core_attn_out = torch.einsum("thm,hdm->thd", core_attn_out, w_vc)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        output, bias = self.linear_proj(core_attn_out)
        return output, bias
