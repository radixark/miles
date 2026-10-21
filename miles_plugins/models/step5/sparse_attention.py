"""Step5 full attention over regions selected by the frozen DSA indexer.

Each KV group keeps up to 512 earlier eight-token regions plus the query's own.
Sequences of at most 4,104 tokens therefore use the dense causal mask. Optional
selection reuse skips the frozen Indexer during full-layer recomputation.
"""

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import is_checkpointing
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.kernels.attention.region_sparse import sparse_region_attention
from miles_plugins.models.step5.indexer import Step5Indexer
from miles_plugins.models.step5.layers import FULL_ATTENTION_ROPE, Step5Attention
from miles_plugins.models.step5.region_selection import RegionSelection

# The kernel's tiles are 8 query tokens times one KV group's heads, tuned for Step5's 16.
_HEADS_PER_GROUP = 16


class Step5SparseCoreAttention(MegatronModule):
    """The core attention of a full-attention layer: ``sparse_region_attention`` over an explicit selection.

    It fills Megatron's core-attention slot, so it takes that slot's constructor arguments. The selection
    is not among the core-attention inputs Megatron passes, so ``Step5SparseAttention`` calls it directly.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        softmax_scale: float | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=config)
        assert attn_mask_type == AttnMaskType.causal, "the kernel masks the query's own region causally"
        assert config.num_attention_heads // config.num_query_groups == _HEADS_PER_GROUP, "the kernel's tile size"
        assert config.params_dtype == torch.bfloat16, "the kernel is validated for BF16 only"
        # TE's default, which the sliding-window layers' dense attention uses
        self.softmax_scale = config.kv_channels**-0.5 if softmax_scale is None else softmax_scale

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, selection: RegionSelection
    ) -> torch.Tensor:
        """Attend over the pack with `query` `[T, heads, 192]` and `key`/`value` `[T, groups, 192]`."""
        return sparse_region_attention(
            query, key, value, selection.region_ids, selection.cu_seqlens, self.softmax_scale
        )


class Step5SparseAttention(Step5Attention):
    """A full-attention layer: the full-attention RoPE, and attention over the indexer's selected regions."""

    rope = FULL_ATTENTION_ROPE

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: int | None = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
        reuse_indexer_on_recompute: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        assert config.head_wise_attn_gate and not config.attention_output_gate, "Step5's head-wise gate only"
        assert not (
            self.checkpoint_core_attention
            or self.offload_qkv_linear
            or self.offload_core_attention
            or self.offload_attn_proj
        ), "the forward below has neither selective core-attention recompute nor activation offloading"
        # The bridge and weight sync name its tensors ``self_attention.indexer.*``.
        self.indexer = Step5Indexer(config, tp_group=self.pg_collection.tp)
        self.reuse_indexer_on_recompute = reuse_indexer_on_recompute

    def _select_regions(self, hidden_states: torch.Tensor, packed_seq_params: PackedSeqParams) -> RegionSelection:
        if not (self.reuse_indexer_on_recompute and self.training and is_checkpointing()):
            return self.indexer.select_regions(hidden_states, packed_seq_params)
        # Megatron's checkpoint disables gradients in forward and enables them in recomputation.
        if torch.is_grad_enabled():
            return packed_seq_params._step5_indexer_selections.pop(self.layer_number)

        selection = self.indexer.select_regions(hidden_states, packed_seq_params)
        # The packed object belongs to this microbatch; layers use separate keys.
        cache = getattr(packed_seq_params, "_step5_indexer_selections", {})
        cache[self.layer_number] = selection
        packed_seq_params._step5_indexer_selections = cache
        return selection

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run packed training attention with per-layer RoPE, sparse selection, and the head-wise gate.

        Follows Megatron `Attention.forward` at `a84b10547` for THD training. The layer's
        RoPE replaces the model-level embedding; THD boundaries supply the causal mask.
        """
        assert inference_context is None and inference_params is None, "Step5 sparse attention has no inference path"
        assert packed_seq_params is not None and packed_seq_params.qkv_format == "thd", "the indexer reads THD packs"
        assert rotary_pos_emb is not None, "Step5 rotates q and k in every layer"
        selection = self._select_regions(hidden_states, packed_seq_params)

        query, key, value, gate = self.get_query_key_value_tensors(hidden_states, head_wise_gate=True)
        # value is still a strided QKV projection view; the sparse kernel requires contiguous storage
        query, key, value = query.squeeze(1), key.squeeze(1), value.squeeze(1).contiguous()
        freqs = self.rotary_pos_emb(packed_seq_params.max_seqlen_q)
        query = apply_rotary_pos_emb(
            query,
            freqs,
            config=self.config,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
            mscale=self._yarn_concentration_factor,
            cp_group=self.pg_collection.cp,
            max_seqlen=packed_seq_params.max_seqlen_q,
        )
        key = apply_rotary_pos_emb(
            key,
            freqs,
            config=self.config,
            cu_seqlens=packed_seq_params.cu_seqlens_kv,
            mscale=self._yarn_concentration_factor,
            cp_group=self.pg_collection.cp,
            max_seqlen=packed_seq_params.max_seqlen_kv,
        )

        core_attn_out = self.core_attention(query, key, value, selection)
        # match Megatron's rounding boundary before multiplying by the gate
        core_attn_out = core_attn_out * torch.sigmoid(gate.squeeze(1)[..., None].float()).to(core_attn_out.dtype)
        return self.linear_proj(core_attn_out.reshape(core_attn_out.shape[0], 1, -1))
