"""Select causal attention regions with Step5's frozen DSA indexer.

Each query keeps up to 512 earlier eight-token regions per KV group plus its own.
Selection is recomputed under `no_grad`; the frozen weights are checkpointed and synced
like every other weight. `indexer_ops` documents the remaining scoring-rounding differences.
"""

import json
import os
from collections.abc import Mapping

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint
from safetensors import safe_open

from miles_plugins.models.step5.indexer_ops import HEADS_PER_GROUP, INDEXER_HEAD_DIM
from miles_plugins.models.step5.region_selection import RegionSelection
from miles_plugins.models.step5.region_selector import (
    CHECKPOINT_NAMES,
    INDEXER_NUM_HEADS,
    SCORE_CHUNK_ELEMENTS,
    IndexerWeights,
    select_regions,
)

# FP32 in the checkpoint and kept FP32 here; the arithmetic rounds them to BF16 as SGLang's load does.
FP32_PARAMS = ("w_weight", "q_norm_weight", "k_norm_weight", "k_norm_bias", "ssmax_s")
# Split by provider group, whose rows are contiguous, across TP ranks; the rest are replicated.
TP_SHARDED_PARAMS = ("q_weight", "w_weight")


class Step5Indexer(MegatronModule):
    """Frozen DSA indexer of one full-attention layer, sharded like the layer's KV groups.

    Provider group ``g`` (indexer heads ``4g .. 4g + 3``) selects for KV group ``g``, so a TP rank holds
    the q/w rows of its own groups while k, z, the norms and ``ssmax_s`` are replicated.
    """

    def __init__(
        self,
        config: TransformerConfig,
        tp_group: torch.distributed.ProcessGroup | None = None,
        score_chunk_elements: int = SCORE_CHUNK_ELEMENTS,
    ) -> None:
        super().__init__(config=config)
        tp_size = 1 if tp_group is None else tp_group.size()
        assert config.num_query_groups * HEADS_PER_GROUP == INDEXER_NUM_HEADS, "one provider group per KV group"
        assert config.num_query_groups % tp_size == 0, "each TP rank must hold whole KV groups"
        assert tp_group is not None or not config.sequence_parallel, "sequence parallelism needs the TP group"
        self.tp_group = tp_group
        self.score_chunk_elements = score_chunk_elements
        self.sequence_parallel = config.sequence_parallel
        self.eps = config.layernorm_epsilon
        self.num_local_groups = config.num_query_groups // tp_size
        local_heads = self.num_local_groups * HEADS_PER_GROUP
        hidden = config.hidden_size
        self.q_weight = _frozen(torch.zeros(local_heads * INDEXER_HEAD_DIM, hidden, dtype=torch.bfloat16))
        self.k_weight = _frozen(torch.zeros(INDEXER_HEAD_DIM, hidden, dtype=torch.bfloat16))
        self.z_weight = _frozen(torch.zeros(INDEXER_HEAD_DIM, hidden, dtype=torch.bfloat16))
        self.w_weight = _frozen(torch.zeros(local_heads, hidden))
        self.q_norm_weight = _frozen(torch.zeros(INDEXER_HEAD_DIM))
        self.k_norm_weight = _frozen(torch.zeros(INDEXER_HEAD_DIM))
        self.k_norm_bias = _frozen(torch.zeros(INDEXER_HEAD_DIM))
        # a scalable-softmax scale no kernel reads; SGLang loads it with the indexer, so exports must carry it
        self.ssmax_s = _frozen(torch.zeros(config.num_attention_heads))
        for name in FP32_PARAMS:
            mark_keep_in_fp32(getattr(self, name))
        for name in TP_SHARDED_PARAMS:
            set_tensor_model_parallel_attributes(getattr(self, name), is_parallel=True, dim=0, stride=1)

    @torch.no_grad()
    def select_regions(self, hidden_states: torch.Tensor, packed_seq_params: PackedSeqParams) -> RegionSelection:
        """The regions SGLang selects for every token of a THD pack, per local KV group.

        ``hidden_states`` is the attention input, the input norm's BF16 output: ``[T, 1, hidden]``, or
        this rank's sequence-parallel shard of it.
        """
        weights = IndexerWeights(
            q_weight=self.q_weight,
            k_weight=self.k_weight,
            z_weight=self.z_weight,
            w_weight=self.w_weight,
            q_norm_weight=self.q_norm_weight,
            k_norm_weight=self.k_norm_weight,
            k_norm_bias=self.k_norm_bias,
        )
        return select_regions(
            self._full_sequence(hidden_states),
            packed_seq_params.cu_seqlens_q,
            weights,
            eps=self.eps,
            score_chunk_elements=self.score_chunk_elements,
        )

    def load_checkpoint_layer(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Copy one layer's full checkpoint tensors, keyed by ``CHECKPOINT_NAMES`` values, into this rank."""
        tp_size = 1 if self.tp_group is None else self.tp_group.size()
        tp_rank = 0 if self.tp_group is None else self.tp_group.rank()
        for name, checkpoint_name in CHECKPOINT_NAMES.items():
            tensor = tensors[checkpoint_name]
            if name in TP_SHARDED_PARAMS:
                tensor = tensor.chunk(tp_size)[tp_rank]
            param = getattr(self, name)
            assert tensor.shape == param.shape, f"{checkpoint_name}: checkpoint {tuple(tensor.shape)}, {param.shape}"
            with torch.no_grad():
                param.copy_(tensor)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        state_dict = self.state_dict(prefix="", keep_vars=True)
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            tensor_parallel_layers_axis_map={name: 0 for name in TP_SHARDED_PARAMS},
            sharded_offsets=sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def _full_sequence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.ndim == 3 and hidden_states.shape[1] == 1, "the indexer reads a THD pack"
        assert hidden_states.dtype == torch.bfloat16, "the indexer projections read BF16 hidden states"
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
            )
        return hidden_states.squeeze(1)


def read_checkpoint_layer(checkpoint: str, layer_index: int) -> dict[str, torch.Tensor]:
    """One layer's indexer tensors from an HF safetensors checkpoint, keyed by ``CHECKPOINT_NAMES`` values."""
    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    tensors = {}
    for checkpoint_name in CHECKPOINT_NAMES.values():
        name = f"model.layers.{layer_index}.self_attn.{checkpoint_name}"
        with safe_open(os.path.join(checkpoint, weight_map[name]), framework="pt") as f:
            tensors[checkpoint_name] = f.get_tensor(name)
    return tensors


def _frozen(tensor: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(tensor, requires_grad=False)
