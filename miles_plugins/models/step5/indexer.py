"""Select causal attention regions with Step5's frozen DSA indexer.

Each query keeps up to 512 earlier eight-token regions per KV group plus its own.
Selection runs under `no_grad`; the frozen weights are checkpointed and synced
like every other weight. `indexer_ops` documents the remaining scoring-rounding differences.
Attention can reuse cached block IDs to skip the frozen Indexer during backward recomputation.
"""

import itertools
import json
import os
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint
from safetensors import safe_open

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
from miles_plugins.models.step5.layers import FULL_ATTENTION_ROPE
from miles_plugins.models.step5.region_selection import MAX_SELECTED_REGIONS, REGION_SIZE, TOP_REGIONS, RegionSelection

INDEXER_NUM_HEADS = 16
INDEXER_ROPE = Llama3Rope(
    theta=FULL_ATTENTION_ROPE.theta,
    factor=1.0,
    low_freq_factor=1.0,
    high_freq_factor=32.0,
    original_max_position_embeddings=1 << 20,
)

# Megatron parameter -> its checkpoint tensor under ``model.layers.N.self_attn``.
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
# FP32 in the checkpoint and kept FP32 here; the arithmetic rounds them to BF16 as SGLang's load does.
FP32_PARAMS = ("w_weight", "q_norm_weight", "k_norm_weight", "k_norm_bias", "ssmax_s")
# Split by provider group, whose rows are contiguous, across TP ranks; the rest are replicated.
TP_SHARDED_PARAMS = ("q_weight", "w_weight")

# Positions before this one have at most TOP_REGIONS earlier regions and keep them all, unscored.
_FIRST_SCORED_POSITION = (TOP_REGIONS + 1) * REGION_SIZE
# Default bound on the FP32 dot products alive at once, [query rows, local heads, regions]: 64 MiB.
SCORE_CHUNK_ELEMENTS = 1 << 24
# Queries are prepared this many rows at a time, then scored in memory-bounded chunks.
_QUERY_BLOCK_ROWS = 2048
# Below 16 rows cuBLAS runs the BF16 projections as a GEMV that rounds differently; SGLang's prefill
# projects whole prompts, so query projections always cover at least this many rows.
_MIN_GEMM_ROWS = 16


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
        tokens = self._full_sequence(hidden_states)
        cu_seqlens = packed_seq_params.cu_seqlens_q
        bounds = cu_seqlens.tolist()
        # Rows outside every sequence would stay unwritten, and the sparse kernel reads int32 boundaries.
        assert cu_seqlens.dtype == torch.int32 and bounds[0] == 0 and bounds[-1] == tokens.shape[0], (
            f"the pack's sequences must cover its {tokens.shape[0]} tokens: {cu_seqlens.dtype} cu_seqlens "
            f"from {bounds[0]} to {bounds[-1]}"
        )
        region_ids = torch.empty(
            (tokens.shape[0], self.num_local_groups, MAX_SELECTED_REGIONS), dtype=torch.int32, device=tokens.device
        )
        for start, end in itertools.pairwise(bounds):
            self._select_sequence(tokens[start:end], out=region_ids[start:end])
        return RegionSelection(region_ids=region_ids, cu_seqlens=cu_seqlens)

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

    def _select_sequence(self, tokens: torch.Tensor, *, out: torch.Tensor) -> None:
        """Fill ``out`` ``[L, local groups, 513]`` with one sequence's selection; ``tokens`` is ``[L, hidden]``."""
        length = tokens.shape[0]
        out.copy_(_unscored_region_ids(length, device=tokens.device)[:, None])
        if length <= _FIRST_SCORED_POSITION:
            return
        cos, sin = rope_cos_sin(INDEXER_ROPE, length, tokens.device)
        region_keys = self._region_keys(tokens, cos, sin)
        for block in _row_slices(_FIRST_SCORED_POSITION, length, _QUERY_BLOCK_ROWS):
            query, weights = self._queries(tokens, block, cos[block], sin[block])
            self._select_block(query, weights, region_keys, first_row=block.start, out=out)

    def _select_block(self, query, weights, region_keys, *, first_row: int, out: torch.Tensor) -> None:
        """Score one block of queries against the regions before them, in memory-bounded row chunks."""
        local_heads = self.num_local_groups * HEADS_PER_GROUP
        rows_per_chunk = max(1, self.score_chunk_elements // (local_heads * region_keys.shape[0]))
        for local in _row_slices(0, query.shape[0], rows_per_chunk):
            rows = slice(first_row + local.start, first_row + local.stop)
            scores = score_regions(query[local], weights[local], region_keys[: (rows.stop - 1) // REGION_SIZE])
            candidates = torch.arange(rows.start, rows.stop, device=query.device) // REGION_SIZE
            out[rows, :, :TOP_REGIONS] = top_regions(scores, candidates)

    def _region_keys(self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """E4M3 ``[complete regions, 256]``: each region's compressed index key."""
        complete = tokens.shape[0] // REGION_SIZE * REGION_SIZE
        tokens = tokens[:complete]
        key = norm_rope_key(
            F.linear(tokens, self.k_weight),
            self.k_norm_weight,
            self.k_norm_bias,
            cos[:complete],
            sin[:complete],
            self.eps,
        )
        logits = F.linear(tokens, self.z_weight)
        return compress_regions(
            key.view(-1, REGION_SIZE, INDEXER_HEAD_DIM), logits.view(-1, REGION_SIZE, INDEXER_HEAD_DIM)
        )

    def _queries(self, tokens: torch.Tensor, rows: slice, cos: torch.Tensor, sin: torch.Tensor):
        """For ``tokens[rows]``: E4M3 queries ``[rows, groups, 4, 256]`` and FP32 head weights ``[rows, groups, 4]``."""
        first = max(0, min(rows.start, rows.stop - _MIN_GEMM_ROWS))
        query = F.linear(tokens[first : rows.stop], self.q_weight)[rows.start - first :]
        count = query.shape[0]
        query = to_e4m3(
            norm_rope_query(query.view(count, -1, INDEXER_HEAD_DIM), self.q_norm_weight, cos, sin, self.eps)
        )
        weights = batch_invariant_linear(tokens[rows], self.w_weight.to(torch.bfloat16))
        groups = (count, self.num_local_groups, HEADS_PER_GROUP)
        return query.view(*groups, INDEXER_HEAD_DIM), weights.float().view(groups) * WEIGHT_SCALE


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


def _row_slices(start: int, stop: int, step: int) -> list[slice]:
    return [slice(row, min(row + step, stop)) for row in range(start, stop, step)]


def _frozen(tensor: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(tensor, requires_grad=False)


def _unscored_region_ids(length: int, *, device: torch.device) -> torch.Tensor:
    """int32 ``[L, 513]``: regions ``0 .. min(c, 512) - 1``, then the token's own region ``c``, then -1.

    Complete for positions whose ``c = p // 8`` is at most TOP_REGIONS; later rows keep ``c`` in the
    last slot and get their scored regions written over the rest.
    """
    current = torch.arange(length, device=device)[:, None] // REGION_SIZE
    history = current.clamp(max=TOP_REGIONS)
    slot = torch.arange(MAX_SELECTED_REGIONS, device=device)
    return torch.where(slot < history, slot, torch.where(slot == history, current, -1)).to(torch.int32)
