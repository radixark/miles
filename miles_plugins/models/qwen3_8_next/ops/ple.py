"""Qwen3.8-Next PLE: n-gram hashing, token-id side channel, frozen table, layer.

Hashing is Megatron-free (unit-tested vs sglang). The side channel has no silent
default: PLE with nothing published raises. The 102GB table is host-resident,
TP-row-sharded, and deliberately NOT a checkpointed parameter.
"""

import logging
import threading

import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles.kernels.embedding.ple_gate_conv import ple_gate_conv
from miles.kernels.embedding.ple_gather import gather_ple_rows
from miles_plugins.models.qwen3_8_next.ops.ngram import (
    ngram_hash_ids,
    ple_table_shards,
    read_ple_hash_tensors,
    read_ple_shard_into,
)

logger = logging.getLogger(__name__)


_state = threading.local()


def current_ple_batch():
    batch = getattr(_state, "batch", None)
    if batch is None:
        raise RuntimeError(
            "PLE ran with no n-gram ids published; publish_ple_batch(ngram_ids, "
            "cu_seqlens) must run before the forward (model_provider hooks do this). "
            "Skipping PLE silently would change the logits without failing."
        )
    return batch


def publish_ple_batch(ngram_ids: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> None:
    """Pre-hook/post-hook publication (no scope survives between paired hooks)."""
    _state.batch = (ngram_ids, cu_seqlens)


def clear_ple_batch() -> None:
    _state.batch = None


class Qwen38NextFrozenNGramEmbedding(MegatronModule):
    """Frozen host-resident n-gram table, row-sharded over TP."""

    def __init__(self, config: TransformerConfig, layer_number: int, tp_group=None):
        super().__init__(config)
        self.layer_number = layer_number
        self.hf_layer_index = layer_number - 1
        heads = (config.qwen3_8_next_ngram_size - 1) * config.qwen3_8_next_heads_per_ngram
        self.embedding_dim = config.qwen3_8_next_ple_embed_dim // heads
        self.num_shards = config.qwen3_8_next_split_ngram_parts
        self.ngram_size = config.qwen3_8_next_ngram_size
        self._heads_per_ngram = config.qwen3_8_next_heads_per_ngram
        self.eos_token_id = getattr(config, "qwen3_8_next_eos_token_id", 0)
        self.tp_group = tp_group

        tp_size = tp_group.size() if tp_group is not None else 1
        tp_rank = tp_group.rank() if tp_group is not None else 0
        if self.num_shards % tp_size:
            raise ValueError(
                f"split_ngram_parts={self.num_shards} must be divisible by the "
                f"tensor-parallel size {tp_size}: shards are assigned whole so that "
                "changing TP only changes which of the fixed HF shards a rank reads."
            )
        self.shards_per_rank = self.num_shards // tp_size
        self.shard_ids = list(range(tp_rank * self.shards_per_rank, (tp_rank + 1) * self.shards_per_rank))

        self.rows_per_shard = getattr(config, "qwen3_8_next_ngram_rows_per_shard", None)
        if self.rows_per_shard is None:
            hf = getattr(config, "qwen3_8_next_hf_checkpoint", None)
            if hf is None:
                raise ValueError(
                    "PLE shard height unknown: neither qwen3_8_next_ngram_rows_per_shard "
                    "nor qwen3_8_next_hf_checkpoint is set on the config; deriving it "
                    "from ngram_vocab_size_base drifts 12 rows/shard."
                )
            self.rows_per_shard = int(ple_table_shards(hf, self.hf_layer_index)[0].shape[0])
        self.row_start = self.shard_ids[0] * self.rows_per_shard
        self.row_end = (self.shard_ids[-1] + 1) * self.rows_per_shard

        self.table = torch.empty(
            (self.row_end - self.row_start, self.embedding_dim),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        self._loaded = False
        self._hf_checkpoint = getattr(config, "qwen3_8_next_hf_checkpoint", None)

        self.register_buffer("layer_multipliers", torch.zeros(self.ngram_size, dtype=torch.long), persistent=False)
        self.register_buffer("ngram_heads_vocab_sizes", torch.zeros(heads, dtype=torch.long), persistent=False)
        self.register_buffer("ngram_heads_offsets", torch.zeros(heads, dtype=torch.long), persistent=False)

        if self._hf_checkpoint is not None:
            self.load_metadata_from_hf(self._hf_checkpoint)

    def load_metadata_from_hf(self, hf_checkpoint: str) -> None:
        """Load the three integer tensors that parameterise the hash."""
        for name, tensor in read_ple_hash_tensors(hf_checkpoint, self.hf_layer_index).items():
            getattr(self, name).copy_(tensor)

    def load_from_hf(self, hf_checkpoint: str) -> None:
        """Fill the table from the HF safetensors."""
        shards = ple_table_shards(hf_checkpoint, self.hf_layer_index)
        for i, shard_id in enumerate(self.shard_ids):
            rows = i * self.rows_per_shard
            shard = shards[shard_id]
            read_ple_shard_into(shard, self.table[rows : rows + shard.shape[0]])

        self._loaded = True

    def compute_ngram_ids(self, contexts: Tensor) -> Tensor:
        """``[T, ngram_size]`` sliding windows -> ``[T, n_heads]`` row ids."""
        return ngram_hash_ids(
            contexts,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            self.ngram_size,
            self._heads_per_ngram,
            self.eos_token_id,
        )

    def forward(self, ids: Tensor) -> Tensor:
        """``[T, n_heads]`` int64 -> ``[T, n_heads * embedding_dim]`` bf16."""
        if not self._loaded:
            if self._hf_checkpoint is None:
                raise RuntimeError(
                    "PLE table never loaded and config.qwen3_8_next_hf_checkpoint "
                    "is unset; a zero table changes logits without failing."
                )
            self.load_from_hf(self._hf_checkpoint)
        assert self.table.device.type == "cpu", (
            f"PLE table must stay on the host, found {self.table.device}. Something "
            "moved it -- most likely by registering it as a buffer again."
        )
        assert self.table.is_pinned(), "PLE table lost its pinning"
        rows = gather_ple_rows(self.table, ids, self.row_start, self.row_end)
        out = rows.flatten(start_dim=-2)
        if self.tp_group is not None and self.tp_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.tp_group)
        return out


class Qwen38NextPLE(MegatronModule):
    """PLE increment for the hyper-connection state."""

    def __init__(self, config: TransformerConfig, layer_number: int, tp_group=None):
        super().__init__(config)
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.embed_dim = config.qwen3_8_next_ple_embed_dim
        wide = self.n * self.hidden_size

        self.ple_embedding = Qwen38NextFrozenNGramEmbedding(config, layer_number=layer_number, tp_group=tp_group)

        self.key_proj = TELinear(
            self.embed_dim,
            wide,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.value_proj = TELinear(
            self.embed_dim,
            self.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        dtype = config.params_dtype
        self.norm_key = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.norm_query = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.norm_conv = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))

        kernel = config.qwen3_8_next_ple_conv_kernel_size
        # sglang dilates the short conv by the n-gram size; the checkpoint has no separate field
        self.conv_dilation = config.qwen3_8_next_ngram_size
        self.conv1d_weight = torch.nn.Parameter(torch.zeros(wide, 1, kernel, dtype=dtype))

        for p in (self.norm_key, self.norm_query, self.norm_conv, self.conv1d_weight):
            p.sequence_parallel = config.sequence_parallel

    def forward(self, hc_state: Tensor, ngram_ids: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
        """``hc_state`` ``[T, n*C]``, ``ngram_ids`` ``[T, n_heads]`` -> increment ``[T, n*C]``."""
        if hc_state.dim() != 2 or ngram_ids.dim() != 2:
            raise RuntimeError(
                "PLE takes a flat token axis: expected hc_state [T, n*C] and ngram_ids "
                f"[T, heads], got {tuple(hc_state.shape)} and {tuple(ngram_ids.shape)}"
            )
        if hc_state.shape[0] != ngram_ids.shape[0]:
            raise RuntimeError(f"PLE token counts differ: state {hc_state.shape[0]} vs ids {ngram_ids.shape[0]}")

        embeddings = self.ple_embedding(ngram_ids)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)

        if hc_state.shape[-1] != self.n * self.hidden_size:
            raise RuntimeError(
                "PLE hidden size does not match the hyper-connection layout: expected "
                f"{self.n * self.hidden_size}, got {hc_state.shape[-1]}"
            )

        return ple_gate_conv(
            hc_state,
            key,
            value,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            self.conv1d_weight,
            self.n,
            self.norm_eps,
            self.conv_dilation,
            cu_seqlens,
        )
