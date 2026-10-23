"""Qwen3.8-Next full-attention layer: QSA indexer + sparse attention.

Each full-attention layer keeps ``indexer_budget`` key tokens per query and attends
only those.
"""

import torch
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from miles.kernels.attention.qsa import qsa_block_sparse_attention, qsa_sparse_attention
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import Qwen38NextQSAIndexer
from miles_plugins.models.qwen3_8_next.ops.qsa_selection import build_block_form, merge_tail_and_clip


class Qwen38NextQSACoreAttention(MegatronModule):
    """Core attention restricted to the indexer's selection."""

    def __init__(self, config, layer_number: int, owner):
        super().__init__(config)
        self.layer_number = layer_number
        object.__setattr__(self, "_owner", owner)
        self.softmax_scale = config.kv_channels**-0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask=None, **kwargs):
        selection = getattr(self._owner, "_qsa_selection", None)
        if selection is None:
            raise RuntimeError(
                "QSA core attention ran with no selection published. "
                "Qwen38NextAttention.forward sets it before delegating; reaching here "
                "means core_attention was called out of band."
            )
        block_form = getattr(self._owner, "_qsa_block_form", None)
        if query.dim() == 3:
            if block_form is not None:
                sel_bitmap, lo, hi, blk_base, tok_base, block_first, block_last, blk = block_form
                return qsa_block_sparse_attention(
                    query,
                    key,
                    value,
                    sel_bitmap,
                    lo,
                    hi,
                    blk_base,
                    tok_base,
                    block_first,
                    block_last,
                    self.softmax_scale,
                    blk,
                ).reshape(query.shape[0], -1)
            return qsa_sparse_attention(query, key, value, selection, self.softmax_scale).reshape(query.shape[0], -1)

        if query.dim() != 4:
            raise RuntimeError(f"QSA core attention expected a 3D (thd) or 4D (sbhd) query, got {tuple(query.shape)}")

        s, b, hq, d = query.shape
        out = [
            qsa_sparse_attention(query[:, i], key[:, i], value[:, i], selection, self.softmax_scale) for i in range(b)
        ]
        return torch.stack(out, dim=1).reshape(s, b, hq * d)


class Qwen38NextAttention(SelfAttention):
    """Megatron self-attention whose key set is chosen by a QSA indexer."""

    def __init__(self, config, submodules, layer_number=1, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        if self.checkpoint_core_attention:
            # the selection is cleared after forward, so a backward-time core_attn recompute would find none
            raise NotImplementedError(
                "Qwen3.8-Next QSA does not support selective core_attn recompute; "
                "use full recompute or drop core_attn from --recompute-modules"
            )
        if config.rotary_interleaved:
            raise NotImplementedError("the QSA indexer mirrors sglang's neox RoPE; rotary_interleaved is unsupported")
        self.indexer = Qwen38NextQSAIndexer(config, layer_number=layer_number)
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio
        self.core_attention = Qwen38NextQSACoreAttention(config, layer_number, owner=self)
        self._qsa_selection = None
        self._qsa_block_form = None

    @staticmethod
    def _packed_positions(cu_seqlens: Tensor, total: int) -> Tensor:
        """Positions restarting at 0 for each sequence in a packed batch."""
        idx = torch.arange(total, device=cu_seqlens.device)
        starts = cu_seqlens[:-1].long()
        seg = torch.zeros(total, dtype=torch.long, device=cu_seqlens.device)
        seg[starts[1:]] = 1
        seg = seg.cumsum(0)
        return idx - starts[seg]

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        packed = kwargs.get("packed_seq_params")

        indexer_states = hidden_states
        if getattr(self.config, "sequence_parallel", False):
            if get_tensor_model_parallel_world_size() > 1:
                with torch.no_grad():
                    indexer_states = gather_from_sequence_parallel_region(
                        hidden_states,
                        tensor_parallel_output_grad=False,
                        group=get_tensor_model_parallel_group(),
                    )
        seq = indexer_states.shape[0]
        if packed is not None:
            cu = getattr(packed, "cu_seqlens_q", None)
            if cu is None:
                raise NotImplementedError(
                    "packed_seq_params without cu_seqlens_q: QSA needs the sequence "
                    "boundaries to place positions and to keep attention inside a "
                    "document."
                )
            self._qsa_cu_seqlens = cu
            positions = self._packed_positions(cu, seq)
        else:
            self._qsa_cu_seqlens = None
            positions = torch.arange(seq, device=hidden_states.device)
        rotary_pos_emb = kwargs.get("rotary_pos_emb")
        if rotary_pos_emb is None:
            raise RuntimeError("the QSA indexer reuses the attention RoPE, but no rotary_pos_emb was passed")
        with torch.no_grad():
            selection = self.indexer(indexer_states[:, 0], positions, rotary_pos_emb, cu_seqlens=self._qsa_cu_seqlens)
            self._qsa_selection, seq_start = merge_tail_and_clip(selection, positions, self.compress_ratio)
            self._qsa_block_form = build_block_form(
                self._qsa_selection, positions, seq_start, self._qsa_cu_seqlens, self.compress_ratio
            )
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._qsa_selection = None
            self._qsa_cu_seqlens = None
            self._qsa_block_form = None
