"""Qwen3.8-Next full-attention layer: QSA indexer + sparse attention.

12 of the 48 layers are full attention. Each projects its own indexer queries and
compressed keys, scores them, and keeps a budget of ``indexer_budget`` key tokens
per query; attention then reads only those.

"""

import torch
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from miles_plugins.models.qwen3_8_next.ops.kernel.qsa_block_sparse_attn import qsa_block_sparse_attention_triton
from miles_plugins.models.qwen3_8_next.ops.kernel.qsa_sparse_attn import qsa_sparse_attention_triton
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import PackedBlockLayout, Qwen38NextQSAIndexer


class Qwen38NextQSACoreAttention(MegatronModule):
    """Core attention restricted to the indexer's selection."""

    def __init__(self, config, layer_number: int, owner):
        super().__init__(config)
        self.layer_number = layer_number
        object.__setattr__(self, "_owner", owner)
        self.softmax_scale = config.kv_channels**-0.5
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio

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
                sel_bitmap, lo, hi, blk_base, tok_base, blk = block_form
                return qsa_block_sparse_attention_triton(
                    query, key, value, sel_bitmap, lo, hi, blk_base, tok_base, self.softmax_scale, blk
                ).reshape(query.shape[0], -1)
            return qsa_sparse_attention_triton(query, key, value, selection, self.softmax_scale).reshape(
                query.shape[0], -1
            )

        if query.dim() != 4:
            raise RuntimeError(
                f"QSA core attention expected a 3D (thd) or 4D (sbhd) query, got " f"{tuple(query.shape)}"
            )

        s, b, hq, d = query.shape
        out = [
            qsa_sparse_attention_triton(query[:, i], key[:, i], value[:, i], selection, self.softmax_scale)
            for i in range(b)
        ]
        return torch.stack(out, dim=1).reshape(s, b, hq * d)


class Qwen38NextAttention(SelfAttention):
    """Megatron self-attention whose key set is chosen by a QSA indexer."""

    def __init__(self, config, submodules, layer_number=1, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
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
        with torch.no_grad():
            use_mrope = getattr(self.config, "position_embedding_type", None) == "mrope"
            rotary_pos_emb = kwargs.get("rotary_pos_emb") if use_mrope else None
            selection = self.indexer(
                indexer_states[:, 0],
                positions,
                cu_seqlens=self._qsa_cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )
            seq_start = torch.arange(seq, device=positions.device) - positions
            r = self.compress_ratio
            tail_in_seq = (positions + 1) // r * r
            offs = torch.arange(r, device=positions.device)
            tail_pos = tail_in_seq.unsqueeze(1) + offs.unsqueeze(0)  # in-seq
            tail_idx = seq_start.unsqueeze(1) + tail_pos  # pack index
            tail_idx = torch.where(
                tail_pos <= positions.unsqueeze(1),
                tail_idx,
                torch.full_like(tail_idx, -1),
            )
            merged = torch.cat([selection, tail_idx.to(selection.dtype)], dim=1)
            pack_pos = torch.arange(seq, device=positions.device).unsqueeze(1)
            seg_lo = seq_start.unsqueeze(1)
            ok = (merged >= seg_lo) & (merged <= pack_pos) & (merged >= 0)
            self._qsa_selection = torch.where(ok, merged, torch.full_like(merged, -1))
            self._qsa_block_form = self._publish_block_form(self._qsa_selection, positions, seq_start, seq)
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._qsa_selection = None
            self._qsa_cu_seqlens = None
            self._qsa_block_form = None

    def _publish_block_form(self, selection: Tensor, positions: Tensor, seq_start: Tensor, seq: int):
        """Selection rows -> the block form the tensor-core kernel consumes.

        Block ids come from the indexer's own per-sequence grid (``PackedBlockLayout``), not
        from a formula over ``seq_start``: a ceil(seq_start / ratio) base looks right but
        collides for some length combinations (lens [301, 211] puts one sequence's last
        block on the next sequence's first), which would silently mix selections.
        """
        ratio = self.compress_ratio
        cu = self._qsa_cu_seqlens
        if cu is not None and cu.numel() > 2:
            layout = PackedBlockLayout(cu, positions, ratio)
            blk_base = layout.token_block_start.to(torch.int32)
            tok_base = layout.token_start.to(torch.int32)
            num_blocks = layout.num_blocks
        else:
            blk_base = torch.zeros(seq, dtype=torch.int32, device=selection.device)
            tok_base = torch.zeros(seq, dtype=torch.int32, device=selection.device)
            num_blocks = -(-seq // ratio)

        sel_bitmap = torch.zeros(seq, num_blocks, dtype=torch.uint8, device=selection.device)
        valid = selection >= 0
        if bool(valid.any()):
            rows = torch.arange(seq, device=selection.device).unsqueeze(1).expand_as(selection)
            blk = blk_base.unsqueeze(1).long() + (selection - tok_base.unsqueeze(1)).div(ratio, rounding_mode="floor")
            sel_bitmap[rows[valid], blk[valid].long().clamp_(0, num_blocks - 1)] = 1
        lo = tok_base
        hi = (positions + seq_start).to(torch.int32)
        return sel_bitmap, lo, hi, blk_base, tok_base, ratio
