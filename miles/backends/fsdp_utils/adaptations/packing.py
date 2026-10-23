"""Packed-document boundaries for the FSDP backend.

The FSDP backend packs several documents into one ``[1, T]`` forward row; stateful layers (Mamba2,
GatedDeltaNet) and attention must reset per document. ``get_batch`` records the boundaries
(``cu_seqlens``, ``max_seqlen``) and ``hf_attention_packing_kwargs`` / ``hf_packing_kwargs`` hand them
to HF modeling. NemotronH's patch only sees ``position_ids`` inside the model forward, so
``packed_seq_context`` re-derives them there.
"""

import functools
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PackedSeqContext:
    cu_seqlens: torch.Tensor  # int32 [num_docs + 1]
    seq_idx: torch.Tensor  # int32 [1, T]
    max_seqlen: int


def packed_seq_context(position_ids: torch.Tensor | None) -> PackedSeqContext | None:
    """Derive per-document boundaries from packed ``position_ids``, or ``None`` when not packing."""
    # FSDP requires context_parallel_size == 1, so this sees the full packed row, not a zigzag CP shard.
    if position_ids is None or position_ids.dim() != 2 or position_ids.shape[0] != 1:
        return None
    pos = position_ids.reshape(-1)
    starts = (pos == 0).nonzero(as_tuple=True)[0]
    if starts.numel() <= 1:
        return None  # single document -> packing is a no-op
    total = torch.tensor([pos.numel()], device=pos.device, dtype=starts.dtype)
    cu_seqlens = torch.cat([starts, total]).to(torch.int32)
    seq_idx = (torch.cumsum((pos == 0).to(torch.int32), dim=0) - 1).to(torch.int32).unsqueeze(0).contiguous()
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
    return PackedSeqContext(cu_seqlens=cu_seqlens, seq_idx=seq_idx, max_seqlen=max_seqlen)


HF_PACKING_KWARG_NAMES = frozenset({"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx"})


def hf_attention_packing_kwargs(*, cu_seqlens: torch.Tensor, max_seqlen: int) -> dict[str, object]:
    """The varlen boundaries HF attention reads; ``max_length`` stays an int, which FlashAttention-4 needs."""
    return {
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": max_seqlen,
        "max_length_k": max_seqlen,
    }


def hf_packing_kwargs(
    *, cu_seqlens: torch.Tensor, cu_seqlens_host: tuple[int, ...], max_seqlen: int
) -> dict[str, object]:
    """The micro-batch's ``get_batch`` boundaries, as the kwargs HF's ``DataCollatorWithFlattening`` emits.

    HF models forward these through ``**kwargs`` to every layer: flash attention reads ``cu_seq_lens_*`` /
    ``max_length_*``, and linear-attention layers read ``seq_idx`` and ``cu_seq_lens_q``.
    """
    # the host-side length keeps repeat_interleave from syncing the device to size its output
    seq_idx = torch.repeat_interleave(
        torch.arange(cu_seqlens.numel() - 1, dtype=torch.int32, device=cu_seqlens.device),
        cu_seqlens.diff(),
        output_size=cu_seqlens_host[-1],
    )
    return {
        **hf_attention_packing_kwargs(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen),
        "seq_idx": seq_idx.unsqueeze(0),
    }


def keep_packing_kwargs_out_of_vision(model: nn.Module) -> None:
    """Drop the language model's packing kwargs at the vision tower's forward; idempotent, and a no-op
    without a tower. HF hands the tower its ``**kwargs`` too, so ``seq_idx`` would otherwise reach it."""
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None or getattr(visual.forward, "_drops_lm_packing_kwargs", False):
        return
    vision_forward = visual.forward

    @functools.wraps(vision_forward)
    def forward(*args, **kwargs):
        return vision_forward(*args, **{k: v for k, v in kwargs.items() if k not in HF_PACKING_KWARG_NAMES})

    forward._drops_lm_packing_kwargs = True
    visual.forward = forward
