"""Shared packed-document boundary derivation for the FSDP backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackedSeqContext:
    cu_seqlens: "object"  # torch.Tensor int32 [num_docs+1]
    seq_idx: "object"  # torch.Tensor int32 [1, T]
    max_seqlen: int


def packed_seq_context(position_ids):
    """Derive per-document boundaries from packed ``position_ids``, or ``None`` when not packing."""
    import torch

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
