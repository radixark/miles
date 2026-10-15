"""Where each attention layer finds its keys in a packed sequence, with and without context parallelism.

Under CP every rank holds a load-balanced shard of the sequence's tokens. The sliding-window layers attend
the rank's queries over the window of gathered keys before them; the full-attention layers gather the
whole sequence, attend it, and keep the rank's rows.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_single_autograd
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.models.common.attention import VarlenMetadata, create_varlen_metadata_for_document


@dataclass(frozen=True)
class ContextParallelLayout:
    """``local_indices``: the sequence positions of this rank's tokens, ascending.
    ``restore_indices``: the sequence order of the rank-concatenated shards."""

    group: dist.ProcessGroup
    local_indices: torch.Tensor
    restore_indices: torch.Tensor

    @classmethod
    def build(cls, mesh: DeviceMesh, *, load_balancer: str, seq_len: int, device) -> "ContextParallelLayout":
        group = mesh.get_group()
        sequence_order = torch.arange(seq_len, device=device).unsqueeze(0)
        (local,), _ = cp_shard(mesh, (sequence_order,), None, load_balancer)
        local_indices = local.flatten()
        return cls(
            group=group,
            local_indices=local_indices,
            restore_indices=_gather_tokens(local_indices, group).argsort(),
        )


@dataclass(frozen=True)
class PackedSequence:
    """``cu_seqlens``: the document boundaries of the whole sequence. ``window``: this rank's queries
    against their key spans, ``window_key_indices`` the sequence positions those spans read (None when
    the spans are the documents themselves)."""

    cu_seqlens: torch.Tensor
    window: VarlenMetadata
    window_key_indices: torch.Tensor | None = None
    cp_layout: ContextParallelLayout | None = None

    def gather_tokens(self, x_TD: torch.Tensor) -> torch.Tensor:
        """The whole sequence's rows of ``x_TD``, in sequence order; gradients return to their owners."""
        if self.cp_layout is None:
            return x_TD
        gathered = all_gather_single_autograd(x_TD.contiguous(), 0, self.cp_layout.group)
        return gathered.index_select(0, self.cp_layout.restore_indices)

    def gather_tokens_detached(self, x_TD: torch.Tensor) -> torch.Tensor:
        if self.cp_layout is None:
            return x_TD
        return _gather_tokens(x_TD, self.cp_layout.group).index_select(0, self.cp_layout.restore_indices)

    def local_rows(self, x_TD: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a whole-sequence tensor."""
        if self.cp_layout is None:
            return x_TD
        return x_TD.index_select(0, self.cp_layout.local_indices)

    def window_keys(self, local_TD: torch.Tensor) -> torch.Tensor:
        """The rows the sliding-window layers attend, from this rank's rows of ``local_TD``."""
        if self.window_key_indices is None:
            return local_TD
        return self.gather_tokens(local_TD).index_select(0, self.window_key_indices)


def build_packed_sequence(
    positions: torch.Tensor, cp_layout: ContextParallelLayout | None, *, window: int
) -> PackedSequence:
    """``positions`` ``[1, T]`` restart at 0 on each document; ``window`` keys end at the query, inclusive."""
    if cp_layout is None:
        masks = create_varlen_metadata_for_document(positions, include_host_offsets=True)
        return PackedSequence(cu_seqlens=masks.cu_seq_q, window=masks)
    with torch.no_grad():
        gathered = _gather_tokens(positions.flatten(), cp_layout.group)
        positions = gathered.index_select(0, cp_layout.restore_indices).unsqueeze(0)
    masks = create_varlen_metadata_for_document(positions, include_host_offsets=True)
    query_spans, key_spans = _window_spans(masks.cu_seq_q, cp_layout.local_indices, window=window)
    device = positions.device
    return PackedSequence(
        cu_seqlens=masks.cu_seq_q,
        window=VarlenMetadata(
            cu_seq_q=_offsets(query_spans, device),
            cu_seq_k=_offsets(key_spans, device),
            max_q=max(end - start for start, end in query_spans),
            max_k=max(end - start for start, end in key_spans),
        ),
        window_key_indices=torch.cat([torch.arange(start, end) for start, end in key_spans]).to(device),
        cp_layout=cp_layout,
    )


def _window_spans(
    cu_seqlens: torch.Tensor, local_indices: torch.Tensor, *, window: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split this rank's tokens into runs, consecutive positions of one document, and give each run the
    keys its queries see: from ``window - 1`` before its first query to its last.

    Varlen attention aligns a shorter query span to the end of its key span, so query ``i`` of a run
    attends exactly its own window.
    """
    document = torch.searchsorted(cu_seqlens, local_indices, right=True) - 1
    breaks = (local_indices[1:] != local_indices[:-1] + 1) | (document[1:] != document[:-1])
    starts = [0, *(breaks.nonzero().flatten() + 1).tolist()]
    ends = [*starts[1:], local_indices.shape[0]]
    firsts = local_indices[starts].tolist()
    lasts = local_indices[[end - 1 for end in ends]].tolist()
    document_starts = cu_seqlens[document[starts]].tolist()
    query_spans = list(zip(starts, ends, strict=True))
    key_spans = [
        (max(document_start, first - (window - 1)), last + 1)
        for document_start, first, last in zip(document_starts, firsts, lasts, strict=True)
    ]
    return query_spans, key_spans


def _offsets(spans: list[tuple[int, int]], device) -> torch.Tensor:
    lengths = torch.tensor([0] + [end - start for start, end in spans], dtype=torch.int32)
    return lengths.cumsum(0, dtype=torch.int32).to(device)


def _gather_tokens(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = local.new_empty((dist.get_world_size(group) * local.shape[0], *local.shape[1:]))
    dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    return gathered
