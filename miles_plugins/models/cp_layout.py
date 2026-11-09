"""Immutable CPU copy plans for Megatron zigzag and FLA packed CP layouts."""

from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class CPTransferPlan:
    send: tuple[tuple[tuple[int, int, int], ...], ...]
    recv: tuple[tuple[tuple[int, int, int], ...], ...]
    send_sizes: tuple[int, ...]
    recv_sizes: tuple[int, ...]


@dataclass(frozen=True)
class CPLayoutMetadata:
    cp_size: int
    cp_rank: int
    global_boundaries: tuple[int, ...]
    local_total_len: int
    zigzag_to_contiguous: CPTransferPlan
    contiguous_to_zigzag: CPTransferPlan


def _freeze_plan(send, recv):
    def freeze(peers):
        result = []
        for intervals in peers:
            merged = []
            for src, dst, length in sorted(intervals):
                if merged and (src, dst) == (merged[-1][0] + merged[-1][2], merged[-1][1] + merged[-1][2]):
                    prev_src, prev_dst, prev_length = merged.pop()
                    merged.append((prev_src, prev_dst, prev_length + length))
                else:
                    merged.append((src, dst, length))
            result.append(tuple(merged))
        return tuple(result)

    send, recv = freeze(send), freeze(recv)
    return CPTransferPlan(
        send=send,
        recv=recv,
        send_sizes=tuple(sum(length for _, _, length in peer) for peer in send),
        recv_sizes=tuple(sum(length for _, _, length in peer) for peer in recv),
    )


def build_cp_layout_metadata(global_boundaries, cp_size, cp_rank):
    """Build both directions from CPU global boundaries, including final padding."""
    boundaries = tuple(global_boundaries)
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("GDN CP requires a positive size and a rank within the group")
    if len(boundaries) < 2 or any(type(value) is not int for value in boundaries):
        raise ValueError("GDN CP requires one-dimensional integer packed sequence boundaries")
    if boundaries[0] != 0 or any(end <= start for start, end in pairwise(boundaries)):
        raise ValueError("GDN CP requires strictly increasing boundaries starting at zero")
    global_lengths = tuple(end - start for start, end in pairwise(boundaries))
    if any(length % cp_size for length in global_lengths):
        raise ValueError(f"GDN CP sequence lengths must be divisible by cp_size={cp_size}")
    local_lengths = tuple(length // cp_size for length in global_lengths)
    local_total = boundaries[-1] // cp_size
    forward_send = [[] for _ in range(cp_size)]
    forward_recv = [[] for _ in range(cp_size)]
    inverse_send = [[] for _ in range(cp_size)]
    inverse_recv = [[] for _ in range(cp_size)]

    local_start = 0
    for global_start, global_length, local_length in zip(boundaries[:-1], global_lengths, local_lengths, strict=True):
        for src_rank in range(cp_size):
            if global_length % (2 * cp_size) == 0:
                half = local_length // 2
                chunks = (
                    (local_start, global_start + src_rank * half, half),
                    (local_start + half, global_start + (2 * cp_size - 1 - src_rank) * half, half),
                )
            else:
                chunks = ((local_start, global_start + src_rank * local_length, local_length),)
            for src_start, packed_start, length in chunks:
                while length:
                    dst_rank, dst_start = divmod(packed_start, local_total)
                    count = min(length, local_total - dst_start)
                    if src_rank == cp_rank:
                        forward_send[dst_rank].append((src_start, dst_start, count))
                        inverse_recv[dst_rank].append((dst_start, src_start, count))
                    if dst_rank == cp_rank:
                        forward_recv[src_rank].append((src_start, dst_start, count))
                        inverse_send[src_rank].append((dst_start, src_start, count))
                    src_start += count
                    packed_start += count
                    length -= count
        local_start += local_length

    return CPLayoutMetadata(
        cp_size=cp_size,
        cp_rank=cp_rank,
        global_boundaries=boundaries,
        local_total_len=local_total,
        zigzag_to_contiguous=_freeze_plan(forward_send, forward_recv),
        contiguous_to_zigzag=_freeze_plan(inverse_send, inverse_recv),
    )
