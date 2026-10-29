from collections.abc import Callable

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


def get_data_replica_rank_and_size(parallel_state: ParallelState, placement: WeightUpdatePlacement) -> tuple[int, int]:
    """(replica_rank, replica_size): this rank's index among the ranks that hold
    identical data after gathering per ``placement``, and their count. Collective."""
    assert (
        placement.gather_tp and placement.gather_ep
    ), f"ranks hold identical data only once TP and EP are gathered: {placement}"
    if placement.gather_pp:
        return dist.get_rank(), dist.get_world_size()

    column_id = min(dist.get_process_group_ranks(parallel_state.pp.group))
    all_column_ids: list = [None] * dist.get_world_size()
    dist.all_gather_object(all_column_ids, column_id)
    return sorted(set(all_column_ids)).index(column_id), dist.get_world_size() // parallel_state.pp.size


# (bucket stream index, byte size) -> the flat uint8 host tensor an owned bucket is copied into
HostBufferFactory = Callable[[int, int], torch.Tensor]


def _new_host_buffer(index: int, nbytes: int) -> torch.Tensor:
    # unpinned, since `.to("cpu", non_blocking=True)` pins memory that CUDA's host allocator keeps cached
    return torch.empty(nbytes, dtype=torch.uint8)


class OwnedBuckets:
    """The buckets one rank publishes, copied to the host: each bucket goes to the replica rank with the fewest
    bytes so far, so the ranks holding identical gathered data share the writes evenly. An owned bucket lands in
    one flat host buffer from `host_buffer`, its tensors ordered by dtype (see `pack_by_dtype`)."""

    def __init__(
        self, *, replica_rank: int, replica_size: int, host_buffer: HostBufferFactory = _new_host_buffer
    ) -> None:
        self._replica_rank = replica_rank
        self._replica_size = replica_size
        self._host_buffer = host_buffer
        self.reset()

    def reset(self) -> None:
        self._next_index = 0
        self._bytes_per_rank = [0] * self._replica_size
        self._owned: list[tuple[int, list[tuple[str, torch.Tensor]]]] = []

    def offer(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # every rank sees the same buckets in the same order, so every rank picks the same owner
        owner = min(range(self._replica_size), key=lambda rank: (self._bytes_per_rank[rank], rank))
        nbytes = sum(tensor.nbytes for _, tensor in bucket)
        self._bytes_per_rank[owner] += nbytes
        if owner == self._replica_rank:
            # copied out now: kept on the device until the stream ends, the owned buckets would add
            # model/replica_size to each rank's peak device memory
            flat = self._host_buffer(self._next_index, nbytes)
            self._owned.append((self._next_index, _copy_into(pack_by_dtype(bucket), flat)))
        self._next_index += 1
        # the caller's loop holds this list while the next bucket is gathered; empty it so its device tensors free first
        bucket.clear()

    def take(self) -> list[tuple[int, list[tuple[str, torch.Tensor]]]]:
        """The owned host buckets with their stream index, handed over so none stays alive after the sync."""
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()  # the non-blocking host copies have landed
        owned, self._owned = self._owned, []
        return owned


def pack_by_dtype(bucket: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    """The bucket ordered by dtype, widest first, so each dtype fills one aligned, contiguous byte range.

    Engines load a bucket by name, so its order is free.
    """
    return sorted(bucket, key=lambda item: (-item[1].element_size(), str(item[1].dtype)))


def _copy_into(packed: list[tuple[str, torch.Tensor]], flat: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Copy each tensor of `packed` to its place in `flat`, back to back, and return the host views."""
    host_bucket, offset = [], 0
    for name, tensor in packed:
        host = flat[offset : offset + tensor.nbytes].view(tensor.dtype).view(tensor.shape)
        # non-blocking, so the copy overlaps the next gather; take() waits for it
        host_bucket.append((name, host.copy_(tensor, non_blocking=True)))
        offset += tensor.nbytes
    return host_bucket
