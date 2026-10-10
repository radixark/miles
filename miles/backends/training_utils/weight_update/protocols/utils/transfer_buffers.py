import concurrent.futures
from collections.abc import Callable, Sequence
from concurrent.futures import Future

import torch


class TransferBuffers:
    """Memory a p2p sender loads weights into and a transport writes them to the rollout engines from.

    The protocol creates it once per process on the device the transport reads from and registers each buffer with
    the transport. For each load it calls `acquire`, loads into the buffer, starts the writes that read it, and hands
    them to `release_after`. A buffer is handed out again only after every write reading it has returned.
    """

    def __init__(
        self,
        num_buffers: int,
        buffer_nbytes: int,
        *,
        device: torch.device,
        register_memory: Callable[[torch.Tensor], None],
    ) -> None:
        self.buffer_nbytes = buffer_nbytes
        self._buffers = [_allocate_transfer_buffer(buffer_nbytes, device) for _ in range(num_buffers)]
        for buffer in self._buffers:
            register_memory(buffer)
        self._pending_writes_by_buffer_index: list[list[Future]] = [[] for _ in self._buffers]
        self._next_buffer_index = 0

    def acquire(self) -> torch.Tensor:
        """Returns the next buffer once every write reading it has returned, failed or not; the caller reports
        failed writes. No timeout here: the transport bounds each write."""
        buffer_index = self._next_buffer_index
        self._next_buffer_index = (buffer_index + 1) % len(self._buffers)
        concurrent.futures.wait(self._pending_writes_by_buffer_index[buffer_index])
        self._pending_writes_by_buffer_index[buffer_index] = []
        return self._buffers[buffer_index]

    def release_after(self, buffer: torch.Tensor, writes: Sequence[Future]) -> None:
        """Records the writes that read `buffer`; `acquire` hands it out again only after they return."""
        (buffer_index,) = [index for index, candidate in enumerate(self._buffers) if candidate is buffer]
        self._pending_writes_by_buffer_index[buffer_index] += writes


def _allocate_transfer_buffer(buffer_nbytes: int, device: torch.device) -> torch.Tensor:
    # host buffers are pinned in place: a pageable copy first would stay resident on aarch64 hosts
    return torch.empty(buffer_nbytes, dtype=torch.uint8, device=device, pin_memory=device.type == "cpu")
