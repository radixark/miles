import concurrent.futures
from collections.abc import Callable, Sequence
from concurrent.futures import Future

import torch

from miles.backends.training_utils.weight_update.protocols.transports.mooncake import RemoteShard


class TransferBuffers:
    """Memory a p2p sender loads weights into and a transport writes them to the rollout engines from.

    The protocol creates it once per process on the device the transport reads from and registers each buffer with
    the transport. For each load it calls `acquire`, loads into the buffer, starts the writes that read it, and hands
    them to `release_after`. A buffer is handed out again only after every write reading it has returned; a write
    that failed or is still running after `write_timeout` fails the update when its buffer is next acquired or when
    the protocol waits for all writes.
    """

    def __init__(
        self,
        num_buffers: int,
        buffer_nbytes: int,
        *,
        device: torch.device,
        register_memory: Callable[[torch.Tensor], None],
        unregister_memory: Callable[[torch.Tensor], None],
        write_timeout: float,
    ) -> None:
        self.buffer_nbytes = buffer_nbytes
        self._unregister_memory = unregister_memory
        self._write_timeout = write_timeout
        self._buffers = [_allocate_transfer_buffer(buffer_nbytes, device) for _ in range(num_buffers)]
        for buffer in self._buffers:
            register_memory(buffer)
        self._pending_writes_by_buffer_index: list[list[tuple[RemoteShard, Future]]] = [[] for _ in self._buffers]
        self._next_buffer_index = 0

    def acquire(self) -> torch.Tensor:
        """Returns the next buffer once every write reading it has returned."""
        buffer_index = self._next_buffer_index
        self._next_buffer_index = (buffer_index + 1) % len(self._buffers)
        self._wait_for_writes_reading([buffer_index])
        return self._buffers[buffer_index]

    def release_after(self, buffer: torch.Tensor, pending_writes: Sequence[tuple[RemoteShard, Future]]) -> None:
        """Records the writes that read `buffer`; `acquire` hands it out again only after they return."""
        (buffer_index,) = [index for index, candidate in enumerate(self._buffers) if candidate is buffer]
        self._pending_writes_by_buffer_index[buffer_index] += pending_writes

    def wait_for_all_writes(self) -> None:
        """Waits for every write still reading a buffer."""
        self._wait_for_writes_reading(range(len(self._buffers)))

    def release(self) -> None:
        """Waits for every write still reading a buffer, then unregisters the buffers from the transport."""
        self.wait_for_all_writes()
        for buffer in self._buffers:
            self._unregister_memory(buffer)

    def _wait_for_writes_reading(self, buffer_indices: Sequence[int]) -> None:
        pending_writes = [write for index in buffer_indices for write in self._pending_writes_by_buffer_index[index]]
        try:
            _raise_if_any_write_failed(pending_writes, timeout=self._write_timeout)
        finally:
            for index in buffer_indices:
                self._pending_writes_by_buffer_index[index] = [
                    (remote_shard, write)
                    for remote_shard, write in self._pending_writes_by_buffer_index[index]
                    if not write.done()
                ]


def _raise_if_any_write_failed(pending_writes: list[tuple[RemoteShard, Future]], timeout: float) -> None:
    _, unfinished_writes = concurrent.futures.wait([write for _, write in pending_writes], timeout=timeout)
    failures = []
    for remote_shard, write in pending_writes:
        if write in unfinished_writes:
            reason = f"still running after {timeout}s"
        elif write.exception() is not None:
            reason = repr(write.exception())
        else:
            continue
        failures.append(
            f"rollout engine {remote_shard.rollout_engine_ind} rank {remote_shard.rollout_engine_rank} "
            f"(session {remote_shard.session_id}): {reason}"
        )
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(pending_writes)} p2p writes failed:\n" + "\n".join(failures))


def _allocate_transfer_buffer(buffer_nbytes: int, device: torch.device) -> torch.Tensor:
    # host buffers are pinned in place: a pageable copy first would stay resident on aarch64 hosts
    return torch.empty(buffer_nbytes, dtype=torch.uint8, device=device, pin_memory=device.type == "cpu")
