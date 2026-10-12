"""A transfer buffer must never be loaded again while a write still reads it."""

import threading
from concurrent.futures import Future

import pytest
import torch

import miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers as transfer_buffers_module
from miles.backends.training_utils.weight_update.protocols.transports.mooncake import RemoteShard
from miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers import TransferBuffers

_RETURN_TIMEOUT_SECONDS = 10.0
_STILL_BLOCKED_SECONDS = 0.2


@pytest.fixture
def make_transfer_buffers(monkeypatch: pytest.MonkeyPatch):
    # CPU CI has no pinned memory
    monkeypatch.setattr(
        transfer_buffers_module,
        "_allocate_transfer_buffer",
        lambda buffer_nbytes, device: torch.empty(buffer_nbytes, dtype=torch.uint8, device=device),
    )
    registered: list[tuple[int, int]] = []

    def make(
        num_buffers: int, buffer_nbytes: int = 16, write_timeout: float = _RETURN_TIMEOUT_SECONDS
    ) -> tuple[TransferBuffers, list[tuple[int, int]]]:
        transfer_buffers = TransferBuffers(
            num_buffers,
            buffer_nbytes,
            device=torch.device("cpu"),
            register_memory=lambda buffer: registered.append((buffer.data_ptr(), buffer.numel())),
            unregister_memory=lambda buffer: registered.remove((buffer.data_ptr(), buffer.numel())),
            write_timeout=write_timeout,
        )
        return transfer_buffers, registered

    return make


def _remote_shard(session_id: str) -> RemoteShard:
    return RemoteShard(
        rollout_engine_ind=0,
        rollout_engine_rank=0,
        runner_role="target",
        session_id=session_id,
        weight_locations_by_name={},
    )


def _start_in_thread(target) -> threading.Event:
    returned = threading.Event()

    def run() -> None:
        target()
        returned.set()

    threading.Thread(target=run, daemon=True).start()
    return returned


def test_a_buffer_is_not_handed_out_while_a_write_still_reads_it(make_transfer_buffers) -> None:
    """Loading into a buffer a write still reads would send the next weights' bytes in place of the old ones."""
    transfer_buffers, _ = make_transfer_buffers(num_buffers=2)
    first_buffer = transfer_buffers.acquire()
    pending_write = Future()
    transfer_buffers.release_after(first_buffer, [(_remote_shard("session-a"), pending_write)])
    transfer_buffers.release_after(transfer_buffers.acquire(), [])

    acquired: list[torch.Tensor] = []
    returned = _start_in_thread(lambda: acquired.append(transfer_buffers.acquire()))

    assert not returned.wait(timeout=_STILL_BLOCKED_SECONDS)
    pending_write.set_result(None)
    assert returned.wait(timeout=_RETURN_TIMEOUT_SECONDS)
    assert acquired == [first_buffer]


def test_a_failed_write_fails_the_next_acquire_of_its_buffer_and_frees_it(make_transfer_buffers) -> None:
    """Loading on after a failed write would only queue more writes behind a broken rollout engine."""
    transfer_buffers, _ = make_transfer_buffers(num_buffers=1)
    buffer = transfer_buffers.acquire()
    failed_write = Future()
    failed_write.set_exception(RuntimeError("write failed"))
    transfer_buffers.release_after(buffer, [(_remote_shard("session-a"), failed_write)])

    with pytest.raises(RuntimeError, match="session session-a.*write failed"):
        transfer_buffers.acquire()
    assert transfer_buffers.acquire() is buffer


def test_a_write_still_running_after_the_timeout_fails_acquire_and_keeps_its_buffer(make_transfer_buffers) -> None:
    """A hung write must fail the update within the write timeout, and its buffer stays reserved while it reads."""
    transfer_buffers, _ = make_transfer_buffers(num_buffers=1, write_timeout=_STILL_BLOCKED_SECONDS)
    buffer = transfer_buffers.acquire()
    running_write = Future()
    transfer_buffers.release_after(buffer, [(_remote_shard("session-a"), running_write)])

    with pytest.raises(RuntimeError, match=f"session session-a.*still running after {_STILL_BLOCKED_SECONDS}s"):
        transfer_buffers.acquire()
    with pytest.raises(RuntimeError, match="still running"):
        transfer_buffers.acquire()
    running_write.set_result(None)
    assert transfer_buffers.acquire() is buffer


def test_each_buffer_is_registered_once_over_its_whole_allocation(make_transfer_buffers) -> None:
    """A write reads anywhere in a buffer, so its whole range must be registered, and reuse must not register again."""
    transfer_buffers, registered = make_transfer_buffers(num_buffers=2, buffer_nbytes=64)

    buffers = []
    for _ in range(4):
        buffer = transfer_buffers.acquire()
        buffers.append(buffer)
        transfer_buffers.release_after(buffer, [])

    assert sorted(registered) == sorted({(buffer.data_ptr(), 64) for buffer in buffers})
