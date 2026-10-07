"""A transfer buffer must never be loaded again while a write still reads it."""

import threading
from concurrent.futures import Future

import pytest
import torch

import miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers as transfer_buffers_module
from miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers import TransferBuffers

_WAIT_BOUND = 10.0
_STILL_BLOCKED = 0.2


@pytest.fixture
def make_transfer_buffers(monkeypatch: pytest.MonkeyPatch):
    # CPU CI has no pinned memory
    monkeypatch.setattr(
        transfer_buffers_module,
        "_allocate_transfer_buffer",
        lambda buffer_nbytes, device: torch.empty(buffer_nbytes, dtype=torch.uint8, device=device),
    )
    registered: list[tuple[int, int]] = []

    def make(num_buffers: int, buffer_nbytes: int = 16) -> tuple[TransferBuffers, list[tuple[int, int]]]:
        transfer_buffers = TransferBuffers(
            num_buffers,
            buffer_nbytes,
            device=torch.device("cpu"),
            register_memory=lambda buffer: registered.append((buffer.data_ptr(), buffer.numel())),
        )
        return transfer_buffers, registered

    return make


def _in_thread(target) -> threading.Event:
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
    transfer_buffers.release_after(first_buffer, [pending_write])
    transfer_buffers.release_after(transfer_buffers.acquire(), [])

    acquired: list[torch.Tensor] = []
    returned = _in_thread(lambda: acquired.append(transfer_buffers.acquire()))

    assert not returned.wait(timeout=_STILL_BLOCKED)
    pending_write.set_result(None)
    assert returned.wait(timeout=_WAIT_BOUND)
    assert acquired == [first_buffer]


def test_a_failed_write_frees_its_buffer_without_raising(make_transfer_buffers) -> None:
    """What a failed write means is the caller's decision; the buffer is free once the write returned."""
    transfer_buffers, _ = make_transfer_buffers(num_buffers=1)
    buffer = transfer_buffers.acquire()
    failed_write = Future()
    failed_write.set_exception(RuntimeError("write failed"))
    transfer_buffers.release_after(buffer, [failed_write])

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


def test_wait_for_writes_returns_after_the_writes_of_every_buffer(make_transfer_buffers) -> None:
    """The end of an update must not return while any write still reads transfer memory."""
    transfer_buffers, _ = make_transfer_buffers(num_buffers=2)
    writes = [Future(), Future()]
    for write in writes:
        transfer_buffers.release_after(transfer_buffers.acquire(), [write])

    returned = _in_thread(transfer_buffers.wait_for_writes)

    writes[0].set_result(None)
    assert not returned.wait(timeout=_STILL_BLOCKED)
    writes[1].set_result(None)
    assert returned.wait(timeout=_WAIT_BOUND)
