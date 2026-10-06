"""A staging buffer must never be loaded again while a write still reads it."""

import threading
from concurrent.futures import Future

import pytest
import torch

import miles.backends.training_utils.weight_update.protocols.utils.staging_buffers as staging_buffers_module
from miles.backends.training_utils.weight_update.protocols.utils.staging_buffers import StagingBuffers

_WAIT_BOUND = 10.0
_STILL_BLOCKED = 0.2


@pytest.fixture
def make_staging_buffers(monkeypatch: pytest.MonkeyPatch):
    # CPU CI has no pinned memory
    monkeypatch.setattr(
        staging_buffers_module,
        "_allocate_staging_buffer",
        lambda buffer_bytes, device: torch.empty(buffer_bytes, dtype=torch.uint8, device=device),
    )
    registered: list[tuple[int, int]] = []

    def make(num_buffers: int, buffer_bytes: int = 16) -> tuple[StagingBuffers, list[tuple[int, int]]]:
        staging_buffers = StagingBuffers(
            num_buffers,
            buffer_bytes,
            device=torch.device("cpu"),
            register_memory=lambda buffer: registered.append((buffer.data_ptr(), buffer.numel())),
        )
        return staging_buffers, registered

    return make


def _in_thread(target) -> threading.Event:
    returned = threading.Event()

    def run() -> None:
        target()
        returned.set()

    threading.Thread(target=run, daemon=True).start()
    return returned


def test_a_buffer_is_not_handed_out_while_a_write_still_reads_it(make_staging_buffers) -> None:
    """Loading into a buffer a write still reads would send the next weights' bytes in place of the old ones."""
    staging_buffers, _ = make_staging_buffers(num_buffers=2)
    first_buffer = staging_buffers.acquire()
    pending_write = Future()
    staging_buffers.release_after(first_buffer, [pending_write])
    staging_buffers.release_after(staging_buffers.acquire(), [])

    acquired: list[torch.Tensor] = []
    returned = _in_thread(lambda: acquired.append(staging_buffers.acquire()))

    assert not returned.wait(timeout=_STILL_BLOCKED)
    pending_write.set_result(None)
    assert returned.wait(timeout=_WAIT_BOUND)
    assert acquired == [first_buffer]


def test_a_failed_write_frees_its_buffer_without_raising(make_staging_buffers) -> None:
    """What a failed write means is the caller's decision; the buffer is free once the write returned."""
    staging_buffers, _ = make_staging_buffers(num_buffers=1)
    buffer = staging_buffers.acquire()
    failed_write = Future()
    failed_write.set_exception(RuntimeError("write failed"))
    staging_buffers.release_after(buffer, [failed_write])

    assert staging_buffers.acquire() is buffer


def test_each_buffer_is_registered_once_over_its_whole_allocation(make_staging_buffers) -> None:
    """A write reads anywhere in a buffer, so its whole range must be registered, and reuse must not register again."""
    staging_buffers, registered = make_staging_buffers(num_buffers=2, buffer_bytes=64)

    buffers = []
    for _ in range(4):
        buffer = staging_buffers.acquire()
        buffers.append(buffer)
        staging_buffers.release_after(buffer, [])

    assert sorted(registered) == sorted({(buffer.data_ptr(), 64) for buffer in buffers})


def test_wait_for_writes_returns_after_the_writes_of_every_buffer(make_staging_buffers) -> None:
    """The end of an update must not return while any write still reads staging memory."""
    staging_buffers, _ = make_staging_buffers(num_buffers=2)
    writes = [Future(), Future()]
    for write in writes:
        staging_buffers.release_after(staging_buffers.acquire(), [write])

    returned = _in_thread(staging_buffers.wait_for_writes)

    writes[0].set_result(None)
    assert not returned.wait(timeout=_STILL_BLOCKED)
    writes[1].set_result(None)
    assert returned.wait(timeout=_WAIT_BOUND)
