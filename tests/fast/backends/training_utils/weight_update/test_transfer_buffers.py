"""A transfer buffer stays unchanged for healthy cells and registered for every unfinished write."""

import threading
from argparse import Namespace
from concurrent.futures import Future

import pytest
import torch

import miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers as transfer_buffers_module
from miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers import PendingWrite, TransferBuffers
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater

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


def _pending_write(session_id: str, future: Future[None]) -> PendingWrite:
    return PendingWrite(
        future=future,
        cell_updater=_RolloutCellUpdater(
            args=Namespace(ft_components=["weight_update"], update_weight_engine_request_timeout=1),
            cell_id=session_id,
            api_client=None,
        ),
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
    transfer_buffers.release_after(first_buffer, [_pending_write("session-a", pending_write)])
    transfer_buffers.release_after(transfer_buffers.acquire(), [])

    acquired: list[torch.Tensor] = []
    returned = _start_in_thread(lambda: acquired.append(transfer_buffers.acquire()))

    assert not returned.wait(timeout=_STILL_BLOCKED_SECONDS)
    pending_write.set_result(None)
    assert returned.wait(timeout=_RETURN_TIMEOUT_SECONDS)
    assert acquired == [first_buffer]


def test_a_failed_write_marks_its_cell_and_allows_reuse(make_transfer_buffers) -> None:
    """Loading on after a failed write would only queue more writes behind a broken rollout engine."""
    transfer_buffers, _ = make_transfer_buffers(num_buffers=1)
    buffer = transfer_buffers.acquire()
    failed_write = Future()
    failed_write.set_exception(RuntimeError("write failed"))
    pending = _pending_write("session-a", failed_write)
    transfer_buffers.release_after(buffer, [pending])

    assert transfer_buffers.acquire() is buffer
    assert pending.cell_updater.is_errored
    assert str(pending.cell_updater._error) == "write failed"
    assert transfer_buffers.acquire() is buffer


def test_a_timed_out_write_allows_reuse_but_prevents_unregister_until_it_returns(make_transfer_buffers) -> None:
    """A timed-out cell can absorb late reads, but the native writer must never read unregistered memory."""
    transfer_buffers, registered = make_transfer_buffers(num_buffers=1, write_timeout=_STILL_BLOCKED_SECONDS)
    buffer = transfer_buffers.acquire()
    running_write = Future()
    running_write.set_running_or_notify_cancel()
    pending = _pending_write("session-a", running_write)
    transfer_buffers.release_after(buffer, [pending])

    assert transfer_buffers.acquire() is buffer
    assert pending.cell_updater.is_errored
    assert transfer_buffers.acquire() is buffer
    with pytest.raises(RuntimeError, match="still running"):
        transfer_buffers.release()
    assert registered == [(buffer.data_ptr(), buffer.numel())]
    running_write.set_result(None)
    transfer_buffers.release()
    assert registered == []


def test_each_buffer_is_registered_once_over_its_whole_allocation(make_transfer_buffers) -> None:
    """A write reads anywhere in a buffer, so its whole range must be registered, and reuse must not register again."""
    transfer_buffers, registered = make_transfer_buffers(num_buffers=2, buffer_nbytes=64)

    buffers = []
    for _ in range(4):
        buffer = transfer_buffers.acquire()
        buffers.append(buffer)
        transfer_buffers.release_after(buffer, [])

    assert sorted(registered) == sorted({(buffer.data_ptr(), 64) for buffer in buffers})


def test_release_waits_for_every_cell_even_if_another_write_has_failed(make_transfer_buffers) -> None:
    """A failed completed write cannot hide another cell that is still reading the allocation."""
    buffers, registered = make_transfer_buffers(num_buffers=1, write_timeout=0.01)
    buffer = buffers.acquire()
    failed, running = Future(), Future()
    failed.set_exception(RuntimeError("failed"))
    buffers.release_after(buffer, [_pending_write("failed", failed), _pending_write("running", running)])

    with pytest.raises(RuntimeError, match="still running"):
        buffers.release()
    assert registered == [(buffer.data_ptr(), buffer.numel())]
    running.set_result(None)
    buffers.release()
    assert registered == []


def test_cancelled_queued_writes_do_not_prevent_releasing_memory(make_transfer_buffers) -> None:
    """A cancelled task never reads the buffer and must not be mistaken for a running native write."""
    buffers, registered = make_transfer_buffers(num_buffers=1, write_timeout=0)
    buffer = buffers.acquire()
    cancelled = Future()
    cancelled.cancel()
    buffers.release_after(buffer, [_pending_write("cancelled", cancelled)])

    buffers.release()
    assert registered == []


def test_failed_cell_writes_stay_registered_without_affecting_healthy_replacements(make_transfer_buffers) -> None:
    """Failed-cell writes must retain memory while a healthy replacement keeps its own state."""
    buffers, registered = make_transfer_buffers(num_buffers=2, write_timeout=0)
    first, second = buffers.acquire(), buffers.acquire()
    running, queued, healthy = Future(), Future(), Future()
    running.set_running_or_notify_cancel()
    old = _pending_write("cell-a", running)
    replacement = _pending_write("cell-a", healthy)
    buffers.release_after(first, [old])
    buffers.release_after(second, [PendingWrite(future=queued, cell_updater=old.cell_updater), replacement])
    first_error = RuntimeError("first")
    old.cell_updater.mark_errored(first_error)
    old.cell_updater.mark_errored(RuntimeError("second"))

    assert buffers.acquire() is first

    assert not running.done()
    assert not healthy.cancelled()
    assert not replacement.cell_updater.is_errored
    assert old.cell_updater._error is first_error
    with pytest.raises(RuntimeError, match="still running"):
        buffers.release()
    assert len(registered) == 2
    running.set_result(None)
    queued.set_result(None)
    healthy.set_result(None)
    buffers.wait_for_all_writes()
    buffers.release()
    assert registered == []


def test_failure_before_registration_does_not_hide_an_unfinished_write(make_transfer_buffers) -> None:
    """A failed cell must not hide unfinished readers registered after its failure."""
    buffers, registered = make_transfer_buffers(num_buffers=1, write_timeout=0)
    buffer = buffers.acquire()
    queued = Future()
    pending = _pending_write("cell-a", queued)
    pending.cell_updater.mark_errored(RuntimeError("already failed"))
    buffers.release_after(buffer, [pending])

    assert buffers.acquire() is buffer

    with pytest.raises(RuntimeError, match="still running"):
        buffers.release()
    assert registered == [(buffer.data_ptr(), buffer.numel())]
    queued.set_result(None)
    buffers.release()
    assert registered == []


def test_completion_before_tracking_does_not_retain_the_future(make_transfer_buffers) -> None:
    """An already completed write must allow buffer reuse and allocation release."""
    buffers, registered = make_transfer_buffers(num_buffers=1, write_timeout=0)
    buffer = buffers.acquire()
    done = Future()
    done.set_result(None)
    buffers.release_after(buffer, [_pending_write("cell-a", done)])

    assert buffers.acquire() is buffer
    buffers.release()
    assert registered == []
