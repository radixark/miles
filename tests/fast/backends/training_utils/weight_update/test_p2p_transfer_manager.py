import threading
from concurrent.futures import Future, TimeoutError

import pytest


def _blocking_task(started: threading.Event, release: threading.Event):
    def run() -> None:
        started.set()
        assert release.wait(timeout=30.0)

    return run


class TestSubmit:
    """Submission of P2P writes to the background executor."""

    def test_a_future_handed_to_the_caller_is_still_tracked_for_the_bulk_wait(self, p2p_transfer_utils) -> None:
        """The last engine rank never calls result() itself, so its write must still be awaited by wait_transfers."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)
        started, release = threading.Event(), threading.Event()

        future = manager.submit(_blocking_task(started, release))
        assert started.wait(timeout=30.0)

        assert manager.transfer_futures == [future]
        assert not future.done()

        release.set()
        manager.wait_transfers()

        assert future.done()
        assert manager.transfer_futures == []

    def test_every_submitted_write_runs_before_the_bulk_wait_returns(self, p2p_transfer_utils) -> None:
        """More writes than workers are submitted per bucket, so the queued ones must finish too."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)
        lock = threading.Lock()
        finished: list[int] = []

        for index in range(6):

            def run(index: int = index) -> None:
                with lock:
                    finished.append(index)

            manager.submit(run)

        manager.wait_transfers()

        assert sorted(finished) == list(range(6))


class TestWaitTransfers:
    """Collection of the background P2P writes at the end of a weight update."""

    def test_a_completed_round_of_transfers_finishes_quietly(self, p2p_transfer_utils) -> None:
        """The happy path must stay silent and forget the futures it already collected."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)
        manager.submit(lambda: None)

        manager.wait_transfers()

        assert manager.transfer_futures == []

    def test_a_failed_transfer_is_raised_to_the_caller(self, p2p_transfer_utils) -> None:
        """A silently logged RDMA failure would publish a half-written weight version, so it must propagate."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)

        def failing() -> None:
            raise RuntimeError("[P2P-Shared] Transfer failed for session s0, error: -1")

        manager.submit(failing)

        with pytest.raises(RuntimeError, match="error: -1"):
            manager.wait_transfers()

    def test_a_timed_out_transfer_propagates_timeout(self, p2p_transfer_utils) -> None:
        """An unfinished write must fail the update when its wait deadline expires."""
        manager = p2p_transfer_utils.P2PTransferManager(transfer_timeout=0)
        manager.transfer_futures.append(Future())

        with pytest.raises(TimeoutError):
            manager.wait_transfers()
