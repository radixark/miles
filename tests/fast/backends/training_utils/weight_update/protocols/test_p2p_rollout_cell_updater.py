import threading
import time

_TRANSFER_TIMEOUT = 30.0
from argparse import Namespace
from types import ModuleType
from typing import Any

import pytest


class _RecordingTransferEngine:
    def __init__(self, return_code: int = 0, error: Exception | None = None, gate: threading.Event | None = None):
        self.calls: list[tuple[str, list[int], list[int], list[int]]] = []
        self._return_code = return_code
        self._error = error
        self._gate = gate

    def batch_transfer_sync_write(
        self, session_id: str, source_ptrs: list[int], target_ptrs: list[int], source_lens: list[int]
    ) -> int:
        if self._gate is not None:
            self._gate.wait(timeout=30)
        self.calls.append((session_id, list(source_ptrs), list(target_ptrs), list(source_lens)))
        if self._error is not None:
            raise self._error
        return self._return_code


def _remote_session(
    p2p_rollout_cell_updater: ModuleType,
    p2p_transfer_utils: ModuleType,
    session_id: str,
    weights: dict[str, tuple[int, int, int]],
) -> Any:
    return p2p_rollout_cell_updater.RemoteWeightInfo(
        session_id,
        {name: p2p_transfer_utils.RemoteWeightLocation(*location) for name, location in weights.items()},
    )


class TestDoP2PWriteOneSession:
    """The single-session write from the shared CPU pinned buffers."""

    def test_every_registered_weight_is_written_in_one_batch(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """One batch call carries the whole bucket, addressed source pointer by target pointer."""
        transfer_engine = _RecordingTransferEngine()
        session = _remote_session(
            p2p_rollout_cell_updater,
            p2p_transfer_utils,
            "session-a",
            {"w0": (0x2000, 4, 2), "w1": (0x3000, 8, 2)},
        )

        p2p_rollout_cell_updater._do_p2p_write_one_session(
            transfer_engine,
            session,
            ["w0", "w1"],
            {"w0": (0x1000, 4, 2), "w1": (0x1100, 8, 2)},
        )

        assert transfer_engine.calls == [("session-a", [0x1000, 0x1100], [0x2000, 0x3000], [8, 16])]

    def test_an_unregistered_weight_is_rejected(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A weight with no local registration has no pinned source buffer to read from."""
        session = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-a", {"w0": (0x2000, 4, 2)})

        with pytest.raises(AssertionError, match="_weight_memory_registry of w0"):
            p2p_rollout_cell_updater._do_p2p_write_one_session(_RecordingTransferEngine(), session, ["w0"], {})

    def test_a_weight_the_remote_session_does_not_hold_is_rejected(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """Skipping a weight silently would pair the remaining source pointers with the wrong targets."""
        session = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-a", {"w0": (0x2000, 4, 2)})

        with pytest.raises(AssertionError, match="Pointer count mismatch"):
            p2p_rollout_cell_updater._do_p2p_write_one_session(
                _RecordingTransferEngine(),
                session,
                ["w0", "w1"],
                {"w0": (0x1000, 4, 2), "w1": (0x1100, 4, 2)},
            )

    def test_a_weight_spanning_a_different_number_of_bytes_remotely_is_rejected(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """Writing a longer source into a shorter target buffer would run past the remote allocation."""
        session = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-a", {"w0": (0x2000, 4, 1)})

        with pytest.raises(AssertionError, match="run past the target buffer"):
            p2p_rollout_cell_updater._do_p2p_write_one_session(
                _RecordingTransferEngine(), session, ["w0"], {"w0": (0x1000, 4, 2)}
            )

    def test_an_empty_bucket_transfers_nothing(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A bucket with no name must not issue an empty batch to the transfer engine."""
        transfer_engine = _RecordingTransferEngine()
        session = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-a", {"w0": (0x2000, 4, 2)})

        p2p_rollout_cell_updater._do_p2p_write_one_session(transfer_engine, session, [], {"w0": (0x1000, 4, 2)})

        assert transfer_engine.calls == []

    def test_a_failing_return_code_raises(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A negative return code means the remote never received the weights."""
        session = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-a", {"w0": (0x2000, 4, 2)})

        with pytest.raises(RuntimeError, match="Transfer failed for session session-a"):
            p2p_rollout_cell_updater._do_p2p_write_one_session(
                _RecordingTransferEngine(return_code=-1), session, ["w0"], {"w0": (0x1000, 4, 2)}
            )


def _cell_updater(p2p_rollout_cell_updater: ModuleType, cell_id: str = "cell-a") -> Any:
    return p2p_rollout_cell_updater._P2PRolloutCellUpdater(
        args=Namespace(update_weight_engine_request_timeout=10.0), cell_id=cell_id, api_client=None
    )


class TestSubmitWrite:
    """One updater per rollout cell, each owning the writes addressed to that cell."""

    def test_a_write_is_addressed_to_the_remote_session_of_that_rollout_engine_rank(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """The rank decides which remote session the bytes land in, so picking the wrong one corrupts a peer."""
        updater = _cell_updater(p2p_rollout_cell_updater)
        updater.targets_by_rollout_engine_rank = {
            0: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-0", {"w": (0x1000, 2, 4)}),
            1: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-1", {"w": (0x2000, 2, 4)}),
        }
        engine = _RecordingTransferEngine()

        updater.submit_write(
            rollout_engine_rank=1,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=engine,
        )

        assert [session_id for session_id, _, _, _ in engine.calls] == ["session-1"]

    def test_every_submitted_write_is_drained_once_and_then_forgotten(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A drained future left in the queue would be waited on again by the next bucket."""
        updater = _cell_updater(p2p_rollout_cell_updater)
        updater.targets_by_rollout_engine_rank = {
            0: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-0", {"w": (0x1000, 2, 4)})
        }
        engine = _RecordingTransferEngine()
        for _ in range(2):
            updater.submit_write(
                rollout_engine_rank=0,
                names=["w"],
                weight_memory_registry={"w": (0x30, 2, 4)},
                transfer_engine=engine,
            )

        updater.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)
        updater.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)

        assert updater._pending_writes == []

    def test_a_broken_write_is_blamed_on_the_cell_it_was_addressed_to(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """The failure must mark the cell whose write it was, not whichever cell is drained first."""
        broken = _cell_updater(p2p_rollout_cell_updater, cell_id="cell-broken")
        healthy = _cell_updater(p2p_rollout_cell_updater, cell_id="cell-healthy")
        for updater, session in ((broken, "session-broken"), (healthy, "session-healthy")):
            updater.targets_by_rollout_engine_rank = {
                0: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, session, {"w": (0x1000, 2, 4)})
            }
        broken.submit_write(
            rollout_engine_rank=0,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=_RecordingTransferEngine(return_code=-1),
        )
        healthy.submit_write(
            rollout_engine_rank=0,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=_RecordingTransferEngine(),
        )

        broken.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)
        healthy.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)

        assert (broken.is_errored, healthy.is_errored) == (True, False)

    def test_each_updater_keeps_the_targets_of_its_own_rollout_cell(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """Sharing one target map across cells would send a cell's weights into another cell's memory."""
        first = _cell_updater(p2p_rollout_cell_updater, cell_id="cell-a")
        second = _cell_updater(p2p_rollout_cell_updater, cell_id="cell-b")
        first.targets_by_rollout_engine_rank = {
            0: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-first", {"w": (0x1000, 2, 4)})
        }

        assert second.targets_by_rollout_engine_rank == {}
        assert (first.cell_id, second.cell_id) == ("cell-a", "cell-b")


class TestErroredCellDropsItsWrites:
    """Once a cell has lost the update, nothing more may be written into its memory."""

    def test_a_write_submitted_after_the_failure_never_reaches_the_transfer_engine(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """The remote may already be serving again, so a late write would corrupt what it serves."""
        updater = _cell_updater(p2p_rollout_cell_updater)
        updater.targets_by_rollout_engine_rank = {
            0: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-0", {"w": (0x1000, 2, 4)})
        }
        updater.mark_errored(RuntimeError("lost"))
        engine = _RecordingTransferEngine()

        updater.submit_write(
            rollout_engine_rank=0,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=engine,
        )

        assert engine.calls == []

    def test_a_write_already_queued_is_skipped_when_the_cell_fails_before_it_runs(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """The queue is drained by another thread, so the check has to happen where the write runs."""
        updater = _cell_updater(p2p_rollout_cell_updater)
        target = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-0", {"w": (0x1000, 2, 4)})
        engine = _RecordingTransferEngine()
        updater.mark_errored(RuntimeError("lost"))

        updater._write_if_active(engine, target, ["w"], {"w": (0x30, 2, 4)})

        assert engine.calls == []

    def test_draining_an_errored_cell_is_a_noop(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """Waiting on the writes of a cell nobody will hear from again only burns the deadline."""
        updater = _cell_updater(p2p_rollout_cell_updater)
        updater.targets_by_rollout_engine_rank = {
            0: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-0", {"w": (0x1000, 2, 4)})
        }
        updater.submit_write(
            rollout_engine_rank=0,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=_RecordingTransferEngine(),
        )
        updater.mark_errored(RuntimeError("lost"))

        updater.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)

        assert len(updater._pending_writes) == 1


class TestDrainingBlamesTheCell:
    """Draining a cell's queue turns a broken write into that cell's error instead of raising."""

    def test_a_failed_write_is_reported_as_the_cells_error_and_not_raised(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """Raising here would abort the whole trainer instead of only giving up on this cell."""
        updater = _cell_updater(p2p_rollout_cell_updater)
        updater.targets_by_rollout_engine_rank = {
            0: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-0", {"w": (0x1000, 2, 4)})
        }
        updater.submit_write(
            rollout_engine_rank=0,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=_RecordingTransferEngine(return_code=-1),
        )

        updater.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)

        assert updater.is_errored is True

    def test_a_drained_queue_is_not_waited_on_again(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A future left behind would be re-raised against the next bucket that drains the cell."""
        updater = _cell_updater(p2p_rollout_cell_updater)
        updater.targets_by_rollout_engine_rank = {
            0: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-0", {"w": (0x1000, 2, 4)})
        }
        updater.submit_write(
            rollout_engine_rank=0,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=_RecordingTransferEngine(),
        )

        updater.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)

        assert updater._pending_writes == []


def _make_args() -> Namespace:
    return Namespace(update_weight_engine_request_timeout=10.0, p2p_transfer_timeout=_TRANSFER_TIMEOUT)


class _FakeFuture:
    def __init__(self, *, error: BaseException | None = None, blocks: bool = False) -> None:
        self.result_calls = 0
        self.cancelled = False
        self._error = error
        self._blocks = blocks

    def result(self, timeout: float | None = None) -> None:
        self.result_calls += 1
        if self._error is not None:
            raise self._error
        if self._blocks:
            assert timeout is not None
            time.sleep(timeout)
            raise TimeoutError("the write never finished")

    def cancel(self) -> bool:
        self.cancelled = True
        return True


class _FakeExecutor:
    def __init__(self, futures: list[_FakeFuture]) -> None:
        self.submitted = 0
        self._futures = futures

    def submit(self, fn: Any, *args: Any) -> _FakeFuture:
        future = self._futures[self.submitted]
        self.submitted += 1
        return future


def _make_updater(p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType, *, cell_id: str = "cell-0"):
    updater = p2p_rollout_cell_updater._P2PRolloutCellUpdater(args=_make_args(), cell_id=cell_id, api_client=object())
    updater.targets_by_rollout_engine_rank = {
        0: p2p_transfer_utils.RemoteWeightInfo(
            session_id=f"session-{cell_id}",
            weights_info={"w": p2p_transfer_utils.RemoteWeightLocation(address=0x2000, numel=4, element_size=2)},
        )
    }
    return updater


_REGISTRY = {"w": (0x1000, 4, 2)}


class TestPerCellWriteThread:
    def test_a_stuck_cell_does_not_hold_up_another_cells_write(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A shared pool lets one hung engine occupy the workers the healthy cells need."""
        stuck = _make_updater(p2p_rollout_cell_updater, p2p_transfer_utils, cell_id="cell-stuck")
        healthy = _make_updater(p2p_rollout_cell_updater, p2p_transfer_utils, cell_id="cell-healthy")
        gate = threading.Event()
        stuck_engine = _RecordingTransferEngine(gate=gate)
        healthy_engine = _RecordingTransferEngine()

        try:
            stuck.submit_write(
                rollout_engine_rank=0, names=["w"], weight_memory_registry=_REGISTRY, transfer_engine=stuck_engine
            )
            assert stuck_engine.entered.wait(timeout=10)
            healthy.submit_write(
                rollout_engine_rank=0, names=["w"], weight_memory_registry=_REGISTRY, transfer_engine=healthy_engine
            )
            healthy.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)

            assert healthy_engine.calls != []
            assert stuck_engine.calls == []
            assert not healthy.is_errored
        finally:
            gate.set()

    def test_two_cells_write_on_two_different_threads(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """One worker per cell is what keeps a hung write confined to its own cell."""
        first = _make_updater(p2p_rollout_cell_updater, p2p_transfer_utils, cell_id="cell-a")
        second = _make_updater(p2p_rollout_cell_updater, p2p_transfer_utils, cell_id="cell-b")
        gate = threading.Event()
        first_engine = _RecordingTransferEngine(gate=gate)
        second_engine = _RecordingTransferEngine()

        try:
            first.submit_write(
                rollout_engine_rank=0, names=["w"], weight_memory_registry=_REGISTRY, transfer_engine=first_engine
            )
            assert first_engine.entered.wait(timeout=10)
            second.submit_write(
                rollout_engine_rank=0, names=["w"], weight_memory_registry=_REGISTRY, transfer_engine=second_engine
            )
            second.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)

            assert first_engine.thread_idents[0] != second_engine.thread_idents[0]
        finally:
            gate.set()

    def test_one_cell_writes_its_own_buckets_in_submission_order(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A single worker per cell is what keeps a later bucket from overtaking an earlier one."""
        updater = _make_updater(p2p_rollout_cell_updater, p2p_transfer_utils)
        engine = _RecordingTransferEngine()

        for _ in range(4):
            updater.submit_write(
                rollout_engine_rank=0, names=["w"], weight_memory_registry=_REGISTRY, transfer_engine=engine
            )
        updater.wait_for_pending_writes(timeout=_TRANSFER_TIMEOUT)

        assert len(engine.calls) == 4
        assert len(set(engine.thread_idents)) == 1
