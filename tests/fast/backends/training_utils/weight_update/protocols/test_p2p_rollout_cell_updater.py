import threading
from concurrent.futures import Future
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


class _RecordingTransferManager:
    def __init__(self) -> None:
        self.submissions: list[tuple] = []

    def submit(self, fn, *args) -> Future:
        self.submissions.append(args)
        future: Future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as e:
            future.set_exception(e)
        return future


def _cell_updater(p2p_rollout_cell_updater: ModuleType, cell_id: str = "cell-a") -> Any:
    return p2p_rollout_cell_updater._P2PRolloutCellUpdater(cell_id=cell_id, api_client=None)


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
        manager = _RecordingTransferManager()
        engine = _RecordingTransferEngine()

        updater.submit_write(
            rollout_engine_rank=1,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=engine,
            transfer_manager=manager,
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
        manager = _RecordingTransferManager()
        engine = _RecordingTransferEngine()
        for _ in range(2):
            updater.submit_write(
                rollout_engine_rank=0,
                names=["w"],
                weight_memory_registry={"w": (0x30, 2, 4)},
                transfer_engine=engine,
                transfer_manager=manager,
            )

        updater.wait_for_pending_writes()
        updater.wait_for_pending_writes()

        assert len(manager.submissions) == 2
        assert updater._pending_writes == []

    def test_a_broken_write_surfaces_when_its_own_updater_is_drained(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """The failure must reach the cell whose write it was, not whichever cell is drained first."""
        broken = _cell_updater(p2p_rollout_cell_updater, cell_id="cell-broken")
        healthy = _cell_updater(p2p_rollout_cell_updater, cell_id="cell-healthy")
        for updater, session in ((broken, "session-broken"), (healthy, "session-healthy")):
            updater.targets_by_rollout_engine_rank = {
                0: _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, session, {"w": (0x1000, 2, 4)})
            }
        manager = _RecordingTransferManager()
        broken.submit_write(
            rollout_engine_rank=0,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=_RecordingTransferEngine(return_code=-1),
            transfer_manager=manager,
        )
        healthy.submit_write(
            rollout_engine_rank=0,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=_RecordingTransferEngine(),
            transfer_manager=manager,
        )

        with pytest.raises(RuntimeError):
            broken.wait_for_pending_writes()
        healthy.wait_for_pending_writes()

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
            transfer_manager=_RecordingTransferManager(),
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
        manager = _RecordingTransferManager()
        updater.submit_write(
            rollout_engine_rank=0,
            names=["w"],
            weight_memory_registry={"w": (0x30, 2, 4)},
            transfer_engine=_RecordingTransferEngine(),
            transfer_manager=manager,
        )
        updater.mark_errored(RuntimeError("lost"))

        updater.wait_for_pending_writes()

        assert len(updater._pending_writes) == 1
