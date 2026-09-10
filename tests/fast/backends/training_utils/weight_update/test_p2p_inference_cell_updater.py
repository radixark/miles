import threading
import time
from concurrent.futures import wait
from pathlib import Path

import pytest

from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils import fault_hooks
from miles.utils.test_utils.fault_hooks import FaultHookRegistry, FaultHookRequest

_REGISTRY = {"layer.0": (0x1000, 4, 2), "layer.1": (0x2000, 8, 2)}
_NAMES = ["layer.0", "layer.1"]


class _RecordingTransferEngine:
    def __init__(
        self,
        failing_sessions: set[str] | None = None,
        gate: threading.Event | None = None,
        started: threading.Event | None = None,
    ) -> None:
        self.writes: list[tuple[str, list[int], list[int], list[int]]] = []
        self._failing_sessions = failing_sessions if failing_sessions is not None else set()
        self._gate = gate
        self._started = started

    def batch_transfer_sync_write(
        self, session_id: str, source_ptrs: list[int], target_ptrs: list[int], source_lens: list[int]
    ) -> int:
        if self._started is not None:
            self._started.set()
        if self._gate is not None:
            assert self._gate.wait(timeout=30.0)
        self.writes.append((session_id, list(source_ptrs), list(target_ptrs), list(source_lens)))
        return -1 if session_id in self._failing_sessions else 0


def _remote_weight_info(utils, session_id: str, base_address: int, names: list[str] | None = None):
    return utils.RemoteWeightInfo(
        session_id,
        {
            name: utils.RemoteWeightLocation(base_address + index, _REGISTRY[name][1], _REGISTRY[name][2])
            for index, name in enumerate(names if names is not None else _NAMES)
        },
    )


def _cell_updater(module, transfer_timeout, engine, *, cell_id: str, targets: dict[int, object], health=None):
    if health is None:
        health = InferenceCellHealth([cell_id])
    return module.P2PInferenceCellUpdater(
        cell_id=cell_id,
        transfer_engine=engine,
        health=health,
        transfer_timeout=transfer_timeout,
        targets_by_engine_rank=targets,
    )


@pytest.fixture
def transfer_timeout() -> float:
    return 30.0


class TestTargetRouting:
    """One cell updater drives one inference cell across all of its engine ranks."""

    def test_queued_write_reaches_its_fault_hook_before_native_transfer(
        self, p2p_inference_cell_updater, p2p_transfer_utils, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The actual write thread dispatches the scoped hook before transferring any bytes."""
        event_logger = EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"))
        monkeypatch.setattr(fault_hooks, "get_event_logger", lambda: event_logger)
        registry = FaultHookRegistry()
        request = FaultHookRequest(
            request_id="p2p-hit",
            instance_id=registry.instance_id,
            hook="trainer_before_weight_send",
            mode="exit",
        )
        registry.arm(request)
        engine = _RecordingTransferEngine()
        updater = _cell_updater(
            p2p_inference_cell_updater,
            30.0,
            engine,
            cell_id="cell-hook",
            targets={0: _remote_weight_info(p2p_transfer_utils, "session-hook", 0xA000)},
        )
        threads: list[int] = []

        def terminate(**kwargs) -> None:
            threads.append(threading.get_native_id())
            assert engine.writes == []
            raise RuntimeError("Injected write failure")

        monkeypatch.setattr(fault_hooks, "inject_fault", terminate)
        try:
            with registry.weight_update_scope(weight_version=37, update_id="update-37"):
                future = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
            assert future is not None
            with pytest.raises(RuntimeError, match="Injected write failure"):
                future.result(timeout=30.0)
            record = registry.read(request_id=request.request_id, instance_id=registry.instance_id)
            assert record.weight_version == 37
            assert record.update_id == "update-37"
            assert record.status == "failed"
            assert len(threads) == 1 and threads[0] != threading.get_native_id()
            assert engine.writes == []
        finally:
            assert updater.dispose() is None

    def test_two_cells_at_the_same_engine_rank_write_to_their_own_sessions(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """The engine ranks share a CPU replica, so a cell writing into another cell's session corrupts that cell."""
        engine = _RecordingTransferEngine()
        first = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )
        second = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-1",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-1-rank-0", 0xB000)},
        )

        first.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        second.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        first.wait_for_pending_writes()
        second.wait_for_pending_writes()

        assert sorted(session_id for session_id, _s, _t, _l in engine.writes) == ["cell-0-rank-0", "cell-1-rank-0"]
        by_session = {session_id: target_ptrs for session_id, _s, target_ptrs, _l in engine.writes}
        assert by_session["cell-0-rank-0"] == [0xA000, 0xA001]
        assert by_session["cell-1-rank-0"] == [0xB000, 0xB001]

    def test_a_cell_writes_to_the_target_of_the_requested_engine_rank(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """One cell owns several TP ranks, and each rank holds a different shard of the model."""
        engine = _RecordingTransferEngine()
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={
                0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000),
                1: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-1", 0xC000),
            },
        )

        updater.submit_write(engine_rank=1, names=_NAMES, weight_memory_registry=_REGISTRY)
        updater.wait_for_pending_writes()

        assert [session_id for session_id, _s, _t, _l in engine.writes] == ["cell-0-rank-1"]
        assert engine.writes[0][2] == [0xC000, 0xC001]

    def test_every_target_is_written_from_the_same_shared_source_buffers(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """All cells share one set of pinned CPU buffers, so any per-cell copy of the source is a bug."""
        engine = _RecordingTransferEngine()
        first = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )
        second = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-1",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-1-rank-0", 0xB000)},
        )

        first.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        second.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        first.wait_for_pending_writes()
        second.wait_for_pending_writes()

        assert {tuple(source_ptrs) for _sid, source_ptrs, _t, _l in engine.writes} == {(0x1000, 0x2000)}
        assert {tuple(source_lens) for _sid, _s, _t, source_lens in engine.writes} == {(8, 16)}


class TestSubmissionSemantics:
    """Writes run in the background so the last engine rank never blocks the bucket stream."""

    def test_a_write_is_left_running_in_the_background(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """Blocking inside submit_write would serialize every cell and defeat the fire-and-forget last rank."""
        gate = threading.Event()
        engine = _RecordingTransferEngine(gate=gate)
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )

        try:
            future = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
            assert not future.done()
            assert updater._pending_writes == [future]
        finally:
            gate.set()

        updater.wait_for_pending_writes()

        assert future.done()

    def test_a_rejected_transfer_surfaces_through_the_future(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """A failed RDMA write must not be swallowed at submission, where nobody is watching for it."""
        engine = _RecordingTransferEngine(failing_sessions={"cell-0-rank-0"})
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )

        future = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)

        with pytest.raises(RuntimeError, match="cell-0-rank-0"):
            future.result(timeout=30.0)

    def test_an_unregistered_source_parameter_is_rejected(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """Writing a parameter whose pinned buffer was never registered would send an arbitrary address."""
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            _RecordingTransferEngine(),
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )

        future = updater.submit_write(engine_rank=0, names=["missing"], weight_memory_registry=_REGISTRY)

        with pytest.raises(AssertionError, match="missing"):
            future.result(timeout=30.0)

    def test_a_target_that_does_not_expect_every_parameter_is_rejected(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """Fewer target addresses than sources would pair the wrong buffers and silently corrupt the target."""
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            _RecordingTransferEngine(),
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000, names=["layer.0"])},
        )

        future = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)

        with pytest.raises(AssertionError, match="Pointer count mismatch"):
            future.result(timeout=30.0)

    def test_a_target_buffer_of_another_size_is_rejected(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """A target that registered a smaller buffer would be written past its end by this source span."""
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            _RecordingTransferEngine(),
            cell_id="cell-0",
            targets={
                0: p2p_transfer_utils.RemoteWeightInfo(
                    "cell-0-rank-0",
                    {
                        "layer.0": p2p_transfer_utils.RemoteWeightLocation(0xA000, 4, 2),
                        "layer.1": p2p_transfer_utils.RemoteWeightLocation(0xA001, 4, 2),
                    },
                )
            },
        )

        future = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)

        with pytest.raises(AssertionError, match="run past the target buffer"):
            future.result(timeout=30.0)


class TestErrorState:
    """An errored cell stops receiving work while its neighbours keep being written to."""

    def test_an_errored_cell_submits_nothing(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """Writing on after the first failure keeps a doomed engine busy and delays the healthy cells."""
        engine = _RecordingTransferEngine()
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )
        updater.mark_errored(RuntimeError("boom"))

        assert updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY) is None

        updater.wait_for_pending_writes()

        assert engine.writes == []
        assert updater._pending_writes == []

    def test_only_the_errored_cell_stops_being_written(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """Isolating a failed target is the whole point: the other cells must still get the full bucket."""
        engine = _RecordingTransferEngine()
        health = InferenceCellHealth(["cell-0", "cell-1"])
        first = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
            health=health,
        )
        second = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-1",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-1-rank-0", 0xB000)},
            health=health,
        )
        first.mark_errored(RuntimeError("boom"))

        first.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        second.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        first.wait_for_pending_writes()
        second.wait_for_pending_writes()

        assert [session_id for session_id, _s, _t, _l in engine.writes] == ["cell-1-rank-0"]
        assert second.is_errored is False
        assert health.errored_cell_ids == ["cell-0"]

    def test_the_first_failure_of_a_cell_is_the_one_reported(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """Later failures are consequences of the first one, which is the evidence worth keeping."""
        first_error = RuntimeError("the transfer engine rejected the write")
        health = InferenceCellHealth(["cell-0"])
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            _RecordingTransferEngine(),
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
            health=health,
        )

        updater.mark_errored(first_error)
        updater.mark_errored(RuntimeError("resume also failed"))

        assert updater.is_errored is True
        assert health.error_of("cell-0") is first_error


class TestStickyRefusal:
    """A cell that was errored or disposed must not write, not even from work queued before that."""

    def test_a_queued_write_is_skipped_once_the_cell_has_failed(
        self, p2p_inference_cell_updater, p2p_transfer_utils
    ) -> None:
        """The queued write runs only when the stuck one returns, long after the cell was given up on."""
        started = threading.Event()
        release = threading.Event()
        engine = _RecordingTransferEngine(gate=release, started=started)
        updater = _cell_updater(
            p2p_inference_cell_updater,
            30.0,
            engine,
            cell_id="cell-0",
            targets={
                0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000),
                1: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-1", 0xC000),
            },
        )

        try:
            updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
            assert started.wait(timeout=30.0)
            queued = updater.submit_write(engine_rank=1, names=_NAMES, weight_memory_registry=_REGISTRY)
            updater.mark_errored(RuntimeError("the cell was given up on while the first write was stuck"))
            release.set()
            queued.result(timeout=30.0)

            assert [session_id for session_id, _s, _t, _l in engine.writes] == ["cell-0-rank-0"]
        finally:
            release.set()
            updater.dispose()

    def test_a_queued_write_is_skipped_once_the_cell_was_disposed(
        self, p2p_inference_cell_updater, p2p_transfer_utils
    ) -> None:
        """The engine of a disposed incarnation was replaced, so its session id addresses somebody else's memory."""
        started = threading.Event()
        release = threading.Event()
        engine = _RecordingTransferEngine(gate=release, started=started)
        updater = _cell_updater(
            p2p_inference_cell_updater,
            30.0,
            engine,
            cell_id="cell-0",
            targets={
                0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000),
                1: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-1", 0xC000),
            },
        )
        stalled = None

        try:
            updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
            assert started.wait(timeout=30.0)
            queued = updater.submit_write(engine_rank=1, names=_NAMES, weight_memory_registry=_REGISTRY)
            stalled = updater.dispose()
            release.set()
            queued.result(timeout=30.0)

            assert [session_id for session_id, _s, _t, _l in engine.writes] == ["cell-0-rank-0"]
            assert updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY) is None
        finally:
            release.set()
            if stalled is not None:
                stalled.close(timeout=30.0)


class TestWriteCollection:
    """A write that fails is the failure of the cell it was addressed to, not of the trainer rank."""

    def test_a_failed_write_errors_only_its_own_cell(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """Raising out of the bucket stream would abandon the update for every other cell at once."""
        engine = _RecordingTransferEngine(failing_sessions={"cell-0-rank-0"})
        health = InferenceCellHealth(["cell-0", "cell-1"])
        first = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
            health=health,
        )
        second = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-1",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-1-rank-0", 0xB000)},
            health=health,
        )

        first.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        second.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        first.wait_for_pending_writes()
        second.wait_for_pending_writes()

        assert health.errored_cell_ids == ["cell-0"]
        assert isinstance(health.error_of("cell-0"), RuntimeError)
        assert second.is_errored is False
        assert first._pending_writes == []

    def test_completed_writes_are_collected_even_when_one_failed(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """Collect every completed write even when an earlier result reports a failure."""
        engine = _RecordingTransferEngine(failing_sessions={"cell-0-rank-0"})
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={
                0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000),
                1: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-1", 0xC000),
            },
        )

        futures = [
            updater.submit_write(engine_rank=rank, names=_NAMES, weight_memory_registry=_REGISTRY) for rank in (0, 1)
        ]
        _done, pending = wait(futures, timeout=30.0)
        assert not pending
        updater.wait_for_pending_writes()

        assert sorted(session_id for session_id, _s, _t, _l in engine.writes) == ["cell-0-rank-0", "cell-0-rank-1"]
        assert updater.is_errored is True
        assert updater._pending_writes == []

    def test_a_single_awaited_write_is_attributed_without_touching_the_others(
        self, p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout
    ) -> None:
        """The non-last engine rank is awaited one write at a time, and the last rank must stay in flight."""
        engine = _RecordingTransferEngine(failing_sessions={"cell-0-rank-0"})
        updater = _cell_updater(
            p2p_inference_cell_updater,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={
                0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000),
                1: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-1", 0xC000),
            },
        )

        first = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        updater.wait_for_write(first)
        left_running = updater._pending_writes

        assert updater.is_errored is True
        assert left_running == []
        assert updater.submit_write(engine_rank=1, names=_NAMES, weight_memory_registry=_REGISTRY) is None

    def test_a_write_that_outlives_the_timeout_stays_tracked(
        self, p2p_inference_cell_updater, p2p_transfer_utils
    ) -> None:
        """A timed-out write is still reading the shared buffers, so forgetting it loses the only handle on it."""
        gate = threading.Event()
        engine = _RecordingTransferEngine(gate=gate)
        updater = _cell_updater(
            p2p_inference_cell_updater,
            0.05,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )

        try:
            future = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
            updater.wait_for_pending_writes()

            assert updater.is_errored is True
            assert updater._pending_writes == [future]
            assert updater.take_unfinished_writes() == [future]
        finally:
            gate.set()

        assert future.result(timeout=30.0) is None


class TestCollectionBudget:
    """A cell that lost one write is collected without paying the transfer timeout again for each queued one."""

    def _stuck_cell(self, module, utils, *, transfer_timeout: float, engine_ranks: list[int]):
        gate = threading.Event()
        engine = _RecordingTransferEngine(gate=gate)
        updater = _cell_updater(
            module,
            transfer_timeout,
            engine,
            cell_id="cell-0",
            targets={
                rank: _remote_weight_info(utils, f"cell-0-rank-{rank}", 0xA000 + 0x1000 * rank)
                for rank in engine_ranks
            },
        )
        futures = [
            updater.submit_write(engine_rank=rank, names=_NAMES, weight_memory_registry=_REGISTRY)
            for rank in engine_ranks
        ]
        return gate, engine, updater, futures

    def test_the_queued_writes_of_a_failed_cell_are_not_each_waited_for(
        self, p2p_inference_cell_updater, p2p_transfer_utils
    ) -> None:
        """One stuck write plus a bucket stream would multiply the timeout until the trainer deadline fires."""
        gate, _engine, updater, futures = self._stuck_cell(
            p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout=0.2, engine_ranks=[0, 1, 2, 3, 4]
        )

        try:
            started = time.monotonic()
            updater.wait_for_pending_writes()
            elapsed = time.monotonic() - started
        finally:
            gate.set()

        assert elapsed < 0.9
        assert updater.is_errored is True
        assert all(future.cancelled() for future in futures[1:])
        assert updater._pending_writes == [futures[0]]

    def test_no_queued_write_runs_after_the_stuck_one_returns(
        self, p2p_inference_cell_updater, p2p_transfer_utils
    ) -> None:
        """A late write to a cell the trainer already gave up on reaches an engine that is being replaced."""
        gate, engine, updater, futures = self._stuck_cell(
            p2p_inference_cell_updater, p2p_transfer_utils, transfer_timeout=0.05, engine_ranks=[0, 1, 2]
        )

        updater.wait_for_pending_writes()
        gate.set()
        futures[0].result(timeout=30.0)

        assert [session_id for session_id, _s, _t, _l in engine.writes] == ["cell-0-rank-0"]

    def test_a_healthy_cell_still_gets_its_full_budget(self, p2p_inference_cell_updater, p2p_transfer_utils) -> None:
        """The shortcut must apply to failed cells only; a slow healthy write has to be waited for."""
        gate = threading.Event()
        gate.set()
        engine = _RecordingTransferEngine(gate=gate)
        updater = _cell_updater(
            p2p_inference_cell_updater,
            30.0,
            engine,
            cell_id="cell-0",
            targets={
                0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000),
                1: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-1", 0xB000),
            },
        )

        updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        updater.submit_write(engine_rank=1, names=_NAMES, weight_memory_registry=_REGISTRY)
        updater.wait_for_pending_writes()

        assert updater.is_errored is False
        assert sorted(session_id for session_id, _s, _t, _l in engine.writes) == ["cell-0-rank-0", "cell-0-rank-1"]
        assert updater._pending_writes == []
