import threading

import pytest

from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth

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


def _cell_updater(module, manager, engine, *, cell_id: str, targets: dict[int, object], health=None):
    if health is None:
        health = InferenceCellHealth([cell_id])
    return module.P2PInferenceCellUpdater(
        cell_id=cell_id,
        transfer_engine=engine,
        transfer_manager=manager,
        health=health,
        targets_by_engine_rank=targets,
    )


@pytest.fixture
def manager(p2p_transfer_utils):
    return p2p_transfer_utils.P2PTransferManager(num_workers=4, transfer_timeout=30.0)


class TestTargetRouting:
    """One cell updater drives one inference cell across all of its engine ranks."""

    def test_two_cells_at_the_same_engine_rank_write_to_their_own_sessions(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """The engine ranks share a CPU replica, so a cell writing into another cell's session corrupts that cell."""
        engine = _RecordingTransferEngine()
        first = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )
        second = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-1",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-1-rank-0", 0xB000)},
        )

        first.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        second.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        manager.wait_transfers()

        assert sorted(session_id for session_id, _s, _t, _l in engine.writes) == ["cell-0-rank-0", "cell-1-rank-0"]
        by_session = {session_id: target_ptrs for session_id, _s, target_ptrs, _l in engine.writes}
        assert by_session["cell-0-rank-0"] == [0xA000, 0xA001]
        assert by_session["cell-1-rank-0"] == [0xB000, 0xB001]

    def test_a_cell_writes_to_the_target_of_the_requested_engine_rank(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """One cell owns several TP ranks, and each rank holds a different shard of the model."""
        engine = _RecordingTransferEngine()
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-0",
            targets={
                0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000),
                1: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-1", 0xC000),
            },
        )

        updater.submit_write(engine_rank=1, names=_NAMES, weight_memory_registry=_REGISTRY)
        manager.wait_transfers()

        assert [session_id for session_id, _s, _t, _l in engine.writes] == ["cell-0-rank-1"]
        assert engine.writes[0][2] == [0xC000, 0xC001]

    def test_every_target_is_written_from_the_same_shared_source_buffers(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """All cells share one set of pinned CPU buffers, so any per-cell copy of the source is a bug."""
        engine = _RecordingTransferEngine()
        first = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )
        second = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-1",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-1-rank-0", 0xB000)},
        )

        first.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        second.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        manager.wait_transfers()

        assert {tuple(source_ptrs) for _sid, source_ptrs, _t, _l in engine.writes} == {(0x1000, 0x2000)}
        assert {tuple(source_lens) for _sid, _s, _t, source_lens in engine.writes} == {(8, 16)}


class TestSubmissionSemantics:
    """Writes run in the background so the last engine rank never blocks the bucket stream."""

    def test_a_write_is_left_running_in_the_background(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """Blocking inside submit_write would serialize every cell and defeat the fire-and-forget last rank."""
        gate = threading.Event()
        engine = _RecordingTransferEngine(gate=gate)
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )

        try:
            future = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
            assert not future.done()
            assert manager.transfer_futures == [future]
        finally:
            gate.set()

        manager.wait_transfers()

        assert future.done()

    def test_a_rejected_transfer_surfaces_through_the_future(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """A failed RDMA write must not be swallowed at submission, where nobody is watching for it."""
        engine = _RecordingTransferEngine(failing_sessions={"cell-0-rank-0"})
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )

        future = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)

        with pytest.raises(RuntimeError, match="cell-0-rank-0"):
            future.result(timeout=30.0)

    def test_an_unregistered_source_parameter_is_rejected(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """Writing a parameter whose pinned buffer was never registered would send an arbitrary address."""
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            _RecordingTransferEngine(),
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )

        future = updater.submit_write(engine_rank=0, names=["missing"], weight_memory_registry=_REGISTRY)

        with pytest.raises(AssertionError, match="missing"):
            future.result(timeout=30.0)

    def test_a_target_that_does_not_expect_every_parameter_is_rejected(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """Fewer target addresses than sources would pair the wrong buffers and silently corrupt the target."""
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            _RecordingTransferEngine(),
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000, names=["layer.0"])},
        )

        future = updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)

        with pytest.raises(AssertionError, match="Pointer count mismatch"):
            future.result(timeout=30.0)

    def test_a_target_buffer_of_another_size_is_rejected(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """A target that registered a smaller buffer would be written past its end by this source span."""
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
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

    def test_an_errored_cell_submits_nothing(self, p2p_inference_cell_updater, p2p_transfer_utils, manager) -> None:
        """Writing on after the first failure keeps a doomed engine busy and delays the healthy cells."""
        engine = _RecordingTransferEngine()
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
        )
        updater.mark_errored(RuntimeError("boom"))

        assert updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY) is None

        manager.wait_transfers()

        assert engine.writes == []
        assert manager.transfer_futures == []

    def test_only_the_errored_cell_stops_being_written(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """Isolating a failed target is the whole point: the other cells must still get the full bucket."""
        engine = _RecordingTransferEngine()
        health = InferenceCellHealth(["cell-0", "cell-1"])
        first = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-0",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-0-rank-0", 0xA000)},
            health=health,
        )
        second = _cell_updater(
            p2p_inference_cell_updater,
            manager,
            engine,
            cell_id="cell-1",
            targets={0: _remote_weight_info(p2p_transfer_utils, "cell-1-rank-0", 0xB000)},
            health=health,
        )
        first.mark_errored(RuntimeError("boom"))

        first.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        second.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY)
        manager.wait_transfers()

        assert [session_id for session_id, _s, _t, _l in engine.writes] == ["cell-1-rank-0"]
        assert second.is_errored is False
        assert health.errored_cell_ids == ["cell-0"]

    def test_the_first_failure_of_a_cell_is_the_one_reported(
        self, p2p_inference_cell_updater, p2p_transfer_utils, manager
    ) -> None:
        """Later failures are consequences of the first one, which is the evidence worth keeping."""
        first_error = RuntimeError("the transfer engine rejected the write")
        health = InferenceCellHealth(["cell-0"])
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
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
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=1, transfer_timeout=30.0)
        engine = _RecordingTransferEngine(gate=release, started=started)
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
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
            if manager.executor is not None:
                manager.executor.shutdown(wait=True)

    def test_a_queued_write_is_skipped_once_the_cell_was_disposed(
        self, p2p_inference_cell_updater, p2p_transfer_utils
    ) -> None:
        """The engine of a disposed incarnation was replaced, so its session id addresses somebody else's memory."""
        started = threading.Event()
        release = threading.Event()
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=1, transfer_timeout=30.0)
        engine = _RecordingTransferEngine(gate=release, started=started)
        updater = _cell_updater(
            p2p_inference_cell_updater,
            manager,
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
            updater.dispose()
            release.set()
            queued.result(timeout=30.0)

            assert [session_id for session_id, _s, _t, _l in engine.writes] == ["cell-0-rank-0"]
            assert updater.submit_write(engine_rank=0, names=_NAMES, weight_memory_registry=_REGISTRY) is None
        finally:
            release.set()
            if manager.executor is not None:
                manager.executor.shutdown(wait=True)
