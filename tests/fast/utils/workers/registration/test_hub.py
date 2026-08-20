from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from miles.utils.workers.naming import _worker_name_of_cell, compute_cell_id
from miles.utils.workers.registration.hub import REPORTER_TTL_SECONDS, RegistrationHub
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import HostAndPort

_REPORTER = "miles-run-r1-inference"
_POOL_ID = f"{_REPORTER}-inference-engine-0-0"
_OTHER_REPORTER = "miles-run-r1-inference-b"
_OTHER_POOL_ID = f"{_OTHER_REPORTER}-inference-engine-0-0"
_PROVIDER_MODULE = "miles.utils.workers.registration.hub"
_RUN_UUID = "run-uuid-1"
_POLL_INTERVAL_SECONDS = 0.001


def _cell_id(cell_index: int, *, pool_id: str = _POOL_ID) -> str:
    return compute_cell_id(pool_id=pool_id, cell_index=cell_index)


def _cell(
    cell_index: int,
    *,
    host: str = "10.0.0.5",
    model_id: str = "default",
    generation: int = 0,
    reporter_id: str = _REPORTER,
    pool_id: str = _POOL_ID,
) -> RegisteredCellInfo:
    cell_id = _cell_id(cell_index, pool_id=pool_id)
    return RegisteredCellInfo(
        reporter_id=reporter_id,
        info=CellInfo(
            cell_id=cell_id,
            pool_id=pool_id,
            alive=True,
            worker_names=[_worker_name_of_cell(cell_id)],
            workers_hash=f"hash-{host}",
            meta=dict(model_id=model_id, worker_type="regular"),
        ),
        workers=[
            WorkerInfo(
                name=_worker_name_of_cell(cell_id),
                generation=generation,
                self_addrs={"primary": HostAndPort(host=host, port=8000)},
                gpu_ids=[0],
            )
        ],
    )


def _other_cell(cell_index: int, *, host: str = "10.0.0.5") -> RegisteredCellInfo:
    return _cell(cell_index, host=host, reporter_id=_OTHER_REPORTER, pool_id=_OTHER_POOL_ID)


def _snapshot(
    cells: list[RegisteredCellInfo], *, reporter_id: str = _REPORTER, sequence_number: int = 1
) -> RegistrationSnapshot:
    return RegistrationSnapshot(
        run_uuid=_RUN_UUID, reporter_id=reporter_id, sequence_number=sequence_number, cells=cells
    )


@pytest.fixture(autouse=True)
def _poll_without_delay():
    with patch(f"{_PROVIDER_MODULE}.REGISTERED_CELLS_POLL_INTERVAL_SECONDS", _POLL_INTERVAL_SECONDS):
        yield


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Watcher:
    def __init__(self) -> None:
        self.observations: list[tuple[str, CellInfo | None]] = []
        self.failing_cell_ids: set[str] = set()

    async def reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        if cell_id in self.failing_cell_ids:
            raise RuntimeError(f"cell {cell_id} refuses to be reconciled")
        self.observations.append((cell_id, observed))


async def _watched(**kwargs) -> tuple[RegistrationHub, _Watcher]:
    watcher = _Watcher()
    provider = RegistrationHub(run_uuid=_RUN_UUID, **kwargs)
    await _start_watch(provider, watcher)
    return provider, watcher


async def _start_watch(provider: RegistrationHub, watcher: _Watcher):
    return await provider.watch_cells(watcher.reconcile)


async def _apply(provider: RegistrationHub, snapshot: RegistrationSnapshot) -> None:
    await provider.ingest(snapshot)
    await _drain()


async def _drain() -> None:
    for _ in range(50):
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


class TestSnapshotMembership:
    async def test_a_first_snapshot_announces_every_cell_it_carries(self):
        """A snapshot is the whole membership of its deployment, so the run takes all of it in at once."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0), _cell(1)]))

        assert sorted(provider._cell_of_id) == [_cell_id(0), _cell_id(1)]
        assert [cell_id for cell_id, _observed in watcher.observations] == [_cell_id(0), _cell_id(1)]

    async def test_an_unchanged_cell_is_not_announced_twice(self):
        """The run adds a cell once; announcing it again would tear down a serving engine to rebuild it."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))
        await _apply(provider, _snapshot([_cell(0)], sequence_number=2))

        assert len(watcher.observations) == 1

    async def test_a_cell_that_changed_is_announced_anew(self):
        """A cell rebuilt on another host serves from another address, and the old one answers nothing."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))
        await _apply(provider, _snapshot([_cell(0, host="10.0.0.6")], sequence_number=2))

        (_first, (cell_id, observed)) = watcher.observations
        assert cell_id == _cell_id(0)
        assert observed.workers_hash == "hash-10.0.0.6"

    async def test_a_cell_the_snapshot_stops_naming_is_removed(self):
        """Membership is level, so omission is how a deployment says a cell is gone; there is no death message."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence_number=1))
        await _apply(provider, _snapshot([_cell(0)], sequence_number=2))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]
        assert watcher.observations[-1] == (_cell_id(1), None)

    async def test_a_late_snapshot_is_ignored(self):
        """A snapshot that crossed the wan slowly would otherwise resurrect cells the run already dropped."""
        provider, _watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence_number=5))
        await _apply(provider, _snapshot([_cell(0)], sequence_number=4))

        assert sorted(provider._cell_of_id) == [_cell_id(0), _cell_id(1)]

    async def test_a_cell_reported_by_two_deployments_is_refused(self):
        """One cell id names one cell, and the second owner could remove the cell of the first."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)]))

        with pytest.raises(AssertionError, match="is reported by both"):
            await provider.ingest(_snapshot([_cell(0, reporter_id="other")], reporter_id="other"))

    async def test_a_snapshot_of_another_run_is_refused(self):
        """A resume reusing a run id leaves the reporter of the previous training announcing into this one."""
        provider, _watcher = await _watched()

        with pytest.raises(AssertionError, match="carries run_uuid"):
            await provider.ingest(_snapshot([_cell(0)]).model_copy(update=dict(run_uuid="another-run")))

    async def test_a_snapshot_carrying_a_cell_of_another_reporter_is_refused(self):
        """A snapshot is the membership of one deployment, so a cell of another one was assembled in by mistake."""
        provider, _watcher = await _watched()

        with pytest.raises(AssertionError, match="snapshot of reporter"):
            await provider.ingest(_snapshot([_cell(0, reporter_id="other")]))

    async def test_a_cell_carried_twice_by_one_snapshot_is_refused(self):
        """Either entry could be the truth, so holding one would hold a membership nobody announced."""
        provider, _watcher = await _watched()

        with pytest.raises(AssertionError, match="more than once"):
            await provider.ingest(_snapshot([_cell(0), _cell(0, host="10.0.0.6")]))

        assert sorted(provider._cell_of_id) == []

    async def test_a_cell_that_does_not_name_its_own_pool_is_refused(self):
        """The run parses a cell id to address the workers of that cell."""
        provider, _watcher = await _watched()
        cell = _cell(0)
        cell = cell.model_copy(update=dict(info=cell.info.model_copy(update=dict(cell_id="not-a-cell-id"))))

        with pytest.raises(AssertionError, match="does not read as"):
            await provider.ingest(_snapshot([cell]))

    async def test_a_cell_whose_workers_belong_to_another_cell_is_refused(self):
        """The run addresses a worker by parsing its name, so a name from another cell would route nowhere."""
        provider, _watcher = await _watched()
        cell = _cell(0).model_copy(update=dict(workers=_cell(1).workers))

        with pytest.raises(AssertionError, match="name cells"):
            await provider.ingest(_snapshot([cell]))


class TestPartitioningCellsByReporter:
    async def test_a_snapshot_replaces_only_the_cells_of_the_reporter_that_sent_it(self):
        """A snapshot declares one deployment's membership, so taking it as the run's would drop every other engine."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0), _cell(1)]))
        await _apply(provider, _snapshot([_other_cell(0), _other_cell(1)], reporter_id=_OTHER_REPORTER))

        await _apply(provider, _snapshot([_other_cell(0)], reporter_id=_OTHER_REPORTER, sequence_number=2))

        assert sorted(provider._cell_of_id) == sorted([_cell_id(0), _cell_id(1), _cell_id(0, pool_id=_OTHER_POOL_ID)])

    async def test_each_reporter_is_sequenced_on_its_own(self):
        """Deployments count their own snapshots, so one that has run for longer must not silence a fresh one."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)], sequence_number=7))

        await _apply(provider, _snapshot([_other_cell(0)], reporter_id=_OTHER_REPORTER, sequence_number=1))

        assert sorted(provider._cell_of_id) == sorted([_cell_id(0), _cell_id(0, pool_id=_OTHER_POOL_ID)])


class TestResendingTheSameMembership:
    async def test_the_same_snapshot_sent_again_leaves_the_membership_as_it_was(self):
        """Every tick carries the whole membership, so the steady state has to be idempotent."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))

        await _apply(provider, _snapshot([_cell(0)], sequence_number=2))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]

    async def test_resending_it_announces_no_change_to_the_watcher(self):
        """A membership that did not move must not churn the cells the run reconciles."""
        provider, watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))
        await _drain()
        watcher.observations.clear()

        await _apply(provider, _snapshot([_cell(0)], sequence_number=2))
        await _drain()

        assert watcher.observations == []


class TestAddressingRegisteredCells:
    async def test_the_addresses_of_a_registered_worker_are_the_ones_reported(self):
        """The run calls the engine at the address its own deployment observed, never at one derived here."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)]))

        addrs = await provider.get_addrs(_worker_name_of_cell(_cell_id(0)))

        assert addrs["primary"] == HostAndPort(host="10.0.0.5", port=8000)

    async def test_a_registered_worker_keeps_the_generation_its_own_deployment_gave_it(self):
        """The dashboard identifies an engine by its generation, so a restarted pod has to look like a new one."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0, generation=3)]))

        ((worker_info,),) = provider.get_worker_infos(cell_ids=[_cell_id(0)])

        assert worker_info.generation == 3

    async def test_a_watcher_that_starts_late_is_replayed_the_cells_already_reported(self):
        """A snapshot may land before the run watches, and nothing announces that cell a second time."""
        provider = RegistrationHub(run_uuid=_RUN_UUID)
        await _apply(provider, _snapshot([_cell(0)]))

        watcher = _Watcher()
        await _start_watch(provider, watcher)

        assert [cell_id for cell_id, _observed in watcher.observations] == [_cell_id(0)]


class TestFailedReconciliation:
    async def test_a_cell_the_run_could_not_take_in_is_offered_again_on_the_next_poll(self):
        """A cell whose reconcile raised must be retried from the membership this run already holds."""
        provider, watcher = await _watched()
        watcher.failing_cell_ids = {_cell_id(0)}

        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))

        assert watcher.observations == []
        assert sorted(provider._cell_of_id) == [_cell_id(0)]

        watcher.failing_cell_ids = set()
        await _drain()

        assert [cell_id for cell_id, _observed in watcher.observations] == [_cell_id(0)]


class TestDroppingAReporterThatStoppedReporting:
    async def test_a_reporter_that_stopped_reporting_loses_every_cell_it_announced(self):
        """Its deployment is gone or unreachable, and a request routed to those engines answers nothing."""
        clock = _FakeClock()
        provider, watcher = await _watched(clock=clock)
        await _apply(provider, _snapshot([_cell(0)]))

        clock.now += REPORTER_TTL_SECONDS + 1.0
        await _drain()

        assert sorted(provider._cell_of_id) == []
        assert watcher.observations[-1] == (_cell_id(0), None)

    async def test_a_reporter_still_inside_its_deadline_keeps_its_cells(self):
        """A reporter reports far more often than this, so dropping one early would flap a healthy deployment."""
        clock = _FakeClock()
        provider, _watcher = await _watched(clock=clock)
        await _apply(provider, _snapshot([_cell(0)]))

        clock.now += REPORTER_TTL_SECONDS - 1.0
        await _drain()

        assert sorted(provider._cell_of_id) == [_cell_id(0)]

    async def test_a_reporter_that_comes_back_is_taken_in_again(self):
        """Being dropped is not a verdict on the deployment; it announces itself anew like any first time."""
        clock = _FakeClock()
        provider, _watcher = await _watched(clock=clock)
        await _apply(provider, _snapshot([_cell(0)]))

        clock.now += REPORTER_TTL_SECONDS + 1.0
        await _drain()
        await _apply(provider, _snapshot([_cell(0)], sequence_number=9))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]

    async def test_a_snapshot_dropped_as_late_does_not_keep_its_reporter_alive(self):
        """A reporter that restarted resends from sequence 1, and only the deadline ends that stalemate."""
        clock = _FakeClock()
        provider = RegistrationHub(run_uuid=_RUN_UUID, clock=clock)
        await _apply(provider, _snapshot([_cell(0)], sequence_number=5))

        clock.now += REPORTER_TTL_SECONDS - 1.0
        await provider.ingest(_snapshot([_cell(0)], sequence_number=1))
        clock.now += 2.0
        provider._remove_stale_reporters()

        assert sorted(provider._state_of_reporter_id) == []
        assert sorted(provider._cell_of_id) == []

    async def test_dropping_a_reporter_forgets_the_reporter_itself(self):
        """A state left behind keeps the reporter's sequence number, which would refuse its fresh snapshots."""
        clock = _FakeClock()
        provider = RegistrationHub(run_uuid=_RUN_UUID, clock=clock)
        await _apply(provider, _snapshot([_cell(0)]))
        assert sorted(provider._state_of_reporter_id) == [_REPORTER]

        clock.now += REPORTER_TTL_SECONDS + 1.0
        provider._remove_stale_reporters()

        assert sorted(provider._state_of_reporter_id) == []

    async def test_the_sweep_stops_with_the_watcher_it_was_started_for(self):
        """A hub nobody watches serves nobody, and a task outliving it would keep the whole hub alive."""
        clock = _FakeClock()
        provider = RegistrationHub(run_uuid=_RUN_UUID, clock=clock)
        stop_watch = await _start_watch(provider, _Watcher())

        await stop_watch()
        clock.now += REPORTER_TTL_SECONDS + 1.0
        await _apply(provider, _snapshot([_cell(0)]))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]
