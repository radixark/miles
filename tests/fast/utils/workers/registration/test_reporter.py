from __future__ import annotations

import asyncio

import pytest

from miles.utils.workers.registration.models import RegistrationSnapshot
from miles.utils.workers.registration.reporter import RegistrationReporter, RegistrationReporterWorker
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

_POOL_ID = "inference-engine-0-0"
_REPORTER_ID = "miles-run-r1-inference"


class _FakeEngineProvider(BaseWorkerProvider):
    def __init__(self, *, cell_indices: list[int], worker_type: str = "regular", generation: int = 0) -> None:
        self.cell_indices = list(cell_indices)
        self.worker_type = worker_type
        self.generation = generation
        self.reconcile: CellReconcileFn | None = None
        self.stopped = False

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise NotImplementedError

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [
            [
                WorkerInfo(
                    name=f"{cell_id}-0",
                    generation=self.generation,
                    self_addrs={"primary": HostAndPort(host="10.0.0.5", port=8000)},
                    gpu_ids=[0],
                    worker_class=None,
                )
            ]
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        self.reconcile = reconcile
        for cell_index in self.cell_indices:
            await reconcile(f"{_POOL_ID}-{cell_index}", _cell_info(cell_index, worker_type=self.worker_type))

        async def _stop() -> None:
            self.stopped = True

        return _stop


class _FakeHub:
    def __init__(self) -> None:
        self.snapshots: list[RegistrationSnapshot] = []
        self.ready_timeouts: list[float] = []

    async def wait_ready(self, *, timeout: float) -> None:
        self.ready_timeouts.append(timeout)

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        self.snapshots.append(snapshot)


def _cell_info(cell_index: int, *, workers_hash: str = "hash-1", worker_type: str = "regular") -> CellInfo:
    return CellInfo(
        cell_id=f"{_POOL_ID}-{cell_index}",
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[f"{_POOL_ID}-{cell_index}-0"],
        workers_hash=workers_hash,
        meta=dict(model_id="default", worker_type=worker_type),
    )


def _reporter(*, provider: _FakeEngineProvider, hub_endpoint: _FakeHub):
    return RegistrationReporter(
        reporter_id=_REPORTER_ID,
        hub_endpoint=hub_endpoint,
        worker_provider=provider,
    )


async def _synced(
    *,
    cell_indices: tuple[int, ...] = (0,),
    worker_type: str = "regular",
    hub_endpoint: _FakeHub | None = None,
    generation: int = 0,
) -> tuple[RegistrationReporter, _FakeEngineProvider, _FakeHub]:
    provider = _FakeEngineProvider(cell_indices=list(cell_indices), worker_type=worker_type, generation=generation)
    hub_endpoint = hub_endpoint if hub_endpoint is not None else _FakeHub()
    reporter = _reporter(provider=provider, hub_endpoint=hub_endpoint)
    await provider.watch_cells(reporter._observe)
    return reporter, provider, hub_endpoint


class TestSnapshotContents:
    async def test_a_snapshot_carries_every_cell_of_this_deployment(self):
        """The run replaces this deployment's membership with the snapshot, so it has to be the whole one."""
        reporter, _provider, hub_endpoint = await _synced(cell_indices=(0, 1))

        await reporter._send_once()

        (snapshot,) = hub_endpoint.snapshots
        assert [cell.info.cell_id for cell in snapshot.cells] == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]

    async def test_it_reports_the_names_its_own_deployment_gave_its_cells(self):
        """Those pool ids are born naming this deployment, so a name rewritten here would name nothing."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.info.pool_id == _POOL_ID
        assert [worker.name for worker in cell.workers] == [f"{_POOL_ID}-0-0"]

    async def test_it_reports_the_addresses_its_own_deployment_observed(self):
        """The run calls the engine there, so an address computed anywhere else would be a guess."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.workers[0].self_addrs["primary"] == HostAndPort(host="10.0.0.5", port=8000)

    async def test_it_reports_the_generation_its_own_deployment_observed(self):
        """A restarted engine is a new engine to the run, and only its own deployment counts the restarts."""
        reporter, _provider, hub_endpoint = await _synced(generation=3)

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.workers[0].generation == 3

    async def test_it_reports_a_prefill_or_decode_deployment_like_any_other(self):
        """A deployment of split engines announces them the same way, and the run addresses them the same way."""
        reporter, _provider, hub_endpoint = await _synced(worker_type="prefill")

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.info.meta["worker_type"] == "prefill"


class TestSnapshotSequencing:
    async def test_an_unchanged_membership_is_sent_whole_again(self):
        """Every tick declares the whole membership, so the run never depends on a message it may have missed."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()
        await reporter._send_once()

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[0].cells] == [f"{_POOL_ID}-0"]
        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[1].cells] == [f"{_POOL_ID}-0"]

    async def test_a_changed_membership_is_sent_whole(self):
        """A cell that appeared or went away has to reach the run in the very next snapshot."""
        reporter, provider, hub_endpoint = await _synced()
        await reporter._send_once()

        await provider.reconcile(f"{_POOL_ID}-1", _cell_info(1))
        await reporter._send_once()

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[1].cells] == [
            f"{_POOL_ID}-0",
            f"{_POOL_ID}-1",
        ]


class TestReporterWorker:
    def test_the_worker_the_engine_release_serves_answers_no_call(self):
        """Its whole value is the reporting loop its constructor starts, so it exposes nothing to call."""
        assert collect_rpc_method_specs(RegistrationReporterWorker) == {}


class TestReporterLifecycle:
    async def test_it_waits_for_the_hub_before_it_watches_its_own_cells(self):
        """The hub_endpoint comes up with the run, and a snapshot into nothing is a wasted period."""
        provider = _FakeEngineProvider(cell_indices=[0])
        hub_endpoint = _FakeHub()
        run = asyncio.create_task(_reporter(provider=provider, hub_endpoint=hub_endpoint).run())
        for _ in range(100):
            await asyncio.sleep(0)
            if provider.reconcile is not None:
                break

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert hub_endpoint.ready_timeouts
        assert provider.stopped
