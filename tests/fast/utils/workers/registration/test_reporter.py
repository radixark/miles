from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import random
import threading
from argparse import Namespace
from unittest.mock import patch

import pytest

from miles.utils.workers.naming import _worker_name_of_cell, compute_cell_id
from miles.utils.workers.registration.models import RegistrationSnapshot
from miles.utils.workers.registration.reporter import (
    SNAPSHOT_INTERVAL_SECONDS,
    RegistrationReporter,
    RegistrationReporterWorker,
    _DebouncedIntervalTrigger,
    _exit_on_reporter_stop,
)
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

_POOL_ID = "inference-engine-0-0"
_REPORTER_ID = "miles-run-r1-inference"
_RUN_UUID = "run-uuid-1"


def _cell_id(cell_index: int) -> str:
    return compute_cell_id(pool_id=_POOL_ID, cell_index=cell_index)


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
                    name=_worker_name_of_cell(cell_id),
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
            await reconcile(_cell_id(cell_index), _cell_info(cell_index, worker_type=self.worker_type))

        async def _stop() -> None:
            self.stopped = True

        return _stop


class _FakeHub:
    def __init__(self) -> None:
        self.snapshots: list[RegistrationSnapshot] = []
        self.ready_timeouts: list[float] = []
        self.pin_drops: list[bool] = []

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        self.ready_timeouts.append(timeout)
        self.pin_drops.append(allow_server_uuid_change)

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        self.snapshots.append(snapshot)


class _HungHub(_FakeHub):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = asyncio.Event()

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        try:
            await asyncio.Future()
        finally:
            self.cancelled.set()


class _FailOnceHub(_FakeHub):
    def __init__(self) -> None:
        super().__init__()
        self.accepted = asyncio.Event()

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        self.snapshots.append(snapshot)
        if len(self.snapshots) == 1:
            raise RuntimeError("first ingest failed")
        self.accepted.set()


def _cell_info(cell_index: int, *, workers_hash: str = "hash-1", worker_type: str = "regular") -> CellInfo:
    return CellInfo(
        cell_id=_cell_id(cell_index),
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[_worker_name_of_cell(_cell_id(cell_index))],
        workers_hash=workers_hash,
        meta=dict(model_id="default", worker_type=worker_type),
    )


class _FakeTrigger:
    """Fires as soon as anything asks, so a reporter test never waits on a real interval."""

    def __init__(self) -> None:
        self.interval_seconds = SNAPSHOT_INTERVAL_SECONDS
        self.notified = 0

    def notify(self) -> None:
        self.notified += 1

    async def wait(self) -> None:
        # answering without a timer turn starves the loop this runs in: the reporter sends in a tight
        # cycle and nothing else gets a turn, including the cancellation that ends the test.
        await asyncio.sleep(0.001)


def _reporter(*, provider: _FakeEngineProvider, hub_endpoint: _FakeHub):
    return RegistrationReporter(
        run_uuid=_RUN_UUID,
        reporter_id=_REPORTER_ID,
        hub_endpoint=hub_endpoint,
        worker_provider=provider,
        _trigger=_FakeTrigger(),
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
        assert [cell.info.cell_id for cell in snapshot.cells] == [_cell_id(0), _cell_id(1)]

    async def test_it_reports_the_names_its_own_deployment_gave_its_cells(self):
        """Those pool ids are born naming this deployment, so a name rewritten here would name nothing."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.info.pool_id == _POOL_ID
        assert [worker.name for worker in cell.workers] == [_worker_name_of_cell(_cell_id(0))]

    async def test_it_reports_the_addresses_its_own_deployment_observed(self):
        """The run calls the engine there, so an address computed anywhere else would be a guess."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.workers[0].self_addrs["primary"] == HostAndPort(host="10.0.0.5", port=8000)

    async def test_it_names_the_run_it_was_deployed_for(self):
        """The deployment script gives every component the same --run-uuid, and the run refuses any other."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()

        assert hub_endpoint.snapshots[0].run_uuid == _RUN_UUID

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
    async def test_every_snapshot_carries_a_higher_sequence_than_the_last(self):
        """The run drops a snapshot that arrived late, and only the sequence number tells it which one that is."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()
        await reporter._send_once()

        assert [snapshot.sequence_number for snapshot in hub_endpoint.snapshots] == [1, 2]

    async def test_an_unchanged_membership_is_sent_whole_again(self):
        """Every tick declares the whole membership, so the run never depends on a message it may have missed."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()
        await reporter._send_once()

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[0].cells] == [_cell_id(0)]
        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[1].cells] == [_cell_id(0)]

    async def test_a_changed_membership_is_sent_whole(self):
        """A cell that appeared or went away has to reach the run in the very next snapshot."""
        reporter, provider, hub_endpoint = await _synced()
        await reporter._send_once()

        await provider.reconcile(_cell_id(1), _cell_info(1))
        await reporter._send_once()

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[1].cells] == [
            _cell_id(0),
            _cell_id(1),
        ]


class TestObservingItsOwnCells:
    async def test_a_cell_that_came_or_went_wakes_the_reporter(self):
        """Waiting out the whole period would leave the run without engines its deployment already has."""
        reporter, provider, _hub_endpoint = await _synced()
        assert reporter._trigger.notified == 1

        await provider.reconcile(_cell_id(1), _cell_info(1))
        await provider.reconcile(_cell_id(1), None)

        assert reporter._trigger.notified == 3
        assert sorted(reporter._info_of_cell_id) == [_cell_id(0)]


class TestReporterWorker:
    def test_construction_starts_reporting_and_a_stopped_reporter_exits_the_process(self) -> None:
        """Construction starts reporting, and an ended reporting loop makes the deployment exit."""
        reporting_started = threading.Event()
        exit_called = threading.Event()

        class FakeReporter:
            async def run(self) -> None:
                reporting_started.set()

        args = Namespace(env_report_interval_seconds=0)
        with patch("os._exit", side_effect=lambda _code: exit_called.set()) as exit_process:
            RegistrationReporterWorker(args=args, reporter=FakeReporter())

            assert reporting_started.wait(timeout=5.0)
            assert exit_called.wait(timeout=5.0)
            exit_process.assert_called_once_with(1)

    def test_the_worker_the_engine_release_serves_answers_no_call(self):
        """Its whole value is the reporting loop its constructor starts, so it exposes nothing to call."""
        assert collect_rpc_method_specs(RegistrationReporterWorker) == {}


class _RecordingProvider(_FakeEngineProvider):
    def __init__(self, events: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.events = events

    async def init(self) -> None:
        self.events.append("init")

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        self.events.append("watch")
        return await super().watch_cells(reconcile)


class _InitRequiringProvider(_FakeEngineProvider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.initialized = False

    async def init(self) -> None:
        self.initialized = True

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        assert self.initialized, "the engines this provider serves only exist once init() discovered them"
        return await super().watch_cells(reconcile)


class _RecordingHub(_FakeHub):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def wait_ready(self, *, timeout: float) -> None:
        self.events.append("wait_ready")
        await super().wait_ready(timeout=timeout)


async def _run_until_watching(reporter: RegistrationReporter, provider: _FakeEngineProvider) -> None:
    run = asyncio.create_task(reporter.run())
    for _ in range(100):
        await asyncio.sleep(0)
        if provider.reconcile is not None:
            break

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run


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

    async def test_it_initializes_the_worker_provider_before_anything_else(self):
        """InferenceController.init() does the same, and a provider is not usable before it."""
        events: list[str] = []
        provider = _RecordingProvider(events, cell_indices=[0])

        await _run_until_watching(_reporter(provider=provider, hub_endpoint=_RecordingHub(events)), provider)

        assert events[:3] == ["init", "wait_ready", "watch"]

    async def test_a_provider_whose_cells_init_discovers_is_watched_without_tripping_its_own_check(self):
        """The static external-engine provider asserted this and took the whole reporter down with it."""
        provider = _InitRequiringProvider(cell_indices=[0])

        await _run_until_watching(_reporter(provider=provider, hub_endpoint=_FakeHub()), provider)

        assert provider.reconcile is not None

    async def test_the_cells_init_discovered_reach_the_hub(self):
        """A split run on --rollout-external-engine-addrs registered no engine at all, and then timed out."""
        provider = _InitRequiringProvider(cell_indices=[0])
        hub_endpoint = _FakeHub()
        reporter = _reporter(provider=provider, hub_endpoint=hub_endpoint)

        await _run_until_watching(reporter, provider)
        await reporter._send_once()

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[-1].cells] == [_cell_id(0)]

    async def test_a_hung_ingest_is_cancelled_at_the_timeout(self):
        """A hub call that never answers is cancelled when its ingest deadline expires."""
        hub_endpoint = _HungHub()
        reporter, _provider, _hub_endpoint = await _synced(hub_endpoint=hub_endpoint)

        with patch("miles.utils.workers.registration.reporter.INGEST_TIMEOUT_SECONDS", 0.001):
            with pytest.raises(TimeoutError):
                await reporter._send_once()

        assert hub_endpoint.cancelled.is_set()

    async def test_a_failed_ingest_is_logged_and_the_next_snapshot_is_sent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A transient ingest failure is logged while reporting continues with a newer snapshot sequence."""
        provider = _FakeEngineProvider(cell_indices=[0])
        hub_endpoint = _FailOnceHub()
        reporter = _reporter(provider=provider, hub_endpoint=hub_endpoint)

        with caplog.at_level(logging.WARNING):
            run = asyncio.create_task(reporter.run())
            await asyncio.wait_for(hub_endpoint.accepted.wait(), timeout=5.0)

        assert not run.done()
        assert [snapshot.sequence_number for snapshot in hub_endpoint.snapshots[:2]] == [1, 2]
        assert f"Reporting the cells of {_REPORTER_ID} failed" in caplog.text

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert provider.stopped


class TestReporterFailure:
    def test_a_stopped_reporter_forces_the_deployment_to_exit(self):
        """Any stopped reporter makes its deployment exit unsuccessfully instead of appearing healthy."""
        completed: concurrent.futures.Future[None] = concurrent.futures.Future()
        completed.set_result(None)
        failed: concurrent.futures.Future[None] = concurrent.futures.Future()
        failed.set_exception(RuntimeError("reporting stopped"))

        with patch("miles.utils.workers.registration.reporter.os._exit") as exit_process:
            _exit_on_reporter_stop(completed)
            _exit_on_reporter_stop(failed)

        assert [call.args for call in exit_process.call_args_list] == [(1,), (1,)]


def _trigger(*, interval_seconds: float = 0.05, jitter_ratio: float = 0.0, debounce_seconds: float = 0.0):
    return _DebouncedIntervalTrigger(
        interval_seconds=interval_seconds,
        jitter_ratio=jitter_ratio,
        debounce_seconds=debounce_seconds,
        rng=random.Random(0),
    )


class TestDebouncedIntervalTrigger:
    async def test_it_fires_on_its_own_when_nothing_notified_it(self):
        """A membership nobody changed still has to be re-announced, or the run forgets this reporter."""
        await asyncio.wait_for(_trigger().wait(), timeout=5.0)

    async def test_a_notification_fires_it_before_the_interval_is_up(self):
        """A cell that came or went has to reach the run now, not up to a whole period later."""
        trigger = _trigger(interval_seconds=1000.0)
        trigger.notify()

        await asyncio.wait_for(trigger.wait(), timeout=5.0)

    async def test_it_settles_before_firing_so_a_burst_of_changes_sends_once(self):
        """Cells arrive in bursts, and one snapshot per cell would be a snapshot storm."""
        trigger = _trigger(interval_seconds=1000.0, debounce_seconds=0.05)
        trigger.notify()

        await asyncio.wait_for(trigger.wait(), timeout=5.0)

        assert not trigger._changed.is_set()

    async def test_a_change_during_the_settling_window_is_not_carried_into_the_next_wait(self):
        """It is already part of the snapshot this wait is about to send, so firing again would send a duplicate."""
        trigger = _trigger(interval_seconds=1000.0, debounce_seconds=0.05)
        trigger.notify()

        waiting = asyncio.ensure_future(trigger.wait())
        await asyncio.sleep(0)
        trigger.notify()
        await asyncio.wait_for(waiting, timeout=5.0)

        assert not trigger._changed.is_set()

    async def test_the_period_it_waits_is_jittered_around_the_interval(self):
        """Every reporter of a run wakes on its own schedule, so they never pile onto the hub_endpoint together."""
        trigger = _trigger(interval_seconds=10.0, jitter_ratio=0.2)

        periods = {trigger._compute_next_interval_seconds() for _ in range(20)}

        assert len(periods) > 1
        assert all(8.0 <= period <= 12.0 for period in periods)

    async def test_no_jitter_asks_for_exactly_the_interval(self):
        """The jitter is a spread around the configured period, not a replacement for it."""
        assert _trigger(interval_seconds=10.0, jitter_ratio=0.0)._compute_next_interval_seconds() == 10.0


class _FailingHub(_FakeHub):
    def __init__(self, failures: int, error: BaseException) -> None:
        super().__init__()
        self.failures = failures
        self.error = error

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        await super().registration_ingest(snapshot=snapshot)


class _RestartedHub(_FakeHub):
    """Answers like a hub whose process was replaced: it refuses every call made on the old pin."""

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        if True not in self.pin_drops:
            raise ServerRestartedError("the hub answering now is a different process")
        await super().registration_ingest(snapshot=snapshot)


async def _report_until(reporter: RegistrationReporter, condition) -> None:
    run = asyncio.create_task(reporter.run())
    for _ in range(1000):
        await asyncio.sleep(0.001)
        if condition():
            break

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert condition()


class TestAHubThatRestartedUnderTheReporter:
    async def test_it_awaits_the_new_hub_with_the_pin_dropped(self):
        """Dropping the pin is the only way the handle stops talking to a process that is gone."""
        hub_endpoint = _RestartedHub()
        provider = _FakeEngineProvider(cell_indices=[0])

        await _report_until(_reporter(provider=provider, hub_endpoint=hub_endpoint), lambda: hub_endpoint.snapshots)

        assert hub_endpoint.pin_drops[:2] == [False, True]

    async def test_it_goes_on_reporting_into_the_process_that_answers_now(self):
        """The run waits for engines that are running until this deployment reaches the new hub."""
        hub_endpoint = _RestartedHub()
        provider = _FakeEngineProvider(cell_indices=[0])

        await _report_until(_reporter(provider=provider, hub_endpoint=hub_endpoint), lambda: hub_endpoint.snapshots)

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[0].cells] == [_cell_id(0)]

    async def test_an_ordinary_failure_leaves_the_pin_where_it_was(self):
        """Re-awaiting readiness on every hiccup would accept whatever process answered next."""
        hub_endpoint = _FailingHub(failures=1, error=RuntimeError("the hub was busy"))
        provider = _FakeEngineProvider(cell_indices=[0])

        await _report_until(_reporter(provider=provider, hub_endpoint=hub_endpoint), lambda: hub_endpoint.snapshots)

        assert hub_endpoint.pin_drops == [False]

    async def test_a_hub_that_never_restarted_is_awaited_once(self):
        """Every reporter of every run takes this path, and it may not wait for readiness twice."""
        hub_endpoint = _FakeHub()
        provider = _FakeEngineProvider(cell_indices=[0])

        await _report_until(_reporter(provider=provider, hub_endpoint=hub_endpoint), lambda: hub_endpoint.snapshots)

        assert hub_endpoint.pin_drops == [False]
