import asyncio
import random
from builtins import ExceptionGroup
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from tests.fast.utils.soak.utils import AsyncStubFaultForm, typed_cell
from tests.utils.soak.action import SoakActionError
from tests.utils.soak.config import SoakTimeouts
from tests.utils.soak.core import SoakActionScheduler
from tests.utils.soak.observer import SoakObserver
from tests.utils.soak.runner import SoakRunner
from tests.utils.soak.state import (
    EventLog,
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakObservation,
)
from tests.utils.soak.views import compute_num_successful_injections_of_form


@pytest.mark.parametrize("stop_during_read", [False, True])
async def test_transient_reads_and_action_failures_preserve_progress_and_stop_order(
    monkeypatch: pytest.MonkeyPatch, stop_during_read: bool
) -> None:
    """Retry unavailable observations and refused actions, but never dispatch after a stop observed during reading."""
    stop = asyncio.Event()
    log = EventLog()
    reads = 0
    attempts = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal reads
        reads += 1
        if reads == 1 and not stop_during_read:
            return httpx.Response(503)
        if stop_during_read:
            stop.set()
        return httpx.Response(200, json={"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]})

    async def execute(request: SoakActionRequest) -> dict:
        stored = [
            event.request
            for event in log.events
            if isinstance(event, SoakActionRequestedEvent) and event.request.request_id == request.request_id
        ]
        assert stored == [request]
        assert not any(
            isinstance(event, SoakActionResultEvent) and event.request_id == request.request_id for event in log.events
        )
        attempts.append(request.request_id)
        if len(attempts) == 1:
            raise httpx.ConnectError("fault request refused")
        stop.set()
        return {"request_id": request.request_id}

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs)
    )
    forms = {"actor": [AsyncStubFaultForm(name="fault", execute=execute)]}
    runner = SoakRunner(
        observer=SoakObserver(base_url="http://control", cell_types={"actor"}),
        scheduler=SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1e-12}, forms=forms),
        forms=forms,
        event_log=log,
        poll_interval_seconds=0,
    )
    async with asyncio.timeout(5):
        await runner.run(stop)
    results = [event for event in log.events if isinstance(event, SoakActionResultEvent)]
    applied = [event for event in log.events if isinstance(event, SoakActionAppliedEvent)]
    if stop_during_read:
        assert not attempts and not results and not applied
    else:
        assert len(attempts) == 2
        assert [event.request_id for event in results] == attempts
        assert [event.returned for event in results] == [False, True]
        assert [event.request_id for event in applied] == [attempts[1]]
        assert any(isinstance(event, SoakObservation) and event.cells is None for event in log.events)
    assert isinstance(log.events[-1], SoakObservation)


async def test_failed_action_retains_partial_evidence_without_becoming_applied() -> None:
    """Partial victim receipts remain auditable without turning a failed batch into successful coverage."""
    evidence = {"victim_outcomes": {"a": {"receipt": {"exited_pids": [42]}}, "b": {"error": "stale"}}}

    async def execute(request: SoakActionRequest) -> None:
        raise SoakActionError("Partial failure", evidence=evidence)

    form = AsyncStubFaultForm(name="batch", execute=execute)
    forms = {"actor": [form]}
    log = EventLog()
    runner = SoakRunner(
        observer=SoakObserver(base_url="http://control", cell_types={"actor"}),
        scheduler=SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1}, forms=forms),
        forms=forms,
        event_log=log,
    )
    request = SoakActionRequest(target=typed_cell("actor-0", "actor"), form_name="batch", harms_cell=True)
    log.note_action_requested(request)
    with pytest.raises(SoakActionError):
        await runner._execute(request)
    results = [event for event in log.events if isinstance(event, SoakActionResultEvent)]
    assert len(results) == 1 and not results[0].returned
    assert results[0].evidence == evidence
    assert not any(isinstance(event, SoakActionAppliedEvent) for event in log.events)


async def test_total_budget_cancels_a_stuck_observation_and_records_the_final_snapshot() -> None:
    """The session deadline remains effective while the observer is blocked."""
    cleaned = asyncio.Event()

    async def observe() -> SoakObservation:
        if cleaned.is_set():
            return SoakObservation(cells=[])
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return SoakObservation(cells=None)

    log = EventLog()
    runner = SoakRunner(
        observer=SoakObserver(base_url="http://control", cell_types=set()),
        scheduler=SoakActionScheduler(rng=random.Random(0), mean_intervals={}, forms={}),
        forms={},
        event_log=log,
        poll_interval_seconds=0,
        timeouts=SoakTimeouts(run_seconds=0.01, observation_seconds=60, final_observation_seconds=1),
    )
    with patch.object(SoakObserver, "observe", side_effect=observe):
        with pytest.raises(TimeoutError):
            await runner.run(asyncio.Event())
    assert cleaned.is_set()
    assert isinstance(log.events[-1], SoakObservation) and log.events[-1].cells == []


async def test_observation_timeout_records_failure_after_cancelling_the_read() -> None:
    """A timed-out observer releases its work and replaces stale facts with an explicit failed reading."""
    cleaned = asyncio.Event()

    async def observe() -> SoakObservation:
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    log = EventLog()
    log.note_observation(SoakObservation(cells=[typed_cell("actor-0", "actor")]))
    runner = SoakRunner(
        observer=SoakObserver(base_url="http://control", cell_types=set()),
        scheduler=SoakActionScheduler(rng=random.Random(0), mean_intervals={}, forms={}),
        forms={},
        event_log=log,
    )
    with patch.object(SoakObserver, "observe", side_effect=observe):
        await runner._observe_and_record(timeout_seconds=0.01)
    assert cleaned.is_set()
    assert isinstance(log.events[-1], SoakObservation)
    assert log.events[-1].cells is None and "observation" in log.events[-1].errors


@pytest.mark.parametrize("evidence", [None, {"exited_pids": [42]}])
async def test_runner_records_applied_before_result_only_when_form_returns_evidence(evidence: dict | None) -> None:
    """A completed command earns coverage only when its form supplies actual effect evidence."""

    async def execute(request: SoakActionRequest) -> dict | None:
        return evidence

    log = EventLog()
    forms = {"actor": [AsyncStubFaultForm(name="fault", execute=execute)]}
    runner = SoakRunner(
        observer=SoakObserver(base_url="http://control", cell_types={"actor"}),
        scheduler=SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1}, forms=forms),
        forms=forms,
        event_log=log,
    )
    request = SoakActionRequest(target=typed_cell("actor-0", "actor"), form_name="fault", harms_cell=True)
    log.note_action_requested(request)
    await runner._execute(request)
    assert isinstance(log.events[-1], SoakActionResultEvent) and log.events[-1].returned
    assert compute_num_successful_injections_of_form(log.events, form_name="fault") == int(evidence is not None)
    if evidence is not None:
        assert isinstance(log.events[-2], SoakActionAppliedEvent)
        assert log.events[-2].evidence == evidence


def test_pending_action_does_not_block_observation_and_stop_reaps_it_before_final_snapshot() -> None:
    """Observe during an action, keep admission serial, and collect cancellation before the final reading."""

    async def scenario() -> None:
        stop = asyncio.Event()
        entered = asyncio.Event()
        cleaned = asyncio.Event()
        log = EventLog()
        observed: list[SoakObservation] = []

        async def execute(request: SoakActionRequest) -> None:
            assert any(
                isinstance(event, SoakActionRequestedEvent) and event.request == request for event in log.events
            )
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        async def observe() -> SoakObservation:
            if observed:
                assert entered.is_set()
                if len(observed) == 1:
                    assert not cleaned.is_set()
                    stop.set()
                else:
                    assert cleaned.is_set()
                    assert isinstance(log.events[-1], SoakActionResultEvent)
            snapshot = SoakObservation(cells=[typed_cell(f"actor-{i}", "actor") for i in range(3)])
            observed.append(snapshot)
            return snapshot

        forms = {"actor": [AsyncStubFaultForm(name="slow", execute=execute)]}
        runner = SoakRunner(
            observer=SoakObserver(base_url="http://control", cell_types={"actor"}),
            scheduler=SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1e-12}, forms=forms),
            forms=forms,
            event_log=log,
            poll_interval_seconds=0,
        )
        with patch.object(SoakObserver, "observe", side_effect=observe):
            async with asyncio.timeout(5):
                await runner.run(stop)

        assert len(observed) == 3
        requests = [event for event in log.events if isinstance(event, SoakActionRequestedEvent)]
        results = [event for event in log.events if isinstance(event, SoakActionResultEvent)]
        assert len(requests) == len(results) == 1
        assert results[0].request_id == requests[0].request.request_id
        assert not results[0].returned
        assert results[0].error == "CancelledError()"
        assert isinstance(log.events[-1], SoakObservation)

    asyncio.run(scenario())


def test_unexpected_action_failure_stops_runner_without_waiting_for_stop_request() -> None:
    """Programming failures propagate out of the runner after recording the result and final observation."""

    async def scenario() -> None:
        async def execute(request: SoakActionRequest) -> None:
            raise RuntimeError("broken action")

        log = EventLog()
        forms = {"actor": [AsyncStubFaultForm(name="broken", execute=execute)]}
        runner = SoakRunner(
            observer=SoakObserver(base_url="http://control", cell_types={"actor"}),
            scheduler=SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1e-12}, forms=forms),
            forms=forms,
            event_log=log,
            poll_interval_seconds=0,
        )
        stop = asyncio.Event()
        with patch.object(
            SoakObserver,
            "observe",
            AsyncMock(return_value=SoakObservation(cells=[typed_cell(f"actor-{i}", "actor") for i in range(3)])),
        ):
            async with asyncio.timeout(5):
                with pytest.raises(ExceptionGroup):
                    await runner.run(stop)

        assert not stop.is_set()
        results = [event for event in log.events if isinstance(event, SoakActionResultEvent)]
        assert results
        assert all(not event.returned and event.error == "RuntimeError('broken action')" for event in results)
        assert isinstance(log.events[-1], SoakObservation)

    asyncio.run(scenario())
