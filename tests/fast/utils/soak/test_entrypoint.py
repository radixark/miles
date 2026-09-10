import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest
from tests.fast.utils.soak.utils import (
    SERVING,
    AsyncStubFaultForm,
    api_server_fault_forms,
    fixed_fault_forms,
    intervals,
    staged,
    typed_cell,
)
from tests.utils.soak import entrypoint, views
from tests.utils.soak.observer import SoakObserver
from tests.utils.soak.runner import SoakRunner
from tests.utils.soak.state import SoakActionRequest, SoakObservation


def test_stop_and_join_takes_one_last_snapshot_before_the_log_is_read() -> None:
    """Regression: a recovery completing after the final poll must not be lost to a race."""
    handle = entrypoint.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type=intervals(("rollout",), 1e9),
        cell_fault_forms=api_server_fault_forms(),
    )

    with patch.object(
        SoakObserver,
        "observe",
        AsyncMock(return_value=SoakObservation(cells=[staged("rollout-engine-0", SERVING)])),
    ):
        handle.start()
        handle.stop_and_join()

    assert views.compute_states_of_cell_name(handle.event_log.events) == {"rollout-engine-0": [SERVING]}


def test_handle_forwards_the_poll_interval() -> None:
    """The configured polling cadence reaches the injection loop."""
    captured: dict[str, object] = {}
    finished = threading.Event()

    async def capture_loop(self: SoakRunner, stop_event: asyncio.Event) -> None:
        captured["poll_interval_seconds"] = self._poll_interval_seconds
        finished.set()

    handle = entrypoint.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type=intervals(("rollout",), 1e9),
        cell_fault_forms=api_server_fault_forms(),
        poll_interval_seconds=0.25,
    )

    with patch.object(SoakRunner, "run", capture_loop):
        handle.start()
        assert finished.wait(timeout=30)
        handle._worker.join(timeout_seconds=30)

    assert captured["poll_interval_seconds"] == 0.25


def test_an_injector_that_outlives_the_join_fails_instead_of_racing_the_log() -> None:
    """Reading the log beside a still-running injector would assert on a half-written history."""
    released = threading.Event()
    entered = threading.Event()

    async def slow_inject(request: SoakActionRequest) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            async with asyncio.timeout(30):
                while not released.is_set():
                    await asyncio.sleep(0.01)

    handle = entrypoint.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type=intervals(("actor",), 1e-12),
        cell_fault_forms=fixed_fault_forms([AsyncStubFaultForm(name="slow", execute=slow_inject)]),
        poll_interval_seconds=0,
    )

    with (
        patch.object(
            SoakObserver,
            "observe",
            AsyncMock(return_value=SoakObservation(cells=[typed_cell(f"actor-{i}", "actor") for i in range(3)])),
        ),
        patch.object(entrypoint, "STOP_AND_JOIN_TIMEOUT_SECONDS", 0.2),
    ):
        handle.start()
        try:
            assert entered.wait(timeout=30)
            with pytest.raises(AssertionError, match="still mid-injection"):
                handle.stop_and_join()
        finally:
            released.set()
            handle._worker.join(timeout_seconds=30)
