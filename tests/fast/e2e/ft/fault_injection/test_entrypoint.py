import random
import threading
from unittest.mock import patch

import pytest

from tests.e2e.ft.conftest_ft.fault_injection import core, entrypoint, views
from tests.fast.e2e.ft.fault_injection.utils import (
    intervals,
    api_server_fault_forms,
    fixed_fault_forms,
    StubFaultForm,
    typed_cell,
    SERVING,
    mock_response,
    staged,
)


def test_stop_and_join_takes_one_last_snapshot_before_the_log_is_read() -> None:
    """Regression: a recovery completing after the final poll must not be lost to a race."""
    handle = entrypoint.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type=intervals(("rollout",), 1e9),
        cell_fault_forms=api_server_fault_forms(),
    )

    with patch.object(core, "requests") as mock_requests:
        mock_requests.get.side_effect = lambda url, timeout: mock_response(
            {"items": [staged("rollout-engine-0", SERVING)]}
        )
        handle.start()
        handle.stop_and_join()

    assert views.compute_states_of_cell_name(handle.event_log.events) == {"rollout-engine-0": [SERVING]}


def test_handle_forwards_the_quiescent_poll_override() -> None:
    """A short scenario may rely on the controller's atomic injection gate."""
    captured: dict[str, object] = {}
    finished = threading.Event()

    def capture_loop(**kwargs: object) -> None:
        captured.update(kwargs)
        finished.set()

    handle = entrypoint.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type=intervals(("rollout",), 1e9),
        cell_fault_forms=api_server_fault_forms(),
        poll_interval_seconds=0.25,
        quiescent_polls_required=1,
    )

    with patch.object(entrypoint, "run_fault_injection_loop", capture_loop):
        handle.start()
        assert finished.wait(timeout=30)
        handle._worker.join(timeout_seconds=30)

    assert captured["quiescent_polls_required"] == 1
    assert captured["poll_interval_seconds"] == 0.25


def test_an_injector_that_outlives_the_join_fails_instead_of_racing_the_log() -> None:
    """Reading the log beside a still-running injector would assert on a half-written history."""
    released = threading.Event()
    entered = threading.Event()

    def slow_inject(cell: dict, rng: random.Random) -> None:
        entered.set()
        released.wait(timeout=30)

    handle = entrypoint.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type=intervals(("actor",), 1e-12),
        cell_fault_forms=fixed_fault_forms([StubFaultForm("slow", slow_inject)]),
        poll_interval_seconds=0,
        quiescent_polls_required=1,
    )

    with (
        patch.object(core, "requests") as mock_requests,
        patch.object(entrypoint, "STOP_AND_JOIN_TIMEOUT_SECONDS", 0.2),
    ):
        mock_requests.get.side_effect = lambda url, timeout: mock_response(
            {"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]}
        )
        handle.start()
        try:
            assert entered.wait(timeout=30)
            with pytest.raises(AssertionError, match="still mid-injection"):
                handle.stop_and_join()
        finally:
            released.set()
            handle._worker.join(timeout_seconds=30)
