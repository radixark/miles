from unittest.mock import patch

from tests.e2e.ft.conftest_ft.fault_injection import core, entrypoint, views
from tests.fast.e2e.ft.fault_injection.utils import SERVING, mock_response, staged


def test_stop_and_join_takes_one_last_snapshot_before_the_log_is_read() -> None:
    """Regression: a recovery completing after the final poll must not be lost to a race."""
    handle = entrypoint.FaultInjectorHandle(
        base_url="http://control", seed=0, mean_interval_seconds=1e9, cell_type="rollout"
    )

    with patch.object(core, "requests") as mock_requests:
        mock_requests.get.side_effect = lambda url, timeout: mock_response(
            {"items": [staged("rollout-engine-0", SERVING)]}
        )
        handle.start()
        handle.stop_and_join(timeout_seconds=5)

    assert views.compute_states_of_cell_name(handle.event_log.events) == {"rollout-engine-0": [SERVING]}
