import pytest

from tests.fast.utils.workers.rpc.server.lifecycle_fixtures import _run_process_probe, _shutdown_probe


class TestShutdownDeadline:
    @pytest.mark.parametrize("mode", ["sync", "async"])
    def test_unresponsive_work_cannot_prevent_process_shutdown(self, mode: str) -> None:
        """Shutdown terminates the process when running work cannot be cancelled."""
        assert _run_process_probe(target=_shutdown_probe, mode=mode) == 1

    def test_completed_shutdown_cancels_the_process_exit_deadline(self) -> None:
        """A successfully closed server must not terminate its process later."""
        assert _run_process_probe(target=_shutdown_probe, mode="normal") == 0
