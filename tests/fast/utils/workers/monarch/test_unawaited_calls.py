import logging

import pytest

pytest.importorskip("monarch")

from miles.utils.workers.monarch.runtime import _log_if_failed
from miles.utils.workers.worker_handle import WorkerUnreachableError


async def _raising(error: BaseException) -> None:
    raise error


class TestACallNobodyAwaits:
    async def test_its_failure_is_logged(self, caplog: pytest.LogCaptureFixture):
        """A command whose run fails would otherwise leave its cell alive with nothing running and no trace."""
        with caplog.at_level(logging.ERROR):
            await _log_if_failed(_raising(RuntimeError("run failed")), describe="run")

        assert "'run', which nobody awaits, failed on its worker" in caplog.text
        assert "run failed" in caplog.text

    async def test_a_worker_that_is_gone_is_left_to_the_liveness_scan(self, caplog: pytest.LogCaptureFixture):
        """Every teardown ends a running command this way; the liveness scan already reports the lost cell."""
        with caplog.at_level(logging.ERROR):
            await _log_if_failed(_raising(WorkerUnreachableError("stopped")), describe="run")

        assert caplog.text == ""
