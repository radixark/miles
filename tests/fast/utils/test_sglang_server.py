from io import StringIO
from unittest.mock import MagicMock

import pytest

from tests.e2e.sglang.utils import sglang_server


@pytest.mark.parametrize(
    ("error", "returncode", "should_terminate"),
    [
        (TimeoutError("not ready"), None, True),
        (RuntimeError("exited early"), 1, False),
    ],
)
def test_start_sglang_server_cleans_up_after_readiness_failure(monkeypatch, error, returncode, should_terminate):
    log_file = StringIO()
    process = MagicMock()
    process.poll.return_value = returncode
    monkeypatch.setattr(sglang_server.Path, "open", lambda *_args, **_kwargs: log_file)
    monkeypatch.setattr(sglang_server.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(sglang_server, "_wait_for_ready", MagicMock(side_effect=error))

    with pytest.raises(type(error)) as exc_info:
        sglang_server.start_sglang_server(model_path="model", port=34001)

    assert exc_info.value is error
    process.poll.assert_called_once_with()
    if should_terminate:
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=sglang_server.DEFAULT_SHUTDOWN_TIMEOUT_SECS)
    else:
        process.terminate.assert_not_called()
        process.wait.assert_not_called()
    process.kill.assert_not_called()
    assert log_file.closed


def test_start_sglang_server_keeps_running_server_open(monkeypatch):
    log_file = StringIO()
    process = MagicMock()
    process.poll.return_value = None
    monkeypatch.setattr(sglang_server.Path, "open", lambda *_args, **_kwargs: log_file)
    monkeypatch.setattr(sglang_server.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(sglang_server, "_wait_for_ready", MagicMock())

    server = sglang_server.start_sglang_server(model_path="model", port=34001)

    process.terminate.assert_not_called()
    assert not log_file.closed
    server.stop()
    process.terminate.assert_called_once_with()
    assert log_file.closed
