from concurrent.futures import Future
from enum import IntEnum
from threading import Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles.utils import gds_io


class _Error(Exception):
    def __init__(self, status):
        self.status = status


class _Status(IntEnum):
    INVALID_VALUE = 5022
    PERMISSION_DENIED = 5025


def _runtime(monkeypatch, *, enabled=(), errors=None):
    names = ("PROPERTIES_ALLOW_COMPAT_MODE", "FORCE_COMPAT_MODE", "PROPERTIES_POSIX_IO_MODE", "GDS_FALLBACK_IO")
    calls = []

    def get(name):
        assert calls and calls[0] == "open", "Configuration must be read after driver initialization"
        calls.append(name)
        if name in (errors or {}):
            raise _Error(errors[name])
        return name in enabled

    monkeypatch.setattr(gds_io, "_DRIVER_OPENED", False)
    return (
        SimpleNamespace(
            BoolConfigParameter=SimpleNamespace(**{name: name for name in names}),
            OpError=_Status,
            cuFileError=_Error,
            driver_open=lambda: calls.append("open"),
            get_parameter_bool=get,
        ),
        calls,
    )


def test_config_initializes_once_and_accepts_only_unknown_optional_flags(monkeypatch):
    runtime, calls = _runtime(
        monkeypatch,
        errors={"PROPERTIES_POSIX_IO_MODE": _Status.INVALID_VALUE, "GDS_FALLBACK_IO": _Status.INVALID_VALUE},
    )
    gds_io._require_direct_configuration(runtime)
    gds_io._require_direct_configuration(runtime)
    assert calls.count("open") == 1


@pytest.mark.parametrize("name", ["PROPERTIES_ALLOW_COMPAT_MODE", "FORCE_COMPAT_MODE", "GDS_FALLBACK_IO"])
def test_config_rejects_enabled_fallback(monkeypatch, name):
    runtime, _ = _runtime(monkeypatch, enabled=(name,))
    with pytest.raises(RuntimeError, match=name):
        gds_io._require_direct_configuration(runtime)


@pytest.mark.parametrize(
    ("name", "status"),
    [("PROPERTIES_ALLOW_COMPAT_MODE", _Status.INVALID_VALUE), ("GDS_FALLBACK_IO", _Status.PERMISSION_DENIED)],
)
def test_config_never_ignores_required_or_other_errors(monkeypatch, name, status):
    runtime, _ = _runtime(monkeypatch, errors={name: status})
    with pytest.raises(_Error):
        gds_io._require_direct_configuration(runtime)


def test_failed_driver_initialization_does_not_mark_open(monkeypatch):
    runtime, _ = _runtime(monkeypatch)
    runtime.driver_open = MagicMock(side_effect=OSError("driver unavailable"))
    with pytest.raises(OSError, match="driver unavailable"):
        gds_io._require_direct_configuration(runtime)
    assert not gds_io._DRIVER_OPENED


def test_close_drains_every_operation_before_deregister_even_after_failure(monkeypatch):
    calls = []
    failed = Future()
    failed.set_exception(OSError("short write"))
    completed = MagicMock()
    completed.result.side_effect = lambda: calls.append("remaining operation")
    backend = gds_io.GdsBackend.__new__(gds_io.GdsBackend)
    backend._lock = Lock()
    backend._closed = False
    backend._futures = [failed, completed]
    backend._owns_executor = False
    backend._handle, backend._fd = 123, 456
    backend._cufile = SimpleNamespace(handle_deregister=lambda _: calls.append("deregister"))
    monkeypatch.setattr(gds_io.os, "close", lambda _: calls.append("close fd"))
    with pytest.raises(OSError, match="short write"):
        backend.close()
    assert calls == ["remaining operation", "deregister", "close fd"]
    backend.close()
    assert backend._futures == []
