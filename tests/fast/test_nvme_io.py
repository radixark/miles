from concurrent.futures import Future
from threading import Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from miles.utils import nvme_io


def test_backend_opens_direct_io_without_buffered_fallback(monkeypatch):
    direct_flag = getattr(nvme_io.os, "O_DIRECT", 0x4000)
    monkeypatch.setattr(nvme_io.os, "O_DIRECT", direct_flag, raising=False)
    monkeypatch.setattr(nvme_io.os, "preadv", MagicMock(), raising=False)
    monkeypatch.setattr(nvme_io.os, "pwritev", MagicMock(), raising=False)
    opener = MagicMock(side_effect=OSError("direct I/O unavailable"))
    monkeypatch.setattr(nvme_io.os, "open", opener)
    pool = SimpleNamespace(device=torch.device("cuda:0"))
    with pytest.raises(OSError, match="direct I/O unavailable"):
        nvme_io.NvmeBackend("baseline.bin", "cuda:0", staging=pool)
    opener.assert_called_once()
    assert opener.call_args.args[1] & direct_flag


@pytest.mark.parametrize(("actual", "expected"), [(4096, 4096), (17, 17)])
def test_read_accepts_exact_requested_or_known_eof_count(monkeypatch, actual, expected):
    reader = MagicMock(return_value=actual)
    monkeypatch.setattr(nvme_io.os, "preadv", reader, raising=False)
    buffer = memoryview(bytearray(4096))
    nvme_io._read_exact(123, buffer, 8192, expected)
    reader.assert_called_once_with(123, [buffer], 8192)


@pytest.mark.parametrize(("actual", "expected"), [(0, 4096), (17, 4096), (16, 17)])
def test_unexpected_short_reads_fail_before_gpu_copy(monkeypatch, actual, expected):
    monkeypatch.setattr(nvme_io.os, "preadv", lambda *args: actual, raising=False)
    with pytest.raises(OSError, match=f"expected {expected} bytes, received {actual}"):
        nvme_io._read_exact(123, memoryview(bytearray(4096)), 0, expected)


def test_short_write_is_never_accepted(monkeypatch):
    monkeypatch.setattr(nvme_io.os, "pwritev", lambda *args: 2048, raising=False)
    with pytest.raises(OSError, match="expected 4096 bytes, received 2048"):
        nvme_io._write_exact(123, memoryview(bytearray(4096)), 0)


def test_close_drains_all_transfers_even_after_failure(monkeypatch):
    calls = []
    failed = Future()
    failed.set_exception(OSError("short write"))
    completed = MagicMock()
    completed.result.side_effect = lambda: calls.append("remaining transfer")
    backend = nvme_io.NvmeBackend.__new__(nvme_io.NvmeBackend)
    backend._lock = Lock()
    backend._closed = False
    backend._futures = [failed, completed]
    backend._owns_executor = False
    backend._fd = 456
    monkeypatch.setattr(nvme_io.os, "close", lambda _: calls.append("close fd"))
    with pytest.raises(OSError, match="short write"):
        backend.close()
    assert calls == ["remaining transfer", "close fd"]
    backend.close()
    assert backend._futures == []
