"""The Adam store page-cache window, checked against kernel residency through mincore."""

import mmap
import os
import threading

import pytest
from tests.fast.optimizers.nvme_setup_fixtures import require_evictable, resident_pages

from miles_plugins.optimizers import nvme_stream


_FILE_BYTES = 256 * mmap.PAGESIZE


@pytest.fixture
def written_file(tmp_path):
    """Opens a file freshly written through the page cache, on a filesystem that can evict it."""
    require_evictable(tmp_path)
    fds = []

    def write(name: str) -> int:
        fd = os.open(str(tmp_path / name), os.O_RDWR | os.O_CREAT, 0o600)
        fds.append(fd)
        os.pwrite(fd, b"\x01" * _FILE_BYTES, 0)
        return fd

    yield write
    for fd in fds:
        os.close(fd)


def test_page_cache_window_drops_the_oldest_file_before_a_release_past_its_cap_returns(written_file):
    window = nvme_stream._PageCacheWindow(cap_bytes=_FILE_BYTES * 3 // 2)
    older, newer = written_file("older.bin"), written_file("newer.bin")
    window.release(older, _FILE_BYTES)
    window.release(newer, _FILE_BYTES)

    assert resident_pages(older) == 0
    window.drain()
    assert resident_pages(newer) == 0
    assert os.pread(older, 1, 0) == b"\x01"


def test_a_release_within_the_cap_returns_while_the_writeback_is_still_running(monkeypatch):
    """The window's point: a flush under the cap must not wait on any writeback, its own or an older one."""
    writeback_may_finish = threading.Event()
    dropped = []

    def slow_drop(fd: int) -> None:
        # Bounded, so a release that waits on its own writeback fails the assertion below instead of hanging.
        writeback_may_finish.wait(timeout=10)
        dropped.append(fd)

    monkeypatch.setattr(nvme_stream, "_drop_from_page_cache", slow_drop)
    window = nvme_stream._PageCacheWindow(cap_bytes=2)
    window.release(10, 1)
    window.release(11, 1)

    assert dropped == []
    writeback_may_finish.set()
    window.drain()
    assert dropped == [10, 11]


@pytest.mark.parametrize("by_path", [False, True], ids=["store-fd", "checkpoint-path"])
def test_zero_cap_drops_each_file_before_its_release_returns(written_file, tmp_path, by_path):
    window = nvme_stream._PageCacheWindow(cap_bytes=0)
    fd = written_file("bucket.bin")
    window.release(str(tmp_path / "bucket.bin") if by_path else fd, _FILE_BYTES)

    assert resident_pages(fd) == 0


def test_page_cache_window_raises_a_failed_writeback_on_the_caller():
    """The writer thread's errors must not vanish with it."""
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("page-cache control is Linux-only")
    window = nvme_stream._PageCacheWindow(cap_bytes=1 << 30)

    # Not an open descriptor, so the writeback fails with EBADF: on this release if the writer is already done, else on
    # the drain.
    with pytest.raises(OSError):
        window.release(1 << 30, 1)
        window.drain()
