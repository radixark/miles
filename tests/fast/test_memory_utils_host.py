"""The memory report says what this process holds on the host, and a train
step hands its heap and pinned cache back."""

import sys

import pytest
import torch

from miles.utils import memory_utils

STATUS = "Name:\tray::Actor\nVmPeak:\t5330000000 kB\nVmRSS:\t248000000 kB\nVmHWM:\t249200000 kB\nRssAnon:\t67500000 kB\nRssFile:\t1900000 kB\nRssShmem:\t178600000 kB\nThreads:\t80\n"


def test_host_memory_reads_the_status_fields():
    out = memory_utils.host_memory(STATUS)
    assert (out["rss_GB"], out["rss_peak_GB"], out["anon_GB"], out["shmem_GB"], out["file_GB"]) == (
        236.51,
        237.66,
        64.37,
        170.33,
        1.81,
    )


def test_host_memory_survives_a_missing_status_file(monkeypatch):
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no proc")))
    out = memory_utils.host_memory()
    assert "rss_GB" not in out


def test_the_report_carries_the_host_part(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device for the GPU part")
    report = memory_utils.available_memory()
    assert report["host"]["rss_GB"] > 0 and "anon_GB" in report["host"]


def test_the_report_carries_peaks_since_the_last_reset():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    big = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    del big
    assert memory_utils.available_memory()["max_allocated_GB"] >= 0.25
    torch.accelerator.reset_peak_memory_stats(torch.accelerator.current_device_index())
    report = memory_utils.available_memory()
    assert report["max_allocated_GB"] == report["allocated_GB"]


def test_trim_host_heap_is_a_no_op_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert memory_utils.trim_host_heap() is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc only")
def test_trim_host_heap_calls_libc():
    assert memory_utils.trim_host_heap() is True


def test_release_host_memory_collects_empties_and_trims(monkeypatch):
    calls = []
    monkeypatch.setattr(memory_utils.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: calls.append("pinned"), raising=False)
    monkeypatch.setattr(memory_utils, "trim_host_heap", lambda: calls.append("trim") or True)
    memory_utils.release_host_memory()
    assert calls == ["gc", "pinned", "trim"]
