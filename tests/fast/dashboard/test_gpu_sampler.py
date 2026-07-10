import logging
import time

import pytest

from miles.dashboard.gpu_sampler import GpuSampler
from miles.dashboard.store import GpuSample


class FakeNvml:
    """Just enough of pynvml for the sampler: handle == device index."""

    def __init__(self, count=2, fail_init=False, failing_devices=()):
        self.count = count
        self.fail_init = fail_init
        self.failing_devices = set(failing_devices)

    def nvmlInit(self):
        if self.fail_init:
            raise RuntimeError("driver/library version mismatch")

    def nvmlDeviceGetCount(self):
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return f"GPU-fake-{handle}"

    def nvmlDeviceGetUtilizationRates(self, handle):
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return type("Util", (), {"gpu": 40 + handle})()

    def nvmlDeviceGetMemoryInfo(self, handle):
        return type("Mem", (), {"used": (handle + 1) * 1024 * 1024 * 1024})()  # GiB in bytes

    def nvmlDeviceGetPowerUsage(self, handle):
        return 600_000 + handle  # milliwatts


class PushSpy:
    def __init__(self):
        self.calls: list[tuple[str, list[GpuSample]]] = []

    def __call__(self, node, batch):
        self.calls.append((node, batch))


def test_sample_once_converts_units():
    push = PushSpy()
    sampler = GpuSampler(push, node="10.0.0.1", nvml=FakeNvml(count=2))
    assert sampler.available
    assert sampler.gpu_uuids() == ["GPU-fake-0", "GPU-fake-1"]

    assert sampler.sample_once(ts=10.0) == 2
    sampler.flush()
    [(node, batch)] = push.calls
    assert node == "10.0.0.1"
    assert batch == [
        GpuSample(ts=10.0, node="10.0.0.1", gpu=0, util=40, mem_mb=1024, power_w=600),
        GpuSample(ts=10.0, node="10.0.0.1", gpu=1, util=41, mem_mb=2048, power_w=600),
    ]


def test_flush_clears_buffer_and_skips_empty():
    push = PushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=1))
    sampler.flush()  # empty: no call
    assert push.calls == []

    sampler.sample_once(ts=1.0)
    sampler.flush()
    sampler.flush()  # cleared: no duplicate push
    assert len(push.calls) == 1


def test_nvml_init_failure_disables_sampler(caplog):
    push = PushSpy()
    with caplog.at_level(logging.WARNING):
        sampler = GpuSampler(push, node="n", nvml=FakeNvml(fail_init=True))
    assert not sampler.available
    assert sampler.start() is False
    assert sampler.sample_once(ts=1.0) == 0
    assert push.calls == []
    assert any("NVML unavailable" in r.message for r in caplog.records)


def test_failing_device_is_skipped_others_report(caplog):
    push = PushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=3, failing_devices={1}))
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push.calls
    assert [s.gpu for s in batch] == [0, 2]
    assert any("skipping this tick" in r.message for r in caplog.records)


def test_thread_lifecycle_flushes_on_stop():
    push = PushSpy()
    sampler = GpuSampler(push, node="n", interval=0.01, nvml=FakeNvml(count=1))
    assert sampler.start() is True
    time.sleep(0.08)
    sampler.stop()
    assert push.calls, "stop() must flush buffered samples"
    total = sum(len(batch) for _, batch in push.calls)
    assert total >= 3  # ~8 ticks at 10ms; generous margin against scheduler jitter


def test_interval_must_be_positive():
    with pytest.raises(AssertionError):
        GpuSampler(lambda n, b: None, node="n", interval=0, nvml=FakeNvml())


def test_real_nvml_when_gpus_present():
    # Guards the FakeNvml against drifting from the real pynvml API surface;
    # runs wherever a GPU + driver exist (devbox/CI-GPU), skips elsewhere.
    pynvml = pytest.importorskip("pynvml")
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception:
        pytest.skip("no usable NVML device")

    push = PushSpy()
    sampler = GpuSampler(push, node="local", nvml=pynvml)
    assert sampler.available
    assert sampler.sample_once(ts=1.0) >= 1
    sampler.flush()
    [(_, batch)] = push.calls
    sample = batch[0]
    assert 0 <= sample.util <= 100
    assert sample.mem_mb >= 0 and sample.power_w >= 0
    assert sampler.gpu_uuids()[0].startswith("GPU-")
