import logging
import time

import pytest

from miles.dashboard.gpu_sampler import GpuSampler
from miles.dashboard.store import GpuProcessSample, GpuSample


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

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return [type("Proc", (), {"pid": 1000 + handle, "usedGpuMemory": (handle + 1) * 512 * 1024 * 1024})()]

    def nvmlSystemGetProcessName(self, pid):
        return f"proc-{pid}"


class PushSpy:
    def __init__(self):
        self.calls: list[tuple[str, list[GpuSample]]] = []

    def __call__(self, node, batch):
        self.calls.append((node, batch))


class ProcessPushSpy:
    def __init__(self):
        self.calls: list[tuple[str, list[GpuProcessSample]]] = []

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


def test_sample_processes_once_converts_units():
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=2), push_processes=push_processes)
    assert sampler.sample_processes_once(ts=5.0) == 2
    sampler.flush()
    [(node, batch)] = push_processes.calls
    assert node == "n"
    assert batch == [
        GpuProcessSample(ts=5.0, node="n", gpu=0, pid=1000, name="proc-1000", mem_mb=512),
        GpuProcessSample(ts=5.0, node="n", gpu=1, pid=1001, name="proc-1001", mem_mb=1024),
    ]


def test_failing_device_skipped_for_process_sampling(caplog):
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=3, failing_devices={1}), push_processes=push_processes)
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_processes_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push_processes.calls
    assert [s.gpu for s in batch] == [0, 2]
    assert any("skipping this tick" in r.message for r in caplog.records)


def test_process_batch_dropped_silently_without_push_processes():
    # sampling still works if called directly, but flush() must not crash or
    # invent a push destination — the buffer is just discarded (design: the
    # feature is a no-op end-to-end when the collector never wires it up)
    push = PushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=1))
    assert sampler.sample_processes_once(ts=1.0) == 1
    sampler.flush()
    assert push.calls == []


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
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push, node="local", nvml=pynvml, push_processes=push_processes)
    assert sampler.available
    assert sampler.sample_once(ts=1.0) >= 1
    # idle test GPUs may have zero compute processes — asserting >= 0 just
    # guards that the real nvmlDeviceGetComputeRunningProcesses call doesn't raise
    assert sampler.sample_processes_once(ts=1.0) >= 0
    sampler.flush()
    [(_, batch)] = push.calls
    sample = batch[0]
    assert 0 <= sample.util <= 100
    assert sample.mem_mb >= 0 and sample.power_w >= 0
    assert sampler.gpu_uuids()[0].startswith("GPU-")
    if push_processes.calls:
        proc_sample = push_processes.calls[0][1][0]
        assert proc_sample.pid > 0 and proc_sample.mem_mb >= 0 and proc_sample.name
