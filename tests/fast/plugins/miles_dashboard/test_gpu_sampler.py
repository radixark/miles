import logging
import sys
import time

import pytest

from miles.dashboard import gpu_sampler as gpu_sampler_module
from miles.dashboard.gpu_sampler import GpuSampler
from miles.dashboard.store import GpuProcessSample, GpuSample


class FakeNvml:
    """Just enough of pynvml for the sampler: handle == device index. UUIDs
    and process names come back as bytes, as the real pynvml returns them."""

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
        return f"GPU-fake-{handle}".encode()

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
        return f"proc-{pid}".encode()


class FakeAmdSmi:
    """Just enough of the amdsmi API (ROCm >= 6.1 dict shapes); handle == device index."""

    def __init__(self, count=2, *, fail_init=False, failing_devices=()):
        self.count = count
        self.fail_init = fail_init
        self.failing_devices = set(failing_devices)

    def amdsmi_init(self):
        if self.fail_init:
            raise RuntimeError("AMD SMI initialization failed")

    def amdsmi_get_processor_handles(self):
        return list(range(self.count))

    def amdsmi_get_gpu_device_uuid(self, handle):
        return f"GPU-amd-{handle}"

    def amdsmi_get_gpu_activity(self, handle):
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return {"gfx_activity": 70 + handle, "umc_activity": 10, "mm_activity": 0}

    def amdsmi_get_gpu_vram_usage(self, handle):
        # AMD SMI reports vram_used in MiB, not bytes.
        return {"vram_total": 192 * 1024, "vram_used": (handle + 1) * 2048}

    def amdsmi_get_power_info(self, handle):
        # Keep values distinct to catch selection of the wrong power field.
        return {
            "socket_power": 500 + handle,
            "current_socket_power": 475 + handle,
            "average_socket_power": 250 + handle,
        }

    def amdsmi_get_gpu_process_list(self, handle):
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return [
            {
                "pid": 2000 + handle,
                "name": f"amd-proc-{handle}" if handle == 0 else "N/A",
                "memory_usage": {"vram_mem": (handle + 1) * 768 * 1024 * 1024},
            },
            {"pid": 9000 + handle, "name": "kfd-bystander", "memory_usage": {"vram_mem": 0}},
        ]


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
    sampler = GpuSampler(push=push, node="10.0.0.1", nvml=FakeNvml(count=2))
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


def test_amd_sample_once_preserves_native_units_and_uuids():
    push = PushSpy()
    sampler = GpuSampler(push=push, node="amd-node", amdsmi=FakeAmdSmi(count=2))
    assert sampler.available
    assert sampler.gpu_uuids() == ["GPU-amd-0", "GPU-amd-1"]

    assert sampler.sample_once(ts=11.0) == 2
    sampler.flush()
    [(node, batch)] = push.calls
    assert node == "amd-node"
    assert batch == [
        GpuSample(ts=11.0, node="amd-node", gpu=0, util=70, mem_mb=2048, power_w=500),
        GpuSample(ts=11.0, node="amd-node", gpu=1, util=71, mem_mb=4096, power_w=501),
    ]


@pytest.mark.parametrize(
    "power,expected",
    [
        # MI300+/ROCm 7.x: the unified field carries current power and wins.
        ({"socket_power": 500, "current_socket_power": 475, "average_socket_power": 250}, 500),
        # ROCm 7.x with the unified sensor unavailable ("N/A" per field).
        ({"socket_power": "N/A", "current_socket_power": 475, "average_socket_power": 250}, 475),
        # ROCm 6.0 numeric uint16 sentinel; no unified field on 6.x.
        ({"current_socket_power": 0xFFFF, "average_socket_power": 250}, 250),
        # Pre-MI300 6.x wrappers only expose the average.
        ({"average_socket_power": 250}, 250),
    ],
)
def test_amd_socket_power_prefers_current_then_falls_back(power, expected):
    assert gpu_sampler_module._amd_socket_power(power) == expected


def test_amd_socket_power_raises_when_all_fields_unavailable():
    with pytest.raises(ValueError, match="socket power unavailable"):
        gpu_sampler_module._amd_socket_power({"socket_power": "N/A", "current_socket_power": 0xFFFF})


def test_amd_unavailable_power_reports_zero_but_keeps_util_and_mem():
    push = PushSpy()
    amdsmi = FakeAmdSmi(count=1)
    amdsmi.amdsmi_get_power_info = lambda handle: {"socket_power": "N/A", "current_socket_power": "N/A"}
    sampler = GpuSampler(push=push, node="n", amdsmi=amdsmi)

    assert sampler.sample_once(ts=1.0) == 1
    sampler.flush()
    [sample] = push.calls[0][1]
    assert (sample.util, sample.mem_mb, sample.power_w) == (70, 2048, 0)


def test_flush_clears_buffer_and_skips_empty():
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(count=1))
    sampler.flush()  # empty: no call
    assert push.calls == []

    sampler.sample_once(ts=1.0)
    sampler.flush()
    sampler.flush()  # cleared: no duplicate push
    assert len(push.calls) == 1


def test_nvml_init_failure_disables_sampler(caplog):
    push = PushSpy()
    with caplog.at_level(logging.WARNING):
        sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(fail_init=True))
    assert not sampler.available
    assert sampler.start() is False
    assert sampler.sample_once(ts=1.0) == 0
    assert push.calls == []
    assert any("NVML unavailable" in r.message for r in caplog.records)


@pytest.mark.parametrize("backend", ["nvml", "amdsmi"])
def test_zero_devices_disable_sampler(backend):
    fake = FakeNvml(count=0) if backend == "nvml" else FakeAmdSmi(count=0)
    sampler = GpuSampler(push=PushSpy(), node="n", **{backend: fake})
    assert not sampler.available
    assert sampler.gpu_uuids() == []


def test_production_auto_detection_falls_back_from_nvml_to_amdsmi(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml(fail_init=True))
    monkeypatch.setitem(sys.modules, "amdsmi", FakeAmdSmi(count=1))

    sampler = GpuSampler(push=PushSpy(), node="n")

    assert sampler.available
    assert sampler.gpu_uuids() == ["GPU-amd-0"]


def test_explicit_injection_does_not_probe_the_other_backend(monkeypatch):
    class BoomAmdSmi:
        def amdsmi_init(self):
            pytest.fail("explicit NVML injection must not probe AMD SMI")

    monkeypatch.setitem(sys.modules, "amdsmi", BoomAmdSmi())

    sampler = GpuSampler(push=PushSpy(), node="n", nvml=FakeNvml(fail_init=True))

    assert not sampler.available


def test_missing_optional_backends_disable_sampler(monkeypatch, caplog):
    # None in sys.modules makes `import pynvml` raise ImportError, as if absent.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setitem(sys.modules, "amdsmi", None)
    with caplog.at_level(logging.WARNING):
        sampler = GpuSampler(push=PushSpy(), node="n")

    assert not sampler.available
    assert sampler.start() is False
    assert any("GPU telemetry unavailable" in record.message for record in caplog.records)


def test_only_one_backend_may_be_injected():
    with pytest.raises(AssertionError, match="inject only one"):
        GpuSampler(push=PushSpy(), node="n", nvml=FakeNvml(), amdsmi=FakeAmdSmi())


def test_failing_device_is_skipped_others_report(caplog):
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(count=3, failing_devices={1}))
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push.calls
    assert [s.gpu for s in batch] == [0, 2]
    assert any("skipping this tick" in r.message for r in caplog.records)


def test_amd_failing_device_is_skipped_while_others_report(caplog):
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", amdsmi=FakeAmdSmi(count=3, failing_devices={1}))
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push.calls
    assert [sample.gpu for sample in batch] == [0, 2]
    assert any("AMD SMI read failed for gpu 1" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "method,payload",
    [
        ("amdsmi_get_gpu_activity", {"gfx_activity": "N/A", "umc_activity": 10, "mm_activity": 0}),
        ("amdsmi_get_gpu_vram_usage", {"vram_total": 192 * 1024, "vram_used": "N/A"}),
    ],
)
def test_amd_degraded_metric_value_skips_device_while_others_report(method, payload, caplog):
    push = PushSpy()
    amdsmi = FakeAmdSmi(count=2)
    original = getattr(amdsmi, method)
    setattr(amdsmi, method, lambda handle: payload if handle == 0 else original(handle))
    sampler = GpuSampler(push=push, node="n", amdsmi=amdsmi)

    with caplog.at_level(logging.WARNING):
        assert sampler.sample_once(ts=1.0) == 1
    sampler.flush()
    assert [sample.gpu for sample in push.calls[0][1]] == [1]
    assert any("AMD SMI read failed for gpu 0" in record.message for record in caplog.records)


def test_amd_uses_smi_visible_order_without_refiltering_process_env(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2")
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", amdsmi=FakeAmdSmi(count=3))

    assert sampler.sample_once(ts=1.0) == 3
    sampler.flush()
    assert [sample.gpu for sample in push.calls[0][1]] == [0, 1, 2]


def test_thread_lifecycle_flushes_on_stop():
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", interval=0.01, nvml=FakeNvml(count=1))
    assert sampler.start() is True
    time.sleep(0.08)
    sampler.stop()
    assert push.calls, "stop() must flush buffered samples"
    total = sum(len(batch) for _, batch in push.calls)
    assert total >= 3  # ~8 ticks at 10ms; generous margin against scheduler jitter


def test_sample_processes_once_converts_units():
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(count=2), push_processes=push_processes)
    assert sampler.sample_processes_once(ts=5.0) == 2
    sampler.flush()
    [(node, batch)] = push_processes.calls
    assert node == "n"
    assert batch == [
        GpuProcessSample(ts=5.0, node="n", gpu=0, pid=1000, name="proc-1000", mem_mb=512),
        GpuProcessSample(ts=5.0, node="n", gpu=1, pid=1001, name="proc-1001", mem_mb=1024),
    ]


def test_amd_processes_convert_bytes_fall_back_on_name_and_drop_zero_vram():
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push=PushSpy(), node="n", amdsmi=FakeAmdSmi(count=2), push_processes=push_processes)

    assert sampler.sample_processes_once(ts=5.0) == 2
    sampler.flush()
    [(node, batch)] = push_processes.calls
    assert node == "n"
    # the zero-VRAM kfd-bystander pids are filtered out
    assert batch == [
        GpuProcessSample(ts=5.0, node="n", gpu=0, pid=2000, name="amd-proc-0", mem_mb=768),
        GpuProcessSample(ts=5.0, node="n", gpu=1, pid=2001, name="pid 2001", mem_mb=1536),
    ]


def test_failing_device_skipped_for_process_sampling(caplog):
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(
        push=push, node="n", nvml=FakeNvml(count=3, failing_devices={1}), push_processes=push_processes
    )
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_processes_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push_processes.calls
    assert [s.gpu for s in batch] == [0, 2]
    assert any("skipping this tick" in r.message for r in caplog.records)


def test_amd_failing_device_is_skipped_for_process_sampling(caplog):
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(
        push=PushSpy(),
        node="n",
        amdsmi=FakeAmdSmi(count=3, failing_devices={1}),
        push_processes=push_processes,
    )
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_processes_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push_processes.calls
    assert [sample.gpu for sample in batch] == [0, 2]
    assert any("AMD SMI process query failed for gpu 1" in record.message for record in caplog.records)


def test_process_batch_dropped_silently_without_push_processes():
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(count=1))
    assert sampler.sample_processes_once(ts=1.0) == 1
    sampler.flush()
    assert push.calls == []


def test_interval_must_be_positive():
    with pytest.raises(AssertionError):
        GpuSampler(push=lambda n, b: None, node="n", interval=0, nvml=FakeNvml())


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
    sampler = GpuSampler(push=push, node="local", nvml=pynvml, push_processes=push_processes)
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


def test_real_amdsmi_when_gpus_present():
    amdsmi = pytest.importorskip("amdsmi")
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push=push, node="local", amdsmi=amdsmi, push_processes=push_processes)
    if not sampler.available:
        pytest.skip("no usable AMD SMI device")

    assert sampler.sample_once(ts=1.0) >= 1
    assert sampler.sample_processes_once(ts=1.0) >= 0
    sampler.flush()
    [(_, batch)] = push.calls
    sample = batch[0]
    assert 0 <= sample.util <= 100
    assert sample.mem_mb >= 0 and sample.power_w >= 0
    assert sampler.gpu_uuids()[0]
    if push_processes.calls:
        proc_sample = push_processes.calls[0][1][0]
        assert proc_sample.pid > 0 and proc_sample.mem_mb > 0 and proc_sample.name


class TestConstructorContract:
    def test_a_positional_push_is_rejected(self):
        """The sampler is built as a ray actor, so push must be bound by keyword only."""
        with pytest.raises(TypeError):
            GpuSampler(PushSpy(), node="n", nvml=FakeNvml())
