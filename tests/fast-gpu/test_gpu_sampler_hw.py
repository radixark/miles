from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["short"])
register_rocm_ci(est_time=60, suite="stage-c-4-gpu-mi350", labels=["amd"])

import torch

from miles.dashboard.gpu_sampler import GpuSampler


class PushSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, node, batch):
        self.calls.append((node, batch))


def test_auto_detection_picks_the_backend_matching_the_hardware():
    sampler = GpuSampler(push=PushSpy(), node="ci")
    assert sampler.available, "no GPU telemetry backend initialized on a GPU runner"
    expected = "AMD SMI" if torch.version.hip else "NVML"
    assert sampler._provider.name == expected


def test_every_device_reports_telemetry_and_processes():
    push = PushSpy()
    push_processes = PushSpy()
    sampler = GpuSampler(push=push, node="ci", push_processes=push_processes)
    assert sampler.available

    uuids = sampler.gpu_uuids()
    count = len(uuids)
    assert count >= 2, f"expected a multi-GPU CI runner, saw {count} device(s)"
    assert all(uuids)

    assert sampler.sample_once(ts=1.0) == count
    assert sampler.sample_processes_once(ts=1.0) >= 0
    sampler.flush()
    [(_, batch)] = push.calls
    assert [sample.gpu for sample in batch] == list(range(count))
    for sample in batch:
        assert 0 <= sample.util <= 100
        assert sample.mem_mb >= 0 and sample.power_w >= 0
    if push_processes.calls:
        for process_sample in push_processes.calls[0][1]:
            assert process_sample.pid > 0 and process_sample.mem_mb >= 0 and process_sample.name


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
