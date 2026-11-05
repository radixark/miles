"""The memory report tells this process's share of a GPU from a colocated
engine's: ``used_GB`` is device-wide, the ``processes`` list is per process."""

import logging

import pytest
import torch

from miles.utils import memory_utils

CSV = "11720, 72888, ray::TorchtitanTrainRayActorWithConcurrencyGroups\n13978, 4946, /opt/sglang/bin/python3\n14266, 195834, sglang::scheduler_TP0_EP0\n"


def test_compute_apps_rows_become_records():
    assert memory_utils.parse_compute_apps(CSV) == [
        {"pid": 11720, "used_MiB": 72888, "name": "ray::TorchtitanTrainRayActorWithConcurrencyGroups"},
        {"pid": 13978, "used_MiB": 4946, "name": "/opt/sglang/bin/python3"},
        {"pid": 14266, "used_MiB": 195834, "name": "sglang::scheduler_TP0_EP0"},
    ]


def test_a_row_that_does_not_parse_is_skipped():
    assert memory_utils.parse_compute_apps("No running processes found\n12, 5, x\n") == [
        {"pid": 12, "used_MiB": 5, "name": "x"}
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device for the uuid")
def test_gpu_processes_asks_nvidia_smi_for_this_device(monkeypatch):
    seen = {}

    def query(args):
        seen["args"] = args
        return CSV

    rows = memory_utils.gpu_processes(query=query)
    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    assert seen["args"][1] == f"--id=GPU-{uuid}"
    assert [row["pid"] for row in rows] == [11720, 13978, 14266]


def test_a_failing_nvidia_smi_is_a_warning_not_an_error(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: type("P", (), {"uuid": "abc"})())

    def query(args):
        raise FileNotFoundError("nvidia-smi")

    with caplog.at_level(logging.WARNING):
        assert memory_utils.gpu_processes(query=query) == []
    assert "nvidia-smi" in caplog.text


def test_reset_peak_memory_leaves_an_uninitialized_runtime_alone(monkeypatch):
    import torch

    from miles.utils import memory_utils

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.accelerator, "reset_peak_memory_stats", lambda *_args: pytest.fail("touched the device runtime")
    )
    memory_utils.reset_peak_memory()


def test_reset_peak_memory_resets_the_current_device(monkeypatch):
    import torch

    from miles.utils import memory_utils

    reset: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda device: reset.append(device))
    memory_utils.reset_peak_memory()
    assert reset == [3]
