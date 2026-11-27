from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from miles.utils.external_utils.colocate_pairing import devices, publish_devices
from miles.utils.external_utils.colocate_pairing.pods import PodCoordinate


class TestDiscoverAssignedGpuUuids:
    def test_returns_ordered_uuids_from_nvidia_smi(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(
            devices.subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="GPU-1111\nGPU-4444\nGPU-6666\nGPU-7777\n",
                stderr="",
            ),
        )

        assert devices.discover_assigned_gpu_uuids(expected_count=4) == (
            "GPU-1111",
            "GPU-4444",
            "GPU-6666",
            "GPU-7777",
        )

    def test_applies_numeric_cuda_visible_devices_subset(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
        monkeypatch.setattr(
            devices.subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="GPU-0\nGPU-1\nGPU-2\nGPU-3\n",
                stderr="",
            ),
        )

        assert devices.discover_assigned_gpu_uuids(expected_count=2) == ("GPU-3", "GPU-1")

    def test_applies_uuid_cuda_visible_devices_subset(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-c,GPU-a")
        monkeypatch.setattr(
            devices.subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="GPU-a\nGPU-b\nGPU-c\n",
                stderr="",
            ),
        )

        assert devices.discover_assigned_gpu_uuids(expected_count=2) == ("GPU-c", "GPU-a")

    def test_fails_when_gpu_count_does_not_match_expected(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(
            devices.subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="GPU-a\nGPU-b\n",
                stderr="",
            ),
        )

        with pytest.raises(RuntimeError, match="expected 4 assigned GPU"):
            devices.discover_assigned_gpu_uuids(expected_count=4)


class TestPublishDevicesEntrypoint:
    def test_writes_atomic_pairing_file_and_execs_trainer_command(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("MILES_POD_UID", "trainer-uid-42")
        monkeypatch.setenv("MILES_NODE_NAME", "gpu-node-71-29")
        monkeypatch.setenv("MILES_CELL_INDEX", "0")
        monkeypatch.setenv("MILES_POD_INDEX", "1")
        monkeypatch.setattr(
            publish_devices,
            "discover_assigned_gpu_uuids",
            lambda *, expected_count: ("GPU-b", "GPU-e", "GPU-f", "GPU-h"),
        )
        exec_calls: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(publish_devices.os, "execvp", lambda prog, argv: exec_calls.append((prog, argv)))

        pairing_dir = tmp_path / "pairing"
        publish_devices.main(
            [
                "--pairing-dir",
                str(pairing_dir),
                "--pool-id",
                "trainer-engine-actor",
                "--expected-gpus",
                "4",
                "--",
                "python",
                "-m",
                "miles.utils.workers.serving.serve",
            ]
        )

        coord = PodCoordinate(pool_id="trainer-engine-actor", cell_index=0, pod_in_cell_index=1)
        recorded = devices.read_trainer_devices(pairing_dir=pairing_dir, coord=coord)
        assert recorded == devices.TrainerDevices(
            pod_uid="trainer-uid-42",
            node_name="gpu-node-71-29",
            gpu_uuids=("GPU-b", "GPU-e", "GPU-f", "GPU-h"),
        )
        assert exec_calls == [("python", ["python", "-m", "miles.utils.workers.serving.serve"])]
