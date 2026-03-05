from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil
import ray

logger = logging.getLogger(__name__)

_TRAINING_CMDLINE_PATTERNS = ("megatron", "run_deepseek", "run_train", "torchrun")

_GPU_STRESS_SCRIPT = Path(__file__).parent / "gpu_stress.py"


def _kill_if_exists(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _remove_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


@ray.remote(num_gpus=0)
class FaultInjectorActor:
    """Ray Actor for injecting faults on a specific node during E2E tests.

    Deployed to a target node via NodeAffinitySchedulingStrategy.
    """

    def __init__(self) -> None:
        self._stress_pids: list[int] = []
        self._filled_paths: list[str] = []

    def find_training_processes(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for proc in psutil.process_iter(["pid", "cmdline", "name"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline).lower()

                if any(pattern in cmdline_str for pattern in _TRAINING_CMDLINE_PATTERNS):
                    results.append({
                        "pid": proc.info["pid"],
                        "name": proc.info.get("name", ""),
                        "cmdline": cmdline,
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return results

    def kill_process(self, pid: int, sig: int = signal.SIGKILL) -> None:
        logger.info("kill_process pid=%d sig=%d", pid, sig)
        os.kill(pid, sig)

    def stop_process(self, pid: int) -> None:
        logger.info("stop_process pid=%d (SIGSTOP)", pid)
        os.kill(pid, signal.SIGSTOP)

    def continue_process(self, pid: int) -> None:
        logger.info("continue_process pid=%d (SIGCONT)", pid)
        os.kill(pid, signal.SIGCONT)

    def start_gpu_stress(self) -> int:
        proc = subprocess.Popen(
            [sys.executable, str(_GPU_STRESS_SCRIPT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._stress_pids.append(proc.pid)
        logger.info("start_gpu_stress pid=%d", proc.pid)
        return proc.pid

    def stop_gpu_stress(self, pid: int) -> None:
        logger.info("stop_gpu_stress pid=%d", pid)
        _kill_if_exists(pid)
        if pid in self._stress_pids:
            self._stress_pids.remove(pid)

    def fill_disk(self, path: str, size_bytes: int) -> None:
        logger.info("fill_disk path=%s size_bytes=%d", path, size_bytes)
        chunk_size = 64 * 1024 * 1024
        chunk = b"\0" * chunk_size

        with open(path, "wb") as f:
            remaining = size_bytes
            while remaining > 0:
                write_size = min(chunk_size, remaining)
                f.write(chunk[:write_size])
                remaining -= write_size
        self._filled_paths.append(path)

    def cleanup_disk(self, path: str) -> None:
        logger.info("cleanup_disk path=%s", path)
        _remove_if_exists(path)
        if path in self._filled_paths:
            self._filled_paths.remove(path)

    def cleanup_all(self) -> None:
        logger.info("cleanup_all stress_pids=%s filled_paths=%s", self._stress_pids, self._filled_paths)
        for pid in list(self._stress_pids):
            _kill_if_exists(pid)
        self._stress_pids.clear()

        for path in list(self._filled_paths):
            _remove_if_exists(path)
        self._filled_paths.clear()


def deploy_fault_injector(node_id: str) -> ray.actor.ActorHandle:
    """Create and deploy a FaultInjectorActor to a specific node."""
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=False,
    )
    actor = FaultInjectorActor.options(  # type: ignore[attr-defined]
        scheduling_strategy=scheduling_strategy,
        name=f"fault_injector_{node_id}",
    ).remote()
    return actor
