from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import signal
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psutil
import ray

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0)
class FaultInjectorActor:
    """Ray Actor for injecting faults on a specific node during E2E tests.

    Deployed to a target node via NodeAffinitySchedulingStrategy.
    """

    def __init__(self) -> None:
        self._stress_pids: list[int] = []
        self._filled_paths: list[str] = []
        self._exception_flag_paths: list[str] = []

    def find_training_processes(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for proc in psutil.process_iter(["pid", "cmdline", "name"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline).lower()

                if any(pattern in cmdline_str for pattern in _TRAINING_CMDLINE_PATTERNS):
                    results.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info.get("name", ""),
                            "cmdline": cmdline,
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return results

    def find_sglang_processes(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for proc in psutil.process_iter(["pid", "cmdline", "name"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline).lower()

                if any(pattern in cmdline_str for pattern in _SGLANG_CMDLINE_PATTERNS):
                    results.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info.get("name", ""),
                            "cmdline": cmdline,
                        }
                    )
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
        self._filled_paths.append(path)

        chunk_size = 64 * 1024 * 1024
        chunk = b"\0" * chunk_size

        try:
            with open(path, "wb") as f:
                remaining = size_bytes
                while remaining > 0:
                    write_size = min(chunk_size, remaining)
                    f.write(chunk[:write_size])
                    remaining -= write_size
        except Exception:
            logger.warning("fill_disk write failed path=%s", path, exc_info=True)
            _remove_if_exists(path)
            self._filled_paths.remove(path)
            raise

    def trigger_gpu_xid(self) -> None:
        """Trigger XID 13 (Graphics Engine Exception) via illegal memory access.

        Compiles trigger_xid.cu on first call (cached at /tmp/trigger_xid),
        then runs the binary.  The short-lived process creates its own CUDA
        context which is destroyed on exit, so other GPU workloads are not
        affected.  XID 13 should appear in /dev/kmsg within ~1 s.
        """
        _ensure_trigger_xid_binary()
        logger.info("trigger_gpu_xid: running %s", _TRIGGER_XID_BINARY)
        result = subprocess.run(
            [str(_TRIGGER_XID_BINARY)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        logger.info(
            "trigger_gpu_xid: exit=%d stdout=%s stderr=%s",
            result.returncode,
            result.stdout[:500] if result.stdout else "",
            result.stderr[:500] if result.stderr else "",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"trigger_gpu_xid failed with exit code {result.returncode}: "
                f"stdout={result.stdout[:500] if result.stdout else ''} "
                f"stderr={result.stderr[:500] if result.stderr else ''}"
            )

    def write_exception_flag(self, path: str) -> None:
        logger.info("write_exception_flag path=%s", path)
        Path(path).touch()
        if path not in self._exception_flag_paths:
            self._exception_flag_paths.append(path)

    def broadcast_exception_flags(self, inject_dir: str, ranks: list[int]) -> None:
        """Write per-rank exception flags so every target rank independently
        consumes its own flag file, avoiding the single-path race where
        only the first process to check would consume the injection."""
        from miles.utils.ft.utils.env import build_exception_inject_flag_path

        dir_path = Path(inject_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        for rank in ranks:
            flag = build_exception_inject_flag_path(dir_path, rank=rank)
            logger.info("broadcast_exception_flag rank=%d path=%s", rank, flag)
            flag.touch()
            str_flag = str(flag)
            if str_flag not in self._exception_flag_paths:
                self._exception_flag_paths.append(str_flag)

    def remove_exception_flag(self, path: str) -> None:
        logger.info("remove_exception_flag path=%s", path)
        _remove_if_exists(path)
        if path in self._exception_flag_paths:
            self._exception_flag_paths.remove(path)

    def cleanup_disk(self, path: str) -> None:
        logger.info("cleanup_disk path=%s", path)
        _remove_if_exists(path)
        if path in self._filled_paths:
            self._filled_paths.remove(path)

    def cleanup_all(self) -> None:
        logger.info(
            "cleanup_all stress_pids=%s filled_paths=%s exception_flag_paths=%s",
            self._stress_pids,
            self._filled_paths,
            self._exception_flag_paths,
        )
        for pid in list(self._stress_pids):
            _kill_if_exists(pid)
        self._stress_pids.clear()

        for path in list(self._filled_paths):
            _remove_if_exists(path)
        self._filled_paths.clear()

        for path in list(self._exception_flag_paths):
            _remove_if_exists(path)
        self._exception_flag_paths.clear()


def deploy_fault_injector(
    node_id: str,
    ft_id: str = "",
) -> ray.actor.ActorHandle:
    """Create and deploy a FaultInjectorActor to a specific node."""
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=False,
    )
    name_prefix = f"fault_injector_{ft_id}_" if ft_id else "fault_injector_"
    actor_name = f"{name_prefix}{node_id}"
    logger.info("fault_injector: deploying to node_id=%s, actor_name=%s", node_id, actor_name)
    actor = FaultInjectorActor.options(  # type: ignore[attr-defined]
        scheduling_strategy=scheduling_strategy,
        name=actor_name,
    ).remote()
    return actor


_TRAINING_CMDLINE_PATTERNS = ("megatron", "run_deepseek", "run_train", "torchrun")
_SGLANG_CMDLINE_PATTERNS = ("sglang", "launch_server")

_GPU_STRESS_SCRIPT = Path(__file__).parent / "gpu_stress.py"
_TRIGGER_XID_SOURCE = Path(__file__).parent / "trigger_xid.cu"
_TRIGGER_XID_BINARY = Path("/tmp/trigger_xid")


_TRIGGER_XID_LOCK = Path("/tmp/trigger_xid.lock")


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _compile_to_temp_then_replace(source: Path, target: Path) -> None:
    tmp_path = Path(f"{target}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        logger.info("fault_injector: compiling %s -> %s (via %s)", source, target, tmp_path)
        subprocess.run(
            ["nvcc", "-o", str(tmp_path), str(source)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        os.replace(str(tmp_path), str(target))
        logger.info("fault_injector: compilation complete target=%s", target)
    except Exception:
        logger.error("fault_injector: compilation failed source=%s", source, exc_info=True)
        tmp_path.unlink(missing_ok=True)
        raise


def _ensure_trigger_xid_binary() -> None:
    with _file_lock(_TRIGGER_XID_LOCK):
        if _TRIGGER_XID_BINARY.exists() and os.access(_TRIGGER_XID_BINARY, os.X_OK):
            logger.debug("fault_injector: trigger_xid binary already exists at %s", _TRIGGER_XID_BINARY)
            return
        _compile_to_temp_then_replace(source=_TRIGGER_XID_SOURCE, target=_TRIGGER_XID_BINARY)


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
