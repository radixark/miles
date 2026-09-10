import atexit
import logging
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import ExitStack

import psutil

logger = logging.getLogger(__name__)

_TRAMPOLINE_MODULE = "miles.utils.workers.process_trampoline"


def launch_bound_subprocess(argv: list[str], *, envs: dict[str, str]) -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, "-m", _TRAMPOLINE_MODULE, str(os.getpid()), *argv],
        env={**os.environ, **envs},
        start_new_session=True,
    )
    atexit.register(terminate_process_tree, process)
    return process


def terminate_process_tree(process: subprocess.Popen, *, sigkill_timeout: float = 5.0) -> None:
    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=sigkill_timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Process %d did not exit within %.1fs after SIGTERM; escalating to SIGKILL", process.pid, sigkill_timeout
        )
    _signal_process_group(process.pid, signal.SIGKILL)
    process.wait()


def kill_process_tree(process: subprocess.Popen) -> None:
    _signal_process_group(process.pid, signal.SIGKILL)


def kill_process_tree_and_wait(process: subprocess.Popen, *, timeout_seconds: float = 5.0) -> list[int]:
    root = psutil.Process(process.pid)
    deadline = time.monotonic() + timeout_seconds
    with ExitStack() as resources:
        handles = {root.pid: _freeze_process(root, resources=resources, deadline=deadline)}
        while children := [child for child in root.children(recursive=True) if child.pid not in handles]:
            for child in children:
                handles[child.pid] = _freeze_process(child, resources=resources, deadline=deadline)
        for fd in handles.values():
            signal.pidfd_send_signal(fd, signal.SIGKILL)
        pending = set(handles.values())
        while pending:
            readable, _, _ = select.select(list(pending), [], [], max(0.0, deadline - time.monotonic()))
            if not readable:
                raise TimeoutError(f"Process tree {process.pid} did not exit within {timeout_seconds}s")
            pending.difference_update(readable)
    return list(handles)


def _freeze_process(observed: psutil.Process, *, resources: ExitStack, deadline: float) -> int:
    fd = os.pidfd_open(observed.pid)
    resources.callback(os.close, fd)
    if not observed.is_running() or select.select([fd], [], [], 0)[0]:
        raise ProcessLookupError(f"Process {observed.pid} exited before injection")
    if observed.status() != psutil.STATUS_STOPPED:
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
        resources.callback(_idempotent_resume_process, fd)
    while observed.is_running() and observed.status() != psutil.STATUS_STOPPED:
        if select.select([fd], [], [], 0)[0]:
            raise ProcessLookupError(f"Process {observed.pid} exited before it stopped")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Process {observed.pid} did not stop before the injection deadline")
        time.sleep(0.01)
    if not observed.is_running():
        raise ProcessLookupError(f"Process {observed.pid} changed before injection")
    return fd


def _idempotent_resume_process(fd: int) -> None:
    try:
        signal.pidfd_send_signal(fd, signal.SIGCONT)
    except ProcessLookupError:
        pass


def kill_process(process: subprocess.Popen) -> None:
    try:
        os.kill(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass
