import atexit
import ctypes
import os
import signal
import subprocess
import sys
from functools import partial

_PR_SET_PDEATHSIG = 1

_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None


def launch_bound_subprocess(argv: list[str], *, envs: dict[str, str]) -> subprocess.Popen:
    parent_pid = os.getpid()
    process = subprocess.Popen(
        argv,
        env={**os.environ, **envs},
        start_new_session=True,
        preexec_fn=partial(_set_parent_death_signal, expected_parent_pid=parent_pid) if _LIBC is not None else None,
    )
    atexit.register(_terminate_process_tree, process)
    return process


def _terminate_process_tree(process: subprocess.Popen, *, sigkill_timeout: float = 5.0) -> None:
    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=sigkill_timeout)
    except subprocess.TimeoutExpired:
        pass
    _signal_process_group(process.pid, signal.SIGKILL)
    process.wait()


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass


def _set_parent_death_signal(*, expected_parent_pid: int) -> None:
    _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    if os.getppid() != expected_parent_pid:
        os._exit(1)
