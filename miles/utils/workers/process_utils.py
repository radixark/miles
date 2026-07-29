import atexit
import ctypes
import os
import signal
import subprocess
import sys

_PR_SET_PDEATHSIG = 1

_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None


def launch_bound_subprocess(argv: list[str], *, envs: dict[str, str]) -> subprocess.Popen:
    process = subprocess.Popen(
        argv,
        env={**os.environ, **envs},
        start_new_session=True,
        preexec_fn=_set_parent_death_signal if _LIBC is not None else None,
    )
    atexit.register(terminate_process_tree, process)
    return process


def terminate_process_tree(process: subprocess.Popen, *, sigkill_timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return

    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=sigkill_timeout)
    except subprocess.TimeoutExpired:
        _signal_process_group(process.pid, signal.SIGKILL)
        process.wait()


def kill_process_tree(process: subprocess.Popen) -> None:
    _signal_process_group(process.pid, signal.SIGKILL)


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass


def _set_parent_death_signal() -> None:
    _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
