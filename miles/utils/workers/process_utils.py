import atexit
import logging
import os
import signal
import subprocess
import sys

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


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass
