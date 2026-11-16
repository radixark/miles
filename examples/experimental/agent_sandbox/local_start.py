"""A sandbox start function that runs each call's agent worker as a process on this host.

It runs a worker per call without a sandbox service:

    --custom-agent-function-mode worker
    --custom-agent-function-worker examples.experimental.agent_sandbox.local_start.start
"""

import os
import socket
import subprocess
import sys
from collections.abc import Callable

import psutil

from miles.rollout.agentic.agent_function import _CANCEL_GRACE_S, _LEFTOVER_CLEANUP_S

_STOP_TIMEOUT_S = 30.0


def start(task_id: str, tasks_dir: str) -> tuple[Callable[[], None], str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "miles.rollout.agentic.worker", "--host", "127.0.0.1", "--port", str(port)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, sys.path))},
    )
    return (lambda: _stop(proc)), f"http://127.0.0.1:{port}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _stop(proc: subprocess.Popen) -> None:
    _wait_for_agent_exit(proc)
    proc.terminate()
    try:
        proc.wait(timeout=_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _wait_for_agent_exit(proc: subprocess.Popen) -> None:
    try:
        agents = psutil.Process(proc.pid).children()
    except psutil.NoSuchProcess:
        return
    psutil.wait_procs(agents, timeout=_CANCEL_GRACE_S + _LEFTOVER_CLEANUP_S)
