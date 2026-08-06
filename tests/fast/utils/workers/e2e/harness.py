from __future__ import annotations

import contextlib
import dataclasses
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKER_PATH = "tests.fast.utils.workers.e2e.e2e_worker.make_worker"

READY_TIMEOUT_SECONDS = 60.0
STOP_TIMEOUT_SECONDS = 15.0
KILL_TIMEOUT_SECONDS = 10.0


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclasses.dataclass
class ServerProcess:
    port: int
    process: subprocess.Popen
    log_path: Path

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def logs(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path.exists() else ""

    def is_running(self) -> bool:
        return self.process.poll() is None

    def signal(self, signal_number: int) -> None:
        if self.is_running():
            self.process.send_signal(signal_number)

    def wait(self, timeout: float) -> int | None:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self) -> int | None:
        if not self.is_running():
            return self.process.returncode

        self.process.terminate()
        exit_code = self.wait(STOP_TIMEOUT_SECONDS)
        if exit_code is None:
            self.kill()
            exit_code = self.wait(KILL_TIMEOUT_SECONDS)
        return exit_code

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)


def spawn_server(
    *,
    state_dir: Path,
    log_path: Path,
    port: int | None = None,
    worker_argv: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    worker_path: str = WORKER_PATH,
) -> ServerProcess:
    port = reserve_port() if port is None else port

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra_env or {})

    argv = [sys.executable, "-m", "miles.utils.workers.serving.serve_inner", "--worker", worker_path]
    argv += ["--host", "127.0.0.1", "--port", str(port)]
    argv += ["--", "--state-dir", str(state_dir)]
    argv += worker_argv or []

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )

    return ServerProcess(port=port, process=process, log_path=log_path)


def wait_until_serving(server: ServerProcess, timeout: float = READY_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        exit_code = server.process.poll()
        assert exit_code is None, f"server exited with {exit_code} before serving:\n{server.logs()}"
        with contextlib.suppress(httpx.TransportError):
            if httpx.get(f"{server.url}/v1/health", timeout=2.0, trust_env=False).status_code == 200:
                return
        time.sleep(0.05)

    raise AssertionError(f"server never became ready within {timeout}s:\n{server.logs()}")
