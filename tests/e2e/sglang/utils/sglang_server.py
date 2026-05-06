from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import requests

from miles.utils.http_utils import find_available_port

DEFAULT_HOST = "127.0.0.1"
DEFAULT_BASE_PORT = 34000
DEFAULT_STARTUP_TIMEOUT_SECS = 900.0
DEFAULT_SHUTDOWN_TIMEOUT_SECS = 30.0


@dataclass
class SGLangServer:
    process: subprocess.Popen
    host: str
    port: int
    log_path: Path
    _log_file: IO[str]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self, timeout_secs: float = DEFAULT_SHUTDOWN_TIMEOUT_SECS) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_secs)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout_secs)
        self._log_file.close()


def start_sglang_server(
    *,
    model_path: str,
    host: str = DEFAULT_HOST,
    port: int | None = None,
    startup_timeout_secs: float = DEFAULT_STARTUP_TIMEOUT_SECS,
    enable_deterministic_inference: bool = True,
    extra_args: list[str] | None = None,
) -> SGLangServer:
    if port is None:
        port = find_available_port(DEFAULT_BASE_PORT)

    log_path = Path(f"/tmp/sglang_e2e_{port}.log")
    log_file = log_path.open("w", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    if enable_deterministic_inference:
        cmd.append("--enable-deterministic-inference")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    server = SGLangServer(process=process, host=host, port=port, log_path=log_path, _log_file=log_file)

    _wait_for_ready(server, timeout_secs=startup_timeout_secs)
    return server


def _wait_for_ready(server: SGLangServer, *, timeout_secs: float) -> None:
    deadline = time.monotonic() + timeout_secs
    last_error = ""

    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            log_tail = _read_log_tail(server.log_path)
            raise RuntimeError(
                "SGLang server exited early. " f"Exit code: {server.process.returncode}. " f"Log tail:\n{log_tail}"
            )

        try:
            response = requests.get(f"{server.base_url}/health", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"status_code={response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)

        time.sleep(1.0)

    log_tail = _read_log_tail(server.log_path)
    raise TimeoutError(
        "Timed out waiting for SGLang server to become healthy. "
        f"Last error: {last_error}. "
        f"Log tail:\n{log_tail}"
    )


def _read_log_tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""

    content = path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    return "\n".join(lines[-max_lines:])
