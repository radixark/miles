from __future__ import annotations

import asyncio
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


def port_is_refused(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) != 0


@dataclasses.dataclass
class ProxyRequest:
    at: float
    verb: str
    path: str
    body: bytes


class FlakyProxy:
    """A local TCP relay that records requests and can inject HTTP failures."""

    def __init__(self, upstream_port: int | None) -> None:
        self._upstream_port = upstream_port
        self._server: asyncio.Server | None = None
        self.requests: list[ProxyRequest] = []
        self.reject_status: int | None = None
        self.reject_remaining = 0
        self.drop_remaining = 0
        self.strip_boot_uuid = False
        self.rewrite_boot_uuid: str | None = None
        self.record_only = False

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.sockets[0].getsockname()[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def submits(self, method: str) -> list[ProxyRequest]:
        return [r for r in self.requests if r.verb == "POST" and r.path == f"/v1/{method}"]

    def reject_next(self, count: int, status: int) -> None:
        self.reject_remaining = count
        self.reject_status = status

    def drop_next(self, count: int) -> None:
        self.drop_remaining = count

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await _read_http_message(reader)
            if request is None:
                return

            verb, path = _parse_request_line(request)
            self.requests.append(ProxyRequest(at=time.monotonic(), verb=verb, path=path, body=request))

            if self.record_only:
                _write_simple(writer, 503, b"record-only proxy")
                await writer.drain()
                return

            if self.reject_remaining != 0 and self.reject_status is not None:
                self.reject_remaining -= 1
                _write_simple(writer, self.reject_status, b"injected failure")
                await writer.drain()
                return

            assert self._upstream_port is not None
            response = await _forward(self._upstream_port, request)

            if self.drop_remaining != 0:
                self.drop_remaining -= 1
                return

            writer.write(self._maybe_rewrite(response))
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    def _maybe_rewrite(self, response: bytes) -> bytes:
        if not self.strip_boot_uuid and self.rewrite_boot_uuid is None:
            return response

        head, separator, body = response.partition(b"\r\n\r\n")
        kept = []
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"x-miles-boot-uuid:"):
                if self.strip_boot_uuid:
                    continue
                line = b"x-miles-boot-uuid: " + self.rewrite_boot_uuid.encode()
            kept.append(line)
        return b"\r\n".join(kept) + separator + body


async def _forward(upstream_port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", upstream_port)
    try:
        writer.write(request)
        await writer.drain()
        return await _read_http_message(reader, require_body=True) or b""
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def _read_http_message(reader: asyncio.StreamReader, require_body: bool = False) -> bytes | None:
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = await reader.read(4096)
        if not chunk:
            return head or None
        head += chunk

    header_blob, _, rest = head.partition(b"\r\n\r\n")
    length = 0
    for line in header_blob.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])

    body = rest
    while len(body) < length:
        chunk = await reader.read(4096)
        if not chunk:
            break
        body += chunk

    if require_body and length == 0 and b"transfer-encoding: chunked" in header_blob.lower():
        while not body.endswith(b"0\r\n\r\n"):
            chunk = await reader.read(4096)
            if not chunk:
                break
            body += chunk

    return header_blob + b"\r\n\r\n" + body


def _parse_request_line(request: bytes) -> tuple[str, str]:
    first_line = request.split(b"\r\n", 1)[0].decode(errors="replace")
    parts = first_line.split(" ")
    return (parts[0], parts[1].split("?")[0]) if len(parts) >= 2 else ("", "")


def _write_simple(writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
    writer.write(
        f"HTTP/1.1 {status} Injected\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
    )


class ConnectionCountingRelay:
    def __init__(self, upstream_port: int) -> None:
        self._upstream_port = upstream_port
        self._server: asyncio.Server | None = None
        self.accepted = 0

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.sockets[0].getsockname()[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.accepted += 1
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", self._upstream_port)

        try:
            await asyncio.gather(_pump(reader, upstream_writer), _pump(upstream_reader, writer))
        finally:
            for stream in (upstream_writer, writer):
                with contextlib.suppress(Exception):
                    stream.close()


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(ConnectionResetError, BrokenPipeError):
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()

    with contextlib.suppress(Exception):
        writer.write_eof()
