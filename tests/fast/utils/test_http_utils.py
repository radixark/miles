"""Tests for wait_for_server_ready in http_utils.

wait_for_server_ready() polls a TCP port in a loop until the server is
accepting connections.  Each iteration:
  1. Check if the process is still alive (if a process handle was given).
  2. Try ``socket.create_connection((host, port))``.
  3. If it connects → server is ready, return.
  4. If OSError (connection refused) → sleep 0.5s, retry.
  5. If ``time.time()`` exceeds the deadline → raise RuntimeError.

TestWaitForServerReady uses real sockets/threads to verify end-to-end
behaviour (port already open, delayed open, timeout, dead process).

TestWaitForServerReadySimulatedDelays uses mocks to test long startup
delays (5s / 10s / 20s) without actually waiting.  The trick:
  - Mock ``time.time()``  → returns a fake clock that we control.
  - Mock ``time.sleep()`` → doesn't really wait, just advances the fake clock.
  - Mock ``socket.create_connection()`` → raises OSError for the first N
    calls (simulating "port not ready"), then returns a fake socket
    (simulating "port ready").
This lets us simulate 20 seconds of polling in <1ms of real time.
"""

import asyncio
import multiprocessing
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from miles.utils import http_utils
from miles.utils.http_utils import GeneralHttpClientProvider, wait_for_server_ready, wait_tcp_ready


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _listen_after_delay(host: str, port: int, delay: float, stop_event: threading.Event):
    """Open a TCP listener after *delay* seconds, keep it open until *stop_event* is set."""
    time.sleep(delay)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    stop_event.wait()
    srv.close()


# ---------------------------------------------------------------------------
# Real-network tests (use actual sockets and threads)
# ---------------------------------------------------------------------------


class TestWaitForServerReady:
    def test_returns_when_port_is_already_open(self):
        """Server is already listening → should return immediately."""
        port = _find_free_port()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        try:
            wait_for_server_ready("127.0.0.1", port, timeout=2)
        finally:
            srv.close()

    def test_waits_until_port_opens(self):
        """Server starts listening after 1s → should wait and then return."""
        port = _find_free_port()
        stop = threading.Event()
        t = threading.Thread(target=_listen_after_delay, args=("127.0.0.1", port, 1.0, stop))
        t.daemon = True
        t.start()
        try:
            start = time.time()
            wait_for_server_ready("127.0.0.1", port, timeout=10)
            elapsed = time.time() - start
            assert elapsed >= 0.8, f"Should have waited ~1s, waited {elapsed:.2f}s"
        finally:
            stop.set()

    def test_raises_on_timeout(self):
        """No server on the port → should raise after timeout."""
        port = _find_free_port()
        with pytest.raises(RuntimeError, match="not ready after"):
            wait_for_server_ready("127.0.0.1", port, timeout=1)

    def test_raises_when_process_dies(self):
        """Process exits before port is ready → should raise immediately."""
        port = _find_free_port()

        def _die_immediately():
            pass

        proc = multiprocessing.Process(target=_die_immediately)
        proc.start()
        proc.join()  # ensure it's dead before we call wait

        with pytest.raises(RuntimeError, match="process died"):
            wait_for_server_ready("127.0.0.1", port, process=proc, timeout=5)

    def test_raises_when_subprocess_dies(self) -> None:
        """Subprocess exits before port is ready and raises immediately."""
        process: subprocess.Popen[bytes] = subprocess.Popen([sys.executable, "-c", "raise SystemExit(2)"])
        process.wait(timeout=5)

        with pytest.raises(RuntimeError, match="process died"):
            wait_for_server_ready("127.0.0.1", 0, process=process, timeout=5)

    def test_waits_for_a_live_subprocess_to_open_its_port(self) -> None:
        """A still-running subprocess counts as alive, so a port opened later is awaited."""
        port = _find_free_port()
        stop = threading.Event()
        process: subprocess.Popen[bytes] = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        listener = threading.Thread(target=_listen_after_delay, args=("127.0.0.1", port, 1.0, stop))
        listener.daemon = True
        listener.start()

        try:
            wait_for_server_ready("127.0.0.1", port, process=process, timeout=10)
            assert process.poll() is None
        finally:
            stop.set()
            process.kill()
            process.wait(timeout=5)

    def test_raises_when_subprocess_exits_while_waiting_for_port(self) -> None:
        """A subprocess alive at the first poll but exiting later fails as a dead process."""
        port = _find_free_port()
        process: subprocess.Popen[bytes] = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])

        try:
            assert process.poll() is None
            with pytest.raises(RuntimeError, match="process died"):
                wait_for_server_ready("127.0.0.1", port, process=process, timeout=30)
        finally:
            process.kill()
            process.wait(timeout=5)


# ---------------------------------------------------------------------------
# Simulated-delay tests (mock time + socket so tests run instantly)
#
# How it works:
#   wait_for_server_ready() calls time.time() / time.sleep() / socket.create_connection()
#   in a loop.  We replace all three:
#
#   - fake_time_fn():  returns a counter we control (starts at 0.0)
#   - fake_sleep(d):   advances the counter by d (no real waiting)
#   - fake_connect():  raises OSError for the first N calls, then succeeds
#
#   This makes the loop run through all its iterations at full speed while
#   the function "thinks" real time is passing.  A 20s simulated delay
#   finishes in <1ms of wall-clock time.
# ---------------------------------------------------------------------------


class _FakeSocket:
    """Minimal stand-in for a connected socket (used as context manager)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestWaitForServerReadySimulatedDelays:
    @pytest.mark.parametrize("simulated_delay_s", [5, 10, 20])
    def test_succeeds_after_simulated_delay(self, simulated_delay_s):
        """Port becomes available after simulated_delay_s seconds.

        wait_for_server_ready polls every 0.5s, so it should take
        (simulated_delay_s / 0.5) failed attempts before succeeding.
        """
        poll_interval = 0.5  # matches the sleep(0.5) inside wait_for_server_ready
        polls_until_ready = int(simulated_delay_s / poll_interval)
        call_count = 0
        fake_time = [0.0]

        def fake_time_fn():
            return fake_time[0]

        def fake_sleep(duration):
            # Don't really sleep — just advance the fake clock.
            fake_time[0] += duration

        def fake_connect(addr, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count <= polls_until_ready:
                # Simulate "port not listening yet"
                raise OSError("Connection refused")
            # Simulate "port is now accepting connections"
            return _FakeSocket()

        with (
            patch("miles.utils.http_utils.time.time", side_effect=fake_time_fn),
            patch("miles.utils.http_utils.time.sleep", side_effect=fake_sleep),
            patch("miles.utils.http_utils.socket.create_connection", side_effect=fake_connect),
        ):
            wait_for_server_ready("127.0.0.1", 9999, timeout=simulated_delay_s + 10)

        # Should have polled exactly polls_until_ready times (fail) + 1 (success)
        assert call_count == polls_until_ready + 1

    @pytest.mark.parametrize("timeout", [5, 10, 20])
    def test_timeout_after_simulated_duration(self, timeout):
        """Port never opens → should raise after exactly *timeout* simulated seconds."""
        fake_time = [0.0]

        def fake_time_fn():
            return fake_time[0]

        def fake_sleep(duration):
            fake_time[0] += duration

        def fake_connect(addr, timeout=None):
            # Always fail — server never starts
            raise OSError("Connection refused")

        with (
            patch("miles.utils.http_utils.time.time", side_effect=fake_time_fn),
            patch("miles.utils.http_utils.time.sleep", side_effect=fake_sleep),
            patch("miles.utils.http_utils.socket.create_connection", side_effect=fake_connect),
        ):
            with pytest.raises(RuntimeError, match=f"not ready after {timeout}s"):
                wait_for_server_ready("127.0.0.1", 9999, timeout=timeout)

        # The fake clock should have advanced past the timeout
        assert fake_time[0] >= timeout


class _FakeWriter:
    """Minimal stand-in for the writer half of an opened connection."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


class TestWaitTcpReady:
    async def test_keeps_retrying_until_the_port_accepts(self, monkeypatch):
        """Readiness depends on the endpoint alone, retrying while it refuses connections."""
        attempts: list[tuple[str, int]] = []
        writer = _FakeWriter()

        async def fake_open_connection(host, port):
            attempts.append((host, port))
            if len(attempts) < 3:
                raise OSError("Connection refused")
            return object(), writer

        monkeypatch.setattr(http_utils, "_CONNECT_RETRY_INTERVAL_SECONDS", 0)
        monkeypatch.setattr(http_utils.asyncio, "open_connection", fake_open_connection)

        await wait_tcp_ready("[2001:db8::7]", 23456, timeout=30)

        assert attempts == [("2001:db8::7", 23456)] * 3
        assert writer.closed

    async def test_gives_up_when_the_deadline_passes(self, monkeypatch):
        """A port that never opens fails with a timeout instead of blocking forever."""

        async def fake_open_connection(host, port):
            raise OSError("Connection refused")

        monkeypatch.setattr(http_utils, "_CONNECT_RETRY_INTERVAL_SECONDS", 0)
        monkeypatch.setattr(http_utils.asyncio, "open_connection", fake_open_connection)

        with pytest.raises(RuntimeError, match="Server at 127.0.0.1:23456 not ready after 0.05s"):
            await wait_tcp_ready("127.0.0.1", 23456, timeout=0.05)

    async def test_a_connection_that_never_answers_is_one_refused_attempt(self, monkeypatch):
        """A syn that hangs must not hold the whole budget; each attempt has its own small timeout."""
        attempts: list[tuple[str, int]] = []

        async def fake_open_connection(host, port):
            attempts.append((host, port))
            await asyncio.sleep(10)

        monkeypatch.setattr(http_utils, "_CONNECT_ATTEMPT_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(http_utils, "_CONNECT_RETRY_INTERVAL_SECONDS", 0)
        monkeypatch.setattr(http_utils.asyncio, "open_connection", fake_open_connection)

        with pytest.raises(RuntimeError, match="not ready"):
            await wait_tcp_ready("127.0.0.1", 23456, timeout=0.05)

        assert len(attempts) > 1


class TestGeneralHttpClientProvider:
    """The provider hands out one httpx client per event loop."""

    async def test_the_same_loop_gets_the_same_client(self):
        """Two calls on one loop must share one connection pool."""
        assert GeneralHttpClientProvider.client() is GeneralHttpClientProvider.client()

    async def test_a_different_loop_gets_a_different_client(self):
        """A client's connections belong to the loop that opened them; reusing it elsewhere fails."""
        mine = GeneralHttpClientProvider.client()
        others: list[object] = []

        async def _on_the_other_loop():
            return GeneralHttpClientProvider.client()

        thread = threading.Thread(target=lambda: others.append(asyncio.run(_on_the_other_loop())))
        thread.start()
        thread.join(timeout=10)

        assert others and others[0] is not mine

    def test_calling_it_off_loop_fails_loudly(self):
        """Building the client outside a loop would bind it to whichever loop ran next."""
        with pytest.raises(RuntimeError):
            GeneralHttpClientProvider.client()

    async def test_the_client_has_no_read_timeout_but_a_short_connect_timeout(self):
        """A weight-update request blocks until the collective forms; an unreachable host must not."""
        timeout = GeneralHttpClientProvider.client().timeout

        assert timeout.read is None
        assert timeout.write is None
        assert timeout.pool is None
        assert timeout.connect == 10.0

    async def test_the_pool_is_configured_without_a_connection_cap(self):
        """A capped pool queues the last requests behind the collective waiting for them."""
        assert GeneralHttpClientProvider._LIMITS.max_connections is None
        assert GeneralHttpClientProvider._LIMITS.max_keepalive_connections is None

    async def test_more_requests_than_httpxs_default_cap_reach_the_server_at_once(self):
        """101 engines must all arrive before the caller joins the collective, and httpx caps
        connections at 100 by default."""
        num_requests = 101
        arrived = _ArrivalGate()
        server = _BlockingHttpServer(arrived)

        try:
            client = GeneralHttpClientProvider.client()
            requests = [asyncio.create_task(client.get(server.url)) for _ in range(num_requests)]

            deadline = time.monotonic() + 60
            while arrived.count < num_requests:
                assert time.monotonic() < deadline, (
                    f"only {arrived.count}/{num_requests} requests reached the server; the rest are "
                    "queued behind the connection cap"
                )
                await asyncio.sleep(0.01)

            arrived.release()
            assert [response.status_code for response in await asyncio.gather(*requests)] == [200] * num_requests
        finally:
            arrived.release()
            server.close()


class _ArrivalGate:
    def __init__(self):
        self.count = 0
        self._lock = threading.Lock()
        self._released = threading.Event()

    def wait_for_release(self) -> None:
        with self._lock:
            self.count += 1
        self._released.wait(timeout=60)

    def release(self) -> None:
        self._released.set()


class _BlockingHttpServer:
    def __init__(self, gate: _ArrivalGate):
        handler = self._make_handler(gate)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/health_generate"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @staticmethod
    def _make_handler(gate: _ArrivalGate):
        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                gate.wait_for_release()
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format, *args):
                pass

        return _Handler
