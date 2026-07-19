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

import multiprocessing
import socket
import threading
import time
from unittest.mock import patch

import httpx
import pytest

from miles.utils.http_utils import _post, wait_for_server_ready


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


# ---------------------------------------------------------------------------
# _post idempotency tests (regression for #1334: double-generate on retry).
#
# _post() retries up to max_retries on failure. For non-idempotent calls such
# as /generate, retrying *after* the request reached the server can re-issue
# generation. The idempotent=False path must therefore retry ONLY pre-send
# connection-setup errors and re-raise everything else immediately, while the
# idempotent=True path keeps retrying any error.
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for an httpx.Response that always succeeds."""

    def raise_for_status(self):
        pass

    def json(self):
        return {"ok": True}


class _FakeClient:
    """Async httpx-like client whose POST raises a configured error N times.

    Counts every POST attempt so tests can assert how many times _post retried.
    """

    def __init__(self, error, fail_times):
        self._error = error
        self._fail_times = fail_times
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._error
        return _FakeResponse()


@pytest.mark.asyncio
async def test_non_idempotent_does_not_retry_post_send_error():
    """A post-send error on a non-idempotent call must NOT retry (no double-generate)."""
    # ReadTimeout is raised after the request was sent (server may be generating).
    client = _FakeClient(httpx.ReadTimeout("server still generating"), fail_times=99)
    with patch("miles.utils.http_utils.asyncio.sleep"):
        with pytest.raises(httpx.ReadTimeout):
            await _post(client, "http://x/generate", {"input_ids": [1]}, max_retries=60, idempotent=False)
    assert client.calls == 1, "non-idempotent /generate must not re-send after a post-send error"


@pytest.mark.asyncio
async def test_non_idempotent_still_retries_connect_error():
    """A pre-send connection error is safe to retry even for non-idempotent calls."""
    # ConnectError happens before any request bytes reach the server.
    client = _FakeClient(httpx.ConnectError("connection refused"), fail_times=3)
    with patch("miles.utils.http_utils.asyncio.sleep"):
        out = await _post(client, "http://x/generate", {"input_ids": [1]}, max_retries=60, idempotent=False)
    assert out == {"ok": True}
    assert client.calls == 4, "should retry the 3 connect failures then succeed"


@pytest.mark.asyncio
async def test_idempotent_retries_post_send_error():
    """Idempotent calls keep the original behaviour: retry any error."""
    client = _FakeClient(httpx.ReadTimeout("transient"), fail_times=2)
    with patch("miles.utils.http_utils.asyncio.sleep"):
        out = await _post(client, "http://x/health", {}, max_retries=60, idempotent=True)
    assert out == {"ok": True}
    assert client.calls == 3, "idempotent path should retry the 2 failures then succeed"


@pytest.mark.asyncio
async def test_programming_error_is_not_retried():
    """Non-HTTP errors (e.g. a bug) must fail fast, not retry max_retries times."""
    client = _FakeClient(ValueError("bug"), fail_times=99)
    with patch("miles.utils.http_utils.asyncio.sleep"):
        with pytest.raises(ValueError):
            await _post(client, "http://x/health", {}, max_retries=60, idempotent=True)
    assert client.calls == 1, "a programming error must not be retried"
