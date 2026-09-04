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
import inspect
import multiprocessing
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import patch

import httpx
import pytest
import ray
from tests.fast.utils.fake_ray_ids import fake_ray_node_id

from miles.utils import http_utils
from miles.utils.http_utils import (
    GeneralHttpClientProvider,
    wait_for_server_ready,
    wait_tcp_ready,
    wait_tcp_ready_async,
)


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


class TestWaitTcpReady:
    def test_keeps_retrying_until_the_port_accepts(self):
        """Readiness depends on the endpoint alone, retrying while it refuses connections."""
        attempts: list[tuple[tuple[str, int], float | None]] = []
        sleeps: list[float] = []
        fake_time = [0.0]

        def fake_sleep(duration):
            sleeps.append(duration)
            fake_time[0] += duration

        def fake_connect(addr, timeout=None):
            attempts.append((addr, timeout))
            if len(attempts) < 3:
                raise OSError("Connection refused")
            return _FakeSocket()

        with (
            patch("miles.utils.http_utils.time.time", side_effect=lambda: fake_time[0]),
            patch("miles.utils.http_utils.time.sleep", side_effect=fake_sleep),
            patch("miles.utils.http_utils.socket.create_connection", side_effect=fake_connect),
        ):
            wait_tcp_ready("[2001:db8::7]", 23456, timeout=30)

        assert attempts == [(("2001:db8::7", 23456), 1)] * 3
        assert sleeps == [0.5, 0.5]

    def test_gives_up_when_the_deadline_passes(self):
        """A port that never opens fails with a timeout instead of blocking forever."""
        fake_time = [0.0]

        def fake_sleep(duration):
            fake_time[0] += duration

        def fake_connect(addr, timeout=None):
            raise OSError("Connection refused")

        with (
            patch("miles.utils.http_utils.time.time", side_effect=lambda: fake_time[0]),
            patch("miles.utils.http_utils.time.sleep", side_effect=fake_sleep),
            patch("miles.utils.http_utils.socket.create_connection", side_effect=fake_connect),
        ):
            with pytest.raises(RuntimeError, match="Server at 127.0.0.1:23456 not ready after 1s"):
                wait_tcp_ready("127.0.0.1", 23456, timeout=1)

        assert fake_time[0] >= 1


class TestWaitTcpReadyAsync:
    async def test_it_returns_once_the_port_accepts(self):
        """The async probe must still answer the question the blocking one answered."""
        server = await asyncio.start_server(lambda reader, writer: None, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        try:
            await asyncio.wait_for(wait_tcp_ready_async("127.0.0.1", port, timeout=5), timeout=5)
        finally:
            server.close()
            await server.wait_closed()

    async def test_a_closed_port_leaves_the_event_loop_free(self):
        """The blocking probe froze the whole startup loop for up to two minutes per router."""
        ticks = 0

        async def _tick() -> None:
            nonlocal ticks
            for _ in range(5):
                await asyncio.sleep(0.02)
                ticks += 1

        ticker = asyncio.create_task(_tick())
        with pytest.raises(RuntimeError, match="not ready after"):
            await wait_tcp_ready_async("127.0.0.1", _find_free_port(), timeout=1.2)
        await asyncio.wait_for(ticker, timeout=1)

        assert ticks == 5

    async def test_it_gives_up_when_the_deadline_passes(self):
        """A port that never opens must fail the caller rather than be awaited forever."""
        port = _find_free_port()

        with pytest.raises(RuntimeError, match=f"Server at 127.0.0.1:{port} not ready after 0.2s"):
            await wait_tcp_ready_async("127.0.0.1", port, timeout=0.2)

    async def test_a_bracketed_ipv6_host_is_unwrapped_before_connecting(self):
        """Addresses come in wrapped for urls, and the socket layer rejects the brackets."""
        connected: list[str] = []

        async def _fake_open_connection(host: str, port: int):
            connected.append(host)
            raise ConnectionRefusedError

        with patch("miles.utils.http_utils.asyncio.open_connection", side_effect=_fake_open_connection):
            with pytest.raises(RuntimeError):
                await wait_tcp_ready_async("[::1]", 23456, timeout=0.01)

        assert connected == ["::1"]


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

    async def test_the_client_has_no_read_timeout_but_finite_connect_write_and_pool_deadlines(self):
        """A weight-update request blocks until the collective forms, but a stuck peer, a full send
        buffer or an exhausted pool must never block a control-path caller forever."""
        timeout = GeneralHttpClientProvider.client().timeout

        assert timeout.read is None
        assert timeout.connect == 10.0
        assert timeout.write == 60.0
        assert timeout.pool == 60.0

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


class TestGeneralHttpClientTimeoutPolicy:
    """Only the read deadline of the shared client may be unbounded."""

    async def test_read_is_the_only_deadline_left_unbounded(self):
        """A peer that stops draining the socket must not be able to block a caller forever."""
        timeout = GeneralHttpClientProvider.client().timeout

        assert [name for name, seconds in timeout.as_dict().items() if seconds is None] == ["read"]

    async def test_the_bounded_deadlines_are_positive_and_small_enough_to_fire(self):
        """A zero deadline fails every request and an astronomical one is no deadline at all."""
        deadlines = GeneralHttpClientProvider.client().timeout.as_dict()

        for name in ("connect", "write", "pool"):
            assert deadlines[name] is not None and 0 < deadlines[name] <= 300, name

    async def test_the_bounded_deadlines_track_the_class_constants(self):
        """Retuning a timeout constant must move the client's deadline instead of silently desyncing."""
        timeout = GeneralHttpClientProvider.client().timeout

        assert timeout.connect == GeneralHttpClientProvider._CONNECT_TIMEOUT
        assert timeout.write == GeneralHttpClientProvider._WRITE_TIMEOUT
        assert timeout.pool == GeneralHttpClientProvider._POOL_TIMEOUT

    async def test_every_request_carries_the_bounded_deadlines_down_to_the_transport(self):
        """Deadlines bound a stuck peer only if httpx hands them to the transport on each request."""
        recorded = _RecordingTransport()
        timeout = GeneralHttpClientProvider.client().timeout

        async with httpx.AsyncClient(timeout=timeout, transport=recorded) as client:
            await client.post("http://weight-update.invalid/update_weights", json={})

        assert recorded.deadlines == timeout.as_dict()
        assert recorded.deadlines["write"] is not None
        assert recorded.deadlines["pool"] is not None

    async def test_a_client_built_on_another_loop_keeps_the_same_deadlines(self):
        """Every event loop gets a fresh client, and each one must carry the same timeout policy."""
        mine = GeneralHttpClientProvider.client().timeout.as_dict()
        theirs: list[dict[str, float | None]] = []

        async def _on_the_other_loop() -> dict[str, float | None]:
            return GeneralHttpClientProvider.client().timeout.as_dict()

        thread = threading.Thread(target=lambda: theirs.append(asyncio.run(_on_the_other_loop())))
        thread.start()
        thread.join(timeout=10)

        assert theirs == [mine]


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


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.deadlines: dict[str, float | None] | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.deadlines = request.extensions["timeout"]
        return httpx.Response(200)


class TestDistributedPostActors:
    def test_the_poster_actor_is_constructed_with_keyword_arguments(self, monkeypatch):
        """A positional handoff silently binds to the wrong parameter once the actor grows another one."""
        recorded: list[tuple[tuple, dict]] = []

        class _FakeActorClass:
            def options(self, **_options):
                return self

            def remote(self, *call_args, **call_kwargs):
                recorded.append((call_args, call_kwargs))
                return object()

        monkeypatch.setattr(ray, "nodes", lambda: [{"NodeID": fake_ray_node_id(0), "Alive": True}])
        monkeypatch.setattr(ray, "remote", lambda _cls: _FakeActorClass())
        monkeypatch.setattr(http_utils, "_post_actors", [])
        monkeypatch.setattr(http_utils, "_client_concurrency", 7)

        http_utils._init_ray_distributed_post(SimpleNamespace(num_gpus_per_node=2))

        assert recorded == [((), {"concurrency": 8})] * 2


class _PosterActorInit(NamedTuple):
    actor_class: type
    calls: list[tuple[tuple, dict]]


class _RecordingRemoteActorClass:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def options(self, **_options: Any) -> "_RecordingRemoteActorClass":
        return self

    def remote(self, *call_args: Any, **call_kwargs: Any) -> object:
        self.calls.append((call_args, call_kwargs))
        return object()


def _run_init_ray_distributed_post(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_gpus_per_node: int = 1,
    client_concurrency: int = 7,
    nodes: list[dict] | None = None,
) -> _PosterActorInit:
    captured: dict[str, type] = {}
    remote_actor_class = _RecordingRemoteActorClass()

    def _fake_remote(cls: type) -> _RecordingRemoteActorClass:
        captured["actor_class"] = cls
        return remote_actor_class

    monkeypatch.setattr(ray, "nodes", lambda: nodes or [{"NodeID": fake_ray_node_id(0), "Alive": True}])
    monkeypatch.setattr(ray, "remote", _fake_remote)
    monkeypatch.setattr(http_utils, "_post_actors", [])
    monkeypatch.setattr(http_utils, "_client_concurrency", client_concurrency)

    http_utils._init_ray_distributed_post(SimpleNamespace(num_gpus_per_node=num_gpus_per_node))

    return _PosterActorInit(actor_class=captured["actor_class"], calls=remote_actor_class.calls)


class TestPosterActorKeywordOnlyConstruction:
    def test_the_poster_actor_refuses_a_positional_concurrency(self, monkeypatch):
        """Constructing the poster actor positionally must fail so a later parameter cannot silently steal the slot."""
        actor_class = _run_init_ray_distributed_post(monkeypatch).actor_class

        with pytest.raises(TypeError):
            actor_class(7)

    def test_every_poster_actor_constructor_parameter_is_keyword_only(self, monkeypatch):
        """No poster actor constructor parameter may be positionally bindable."""
        actor_class = _run_init_ray_distributed_post(monkeypatch).actor_class

        parameters = list(inspect.signature(actor_class.__init__).parameters.values())[1:]

        assert [parameter.kind for parameter in parameters] == [inspect.Parameter.KEYWORD_ONLY] * len(parameters)
        assert parameters

    def test_the_recorded_poster_keywords_bind_to_the_actor_constructor(self, monkeypatch):
        """The keywords the call site sends must name real poster actor constructor parameters."""
        init = _run_init_ray_distributed_post(monkeypatch)
        signature = inspect.signature(init.actor_class.__init__)

        for call_args, call_kwargs in init.calls:
            assert call_args == ()
            signature.bind(object(), *call_args, **call_kwargs)

    def test_the_poster_actor_is_constructed_by_keyword_on_every_node_slot(self, monkeypatch):
        """Every actor created across nodes and per-node slots receives its concurrency by keyword."""
        init = _run_init_ray_distributed_post(
            monkeypatch,
            num_gpus_per_node=2,
            client_concurrency=10,
            nodes=[{"NodeID": fake_ray_node_id(0), "Alive": True}, {"NodeID": fake_ray_node_id(1), "Alive": True}],
        )

        assert init.calls == [((), {"concurrency": 6})] * 4
