import asyncio
from collections.abc import Callable
from urllib.parse import urlencode

import httpx
import pytest
import requests
from fastapi import Request

from miles.router import router as router_module
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, default_process_fn
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


def make_router_config(router_port: int, **overrides) -> MilesRouterConfig:
    defaults = dict(
        host="127.0.0.1",
        port=router_port,
        health_check_interval=1.0,
        health_check_failure_threshold=3,
        max_connections=100,
        timeout=None,
    )
    defaults.update(overrides)
    return MilesRouterConfig(**defaults)


def create_mock_worker(start_port: int = 30000) -> MockSGLangServer:
    port = find_available_port(start_port)
    return MockSGLangServer(
        model_name="Qwen/Qwen3-0.6B",
        process_fn=default_process_fn,
        host="127.0.0.1",
        port=port,
        latency=0.0,
    )


def make_add_worker_request(worker_url: str) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/add_worker",
        "query_string": urlencode({"url": worker_url}).encode(),
        "headers": [],
    }
    return Request(scope)


class RouterEnv:
    def __init__(self, router: MilesRouter, server: UvicornThreadServer):
        self.router = router
        self.server = server

    @property
    def url(self) -> str:
        return self.server.url


@pytest.fixture
def router_env():
    config = make_router_config(find_available_port(20000))
    router = MilesRouter(config, verbose=False)
    server = UvicornThreadServer(router.app, host=config.host, port=config.port)
    server.start()
    yield RouterEnv(router, server)
    server.stop()


@pytest.fixture
def mock_worker():
    server = create_mock_worker()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def mock_worker_factory():
    servers = []

    def _create():
        start_port = 30000 + len(servers) * 100
        server = create_mock_worker(start_port)
        server.start()
        servers.append(server)
        return server

    yield _create
    for s in servers:
        s.stop()


@pytest.fixture
def router_factory():
    def _create(**overrides) -> MilesRouter:
        config = make_router_config(find_available_port(20000), **overrides)
        return MilesRouter(config, verbose=False)

    return _create


class TestMilesRouterInitialization:
    async def test_client_uses_configured_connection_limit_and_timeout(self):
        """The router builds its HTTP client with the connection limit and timeout taken from the config."""
        config = make_router_config(20000, max_connections=7, timeout=3.5)

        router = MilesRouter(config, verbose=False)

        try:
            assert isinstance(router.client, httpx.AsyncClient)
            assert router.client._transport._pool._max_connections == 7
            assert router.client.timeout.connect == 3.5
            assert router.client.timeout.read == 3.5
            assert router.client.timeout.write == 3.5
            assert router.client.timeout.pool == 3.5
        finally:
            await router.client.aclose()


class TestWorkerManagement:
    def test_add_worker_via_query_param(self, router_env: RouterEnv):
        worker_url = "http://127.0.0.1:30001"
        r = requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0)
        r.raise_for_status()

        assert r.json()["status"] == "success"
        assert worker_url in router_env.router.worker_request_counts
        assert router_env.router.worker_request_counts[worker_url] == 0

    def test_add_worker_via_body(self, router_env: RouterEnv):
        worker_url = "http://127.0.0.1:30002"
        r = requests.post(f"{router_env.url}/add_worker", json={"url": worker_url}, timeout=5.0)
        r.raise_for_status()

        assert r.json()["status"] == "success"
        assert worker_url in router_env.router.worker_request_counts

    def test_add_worker_duplicate(self, router_env: RouterEnv):
        worker_url = "http://127.0.0.1:30003"
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()

        assert len(router_env.router.worker_request_counts) == 1
        assert worker_url in router_env.router.worker_request_counts

    def test_add_worker_missing_url(self, router_env: RouterEnv):
        r = requests.post(f"{router_env.url}/add_worker", json={}, timeout=5.0)
        assert r.status_code == 400
        assert "error" in r.json()

    def test_list_workers(self, router_env: RouterEnv):
        worker_urls = ["http://127.0.0.1:30001", "http://127.0.0.1:30002"]
        for url in worker_urls:
            requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0)

        r = requests.get(f"{router_env.url}/list_workers", timeout=5.0)
        r.raise_for_status()
        assert set(r.json()["urls"]) == set(worker_urls)

    def test_remove_worker_via_query_param(self, router_env: RouterEnv):
        """A disposed cell must be able to deregister, or its dead url is routed to forever."""
        worker_url = "http://127.0.0.1:30011"
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()

        r = requests.post(f"{router_env.url}/remove_worker", params={"url": worker_url}, timeout=5.0)
        r.raise_for_status()

        assert r.json()["status"] == "success"
        assert worker_url not in router_env.router.worker_request_counts

    def test_remove_worker_via_body(self, router_env: RouterEnv):
        """Removal accepts the same body form as add_worker so callers need no special case."""
        worker_url = "http://127.0.0.1:30012"
        requests.post(f"{router_env.url}/add_worker", json={"url": worker_url}, timeout=5.0).raise_for_status()

        requests.post(f"{router_env.url}/remove_worker", json={"url": worker_url}, timeout=5.0).raise_for_status()

        assert worker_url not in router_env.router.worker_request_counts

    def test_remove_worker_clears_the_failure_and_dead_bookkeeping(self, router_env: RouterEnv):
        """Leaving a removed url quarantined would keep it in the containers this fix drains."""
        worker_url = "http://127.0.0.1:30013"
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()
        router_env.router.worker_failure_counts[worker_url] = 3
        router_env.router.dead_workers.add(worker_url)

        requests.post(f"{router_env.url}/remove_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()

        assert worker_url not in router_env.router.worker_failure_counts
        assert worker_url not in router_env.router.dead_workers

    def test_remove_worker_is_idempotent(self, router_env: RouterEnv):
        """Teardown can race a health-check eviction, and a second removal must not error."""
        worker_url = "http://127.0.0.1:30014"

        r = requests.post(f"{router_env.url}/remove_worker", params={"url": worker_url}, timeout=5.0)

        assert r.status_code == 200

    def test_remove_worker_missing_url(self, router_env: RouterEnv):
        """Without a url there is nothing to remove, and silently succeeding would hide the caller bug."""
        r = requests.post(f"{router_env.url}/remove_worker", json={}, timeout=5.0)

        assert r.status_code == 400
        assert "error" in r.json()

    def test_a_removed_worker_disappears_from_list_workers(self, router_env: RouterEnv):
        """list_workers is what the rollout side aborts against, so dead urls must leave it."""
        kept, removed = "http://127.0.0.1:30015", "http://127.0.0.1:30016"
        for url in (kept, removed):
            requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0).raise_for_status()

        requests.post(f"{router_env.url}/remove_worker", params={"url": removed}, timeout=5.0).raise_for_status()

        r = requests.get(f"{router_env.url}/list_workers", timeout=5.0)
        r.raise_for_status()
        assert r.json()["urls"] == [kept]


class TestLoadBalancing:
    def test_use_url_selects_min_load(self, router_factory):
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 5, "http://w2:8000": 2, "http://w3:8000": 8}

        selected = router._use_url()
        assert selected == "http://w2:8000"
        assert router.worker_request_counts["http://w2:8000"] == 3

    def test_use_url_excludes_dead_workers(self, router_factory):
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 5, "http://w2:8000": 1, "http://w3:8000": 3}
        router.dead_workers = {"http://w2:8000"}

        selected = router._use_url()
        assert selected == "http://w3:8000"
        assert router.worker_request_counts["http://w3:8000"] == 4

    def test_use_url_raises_when_all_dead(self, router_factory):
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 0}
        router.dead_workers = {"http://w1:8000"}

        with pytest.raises(RuntimeError, match="No healthy workers"):
            router._use_url()


class TestFinishingARequestAfterDeregistration:
    def test_a_finished_request_decrements_its_workers_count(self, router_factory):
        """The ordinary path still hands the slot back to a registered worker."""
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 2}

        router._finish_url("http://w1:8000")

        assert router.worker_request_counts == {"http://w1:8000": 1}

    def test_a_request_that_outlives_its_workers_registration_finishes_quietly(self, router_factory):
        """Deregistering a worker mid-request must not turn the already proxied response into a 500."""
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 0, "http://w2:8000": 0}
        worker_url = router._use_url()
        router.worker_request_counts.pop(worker_url)

        router._finish_url(worker_url)

        assert worker_url not in router.worker_request_counts

    def test_a_stale_finish_does_not_charge_a_replacement_that_took_the_same_url(self, router_factory):
        """A request from the previous registration must not drive the re-registered worker's count negative."""
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 0}
        worker_url = router._use_url()
        router.worker_request_counts.pop(worker_url)
        router.worker_request_counts[worker_url] = 0

        router._finish_url(worker_url)

        assert router.worker_request_counts[worker_url] == 0
        assert router._use_url() == worker_url


class FakeSleepClock:
    def __init__(self, *, stop_after: int, on_sleep: Callable[[], None] | None = None):
        self.sleeps: list[float] = []
        self._stop_after = stop_after
        self._on_sleep = on_sleep

    async def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        if self._on_sleep is not None:
            self._on_sleep()
        if len(self.sleeps) >= self._stop_after:
            raise asyncio.CancelledError


class ScriptedWorkerHealth:
    def __init__(self, results: list[bool]):
        self.checked_urls: list[str] = []
        self._results = list(results)

    async def check(self, url: str) -> tuple[str, bool]:
        self.checked_urls.append(url)
        return url, self._results.pop(0)


class DeregisteringWorkerHealth:
    def __init__(self, router: MilesRouter, *, results: dict[str, bool], deregister: set[str]):
        self.checked_urls: list[str] = []
        self._router = router
        self._results = results
        self._deregister = deregister

    async def check(self, url: str) -> tuple[str, bool]:
        self.checked_urls.append(url)
        if url in self._deregister:
            self._router.worker_request_counts.pop(url, None)
            self._router.worker_failure_counts.pop(url, None)
            self._router.dead_workers.discard(url)
        return url, self._results[url]


# TODO: extract main body inside `_health_check_loop`, then can test that function
class TestHealthCheck:
    def test_check_worker_health_success(self, router_factory, mock_worker: MockSGLangServer):
        router = router_factory()
        url, healthy = asyncio.run(router._check_worker_health(mock_worker.url))
        assert url == mock_worker.url
        assert healthy is True

    def test_check_worker_health_failure(self, router_factory):
        router = router_factory()
        url, healthy = asyncio.run(router._check_worker_health("http://127.0.0.1:59999"))
        assert url == "http://127.0.0.1:59999"
        assert healthy is False

    def test_health_check_loop_waits_for_configured_interval(self, router_factory, monkeypatch: pytest.MonkeyPatch):
        """The background health check loop waits the configured interval before every round."""
        router = router_factory(health_check_interval=7.5)
        router.worker_request_counts = {"http://w1:8000": 0}
        router._check_worker_health = ScriptedWorkerHealth([True, True, True]).check
        clock = FakeSleepClock(stop_after=3)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert clock.sleeps == [7.5, 7.5, 7.5]

    def test_only_configured_consecutive_failures_quarantine_worker(
        self, router_factory, monkeypatch: pytest.MonkeyPatch
    ):
        """A worker is quarantined only once it fails the configured number of checks in a row, and a success resets the count."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=4)
        router.worker_request_counts = {worker_url: 0}
        router._check_worker_health = ScriptedWorkerHealth(
            [False, False, False, True, False, False, False, False]
        ).check
        dead_worker_snapshots: list[set[str]] = []
        clock = FakeSleepClock(stop_after=9, on_sleep=lambda: dead_worker_snapshots.append(set(router.dead_workers)))
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert dead_worker_snapshots == [set(), set(), set(), set(), set(), set(), set(), set(), {worker_url}]
        assert router.worker_failure_counts[worker_url] == 4


class TestStaleHealthResults:
    def test_a_probe_result_for_a_worker_removed_meanwhile_is_dropped(
        self, router_factory, monkeypatch: pytest.MonkeyPatch
    ):
        """A failure observed on a deregistered url would quarantine the replacement that later takes it."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {worker_url: 0}
        router.worker_failure_counts = {worker_url: 0}

        async def _check_and_remove(url: str) -> tuple[str, bool]:
            router.worker_request_counts.pop(url, None)
            router.worker_failure_counts.pop(url, None)
            return url, False

        router._check_worker_health = _check_and_remove
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert router.dead_workers == set()
        assert worker_url not in router.worker_failure_counts

    def test_a_probe_result_for_the_current_registration_still_counts(
        self, router_factory, monkeypatch: pytest.MonkeyPatch
    ):
        """Dropping every result would leave a genuinely dead worker in the routing pool forever."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {worker_url: 0}
        router._check_worker_health = ScriptedWorkerHealth([False]).check
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert router.dead_workers == {worker_url}

    def test_readding_a_quarantined_url_puts_it_back_in_the_pool(self, router_env: RouterEnv):
        """A stale probe can quarantine a url after its removal, and the replacement engine must not inherit that."""
        worker_url = "http://127.0.0.1:30017"
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()
        requests.post(f"{router_env.url}/remove_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()
        router_env.router.dead_workers.add(worker_url)

        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()

        assert worker_url not in router_env.router.dead_workers

    def test_a_healthy_probe_result_for_a_removed_worker_does_not_revive_its_bookkeeping(
        self, router_factory: Callable[..., MilesRouter], monkeypatch: pytest.MonkeyPatch
    ):
        """A success observed on a deregistered url must not put the url back into the failure bookkeeping either."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {worker_url: 0}
        router.worker_failure_counts = {worker_url: 0}
        router._check_worker_health = DeregisteringWorkerHealth(
            router, results={worker_url: True}, deregister={worker_url}
        ).check
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert router.worker_request_counts == {}
        assert router.worker_failure_counts == {}

    def test_a_stale_result_does_not_abandon_the_rest_of_the_round(
        self, router_factory: Callable[..., MilesRouter], monkeypatch: pytest.MonkeyPatch
    ):
        """One deregistered url must only skip its own result, while the other workers of the same round are still judged."""
        removed_url, kept_url = "http://w1:8000", "http://w2:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {removed_url: 0, kept_url: 0}
        router.worker_failure_counts = {removed_url: 0, kept_url: 0}
        router._check_worker_health = DeregisteringWorkerHealth(
            router, results={removed_url: False, kept_url: False}, deregister={removed_url}
        ).check
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert router.dead_workers == {kept_url}
        assert removed_url not in router.worker_failure_counts

    def test_a_url_freed_during_a_probe_is_routable_again_once_its_replacement_registers(
        self, router_factory: Callable[..., MilesRouter], monkeypatch: pytest.MonkeyPatch
    ):
        """A dispose racing a probe must not stop the engine that later takes the same address from receiving traffic."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {worker_url: 0}
        router.worker_failure_counts = {worker_url: 0}
        router._check_worker_health = DeregisteringWorkerHealth(
            router, results={worker_url: False}, deregister={worker_url}
        ).check
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())
        asyncio.run(router.add_worker(make_add_worker_request(worker_url)))

        assert router._use_url() == worker_url

    def test_a_duplicate_registration_does_not_lift_a_live_workers_quarantine(
        self, router_factory: Callable[..., MilesRouter]
    ):
        """Only a fresh registration clears quarantine, because reconnecting a still-registered dead worker needs a weight resync."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_failure_threshold=3)
        router.worker_request_counts = {worker_url: 2}
        router.worker_failure_counts = {worker_url: 3}
        router.dead_workers = {worker_url}

        asyncio.run(router.add_worker(make_add_worker_request(worker_url)))

        assert router.dead_workers == {worker_url}
        assert router.worker_failure_counts[worker_url] == 3
        assert router.worker_request_counts[worker_url] == 2


class TestProxyIntegration:
    def test_proxy_forwards_request(self, router_env: RouterEnv, mock_worker: MockSGLangServer):
        requests.post(f"{router_env.url}/add_worker", params={"url": mock_worker.url}, timeout=5.0).raise_for_status()

        payload = {"input_ids": [1, 2, 3], "return_logprob": True}
        r = requests.post(f"{router_env.url}/generate", json=payload, timeout=10.0)
        r.raise_for_status()

        assert "text" in r.json()
        assert len(mock_worker.request_log) == 1
        assert mock_worker.request_log[0] == payload

    def test_proxy_multi_worker(self, router_env: RouterEnv, mock_worker_factory):
        worker1, worker2 = mock_worker_factory(), mock_worker_factory()
        requests.post(f"{router_env.url}/add_worker", params={"url": worker1.url}, timeout=5.0)
        requests.post(f"{router_env.url}/add_worker", params={"url": worker2.url}, timeout=5.0)

        payload = {"input_ids": [1, 2, 3], "return_logprob": True}
        for _ in range(4):
            requests.post(f"{router_env.url}/generate", json=payload, timeout=10.0).raise_for_status()

        all_requests = worker1.request_log + worker2.request_log
        assert len(all_requests) == 4
        assert all(req == payload for req in all_requests)

    def test_proxy_health_endpoint(self, router_env: RouterEnv, mock_worker: MockSGLangServer):
        requests.post(f"{router_env.url}/add_worker", params={"url": mock_worker.url}, timeout=5.0)

        r = requests.get(f"{router_env.url}/health", timeout=5.0)
        r.raise_for_status()
        assert r.json()["status"] == "ok"
