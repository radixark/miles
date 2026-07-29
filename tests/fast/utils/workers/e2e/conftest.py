from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import httpx
import pytest
from tests.fast.utils.workers.e2e.e2e_worker import E2eWorker
from tests.fast.utils.workers.e2e.harness import (
    READY_TIMEOUT_SECONDS,
    FlakyProxy,
    ServerProcess,
    spawn_server,
    wait_until_serving,
)

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle


@pytest.fixture
def state_dir(tmp_path) -> Path:
    path = tmp_path / "state"
    path.mkdir()
    return path


@pytest.fixture
def tag() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture
def spawn(state_dir, tmp_path, request) -> Iterator[Callable[..., ServerProcess]]:
    started: list[ServerProcess] = []

    def start(*, wait: bool = True, **kwargs) -> ServerProcess:
        log_path = tmp_path / f"server-{len(started)}.log"
        server = spawn_server(state_dir=state_dir, log_path=log_path, **kwargs)
        started.append(server)
        if wait:
            wait_until_serving(server)
        return server

    yield start

    for server in started:
        server.stop()
        server.kill()
        if request.node.rep_call is not None and request.node.rep_call.failed:
            print(f"\n--- server log {server.log_path.name} ---\n{server.logs()[-4000:]}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    setattr(item, f"rep_{call.when}", outcome.get_result())


@pytest.fixture(autouse=True)
def _reset_reports(request):
    for phase in ("setup", "call", "teardown"):
        if not hasattr(request.node, f"rep_{phase}"):
            setattr(request.node, f"rep_{phase}", None)


@pytest.fixture(scope="session")
def shared_server(tmp_path_factory) -> Iterator[ServerProcess]:
    directory = tmp_path_factory.mktemp("shared-server")
    state_dir = directory / "state"
    state_dir.mkdir()

    server = spawn_server(state_dir=state_dir, log_path=directory / "server.log")
    wait_until_serving(server)
    yield server

    server.stop()
    server.kill()


@pytest.fixture
async def server(shared_server, request) -> AsyncIterator[ServerProcess]:
    assert shared_server.is_running(), (
        f"the shared server died in an earlier test; a test that stops it must spawn its own:\n"
        f"{shared_server.logs()[-4000:]}"
    )

    yield shared_server

    async with httpx.AsyncClient(base_url=shared_server.url, timeout=30.0, trust_env=False) as client:
        with contextlib.suppress(Exception):
            await RpcWorkerHandle(E2eWorker, server_url=shared_server.url, http_client=client).release_every_gate()

    if request.node.rep_call is not None and request.node.rep_call.failed:
        print(f"\n--- shared server log tail ---\n{shared_server.logs()[-4000:]}")


@pytest.fixture
async def make_handle() -> AsyncIterator[Callable[..., RpcWorkerHandle]]:
    clients: list[httpx.AsyncClient] = []

    def build(target: ServerProcess | FlakyProxy | str, *, worker_cls: type = E2eWorker, **kwargs) -> RpcWorkerHandle:
        url = target if isinstance(target, str) else target.url
        client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), trust_env=False)
        clients.append(client)
        return RpcWorkerHandle(worker_cls, server_url=url, http_client=client, **kwargs)

    yield build

    for client in clients:
        await client.aclose()


@pytest.fixture
async def handle(server, make_handle) -> AsyncIterator[RpcWorkerHandle]:
    worker_handle = make_handle(server)
    await worker_handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield worker_handle


@pytest.fixture
async def raw(server) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=server.url, timeout=30.0, trust_env=False) as client:
        yield client


@pytest.fixture
async def proxy_to(server) -> AsyncIterator[Callable[[], FlakyProxy]]:
    proxies: list[FlakyProxy] = []

    async def build() -> FlakyProxy:
        proxy = FlakyProxy(upstream_port=server.port)
        await proxy.start()
        proxies.append(proxy)
        return proxy

    yield build

    for proxy in proxies:
        await proxy.stop()


@pytest.fixture
async def dead_proxy() -> AsyncIterator[FlakyProxy]:
    proxy = FlakyProxy(upstream_port=None)
    proxy.record_only = True
    await proxy.start()
    yield proxy
    await proxy.stop()
