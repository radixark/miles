import asyncio
import time
from typing import Any

import pytest
from tests.fast.utils.workers.e2e.harness import wait_until_serving

from miles.utils.workers.rpc.client import call as client_module
from miles.utils.workers.rpc.client import handle as handle_module
from miles.utils.workers.rpc.client.misc import RpcProtocolError
from miles.utils.workers.worker_handle import WorkerUnreachableError


@pytest.fixture
def short_retry_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 0.2)
    monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "RETRY_MAX_DELAY_SECONDS", 0.02)
    monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)


class _StaleWorker:
    def demo_removed_method(self) -> int:
        return 0


class TestSubmitNoRetry:
    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    async def test_server_error_gives_up_after_one_attempt(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        tag: str,
        status_code: int,
    ) -> None:
        """A submit 5xx becomes unreachable after exactly one attempt."""
        proxy = await proxy_to()
        proxy.reject_next(count=1, status=status_code)
        handle = make_handle(proxy)

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1

    async def test_bad_request_is_not_retried(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        tag: str,
    ) -> None:
        """A submit 400 raises a protocol error after one attempt."""
        proxy = await proxy_to()
        proxy.reject_next(count=1, status=400)
        handle = make_handle(proxy)

        with pytest.raises(RpcProtocolError) as exc_info:
            await handle.demo_count_sync(tag=tag)

        assert exc_info.value.status_code == 400
        assert len(proxy.submits("demo_count_sync")) == 1

    async def test_unknown_method_is_not_retried(
        self,
        server: Any,
        make_handle: Any,
        short_retry_window: None,
    ) -> None:
        """An unknown server method raises 404 after one submit."""
        handle = make_handle(server, worker_cls=_StaleWorker)

        with pytest.raises(RpcProtocolError) as exc_info:
            await handle.demo_removed_method()

        assert exc_info.value.status_code == 404

    async def test_post_wire_disconnect_is_not_retried(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        tag: str,
    ) -> None:
        """A dropped submit response becomes unreachable after one attempt."""
        proxy = await proxy_to()
        proxy.drop_next(count=1)
        handle = make_handle(proxy)

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1


class TestNeverReachedRetry:
    async def test_unreachable_server_gives_up_after_retry_window(
        self,
        make_handle: Any,
        short_retry_window: None,
    ) -> None:
        """A refused connection retries only until the submit deadline."""
        handle = make_handle("http://127.0.0.1:9")
        started = time.monotonic()

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_sync(a=1, b=1)

        elapsed = time.monotonic() - started
        assert elapsed >= 0.1
        assert elapsed < 1.0

    async def test_late_server_start_is_tolerated(
        self,
        spawn: Any,
        make_handle: Any,
        monkeypatch: pytest.MonkeyPatch,
        tag: str,
    ) -> None:
        """A submit succeeds when a server appears within its retry window."""
        from tests.fast.utils.workers.e2e.harness import reserve_port

        monkeypatch.setattr(client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 3.0)
        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "RETRY_MAX_DELAY_SECONDS", 0.05)
        port = reserve_port()
        handle = make_handle(f"http://127.0.0.1:{port}")

        async def start_later() -> None:
            await asyncio.sleep(0.1)
            wait_until_serving(spawn(port=port, wait=False))

        starter = asyncio.create_task(start_later())
        result = await handle.demo_count_sync(tag=tag)
        await starter

        assert result == 1


class TestWaitReadyRetries:
    async def test_wait_ready_gives_up(self, make_handle: Any, short_retry_window: None) -> None:
        """wait_ready stops at its own deadline."""
        handle = make_handle("http://127.0.0.1:9")
        started = time.monotonic()

        with pytest.raises(WorkerUnreachableError):
            await handle.wait_ready(timeout=0.2)

        assert time.monotonic() - started < 1.0

    async def test_wait_ready_retries_unhealthy_responses(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """wait_ready retries transient health 503 responses."""
        monkeypatch.setattr(handle_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        proxy = await proxy_to()
        proxy.reject_next(count=2, status=503)
        handle = make_handle(proxy)

        await handle.wait_ready(timeout=1.0)


class TestPinnedSubmitFailure:
    async def test_headerless_gateway_error_does_not_repin(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        tag: str,
    ) -> None:
        """A pinned handle keeps its pin and gives up on a submit 503."""
        proxy = await proxy_to()
        handle = make_handle(proxy, require_stable_boot_uuid=True)
        await handle.wait_ready(timeout=1.0)
        proxy.reject_next(count=1, status=503)

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1
