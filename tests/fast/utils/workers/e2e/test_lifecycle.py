import asyncio
import os
import signal
import time
import uuid

import httpx
import pytest
from tests.fast.utils.workers.e2e.harness import (
    READY_TIMEOUT_SECONDS,
    ServerProcess,
    port_is_refused,
    wait_until_serving,
)

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import RpcProtocolError, ServerRestartedError
from miles.utils.workers.worker_handle import WorkerUnreachableError


async def _submit_and_disconnect(server: ServerProcess, method: str, query: dict) -> None:
    async with httpx.AsyncClient(base_url=server.url, timeout=30.0, trust_env=False) as client:
        response = await client.post(f"/v1/{method}", json={"call_id": uuid.uuid4().hex, "query": query})
        assert response.status_code == 200


async def _wait_until_started(handle: RpcWorkerHandle, tag: str) -> None:
    for _ in range(200):
        events = await handle.report_events()
        if any(event.tag == tag and event.phase == "start" for event in events):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"the worker never entered the body of the call tagged {tag}")


class TestRestartDetection:
    async def test_pinned_handle_refuses_a_restarted_server(self, spawn, make_handle, tag):
        """A pinned handle refuses to talk to a different process on the same port."""
        server = spawn()
        pinned = make_handle(server, require_stable_boot_uuid=True)
        assert await pinned.demo_count_sync(tag=tag) == 1

        server.stop()
        spawn(port=server.port)

        with pytest.raises(ServerRestartedError):
            await pinned.demo_count_sync(tag=tag)

    async def test_pinned_handle_does_not_rerun_on_the_new_server(self, spawn, make_handle, tag):
        """The refused call never reaches the restarted worker."""
        server = spawn()
        pinned = make_handle(server, require_stable_boot_uuid=True)
        await pinned.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        server.stop()
        restarted = spawn(port=server.port)

        with pytest.raises(ServerRestartedError):
            await pinned.demo_count_sync(tag=tag)

        fresh = make_handle(restarted)
        assert await fresh.report_counter(tag=tag) == 0

    async def test_unpinned_handle_sees_the_new_process(self, spawn, make_handle, tag):
        """Without pinning, a new call after a restart simply goes to the new process."""
        server = spawn()
        handle = make_handle(server)
        assert await handle.demo_count_sync(tag=tag) == 1

        server.stop()
        spawn(port=server.port)

        assert await handle.demo_count_sync(tag=tag) == 1

    async def test_unpinned_handle_mid_call_restart_reports_unknown_call(self, spawn, make_handle, tag):
        """A call whose server was replaced mid-flight fails loudly instead of hanging."""
        server = spawn()
        handle = make_handle(server, call_timeout_seconds=30.0)

        pending = asyncio.create_task(handle.demo_block_async(tag=tag))
        await asyncio.sleep(0.5)

        server.stop()
        spawn(port=server.port)

        with pytest.raises(RpcProtocolError):
            await pending

    async def test_pinned_handle_created_after_restart_works(self, spawn, make_handle, tag):
        """Pinning after the restart is the supported recovery path."""
        server = spawn()
        server.stop()
        restarted = spawn(port=server.port)

        pinned = make_handle(restarted, require_stable_boot_uuid=True)
        assert await pinned.demo_count_sync(tag=tag) == 1


class TestShutdown:
    async def test_idle_server_exits_promptly_on_sigterm(self, spawn):
        """An idle server shuts down quickly and cleanly."""
        server = spawn()
        started = time.monotonic()
        server.signal(signal.SIGTERM)

        assert server.wait(timeout=15.0) is not None
        assert time.monotonic() - started < 5.0

    async def test_sigterm_abandons_an_in_flight_async_call(self, spawn, state_dir):
        """An accepted call is cancelled rather than allowed to finish."""
        server = spawn()
        await _submit_and_disconnect(server, "demo_marker_after_sleep_async", {"name": "abandoned", "seconds": 2.0})

        server.signal(signal.SIGTERM)
        assert server.wait(timeout=30.0) is not None
        await asyncio.sleep(2.5)
        assert not (state_dir / "abandoned").exists()

    @pytest.mark.parametrize("stop", ["sigterm", "sigkill"])
    async def test_the_worker_gets_no_cancellation_callback(self, spawn, state_dir, tag, stop: str):
        """There is no graceful shutdown, so an in-flight worker never observes cancellation."""
        server = spawn()
        await _submit_and_disconnect(server, "demo_block_until_cancelled", {"tag": tag})

        if stop == "sigterm":
            server.signal(signal.SIGTERM)
        else:
            server.kill()
        assert server.wait(timeout=30.0) is not None
        assert not (state_dir / f"cancelled_{tag}").exists()

    async def test_sigterm_after_the_worker_started_does_not_run_its_cancellation_callback(
        self, spawn, make_handle, state_dir, tag
    ):
        """Even once the worker body is provably running, SIGTERM hands it no cancellation callback."""
        server = spawn()
        await _submit_and_disconnect(server, "demo_block_until_cancelled", {"tag": tag})
        await _wait_until_started(make_handle(server), tag)

        server.signal(signal.SIGTERM)
        assert server.wait(timeout=30.0) is not None

        await asyncio.sleep(0.5)
        assert not (state_dir / f"cancelled_{tag}").exists()

    async def test_port_is_released_for_the_next_process(self, spawn):
        """A stopped server frees its port for an immediate successor."""
        server = spawn()
        server.stop()
        assert port_is_refused(server.port)

        successor = spawn(port=server.port)
        wait_until_serving(successor)

    async def test_no_process_survives_teardown(self, spawn):
        """Stopping the server really reaps the process."""
        server = spawn()
        pid = server.process.pid
        server.stop()

        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


class TestClientAgainstDeadServer:
    async def test_call_submitted_then_server_killed_times_out(
        self,
        spawn,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
        tag,
    ) -> None:
        """Poll failures after server death exhaust as TimeoutError."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)
        server = spawn()
        handle = make_handle(server, call_timeout_seconds=0.3)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        pending = asyncio.create_task(handle.demo_block_async(tag=tag))
        await asyncio.sleep(0.05)
        server.kill()

        with pytest.raises(TimeoutError):
            await pending

    async def test_call_to_a_dead_port_raises_plain_unreachable(
        self,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Never reaching the server raises plain WorkerUnreachableError."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 0.2)
        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "RETRY_MAX_DELAY_SECONDS", 0.02)

        handle = make_handle("http://127.0.0.1:9")
        with pytest.raises(WorkerUnreachableError):
            await handle.demo_sync(a=1, b=1)
