import asyncio

import pytest
from pydantic import ValidationError

from miles.utils.workers.rpc.client.misc import RpcWorkerCallError
from miles.utils.workers.worker_handle import WorkerUnreachableError


async def _submit(raw, method: str, call_id: str, query: dict):
    return await raw.post(f"/v1/{method}", json={"call_id": call_id, "query": query})


async def _poll_until_done(raw, call_id: str):
    for _ in range(50):
        body = (await raw.get(f"/v1/calls/{call_id}", params={"timeout": 2.0})).json()
        if body["status"] != "pending":
            return body
    raise AssertionError("call never finished")


class TestWorkerExceptions:
    async def test_sync_exception_raises_call_error(self, handle):
        """A sync worker exception surfaces as a call error on the client."""
        with pytest.raises(RpcWorkerCallError) as exc_info:
            await handle.demo_sync_raises(message="sync-boom")
        assert "demo_sync_raises" in str(exc_info.value)

    async def test_async_exception_raises_call_error(self, handle):
        """An async worker exception surfaces the same way."""
        with pytest.raises(RpcWorkerCallError):
            await handle.demo_async_raises(message="async-boom")

    async def test_error_carries_the_remote_traceback(self, handle):
        """The client sees the worker's traceback, not just a repr."""
        with pytest.raises(RpcWorkerCallError) as exc_info:
            await handle.demo_sync_raises(message="needle")

        text = str(exc_info.value)
        assert "ValueError: needle" in text
        assert "Traceback (most recent call last)" in text
        assert "e2e_worker.py" in text

    async def test_worker_failure_is_http_200(self, raw, tag):
        """A failing worker call is a successful HTTP exchange with a failed envelope."""
        response = await _submit(raw, "demo_sync_raises", tag, {"message": "x"})
        assert response.status_code == 200

        body = await _poll_until_done(raw, tag)
        assert body["status"] == "failed" and body["result"] is None
        assert "ValueError" in body["error"]

    async def test_failure_does_not_poison_the_server(self, handle):
        """The server keeps serving after a worker raises."""
        with pytest.raises(RpcWorkerCallError):
            await handle.demo_sync_raises(message="x")
        assert await handle.demo_sync(a=1, b=2) == 3

    async def test_failure_does_not_poison_its_executor(self, handle, tag):
        """The executor thread of a failed sync call keeps working."""
        with pytest.raises(RpcWorkerCallError):
            await handle.demo_sync_raises(message="x")
        assert (await handle.demo_instant_sync(tag=tag)).startswith("rpc-")


class TestResultContract:
    async def test_unserializable_result_becomes_a_failure(self, handle):
        """A result the server cannot serialize fails the call instead of stranding it."""
        with pytest.raises(RpcWorkerCallError):
            await handle.demo_unserializable_result()

    async def test_result_type_mismatch_is_caught_client_side(self, handle):
        """A result that does not match the annotation fails validation on the client."""
        with pytest.raises(ValidationError):
            await handle.demo_wrong_result_type()

    async def test_server_survives_a_mismatched_result(self, handle):
        """A validation failure on the client leaves the server usable."""
        with pytest.raises(ValidationError):
            await handle.demo_wrong_result_type()
        assert await handle.demo_sync(a=2, b=2) == 4


class TestNonExceptionExits:
    async def test_systemexit_takes_the_server_down_and_fails_the_call(
        self,
        spawn,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SystemExit kills the server; the caller fails either at submit or at the poll deadline."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)
        server = spawn()
        handle = make_handle(server, call_timeout_seconds=0.3)

        with pytest.raises((WorkerUnreachableError, TimeoutError)):
            await handle.demo_system_exit()

        assert server.wait(timeout=10.0) is not None

    async def test_async_systemexit_takes_the_server_down_and_fails_the_call(
        self,
        spawn,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SystemExit from an async method escapes the event loop too, killing the server and the call."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)
        server = spawn()
        handle = make_handle(server, call_timeout_seconds=0.3)

        with pytest.raises((WorkerUnreachableError, TimeoutError)):
            await handle.demo_system_exit_async()

        assert server.wait(timeout=10.0) is not None

    async def test_calls_in_flight_when_worker_exits_finish_by_deadline(
        self,
        spawn,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A concurrent accepted call dies with the server and reaches its own deadline."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)
        handle = make_handle(spawn(), call_timeout_seconds=0.3)
        other = asyncio.create_task(handle.demo_sleep_async(tag="doomed", seconds=30.0))
        await asyncio.sleep(0.05)

        with pytest.raises((WorkerUnreachableError, TimeoutError)):
            await handle.demo_system_exit()

        with pytest.raises(TimeoutError):
            await other
