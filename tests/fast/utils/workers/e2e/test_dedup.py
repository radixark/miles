import asyncio
from typing import Any

import httpx
import pytest

from miles.utils.workers.worker_handle import WorkerUnreachableError


async def _submit(
    raw: httpx.AsyncClient,
    *,
    method: str,
    call_id: str,
    query: dict[str, Any],
) -> httpx.Response:
    return await raw.post(f"/v1/{method}", json={"call_id": call_id, "query": query})


async def _poll(raw: httpx.AsyncClient, *, call_id: str, timeout: float = 5.0) -> dict[str, Any]:
    return (await raw.get(f"/v1/calls/{call_id}", params={"timeout": timeout})).json()


class TestDuplicateSubmits:
    async def test_identical_resubmit_returns_409(self, raw: httpx.AsyncClient, handle: Any, tag: str) -> None:
        """Resubmitting the same call id and payload fails loudly."""
        first = await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})
        assert first.json() == {"status": "submitted"}
        await _poll(raw, call_id=tag)

        second = await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})

        assert second.status_code == 409
        assert "already submitted" in second.json()["detail"]
        assert await handle.report_counter(tag=tag) == 1

    async def test_in_flight_duplicate_returns_409_immediately(
        self,
        raw: httpx.AsyncClient,
        handle: Any,
        tag: str,
    ) -> None:
        """A duplicate in-flight call fails without waiting for completion."""
        await _submit(raw, method="demo_block_sync", call_id=tag, query={"tag": tag})
        await asyncio.sleep(0.05)

        duplicate = await asyncio.wait_for(
            _submit(raw, method="demo_block_sync", call_id=tag, query={"tag": tag}),
            timeout=1.0,
        )

        assert duplicate.status_code == 409
        await handle.release(tag=tag)
        assert (await _poll(raw, call_id=tag, timeout=2.0))["status"] == "success"

    async def test_concurrent_duplicates_have_one_acceptance(
        self,
        raw: httpx.AsyncClient,
        handle: Any,
        tag: str,
    ) -> None:
        """Concurrent duplicate submits produce one acceptance and nine conflicts."""
        responses = await asyncio.gather(
            *[_submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag}) for _ in range(10)]
        )
        await _poll(raw, call_id=tag)

        assert [response.status_code for response in responses].count(200) == 1
        assert [response.status_code for response in responses].count(409) == 9
        assert await handle.report_counter(tag=tag) == 1

    async def test_duplicate_conflict_leaves_original_pollable(self, raw: httpx.AsyncClient, tag: str) -> None:
        """A duplicate conflict leaves the original outcome pollable."""
        await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})
        duplicate = await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})

        assert duplicate.status_code == 409
        assert (await _poll(raw, call_id=tag))["status"] == "success"


class TestCallIdIdentity:
    async def test_different_payload_same_call_id_returns_409(
        self,
        raw: httpx.AsyncClient,
        handle: Any,
        tag: str,
    ) -> None:
        """Reusing a call id with a different payload returns 409."""
        await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})
        response = await _submit(
            raw,
            method="demo_count_sync",
            call_id=tag,
            query={"tag": f"{tag}other"},
        )

        assert response.status_code == 409
        assert await handle.report_counter(tag=f"{tag}other") == 0

    async def test_different_method_same_call_id_returns_409(self, raw: httpx.AsyncClient, tag: str) -> None:
        """A call id reused on another method returns 409."""
        await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})

        response = await _submit(raw, method="demo_count_async", call_id=tag, query={"tag": tag})

        assert response.status_code == 409

    async def test_key_order_does_not_change_duplicate_failure(self, raw: httpx.AsyncClient, tag: str) -> None:
        """A reordered duplicate payload still returns 409."""
        await _submit(raw, method="demo_sync", call_id=tag, query={"a": 1, "b": 2})

        response = await _submit(raw, method="demo_sync", call_id=tag, query={"b": 2, "a": 1})

        assert response.status_code == 409


class TestNoRetryAfterAmbiguousSubmit:
    async def test_submit_503_is_not_retried(
        self,
        proxy_to: Any,
        make_handle: Any,
        tag: str,
    ) -> None:
        """A submit 503 raises after one wire attempt."""
        proxy = await proxy_to()
        proxy.reject_next(count=1, status=503)
        handle = make_handle(proxy)

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1

    async def test_dropped_submit_response_is_not_retried(
        self,
        proxy_to: Any,
        make_handle: Any,
        handle: Any,
        tag: str,
    ) -> None:
        """A dropped submit response raises once while the accepted call runs."""
        proxy = await proxy_to()
        proxy.drop_next(count=1)
        retrying = make_handle(proxy)

        with pytest.raises(WorkerUnreachableError):
            await retrying.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1
        for _ in range(100):
            if await handle.report_counter(tag=tag) == 1:
                break
            await asyncio.sleep(0)
        assert await handle.report_counter(tag=tag) == 1
