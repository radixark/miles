from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace

import pytest

import miles.rollout.inference_rollout.inference_rollout_train as inference_rollout_train


async def test_request_abort_settles_every_worker_before_propagating_failure(monkeypatch) -> None:
    failure = RuntimeError("first worker abort failed")
    first_finished = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    second_finished = asyncio.Event()
    agent_abort_called = asyncio.Event()

    async def get_worker_urls(args: Namespace) -> list[str]:
        return ["http://worker-0", "http://worker-1"]

    async def post(url: str, payload: dict[str, bool]) -> None:
        assert payload == {"abort_all": True}
        if url == "http://worker-0/abort_request":
            first_finished.set()
            raise failure
        assert url == "http://worker-1/abort_request"
        second_started.set()
        await release_second.wait()
        second_finished.set()

    async def call_agent_abort_hook(args: Namespace) -> None:
        agent_abort_called.set()

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", get_worker_urls)
    monkeypatch.setattr(inference_rollout_train, "post", post)
    monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", call_agent_abort_hook)
    abort_task = asyncio.create_task(inference_rollout_train.request_abort(Namespace()))
    await first_finished.wait()
    await second_started.wait()

    try:
        await asyncio.wait_for(agent_abort_called.wait(), timeout=0.01)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(abort_task), timeout=0.01)
        assert not second_finished.is_set()
    finally:
        release_second.set()
        await asyncio.gather(abort_task, return_exceptions=True)

    with pytest.raises(RuntimeError) as error:
        await abort_task

    assert error.value is failure
    assert second_finished.is_set()
    assert agent_abort_called.is_set()


async def test_request_abort_settles_agent_hook_when_worker_requests_time_out(monkeypatch) -> None:
    worker_started = asyncio.Event()
    worker_cancelled = asyncio.Event()
    agent_abort_finished = asyncio.Event()

    async def get_worker_urls(args: Namespace) -> list[str]:
        return ["http://worker-0"]

    async def post(url: str, payload: dict[str, bool]) -> None:
        worker_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            worker_cancelled.set()
            raise

    async def call_agent_abort_hook(args: Namespace) -> None:
        await asyncio.sleep(0)
        agent_abort_finished.set()

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", get_worker_urls)
    monkeypatch.setattr(inference_rollout_train, "post", post)
    monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", call_agent_abort_hook)
    abort_task = asyncio.create_task(
        asyncio.wait_for(
            inference_rollout_train.request_abort(Namespace()),
            timeout=0.01,
        )
    )
    await worker_started.wait()

    with pytest.raises(asyncio.TimeoutError):
        await abort_task

    assert worker_cancelled.is_set()
    assert agent_abort_finished.is_set()


async def test_request_abort_cancellation_terminates_the_agent_hook(monkeypatch) -> None:
    worker_started = asyncio.Event()
    agent_abort_started = asyncio.Event()
    agent_abort_cancelled = asyncio.Event()
    release_agent_abort = asyncio.Event()

    async def get_worker_urls(args: Namespace) -> list[str]:
        return ["http://worker-0"]

    async def post(url: str, payload: dict[str, bool]) -> None:
        worker_started.set()
        await asyncio.Future()

    async def call_agent_abort_hook(args: Namespace) -> None:
        agent_abort_started.set()
        try:
            await release_agent_abort.wait()
        except asyncio.CancelledError:
            agent_abort_cancelled.set()
            raise

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", get_worker_urls)
    monkeypatch.setattr(inference_rollout_train, "post", post)
    monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", call_agent_abort_hook)
    abort_task = asyncio.create_task(inference_rollout_train.request_abort(Namespace()))
    await worker_started.wait()
    await agent_abort_started.wait()

    abort_task.cancel()
    done, pending = await asyncio.wait({abort_task}, timeout=0.1)
    try:
        assert (done, pending) == ({abort_task}, set())
        assert agent_abort_cancelled.is_set()
    finally:
        release_agent_abort.set()
        await asyncio.gather(abort_task, return_exceptions=True)

    with pytest.raises(asyncio.CancelledError):
        await abort_task


async def test_request_abort_preserves_worker_discovery_failure_after_agent_hook_settles(monkeypatch) -> None:
    discovery_error = RuntimeError("worker discovery failed")
    agent_abort_finished = asyncio.Event()

    async def get_worker_urls(args: Namespace) -> list[str]:
        raise discovery_error

    async def call_agent_abort_hook(args: Namespace) -> None:
        agent_abort_finished.set()

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", get_worker_urls)
    monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", call_agent_abort_hook)

    with pytest.raises(RuntimeError) as error:
        await inference_rollout_train.request_abort(Namespace())

    assert error.value is discovery_error
    assert agent_abort_finished.is_set()


async def test_request_abort_propagates_agent_hook_cancellation(monkeypatch) -> None:
    cancellation = asyncio.CancelledError("agent abort cancelled")

    async def get_worker_urls(args: Namespace) -> list[str]:
        return []

    async def call_agent_abort_hook(args: Namespace) -> None:
        raise cancellation

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", get_worker_urls)
    monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", call_agent_abort_hook)

    with pytest.raises(asyncio.CancelledError) as error:
        await inference_rollout_train.request_abort(Namespace())

    assert error.value is cancellation
