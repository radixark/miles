"""Tests for eager_create_task — contrast with plain asyncio.create_task."""

import asyncio

import pytest

from miles.utils.async_utils import eager_create_task


@pytest.mark.asyncio
@pytest.mark.parametrize("create_mode", ["eager", "plain"])
class TestCreateTaskComparison:
    async def test_returns_asyncio_task(self, create_mode):
        async def coro():
            return 42

        if create_mode == "eager":
            task = await eager_create_task(coro())
        else:
            task = asyncio.create_task(coro())

        assert isinstance(task, asyncio.Task)
        assert await task == 42

    async def test_started_before_next_line(self, create_mode):
        """eager starts immediately; plain does not."""
        started = False

        async def coro():
            nonlocal started
            started = True
            await asyncio.sleep(10)

        if create_mode == "eager":
            task = await eager_create_task(coro())
            assert started, "eager_create_task should have started the task"
        else:
            task = asyncio.create_task(coro())
            assert not started, "plain create_task should NOT have started the task yet"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_dispatch_order(self, create_mode):
        """eager preserves critic-before-actor dispatch order; plain reverses it."""
        order: list[str] = []

        async def critic():
            order.append("critic")
            await asyncio.sleep(0.1)

        async def actor():
            order.append("actor")
            await asyncio.sleep(0.1)

        if create_mode == "eager":
            critic_task = await eager_create_task(critic())
        else:
            critic_task = asyncio.create_task(critic())

        await actor()
        await critic_task

        if create_mode == "eager":
            assert order == ["critic", "actor"]
        else:
            assert order == ["actor", "critic"]

    async def test_exception_propagates(self, create_mode):
        async def failing():
            raise ValueError("boom")

        if create_mode == "eager":
            task = await eager_create_task(failing())
        else:
            task = asyncio.create_task(failing())

        with pytest.raises(ValueError, match="boom"):
            await task

    async def test_result_available(self, create_mode):
        async def compute():
            return {"key": "value"}

        if create_mode == "eager":
            task = await eager_create_task(compute())
        else:
            task = asyncio.create_task(compute())

        assert await task == {"key": "value"}
