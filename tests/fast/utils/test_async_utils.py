"""Tests for eager_create_task — contrast with plain asyncio.create_task."""

import asyncio

import pytest

from miles.utils.async_utils import eager_create_task


async def _eager_create(coro):
    return await eager_create_task(coro)


async def _plain_create(coro):
    return asyncio.create_task(coro)


@pytest.mark.asyncio
@pytest.mark.parametrize("create_fn", [_eager_create, _plain_create], ids=["eager", "plain"])
class TestCreateTaskComparison:
    async def test_returns_asyncio_task(self, create_fn):
        async def coro():
            return 42

        task = await create_fn(coro())
        assert isinstance(task, asyncio.Task)
        assert await task == 42

    async def test_started_before_next_line(self, create_fn):
        """eager starts immediately; plain does not."""
        started = False

        async def coro():
            nonlocal started
            started = True
            await asyncio.sleep(10)

        task = await create_fn(coro())

        if create_fn is _eager_create:
            assert started, "eager_create_task should have started the task"
        else:
            assert not started, "plain create_task should NOT have started the task yet"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_dispatch_order(self, create_fn):
        """eager preserves critic-before-actor dispatch order; plain reverses it."""
        order: list[str] = []

        async def critic():
            order.append("critic")
            await asyncio.sleep(0.1)

        async def actor():
            order.append("actor")
            await asyncio.sleep(0.1)

        critic_task = await create_fn(critic())
        await actor()
        await critic_task

        if create_fn is _eager_create:
            assert order == ["critic", "actor"]
        else:
            assert order == ["actor", "critic"]

    async def test_multiple_tasks_order(self, create_fn):
        """eager dispatches 1,2,3 in order; plain defers all until first yield."""
        order: list[int] = []

        async def append(n: int):
            order.append(n)
            await asyncio.sleep(0)

        tasks = [await create_fn(append(i)) for i in [1, 2, 3]]
        await asyncio.gather(*tasks)

        if create_fn is _eager_create:
            assert order[:3] == [1, 2, 3]
        else:
            assert order[:3] == [1, 2, 3]  # plain also preserves order once they all run

    async def test_exception_propagates(self, create_fn):
        async def failing():
            raise ValueError("boom")

        task = await create_fn(failing())

        with pytest.raises(ValueError, match="boom"):
            await task

    async def test_result_available(self, create_fn):
        async def compute():
            return {"key": "value"}

        task = await create_fn(compute())
        assert await task == {"key": "value"}
