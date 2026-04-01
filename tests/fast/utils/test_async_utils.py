"""Tests for eager_create_task."""

import asyncio

import pytest

from miles.utils.async_utils import eager_create_task


class TestEagerCreateTask:
    @pytest.mark.asyncio
    async def test_returns_asyncio_task(self):
        async def coro():
            return 42

        task = await eager_create_task(coro())

        assert isinstance(task, asyncio.Task)
        assert await task == 42

    @pytest.mark.asyncio
    async def test_task_starts_before_return(self):
        """The task's first code runs before eager_create_task returns."""
        started = False

        async def coro():
            nonlocal started
            started = True
            await asyncio.sleep(10)

        task = await eager_create_task(coro())

        assert started, "Task should have started executing before eager_create_task returned"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_remote_calls_dispatched_before_caller_continues(self):
        """Simulates the Ray .remote() dispatch pattern: side effects in the task
        happen before the caller's next line."""
        dispatch_order: list[str] = []

        async def critic():
            dispatch_order.append("critic_dispatched")
            await asyncio.sleep(0.1)

        async def actor():
            dispatch_order.append("actor_dispatched")
            await asyncio.sleep(0.1)

        critic_task = await eager_create_task(critic())
        # After eager_create_task, critic should have appended already
        assert "critic_dispatched" in dispatch_order

        await actor()
        await critic_task

        assert dispatch_order == ["critic_dispatched", "actor_dispatched"]

    @pytest.mark.asyncio
    async def test_bare_create_task_does_not_start_immediately(self):
        """Contrast: plain asyncio.create_task does NOT start before next line."""
        started = False

        async def coro():
            nonlocal started
            started = True

        _task = asyncio.create_task(coro())
        assert not started, "Plain create_task should NOT have started yet"

        await _task
        assert started

    @pytest.mark.asyncio
    async def test_exception_propagates_on_await(self):
        async def failing():
            raise ValueError("boom")

        task = await eager_create_task(failing())

        with pytest.raises(ValueError, match="boom"):
            await task

    @pytest.mark.asyncio
    async def test_multiple_eager_tasks_preserve_order(self):
        order: list[int] = []

        async def append(n: int):
            order.append(n)
            await asyncio.sleep(0)

        task1 = await eager_create_task(append(1))
        task2 = await eager_create_task(append(2))
        task3 = await eager_create_task(append(3))

        await asyncio.gather(task1, task2, task3)

        assert order[:3] == [1, 2, 3], "Tasks should start in dispatch order"

    @pytest.mark.asyncio
    async def test_result_available_after_await(self):
        async def compute():
            return {"key": "value"}

        task = await eager_create_task(compute())
        result = await task

        assert result == {"key": "value"}
