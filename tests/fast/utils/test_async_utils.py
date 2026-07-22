"""Tests for eager_create_task and AsyncioGatherUtils."""

import asyncio
import logging

import pytest

from miles.utils.async_utils import AsyncioGatherUtils, eager_create_task


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


# ---- AsyncioGatherUtils ----

_ERR1 = RuntimeError("boom")
_ERR2 = ValueError("kaboom")


class TestAsyncioGatherUtilsHasError:
    def test_no_errors(self):
        assert AsyncioGatherUtils.has_error([1, "ok", None]) is False

    def test_empty_list(self):
        assert AsyncioGatherUtils.has_error([]) is False

    def test_single_exception(self):
        assert AsyncioGatherUtils.has_error([_ERR1]) is True

    def test_exception_among_successes(self):
        assert AsyncioGatherUtils.has_error(["ok", _ERR1, 42]) is True

    def test_multiple_exceptions(self):
        assert AsyncioGatherUtils.has_error([_ERR1, _ERR2]) is True

    def test_base_exception_detected(self):
        assert AsyncioGatherUtils.has_error([KeyboardInterrupt()]) is True

    def test_exception_subclass_detected(self):
        assert AsyncioGatherUtils.has_error([OSError("disk")]) is True


class TestAsyncioGatherUtilsLogError:
    def test_logs_nothing_on_all_success(self, caplog):
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error(["ok", 42], debug_name="test_op")

        assert not any("test_op" in r.message for r in caplog.records)

    def test_logs_nothing_on_empty(self, caplog):
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error([], debug_name="test_op")

        assert len(caplog.records) == 0

    def test_logs_single_error_with_index(self, caplog):
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error(["ok", _ERR1], debug_name="test_op")

        error_records = [r for r in caplog.records if "test_op" in r.message]
        assert len(error_records) == 1
        assert "index=1" in error_records[0].message
        assert error_records[0].exc_info is not None

    def test_logs_multiple_errors_with_correct_indices(self, caplog):
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error([_ERR1, "ok", _ERR2], debug_name="my_gather")

        error_records = [r for r in caplog.records if "my_gather" in r.message]
        assert len(error_records) == 2
        assert "index=0" in error_records[0].message
        assert "index=2" in error_records[1].message

    def test_logs_include_debug_name(self, caplog):
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error([_ERR1], debug_name="refresh_cells#coop")

        assert any("refresh_cells#coop" in r.message for r in caplog.records)

    def test_logs_exc_info_is_the_exception(self, caplog):
        err = TypeError("specific")
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error([err], debug_name="op")

        record = [r for r in caplog.records if "op" in r.message][0]
        assert record.exc_info[1] is err
