"""Tests for eager_create_task and AsyncioGatherUtils."""

import asyncio
import concurrent.futures
import logging
import threading

import pytest

from miles.utils import async_utils
from miles.utils.async_utils import AsyncioGatherUtils, AsyncLoopThread, eager_create_task


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


# ---- AsyncLoopThread.submit / wait_futures ----


@pytest.fixture
def loop_thread():
    thread = AsyncLoopThread()
    yield thread
    thread.loop.call_soon_threadsafe(thread.loop.stop)


class TestSubmit:
    def test_submit_returns_before_the_coroutine_finishes(self, loop_thread):
        """Non-blocking dispatch is the whole point: the caller must get control back."""
        release = threading.Event()

        async def blocked():
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return "done"

        future = loop_thread.submit(blocked())
        assert not future.done()

        release.set()
        assert future.result(timeout=10) == "done"

    def test_run_blocks_until_the_result_is_available(self, loop_thread):
        """``run`` is the blocking sibling used by the synchronous ray actor shells."""

        async def compute():
            await asyncio.sleep(0)
            return 7

        assert loop_thread.run(compute()) == 7

    def test_exceptions_surface_at_the_collection_point(self, loop_thread):
        """A failed request must not disappear into the background loop."""

        async def failing():
            raise ValueError("boom")

        future = loop_thread.submit(failing())

        with pytest.raises(ValueError, match="boom"):
            future.result(timeout=10)

    def test_run_propagates_the_coroutine_exception(self, loop_thread):
        """``run`` blocks on the future, so a failure must land on the synchronous caller."""

        async def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            loop_thread.run(failing())

    def test_cancelling_submitted_future_cancels_the_coroutine(self, loop_thread):
        """Dropping a fired request must stop the background work, not orphan it on the loop."""
        started = threading.Event()
        cancelled = threading.Event()

        async def blocked():
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        future = loop_thread.submit(blocked())
        assert started.wait(timeout=10)

        assert future.cancel()
        assert cancelled.wait(timeout=10)
        assert future.cancelled()

    def test_submitting_from_inside_the_loop_thread_is_refused(self, loop_thread):
        """Blocking on that future would freeze the very loop that must resolve it."""

        async def submit_from_the_loop():
            with pytest.raises(AssertionError, match="deadlock"):
                loop_thread.submit(asyncio.sleep(0))

        loop_thread.run(submit_from_the_loop())

    def test_a_fan_out_larger_than_any_thread_pool_still_all_lands(self, loop_thread):
        """This is why the dispatch is a loop and not a thread pool."""
        arrived = threading.Semaphore(0)
        release = threading.Event()

        async def blocked():
            arrived.release()
            await asyncio.get_running_loop().run_in_executor(_UNBOUNDED_POOL, release.wait)
            return "ok"

        futures = [loop_thread.submit(blocked()) for _ in range(200)]
        for _ in range(200):
            assert arrived.acquire(timeout=30)

        release.set()
        assert [f.result(timeout=30) for f in futures] == ["ok"] * 200


_UNBOUNDED_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=256)


class TestFireThenRendezvous:
    """The blocking endpoints need every engine asked before the caller joins."""

    def test_every_request_is_on_the_wire_before_the_caller_joins_the_collective(self, loop_thread):
        num_engines = 32
        barrier = threading.Barrier(num_engines + 1)

        async def engine_call(engine_index: int):
            await asyncio.get_running_loop().run_in_executor(_UNBOUNDED_POOL, barrier.wait)
            return engine_index

        futures = [loop_thread.submit(engine_call(i)) for i in range(num_engines)]

        barrier.wait(timeout=30)

        assert async_utils.wait_futures(futures) == list(range(num_engines))

    def test_collecting_before_firing_the_rest_deadlocks(self, loop_thread):
        """A caller that blocks on the first response never sends the second."""
        barrier = threading.Barrier(2)

        async def engine_call(engine_index: int):
            await asyncio.get_running_loop().run_in_executor(_UNBOUNDED_POOL, barrier.wait, 1.0)
            return engine_index

        first = loop_thread.submit(engine_call(0))

        with pytest.raises(concurrent.futures.TimeoutError):
            first.result(timeout=0.5)

        with pytest.raises(threading.BrokenBarrierError):
            first.result(timeout=10)


async def _failing(message: str):
    raise ValueError(message)


class TestWaitFutures:
    def test_it_preserves_the_submission_order(self, loop_thread):
        """Results are positional: caller i's result must stay at index i."""

        async def slow_first(delay: float, value: int):
            await asyncio.sleep(delay)
            return value

        futures = [loop_thread.submit(slow_first(0.05, 1)), loop_thread.submit(slow_first(0.0, 2))]

        assert async_utils.wait_futures(futures) == [1, 2]

    def test_it_drains_every_future_before_raising(self, loop_thread):
        """Bailing early leaves requests running against engines the caller considers done."""
        finished: list[int] = []

        async def failing():
            raise ValueError("boom")

        async def slow_success(index: int):
            await asyncio.sleep(0.1)
            finished.append(index)
            return index

        futures = [loop_thread.submit(failing()), loop_thread.submit(slow_success(1))]

        with pytest.raises(ValueError, match="boom"):
            async_utils.wait_futures(futures)

        assert finished == [1]

    def test_it_raises_the_first_error_in_submission_order(self, loop_thread):
        """A stable error keeps failures reproducible when several engines fail."""

        async def failing(message: str, delay: float):
            await asyncio.sleep(delay)
            raise ValueError(message)

        futures = [loop_thread.submit(failing("second", 0.1)), loop_thread.submit(failing("first", 0.0))]

        with pytest.raises(ValueError, match="second"):
            async_utils.wait_futures(futures)

    def test_an_empty_fan_out_is_not_an_error(self):
        """A group with no live engines must not blow up the weight update."""
        assert async_utils.wait_futures([]) == []

    def test_every_failure_is_logged_not_just_the_one_that_is_raised(self, caplog):
        """Only the first error propagates, so the others are only ever seen in the log."""
        futures = [async_utils.submit(_failing("first")), async_utils.submit(_failing("second"))]

        with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="first"):
            async_utils.wait_futures(futures)

        assert [record.message for record in caplog.records if "wait_futures" in record.message] == [
            "wait_futures index=0 failed",
            "wait_futures index=1 failed",
        ]

    def test_each_wait_future_failure_logs_its_exception_info(self, caplog):
        """The attached exception is the only detail available for failures that never propagate."""
        futures = [async_utils.submit(_failing("first")), async_utils.submit(_failing("second"))]

        with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="first"):
            async_utils.wait_futures(futures)

        records = [record for record in caplog.records if "wait_futures" in record.message]
        assert [str(record.exc_info[1]) for record in records] == ["first", "second"]

    def test_an_interrupt_is_not_held_back_by_the_remaining_futures(self):
        """One Ctrl-C must kill a hung weight update, not queue behind a silent engine."""
        touched: list[str] = []

        class _Interrupted(concurrent.futures.Future):
            def result(self, timeout=None):
                raise KeyboardInterrupt

        class _NeverFinishes(concurrent.futures.Future):
            def result(self, timeout=None):
                touched.append("second")
                raise AssertionError("the interrupt should have propagated before this")

        with pytest.raises(KeyboardInterrupt):
            async_utils.wait_futures([_Interrupted(), _NeverFinishes()])

        assert touched == []

    def test_a_module_level_submit_reaches_the_shared_background_loop(self):
        """Production fires through the module-level facade, not through a hand-built loop thread."""

        async def record():
            return threading.get_ident()

        idents = async_utils.wait_futures([async_utils.submit(record()) for _ in range(50)])

        assert len(set(idents)) == 1, "every request must run on the one background loop"
        assert threading.get_ident() not in idents, "the caller thread must stay free to enter the collective"
