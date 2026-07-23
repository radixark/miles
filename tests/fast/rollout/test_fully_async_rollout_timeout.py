"""Tests for the per-task timeout + exception-safe requeue contract in
``examples/fully_async/fully_async_rollout.py``.

Covers:
  (a) Normal completion — samples flow through unchanged.
  (b) Timeout exceeded — samples are marked ``Sample.Status.ABORTED`` so the
      outer collector's aborted-requeue path re-adds them to the data buffer.
  (c) Coroutine raises — samples are also marked ``ABORTED`` instead of the
      group being silently dropped (regression test for the previous bare
      ``except Exception: print`` in ``continuous_worker_loop``).
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

from miles.utils.types import Sample

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "fully_async"
    / "fully_async_rollout.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fully_async_rollout_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fa = _load_module()
_run_group_with_timeout = _fa._run_group_with_timeout


def _group(n: int = 2) -> list[Sample]:
    return [Sample(index=i, prompt="p") for i in range(n)]


async def test_normal_completion_returns_samples_unchanged():
    group = _group(3)

    async def ok() -> list[Sample]:
        for s in group:
            s.status = Sample.Status.COMPLETED
            s.response = "done"
        return group

    out = await _run_group_with_timeout(ok(), group, timeout_s=5.0)

    assert out is group
    assert [s.status for s in out] == [Sample.Status.COMPLETED] * 3
    assert all(s.response == "done" for s in out)


async def test_timeout_marks_samples_aborted():
    group = _group(4)

    async def slow() -> list[Sample]:
        await asyncio.sleep(10)
        return group

    out = await _run_group_with_timeout(slow(), group, timeout_s=0.05)

    assert out is group
    assert len(out) == 4
    assert all(s.status == Sample.Status.ABORTED for s in out)


async def test_exception_does_not_drop_group():
    group = _group(2)

    async def boom() -> list[Sample]:
        raise RuntimeError("simulated transport failure")

    out = await _run_group_with_timeout(boom(), group, timeout_s=5.0)

    assert out is group
    assert all(s.status == Sample.Status.ABORTED for s in out)


async def test_subtask_exception_via_gather_does_not_drop_group():
    """Mirrors production bug: ``generate_and_rm_group`` uses ``asyncio.gather``
    without ``return_exceptions=True`` so a single sub-task raise propagates.
    The wrapper must still return the group (with ABORTED status) rather than
    letting the exception escape and silently drop the group.
    """
    group = _group(3)

    async def good() -> Sample:
        await asyncio.sleep(0)
        return Sample(index=0, prompt="p")

    async def bad() -> Sample:
        raise ConnectionResetError("peer hung up")

    async def gather_like():
        # One failing sub-task must propagate out of gather, which is the exact
        # production failure mode this wrapper is designed to contain.
        return await asyncio.gather(good(), bad(), good())

    out = await _run_group_with_timeout(gather_like(), group, timeout_s=5.0)

    assert out is group
    assert all(s.status == Sample.Status.ABORTED for s in out)


async def test_none_timeout_disables_time_bound_but_keeps_exception_safety():
    group = _group(2)

    async def boom() -> list[Sample]:
        raise ValueError("still caught")

    out = await _run_group_with_timeout(boom(), group, timeout_s=None)

    assert out is group
    assert all(s.status == Sample.Status.ABORTED for s in out)


async def test_non_positive_timeout_disables_time_bound():
    group = _group(1)

    async def quick() -> list[Sample]:
        group[0].status = Sample.Status.COMPLETED
        return group

    out = await _run_group_with_timeout(quick(), group, timeout_s=0.0)

    assert out is group
    assert out[0].status == Sample.Status.COMPLETED


async def test_cancellation_propagates_after_marking():
    group = _group(2)

    async def long_running() -> list[Sample]:
        await asyncio.sleep(10)
        return group

    task = asyncio.create_task(_run_group_with_timeout(long_running(), group, timeout_s=5.0))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Samples should have been marked before the cancellation propagated so a
    # caller that catches CancelledError can still requeue.
    assert all(s.status == Sample.Status.ABORTED for s in group)
