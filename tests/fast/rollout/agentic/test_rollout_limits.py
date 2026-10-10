"""Rollout limits hold across a rollout's agent processes and free a slot when its holder dies."""

import asyncio

import pytest

from miles.rollout.agentic.agent_function import call_agent_function
from miles.utils.test_utils import agent_function_stubs as stubs


async def _in_agent_processes(fn, count: int, **kwargs) -> list:
    return await asyncio.gather(*(call_agent_function(fn, mode="subproc", **kwargs) for _ in range(count)))


def _peak_overlap(spans: list[tuple[float, float]]) -> int:
    return max(sum(1 for start, end in spans if start <= t < end) for t, _ in spans)


def test_semaphore_caps_holders_across_agent_processes():
    spans = asyncio.run(_in_agent_processes(stubs.hold_semaphore, 6, name="test-cap", limit=2, hold_s=1.0))
    assert _peak_overlap(spans) <= 2
    assert max(end for _, end in spans) - min(start for start, _ in spans) >= 2.9, "six 1s holds under 2 slots"


def test_lock_serializes_holders_across_agent_processes():
    spans = asyncio.run(_in_agent_processes(stubs.hold_lock, 3, name="test-lock", hold_s=0.5))
    assert _peak_overlap(spans) == 1


def test_a_dead_holder_frees_its_slot():
    with pytest.raises(RuntimeError, match="before returning a result"):
        asyncio.run(call_agent_function(stubs.die_holding_semaphore, mode="subproc", name="test-dead"))
    asyncio.run(
        asyncio.wait_for(
            call_agent_function(stubs.hold_semaphore, mode="subproc", name="test-dead", limit=1, hold_s=0.0), 30
        )
    )
