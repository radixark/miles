"""Node limits hold across the processes of one run and free a slot when its holder dies."""

import asyncio
import os
import time

import pytest
import ray

from miles.rollout.agentic.node_limits import node_lock, node_semaphore


@ray.remote(num_cpus=0.01, max_retries=0)
def _hold_semaphore(name: str, limit: int, hold_s: float) -> tuple[float, float]:
    async def hold() -> tuple[float, float]:
        async with node_semaphore(name, limit):
            start = time.monotonic()
            await asyncio.sleep(hold_s)
            return start, time.monotonic()

    return asyncio.run(hold())


@ray.remote(num_cpus=0.01, max_retries=0)
def _hold_lock(name: str, hold_s: float) -> tuple[float, float]:
    with node_lock(name):
        start = time.monotonic()
        time.sleep(hold_s)
        return start, time.monotonic()


@ray.remote(num_cpus=0.01, max_calls=1, max_retries=0)
def _die_holding(name: str) -> None:
    async def die() -> None:
        async with node_semaphore(name, 1):
            os._exit(1)

    asyncio.run(die())


def _peak_overlap(spans: list[tuple[float, float]]) -> int:
    return max(sum(1 for start, end in spans if start <= t < end) for t, _ in spans)


def test_semaphore_caps_holders_across_processes(ray_local_mode):
    spans = ray.get([_hold_semaphore.remote("test-cap", 2, 1.0) for _ in range(6)])
    assert _peak_overlap(spans) <= 2
    assert max(end for _, end in spans) - min(start for start, _ in spans) >= 2.9, "six 1s holds under 2 slots"


def test_lock_serializes_holders_across_processes(ray_local_mode):
    spans = ray.get([_hold_lock.remote("test-lock", 0.5) for _ in range(3)])
    assert _peak_overlap(spans) == 1


def test_a_dead_holder_frees_its_slot(ray_local_mode):
    with pytest.raises(ray.exceptions.RayError):
        ray.get(_die_holding.remote("test-dead"))
    ray.get(_hold_semaphore.remote("test-dead", 1, 0.0), timeout=30)
