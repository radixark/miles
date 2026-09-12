import asyncio
from contextlib import suppress

import pytest

from tests.fast.tinker.harness import make_service


@pytest.fixture
async def service(tmp_path):
    gateway = make_service(tmp_path)
    run_task = asyncio.create_task(gateway.run())
    try:
        yield gateway
    finally:
        tasks = [run_task, *gateway._create_tasks, *(task for task, _ in gateway._sample_tasks.values())]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
