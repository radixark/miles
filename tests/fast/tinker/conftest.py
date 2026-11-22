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
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task
