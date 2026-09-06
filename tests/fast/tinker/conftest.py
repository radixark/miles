import asyncio

import pytest

from tests.fast.tinker.harness import make_service


@pytest.fixture
async def service(tmp_path):
    """A TinkerService over a FakeBackend with its dispatch loop running."""
    svc = make_service(checkpoint_root=str(tmp_path))
    run_task = asyncio.create_task(svc.run())
    yield svc
    for task in (run_task, getattr(svc, "_sweep_task", None)):
        if task is not None:
            task.cancel()
