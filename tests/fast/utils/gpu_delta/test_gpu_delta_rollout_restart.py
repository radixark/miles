"""The GPU-delta E2E injector waits for a fresh rollout-engine generation."""

import json
from types import SimpleNamespace

import pytest

from miles.utils.test_utils import ft_test_actions
from miles.utils.test_utils.ft_test_actions import FTTestAction, FTTestActionOrchestrationExecutor


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_resume", [False, True])
async def test_rollout_restart_waits_for_fresh_ready_generation(auto_resume, monkeypatch, caplog):
    from functools import partial
    from unittest.mock import AsyncMock

    cell_id = "inference-engine-all-0-0-00001"
    healthy_id = "inference-engine-all-0-0-00000"

    def cell(name, generation, offset):
        return SimpleNamespace(
            alive=True,
            workers_hash=generation,
            worker_names=[f"{name}-00000"],
            meta={"gpu_offset": offset, "model_id": "actor"},
        )

    before = {cell_id: cell(cell_id, "original", 6), healthy_id: cell(healthy_id, "healthy", 4)}
    after = {**before, cell_id: cell(cell_id, "replacement", 6)}
    operations = SimpleNamespace(
        cell_infos=AsyncMock(side_effect=[before, before, after, after, after]),
        suspend=AsyncMock(),
        resume=AsyncMock(),
    )
    controller = SimpleNamespace(
        get_cell_statuses=AsyncMock(
            side_effect=[
                {cell_id: SimpleNamespace(workers_hash="original", phase="Running")},
                {cell_id: SimpleNamespace(workers_hash="original", phase="Running")},
                {cell_id: SimpleNamespace(workers_hash="replacement", phase="Pending")},
                {cell_id: SimpleNamespace(workers_hash="replacement", phase="Running")},
            ]
        )
    )

    async def poll(fn, **_kwargs):
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await fn(60)
        return await fn(60)

    monkeypatch.setattr(ft_test_actions, "retry_until_deadline", poll)
    executor = FTTestActionOrchestrationExecutor(
        actions=[FTTestAction(at_rollout=1, action="restart_rollout_cell_at_end", cell_id=cell_id)],
        restart_rollout_cell=partial(
            ft_test_actions._restart_rollout_cell,
            operations=operations,
            controller=controller,
            pool_ids=["inference-engine-all-0-0"],
            auto_resume=auto_resume,
        ),
    )
    await executor.run_after_step(rollout_id=1)
    operations.suspend.assert_awaited_once_with(cell_id=cell_id)
    if auto_resume:
        operations.resume.assert_not_awaited()
    else:
        operations.resume.assert_awaited_once_with(cell_id=cell_id)
    [record] = [record.message for record in caplog.records if record.message.startswith("[ft test rollout restart] ")]
    receipt = json.loads(record.removeprefix("[ft test rollout restart] "))
    assert receipt["at_rollout"] == 1
    assert receipt["cell_id"] == cell_id
    assert receipt["original"]["workers_hash"] == "original"
    assert receipt["replacement"]["workers_hash"] == "replacement"
    assert receipt["replacement"]["gpu_offset"] == 6
    assert receipt["other_cells"] == {healthy_id: {"before": "healthy", "after": "healthy"}}
