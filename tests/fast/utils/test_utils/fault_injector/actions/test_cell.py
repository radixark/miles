import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.fast.utils.test_utils.fault_injector.fakes import _CellOperations, _Controller

from miles.utils.test_utils.fault_injector.actions import cell as cell_actions
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.actions.cell import (
    RestartRolloutCellAction,
    StartCellAction,
    StopCellAction,
)


class TestCellActionResources:
    async def test_stopping_without_cell_operations_fails_loudly(self) -> None:
        """A process that holds no cell operations must refuse a stop instead of skipping it."""
        with pytest.raises(AssertionError, match="needs the controller's cell operations"):
            await StopCellAction(cell_id="cell-0")(context=FaultHookContext(), resources=FaultHookResources())

    @pytest.mark.parametrize("with_controller,with_operations", [(False, True), (True, False), (False, False)])
    async def test_starting_without_the_controller_and_its_operations_fails_before_resuming(
        self, with_controller: bool, with_operations: bool
    ) -> None:
        """A start must refuse to resume a cell it cannot then observe rejoining."""
        operations = _CellOperations()
        resources = FaultHookResources(
            controller=_Controller(cell_ids=("cell-0",)) if with_controller else None,
            cell_operations=operations if with_operations else None,
        )
        with pytest.raises(AssertionError, match="needs the controller"):
            await StartCellAction(cell_id="cell-0")(context=FaultHookContext(), resources=resources)
        assert operations.started == []


class TestRestartRolloutCell:
    @pytest.mark.parametrize("auto_resume", [False, True])
    @pytest.mark.parametrize("public_action", [False, True])
    async def test_restart_waits_for_a_fresh_ready_generation(
        self,
        auto_resume: bool,
        public_action: bool,
        monkeypatch: pytest.MonkeyPatch,
        parkable_script: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Both entrypoints wait for a new ready generation using category-filtered cell observations."""
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
        observations = iter([before, before, after, after, after])

        async def cell_infos(*, pool_ids: list[str] | None, category: str) -> dict:
            assert pool_ids is None
            assert category == "inference_engine"
            return next(observations)

        operations = SimpleNamespace(
            cell_infos=cell_infos,
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

        monkeypatch.setattr(cell_actions, "retry_until_deadline", poll)
        if public_action:
            from miles.ray import wiring

            capability = SimpleNamespace(
                cell_operations=lambda: operations,
                static_worker_provider=lambda **kwargs: SimpleNamespace(get_handle=lambda name: controller),
            )
            monkeypatch.setattr(wiring, "compute_backend_capability", lambda args: capability)
            args = SimpleNamespace(
                update_weights_interval=1,
                ft_components=("rollout",),
                cluster_backend="kubernetes" if auto_resume else "ray",
            )
            await RestartRolloutCellAction(cell_id=cell_id)(
                context=FaultHookContext(rollout_id=1), resources=FaultHookResources(args=args)
            )
            receipt = json.loads(caplog.records[-1].message.removeprefix("[ft test rollout restart] "))
        else:
            receipt = await cell_actions._restart_rollout_cell(
                cell_id,
                operations=operations,
                controller=controller,
                category="inference_engine",
                auto_resume=auto_resume,
            )
        operations.suspend.assert_awaited_once_with(cell_id=cell_id)
        if auto_resume:
            operations.resume.assert_not_awaited()
        else:
            operations.resume.assert_awaited_once_with(cell_id=cell_id)
        assert receipt["cell_id"] == cell_id
        assert receipt["original"]["workers_hash"] == "original"
        assert receipt["replacement"]["workers_hash"] == "replacement"
        assert receipt["replacement"]["gpu_offset"] == 6
        assert receipt["other_cells"] == {healthy_id: {"before": "healthy", "after": "healthy"}}

    async def test_restart_requires_rollout_fault_tolerance(self, parkable_script):
        args = SimpleNamespace(update_weights_interval=1, ft_components=("train",))
        with pytest.raises(AssertionError, match="requires rollout fault tolerance"):
            await RestartRolloutCellAction(cell_id="inference-engine-all-0-0-00001")(
                context=FaultHookContext(rollout_id=1), resources=FaultHookResources(args=args)
            )

    async def test_restart_rejects_a_cell_outside_the_inference_category(self) -> None:
        """A target excluded by the inference category cannot be suspended by the restart action."""

        async def cell_infos(*, pool_ids: list[str] | None, category: str) -> dict:
            assert pool_ids is None
            assert category == "inference_engine"
            return {}

        operations = SimpleNamespace(cell_infos=cell_infos, suspend=AsyncMock())
        with pytest.raises(AssertionError, match="not an observed rollout cell"):
            await cell_actions._restart_rollout_cell(
                "trainer-actor-0",
                operations=operations,
                controller=SimpleNamespace(),
                category="inference_engine",
                auto_resume=False,
            )
        operations.suspend.assert_not_awaited()
