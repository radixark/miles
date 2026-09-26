import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tests.utils.ft.launch import DEFAULT_TRAIN_SCRIPT

from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.actions.frozen import (
    PARKABLE_TRAIN_SCRIPT,
    SLEEP_FOREVER_INTERVAL_SECONDS,
    SleepForeverAction,
    read_frozen_rollout_id,
    write_frozen_sentinel,
)
from miles.utils.test_utils.fault_injector.controller import _FaultHookController
from miles.utils.test_utils.fault_injector.models import DeclaredFaultHookTarget, FaultHookName, FaultHookOwner, FaultHookRequest
from miles.utils.test_utils.fault_injector.static_source import write_fault_hooks

_SLEEP = FaultHookRequest(request_id="sleep", hook_name=FaultHookName.ORCHESTRATOR_STEP_END, action=SleepForeverAction(), rollout_id=2)


class TestSleepForeverHook:
    @pytest.mark.parametrize("previous", [None, 0])
    async def test_matching_step_parks_forever_and_replaces_the_sentinel(self, configure_hooks: Callable[..., _FaultHookController], parkable_script: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous: int | None) -> None:
        """A reached freeze must write its own step and never hand control to the next step."""
        path = tmp_path / "plan.json"
        write_fault_hooks(path, [_SLEEP])
        if previous is not None:
            write_frozen_sentinel(path, rollout_id=previous)
        hooks = configure_hooks([], owner=FaultHookOwner.ORCHESTRATOR, path=str(path))
        wakes: list[float] = []

        async def sleep(seconds: float) -> None:
            wakes.append(seconds)
            if len(wakes) == 3:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", sleep)
        with pytest.raises(asyncio.CancelledError):
            await hooks._reach_async(FaultHookName.ORCHESTRATOR_STEP_END, {"rollout_id": 2})

        assert wakes == [SLEEP_FOREVER_INTERVAL_SECONDS] * 3
        assert read_frozen_rollout_id(path) == 2

    @pytest.mark.parametrize("requests,rollout_id", [([_SLEEP], 0), ([_SLEEP], 1), ([_SLEEP], 3), ([_SLEEP], 99), ([], 2)])
    async def test_unmatched_or_absent_plan_returns_without_freezing(self, configure_hooks: Callable[..., _FaultHookController], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requests: list[FaultHookRequest], rollout_id: int) -> None:
        """Only a matching freeze may park the loop or leave a sentinel."""
        path = tmp_path / "plan.json"
        write_fault_hooks(path, requests)
        hooks = configure_hooks([], owner=FaultHookOwner.ORCHESTRATOR, path=str(path))

        async def unexpected_sleep(seconds: float) -> None:
            pytest.fail(f"Unexpected sleep: {seconds}")

        monkeypatch.setattr(asyncio, "sleep", unexpected_sleep)
        await hooks._reach_async(FaultHookName.ORCHESTRATOR_STEP_END, {"rollout_id": rollout_id})
        assert read_frozen_rollout_id(path) is None

    @pytest.mark.parametrize("owner", [FaultHookOwner.TRAINER_ACTOR, FaultHookOwner.TRAINER_CONTROLLER])
    async def test_other_owners_do_not_arm_the_orchestrator_freeze(self, configure_hooks: Callable[..., _FaultHookController], owner: FaultHookOwner) -> None:
        """A shared plan must not freeze a cell when it names the orchestration loop."""
        hooks = configure_hooks([_SLEEP], owner=owner)
        await hooks._reach_async(FaultHookName.ORCHESTRATOR_STEP_END, {"rollout_id": 2})

    def test_orchestrator_requests_cannot_name_a_cell(self) -> None:
        """A cell target on an orchestration hook must be rejected instead of ignored."""
        with pytest.raises(ValidationError, match="outside any cell"):
            FaultHookRequest(request_id="invalid", hook_name=FaultHookName.ORCHESTRATOR_STEP_END,
                             action=SleepForeverAction(), target=DeclaredFaultHookTarget(cell_id="trainer-engine-actor-0"))


class TestParkableLoops:
    @pytest.mark.parametrize("script,interval,model_id,message", [
        ("train_async.py", 1, None, "train_async.py has already started"),
        ("train.py", 2, None, "update-weights-interval"),
        ("train.py", 1, "solver", "several policies"),
    ])
    async def test_unsupported_loop_fails_before_writing_a_sentinel(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str, interval: int, model_id: str | None, message: str) -> None:
        """Freezing an incompatible loop must fail before claiming that the run has parked."""
        monkeypatch.setattr(sys, "argv", [f"/miles/{script}"])
        path = tmp_path / "plan.json"
        args = SimpleNamespace(ci_fault_hooks_path=str(path), update_weights_interval=interval, fully_async=False)
        with pytest.raises(AssertionError, match=message):
            await SleepForeverAction()(context=FaultHookContext(rollout_id=2, trainer_model_id=model_id), resources=FaultHookResources(args=args))
        assert read_frozen_rollout_id(path) is None

    def test_the_parkable_script_is_the_default_launch_script(self) -> None:
        """The freeze guard and launch harness must agree on the synchronous script."""
        assert PARKABLE_TRAIN_SCRIPT == DEFAULT_TRAIN_SCRIPT
