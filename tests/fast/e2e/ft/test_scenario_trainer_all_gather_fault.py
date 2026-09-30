from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import app as comparison_app
from tests.e2e.ft.conftest_ft import fault_hook_app, scenario_trainer_all_gather_fault
from tests.e2e.ft.conftest_ft.modes import MODES
from tests.fast.e2e.ft.event_fakes import _update, _write_events
from tests.fast.e2e.scenario_harness import ScenarioHarness, parse_fault_tolerance_args
from tests.utils.soak.ft.checkers.reconfigure import ReconfigureInfo

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.test_utils.fault_injector.actions.process import (
    DeadlockThreadAction,
    KillProcessAction,
    StopProcessAction,
)
from miles.utils.test_utils.fault_injector.controller import _filter_fault_hooks
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookOwner
from miles.utils.test_utils.fault_injector.static_source import read_declared_fault_hooks

_MODE = "kill_train__dp2_tp2"
_LAST_CELL = "trainer-engine-actor-00001"


@pytest.fixture
def harness(scenario_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch) -> ScenarioHarness:
    monkeypatch.setattr(comparison_app, "prepare", scenario_harness.record_prepare)
    monkeypatch.setattr(fault_hook_app, "compare_deterministic_sides", scenario_harness.recorder("compare"))
    monkeypatch.setattr(fault_hook_app, "assert_fault_hooks_fired", scenario_harness.recorder("fired"))
    monkeypatch.setattr(
        scenario_trainer_all_gather_fault, "assert_weight_updates_published", scenario_harness.recorder("published")
    )
    return scenario_harness


class TestTheAllGatherFaultPlan:
    def test_three_faults_hit_the_last_trainer_cells_first_rank_before_the_all_gather(self) -> None:
        """Kill, stop and deadlock must each strike rank 0 of the last cell at its own rollout."""
        requests = scenario_trainer_all_gather_fault._build_fault_hooks(MODES[_MODE], ExecuteTrainConfig())

        assert [(r.rollout_id, r.action) for r in requests] == [
            (1, KillProcessAction()),
            (3, StopProcessAction()),
            (5, DeadlockThreadAction()),
        ]
        for request in requests:
            assert request.hook_name is FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER
            assert (request.target.cell_id, request.target.rank) == (_LAST_CELL, 0)
            assert (request.attempt, request.weight_version, request.delay_ms, request.lifetime_seconds) == (
                None,
                None,
                0.0,
                None,
            )
        assert len({r.request_id for r in requests}) == 3

    def test_each_fault_heals_the_victim_at_the_next_rollout(self) -> None:
        """The victim cell must rejoin from cell 0 right after each faulted update, with every cell alive."""
        assert scenario_trainer_all_gather_fault._expected_reconfigures(MODES[_MODE]) == [
            ReconfigureInfo(rollout_id=r, src_cell_index=0, healed_cell_indices=[1], alive_cell_indices_after=[0, 1])
            for r in (2, 4, 6)
        ]

    def test_every_fault_heals_before_the_run_ends(self) -> None:
        """A fault at the last rollout could never be healed or compared."""
        assert max(scenario_trainer_all_gather_fault.FAULT_ACTION_OF_ROLLOUT_ID) + 1 < (
            scenario_trainer_all_gather_fault.NUM_ROLLOUTS
        )

    @pytest.mark.parametrize(
        "owner,cell_id,rank,expected",
        [
            (FaultHookOwner.TRAINER_ACTOR, _LAST_CELL, 0, 3),
            (FaultHookOwner.TRAINER_ACTOR, _LAST_CELL, 1, 0),
            (FaultHookOwner.TRAINER_ACTOR, "trainer-engine-actor-00000", 0, 0),
            (FaultHookOwner.TRAINER_CONTROLLER, None, None, 0),
        ],
    )
    def test_the_launched_plan_arms_only_the_victim_rank(
        self, harness: ScenarioHarness, owner: FaultHookOwner, cell_id: str | None, rank: int | None, expected: int
    ) -> None:
        """Parsed from the real target launch, the plan must arm the victim rank and nothing else."""
        scenario_trainer_all_gather_fault.run_ci(_MODE)

        target = parse_fault_tolerance_args(harness.launches[1].request.train_args).namespace
        armed = _filter_fault_hooks(read_declared_fault_hooks(target), owner=owner, cell_id=cell_id, rank=rank)
        assert len(armed) == expected


class TestTheAllGatherFaultRun:
    def test_the_target_waits_on_updates_with_the_bounded_timeout_and_trainer_ft(
        self, harness: ScenarioHarness
    ) -> None:
        """A stopped or deadlocked sender must time the update out and heal through trainer fault tolerance."""
        scenario_trainer_all_gather_fault.run_ci(_MODE)

        for launch in harness.launches:
            parsed = parse_fault_tolerance_args(launch.request.train_args)
            assert parsed.ft_components == ["train"]
            assert parsed.namespace.update_weights_timeout == 120.0

    def test_the_target_is_judged_on_the_declared_healing_and_every_faulted_rollout_publishing(
        self, harness: ScenarioHarness
    ) -> None:
        """Every faulted rollout must still publish and the target must heal exactly as planned."""
        scenario_trainer_all_gather_fault.run_ci(_MODE)

        ((_, compare_kwargs),) = harness.calls_of("compare")
        assert compare_kwargs[
            "expected_target_reconfigures"
        ] == scenario_trainer_all_gather_fault._expected_reconfigures(MODES[_MODE])
        ((_, fired_kwargs),) = harness.calls_of("fired")
        assert fired_kwargs["request_ids"] == [
            "kill_process_before_all_gather_at_1",
            "stop_process_before_all_gather_at_3",
            "deadlock_thread_before_all_gather_at_5",
        ]
        ((published_args, published_kwargs),) = harness.calls_of("published")
        assert published_args[0] == Path(compare_kwargs["target_dir"]) / "events"
        assert list(published_kwargs["rollout_ids"]) == [1, 3, 5]

    def test_the_real_target_check_rejects_a_faulted_rollout_that_published_nothing(self, tmp_path: Path) -> None:
        """Healing must not come at the cost of a rollout that served stale weights."""
        _write_events(
            tmp_path, [_update(rollout_id=1), _update(rollout_id=3, published_version=None), _update(rollout_id=5)]
        )

        with pytest.raises(AssertionError, match="Rollout 3 published no weight version"):
            scenario_trainer_all_gather_fault._assert_target_events(tmp_path, MODES[_MODE])
