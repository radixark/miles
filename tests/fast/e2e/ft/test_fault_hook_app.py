from collections.abc import Callable
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import app as comparison_app
from tests.e2e.ft.conftest_ft import fault_hook_app
from tests.e2e.ft.conftest_ft.fault_hook_events import (
    assert_fault_hooks_fired,
    assert_weight_update_failures,
    assert_weight_updates_published,
)
from tests.e2e.ft.conftest_ft.modes import MODES, FTTestMode
from tests.fast.e2e.ft.event_fakes import _hook, _update, _write_events
from tests.fast.e2e.scenario_harness import ScenarioHarness, parse_fault_tolerance_args
from tests.utils.soak.ft.checkers.reconfigure import ReconfigureInfo

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.test_utils.fault_injector.actions.process import KillProcessAction
from miles.utils.test_utils.fault_injector.models import (
    DeclaredFaultHookTarget,
    FaultHookName,
    FaultHookRequest,
    FaultHookStatus,
)
from miles.utils.test_utils.fault_injector.static_source import compute_fault_hooks_arg, read_declared_fault_hooks

_MODE = "kill_train__dp2_tp2"
_NUM_ROLLOUTS = 6
_HOOKS = [
    FaultHookRequest(
        request_id="kill_at_2",
        hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER,
        action=KillProcessAction(),
        target=DeclaredFaultHookTarget(cell_id="trainer-engine-actor-00001", rank=0),
        rollout_id=2,
    )
]
_HEAL = ReconfigureInfo(rollout_id=3, src_cell_index=0, healed_cell_indices=[1], alive_cell_indices_after=[0, 1])


def _create_run_ci(harness: ScenarioHarness, *, ft_components: tuple[str, ...] = ("train",)) -> Callable[[str], None]:
    def build_fault_hooks(mode: FTTestMode, config: ExecuteTrainConfig) -> list[FaultHookRequest]:
        harness.checks.append(("build_fault_hooks", (mode, config), {}))
        return _HOOKS

    _, run_ci = fault_hook_app.create_fault_hook_comparison_app(
        test_name="fault_hook_app",
        num_rollouts=_NUM_ROLLOUTS,
        ft_components=ft_components,
        extra_train_args="--update-weights-timeout 120.0 ",
        build_fault_hooks=build_fault_hooks,
        expected_target_reconfigures=lambda mode: [_HEAL],
        assert_target_events=harness.recorder("target_events"),
    )
    return run_ci


@pytest.fixture
def harness(scenario_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch) -> ScenarioHarness:
    monkeypatch.setattr(comparison_app, "prepare", scenario_harness.record_prepare)
    monkeypatch.setattr(fault_hook_app, "compare_deterministic_sides", scenario_harness.recorder("compare"))
    monkeypatch.setattr(fault_hook_app, "assert_fault_hooks_fired", scenario_harness.recorder("fired"))
    return scenario_harness


class TestTheTwoSidesOfTheFaultHookComparison:
    def test_only_the_target_side_is_launched_with_the_declared_hooks(self, harness: ScenarioHarness) -> None:
        """A baseline carrying the hooks would compare a faulted run against a faulted run."""
        _create_run_ci(harness)(_MODE)

        baseline, target = (parse_fault_tolerance_args(launch.request.train_args) for launch in harness.launches)
        assert baseline.namespace.ci_fault_hooks is None
        assert read_declared_fault_hooks(baseline.namespace) == []
        assert read_declared_fault_hooks(target.namespace) == _HOOKS
        assert target.namespace.ci_fault_hooks_path is None

    def test_the_sides_differ_only_by_their_dumps_and_the_hooks(self, harness: ScenarioHarness) -> None:
        """Any other difference between the launches would make the bitwise comparison meaningless."""
        _create_run_ci(harness)(_MODE)

        baseline, target = harness.launches
        hooks = compute_fault_hooks_arg(_HOOKS)
        assert "/baseline" in baseline.request.train_args
        assert target.request.train_args.count(hooks) == 1
        assert target.request.train_args.replace(hooks, "") == baseline.request.train_args.replace(
            "/baseline", "/target"
        )

    def test_both_sides_run_the_deterministic_p2p_launch_with_the_scenario_arguments(
        self, harness: ScenarioHarness
    ) -> None:
        """The hooks sit on the p2p weight path, so a broadcast or nondeterministic launch would never reach them."""
        _create_run_ci(harness)(_MODE)

        for launch in harness.launches:
            parsed = parse_fault_tolerance_args(launch.request.train_args)
            assert parsed.namespace.update_weight_transfer_mode == "p2p"
            assert parsed.ft_components == ["train"]
            assert parsed.mini_ft_controller_enable
            assert "--debug-deterministic-collective" in launch.argv
            assert launch.value_of("--update-weights-timeout") == "120.0"
            assert launch.value_of("--num-rollout") == str(_NUM_ROLLOUTS)

    def test_each_side_runs_under_its_own_submission(self, harness: ScenarioHarness) -> None:
        """A shared ray submission would let the target stop the baseline it is compared to."""
        _create_run_ci(harness)(_MODE)

        baseline, target = (launch.config.ray_submission_id for launch in harness.launches)
        assert baseline != target
        assert baseline.startswith("miles-soak-") and target.startswith("miles-soak-")

    def test_a_mode_with_other_fault_tolerant_components_is_refused_before_launching(
        self, harness: ScenarioHarness
    ) -> None:
        """Faults declared for one component must not run in a mode that enables ft elsewhere."""
        with pytest.raises(AssertionError, match="must enable ft on exactly those"):
            _create_run_ci(harness, ft_components=("rollout",))(_MODE)

        assert harness.launches == []


class TestWhatTheFaultHookComparisonIsJudgedBy:
    def test_the_sides_are_compared_then_the_hooks_and_target_events_are_checked(
        self, harness: ScenarioHarness
    ) -> None:
        """The target must match bitwise with the declared healing, have fired every hook and pass its own events."""
        _create_run_ci(harness)(_MODE)

        assert [name for name in harness.checker_names if name != "build_fault_hooks"] == [
            "compare",
            "fired",
            "target_events",
        ]
        ((_, compare_kwargs),) = harness.calls_of("compare")
        root = Path(compare_kwargs["target_dir"]).parent
        assert compare_kwargs == {
            "baseline_dir": f"{root}/baseline",
            "target_dir": f"{root}/target",
            "min_trained_rollouts": 2,
            "expected_target_reconfigures": [_HEAL],
        }
        assert harness.calls_of("fired") == [((root / "target" / EVENTS_DIRNAME,), {"request_ids": ["kill_at_2"]})]
        assert harness.calls_of("target_events") == [((root / "target" / EVENTS_DIRNAME, MODES[_MODE]), {})]

    def test_a_failed_launch_is_raised_and_never_judged(self, harness: ScenarioHarness) -> None:
        """A crashed side leaves partial dumps and must not be graded as a comparison."""
        harness.launch_error = RuntimeError("the job exited 1")

        with pytest.raises(RuntimeError, match="the job exited 1"):
            _create_run_ci(harness)(_MODE)

        assert harness.calls_of("compare") == harness.calls_of("fired") == []


class TestAssertFaultHooksFired:
    def test_a_request_whose_last_record_is_fired_passes(self, tmp_path: Path) -> None:
        """A request that went from pending to fired took the fault it declared."""
        _write_events(tmp_path, [_hook("a", FaultHookStatus.PENDING), _hook("a", FaultHookStatus.FIRED)])

        assert_fault_hooks_fired(tmp_path, request_ids=["a"])

    @pytest.mark.parametrize(
        "statuses",
        [
            [FaultHookStatus.PENDING],
            [FaultHookStatus.PENDING, FaultHookStatus.SCHEDULED],
            [FaultHookStatus.PENDING, FaultHookStatus.CLEARED],
            [FaultHookStatus.PENDING, FaultHookStatus.EXPIRED],
            [FaultHookStatus.PENDING, FaultHookStatus.FIRED, FaultHookStatus.FAILED],
        ],
    )
    def test_a_request_that_did_not_end_fired_fails(self, tmp_path: Path, statuses: list[FaultHookStatus]) -> None:
        """Never reached, still waiting, withdrawn, expired or failed all mean the fault was not taken."""
        _write_events(tmp_path, [_hook("a", status) for status in statuses])

        with pytest.raises(AssertionError, match=f"ended as {statuses[-1]}"):
            assert_fault_hooks_fired(tmp_path, request_ids=["a"])

    def test_another_request_firing_does_not_stand_in(self, tmp_path: Path) -> None:
        """Each declared request must have its own FIRED record."""
        _write_events(tmp_path, [_hook("a", FaultHookStatus.FIRED)])

        with pytest.raises(AssertionError, match="Declared fault hook b ended as None"):
            assert_fault_hooks_fired(tmp_path, request_ids=["a", "b"])


class TestAssertWeightUpdatesPublished:
    def test_every_named_rollout_that_published_passes(self, tmp_path: Path) -> None:
        """Faulted rollouts that still published pass, whatever unnamed rollouts did."""
        _write_events(
            tmp_path,
            [_update(rollout_id=1), _update(rollout_id=2, published_version=None), _update(rollout_id=3)],
        )

        assert_weight_updates_published(tmp_path, rollout_ids=[1, 3])

    def test_a_named_rollout_without_any_update_fails(self, tmp_path: Path) -> None:
        """A faulted rollout that never recorded an update proves nothing about publication."""
        _write_events(tmp_path, [_update(rollout_id=1)])

        with pytest.raises(AssertionError, match="Rollout 3 recorded no weight update"):
            assert_weight_updates_published(tmp_path, rollout_ids=[1, 3])

    def test_any_unpublished_update_of_a_named_rollout_fails(self, tmp_path: Path) -> None:
        """One published retry must not hide an update of the same rollout that published nothing."""
        _write_events(
            tmp_path,
            [_update(rollout_id=3), _update(rollout_id=3, published_version=None, failed=["rollout-0"])],
        )

        with pytest.raises(AssertionError, match="Rollout 3 published no weight version"):
            assert_weight_updates_published(tmp_path, rollout_ids=[3])


class TestAssertWeightUpdateFailures:
    def test_the_exact_failed_cells_of_the_faulted_rollout_pass_in_any_order(self, tmp_path: Path) -> None:
        """The faulted rollout fails exactly its victims and every other rollout fails none."""
        _write_events(
            tmp_path,
            [
                _update(rollout_id=2),
                _update(rollout_id=3, failed=["rollout-1", "rollout-0"]),
                _update(rollout_id=None, failed=["rollout-5"]),
                _update(rollout_id=4),
            ],
        )

        assert_weight_update_failures(tmp_path, failed_cell_ids_of_rollout_id={3: ["rollout-0", "rollout-1"]})

    @pytest.mark.parametrize(
        "events,match",
        [
            ([_update(rollout_id=2)], "Rollouts \\[3\\] recorded no weight update"),
            ([_update(rollout_id=3)], "Rollout 3 failed to update cells \\[\\]"),
            ([_update(rollout_id=3, failed=["rollout-0", "rollout-1"])], "Rollout 3 failed to update cells"),
            ([_update(rollout_id=3, failed=["rollout-1"])], "Rollout 3 failed to update cells"),
            (
                [_update(rollout_id=3, failed=["rollout-0"]), _update(rollout_id=4, failed=["rollout-0"])],
                "Rollout 4 failed to update cells",
            ),
        ],
    )
    def test_any_other_failure_pattern_fails(self, tmp_path: Path, events: list[object], match: str) -> None:
        """A missing, spared, over-broad, misdirected or lingering failure must each be rejected."""
        _write_events(tmp_path, events)

        with pytest.raises(AssertionError, match=match):
            assert_weight_update_failures(tmp_path, failed_cell_ids_of_rollout_id={3: ["rollout-0"]})
