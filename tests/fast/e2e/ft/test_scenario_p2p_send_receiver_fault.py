from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import app as comparison_app
from tests.e2e.ft.conftest_ft import fault_hook_app, scenario_p2p_send_receiver_fault
from tests.e2e.ft.conftest_ft.modes import MODES
from tests.fast.e2e.ft.event_fakes import _update, _write_events
from tests.fast.e2e.scenario_harness import ScenarioHarness, parse_fault_tolerance_args

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.test_utils.fault_injector.actions.process import KillProcessAction
from miles.utils.test_utils.fault_injector.actions.remote import ApiServerFaultAction
from miles.utils.test_utils.fault_injector.controller import _filter_fault_hooks
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookOwner
from miles.utils.test_utils.fault_injector.static_source import read_declared_fault_hooks
from miles.utils.workers.types import ClusterBackend

_MODE = "kill_rollout__dp2_tp2"
_SENDER = "trainer-engine-actor-00000"


@pytest.fixture
def harness(scenario_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch) -> ScenarioHarness:
    monkeypatch.setattr(comparison_app, "prepare", scenario_harness.record_prepare)
    monkeypatch.setattr(fault_hook_app, "compare_deterministic_sides", scenario_harness.recorder("compare"))
    monkeypatch.setattr(fault_hook_app, "assert_fault_hooks_fired", scenario_harness.recorder("fired"))
    monkeypatch.setattr(
        scenario_p2p_send_receiver_fault, "assert_weight_update_failures", scenario_harness.recorder("failures")
    )
    return scenario_harness


class TestTheReceiverFaultPlan:
    def test_the_first_sender_kills_the_first_receiver_through_the_api_server_after_the_delay(self) -> None:
        """At rollout 3 the sender's BEFORE_SEND must ask the api server to kill receiver rank 0 after 50 ms."""
        config = ExecuteTrainConfig()
        [request] = scenario_p2p_send_receiver_fault._build_fault_hooks(MODES[_MODE], config)

        assert request.hook_name is FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND
        assert (request.target.cell_id, request.target.rank) == (_SENDER, 0)
        assert (request.rollout_id, request.delay_ms) == (3, 50.0)
        assert request.action == ApiServerFaultAction(
            base_url="http://localhost:18080",
            cell_id=scenario_p2p_send_receiver_fault._receiver_cell_id(config),
            rank=0,
            inner=KillProcessAction(),
        )

    def test_the_receiver_is_the_first_cell_of_the_runs_engine_pool(self) -> None:
        """The receiver name must follow the deployment so a named instance kills its own engine."""
        default = scenario_p2p_send_receiver_fault._receiver_cell_id(ExecuteTrainConfig())
        named = scenario_p2p_send_receiver_fault._receiver_cell_id(ExecuteTrainConfig(deploy_instance_id="blue"))

        assert (default, named) == ("inference-engine-all-0-0-00000", "inference-engine-blue-0-0-00000")

    def test_a_kubernetes_backend_is_refused(self) -> None:
        """Only the ray backend serves the api server on the sending trainer's host."""
        with pytest.raises(AssertionError, match="only the ray backend"):
            scenario_p2p_send_receiver_fault._build_fault_hooks(
                MODES[_MODE], ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES)
            )

    @pytest.mark.parametrize(
        "cell_id,rank,expected", [(_SENDER, 0, 1), (_SENDER, 1, 0), ("trainer-engine-actor-00001", 0, 0)]
    )
    def test_the_launched_plan_arms_only_the_sending_rank(
        self, harness: ScenarioHarness, cell_id: str, rank: int, expected: int
    ) -> None:
        """Parsed from the real target launch, only rank 0 of the first sender may carry the hook."""
        scenario_p2p_send_receiver_fault.run_ci(_MODE)

        target = parse_fault_tolerance_args(harness.launches[1].request.train_args).namespace
        armed = _filter_fault_hooks(
            read_declared_fault_hooks(target), owner=FaultHookOwner.TRAINER_ACTOR, cell_id=cell_id, rank=rank
        )
        assert len(armed) == expected


class TestTheReceiverFaultRun:
    def test_the_run_is_rollout_fault_tolerant_and_never_reconfigures_the_trainer(
        self, harness: ScenarioHarness
    ) -> None:
        """A receiver fault must be absorbed by rollout ft while the trainer keeps every cell."""
        scenario_p2p_send_receiver_fault.run_ci(_MODE)

        for launch in harness.launches:
            assert parse_fault_tolerance_args(launch.request.train_args).ft_components == ["rollout"]
        ((_, compare_kwargs),) = harness.calls_of("compare")
        assert compare_kwargs["expected_target_reconfigures"] == []
        ((_, fired_kwargs),) = harness.calls_of("fired")
        assert fired_kwargs["request_ids"] == ["kill_receiver_before_send_at_3"]

    def test_only_the_receiver_fails_and_only_at_the_faulted_rollout(self, harness: ScenarioHarness) -> None:
        """The target check must demand exactly the receiver failing at rollout 3."""
        scenario_p2p_send_receiver_fault.run_ci(_MODE)

        ((_, kwargs),) = harness.calls_of("failures")
        receiver = scenario_p2p_send_receiver_fault._receiver_cell_id(ExecuteTrainConfig())
        assert kwargs["failed_cell_ids_of_rollout_id"] == {3: [receiver]}

    @pytest.mark.parametrize(
        "failed_at_3,failed_at_4",
        [([], []), (["other"], []), (["RECEIVER"], ["RECEIVER"])],
    )
    def test_the_real_target_check_rejects_any_other_failure(
        self, tmp_path: Path, failed_at_3: list[str], failed_at_4: list[str]
    ) -> None:
        """A spared receiver, another victim or a lingering failure must fail the scenario."""
        receiver = scenario_p2p_send_receiver_fault._receiver_cell_id(ExecuteTrainConfig())
        _write_events(
            tmp_path,
            [
                _update(rollout_id=3, failed=[receiver if c == "RECEIVER" else c for c in failed_at_3]),
                _update(rollout_id=4, failed=[receiver if c == "RECEIVER" else c for c in failed_at_4]),
            ],
        )

        with pytest.raises(AssertionError, match="failed to update cells"):
            scenario_p2p_send_receiver_fault._assert_target_events(tmp_path, MODES[_MODE])
