import random
from datetime import datetime, timezone

import pytest
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.config import SoakCellPolicy, SoakPolicy
from tests.utils.soak.core import SoakActionScheduler
from tests.utils.soak.entrypoint import FaultInjectorHandle
from tests.utils.soak.fault_forms import InjectFaultForm
from tests.utils.soak.hook_fault_form import HookFaultForm
from tests.utils.soak.state import SoakActionRequest, SoakActionRequestedEvent, SoakObservation, SoakScheduleEvent

from miles.utils.audit_utils.event_logger.models import WeightUpdateAssignmentEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget


class TestHookScheduling:
    @pytest.mark.parametrize("case", ["ready", "missing", "stale", "reserved", "missing_assignment", "changed_sender"])
    def test_all_target_hook_requires_the_complete_unreserved_fleet(self, case: str) -> None:
        """A batch cannot silently omit missing, stale or already targeted inference cells."""
        cells = [typed_cell(f"rollout-{index}", "rollout") for index in range(4)]
        cells.extend(typed_cell(f"actor-{index}", "actor") for index in range(2))
        identities = {
            cell["metadata"]["name"]: FaultTarget(
                cell_id=cell["metadata"]["name"], sub_index=0, workers_hash=cell["status"]["workers_hash"]
            )
            for cell in cells
        }
        form = HookFaultForm(
            base_url="http://control",
            failure_mode=FailureMode.SIGKILL,
            hook="trainer_before_weight_send",
            victim_form=InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL),
            all_targets=True,
        )
        scheduler = SoakActionScheduler(
            rng=random.Random(0),
            mean_intervals={"rollout": 1},
            forms={"rollout": [form]},
            policy=SoakPolicy(
                max_concurrent_actions=2,
                cell_policies={"rollout": SoakCellPolicy(expected_cells=4, min_survivors=2)},
            ),
        )
        assignment = WeightUpdateAssignmentEvent(
            timestamp=datetime(2026, 9, 11, tzinfo=timezone.utc),
            source=TrainerControllerProcessIdentity(trainer_id="actor"),
            update_id="previous-update",
            candidate_version=7,
            trainer_incarnations={f"actor-{i}": "generation-0" for i in range(2)},
            targets_by_trainer={
                f"actor-{i}": {f"rollout-{j}": "stale" if case == "stale" else "generation-0" for j in [i, i + 2]}
                for i in range(2)
            },
        )
        if case == "missing":
            cells.pop(0)
        elif case == "changed_sender":
            assignment.trainer_incarnations["actor-0"] = "replaced"
        events = [
            SoakScheduleEvent(due_of_type={"rollout": 0}),
            SoakObservation(
                cells=cells,
                fault_targets=identities,
                training_events=[] if case == "missing_assignment" else [assignment],
            ),
        ]
        if case == "reserved":
            events.append(
                SoakActionRequestedEvent(
                    request=SoakActionRequest(target=cells[0], form_name="inject_fault:sigkill", harms_cell=True)
                )
            )
        request = scheduler.choose(events=events, now=1)

        if case == "ready":
            assert request is not None
            victims = [request, *request.additional_requests]
            assert request.hook_trigger is not None
            expected = assignment.targets_by_trainer[request.hook_trigger.cell_id]
            assert {child.target["metadata"]["name"] for child in victims} == set(expected)
            assert len({child.request_id for child in victims}) == 2
            assert all(child.fault_target == identities[child.target["metadata"]["name"]] for child in victims)
        else:
            assert request is None

    @pytest.mark.parametrize("remote", [False, True])
    def test_default_observer_collects_hook_process_identities(self, remote: bool) -> None:
        """The observer must collect trainer triggers even when only rollout faults are scheduled."""
        kind = "rollout" if remote else "actor"
        form = HookFaultForm(
            base_url="http://control",
            failure_mode=FailureMode.SIGKILL,
            hook="trainer_before_weight_send",
            victim_form=(
                InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL) if remote else None
            ),
        )
        handle = FaultInjectorHandle(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type={kind: 1},
            cell_fault_forms={kind: [form]},
        )

        assert handle._runner is not None
        observer = handle._runner._observer
        expected = {"actor", "rollout"} if remote else {"actor"}
        assert observer.cell_types == expected
        assert observer.fault_target_cell_types == expected

    @pytest.mark.parametrize("remote", [False, True])
    @pytest.mark.parametrize("has_identity", [False, True])
    def test_hook_request_pins_every_required_process(self, remote: bool, has_identity: bool) -> None:
        """Neither a local hook nor a remote trigger can be scheduled without its observed incarnation."""
        victim = InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL)
        form = HookFaultForm(
            base_url="http://control",
            failure_mode=FailureMode.SIGKILL,
            hook="trainer_before_weight_send",
            victim_form=victim if remote else None,
        )
        kind = "rollout" if remote else "actor"
        cells = [typed_cell(f"{kind}-{i}", kind) for i in range(2)]
        if remote:
            cells.append(typed_cell("actor-0", "actor"))
        identities = {
            cell["metadata"]["name"]: FaultTarget(
                cell_id=cell["metadata"]["name"], sub_index=0, workers_hash=cell["status"]["workers_hash"]
            )
            for cell in cells
            if has_identity or cell["metadata"]["name"].startswith("rollout-")
        }
        scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={kind: 1}, forms={kind: [form]})
        request = scheduler.choose(
            events=[SoakScheduleEvent(due_of_type={kind: 0}), SoakObservation(cells=cells, fault_targets=identities)],
            now=1,
        )

        if not has_identity:
            assert request is None
        else:
            assert request is not None
            assert request.fault_target == identities[request.target["metadata"]["name"]]
            assert request.hook_trigger == (identities["actor-0"] if remote else None)

    def test_pending_remote_trigger_is_not_selected_as_a_local_victim(self) -> None:
        """A local fault cannot destroy the trainer still serving another action's trigger."""
        actors = [typed_cell(f"actor-{i}", "actor") for i in range(2)]
        identities = {
            cell["metadata"]["name"]: FaultTarget(
                cell_id=cell["metadata"]["name"], sub_index=0, workers_hash=cell["status"]["workers_hash"]
            )
            for cell in actors
        }
        pending = SoakActionRequest(
            target=typed_cell("rollout-0", "rollout"),
            form_name="remote_hook:pending",
            harms_cell=True,
            hook_trigger=identities["actor-0"],
        )
        form = HookFaultForm(
            base_url="http://control", failure_mode=FailureMode.SIGKILL, hook="trainer_before_all_gather"
        )
        scheduler = SoakActionScheduler(
            rng=random.Random(0),
            mean_intervals={"actor": 1},
            forms={"actor": [form]},
            policy=SoakPolicy(max_concurrent_actions=2),
        )
        request = scheduler.choose(
            events=[
                SoakScheduleEvent(due_of_type={"actor": 0}),
                SoakObservation(cells=actors, fault_targets=identities),
                SoakActionRequestedEvent(request=pending),
            ],
            now=1,
        )

        assert request is not None
        assert request.fault_target == identities["actor-1"]
