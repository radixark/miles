import random

import pytest
from tests.fast.utils.soak.utils import StubFaultForm, typed_cell
from tests.utils.soak.config import SoakCellPolicy, SoakPolicy
from tests.utils.soak.core import SoakActionScheduler
from tests.utils.soak.policy import eligible_cells
from tests.utils.soak.state import (
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakObservation,
    SoakScheduleEvent,
)


@pytest.mark.parametrize("allow_during_recovery", [False, True])
def test_replacement_can_be_faulted_before_recovery_only_when_the_scenario_allows_it(
    allow_during_recovery: bool,
) -> None:
    """A second fault can hit a pending replacement while the other healthy cell survives."""
    original = typed_cell("actor-0", "actor")
    survivor = typed_cell("actor-1", "actor")
    first = SoakActionRequest(target=original, form_name="kill", harms_cell=True)
    replacement = {
        **original,
        "status": {**original["status"], "workers_hash": "replacement", "phase": "Pending"},
    }
    policy = SoakPolicy(
        cell_policies={"actor": SoakCellPolicy(expected_cells=2, allow_during_recovery=allow_during_recovery)}
    )
    forms = {"actor": [StubFaultForm("kill")]}
    scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1}, forms=forms, policy=policy)
    events = [
        SoakScheduleEvent(due_of_type={"actor": 0}, policy=policy),
        SoakObservation(cells=[original, survivor]),
        SoakActionRequestedEvent(request=first),
        SoakActionAppliedEvent(request_id=first.request_id, evidence={"exited_pids": [42]}),
        SoakActionResultEvent(request_id=first.request_id, returned=True),
        SoakObservation(cells=[replacement, survivor]),
    ]
    second = scheduler.choose(events=events, now=100)
    if allow_during_recovery:
        assert second is not None and second.target == replacement
        assert second.request_id != first.request_id
    else:
        assert second is None


def test_stale_health_of_a_submitted_incarnation_cannot_spend_the_last_survivor() -> None:
    """A requested fault reserves its old incarnation until a healthy replacement is observed."""
    cells = [typed_cell(f"actor-{i}", "actor") for i in range(2)]
    request = SoakActionRequest(target=cells[0], form_name="kill", harms_cell=True)
    events = [SoakActionRequestedEvent(request=request)]
    policy = SoakCellPolicy(expected_cells=2, allow_during_recovery=False)
    assert eligible_cells(cells=cells, events=events, policy=policy, harms_cell=True) == []
    replacement = {**cells[0], "status": {**cells[0]["status"], "workers_hash": "replacement"}}
    recovered = [replacement, cells[1]]
    assert eligible_cells(cells=recovered, events=events, policy=policy, harms_cell=True) == recovered


def test_recovery_gate_uses_declared_topology_and_overlap_is_an_explicit_choice() -> None:
    """Two observed cells do not prove a declared three-cell topology has recovered."""
    cells = [typed_cell(f"actor-{i}", "actor") for i in range(2)]
    guarded = SoakCellPolicy(expected_cells=3, allow_during_recovery=False)
    overlap = guarded.model_copy(update={"allow_during_recovery": True})
    assert eligible_cells(cells=cells, events=[], policy=guarded, harms_cell=True) == []
    assert eligible_cells(cells=cells, events=[], policy=overlap, harms_cell=True) == cells
    with pytest.raises(ValueError, match="explicit expected"):
        SoakCellPolicy(allow_during_recovery=False)


def test_zero_survivor_policy_can_target_one_cell_but_unready_rollout_is_not_a_survivor() -> None:
    """Survivor budgets are configurable and an unregistered engine cannot protect another target."""
    cell = typed_cell("rollout-0", "rollout")
    assert eligible_cells(cells=[cell], events=[], policy=SoakCellPolicy(min_survivors=0), harms_cell=True) == [cell]
    unready = typed_cell("rollout-1", "rollout", serving=False)
    assert (
        eligible_cells(
            cells=[cell, unready], events=[], policy=SoakCellPolicy(require_ready_target=True), harms_cell=True
        )
        == []
    )


def test_scheduler_allows_configured_overlap_and_reserves_each_pending_target() -> None:
    """Two concurrent actions target distinct incarnations while preserving the last healthy cell."""
    forms = {"actor": [StubFaultForm("kill")]}
    policy = SoakPolicy(max_concurrent_actions=2)
    scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1}, forms=forms, policy=policy)
    cells = [typed_cell(f"actor-{i}", "actor") for i in range(3)]
    first = SoakActionRequest(target=cells[0], form_name="kill", harms_cell=True)
    events = [
        SoakScheduleEvent(due_of_type={"actor": 0}, policy=policy),
        SoakObservation(cells=cells),
        SoakActionRequestedEvent(request=first),
    ]
    second = scheduler.choose(events=events, now=100)
    assert second is not None and second.target != first.target
    assert second.next_due_at > 100
    events.append(SoakActionRequestedEvent(request=second))
    assert scheduler.choose(events=events, now=200) is None
