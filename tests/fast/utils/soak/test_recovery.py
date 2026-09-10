from datetime import datetime, timedelta, timezone

import pytest
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.recovery import compute_recovery_episodes
from tests.utils.soak.state import SoakActionAppliedEvent, SoakActionRequest, SoakActionRequestedEvent, SoakObservation

from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


@pytest.mark.parametrize("observe_after_receipt", [False, True])
def test_late_receipt_requires_a_new_observation_after_confirmation(observe_after_receipt: bool) -> None:
    """A healthy snapshot taken before a late effect receipt cannot close the recovery."""
    start = datetime(2026, 9, 11, tzinfo=timezone.utc)
    original = typed_cell("rollout-0", "rollout")
    replacement = {**original, "status": {**original["status"], "workers_hash": "replacement"}}
    request = SoakActionRequest(target=original, form_name="kill", harms_cell=True)
    events = [
        SoakActionRequestedEvent(timestamp=start, request=request),
        SoakObservation(timestamp=start + timedelta(seconds=1), cells=[replacement]),
        SoakActionAppliedEvent(timestamp=start + timedelta(seconds=2), request_id=request.request_id, evidence={}),
    ]
    if observe_after_receipt:
        events.append(SoakObservation(timestamp=start + timedelta(seconds=3), cells=[replacement]))
    (episode,) = compute_recovery_episodes(events)
    assert episode.recovered_incarnation == ("replacement" if observe_after_receipt else None)


def test_a_second_fault_cannot_consume_the_first_completed_recovery() -> None:
    """A fault after a completed episode opens new debt even if the old healthy status persists."""
    start = datetime(2026, 9, 11, tzinfo=timezone.utc)
    original = typed_cell("rollout-0", "rollout")
    replacement = {**original, "status": {**original["status"], "workers_hash": "replacement"}}
    first = SoakActionRequest(target=original, form_name="kill", harms_cell=True)
    second = SoakActionRequest(target=replacement, form_name="kill", harms_cell=True)
    episodes = compute_recovery_episodes(
        [
            SoakActionRequestedEvent(timestamp=start, request=first),
            SoakActionAppliedEvent(timestamp=start + timedelta(seconds=1), request_id=first.request_id, evidence={}),
            SoakObservation(timestamp=start + timedelta(seconds=2), cells=[replacement]),
            SoakActionRequestedEvent(timestamp=start + timedelta(seconds=3), request=second),
            SoakActionAppliedEvent(timestamp=start + timedelta(seconds=4), request_id=second.request_id, evidence={}),
            SoakObservation(timestamp=start + timedelta(seconds=5), cells=[replacement]),
        ]
    )
    assert len(episodes) == 2
    assert episodes[0].request_ids == [first.request_id]
    assert episodes[0].recovered_incarnation == "replacement"
    assert episodes[1].request_ids == [second.request_id]
    assert episodes[1].recovered_incarnation is None


@pytest.mark.parametrize("evidence", ["matching", "early", "late", "wrong_generation", "missing", "not_healed"])
def test_trainer_recovery_requires_the_same_new_incarnation_in_a_causal_reconfigure(evidence: str) -> None:
    """Healthy status cannot replace a matching reconfigure after the fault request."""
    start = datetime(2026, 9, 11, tzinfo=timezone.utc)
    original = typed_cell("actor-0", "actor")
    replacement = {**original, "status": {**original["status"], "workers_hash": "replacement"}}
    request = SoakActionRequest(target=original, form_name="kill", harms_cell=True)
    events = [
        SoakActionRequestedEvent(timestamp=start, request=request),
        SoakActionAppliedEvent(timestamp=start + timedelta(seconds=2), request_id=request.request_id, evidence={}),
        SoakObservation(timestamp=start + timedelta(seconds=10), cells=[replacement]),
    ]
    offset = {"early": -1, "late": 11}.get(evidence, 3)
    reconfigure = CellReconfigureEvent(
        timestamp=start + timedelta(seconds=offset),
        source=TrainerControllerProcessIdentity(trainer_id="actor"),
        rollout_id=1,
        quorum_id=1,
        src_cell_index=1,
        healed_cell_indices=[] if evidence == "not_healed" else [0],
        alive_cell_indices_after=[0, 1],
        cell_incarnations_after=(
            {}
            if evidence == "missing"
            else {"actor-0": "generation-0" if evidence == "wrong_generation" else "replacement"}
        ),
    )
    episodes = compute_recovery_episodes(events, reconfigurations=[reconfigure])
    assert len(episodes) == 1
    assert episodes[0].recovered_incarnation == ("replacement" if evidence == "matching" else None)


@pytest.mark.parametrize("malformation", ["unknown", "duplicate", "early"])
def test_invalid_applied_evidence_cannot_silently_disappear_from_recovery(malformation: str) -> None:
    """Unknown, repeated, and causally impossible receipts fail the evidence projection."""
    start = datetime(2026, 9, 11, tzinfo=timezone.utc)
    request = SoakActionRequest(target=typed_cell("rollout-0", "rollout"), form_name="kill", harms_cell=True)
    applied = SoakActionAppliedEvent(
        timestamp=start + timedelta(seconds=-1 if malformation == "early" else 1),
        request_id="unknown" if malformation == "unknown" else request.request_id,
        evidence={},
    )
    events = [SoakActionRequestedEvent(timestamp=start, request=request), applied]
    if malformation == "duplicate":
        events.append(applied)
    with pytest.raises(ValueError):
        compute_recovery_episodes(events)


@pytest.mark.parametrize("final_incarnation", ["generation-0", "generation-1", "generation-2"])
def test_one_recovery_closes_repeated_faults_only_on_an_unharmed_new_incarnation(final_incarnation: str) -> None:
    """Two faults during one recovery need a later serving incarnation that neither fault targeted."""
    start = datetime(2026, 9, 11, tzinfo=timezone.utc)
    original = typed_cell("rollout-0", "rollout")
    replacement = {**original, "status": {**original["status"], "workers_hash": "generation-1", "phase": "Pending"}}
    first = SoakActionRequest(target=original, form_name="kill", harms_cell=True)
    second = SoakActionRequest(target=replacement, form_name="kill", harms_cell=True)
    final = {**original, "status": {**original["status"], "workers_hash": final_incarnation}}
    events = [
        SoakActionRequestedEvent(timestamp=start, request=first),
        SoakActionAppliedEvent(timestamp=start + timedelta(seconds=1), request_id=first.request_id, evidence={}),
        SoakObservation(timestamp=start + timedelta(seconds=2), cells=[replacement]),
        SoakActionRequestedEvent(timestamp=start + timedelta(seconds=3), request=second),
        SoakActionAppliedEvent(timestamp=start + timedelta(seconds=4), request_id=second.request_id, evidence={}),
        SoakObservation(timestamp=start + timedelta(seconds=5), cells=[final]),
    ]
    episodes = compute_recovery_episodes(events)
    assert len(episodes) == 1
    assert episodes[0].request_ids == [first.request_id, second.request_id]
    assert episodes[0].recovered_incarnation == ("generation-2" if final_incarnation == "generation-2" else None)


def test_a_failed_observation_does_not_close_an_outstanding_recovery() -> None:
    """An unreadable cluster is not evidence that a faulted cell has recovered."""
    request = SoakActionRequest(target=typed_cell("rollout-0", "rollout"), form_name="kill", harms_cell=True)
    episodes = compute_recovery_episodes(
        [
            SoakActionRequestedEvent(request=request),
            SoakActionAppliedEvent(request_id=request.request_id, evidence={}),
            SoakObservation(cells=None, errors={"cells": "timeout"}),
        ]
    )
    assert len(episodes) == 1 and episodes[0].recovered_incarnation is None
