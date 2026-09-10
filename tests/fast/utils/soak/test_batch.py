from datetime import timedelta

import pytest
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.batch import expand_fault_batches, validate_fault_batch
from tests.utils.soak.config import SoakCellPolicy
from tests.utils.soak.policy import eligible_cells
from tests.utils.soak.recovery import compute_recovery_episodes
from tests.utils.soak.state import SoakActionAppliedEvent, SoakActionRequest, SoakActionRequestedEvent, SoakObservation


class TestFaultBatch:
    def test_every_requested_victim_is_reserved_before_any_receipt(self, batch_request: SoakActionRequest) -> None:
        """An in-flight batch reserves additional victims as well as its primary target."""
        assert (
            eligible_cells(
                cells=[batch_request.target, batch_request.additional_requests[0].target],
                events=[SoakActionRequestedEvent(request=batch_request)],
                policy=SoakCellPolicy(min_survivors=0),
                harms_cell=True,
            )
            == []
        )

    @pytest.mark.parametrize("corruption", ["duplicate_id", "duplicate_cell", "nested", "harmless"])
    def test_invalid_batches_are_rejected(self, batch_request: SoakActionRequest, corruption: str) -> None:
        """Each batch victim must be a distinct harmful cell request without nested work."""
        child = batch_request.additional_requests[0]
        updates = {
            "duplicate_id": {"request_id": batch_request.request_id},
            "duplicate_cell": {"target": batch_request.target},
            "nested": {"additional_requests": [child]},
            "harmless": {"harms_cell": False},
        }[corruption]
        request = batch_request.model_copy(update={"additional_requests": [child.model_copy(update=updates)]})
        with pytest.raises(ValueError):
            validate_fault_batch(request)

    def test_additional_victim_keeps_its_own_recovery_debt(self, batch_request: SoakActionRequest) -> None:
        """Recovery of the first victim cannot repay another batch victim's debt."""
        request_event = SoakActionRequestedEvent(request=batch_request)
        applied = SoakActionAppliedEvent(
            timestamp=request_event.timestamp + timedelta(seconds=1),
            request_id=batch_request.request_id,
            evidence={"batch_receipts": {batch_request.additional_requests[0].request_id: {"effect": "confirmed"}}},
        )
        recovered = typed_cell("rollout-0", "rollout")
        recovered["status"]["workers_hash"] = "replacement"
        events = [
            request_event,
            applied,
            SoakObservation(timestamp=applied.timestamp + timedelta(seconds=1), cells=[recovered]),
        ]
        episodes = compute_recovery_episodes(events)
        assert len(episodes) == 2
        assert episodes[0].recovered_incarnation == "replacement"
        assert episodes[1].recovered_incarnation is None
        assert expand_fault_batches(expand_fault_batches(events)) == expand_fault_batches(events)

    def test_missing_child_receipt_cannot_project_a_success(self, batch_request: SoakActionRequest) -> None:
        """A successful parent cannot fabricate an effect for an unconfirmed victim."""
        with pytest.raises(ValueError, match="cover exactly"):
            expand_fault_batches(
                [
                    SoakActionRequestedEvent(request=batch_request),
                    SoakActionAppliedEvent(request_id=batch_request.request_id, evidence={"batch_receipts": {}}),
                ]
            )
