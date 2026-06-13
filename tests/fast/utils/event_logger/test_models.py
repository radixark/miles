from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.backends.megatron_utils.types import TrainStepOutcome
from miles.utils.event_logger.models import (
    CellReconfigureEvent,
    Event,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
    WitnessSnapshotParamEvent,
)
from miles.utils.process_identity import MainProcessIdentity, TrainProcessIdentity

_event_adapter = TypeAdapter(Event)

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FIXED_SOURCE = MainProcessIdentity()
_TRAIN_SOURCE = TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)


class TestEventModelsDiscriminatedUnion:
    def test_roundtrip_via_discriminator(self) -> None:
        event = WitnessAllocateIdEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=0,
            attempt=0,
            witness_id_to_sample_index={10: 0, 11: 1},
            counter_after=12,
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessAllocateIdEvent)
        assert parsed.witness_id_to_sample_index == {10: 0, 11: 1}

    def test_discriminator_distinguishes_types(self) -> None:
        e1 = WitnessAllocateIdEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=0,
            attempt=0,
            witness_id_to_sample_index={0: 0},
            counter_after=1,
        )
        e2 = TrainGroupStepEndEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=0,
            cell_outcomes={0: [TrainStepOutcome.NORMAL]},
        )
        p1 = _event_adapter.validate_json(e1.model_dump_json())
        p2 = _event_adapter.validate_json(e2.model_dump_json())
        assert type(p1) is not type(p2)


class TestEventModelsStrictRejectExtraFields:
    def test_extra_field_rejected(self) -> None:
        data = {
            "type": "witness_allocate_id",
            "timestamp": "2026-01-01T00:00:00Z",
            "source": {"component": "main"},
            "rollout_id": 0,
            "attempt": 0,
            "witness_id_to_sample_index": {0: 0},
            "bogus_field": 123,
        }
        with pytest.raises(ValidationError, match="bogus_field"):
            WitnessAllocateIdEvent.model_validate(data)


class TestWitnessAllocateIdEvent:
    def test_json_roundtrip(self) -> None:
        event = WitnessAllocateIdEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=2,
            attempt=0,
            witness_id_to_sample_index={10: 0, 11: 1},
            counter_after=12,
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessAllocateIdEvent)
        assert parsed.rollout_id == 2
        assert parsed.attempt == 0
        assert parsed.witness_id_to_sample_index == {10: 0, 11: 1}


class TestTrainGroupStepEndEvent:
    def test_json_roundtrip(self) -> None:
        event = TrainGroupStepEndEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=3,
            cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: "error"},
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, TrainGroupStepEndEvent)
        assert parsed.rollout_id == 3
        assert parsed.cell_outcomes[0] == [TrainStepOutcome.NORMAL]
        assert parsed.cell_outcomes[1] == "error"


class TestCellReconfigureEvent:
    def test_healing_json_roundtrip(self) -> None:
        """A healing reconfigure event (non-empty healed cells, with src) survives a JSON round-trip."""
        event = CellReconfigureEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=3,
            quorum_id=1,
            src_cell_index=0,
            healed_cell_indices=[2],
            alive_cell_indices_after=[0, 1, 2],
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, CellReconfigureEvent)
        assert parsed.rollout_id == 3
        assert parsed.quorum_id == 1
        assert parsed.src_cell_index == 0
        assert parsed.healed_cell_indices == [2]
        assert parsed.alive_cell_indices_after == [0, 1, 2]

    def test_shrink_json_roundtrip(self) -> None:
        """A pure-shrink reconfigure event (no healed cells, src None) survives a JSON round-trip."""
        event = CellReconfigureEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=2,
            quorum_id=1,
            src_cell_index=None,
            healed_cell_indices=[],
            alive_cell_indices_after=[0],
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, CellReconfigureEvent)
        assert parsed.src_cell_index is None
        assert parsed.healed_cell_indices == []
        assert parsed.alive_cell_indices_after == [0]


class TestWitnessSnapshotParamEventWithStaleIds:
    def test_json_roundtrip(self) -> None:
        event = WitnessSnapshotParamEvent(
            timestamp=_FIXED_TS,
            source=_TRAIN_SOURCE,
            rollout_id=5,
            instance_id="actor_cell0_rank0",
            nonzero_witness_ids=[10, 11, 12],
            stale_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessSnapshotParamEvent)
        assert parsed.stale_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert parsed.nonzero_witness_ids == [10, 11, 12]


class TestDiscriminatedUnionParsesAllEvents:
    def test_all_event_types_parse(self) -> None:
        events = [
            WitnessAllocateIdEvent(
                timestamp=_FIXED_TS,
                source=_FIXED_SOURCE,
                rollout_id=0,
                attempt=0,
                witness_id_to_sample_index={0: 0},
                counter_after=1,
            ),
            TrainGroupStepEndEvent(
                timestamp=_FIXED_TS,
                source=_FIXED_SOURCE,
                rollout_id=0,
                cell_outcomes={0: [TrainStepOutcome.NORMAL]},
            ),
        ]
        for event in events:
            parsed = _event_adapter.validate_json(event.model_dump_json())
            assert type(parsed) is type(event)


class TestCheckEventNaming:
    def test_naming_convention_holds(self) -> None:
        from miles.utils.event_logger.models import _check_event_naming

        _check_event_naming()
