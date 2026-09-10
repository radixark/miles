from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    IssuedSampleGroup,
    TrainerCpuWitnessEvent,
    TrainerWitnessCohortEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership_store import SampleOwnershipEventStore

_SOURCE = SimpleProcessIdentity(component="rollout_executor")
_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _payload(cohort_id: str) -> dict:
    snapshot = TrainerCpuWitnessEvent(
        timestamp=_TIMESTAMP,
        source=_SOURCE,
        replica_id="cell-0",
        rollout_id=1,
        cohort_id=cohort_id,
        sample_counts=[],
        reason="current",
    )
    marker = TrainerWitnessCohortEvent(
        timestamp=_TIMESTAMP,
        source=_SOURCE,
        rollout_id=1,
        cohort_id=cohort_id,
        replica_ids=["cell-0"],
    )
    return {
        "snapshots": [snapshot.model_dump(mode="json")],
        "marker": marker.model_dump(mode="json"),
    }


class TestSampleOwnershipEventStore:
    def test_reads_history_incrementally_and_keeps_the_latest_cohort(self, tmp_path: Path) -> None:
        """Issued history grows while full witness storage remains one replace snapshot."""
        event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
        store = SampleOwnershipEventStore(event_logger)
        event_logger.log(
            DataSourceIssuedSamplesEvent,
            {"rollout_id": 1, "groups": [IssuedSampleGroup(group_index=1, sample_indices=[10])]},
            print_log=False,
        )

        assert len(store.read_history()) == 1
        store.replace_current(_payload("first"))
        store.replace_current(_payload("second"))

        events = store.read_events()
        assert len([event for event in events if isinstance(event, DataSourceIssuedSamplesEvent)]) == 1
        assert store.read_current().marker.cohort_id == "second"
        assert len(list(tmp_path.glob("sample_ownership_current.json"))) == 1

    def test_malformed_history_is_fatal(self, tmp_path: Path) -> None:
        """Accounting history corruption cannot silently remove an issued obligation."""
        event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
        store = SampleOwnershipEventStore(event_logger)
        (tmp_path / "events.jsonl").write_text('{"type":"data_source_issued_samples"}\n')

        with pytest.raises(ValidationError):
            store.read_history()

    def test_malformed_current_snapshot_is_fatal(self, tmp_path: Path) -> None:
        """A corrupt current witness cannot make ownership analysis pass open."""
        event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
        store = SampleOwnershipEventStore(event_logger)
        store.current_path.write_text('{"snapshots":[]}')

        with pytest.raises(ValidationError):
            store.read_current()
