from pathlib import Path

import pytest

from miles.utils.audit_utils import sample_ownership
from miles.utils.audit_utils.event_logger import logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    RolloutHoldingsSnapshotEvent,
    SampleOwner,
    SampleOwnerTransitionEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity


@pytest.fixture
def ownership_event_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    event_dir = tmp_path / "ownership-events"
    monkeypatch.setattr(
        event_logger_module,
        "_event_logger",
        EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="rollout_executor")),
    )
    monkeypatch.setattr(sample_ownership, "_lineage_id", "initial")
    return event_dir


@pytest.fixture
def lost_sample_event_dir(tmp_path: Path) -> Path:
    event_dir = tmp_path / "events"
    event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="rollout_executor"))
    event_logger.log(
        event_cls=SampleOwnerTransitionEvent,
        partial=dict(sample_indices=[1], from_owner=SampleOwner.DATA_SOURCE, to_owner=SampleOwner.IN_FLIGHT),
    )
    event_logger.log(
        event_cls=RolloutHoldingsSnapshotEvent,
        partial=dict(rollout_id=0, holdings={}, replays_samples=False, reason="save"),
    )
    event_logger.close()
    return event_dir
