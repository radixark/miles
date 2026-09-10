from pathlib import Path

import pytest

from miles.utils.audit_utils.event_logger import logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity


@pytest.fixture
def ownership_event_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    event_dir = tmp_path / "ownership-events"
    monkeypatch.setattr(
        event_logger_module,
        "_event_logger",
        EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="rollout_executor")),
    )
    return event_dir
