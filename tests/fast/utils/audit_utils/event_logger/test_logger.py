import weakref
from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger import logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger, EventReader
from miles.utils.audit_utils.event_logger.models import MetricEvent, TrainEngineLocalWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity


class TestEventsDirectoryName:
    def test_events_directory_name_matches_the_on_disk_contract(self) -> None:
        """The exported events directory name remains compatible with stored audit data."""
        assert EVENTS_DIRNAME == "events"


class TestEventReaderSelection:
    def test_selected_reads_do_not_retain_unrelated_checksum_payloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Progress readers retain metrics without caching large checksum histories."""
        writer = EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"))
        writer.log(MetricEvent, {"rollout_id": 1, "metrics": {"train/grad_norm": 1.0}}, print_log=False)
        writer.log(
            TrainEngineLocalWeightChecksumEvent,
            {
                "rollout_id": 1,
                "state": {"param_hashes": {"weight": "x" * 100000}, "buffer_hashes": {}, "optimizer_hashes": []},
            },
            print_log=False,
        )
        original = event_logger_module._event_adapter.validate_json
        unrelated: list[weakref.ReferenceType] = []

        def validate_json(data: bytes, **kwargs: Any) -> Any:
            event = original(data, **kwargs)
            if isinstance(event, TrainEngineLocalWeightChecksumEvent):
                unrelated.append(weakref.ref(event))
            return event

        monkeypatch.setattr(event_logger_module._event_adapter, "validate_json", validate_json)
        reader = EventReader(tmp_path, event_types=(MetricEvent,))
        [metric] = reader.read()
        assert metric.rollout_id == 1
        assert unrelated and all(reference() is None for reference in unrelated)
        assert reader.read() == [metric]
        assert len(EventReader(tmp_path).read()) == 2

    def test_selected_reads_still_reject_malformed_events_in_strict_mode(self, tmp_path: Path) -> None:
        """Selecting event types must not bypass strict event validation."""
        (tmp_path / "events.jsonl").write_text('{"type":"unknown"}\n')
        with pytest.raises(ValueError):
            EventReader(tmp_path, strict=True, event_types=(MetricEvent,)).read()
