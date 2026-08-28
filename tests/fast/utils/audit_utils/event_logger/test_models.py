import pytest
from pydantic import ValidationError

from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


class TestInferenceEngineWeightChecksumEvent:
    def test_none_rollout_id_is_rejected(self) -> None:
        """A checksum event rejects the former null startup rollout identifier."""
        data = {
            "timestamp": "2026-01-01T00:00:00Z",
            "source": {"component": "main"},
            "rollout_id": None,
            "engine_checksums": [{"rank0/embed.weight": "aaa"}],
        }

        with pytest.raises(ValidationError, match="rollout_id"):
            InferenceEngineWeightChecksumEvent.model_validate(data)
