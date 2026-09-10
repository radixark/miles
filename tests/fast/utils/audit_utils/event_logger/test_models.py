import pytest
from pydantic import ValidationError

from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


class TestInferenceEngineWeightChecksumEvent:
    @pytest.mark.parametrize(
        "case", ["valid", "missing_version", "missing_snapshot", "tensor", "duplicate", "model", "empty_identity"]
    )
    def test_identified_checksums_reject_partial_or_contradictory_evidence(self, case: str) -> None:
        """New checksum evidence must bind one version and model to unique nonempty cell snapshots."""
        snapshot = dict(
            model_name="actor", cell_id="engine-0", workers_hash="generation-0", tensors={"rank0/w": "hash"}
        )
        data = dict(
            timestamp="2026-01-01T00:00:00Z",
            source={"component": "main"},
            rollout_id=0,
            weight_version=1,
            engine_checksums=[snapshot["tensors"]],
            engine_snapshots=[snapshot],
        )
        if case == "missing_version":
            data["weight_version"] = None
        elif case == "missing_snapshot":
            data["engine_snapshots"] = []
        elif case == "tensor":
            data["engine_checksums"] = [{"rank0/w": "different"}]
        elif case in {"duplicate", "model"}:
            data["engine_snapshots"].append(
                {**snapshot, **({"cell_id": "engine-1", "model_name": "other"} if case == "model" else {})}
            )
            data["engine_checksums"].append(snapshot["tensors"])
        elif case == "empty_identity":
            snapshot["workers_hash"] = ""
        if case == "valid":
            assert InferenceEngineWeightChecksumEvent.model_validate(data).weight_version == 1
        else:
            with pytest.raises(ValidationError):
                InferenceEngineWeightChecksumEvent.model_validate(data)

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
