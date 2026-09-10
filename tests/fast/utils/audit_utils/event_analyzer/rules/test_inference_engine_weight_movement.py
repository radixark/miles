import pytest

from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_movement import check
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


class TestWeightMovement:
    @pytest.mark.parametrize("case", ["changed", "unchanged", "missing_tensor", "lora", "interval", "unknown"])
    def test_movement_requires_complete_changed_weights_unless_explicitly_excluded(self, case: str) -> None:
        """Unchanged weights and missing tensors fail unless the mode is explicitly excluded."""
        policy = {"lora": ["lora_base_weights"], "interval": ["update_interval"], "unknown": None}.get(case, [])
        events = []
        for version in [2, 1]:
            tensors = {"rank0/w": "initial", "rank0/b": "constant"}
            if version == 2 and case == "changed":
                tensors["rank0/w"] = "updated"
            if version == 2 and case == "missing_tensor":
                del tensors["rank0/b"]
            events.append(
                InferenceEngineWeightChecksumEvent.model_validate(
                    dict(
                        timestamp="2026-01-01T00:00:00Z",
                        source={"component": "main"},
                        rollout_id=version,
                        weight_version=version,
                        engine_checksums=[tensors],
                        movement_skip_reasons=policy,
                        engine_snapshots=[
                            dict(model_name="actor", cell_id="engine", workers_hash="generation", tensors=tensors)
                        ],
                    )
                )
            )
        if case == "unknown":
            with pytest.raises(AssertionError, match="applicability"):
                check(events)
        else:
            assert bool(check(events)) == (case in {"unchanged", "missing_tensor"})
