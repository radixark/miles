from argparse import Namespace
from pathlib import Path

import pytest
import torch
from tests.fast.utils.env_report.typed_report_fixtures import _DataclassReportConfig, _SerializedReportConfig

from miles.utils.env_report.collector import _dump_args


class TestTypedConfigurationReporting:
    def test_pydantic_report_reads_typed_fields_instead_of_a_custom_serializer(self) -> None:
        """Audit collection preserves original typed fields even when application JSON hides their values."""
        config = _SerializedReportConfig(checkpoint=Path("/actual/model"), wandb_key="synthetic-token")
        assert config.model_dump()["checkpoint"] == "serializer-mask"

        dump = _dump_args(config)

        assert dump.values["checkpoint"] == {"$path": "/actual/model"}
        assert dump.values["wandb_key"].startswith("redacted-sha256:")
        assert config.wandb_key == "synthetic-token"
        assert dump.skipped_names == []

    def test_nested_dataclass_retains_path_dtype_and_secret_redaction(self) -> None:
        """Typed dataclass fields remain visible and nested credentials remain hidden."""
        metadata = _DataclassReportConfig(
            checkpoint=Path("/dataclass/model"), dtype=torch.bfloat16, wandb_key="synthetic-token"
        )

        dump = _dump_args(Namespace(metadata=metadata))

        assert dump.values["metadata"]["checkpoint"] == {"$path": "/dataclass/model"}
        assert dump.values["metadata"]["dtype"] == {"$torch": "torch.bfloat16"}
        assert dump.values["metadata"]["wandb_key"].startswith("redacted-sha256:")
        assert metadata.wandb_key == "synthetic-token"
        assert dump.skipped_names == []

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_values_have_local_diagnostics_without_losing_valid_siblings(self, value: float) -> None:
        """Invalid JSON scalars produce explicit diagnostics while valid configuration remains available."""
        dump = _dump_args(Namespace(nested={"bad": value, "valid": 0.25}))

        assert dump.values["nested"]["bad"] == {"$serialization_error": "ValueError: Cannot serialize builtins.float"}
        assert dump.values["nested"]["valid"] == 0.25
        assert dump.skipped_names == []

    def test_shared_nested_values_are_not_mistaken_for_cycles(self) -> None:
        """Two fields may share one live object without creating a circular configuration."""
        shared = {"values": [1, 2]}

        dump = _dump_args(Namespace(first=shared, second=shared))
        shared["values"].append(3)

        assert dump.values == {"first": {"values": [1, 2]}, "second": {"values": [1, 2]}}
        assert dump.values["first"] is not dump.values["second"]
        assert dump.skipped_names == []

    def test_unsupported_mapping_keys_do_not_discard_other_configuration_fields(self) -> None:
        """An unrepresentable key reports a local mapping failure without hiding valid top-level fields."""
        dump = _dump_args(Namespace(mapping={object(): "value"}, valid=0.25))

        assert dump.values["mapping"] == {"$serialization_error": "TypeError: Cannot serialize builtins.dict"}
        assert dump.values["valid"] == 0.25
        assert dump.skipped_names == []
