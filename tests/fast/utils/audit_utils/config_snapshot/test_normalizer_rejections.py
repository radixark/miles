import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from collections.abc import Callable

from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotGeneratedValue, ConfigSnapshotRecord
from miles.utils.audit_utils.config_snapshot.normalizer import normalize_record


class TestTrainingSnapshotRejection:
    @pytest.mark.parametrize("config", [None, {"args": []}])
    def test_training_capture_requires_an_arguments_mapping(
        self, make_record: Callable[..., ConfigSnapshotRecord], config: JsonValue
    ) -> None:
        """Malformed training envelopes cannot normalize as valid process configurations."""
        record = make_record().model_copy(update={"config": config})

        with pytest.raises(ValueError, match="require a config.args object"):
            normalize_record(record)

        assert record.config == config

    @pytest.mark.parametrize("backend", [False, True])
    @pytest.mark.parametrize("rank", [False, -1])
    def test_rank_normalization_rejects_boolean_and_negative_values(
        self, make_record: Callable[..., ConfigSnapshotRecord], backend: bool, rank: JsonValue
    ) -> None:
        """Rank placeholders cannot erase an invalid rank type or negative rank."""
        config = {"backend": {"rank": rank}} if backend else {"rank": rank}
        record = make_record(config=config)

        with pytest.raises(AssertionError, match="Unexpected args.*rank"):
            normalize_record(record)

        assert record.config == {"args": config}

    def test_a_training_backend_must_be_a_mapping(self, make_record: Callable[..., ConfigSnapshotRecord]) -> None:
        """Malformed backend payloads cannot bypass training rank validation."""
        record = make_record(config={"backend": []})

        with pytest.raises(AssertionError, match="config.args.backend object"):
            normalize_record(record)

        assert record.config == {"args": {"backend": []}}


class TestEndpointSnapshotRejection:
    @pytest.mark.parametrize(
        "config,message",
        [
            ({"sglang_model_routers": {"actor": "invalid"}}, "Invalid router endpoint snapshot"),
            ({"sglang_model_routers": {"actor": {"$tuple": ["host", 1, 2]}}}, "Invalid router endpoint snapshot"),
            ({"session_server_instances": [{"addr": None}]}, "Invalid session endpoint snapshot"),
            ({"session_server_instances": [{"addr": "host"}]}, "Invalid session endpoint address"),
            ({"session_server_instances": [{"addr": "host:invalid"}]}, "Invalid session endpoint address"),
        ],
    )
    def test_malformed_endpoint_shapes_and_addresses_remain_errors_when_provenance_is_present(
        self,
        make_record: Callable[..., ConfigSnapshotRecord],
        config: dict[str, JsonValue],
        message: str,
    ) -> None:
        """Generated endpoint metadata cannot conceal malformed router or session records."""
        record = make_record(config=config).model_copy(
            update={"generated_values": [ConfigSnapshotGeneratedValue(kind="host", name="router", value="host")]}
        )
        before = record.model_dump_json()

        with pytest.raises(ValueError, match=message):
            normalize_record(record)

        assert record.model_dump_json() == before
