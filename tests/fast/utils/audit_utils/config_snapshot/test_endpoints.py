from collections.abc import Callable

import pytest

from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotGeneratedValue
from miles.utils.audit_utils.config_snapshot.normalizer import normalize_record
from miles.utils.test_utils.snapshot import dump_snapshot


class TestAllocatedEndpointNormalization:
    def test_dynamic_allocations_use_common_placeholders(self, make_endpoint_record: Callable) -> None:
        """Automatically allocated addresses normalize independently of their concrete host and port."""
        first = make_endpoint_record(host="10.0.0.1", port=30000)
        second = make_endpoint_record(host="10.0.0.9", port=31000)
        assert dump_snapshot(ConfigSnapshotConverter.convert([first])) == dump_snapshot(
            ConfigSnapshotConverter.convert([second])
        )

    def test_shared_and_independent_endpoints_use_the_same_placeholders(self, make_endpoint_record: Callable) -> None:
        """Allocation sharing does not affect the normalized configuration."""
        independent = make_endpoint_record()
        shared = make_endpoint_record(shared=True)
        assert dump_snapshot(ConfigSnapshotConverter.convert([independent])) == dump_snapshot(
            ConfigSnapshotConverter.convert([shared])
        )

    @pytest.mark.parametrize("change", ["order", "missing", "identity"])
    def test_identity_order_and_count_remain_observable(self, make_endpoint_record: Callable, change: str) -> None:
        """Endpoint normalization retains instance identity, ordering, and count."""
        record = make_endpoint_record()
        expected = dump_snapshot(ConfigSnapshotConverter.convert([record]))
        instances = [dict(instance) for instance in record.config["args"]["session_server_instances"]]
        if change == "order":
            instances.reverse()
        elif change == "missing":
            instances.pop()
        else:
            instances[0]["instance_id"] = "another-identity"
        record = record.model_copy(update={"config": {"args": {"session_server_instances": instances}}})
        assert dump_snapshot(ConfigSnapshotConverter.convert([record])) != expected

    def test_static_allocations_remain_exact(self, make_endpoint_record: Callable) -> None:
        """Explicitly configured host and port changes stay visible even when provenance is present."""
        first = make_endpoint_record(dynamic=False)
        second = make_endpoint_record(host="10.0.0.9", port=31000, dynamic=False)
        assert normalize_record(first) == normalize_record(first, generated_values=[])
        assert dump_snapshot(ConfigSnapshotConverter.convert([first])) != dump_snapshot(
            ConfigSnapshotConverter.convert([second])
        )

    def test_unregistered_addresses_remain_exact(self, make_endpoint_record: Callable) -> None:
        """An address without allocation provenance is not inferred to be dynamic."""
        record = make_endpoint_record().model_copy(update={"generated_values": []})
        assert normalize_record(record)["args"]["session_server_instances"][0]["addr"] == "10.0.0.1:30000"

    @pytest.mark.parametrize("generation_field", ["deploy_instance_id", "name"])
    def test_new_deployment_generations_can_reallocate_ports(
        self, make_endpoint_record: Callable, generation_field: str
    ) -> None:
        """A restarted deployment may legitimately allocate a different address."""
        first = make_endpoint_record()
        second = make_endpoint_record(port=31000)
        second = second.model_copy(
            update={"context": second.context.model_copy(update={generation_field: "next", "capture_id": "next"})}
        )
        assert len(ConfigSnapshotConverter.convert([first, second]).processes) == 2

    def test_static_external_host_is_preserved(self, make_endpoint_record: Callable) -> None:
        """The configured public host is retained while its automatically allocated port stabilizes."""
        record = make_endpoint_record()
        instances = [dict(instance) for instance in record.config["args"]["session_server_instances"]]
        instances[0]["external_addr"] = "public.example:30000"
        record = record.model_copy(
            update={
                "config": {"args": {"session_server_instances": instances}},
                "generated_values": [entry for entry in record.generated_values if entry.kind != "external_host"],
            }
        )
        actual = normalize_record(record)
        assert actual["args"]["session_server_instances"][0]["external_addr"] == "public.example:$PORT"

    def test_primary_router_and_per_model_map_use_common_placeholders(self, make_record: Callable) -> None:
        """Registered router addresses normalize only in endpoint fields."""
        values = [
            ConfigSnapshotGeneratedValue(kind=kind, name=value, value=value)
            for kind, value in [("host", "10.0.0.1"), ("port", "30000"), ("port", "30001")]
        ]
        record = make_record(
            config={
                "sglang_model_routers": {
                    name: {"$tuple": ["10.0.0.1", 30000 + i]} for i, name in enumerate(["actor", "ref"])
                },
                "sglang_router_ip": "10.0.0.1",
                "sglang_router_port": 30000,
                "unrelated_host": "10.0.0.1",
                "unrelated_port": 30000,
            }
        ).model_copy(update={"generated_values": values})
        actual = normalize_record(record)["args"]
        assert actual["sglang_model_routers"]["actor"]["$tuple"] == [
            actual["sglang_router_ip"],
            actual["sglang_router_port"],
        ]
        assert actual["sglang_router_ip"] == "$HOST"
        assert actual["sglang_router_port"] == "$PORT"
        assert actual["sglang_model_routers"]["ref"]["$tuple"] == ["$HOST", "$PORT"]
        assert actual["unrelated_host"] == "10.0.0.1"
        assert actual["unrelated_port"] == 30000
