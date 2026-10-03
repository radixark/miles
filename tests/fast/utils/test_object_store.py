import json
import pickle
import shutil
import socket
import subprocess
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.utils import object_store


def _mooncake_available() -> bool:
    if shutil.which("mooncake_master") is None:
        return False
    try:
        from mooncake.structured_object_store import FieldSchema, export_ref  # noqa: F401
    except ImportError:
        return False
    return True


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tolist(value: Any) -> list:
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _make_rollout_data() -> dict[str, Any]:
    return {
        "tokens": [torch.tensor([1, 2, 3], dtype=torch.int32), torch.tensor([4, 5], dtype=torch.int32)],
        "loss_masks": [torch.tensor([1, 1, 1], dtype=torch.int32), torch.tensor([1, 1], dtype=torch.int32)],
        "rewards": [0.5, 1.0],
        "response_lengths": [3, 2],
        "partition": [0, 1],
        "sample_indices": [0, 1],
        "truncated": [0, 0],
        "raw_reward": [0.5, 1.0],
        "total_lengths": [3, 2],
        "prompt": ["hello", "world"],
        "metadata": [{"a": 1}, {"b": 2}],
        "weight_versions": [
            [
                {"spans": [{"version": "2", "abs_start": 1, "abs_end": 3}], "prefill_spans": [], "output_start": 1},
                {
                    "spans": [{"version": "3", "abs_start": 3, "abs_end": 3}],
                    "prefill_spans": [
                        {"version": "1", "abs_start": 0, "abs_end": 2},
                        {"version": "3", "abs_start": 2, "abs_end": 3},
                    ],
                    "output_start": 3,
                },
            ],
            [],
        ],
    }


def _assert_roundtrip_equal(fetched: dict[str, Any], original: dict[str, Any]) -> None:
    assert sorted(fetched.keys()) == sorted(original.keys())
    assert [_tolist(t) for t in fetched["tokens"]] == [_tolist(t) for t in original["tokens"]]
    assert _tolist(fetched["rewards"]) == original["rewards"]
    assert _tolist(fetched["raw_reward"]) == original["raw_reward"]
    assert _tolist(fetched["total_lengths"]) == original["total_lengths"]
    assert list(fetched["prompt"]) == original["prompt"]
    assert list(fetched["weight_versions"]) == original["weight_versions"]
    assert list(fetched["metadata"]) == original["metadata"]


@pytest.fixture(autouse=True)
def _reset_object_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(object_store, "_INSTANCE", None)


class TestDefaultContributeSegment:
    def test_no_local_rank_contributes(self, monkeypatch: pytest.MonkeyPatch):
        """Processes without LOCAL_RANK (drivers, rollout manager) contribute by default."""
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert object_store._default_contribute_segment() is True

    def test_local_rank_zero_contributes(self, monkeypatch: pytest.MonkeyPatch):
        """LOCAL_RANK=0 contributes a segment."""
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert object_store._default_contribute_segment() is True

    def test_nonzero_local_rank_does_not_contribute(self, monkeypatch: pytest.MonkeyPatch):
        """LOCAL_RANK>0 does not contribute a segment."""
        monkeypatch.setenv("LOCAL_RANK", "3")
        assert object_store._default_contribute_segment() is False


@dataclass
class _StubFieldSchema:
    codec: str
    nullable: bool
    metadata: dict[str, Any]


class TestFieldSchemasForValue:
    @pytest.fixture(autouse=True)
    def _stub_field_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(object_store, "FieldSchema", _StubFieldSchema)

    def test_none_spec_returns_none(self):
        """Without a value spec, no field schemas are generated."""
        assert object_store._field_schemas_for_value({"a": 1}, None) is None

    def test_auto_codec_pins_meta_info_section(self):
        """codec='auto' fields go to meta_info; others go to non_tensor_batch."""
        spec = {
            "scalar": object_store.ValueSpec(codec="auto"),
            "ragged": object_store.ValueSpec(codec="typed_ragged", dtype="int32"),
        }
        schemas = object_store._field_schemas_for_value({"scalar": 1, "ragged": [2]}, spec)
        assert schemas["scalar"].metadata["section"] == "meta_info"
        assert schemas["ragged"].metadata["section"] == "non_tensor_batch"

    def test_dtype_included_only_when_set(self):
        """dtype appears in schema metadata only for specs that declare it."""
        spec = {
            "typed": object_store.ValueSpec(codec="typed_ragged", dtype="int32"),
            "untyped": object_store.ValueSpec(codec="msgpack"),
        }
        schemas = object_store._field_schemas_for_value({"typed": [1], "untyped": [2]}, spec)
        assert schemas["typed"].metadata["dtype"] == "int32"
        assert "dtype" not in schemas["untyped"].metadata
        assert all(schema.nullable is False for schema in schemas.values())

    def test_spec_fields_absent_from_value_are_skipped(self):
        """Spec entries for keys missing from the value dict produce no schema."""
        spec = {
            "present": object_store.ValueSpec(codec="auto"),
            "absent": object_store.ValueSpec(codec="auto"),
        }
        schemas = object_store._field_schemas_for_value({"present": 1}, spec)
        assert sorted(schemas.keys()) == ["present"]


class TestSingletonContract:
    def test_double_init_rejected(self):
        """Calling init_instance twice in one process asserts."""
        args = Namespace(object_store_backend="ray", worker_comm_backend="ray")
        object_store.init_instance(args)
        with pytest.raises(AssertionError):
            object_store.init_instance(args)

    def test_unknown_backend_rejected(self):
        """An unknown backend value raises ValueError from the enum lookup."""
        with pytest.raises(ValueError):
            object_store.init_instance(Namespace(object_store_backend="bogus"))


class TestObjectStoreGetResult:
    def test_value_property_and_release_on_exit(self):
        """The context manager exposes the value and calls release_fn exactly once on exit."""
        released: list[Any] = []
        result = object_store.ObjectStoreGetResult(value={"a": 1}, release_fn=released.append)
        assert result.value == {"a": 1}
        with result as value:
            assert value == {"a": 1}
            assert released == []
        assert released == [{"a": 1}]


class TestStoreObjectRefWireValidation:
    def test_a_wire_reference_without_a_backend_tag_is_rejected(self) -> None:
        """A wire reference without a backend discriminator is rejected."""
        adapter = TypeAdapter(object_store.StoreObjectRef)

        with pytest.raises(ValidationError):
            adapter.validate_json('{"payload": "opaque-reference"}')

    def test_a_corrupt_encoded_ray_reference_is_rejected_at_the_wire_boundary(self) -> None:
        """A Ray reference with corrupt cloudpickle data is rejected during wire validation."""
        adapter = TypeAdapter(object_store.StoreObjectRef)

        with pytest.raises(pickle.UnpicklingError):
            adapter.validate_json('{"backend": "ray", "payload": "bm90IGEgcGlja2xl"}')


class TestRayObjectStore:
    @pytest.fixture(scope="class", autouse=True)
    def _ray_minicluster(self, ray_local_mode):
        yield

    def test_roundtrip_and_noop_remove(self):
        """RayObjectStore puts/gets a rollout dict and remove is a no-op."""
        args = Namespace(object_store_backend="ray", worker_comm_backend="ray")
        store = object_store.init_instance(args)
        assert isinstance(store, object_store.RayObjectStore)

        data = _make_rollout_data()
        ref = store.put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        get_result = store.get(ref)
        _assert_roundtrip_equal(get_result.value, data)
        with get_result:
            pass
        store.remove(ref)
        _assert_roundtrip_equal(store.get(ref).value, data)

    def test_get_instance_requires_init(self):
        """get_instance asserts when init_instance was never called."""
        with pytest.raises(AssertionError):
            object_store.get_instance()


@pytest.mark.skipif(not _mooncake_available(), reason="mooncake with structured_object_store API not installed")
class TestMooncakeObjectStore:
    @pytest.fixture(scope="class")
    def mooncake_master_port(self):
        port = _free_port()
        master = subprocess.Popen(
            ["mooncake_master", "--rpc_port", str(port), "--metrics_port", str(_free_port())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        yield port
        master.terminate()
        master.wait(timeout=10)

    def _make_args(self, port: int) -> Namespace:
        return Namespace(
            object_store_backend="mooncake",
            mooncake_store_init_kwargs={
                "protocol": "tcp",
                "master_server_address": f"127.0.0.1:{port}",
                "global_segment_size": "64mb",
                "local_buffer_size": "64mb",
            },
            mooncake_replica_num=1,
        )

    def test_roundtrip_release_and_remove(self, mooncake_master_port: int):
        """MooncakeObjectStore round-trips a rollout dict; remove deletes the object."""
        store = object_store.init_instance(self._make_args(mooncake_master_port))
        assert isinstance(store, object_store.MooncakeObjectStore)

        data = _make_rollout_data()
        ref = store.put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        get_result = store.get(ref)
        _assert_roundtrip_equal(get_result.value, data)
        with get_result:
            pass

        store.remove(ref)
        with pytest.raises(Exception):  # noqa: B017 - mooncake surfaces missing keys as varying exception types
            store.get(ref)

    def test_replica_num_below_one_rejected(self, mooncake_master_port: int):
        """Constructing the store with replica num < 1 raises ValueError."""
        args = self._make_args(mooncake_master_port)
        args.mooncake_replica_num = 0
        with pytest.raises(ValueError):
            object_store.init_instance(args)


class TestMooncakeReplicaFailover:
    @pytest.mark.parametrize(
        "error",
        [
            json.JSONDecodeError("Expecting value", "", 0),
            RuntimeError("get_into_ranges failed for key: expected 8, got -800"),
            RuntimeError("batch_get_into failed for key: expected 8, got -800"),
            RuntimeError("get_into_ranges failed for key: expected 8, got -703"),
            RuntimeError("batch_get_into failed for key: expected 8, got -703"),
        ],
    )
    def test_a_failed_replica_read_recovers_without_losing_the_value(
        self, mooncake_reader: Any, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        """Transient replica transfer failures preserve a subsequent successful read."""
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        mooncake_reader.get.side_effect = [error, {"value": 42}]
        store = object_store.MooncakeObjectStore(
            Namespace(mooncake_store_init_kwargs={"local_hostname": "127.0.0.1"}, mooncake_replica_num=2),
            contribute_segment=False,
        )
        with store.get(object_store._MooncakeStoreObjectRef(payload="ref")) as value:
            assert value == {"value": 42}

    def test_a_permanent_replica_failure_is_raised_after_the_deadline(
        self, mooncake_reader: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unavailable replicated object cannot retry forever."""
        ticks = iter([0.0, 0.0, 31.0])
        monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        error = json.JSONDecodeError("Expecting value", "", 0)
        mooncake_reader.get.side_effect = error
        store = object_store.MooncakeObjectStore(
            Namespace(mooncake_store_init_kwargs={"local_hostname": "127.0.0.1"}, mooncake_replica_num=2),
            contribute_segment=False,
        )
        with pytest.raises(json.JSONDecodeError) as caught:
            store.get(object_store._MooncakeStoreObjectRef(payload="ref"))
        assert caught.value is error
        assert mooncake_reader.get.call_count == 2

    @pytest.mark.parametrize(
        "error",
        [json.JSONDecodeError("Expecting value", "broken", 0), RuntimeError("unsupported structured field encoding")],
    )
    def test_invalid_objects_fail_without_retrying(self, mooncake_reader: Any, error: Exception) -> None:
        """Malformed objects and programming errors are not replica failovers."""
        mooncake_reader.get.side_effect = error
        store = object_store.MooncakeObjectStore(
            Namespace(mooncake_store_init_kwargs={"local_hostname": "127.0.0.1"}, mooncake_replica_num=2),
            contribute_segment=False,
        )
        with pytest.raises(type(error)) as caught:
            store.get(object_store._MooncakeStoreObjectRef(payload="ref"))
        assert caught.value is error
        assert mooncake_reader.get.call_count == 1

    @pytest.mark.parametrize("operation", ["put", "batch_put_from"])
    def test_a_failed_replica_write_retries_the_same_value(
        self, mooncake_reader: Any, monkeypatch: pytest.MonkeyPatch, operation: str
    ) -> None:
        """A departing replica does not lose the already generated rollout batch."""
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        monkeypatch.setattr(object_store, "export_ref", lambda ref: ref, raising=False)
        monkeypatch.setattr(object_store, "ReplicateConfig", Namespace)
        mooncake_reader.put.side_effect = [RuntimeError(f"{operation} failed for key: -800"), "published"]
        store = object_store.MooncakeObjectStore(
            Namespace(mooncake_store_init_kwargs={}, mooncake_replica_num=2), contribute_segment=False
        )
        value = {"tokens": [1, 2, 3]}

        assert store.put(value=value).payload == "published"
        assert len(mooncake_reader.put.call_args_list) == 2
        assert all(call.args[0] is value for call in mooncake_reader.put.call_args_list)
        assert all(call.kwargs["config"].replica_num == 2 for call in mooncake_reader.put.call_args_list)

    def test_a_permanent_write_failure_stops_at_the_deadline(
        self, mooncake_reader: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A permanently unavailable write target fails within the retry budget."""
        monkeypatch.setattr(object_store, "ReplicateConfig", Namespace)
        ticks = iter([0.0, 0.0, 31.0])
        monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        error = RuntimeError("put failed for key: -800")
        mooncake_reader.put.side_effect = error
        store = object_store.MooncakeObjectStore(
            Namespace(mooncake_store_init_kwargs={}, mooncake_replica_num=2), contribute_segment=False
        )

        with pytest.raises(RuntimeError) as caught:
            store.put(value={"tokens": [1]})
        assert caught.value is error
        assert mooncake_reader.put.call_count == 2

    @pytest.mark.parametrize(
        ("replicas", "message"),
        [
            (1, "put failed for key: -800"),
            (2, "put failed for key: -100"),
            (2, "unsupported structured field encoding"),
        ],
    )
    def test_non_failover_write_errors_are_not_retried(
        self, mooncake_reader: Any, monkeypatch: pytest.MonkeyPatch, replicas: int, message: str
    ) -> None:
        """Single-replica writes and unrelated errors retain their failure semantics."""
        monkeypatch.setattr(object_store, "ReplicateConfig", Namespace)
        error = RuntimeError(message)
        mooncake_reader.put.side_effect = error
        store = object_store.MooncakeObjectStore(
            Namespace(mooncake_store_init_kwargs={}, mooncake_replica_num=replicas), contribute_segment=False
        )

        with pytest.raises(RuntimeError) as caught:
            store.put(value={"tokens": [1]})
        assert caught.value is error
        assert mooncake_reader.put.call_count == 1
