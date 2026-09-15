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
    }


def _assert_roundtrip_equal(fetched: dict[str, Any], original: dict[str, Any]) -> None:
    assert sorted(fetched.keys()) == sorted(original.keys())
    assert [_tolist(t) for t in fetched["tokens"]] == [_tolist(t) for t in original["tokens"]]
    assert _tolist(fetched["rewards"]) == original["rewards"]
    assert _tolist(fetched["raw_reward"]) == original["raw_reward"]
    assert _tolist(fetched["total_lengths"]) == original["total_lengths"]
    assert list(fetched["prompt"]) == original["prompt"]
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
