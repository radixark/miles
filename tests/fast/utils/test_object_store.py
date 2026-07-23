import shutil
import socket
import subprocess
import time
from argparse import Namespace
from typing import Any

import pytest
import ray
import torch

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


class TestRayObjectStore:
    @pytest.fixture(scope="class", autouse=True)
    def _ray_minicluster(self):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        yield

    def test_roundtrip_and_noop_remove(self):
        """RayObjectStore puts/gets a rollout dict and remove is a no-op."""
        args = Namespace(rollout_data_transport="object-store")
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
            rollout_data_transport="mooncake",
            mooncake_store_init_kwargs={
                "protocol": "tcp",
                "master_server_address": f"127.0.0.1:{port}",
                "global_segment_size": "64mb",
                "local_buffer_size": "64mb",
            },
            mooncake_rollout_replica_num=1,
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
        args.mooncake_rollout_replica_num = 0
        with pytest.raises(ValueError):
            object_store.init_instance(args)
