from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.utils import async_utils

_BROADCAST_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
_TENSOR_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"

_NAMED_TENSORS = [("model.layers.0.mlp.gate_proj.weight", torch.zeros(2, 2))]


class _RecordingApiClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self.calls.append((name, kwargs))
            return {"success": True}

        return method


def test_distributed_update_does_not_flush_and_stringifies_the_weight_version():
    """The trainer, not a default, decides that a broadcast update skips the cache flush."""
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
        update_weights_from_distributed,
    )

    client = _RecordingApiClient()

    with patch(f"{_BROADCAST_MODULE}.dist") as dist_mock:
        dist_mock.broadcast.return_value = MagicMock()
        futures = update_weights_from_distributed(
            group_name="miles-pp_0",
            group=MagicMock(),
            weight_version=5,
            rollout_engines=[client],
            converted_named_tensors=_NAMED_TENSORS,
        )
        async_utils.wait_futures(futures)

    name, kwargs = client.calls[0]
    assert name == "update_weights_from_distributed"
    assert kwargs["group_name"] == "miles-pp_0"
    assert kwargs.get("flush_cache", False) is False
    assert kwargs["weight_version"] == "5"


def _run_colocated_send(*, client, lora_config=None, lora_name=None):
    pytest.importorskip("sglang")

    from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import _send_to_colocated_engine

    def fake_gather_object(obj, object_gather_list=None, dst=None, group=None):
        object_gather_list[0] = obj

    with patch(f"{_TENSOR_MODULE}.dist") as dist_mock:
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = 1
        dist_mock.gather_object.side_effect = fake_gather_object
        futures, _long_live_tensors = _send_to_colocated_engine(
            _NAMED_TENSORS,
            ipc_engine=client,
            ipc_gather_src=0,
            ipc_gather_group=MagicMock(),
            weight_version=5,
            lora_config=lora_config,
            lora_name=lora_name,
        )
        async_utils.wait_futures(futures)


def test_colocated_base_update_does_not_flush_and_stringifies_the_weight_version():
    """Same explicit no-flush decision on the colocated IPC path."""
    client = _RecordingApiClient()

    _run_colocated_send(client=client)

    name, kwargs = client.calls[0]
    assert name == "update_weights_from_tensor"
    assert kwargs["load_format"] == "flattened_bucket"
    assert kwargs.get("flush_cache", False) is False
    assert kwargs["weight_version"] == "5"


def test_colocated_lora_update_registers_a_fresh_unpinned_adapter():
    """The colocated LoRA send must not upsert, pin or inject added tokens."""
    client = _RecordingApiClient()

    _run_colocated_send(client=client, lora_config={"peft_type": "LORA", "r": 8}, lora_name="adapter-a")

    assert [name for name, _kwargs in client.calls][:1] == ["unload_lora_adapter"]
    kwargs = next(kw for name, kw in client.calls if name == "load_lora_adapter_from_tensors")
    assert kwargs["lora_name"] == "adapter-a"
    assert kwargs.get("pinned", False) is False
    assert kwargs.get("added_tokens_config") is None
    assert kwargs.get("upsert", False) is False
