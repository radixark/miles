from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.utils import async_utils

_BROADCAST_MODULE = "miles.backends.training_utils.weight_update.protocols.broadcast"
_TENSOR_MODULE = "miles.backends.training_utils.weight_update.protocols.cuda_ipc"

_NAMED_TENSORS = [("model.layers.0.mlp.gate_proj.weight", torch.zeros(2, 2))]


class _RecordingApiClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self.calls.append((name, kwargs))
            return {"success": True}

        return method


def test_distributed_update_does_not_flush():
    """The trainer, not a default, decides that a broadcast update skips the cache flush."""
    from miles.backends.training_utils.weight_update.protocols.broadcast import update_weights_from_distributed

    client = _RecordingApiClient()

    with patch(f"{_BROADCAST_MODULE}.dist") as dist_mock:
        dist_mock.broadcast.return_value = MagicMock()
        futures = update_weights_from_distributed(
            group_name="miles-pp_0",
            group=MagicMock(),
            rollout_engines=[client],
            converted_named_tensors=_NAMED_TENSORS,
        )
        async_utils.wait_futures(futures)

    name, kwargs = client.calls[0]
    assert name == "update_weights_from_distributed"
    assert kwargs["group_name"] == "miles-pp_0"
    assert kwargs.get("flush_cache", False) is False


def test_colocated_base_update_does_not_flush():
    """Same explicit no-flush decision on the colocated IPC path."""
    pytest.importorskip("sglang")

    from miles.backends.training_utils.weight_update.protocols.cuda_ipc import _send_to_colocated_engine

    client = _RecordingApiClient()

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
        )
        async_utils.wait_futures(futures)

    name, kwargs = client.calls[0]
    assert name == "update_weights_from_tensor"
    assert kwargs["load_format"] == "flattened_bucket"
    assert kwargs.get("flush_cache", False) is False
