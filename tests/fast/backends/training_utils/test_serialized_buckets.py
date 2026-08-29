from types import SimpleNamespace

import torch

from miles.backends.fsdp_utils import update_weight_utils as fsdp_update_weight
from miles.backends.megatron_utils.update_weight import update_weight_from_tensor as megatron_update_weight
from miles.backends.training_utils.serialized_buckets import (
    align_serialized_bucket_columns,
    empty_flattened_tensor_data,
)


class _RemoteMethod:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _Engine:
    def __init__(self):
        self.update_weights_from_tensor = _RemoteMethod({"success": True})
        self.flush_cache = _RemoteMethod()


class _RejectBucketConstruction:
    supports_multi_dtypes = True

    def __init__(self, **_kwargs):
        raise AssertionError("an empty rank must not construct FlattenedTensorBucket")


class _IdentitySerializer:
    @staticmethod
    def serialize(value, *, output_str):
        assert output_str is True
        return value


def _source_dist(remote_buckets):
    def gather_object(obj, object_gather_list, **_kwargs):
        object_gather_list[:] = [obj, remote_buckets]

    return SimpleNamespace(
        get_rank=lambda: 0,
        get_world_size=lambda _group=None: 2,
        gather_object=gather_object,
    )


def _patch_transport(monkeypatch, module, remote_buckets):
    empty_data = {"flattened_tensor": "empty", "metadata": []}
    monkeypatch.setattr(module, "dist", _source_dist(remote_buckets))
    monkeypatch.setattr(module, "FlattenedTensorBucket", _RejectBucketConstruction)
    monkeypatch.setattr(module, "MultiprocessingSerializer", _IdentitySerializer)
    monkeypatch.setattr(module, "empty_flattened_tensor_data", lambda *, device: empty_data)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    return empty_data


def test_all_empty_ranks_yield_no_columns():
    assert align_serialized_bucket_columns([[], []]) == []


def test_short_rank_pads_with_none():
    remote = {"flattened_tensor": ("remote",), "metadata": ("w",)}
    assert align_serialized_bucket_columns([[], [remote]]) == [[None, remote]]


def test_ragged_dtype_counts_pad_later_columns():
    a0, a1, b0 = "a0", "a1", "b0"
    assert align_serialized_bucket_columns([[a0, a1], [b0]]) == [[a0, b0], [a1, None]]


def test_empty_flattened_tensor_data_is_zero_length_uint8_payload():
    data = empty_flattened_tensor_data(device="cpu")

    assert data["flattened_tensor"].dtype is torch.uint8
    assert data["flattened_tensor"].numel() == 0
    assert data["metadata"] == []


def test_megatron_all_empty_gather_is_noop(monkeypatch):
    _patch_transport(monkeypatch, megatron_update_weight, remote_buckets=[])
    engine = _Engine()

    refs, long_live_tensors = megatron_update_weight._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=7,
    )

    assert refs == []
    assert long_live_tensors == []
    assert engine.update_weights_from_tensor.calls == []


def test_megatron_empty_source_is_padded_for_each_remote_bucket(monkeypatch):
    remote_buckets = ["remote-0", "remote-1"]
    empty_data = _patch_transport(monkeypatch, megatron_update_weight, remote_buckets=remote_buckets)
    engine = _Engine()

    refs, long_live_tensors = megatron_update_weight._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=7,
    )

    assert refs == [{"success": True}, {"success": True}]
    assert long_live_tensors == [empty_data]
    assert engine.update_weights_from_tensor.calls == [
        {
            "serialized_named_tensors": [empty_data, "remote-0"],
            "load_format": "flattened_bucket",
            "weight_version": "7",
            "selector": "all",
        },
        {
            "serialized_named_tensors": [empty_data, "remote-1"],
            "load_format": "flattened_bucket",
            "weight_version": "7",
            "selector": "all",
        },
    ]


def test_fsdp_empty_source_is_padded_for_remote_bucket(monkeypatch):
    empty_data = _patch_transport(monkeypatch, fsdp_update_weight, remote_buckets=["remote"])
    monkeypatch.setattr(fsdp_update_weight, "monkey_patch_torch_reductions", lambda: None)
    monkeypatch.setattr(fsdp_update_weight.ray, "get", lambda ref: ref)
    engine = _Engine()
    updater = fsdp_update_weight.UpdateWeightFromTensor.__new__(fsdp_update_weight.UpdateWeightFromTensor)
    updater._ipc_gather_src = 0
    updater._ipc_gather_group = object()
    updater._ipc_engine = engine

    updater.update_bucket_weights([], weight_version=7)

    assert engine.update_weights_from_tensor.calls == [
        {
            "serialized_named_tensors": [empty_data, "remote"],
            "load_format": "flattened_bucket",
            "flush_cache": False,
            "weight_version": "7",
        }
    ]
    assert engine.flush_cache.calls == [{}]
