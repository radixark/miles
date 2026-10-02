"""CPU wire-format checks against the actual SGLang bucket implementation."""

import importlib.util
import os
import sys
import weakref
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.protocols import broadcast

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])


@pytest.fixture
def bucket_class(monkeypatch):
    # CPU CI supplies SGLANG_SOURCE_ROOT. Load this pure-Torch module directly
    # so format tests do not require SGLang's unrelated GPU runtime imports.
    root = os.environ.get("SGLANG_SOURCE_ROOT")
    if root is None:
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        return FlattenedTensorBucket
    name = "sglang.srt.weight_sync.tensor_bucket"
    spec = importlib.util.spec_from_file_location(name, Path(root) / "sglang/srt/weight_sync/tensor_bucket.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.FlattenedTensorBucket


def _weights():
    return [
        ("expert.weight", torch.arange(16, dtype=torch.uint8).reshape(2, 8).T),
        ("expert.weight_scale", torch.tensor([0, 128, 126, 255], dtype=torch.uint8).view(torch.float8_e4m3fn)),
        ("expert.weight_scale_2", torch.tensor(-0.0, dtype=torch.float32)),
        ("dense.weight", torch.arange(8, dtype=torch.bfloat16).reshape(2, 4).T),
        ("strided_bf16.weight", torch.arange(8, dtype=torch.bfloat16)[::2]),
        ("strided_fp32.weight", torch.arange(8, dtype=torch.float32)[::2]),
        ("empty", torch.empty((0, 2), dtype=torch.float32)),
    ]


def _assert_same_bytes(expected, actual):
    assert [name for name, _ in actual] == [name for name, _ in expected]
    for (_, original), (_, received) in zip(expected, actual, strict=True):
        assert original.shape == received.shape
        assert original.dtype == received.dtype
        assert torch.equal(
            original.contiguous().flatten().view(torch.uint8), received.contiguous().flatten().view(torch.uint8)
        )


@pytest.mark.parametrize("packed", [False, True])
def test_broadcast_roundtrip_preserves_mixed_dtype_bytes_and_scalar_shape(monkeypatch, bucket_class, packed):
    weights = _weights()
    original = [(name, tensor.clone()) for name, tensor in weights]
    clients = [SimpleNamespace(update_weights_from_distributed=Mock(return_value=object())) for _ in range(2)]
    monkeypatch.setattr(broadcast.async_utils, "submit", lambda future: future)
    sent = []
    waits = []
    group = object()

    def send(tensor, src, *, group, async_op):
        assert src == 0 and async_op
        assert tensor.is_contiguous()
        sent.append(tensor.clone())
        return SimpleNamespace(wait=lambda: waits.append(tensor.numel()))

    monkeypatch.setattr(broadcast.dist, "broadcast", send)
    futures = broadcast.update_weights_from_distributed(
        "test-group", group, clients, weights, "target", "broadcast_packed" if packed else "broadcast"
    )
    assert futures == [client.update_weights_from_distributed.return_value for client in clients]
    assert len(sent) == len(waits) == (1 if packed else len(weights))
    for client in clients:
        metadata = client.update_weights_from_distributed.call_args.kwargs
        assert metadata == {
            "names": [name for name, _ in weights],
            "dtypes": [tensor.dtype for _, tensor in weights],
            "shapes": [tensor.shape for _, tensor in weights],
            "selector": "target",
            "group_name": "test-group",
            **({"load_format": "flattened_bucket"} if packed else {}),
        }
    if packed:
        assert sent[0].dtype == torch.uint8
        assert sent[0].numel() == sum(t.numel() * t.element_size() for _, t in weights)
        # Reproduce SGLang's existing distributed receiver with the transmitted
        # metadata: allocate empty tensors, form its bucket, receive, reconstruct.
        receiver = bucket_class(
            named_tensors=[
                (n, torch.empty(s, dtype=d))
                for n, d, s in zip(metadata["names"], metadata["dtypes"], metadata["shapes"], strict=True)
            ]
        )
        receiver.get_flattened_tensor().copy_(sent[0])
        received = receiver.reconstruct_tensors()
    else:
        received = [(name, tensor) for (name, _), tensor in zip(weights, sent, strict=True)]
    _assert_same_bytes(original, received)
    _assert_same_bytes(original, weights)


@pytest.mark.parametrize(
    "failure,error,message",
    [
        ("unaligned", ValueError, "Unaligned flattened weight scale: byte offset 3"),
        ("unsupported", RuntimeError, "mixed-dtype FlattenedTensorBucket"),
        ("allocation", MemoryError, "bucket allocation"),
    ],
)
def test_pack_failure_does_not_dispatch_receivers(monkeypatch, bucket_class, failure, error, message):
    weights = _weights()
    if failure == "unaligned":
        weights = [("byte", torch.ones(3, dtype=torch.uint8)), ("scale", torch.tensor(1.0))]
    elif failure == "unsupported":
        monkeypatch.setattr(bucket_class, "supports_multi_dtypes", False)
    else:
        monkeypatch.setattr(bucket_class, "__init__", Mock(side_effect=MemoryError("bucket allocation")))
    client = SimpleNamespace(update_weights_from_distributed=Mock())
    collective = Mock()
    monkeypatch.setattr(broadcast.dist, "broadcast", collective)
    with pytest.raises(error, match=message):
        broadcast.update_weights_from_distributed("g", object(), [client], weights, transfer_mode="broadcast_packed")
    client.update_weights_from_distributed.assert_not_called()
    collective.assert_not_called()


def test_backing_buffer_lives_through_collective_wait_and_bucket_clears_after_receiver_completion(
    monkeypatch, bucket_class
):
    weights = _weights()
    events = []
    buffers = []
    client = SimpleNamespace(update_weights_from_distributed=Mock(return_value=object()))
    monkeypatch.setattr(broadcast.async_utils, "submit", lambda future: future)

    def send(tensor, *args, **kwargs):
        ref = weakref.ref(tensor)
        buffers.append(ref)

        def wait():
            assert ref() is not None
            assert weights
            events.append("collective_complete")

        return SimpleNamespace(wait=wait)

    def wait_receivers(futures):
        assert futures == [client.update_weights_from_distributed.return_value]
        assert weights
        assert events == ["collective_complete"]
        events.append("receivers_complete")

    monkeypatch.setattr(broadcast.dist, "broadcast", send)
    monkeypatch.setattr(broadcast.async_utils, "wait_futures", wait_receivers)
    updater = broadcast.UpdateWeightFromDistributed.__new__(broadcast.UpdateWeightFromDistributed)
    updater.args = Namespace(update_weight_transfer_mode="broadcast_packed")
    updater._engine_lock = nullcontext()
    updater.group_name = "g"
    updater._model_update_groups = object()
    updater.rollout_engines = [client]
    updater._selector = "target"
    updater.send_bucket(weights)
    assert len(buffers) == 1
    assert events == ["collective_complete", "receivers_complete"]
    assert not weights
    assert client.update_weights_from_distributed.call_args.kwargs["load_format"] == "flattened_bucket"


@pytest.mark.parametrize("backend,colocate", [("megatron", True), ("fsdp", False), ("fsdp", True)])
def test_packed_mode_rejects_backends_that_would_ignore_it(backend, colocate):
    args = Namespace(train_backend=backend, colocate=colocate, update_weight_transfer_mode="broadcast_packed")
    with pytest.raises(ValueError, match="requires Megatron non-colocated"):
        get_weight_transfer_protocol(args)


def test_unknown_mode_cannot_fall_back_to_colocated_transfer():
    args = Namespace(train_backend="megatron", colocate=True, update_weight_transfer_mode="typo")
    with pytest.raises(ValueError, match="Unknown --update-weight-transfer-mode"):
        get_weight_transfer_protocol(args)


@pytest.mark.parametrize("mode", ["broadcast", "broadcast_packed"])
def test_both_broadcast_modes_share_protocol_and_preserve_selection(monkeypatch, mode):
    monkeypatch.setattr(broadcast, "get_parallel_state", lambda: SimpleNamespace(pp=SimpleNamespace(size=1)))
    args = Namespace(train_backend="megatron", colocate=False, update_weight_transfer_mode=mode)
    protocol = get_weight_transfer_protocol(args)
    assert isinstance(protocol, broadcast.UpdateWeightFromDistributed)
    assert protocol.args.update_weight_transfer_mode == mode


def test_direct_caller_without_mode_keeps_legacy_protocol(monkeypatch):
    monkeypatch.setattr(broadcast, "get_parallel_state", lambda: SimpleNamespace(pp=SimpleNamespace(size=1)))
    args = Namespace(train_backend="megatron", colocate=False)
    assert isinstance(get_weight_transfer_protocol(args), broadcast.UpdateWeightFromDistributed)
    assert not hasattr(args, "update_weight_transfer_mode")


def test_low_level_helper_rejects_non_broadcast_mode_before_receiver_dispatch(monkeypatch):
    client = SimpleNamespace(update_weights_from_distributed=Mock())
    collective = Mock()
    monkeypatch.setattr(broadcast.dist, "broadcast", collective)
    with pytest.raises(ValueError, match="Expected a broadcast transfer mode"):
        broadcast.update_weights_from_distributed("g", object(), [client], _weights(), transfer_mode="p2p")
    client.update_weights_from_distributed.assert_not_called()
    collective.assert_not_called()
