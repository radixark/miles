"""Startup checkpoint version declaration must precede the first rollout."""

import asyncio
import threading
import time
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock, patch

import numpy as np
import pytest
import safetensors.numpy
import torch

from miles.backends.training_utils.weight_update.protocols import gpu_delta


class _Engine:
    def __init__(self, protocol, events, index, fail=False):
        self.protocol, self.events, self.index, self.fail = protocol, events, index, fail
        self.version = "default"

    async def update_weight_version(self, weight_version):
        assert not self.protocol._baseline_captured
        np.testing.assert_array_equal(self.protocol._snapshot["w"], [1, 2, 3, 4])
        await asyncio.sleep(0.01 if self.index else 0)
        self.events.append(self.index)
        if self.fail:
            raise RuntimeError("engine rejected base declaration")
        self.version = weight_version
        return {"success": True, "new_version": weight_version}


def _setup(tmp_path, fail=False):
    safetensors.numpy.save_file({"w": np.array([1, 2, 3, 4], dtype=np.uint8)}, tmp_path / "model.safetensors")
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "delta"),
            custom_update_weight_post_write_path=None,
            update_weight_buffer_size=5,
            update_weight_delta_initial_sync=False,
        )
    )
    protocol._plan = {
        "w": {
            "name": "w",
            "dtype": "U8",
            "shape": [4],
            "encoding": "raw_bytes",
            "views": [{"id": "full", "slices": [[0, 4]]}],
        }
    }
    protocol.is_sender = True
    events = []
    protocol.rollout_engines = [_Engine(protocol, events, 0, fail), _Engine(protocol, events, 1)]
    return protocol, events


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    with (
        patch.object(gpu_delta, "_gather_all", side_effect=lambda value: [value]),
        patch.object(gpu_delta, "get_gloo_group", return_value=None),
        patch.object(gpu_delta.dist, "get_rank", return_value=0),
        patch.object(gpu_delta.dist, "broadcast_object_list"),
    ):
        yield


def _buckets(materialize):
    assert materialize
    # The baseline must use the startup checkpoint, not this exported value.
    yield [("w", torch.tensor([5, 6, 7, 8], dtype=torch.uint8))]


def test_export_bucket_stages_after_all_conversions_with_one_stream_dependency(monkeypatch):
    protocol = gpu_delta.UpdateWeightFromGpuDelta(Namespace(custom_update_weight_post_write_path=None))
    bucket = [
        (name, torch.arange(6, dtype=torch.float32).reshape(2, 3) + index) for index, name in enumerate(("a", "b"))
    ]
    protocol._snapshot = {name: torch.zeros(tensor.nbytes, dtype=torch.uint8) for name, tensor in bucket}
    protocol._next_snapshot = {name: torch.empty_like(value) for name, value in protocol._snapshot.items()}
    protocol._seen = set()
    events, caller_stream = [], object()
    staging_active = False
    protocol._staging_stream = Mock()
    protocol._staging_stream.wait_stream.side_effect = lambda stream: events.append(("wait", stream))
    protocol._match_layout = lambda name, tensor: (events.append(("convert", name)), tensor)[1]

    @contextmanager
    def staging(stream):
        nonlocal staging_active
        assert stream is protocol._staging_stream
        staging_active = True
        yield
        staging_active = False

    copy = torch.Tensor.copy_

    def record_copy(destination, source, non_blocking):
        assert non_blocking and staging_active and ("wait", caller_stream) in events
        events.append(("copy", source.data_ptr()))
        return copy(destination, source)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller_stream)
    monkeypatch.setattr(torch.cuda, "stream", staging)
    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda tensor, stream: events.append(("lease", tensor.data_ptr(), stream))
    )
    protocol.send_bucket(bucket)
    assert protocol._error is None
    protocol._staging_stream.wait_stream.assert_called_once_with(caller_stream)
    wait_index = events.index(("wait", caller_stream))
    assert {event[1] for event in events if event[0] == "convert"} == {"a", "b"}
    assert all(index < wait_index for index, event in enumerate(events) if event[0] == "convert")
    copied = [event[1] for event in events if event[0] == "copy"]
    leased = [event[1] for event in events if event[0] == "lease" and event[2] is protocol._staging_stream]
    assert sorted(copied) == sorted(leased) == sorted(tensor.data_ptr() for _, tensor in bucket)
    for name, tensor in bucket:
        assert torch.equal(protocol._next_snapshot[name], tensor.reshape(-1).view(torch.uint8))
        assert not torch.count_nonzero(protocol._snapshot[name])


def test_invalid_export_bucket_does_not_enqueue_a_partial_snapshot(monkeypatch):
    protocol = gpu_delta.UpdateWeightFromGpuDelta(Namespace(custom_update_weight_post_write_path=None))
    protocol._snapshot = {"a": torch.zeros(4, dtype=torch.uint8)}
    protocol._next_snapshot = {"a": torch.zeros(4, dtype=torch.uint8)}
    protocol._seen = set()
    protocol._match_layout = lambda name, tensor: tensor
    protocol._staging_stream = Mock()
    caller_stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", caller_stream)
    protocol.send_bucket([("a", torch.ones(4, dtype=torch.uint8)), ("a", torch.ones(4, dtype=torch.uint8))])
    assert isinstance(protocol._error, ValueError) and "Duplicate canonical tensor owner" in str(protocol._error)
    caller_stream.assert_not_called()
    protocol._staging_stream.wait_stream.assert_not_called()
    assert not torch.count_nonzero(protocol._next_snapshot["a"])


def test_partial_version_acknowledgement_fails_before_rollout_without_replay(tmp_path, single_rank):
    protocol, events = _setup(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="engine rejected base declaration"):
        protocol.begin_sync(1, _buckets)
    assert sorted(events) == [0, 1]  # all submitted RPCs settled before broadcasting failure
    assert not protocol._baseline_captured and protocol._uncommitted
    with pytest.raises(RuntimeError, match="automatic replay is forbidden"):
        protocol.begin_sync(1, _buckets)
    assert len(events) == 2


@pytest.mark.parametrize("duplicate_owner", [False, True])
def test_pipeline_stage_inventory_requires_unique_owners_before_baseline_declaration(
    tmp_path, single_rank, monkeypatch, duplicate_owner
):
    protocol, events = _setup(tmp_path)
    peer_name = "w" if duplicate_owner else "stage1.weight"
    protocol._plan["stage1.weight"] = {
        "name": "stage1.weight",
        "dtype": "U8",
        "shape": [4],
        "encoding": "raw_bytes",
    }

    def gather(value):
        if value is None:  # Existing collective error check.
            return [None] * 4
        # Two PP stages, each with one sender and one transport non-sender.
        # The local stage reads the real checkpoint through begin_sync; only
        # the remote inventory exchange is replaced in this CPU test.
        assert value == ["w"]
        return [value, [], [peer_name], []]

    monkeypatch.setattr(gpu_delta, "_gather_all", gather)
    if duplicate_owner:
        with pytest.raises(RuntimeError, match=r"ownership mismatch:.*duplicates=\['w'\]"):
            protocol.begin_sync(1, _buckets)
        assert not events and not protocol._baseline_captured
        assert [engine.version for engine in protocol.rollout_engines] == ["default", "default"]
    else:
        assert protocol.begin_sync(1, _buckets) is False
        assert protocol._baseline_captured and not protocol._uncommitted and sorted(events) == [0, 1]
        assert not protocol._stream_dir.exists()
        assert protocol._raw_names == ("w",) and protocol._gpu_batch_names == ()
        assert set(protocol._snapshot) == {"w"}
        assert [engine.version for engine in protocol.rollout_engines] == ["0", "0"]


@pytest.mark.parametrize("initial_sync", [False, True])
def test_initial_delta_publishes_loaded_trainer_after_common_baseline(
    tmp_path, single_rank, monkeypatch, initial_sync
):
    protocol, events = _setup(tmp_path)
    protocol.args.update_weight_delta_initial_sync = initial_sync
    protocol._plan_digest = "plan"
    protocol._staging_stream = Mock()
    protocol._next_snapshot = {"w": torch.empty(4, dtype=torch.uint8)}
    protocol._gpu_encoder = Mock(frame_bytes=1 << 20, outer_metrics={})
    protocol._gpu_encoder.wrap_device.return_value = []  # This fixture contains only a raw vector.
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Mock)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(gpu_delta.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(gpu_delta.dist, "gather_object", lambda shard, shards, **kwargs: shards.__setitem__(0, shard))

    assert protocol.begin_sync(1, _buckets) is initial_sync
    assert sorted(events) == [0, 1]
    if not initial_sync:
        assert not protocol._stream_dir.exists()
        assert protocol.begin_sync(1, _buckets) is True
    assert not protocol._seen  # The baseline export must not consume the current update's inventory.
    for bucket in _buckets(materialize=True):
        protocol.send_bucket(bucket)
    protocol.after_base_weights()
    publication = protocol.publish(1)
    assert publication["base_version"] == 0 and publication["target_version"] == 1
    assert publication["summary_counts"]["raw_bytes"] == 4
    assert publication["summary_counts"]["wire_bytes"] == 4
    assert (protocol._version_dir / "owner-00000.bin").read_bytes() == bytes([5, 6, 7, 8])
    np.testing.assert_array_equal(protocol._snapshot["w"], [1, 2, 3, 4])
    protocol.commit_pending_baseline()
    np.testing.assert_array_equal(protocol._snapshot["w"], [5, 6, 7, 8])


def _gpu_pending(monkeypatch, fail_batch=None, omit=None):
    """Exercise protocol ordering on CPU; native tests cover CUDA encode/copy."""
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(update_weight_buffer_size=5, custom_update_weight_post_write_path=None)
    )
    # Deliberately insert names out of order; c exceeds the batch target.
    sizes = {"d": 1, "b": 3, "c": 7, "a": 2}
    protocol._snapshot = {name: torch.zeros(size, dtype=torch.uint8) for name, size in sizes.items()}
    protocol._next_snapshot = {name: torch.full((size,), 7, dtype=torch.uint8) for name, size in sizes.items()}
    protocol._plan = {
        name: {"dtype": "U8", "shape": [size], "views": [], "encoding": "xor_bytes"} for name, size in sizes.items()
    }
    protocol._seen = set(sizes) - ({omit} if omit else set())
    protocol._uncommitted = True
    protocol._encoding_metrics = []
    protocol._gpu_batch_count = 0
    protocol._bulk_encode_s = protocol._encoded_hash_write_s = 0.0
    protocol._raw_tail_wait_s = 0.0
    protocol._raw_cpu_write_s = 0.0
    protocol._staging_stream = object()
    protocol._started = time.monotonic()
    events = []

    class Ready:
        def record(self, stream):
            assert stream is protocol._staging_stream
            events.append("record")

        def synchronize(self):
            events.append("ready")

    def encode(tensors):
        assert "ready" in events
        assert not any(isinstance(event, tuple) and event[0] == "write" for event in events)
        events.append(("encode", [current.numel() for old, current, encoding in tensors]))
        if sum(isinstance(event, tuple) and event[0] == "encode" for event in events) == fail_batch:
            raise RuntimeError("decoder-independent producer failure")
        for old, current, encoding in tensors:
            assert encoding == "xor_bytes"
            assert torch.equal(old, torch.zeros_like(old))
            assert torch.equal(current, torch.full_like(current, 7))
        return [([], [], current.numel(), {"encode_wall_s": 0.01}) for old, current, encoding in tensors]

    def wrap(values):
        assert len(values) == 4
        events.append("wrap-all")
        return [(frames, b"outer", {}, changed, metrics) for frames, _, changed, metrics in values]

    protocol._gpu_encoder = Mock(encode_device=Mock(side_effect=encode), wrap_device=Mock(side_effect=wrap))
    protocol._writer = Mock()
    protocol._writer.outer_metrics = {"outer_hash_write_s": 0.03}
    protocol._writer.add_gpu_outer_tensor.side_effect = lambda name, *args, **kwargs: events.append(("write", name))
    monkeypatch.setattr(gpu_delta.torch.cuda, "Event", Ready)
    protocol._prepare_gpu_schedule()
    return protocol, events


def test_bulk_compression_waits_for_complete_snapshot_and_defers_all_writes(monkeypatch, single_rank):
    protocol, events = _gpu_pending(monkeypatch)
    assert protocol._gpu_batch_names == (("a", "b"), ("c",), ("d",))
    protocol.after_base_weights()
    assert events == [
        "record",
        "ready",
        ("encode", [2, 3]),
        ("encode", [7]),
        ("encode", [1]),
        "wrap-all",
        ("write", "a"),
        ("write", "b"),
        ("write", "c"),
        ("write", "d"),
    ]
    assert protocol._gpu_batch_count == 3
    assert protocol.pending_baseline is protocol._next_snapshot
    assert all(torch.count_nonzero(value) == 0 for value in protocol._snapshot.values())


@pytest.mark.parametrize("fail_raw", [False, True])
def test_raw_cpu_write_bypasses_gpu_batches_and_overlaps_compression(monkeypatch, single_rank, fail_raw):
    protocol, events = _gpu_pending(monkeypatch)
    protocol._plan["scale"] = {"dtype": "F32", "shape": [], "views": [], "encoding": "raw_bytes"}
    protocol._snapshot["scale"] = torch.zeros(4, dtype=torch.uint8)
    protocol._next_snapshot["scale"] = torch.ones(4, dtype=torch.uint8)
    protocol._seen.add("scale")
    protocol._prepare_gpu_schedule()
    raw_started, encoding_started = threading.Event(), threading.Event()
    encode = protocol._gpu_encoder.encode_device.side_effect

    def encode_after_raw_started(tensors):
        assert raw_started.wait(2), "Raw work did not overlap GPU batches"
        encoding_started.set()
        return encode(tensors)

    def raw(name, previous, current, **kwargs):
        assert name == "scale" and kwargs["shape"] == []
        np.testing.assert_array_equal(previous, 0)
        np.testing.assert_array_equal(current, 1)
        raw_started.set()
        assert encoding_started.wait(2), "Raw work serialized the first GPU batch"
        if fail_raw:
            raise OSError("raw owner write failed")
        events.append(("raw", name))

    protocol._gpu_encoder.encode_device.side_effect = encode_after_raw_started
    protocol._writer.add_raw_tensor.side_effect = raw
    assert protocol._raw_names == ("scale",)
    assert protocol._gpu_batch_names == (("a", "b"), ("c",), ("d",))
    if fail_raw:
        with pytest.raises(RuntimeError, match="raw owner write failed"):
            protocol.after_base_weights()
        protocol._writer.close.assert_called_once()
        assert protocol._uncommitted
    else:
        protocol.after_base_weights()
        assert ("raw", "scale") in events and protocol._raw_cpu_write_s > 0
        assert protocol.pending_baseline is protocol._next_snapshot
        protocol._writer.close.assert_not_called()
    assert protocol._gpu_batch_count == 3
    assert not any(call.args[0] == "scale" for call in protocol._writer.add_gpu_outer_tensor.call_args_list)
    assert torch.count_nonzero(protocol._snapshot["scale"]) == 0


def test_cached_gpu_schedule_uses_current_buffers_after_commit(monkeypatch, single_rank):
    protocol, _ = _gpu_pending(monkeypatch)
    schedule = protocol._gpu_batch_names
    protocol.after_base_weights()
    protocol.commit_pending_baseline()
    for snapshot in protocol._next_snapshot.values():
        snapshot.fill_(13)

    def encode(tensors):
        for previous, current, encoding in tensors:
            assert encoding == "xor_bytes"
            assert torch.all(previous == 7) and torch.all(current == 13)
        return [([], [], current.numel(), {"encode_wall_s": 0.01}) for _, current, _ in tensors]

    protocol._gpu_encoder.encode_device.side_effect = encode
    protocol._prepare_gpu_schedule = Mock(side_effect=AssertionError("Update rebuilt immutable owner schedule"))
    protocol._encode_gpu_batches()
    assert protocol._gpu_batch_names is schedule
    protocol._prepare_gpu_schedule.assert_not_called()


@pytest.mark.parametrize("fail_batch,omit", [(2, None), (None, "c")])
def test_bulk_failure_retains_old_baseline_and_never_writes_partial_publication(
    monkeypatch, single_rank, fail_batch, omit
):
    protocol, events = _gpu_pending(monkeypatch, fail_batch=fail_batch, omit=omit)
    with pytest.raises(RuntimeError, match="GPU-delta encoding failed"):
        protocol.after_base_weights()
    assert "ready" in events  # even an incomplete export drains outstanding D2H
    protocol._writer.add_gpu_outer_tensor.assert_not_called()
    protocol._writer.close.assert_called_once()
    assert all(torch.count_nonzero(value) == 0 for value in protocol._snapshot.values())
    assert protocol._uncommitted
    with pytest.raises(RuntimeError, match="automatic replay"):
        protocol.begin_sync(2, None)


@pytest.mark.parametrize("activation_fails", [False, True])
def test_gpu_baseline_swaps_only_after_successful_receiver_activation(monkeypatch, single_rank, activation_fails):
    protocol, _ = _gpu_pending(monkeypatch)
    protocol.after_base_weights()
    old, current = protocol._snapshot, protocol._next_snapshot
    protocol._cohort, protocol.rollout_engines = object(), []

    def publish(version):
        return {
            "summary_counts": dict(tensor_count=4, wire_bytes=1, changed_bytes=13, canonical_bytes=13),
            "manifest_sha256": "test",
            "producer_summary_metrics": {},
        }

    async def activate(*args):
        assert protocol._snapshot is old
        assert protocol.pending_baseline is current
        if activation_fails:
            raise RuntimeError("uncertain receiver resume")

    monkeypatch.setattr(gpu_delta.gpu_delta_metrics, "activation_metrics", lambda value: {})
    monkeypatch.setattr(protocol, "publish", publish)
    monkeypatch.setattr(gpu_delta.gpu_delta_session, "activate_publication", activate)
    if activation_fails:
        with pytest.raises(RuntimeError, match="uncertain receiver resume"):
            protocol.finalize(1)
        assert protocol._snapshot is old and protocol._next_snapshot is current
        assert protocol._uncommitted
    else:
        protocol.finalize(1)
        assert protocol._snapshot is current and protocol._next_snapshot is old
        assert protocol.update_weight_metrics["perf/update_weights_wire_bytes"] == 1
        assert not protocol._uncommitted
