"""Startup checkpoint version declaration must precede the first rollout."""

import asyncio
import threading
import time
from argparse import Namespace
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

    async def update_weight_version(self, *, weight_version):
        assert not self.protocol._baseline_captured
        np.testing.assert_array_equal(self.protocol._snapshot["w"], [1, 2, 3, 4])
        await asyncio.sleep(0.01 if self.index else 0)
        self.events.append(self.index)
        if self.fail:
            raise RuntimeError("engine rejected base declaration")
        self.version = weight_version
        return {"success": True, "new_version": weight_version}


def _setup(tmp_path, *, fail=False):
    safetensors.numpy.save_file({"w": np.array([1, 2, 3, 4], dtype=np.uint8)}, tmp_path / "model.safetensors")
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "delta"),
            custom_update_weight_post_write_path=None,
        )
    )
    protocol._plan = {"w": {"name": "w", "dtype": "U8", "shape": [4], "encoding": "raw_bytes"}}
    protocol.is_sender = True
    events = []
    protocol.rollout_engines = [_Engine(protocol, events, 0, fail), _Engine(protocol, events, 1)]
    return protocol, events


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setenv("WEIGHT_DELTA_ENCODER", "cpu")
    with (
        patch.object(gpu_delta, "_gather_all", side_effect=lambda value: [value]),
        patch.object(gpu_delta, "get_gloo_group", return_value=None),
        patch.object(gpu_delta.dist, "get_rank", return_value=0),
        patch.object(gpu_delta.dist, "broadcast_object_list"),
    ):
        yield


def _buckets(*, materialize):
    assert materialize
    # The baseline must use the startup checkpoint, not this exported value.
    yield [("w", torch.tensor([5, 6, 7, 8], dtype=torch.uint8))]


def test_verified_startup_checkpoint_is_zero_on_every_engine_before_rollout(tmp_path, single_rank):
    protocol, events = _setup(tmp_path)
    assert protocol.begin_sync(1, _buckets) is False  # no optimizer/update version increment
    assert sorted(events) == [0, 1]
    assert [engine.version for engine in protocol.rollout_engines] == ["0", "0"]
    assert protocol._baseline_captured and not protocol._uncommitted
    assert not protocol._stream_dir.exists()  # no pretend startup publication


def test_partial_version_acknowledgement_fails_before_rollout_without_replay(tmp_path, single_rank):
    protocol, events = _setup(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="engine rejected base declaration"):
        protocol.begin_sync(1, _buckets)
    assert sorted(events) == [0, 1]  # all submitted RPCs settled before broadcasting failure
    assert not protocol._baseline_captured and protocol._uncommitted
    with pytest.raises(RuntimeError, match="automatic replay is forbidden"):
        protocol.begin_sync(1, _buckets)
    assert len(events) == 2


def test_inventory_failure_never_declares_base_version(tmp_path, single_rank):
    protocol, events = _setup(tmp_path)
    protocol._plan["missing"] = {"name": "missing", "dtype": "U8", "shape": [4]}
    with pytest.raises(RuntimeError, match="inventory/ownership mismatch"):
        protocol.begin_sync(1, _buckets)
    assert not events and not protocol._baseline_captured


@pytest.mark.parametrize("buffer_size", [0, 5])
def test_gpu_startup_partitions_owner_plan_before_declaring_baseline(tmp_path, single_rank, monkeypatch, buffer_size):
    protocol, events = _setup(tmp_path)
    protocol.encoder_backend = "gpu"
    protocol.args.update_weight_buffer_size = buffer_size
    # Only allocation is replaced; startup reads and verifies the real checkpoint.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    if buffer_size == 0:
        with pytest.raises(RuntimeError, match="baseline capture.*positive update_weight_buffer_size"):
            protocol.begin_sync(1, _buckets)
        assert not events and not protocol._baseline_captured
    else:
        assert protocol.begin_sync(1, _buckets) is False
        assert protocol._raw_names == ("w",) and protocol._gpu_batch_names == ()
        assert sorted(events) == [0, 1]


def _gpu_pending(monkeypatch, *, fail_batch=None, omit=None, wrapped=False):
    """Exercise protocol ordering on CPU; native tests cover CUDA encode/copy."""
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(update_weight_buffer_size=5, custom_update_weight_post_write_path=None)
    )
    protocol.encoder_backend = "gpu"
    protocol.codec = "snappy" if wrapped else "zstd"
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
    protocol._outer_cpu_work_s = protocol._outer_tail_wait_s = 0.0
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
        if not wrapped:
            assert not any(isinstance(event, tuple) and event[0] == "write" for event in events)
        events.append(("encode", [current.numel() for old, current, encoding in tensors]))
        if sum(isinstance(event, tuple) and event[0] == "encode" for event in events) == fail_batch:
            raise RuntimeError("decoder-independent producer failure")
        for old, current, encoding in tensors:
            assert encoding == "xor_bytes"
            assert torch.equal(old, torch.zeros_like(old))
            assert torch.equal(current, torch.full_like(current, 7))
        return [([], [], current.numel(), {"encode_wall_s": 0.01}) for old, current, encoding in tensors]

    protocol._gpu_encoder = Mock(encode=Mock(side_effect=encode))
    protocol._writer = Mock()
    protocol._writer.outer_metrics = {"inner_hash_s": 0.02, "outer_hash_write_s": 0.03}
    protocol._writer.add_encoded_tensor.side_effect = lambda name, *args, **kwargs: events.append(("write", name))
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
        ("write", "a"),
        ("write", "b"),
        ("write", "c"),
        ("write", "d"),
    ]
    assert protocol._gpu_batch_count == 3
    assert protocol.pending_baseline is protocol._next_snapshot
    assert all(torch.count_nonzero(value) == 0 for value in protocol._snapshot.values())
    with pytest.raises(RuntimeError, match="successfully published"):
        protocol.commit_pending_baseline()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("fail_raw", [False, True])
def test_raw_cpu_write_bypasses_gpu_batches_and_overlaps_compression(monkeypatch, single_rank, wrapped, fail_raw):
    protocol, events = _gpu_pending(monkeypatch, wrapped=wrapped)
    protocol._plan["scale"] = {"dtype": "F32", "shape": [], "views": [], "encoding": "raw_bytes"}
    protocol._snapshot["scale"] = torch.zeros(4, dtype=torch.uint8)
    protocol._next_snapshot["scale"] = torch.ones(4, dtype=torch.uint8)
    protocol._seen.add("scale")
    protocol._prepare_gpu_schedule()
    raw_started, encoding_started = threading.Event(), threading.Event()
    encode = protocol._gpu_encoder.encode.side_effect

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

    protocol._gpu_encoder.encode.side_effect = encode_after_raw_started
    protocol._writer.add_raw_tensor.side_effect = raw
    assert protocol._raw_names == ("scale",)
    assert protocol._gpu_batch_names == (("a", "b"), ("c",), ("d",))
    if fail_raw:
        with pytest.raises(RuntimeError, match="raw owner write failed"):
            protocol.after_base_weights()
        protocol._writer.close.assert_called_once()
        assert not protocol._pending_ready and protocol._uncommitted
    else:
        protocol.after_base_weights()
        assert ("raw", "scale") in events and protocol._raw_cpu_write_s > 0
        assert protocol.pending_baseline is protocol._next_snapshot
        protocol._writer.close.assert_not_called()
    assert protocol._gpu_batch_count == 3
    assert not any(call.args[0] == "scale" for call in protocol._writer.add_encoded_tensor.call_args_list)
    assert torch.count_nonzero(protocol._snapshot["scale"]) == 0


def test_cached_gpu_schedule_uses_current_buffers_after_commit(monkeypatch, single_rank):
    protocol, _ = _gpu_pending(monkeypatch)
    schedule = protocol._gpu_batch_names
    protocol.after_base_weights()
    protocol._published = True
    protocol.commit_pending_baseline()
    for snapshot in protocol._next_snapshot.values():
        snapshot.fill_(13)

    def encode(tensors):
        for previous, current, encoding in tensors:
            assert encoding == "xor_bytes"
            assert torch.all(previous == 7) and torch.all(current == 13)
        return [([], [], current.numel(), {"encode_wall_s": 0.01}) for _, current, _ in tensors]

    protocol._gpu_encoder.encode.side_effect = encode
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
    protocol._writer.add_encoded_tensor.assert_not_called()
    protocol._writer.close.assert_called_once()
    assert all(torch.count_nonzero(value) == 0 for value in protocol._snapshot.values())
    assert protocol._uncommitted
    with pytest.raises(RuntimeError, match="automatic replay"):
        protocol.begin_sync(2, None)
    with pytest.raises(RuntimeError, match="completed pending target"):
        _ = protocol.pending_baseline


@pytest.mark.parametrize("activation_fails", [False, True])
def test_gpu_baseline_swaps_only_after_successful_receiver_activation(monkeypatch, single_rank, activation_fails):
    protocol, _ = _gpu_pending(monkeypatch)
    protocol.after_base_weights()
    old, current = protocol._snapshot, protocol._next_snapshot
    protocol._descriptions, protocol.rollout_engines = [], []

    def publish(version):
        protocol._published = True
        return {
            "summary_counts": dict(tensor_count=4, wire_bytes=1, changed_bytes=13, canonical_bytes=13),
            "manifest_sha256": "test",
        }

    async def activate(*args):
        assert protocol._snapshot is old
        assert protocol.pending_baseline is current
        if activation_fails:
            raise RuntimeError("uncertain receiver commit")

    monkeypatch.setattr(protocol, "publish", publish)
    monkeypatch.setattr(gpu_delta.gpu_delta_session, "activate_publication", activate)
    if activation_fails:
        with pytest.raises(RuntimeError, match="uncertain receiver commit"):
            protocol.finalize(1)
        assert protocol._snapshot is old and protocol._next_snapshot is current
        assert protocol._uncommitted
    else:
        protocol.finalize(1)
        assert protocol._snapshot is current and protocol._next_snapshot is old
        assert not protocol._uncommitted
        with pytest.raises(RuntimeError, match="successfully published"):
            protocol.commit_pending_baseline()


@pytest.mark.parametrize("cpu_fails,gpu_fails", [(False, False), (True, False), (False, True)])
def test_outer_worker_overlaps_later_gpu_batch_and_drains_before_error_close(
    monkeypatch, single_rank, cpu_fails, gpu_fails
):
    protocol, events = _gpu_pending(monkeypatch, wrapped=True, fail_batch=2 if gpu_fails else None)
    original_encode = protocol._gpu_encoder.encode.side_effect
    write_started, second_encode = threading.Event(), threading.Event()
    active = set()

    def encode(tensors):
        if protocol._gpu_encoder.encode.call_count == 2:
            assert write_started.wait(2), "First CPU write did not overlap later GPU encoding"
            second_encode.set()
        return original_encode(tensors)

    def write(name, frames, payloads, **kwargs):
        active.add(name)
        try:
            if name == "a":
                write_started.set()
                assert second_encode.wait(2), "CPU work was awaited before submitting the next GPU batch"
                if cpu_fails:
                    raise RuntimeError("outer write failed")
            events.append(("write", name))
        finally:
            active.remove(name)

    def close():
        assert not active
        expected = ["b"] if gpu_fails else ["c", "d"]
        # A failed batch stops at its first failed tensor, but later queued
        # batches must still settle before closing their shared owner file.
        assert all(("write", name) in events for name in expected)
        events.append("closed")

    protocol._gpu_encoder.encode.side_effect = encode
    protocol._writer.add_encoded_tensor.side_effect = write
    protocol._writer.close.side_effect = close
    if cpu_fails or gpu_fails:
        with pytest.raises(RuntimeError, match="GPU-delta encoding failed"):
            protocol.after_base_weights()
        assert events[-1] == "closed"
        protocol._writer.finish_shard.assert_not_called()
        assert not protocol._pending_ready and protocol._uncommitted
        assert all(torch.count_nonzero(value) == 0 for value in protocol._snapshot.values())
    else:
        protocol.after_base_weights()
        assert [("write", name) for name in ("a", "b", "c", "d")] == [
            event for event in events if isinstance(event, tuple) and event[0] == "write"
        ]
        assert protocol._outer_cpu_work_s > 0 and protocol._outer_tail_wait_s > 0
        assert protocol._encoded_hash_write_s == 0.05
        assert protocol.pending_baseline is protocol._next_snapshot
        protocol._writer.close.assert_not_called()


def test_cpu_snappy_failure_drains_encoding_workers_before_closing_partial_publication(
    tmp_path, monkeypatch, single_rank
):
    monkeypatch.setenv("WEIGHT_DELTA_CODEC", "snappy")
    protocol = gpu_delta.UpdateWeightFromGpuDelta(Namespace(custom_update_weight_post_write_path=None))
    protocol._snapshot = {name: np.zeros(1000, dtype=np.uint8) for name in ("a", "b")}
    protocol._plan = {
        name: {"dtype": "U8", "shape": [1000], "views": None, "encoding": "xor_bytes"} for name in ("a", "b")
    }
    protocol._seen, protocol._uncommitted = {"a", "b"}, True
    protocol._encoding_metrics = []
    protocol._encoding_tail_wait_s = 0.0
    protocol._writer = gpu_delta.gpu_delta_publication.PublicationWriter(
        tmp_path, stream_id="s", base_version=0, target_version=1, plan_digest="b" * 64, codec=protocol.codec
    )
    writer = protocol._writer
    append = writer._append_outer
    first_failed = threading.Event()

    def append_or_fail(entry, payloads):
        if entry["name"] == "a":
            first_failed.set()
            raise OSError("outer publication write failed")
        append(entry, payloads)

    monkeypatch.setattr(writer, "_append_outer", append_or_fail)
    protocol._pool = ThreadPoolExecutor(max_workers=2)
    ready = Mock(synchronize=Mock())  # Native tests cover the completed export D2H event.
    current = torch.full((1000,), 7, dtype=torch.uint8)
    failed = protocol._pool.submit(protocol._encode_cpu, "a", current, ready, current)
    assert first_failed.wait(2)
    following = protocol._pool.submit(protocol._encode_cpu, "b", current, ready, current)
    protocol._inflight = deque([failed, following])
    close = writer.close

    def close_after_workers():
        assert failed.done() and following.done()
        assert writer._entries.keys() == {"b"}  # Later accepted work finished before close.
        close()

    monkeypatch.setattr(writer, "close", close_after_workers)
    with pytest.raises(RuntimeError, match="GPU-delta encoding failed.*outer publication write failed"):
        protocol.after_base_weights()
    assert writer._closed and (tmp_path / "owner-00000.bin").stat().st_size > 0
    assert not (tmp_path / "manifest.json").exists()
    assert not protocol._pending_ready and protocol._uncommitted
    np.testing.assert_array_equal(protocol._snapshot["a"], 0)
    with pytest.raises(RuntimeError, match="automatic replay"):
        protocol.begin_sync(2, None)


def test_gpu_outer_wraps_owner_only_after_every_inner_batch(monkeypatch, single_rank):
    protocol, events = _gpu_pending(monkeypatch, wrapped=True)
    protocol.snappy_outer = "gpu"
    protocol._gpu_encoder.encode_device.side_effect = protocol._gpu_encoder.encode.side_effect
    protocol._gpu_encoder.outer_metrics = {"outer_gpu_wall_s": 0.01}

    def wrap(values):
        assert len(values) == 4
        assert [event for event in events if isinstance(event, tuple)] == [("encode", [2, 3]), ("encode", [7]), ("encode", [1])]
        events.append("wrap-all")
        return [(frames, b"outer", {}, changed, metrics) for frames, payloads, changed, metrics in values]

    protocol._gpu_encoder.wrap_device.side_effect = wrap
    protocol._writer.add_gpu_outer_tensor.side_effect = lambda name, *args, **kwargs: events.append(("outer-write", name))
    protocol.after_base_weights()
    assert events == ["record", "ready", ("encode", [2, 3]), ("encode", [7]), ("encode", [1]), "wrap-all", ("outer-write", "a"), ("outer-write", "b"), ("outer-write", "c"), ("outer-write", "d")]
    protocol._writer.add_encoded_tensor.assert_not_called()
    assert protocol._gpu_batch_count == 3 and len(protocol._encoding_metrics) == 4
    for tensor in protocol._snapshot.values():
        assert not tensor.any()
