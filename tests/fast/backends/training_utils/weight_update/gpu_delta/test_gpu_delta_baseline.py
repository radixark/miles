"""Startup checkpoint version declaration must precede the first rollout."""

import asyncio
import json
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import safetensors.numpy
import torch

from miles.backends.training_utils.weight_update.protocols.gpu_delta import protocol as gpu_delta
from miles.utils.gpu_delta import encoder as gpu_delta_encoder
from miles.utils.gpu_delta import publication as gpu_delta_publication


class _Engine:
    def __init__(self, protocol, events, index, fail=False):
        self.protocol, self.events, self.index, self.fail = protocol, events, index, fail
        self.version = "default"

    async def get_weights_delta_info(self, engine_id):
        return {
            "success": True,
            "participants": [
                {
                    "identity": {"engine_id": engine_id, "rank_id": str(self.index), "host_cache_id": engine_id},
                    "plan": {"tensors": list(self.protocol._plan.values())},
                }
            ],
        }

    async def update_weight_version(self, weight_version):
        assert not self.protocol._baseline_captured
        np.testing.assert_array_equal(self.protocol._snapshot["w"], [1, 2, 3, 4])
        await asyncio.sleep(0.01 if self.index else 0)
        self.events.append(self.index)
        if self.fail:
            raise RuntimeError("engine rejected base declaration")
        self.version = weight_version
        return {"success": True, "new_version": weight_version}


def _setup(tmp_path, fail=False, frame_bytes=gpu_delta_publication.FRAME_BYTES, initial_sync=False):
    safetensors.numpy.save_file({"w": np.array([1, 2, 3, 4], dtype=np.uint8)}, tmp_path / "model.safetensors")
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "delta"),
            custom_update_weight_post_write_path=None,
            update_weight_buffer_size=5,
            update_weight_delta_initial_sync=initial_sync,
        ),
        frame_bytes=frame_bytes,
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
    protocol._cohort = gpu_delta.gpu_delta_session.ReceiverCohort([], (), (), (), "plan")
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
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(custom_update_weight_post_write_path=None, update_weight_delta_initial_sync=False)
    )
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
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(custom_update_weight_post_write_path=None, update_weight_delta_initial_sync=False)
    )
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
@pytest.mark.parametrize("codec", gpu_delta_publication.CODECS)
def test_initial_delta_publishes_loaded_trainer_then_switches_to_cached_update_codec(
    tmp_path, single_rank, monkeypatch, initial_sync, codec
):
    if codec == "snappy-zstd":
        monkeypatch.delenv("GPU_DELTA_CODEC", raising=False)
        monkeypatch.delenv("GPU_DELTA_INITIAL_SYNC_CODEC", raising=False)
        initial_codec = "lz4-zstd"
    else:
        monkeypatch.setenv("GPU_DELTA_CODEC", codec)
        initial_codec = "lz4" if codec == "lz4-zstd" else "snappy-zstd"
        monkeypatch.setenv("GPU_DELTA_INITIAL_SYNC_CODEC", initial_codec)
    # Recovery uses the initial-sync codec even when startup sync is disabled.
    frame_bytes = 1 << 22 if initial_sync else gpu_delta_publication.FRAME_BYTES
    protocol, events = _setup(tmp_path, frame_bytes=frame_bytes, initial_sync=initial_sync)
    monkeypatch.setenv("GPU_DELTA_CODEC", "invalid-after-construction")
    monkeypatch.setenv("GPU_DELTA_INITIAL_SYNC_CODEC", "invalid-after-construction")
    protocol._staging_stream = Mock()
    protocol._next_snapshot = {"w": torch.empty(4, dtype=torch.uint8)}
    encoders = {}

    def make_encoder(device, frame_bytes, codec):
        encoder = Mock(frame_bytes=frame_bytes, finalization_metrics={})
        encoder.finish_device.return_value = []  # This fixture contains only a raw vector.
        encoders[codec] = encoder
        return encoder

    encoder_type = Mock(side_effect=make_encoder)
    monkeypatch.setattr(gpu_delta_encoder, "GpuBatchEncoder", encoder_type)
    monkeypatch.setattr(gpu_delta, "get_data_replica_rank_and_size", lambda *args: (0, 1))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Mock)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(gpu_delta.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(gpu_delta.dist, "gather_object", lambda shard, shards, **kwargs: shards.__setitem__(0, shard))

    first_codec = initial_codec if initial_sync else codec
    protocol.connect(protocol.rollout_engines, None, [0, 1], None, None, None)
    assert set(encoders) == {initial_codec, first_codec}
    assert all(encoder.frame_bytes == frame_bytes for encoder in encoders.values())
    assert protocol.begin_sync(1, _buckets) is initial_sync
    assert sorted(events) == [0, 1]
    if not initial_sync:
        assert not protocol._stream_dir.exists()
        assert protocol.begin_sync(1, _buckets) is True
    assert not protocol._seen  # Baseline export does not consume this update's inventory.
    previous = [1, 2, 3, 4]
    for version, expected_codec in enumerate((first_codec, codec, codec), 1):
        if version > 1:
            assert protocol.begin_sync(version, _buckets) is True
        assert protocol._gpu_encoder is encoders[expected_codec]
        current = [4 * version + n for n in range(1, 5)]
        protocol.send_bucket([("w", torch.tensor(current, dtype=torch.uint8))])
        protocol.after_base_weights()
        publication = protocol.publish()
        assert protocol.codec == publication["codec"] == expected_codec
        assert publication["frame_bytes"] == frame_bytes
        assert publication["plan_digest"] == protocol._cohort.plan_digest
        assert publication["base_version"] == version - 1 and publication["target_version"] == version
        assert (
            protocol.publication_metrics["encoded_hash_write_s"]
            == protocol._writer.payload_metrics["matrix_hash_write_s"]
        )
        assert publication["summary_counts"]["raw_bytes"] == publication["summary_counts"]["wire_bytes"] == 4
        assert (protocol._version_dir / "owner-00000.bin").read_bytes() == bytes(current)
        np.testing.assert_array_equal(protocol._snapshot["w"], previous)
        protocol.commit_pending_baseline()
        np.testing.assert_array_equal(protocol._snapshot["w"], current)
        previous = current
    assert encoder_type.call_count == len({initial_codec, codec})
    assert set(encoders) == {initial_codec, codec}


def _gpu_pending(monkeypatch, fail_batch=None):
    """Real worker threads with CPU copies; native tests cover CUDA ordering."""
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(
            update_weight_buffer_size=5,
            custom_update_weight_post_write_path=None,
            update_weight_delta_initial_sync=False,
        )
    )
    # Baseline callback order intentionally differs from lexical order.
    sizes = {"d": 1, "b": 3, "c": 7, "a": 2}
    protocol._snapshot = {name: torch.zeros(size, dtype=torch.uint8) for name, size in sizes.items()}
    protocol._next_snapshot = {name: torch.empty(size, dtype=torch.uint8) for name, size in sizes.items()}
    protocol._plan = {
        name: {"dtype": "U8", "shape": [size], "views": [], "encoding": "xor_bytes"} for name, size in sizes.items()
    }
    protocol._uncommitted = True
    protocol._cohort = gpu_delta.gpu_delta_session.ReceiverCohort([], (), (), (), "plan")
    protocol._staging_stream = Mock()
    protocol._started = time.monotonic()
    protocol._match_layout = lambda name, tensor: tensor
    events = []
    worker_ready = threading.local()

    class Ready:
        def record(self, stream):
            assert stream is protocol._staging_stream
            self.recorded = True
            events.append("record")

        def synchronize(self):
            assert self.recorded
            events.append("drain")

    def wait_event(ready):
        assert ready.recorded
        worker_ready.value = True
        events.append("wait-event")

    def encode(tensors):
        assert worker_ready.value
        assert not any(isinstance(event, tuple) and event[0] == "write" for event in events)
        events.append(("encode", [current.numel() for old, current in tensors]))
        if sum(isinstance(event, tuple) and event[0] == "encode" for event in events) == fail_batch:
            raise RuntimeError("decoder-independent producer failure")
        for old, current in tensors:
            assert torch.equal(old, torch.zeros_like(old))
            assert torch.equal(current, torch.full_like(current, 7))
        return [([], [], current.numel(), {"encode_wall_s": 0.01}) for old, current in tensors]

    def finish(values):
        assert len(values) == 4
        events.append("finish-all")
        return [(frames, b"outer", {}, changed, metrics) for frames, _, changed, metrics in values]

    protocol._gpu_encoder = Mock(encode_device=Mock(side_effect=encode), finish_device=Mock(side_effect=finish))
    protocol._gpu_encoder.stream.wait_event.side_effect = wait_event
    protocol._writer = Mock()
    protocol._writer.payload_metrics = {"matrix_hash_write_s": 0.03}
    protocol._writer.add_encoded_tensor.side_effect = lambda name, *args, **kwargs: events.append(("write", name))
    monkeypatch.setattr(torch.cuda, "Event", Ready)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    protocol._prepare_gpu_schedule()
    _start_update(protocol)
    return protocol, events


def _start_update(protocol):
    protocol._seen = set()
    protocol._error = None
    protocol._encoding_metrics = []
    protocol._encoding_jobs = []
    protocol._encoding_started = None
    protocol._batch_remaining = [len(names) for names in protocol._gpu_batch_names]
    protocol._encoder_pool = ThreadPoolExecutor(max_workers=1)
    protocol._gpu_batch_count = 0
    protocol._bulk_encode_s = 0.0
    protocol._raw_tail_wait_s = protocol._raw_cpu_write_s = 0.0


def _export(protocol, names=None, value=7):
    for name in protocol._snapshot if names is None else names:
        protocol.send_bucket([(name, torch.full_like(protocol._snapshot[name], value))])


def test_ready_batches_overlap_export_without_waiting_for_earlier_incomplete_batch(monkeypatch, single_rank):
    protocol, events = _gpu_pending(monkeypatch)
    assert protocol._gpu_batch_names == (("d", "b"), ("c",), ("a",))
    encoding_started, later_exported = threading.Event(), threading.Event()
    encode = protocol._gpu_encoder.encode_device.side_effect
    caller = threading.get_ident()

    def encode_while_exporting(tensors):
        assert threading.get_ident() != caller
        encoding_started.set()
        assert later_exported.wait(2)
        return encode(tensors)

    protocol._gpu_encoder.encode_device.side_effect = encode_while_exporting
    _export(protocol, ["d", "c"])
    assert encoding_started.wait(2)  # c is ready; incomplete d+b must not block it.
    assert protocol._seen != protocol._snapshot.keys()
    _export(protocol, ["b", "a"])
    later_exported.set()
    protocol.after_base_weights()
    assert [event for event in events if isinstance(event, tuple) and event[0] == "encode"] == [
        ("encode", [7]),
        ("encode", [1, 3]),
        ("encode", [2]),
    ]
    assert events.index("finish-all") < next(
        i for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == "write"
    )
    assert protocol._gpu_batch_count == 3 and not protocol._encoding_jobs
    assert protocol.pending_baseline is protocol._next_snapshot
    assert all(torch.count_nonzero(value) == 0 for value in protocol._snapshot.values())


@pytest.mark.parametrize("fail_raw", [False, True])
def test_raw_cpu_write_bypasses_gpu_batches_and_overlaps_compression(monkeypatch, single_rank, fail_raw):
    protocol, events = _gpu_pending(monkeypatch)
    protocol._plan["scale"] = {"dtype": "F32", "shape": [], "views": [], "encoding": "raw_bytes"}
    protocol._snapshot["scale"] = torch.zeros(4, dtype=torch.uint8)
    protocol._next_snapshot["scale"] = torch.empty(4, dtype=torch.uint8)
    protocol._prepare_gpu_schedule()
    raw_started, encoding_started = threading.Event(), threading.Event()
    encode = protocol._gpu_encoder.encode_device.side_effect

    def encode_after_raw_started(tensors):
        encoding_started.set()
        assert raw_started.wait(2), "Raw work did not overlap GPU batches"
        return encode(tensors)

    def raw(name, previous, current, **kwargs):
        assert "drain" in events
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
    _export(protocol, ["d", "b", "c", "a"])
    _export(protocol, ["scale"], value=1)
    assert protocol._raw_names == ("scale",)
    if fail_raw:
        with pytest.raises(RuntimeError, match="raw owner write failed"):
            protocol.after_base_weights()
        protocol._writer.close.assert_called_once()
        assert protocol._uncommitted
    else:
        protocol.after_base_weights()
        assert ("raw", "scale") in events and protocol._raw_cpu_write_s > 0
        protocol._writer.close.assert_not_called()
    assert protocol._gpu_batch_count == 3
    assert not any(call.args[0] == "scale" for call in protocol._writer.add_encoded_tensor.call_args_list)
    assert torch.count_nonzero(protocol._snapshot["scale"]) == 0


def test_cached_gpu_schedule_uses_current_buffers_after_commit(monkeypatch, single_rank):
    protocol, _ = _gpu_pending(monkeypatch)
    schedule = protocol._gpu_batch_names
    _export(protocol)
    protocol.after_base_weights()
    protocol.commit_pending_baseline()
    _start_update(protocol)

    def encode(tensors):
        for previous, current in tensors:
            assert torch.all(previous == 7) and torch.all(current == 13)
        return [([], [], current.numel(), {"encode_wall_s": 0.01}) for _, current in tensors]

    protocol._gpu_encoder.encode_device.side_effect = encode
    protocol._prepare_gpu_schedule = Mock(side_effect=AssertionError("Update rebuilt immutable owner schedule"))
    _export(protocol, value=13)
    protocol.after_base_weights()
    assert protocol._gpu_batch_names is schedule
    protocol._prepare_gpu_schedule.assert_not_called()


@pytest.mark.parametrize("fail_batch,omit", [(2, None), (None, "c")])
def test_bulk_failure_retains_old_baseline_and_drains_before_closing(monkeypatch, single_rank, fail_batch, omit):
    protocol, events = _gpu_pending(monkeypatch, fail_batch=fail_batch)
    if fail_batch:
        protocol._gpu_encoder.stream.synchronize.side_effect = RuntimeError("deferred stream failure")
    _export(protocol, [name for name in protocol._snapshot if name != omit])
    jobs = [job for _, job in protocol._encoding_jobs]
    with pytest.raises(RuntimeError, match="GPU-delta encoding failed"):
        protocol.after_base_weights()
    assert "drain" in events
    protocol._gpu_encoder.stream.synchronize.assert_called_once()
    assert not protocol._encoding_jobs and all(job.done() for job in jobs)
    protocol._writer.add_encoded_tensor.assert_not_called()
    protocol._writer.close.assert_called_once()
    assert all(torch.count_nonzero(value) == 0 for value in protocol._snapshot.values())
    assert protocol._uncommitted
    with pytest.raises(RuntimeError, match="automatic replay"):
        protocol.begin_sync(2, None)


@pytest.mark.parametrize("activation_fails", [False, True])
def test_gpu_baseline_swaps_only_after_successful_receiver_activation(monkeypatch, single_rank, activation_fails):
    protocol, _ = _gpu_pending(monkeypatch)
    _export(protocol)
    protocol.after_base_weights()
    old, current = protocol._snapshot, protocol._next_snapshot
    identity = {"rank_id": "rank0", "engine_id": "engine-00000"}
    protocol._cohort = gpu_delta.gpu_delta_session.ReceiverCohort(
        [], (identity,), ((identity,),), ("engine-00000",), "plan"
    )
    protocol.rollout_engines = [object()]
    protocol._committed_incarnations = protocol._incarnations()
    protocol._target_version = 1
    protocol._recovery_encode_s = 0
    monkeypatch.setattr(protocol, "_cache_recovery_payload", lambda: None)

    def publish():
        return {
            "summary_counts": dict(tensor_count=4, wire_bytes=1, changed_bytes=13, canonical_bytes=13),
            "manifest_sha256": "test",
            "base_version": 0,
            "producer_summary_metrics": {},
        }

    async def activate(*args):
        assert protocol._snapshot is old
        assert protocol.pending_baseline is current
        if activation_fails:
            raise RuntimeError("uncertain receiver resume")
        return {"resumed_receipts": [identity]}

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


def _ready_protocol(tmp_path, monkeypatch):
    protocol, _ = _setup(tmp_path)
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **(kw | {"pin_memory": False})))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock)
    monkeypatch.setattr(torch.cuda, "Stream", Mock)
    monkeypatch.setattr(torch.cuda, "Event", Mock)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *a: None)
    monkeypatch.setattr(gpu_delta, "get_data_replica_rank_and_size", lambda *a: (0, 1))
    monkeypatch.setattr(gpu_delta.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(gpu_delta.dist, "gather_object", lambda shard, shards, **kw: shards.__setitem__(0, shard))
    monkeypatch.setattr(gpu_delta.gpu_delta_metrics, "activation_metrics", lambda result: {})

    def encoder(device, codec, frame_bytes):
        instance = Mock(frame_bytes=frame_bytes, finalization_metrics={})
        instance.finish_device.return_value = []
        return instance

    monkeypatch.setattr(gpu_delta_encoder, "GpuBatchEncoder", encoder)
    protocol.connect(protocol.rollout_engines, [1, 1], [0, 1], None, None, None)
    assert protocol.begin_sync(1, _buckets) is False
    return protocol


def _activation_result(cohort):
    return {"resumed_receipts": [{}] if cohort.engine_ids else []}


def test_checkpoint_target_reuse_anchor_retention_and_background_overlap(tmp_path, single_rank, monkeypatch):
    protocol = _ready_protocol(tmp_path, monkeypatch)
    exports = []

    def export(materialize):
        exports.append(True)
        yield from _buckets(materialize)

    checkpoint = tmp_path / "iter_0000001"
    protocol.prepare_checkpoint(checkpoint, 0, 1, export)
    cached = protocol._recovery_payload
    assert exports == [True]
    assert not (checkpoint / "gpu_delta/READY.json").exists()
    protocol.finalize_checkpoint()
    assert (checkpoint / "gpu_delta/READY.json").is_file()
    protocol.finalize_checkpoint()  # No pending Megatron save, no second publication.
    assert protocol.begin_sync(1, export, rollout_id=0)
    assert not protocol.requires_export and exports == [True]
    protocol.after_base_weights()

    async def activate(clients, cohort, publication):
        return _activation_result(cohort)

    monkeypatch.setattr(gpu_delta.gpu_delta_session, "activate_publication", activate)
    protocol.finalize(1)
    assert protocol._recovery_payload is cached
    assert protocol._snapshot is not protocol._hf_snapshot
    assert protocol._next_snapshot == {}
    assert protocol._hf_snapshot["w"].tolist() == [1, 2, 3, 4]
    assert protocol._snapshot["w"].tolist() == [5, 6, 7, 8]

    assert protocol.begin_sync(2, export, rollout_id=1) and protocol.requires_export
    protocol.send_bucket([("w", torch.tensor([9, 10, 11, 12], dtype=torch.uint8))])
    protocol.after_base_weights()
    activation_started = threading.Event()
    recovery_finished = threading.Event()
    encode = protocol._cache_recovery_payload

    async def overlapping_activate(clients, cohort, publication):
        activation_started.set()
        while not recovery_finished.is_set():
            await asyncio.sleep(0.001)
        return _activation_result(cohort)

    def overlapping_encode():
        assert activation_started.wait(2), "HTTP activation must start before recovery encoding finishes"
        encode()
        recovery_finished.set()

    monkeypatch.setattr(gpu_delta.gpu_delta_session, "activate_publication", overlapping_activate)
    monkeypatch.setattr(protocol, "_cache_recovery_payload", overlapping_encode)
    protocol.finalize(2)
    assert protocol._hf_snapshot["w"].tolist() == [1, 2, 3, 4]
    assert protocol._snapshot["w"].tolist() == [9, 10, 11, 12]
    assert cached.raw["w"] == bytes([5, 6, 7, 8])
    assert protocol._recovery_payload.raw["w"] == bytes([9, 10, 11, 12])
    # Ordinary updates retain recovery bytes locally without writing an artifact.
    assert not (protocol._stream_dir / "recovery_v000002").exists()


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_restarted_engine_uses_one_shot_and_retains_cache(tmp_path, single_rank, monkeypatch, recovery_fails):
    protocol = _ready_protocol(tmp_path, monkeypatch)
    calls = []

    async def activate(clients, cohort, publication):
        calls.append((cohort.engine_ids, publication["base_version"], publication["target_version"]))
        return _activation_result(cohort)

    monkeypatch.setattr(gpu_delta.gpu_delta_session, "activate_publication", activate)
    assert protocol.begin_sync(1, _buckets, rollout_id=0)
    protocol.send_bucket(next(_buckets(True)))
    protocol.after_base_weights()
    protocol.finalize(1)
    assert calls == [(("engine-00000", "engine-00001"), 0, 1)]
    calls.clear()

    # Existing Miles FT replaced one engine between ordinary weight updates.
    protocol.rollout_engines[1].index = 2
    protocol.connect(protocol.rollout_engines, [1, 1], [0, 1], None, None, None)
    assert protocol.begin_sync(2, _buckets, rollout_id=1)
    protocol.send_bucket([("w", torch.tensor([9, 10, 11, 12], dtype=torch.uint8))])
    protocol.after_base_weights()
    old, target = protocol._snapshot, protocol._next_snapshot
    loads = []

    async def load(manifest_path, release_state):
        assert not release_state
        assert protocol._snapshot is old and protocol._next_snapshot is target
        manifest = json.loads(Path(manifest_path).read_text())
        loads.append((manifest["base_version"], manifest["target_version"], manifest["codec"]))
        if recovery_fails:
            raise RuntimeError("recovery apply failed")
        return {"success": True, "participants": [{"state": "RESUMED", "target_version": 2}]}

    protocol.rollout_engines[1].update_weights_from_delta = load
    if recovery_fails:
        with pytest.raises(RuntimeError, match="recovery apply failed"):
            protocol.finalize(2)
        assert protocol._snapshot is old and protocol._next_snapshot is target and protocol._uncommitted
    else:
        protocol.finalize(2)
        assert protocol._snapshot is target and not protocol._uncommitted
        assert protocol._committed_incarnations == protocol._incarnations()
        # The retained recovery cache leaves this incarnation on the ordinary
        # rolling path at the next update; the initial-sync RPC is not repeated.
        assert protocol.begin_sync(3, _buckets, rollout_id=2)
        protocol.send_bucket([("w", torch.tensor([13, 14, 15, 16], dtype=torch.uint8))])
        protocol.after_base_weights()
        protocol.finalize(3)
        assert calls[-1] == (("engine-00000", "engine-00001"), 2, 3)
    assert calls[0] == (("engine-00000",), 1, 2)
    assert loads == [(0, 2, "lz4-zstd")]
    assert protocol._hf_snapshot["w"].tolist() == [1, 2, 3, 4]


def test_participant_reply_order_does_not_change_engine_incarnation(tmp_path):
    protocol, _ = _setup(tmp_path)
    cohort_type = gpu_delta.gpu_delta_session.ReceiverCohort
    participants = ({"rank_id": "a", "process": "one"}, {"rank_id": "b", "process": "two"})
    protocol._cohort = cohort_type([], participants, (participants,), ("engine",), "plan")
    initial = protocol._incarnations()
    protocol._cohort = cohort_type([], participants, (participants[::-1],), ("engine",), "plan")
    assert initial == protocol._incarnations()
