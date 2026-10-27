import json
import subprocess
import sys
from argparse import Namespace
from concurrent.futures import Future

import pytest
import torch
import torch.multiprocessing

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols import shm
from miles.backends.training_utils.weight_update.protocols.shm import UpdateWeightFromSharedMemory
from miles.backends.training_utils.weight_update.protocols.utils import rank_errors
from miles.backends.training_utils.weight_update.protocols.utils.data_replica import OwnedBuckets

_BUCKETS = [
    [
        ("layers.0.w", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ("norm.w", torch.ones(3, dtype=torch.bfloat16)),
    ],
    [("layers.1.w", torch.full((4,), 7.0))],
    [
        ("lm_head.w", torch.full((2, 2), -1.0, dtype=torch.bfloat16)),
        ("lm_head.scale", torch.tensor(-0.5)),
        ("empty", torch.empty(0, 2, dtype=torch.bfloat16)),
    ],
]

# what the engine does with each payload, in a process that shares nothing with the trainer
_ENGINE_SIDE = """
import gc, json, sys
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
rebuilt = []
for serialized in json.load(sys.stdin):
    payload = MultiprocessingSerializer.deserialize(serialized)
    bucket = FlattenedTensorBucket(flattened_tensor=payload["flattened_tensor"], metadata=payload["metadata"])
    rebuilt.append([[n, str(t.dtype), t.float().tolist()] for n, t in bucket.reconstruct_tensors()])
    # release the mapping as a long-lived engine does once it has loaded the bucket
    del payload, bucket
    gc.collect()
json.dump(rebuilt, sys.stdout)
"""


class _SingleRankDist:
    def get_world_size(self) -> int:
        return 1

    def all_gather_object(self, output: list, value, group=None) -> None:
        output[:] = [value]


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rank_errors, "dist", _SingleRankDist())
    monkeypatch.setattr(rank_errors, "get_gloo_group", lambda: None)


class _RecordingCellUpdater:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.on_submit = lambda: None
        self.is_errored = False

    def submit_client_call(self, name: str, **kwargs) -> Future:
        self.on_submit()
        self.calls.append((name, kwargs))
        done: Future = Future()
        done.set_result({"success": True})
        return done


def _protocol(*, replica_rank: int, replica_size: int, tp_sizes: list[int]) -> UpdateWeightFromSharedMemory:
    protocol = UpdateWeightFromSharedMemory(Namespace())
    protocol.cell_updaters_of_cell_id = {f"cell-{i}": _RecordingCellUpdater() for i in range(len(tp_sizes))}
    protocol._engine_tp_sizes = tp_sizes
    protocol._selector = "all"
    protocol._owned_buckets = OwnedBuckets(replica_rank=replica_rank, replica_size=replica_size)
    protocol._max_in_flight = max(1, shm._MAX_BUCKETS_IN_FLIGHT_PER_HOST // replica_size)
    return protocol


def _publish(protocol: UpdateWeightFromSharedMemory) -> None:
    protocol.begin_sync(weight_version=1, iter_buckets=None)
    for bucket in _BUCKETS:
        protocol.send_bucket(bucket)
    protocol.after_base_weights()


def _rebuild_in_another_process(payloads: list[str]) -> list:
    result = subprocess.run(
        [sys.executable, "-c", _ENGINE_SIDE], input=json.dumps(payloads), capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_another_process_rebuilds_every_tp_rank_bucket():
    """The engine maps the published host copies by name, so they must outlive the call that hands them over."""
    protocol = _protocol(replica_rank=0, replica_size=1, tp_sizes=[1, 2])
    _publish(protocol)

    for updater, tp_size in zip(protocol.cell_updaters_of_cell_id.values(), [1, 2], strict=True):
        assert [name for name, _ in updater.calls] == ["update_weights_from_tensor"] * len(_BUCKETS)
        assert all(kwargs["load_format"] == "flattened_bucket" for _, kwargs in updater.calls)
        assert all(len(kwargs["serialized_named_tensors"]) == tp_size for _, kwargs in updater.calls)
        payloads = [kwargs["serialized_named_tensors"][-1] for _, kwargs in updater.calls]
        expected = [[[n, str(t.dtype), t.float().tolist()] for n, t in bucket] for bucket in _BUCKETS]
        assert _rebuild_in_another_process(payloads) == expected
    protocol.finalize(weight_version=1)


def test_a_replica_rank_publishes_only_its_own_buckets():
    protocol = _protocol(replica_rank=1, replica_size=2, tp_sizes=[1])
    _publish(protocol)

    (updater,) = protocol.cell_updaters_of_cell_id.values()
    (payload,) = updater.calls[0][1]["serialized_named_tensors"]
    assert len(updater.calls) == 1
    assert [name for name, _, _ in _rebuild_in_another_process([payload])[0]] == ["layers.1.w"]


def test_a_sync_keeps_at_most_the_window_of_host_copies_alive(monkeypatch):
    """Host memory must stay a few buckets, not the whole model, however many buckets a sync sends."""
    monkeypatch.setattr(shm, "_MAX_BUCKETS_IN_FLIGHT_PER_HOST", 2)
    protocol = _protocol(replica_rank=0, replica_size=1, tp_sizes=[1])
    (updater,) = protocol.cell_updaters_of_cell_id.values()
    alive_at_submit = []
    updater.on_submit = lambda: alive_at_submit.append(len(protocol._in_flight))
    _publish(protocol)

    assert len(updater.calls) == len(_BUCKETS) > 2
    # the bucket being published joins the ones already in flight
    assert max(alive_at_submit) + 1 == 2
    protocol.finalize(weight_version=1)
    assert not protocol._in_flight


def test_every_receiving_process_can_open_the_bucket_after_the_others_released_it():
    """A receiver releasing the file must not unlink it for the TP ranks that open it later."""
    protocol = _protocol(replica_rank=0, replica_size=1, tp_sizes=[3])
    _publish(protocol)

    (updater,) = protocol.cell_updaters_of_cell_id.values()
    first_bucket = updater.calls[0][1]["serialized_named_tensors"]
    expected = [[[n, str(t.dtype), t.float().tolist()] for n, t in _BUCKETS[0]]]
    # one process per TP rank, one after another, each releasing the file before the next opens it
    for payload in first_bucket:
        assert _rebuild_in_another_process([payload]) == expected
    protocol.finalize(weight_version=1)


def test_the_window_is_shared_by_the_hosts_replica_ranks():
    """Each rank holds its share, so /dev/shm stays near the per-host bound however many ranks publish."""
    assert (
        _protocol(replica_rank=0, replica_size=1, tp_sizes=[1])._max_in_flight == shm._MAX_BUCKETS_IN_FLIGHT_PER_HOST
    )
    assert (
        _protocol(replica_rank=0, replica_size=2, tp_sizes=[1])._max_in_flight
        == shm._MAX_BUCKETS_IN_FLIGHT_PER_HOST // 2
    )
    assert _protocol(replica_rank=0, replica_size=64, tp_sizes=[1])._max_in_flight == 1


def test_a_sync_leaves_the_process_sharing_strategy_alone():
    """The strategy is process-wide; weight sync must not change it for the trainer's other torch.multiprocessing users."""
    before = torch.multiprocessing.get_sharing_strategy()
    protocol = _protocol(replica_rank=0, replica_size=1, tp_sizes=[1])
    _publish(protocol)
    protocol.finalize(weight_version=1)
    assert torch.multiprocessing.get_sharing_strategy() == before


def test_an_errored_cell_gets_no_payload(monkeypatch):
    """Each payload holds a reference to the /dev/shm file until an engine opens it, so one sent nowhere leaks it."""
    protocol = _protocol(replica_rank=0, replica_size=1, tp_sizes=[1, 2])
    healthy, errored = protocol.cell_updaters_of_cell_id.values()
    errored.is_errored = True
    serialize_calls = []
    serialize = shm.MultiprocessingSerializer.serialize
    monkeypatch.setattr(
        shm.MultiprocessingSerializer,
        "serialize",
        lambda *args, **kwargs: serialize_calls.append(1) or serialize(*args, **kwargs),
    )
    _publish(protocol)
    protocol.finalize(weight_version=1)

    assert len(healthy.calls) == len(_BUCKETS) and not errored.calls
    assert len(serialize_calls) == len(_BUCKETS)


def test_only_engines_on_this_host_can_open_the_buckets():
    # 192.0.2.1 and 2001:db8::1 are reserved for documentation (RFC 5737, RFC 3849): no interface holds them
    hosts = ("127.0.0.1", "localhost", "192.0.2.1", "[2001:db8::1]")
    engines = [SGLangApiClient(f"http://{host}:30000") for host in hosts]
    assert shm._engines_off_this_host(engines) == ["http://192.0.2.1:30000", "http://[2001:db8::1]:30000"]
