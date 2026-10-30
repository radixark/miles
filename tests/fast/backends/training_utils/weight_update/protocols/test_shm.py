import errno
import json
import os
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
from miles.backends.training_utils.weight_update.protocols.utils.data_replica import OwnedBuckets, pack_by_dtype
from miles.backends.training_utils.weight_update.protocols.utils.owner import owner_tag

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


@pytest.fixture(autouse=True)
def pinned(monkeypatch: pytest.MonkeyPatch) -> dict[int, torch.Tensor]:
    """The host tensors the platform has page-locked, by address; CUDA's registration needs a GPU."""
    pinned: dict[int, torch.Tensor] = {}

    def pin(tensor: torch.Tensor) -> None:
        assert tensor.data_ptr() not in pinned, "pinned twice"
        pinned[tensor.data_ptr()] = tensor

    monkeypatch.setattr(shm.current_platform, "pin_host_memory", pin)
    monkeypatch.setattr(shm.current_platform, "unpin_host_memory", lambda tensor: pinned.pop(tensor.data_ptr()))
    return pinned


class _RecordingCellUpdater:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.is_errored = False

    def submit_client_call(self, name: str, **kwargs) -> Future:
        self.calls.append((name, kwargs))
        done: Future = Future()
        done.set_result({"success": True})
        return done


def _protocol(
    tmp_path, *, replica_rank: int, replica_size: int, gpu_counts: list[int]
) -> UpdateWeightFromSharedMemory:
    protocol = UpdateWeightFromSharedMemory(Namespace())
    protocol.cell_updaters_of_cell_id = {f"cell-{i}": _RecordingCellUpdater() for i in range(len(gpu_counts))}
    protocol._engine_gpu_counts = gpu_counts
    protocol._selector = "all"
    # one manifest dir per rank: in a run each rank is its own process, with its own manifest
    (tmp_path / f"rank{replica_rank}").mkdir(exist_ok=True)
    protocol._image = shm._SharedImage(str(tmp_path / f"rank{replica_rank}"))
    protocol._owned_buckets = OwnedBuckets(
        replica_rank=replica_rank, replica_size=replica_size, host_buffer=protocol._image.host_buffer
    )
    return protocol


def _publish(protocol: UpdateWeightFromSharedMemory, buckets: list = _BUCKETS) -> None:
    protocol.begin_sync(weight_version=1, iter_buckets=None)
    for bucket in buckets:
        protocol.send_bucket(list(bucket))
    protocol.after_base_weights()


def _published(updater: _RecordingCellUpdater) -> list[str]:
    return [kwargs["serialized_named_tensors"][0] for _, kwargs in updater.calls]


def _expected(buckets: list) -> list:
    # each bucket arrives ordered by dtype
    return [[[n, str(t.dtype), t.float().tolist()] for n, t in pack_by_dtype(bucket)] for bucket in buckets]


def _rebuild_in_another_process(payloads: list[str]) -> list:
    result = subprocess.run(
        [sys.executable, "-c", _ENGINE_SIDE], input=json.dumps(payloads), capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_another_process_rebuilds_every_tp_rank_bucket(tmp_path):
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, gpu_counts=[1, 2])
    _publish(protocol)

    for updater, gpu_count in zip(protocol.cell_updaters_of_cell_id.values(), [1, 2], strict=True):
        assert [name for name, _ in updater.calls] == ["update_weights_from_tensor"] * len(_BUCKETS)
        assert all(kwargs["load_format"] == "flattened_bucket" for _, kwargs in updater.calls)
        assert all(len(kwargs["serialized_named_tensors"]) == gpu_count for _, kwargs in updater.calls)
        payloads = [kwargs["serialized_named_tensors"][-1] for _, kwargs in updater.calls]
        assert _rebuild_in_another_process(payloads) == _expected(_BUCKETS)
    protocol.finalize(weight_version=1)


def test_the_replica_ranks_publish_every_bucket_exactly_once(tmp_path):
    published = []
    for replica_rank in range(2):
        protocol = _protocol(tmp_path, replica_rank=replica_rank, replica_size=2, gpu_counts=[1])
        _publish(protocol)
        (updater,) = protocol.cell_updaters_of_cell_id.values()
        assert _published(updater), f"replica rank {replica_rank} published nothing"
        published += _rebuild_in_another_process(_published(updater))
        protocol.finalize(weight_version=1)

    assert sorted(published) == sorted(_expected(_BUCKETS))


def test_a_later_sync_rewrites_the_same_shared_buckets_in_place(tmp_path, pinned):
    """No sync after the first allocates, pins or faults in host memory; the engine still reads the new weights."""
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, gpu_counts=[1])
    (updater,) = protocol.cell_updaters_of_cell_id.values()
    _publish(protocol)
    protocol.finalize(weight_version=1)
    addresses = [flat.data_ptr() for flat in protocol._image._buckets.values()]

    updater.calls.clear()
    stepped = [[(name, tensor + 1) for name, tensor in bucket] for bucket in _BUCKETS]
    _publish(protocol, stepped)

    assert [flat.data_ptr() for flat in protocol._image._buckets.values()] == addresses
    assert sorted(pinned) == sorted(addresses)
    assert _rebuild_in_another_process(_published(updater)) == _expected(stepped)
    protocol.finalize(weight_version=2)


def test_a_bucket_that_changed_size_gets_a_new_shared_tensor(tmp_path, pinned):
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, gpu_counts=[1])
    (updater,) = protocol.cell_updaters_of_cell_id.values()
    _publish(protocol)
    protocol.finalize(weight_version=1)
    first = protocol._image.flat(1)

    updater.calls.clear()
    resized = [_BUCKETS[0], [("layers.1.w", torch.full((6,), 3.0))], _BUCKETS[2]]
    _publish(protocol, resized)

    assert protocol._image.flat(1).data_ptr() != first.data_ptr()
    assert first.data_ptr() not in pinned, "the replaced tensor stays page-locked"
    assert len(pinned) == len(_BUCKETS)
    assert _rebuild_in_another_process(_published(updater)) == _expected(resized)
    protocol.finalize(weight_version=2)


def test_a_full_dev_shm_fails_every_rank_after_the_stream_not_mid_gather(tmp_path, monkeypatch):
    """The gathers are collective: a rank leaving the stream early would leave the others waiting in them."""

    def full(fd, offset, length):
        raise OSError(errno.ENOSPC, "No space left on device")

    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, gpu_counts=[1])
    (updater,) = protocol.cell_updaters_of_cell_id.values()
    monkeypatch.setattr(shm.os, "posix_fallocate", full)
    protocol.begin_sync(weight_version=1, iter_buckets=None)
    for bucket in _BUCKETS:
        protocol.send_bucket(list(bucket))  # each bucket of the stream is still consumed

    with pytest.raises(RuntimeError, match=r"bucket publish failed on rank 0: OSError: .*No space left"):
        protocol.after_base_weights()
    assert not updater.calls


def test_an_engine_that_never_opens_a_payload_costs_no_new_shared_memory(tmp_path):
    """Unopened payloads keep a reference to their file; the next sync must reuse that file, not add another."""
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, gpu_counts=[1])
    for weight_version in (1, 2, 3):
        _publish(protocol)  # the recording engine opens nothing
        protocol.finalize(weight_version=weight_version)

    listed = json.loads((tmp_path / "rank0" / f"miles-shm-{owner_tag()}.manifest").read_text())
    assert len(listed) == len(_BUCKETS)
    assert all(os.path.exists(path) for path in listed)


def test_a_new_trainer_deletes_the_files_of_trainers_that_are_gone_and_no_live_ones(tmp_path):
    """Ray kills a trainer and its shm manager together, so the next trainer on the host has to clean up after it."""
    host, namespace, _ = owner_tag().rsplit("-", 2)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    owners = [f"{host}-{namespace}-{gone.pid}", owner_tag(), f"other-host-{namespace}-{gone.pid}"]
    for owner in owners:
        bucket = tmp_path / f"torch_{owner}"
        bucket.write_bytes(b"bucket")
        manifest = tmp_path / f"miles-shm-{owner}.manifest"
        manifest.write_text(json.dumps([str(bucket), str(tmp_path / "already_gone")]))

    shm._SharedImage(str(tmp_path))

    kept = [name for owner in owners[1:] for name in (f"torch_{owner}", f"miles-shm-{owner}.manifest")]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(kept)


def test_every_receiving_process_can_open_the_bucket_after_the_others_released_it(tmp_path):
    """A receiver releasing the file must not unlink it for the TP ranks that open it later."""
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, gpu_counts=[3])
    _publish(protocol)

    (updater,) = protocol.cell_updaters_of_cell_id.values()
    # one process per TP rank, one after another, each releasing the file before the next opens it
    for payload in updater.calls[0][1]["serialized_named_tensors"]:
        assert _rebuild_in_another_process([payload]) == _expected(_BUCKETS[:1])
    protocol.finalize(weight_version=1)


def test_a_sync_leaves_the_process_sharing_strategy_alone(tmp_path):
    """The strategy is process-wide; weight sync must not change it for the trainer's other torch.multiprocessing users."""
    before = torch.multiprocessing.get_sharing_strategy()
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, gpu_counts=[1])
    _publish(protocol)
    protocol.finalize(weight_version=1)
    assert torch.multiprocessing.get_sharing_strategy() == before


def test_an_errored_cell_gets_no_payload(tmp_path):
    """Each payload holds a reference to the /dev/shm file until an engine opens it, so none goes where no one will."""
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, gpu_counts=[1, 2])
    healthy, errored = protocol.cell_updaters_of_cell_id.values()
    errored.is_errored = True
    _publish(protocol)
    protocol.finalize(weight_version=1)

    assert len(healthy.calls) == len(_BUCKETS) and not errored.calls


def test_only_engines_on_this_host_can_open_the_buckets():
    # 192.0.2.1 and 2001:db8::1 are reserved for documentation (RFC 5737, RFC 3849): no interface holds them
    hosts = ("127.0.0.1", "localhost", "192.0.2.1", "[2001:db8::1]")
    engines = [SGLangApiClient(f"http://{host}:30000") for host in hosts]
    assert shm._engines_off_this_host(engines) == ["http://192.0.2.1:30000", "http://[2001:db8::1]:30000"]


def test_an_unresolvable_engine_host_counts_as_elsewhere():
    # .invalid never resolves (RFC 2606)
    assert shm._engines_off_this_host([SGLangApiClient("http://engine.invalid:30000")]) == [
        "http://engine.invalid:30000"
    ]


def test_the_payload_survives_sglangs_cuda_only_tensor_reducer(tmp_path, monkeypatch):
    """Megatron's weight iterator installs sglang's reducer, which treats every tensor as CUDA; shm tensors are CPU."""
    from multiprocessing.reduction import ForkingPickler

    from torch.multiprocessing import reductions

    def cuda_only_reducer(tensor):
        raise IndexError("tuple index out of range")

    # an earlier test may have installed sglang's patch, which already saved torch's reducer there
    if not hasattr(reductions, "_reduce_tensor_original"):
        monkeypatch.setattr(reductions, "_reduce_tensor_original", reductions.reduce_tensor, raising=False)
    monkeypatch.setitem(ForkingPickler._extra_reducers, torch.Tensor, cuda_only_reducer)
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, gpu_counts=[1])
    _publish(protocol)

    (updater,) = protocol.cell_updaters_of_cell_id.values()
    assert _rebuild_in_another_process(_published(updater)) == _expected(_BUCKETS)
    protocol.finalize(weight_version=1)
