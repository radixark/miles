import json
import subprocess
import sys
from argparse import Namespace
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols import disk
from miles.backends.training_utils.weight_update.protocols.disk import UpdateWeightFromDisk
from miles.backends.training_utils.weight_update.protocols.utils import rank_errors
from miles.backends.training_utils.weight_update.protocols.utils.data_replica import OwnedBuckets
from miles.backends.training_utils.weight_update.protocols.utils.owner import owner_tag

_BUCKETS = [
    [("layers.0.w", torch.full((2, 3), 0.0)), ("layers.0.b", torch.full((3,), 0.5))],
    [("layers.1.w", torch.full((2, 3), 1.0))],
    [("layers.2.w", torch.full((2, 3), 2.0)), ("norm.w", torch.ones(3, dtype=torch.bfloat16))],
    [("lm_head.w", torch.full((4, 3), 3.0))],
]


class _RecordingCellUpdater:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self._calls = calls

    def submit_client_call(self, name: str, **kwargs) -> Future:
        self._calls.append((name, kwargs))
        done: Future = Future()
        done.set_result({"success": True})
        return done


class _FakeDist:
    """One rank of a world whose all_gather_object returns the maps every rank wrote."""

    def __init__(self, rank: int, world_size: int, gathered: list | None = None) -> None:
        self._rank, self._world_size, self._gathered = rank, world_size, gathered

    def get_rank(self) -> int:
        return self._rank

    def get_world_size(self) -> int:
        return self._world_size

    def barrier(self, group=None) -> None:
        pass

    def all_gather_object(self, output: list, value, group=None) -> None:
        output[:] = self._gathered if self._gathered is not None else [value]


def _protocol(tmp_path: Path, *, replica_rank: int, replica_size: int, calls: list) -> UpdateWeightFromDisk:
    hf_checkpoint = tmp_path / "hf"
    hf_checkpoint.mkdir(exist_ok=True)
    (hf_checkpoint / "config.json").write_text("{}")
    (hf_checkpoint / "model.safetensors.index.json").write_text("{}")
    (hf_checkpoint / "modeling_custom.py").write_text("")
    (hf_checkpoint / "tf_model.h5").write_text("")
    args = Namespace(
        update_weight_disk_dir=str(tmp_path / "sync"),
        hf_checkpoint=str(hf_checkpoint),
        pause_generation_mode="retract",
        fully_async=False,
        colocate=False,
    )
    protocol = UpdateWeightFromDisk(args)
    protocol.run_dir = str(tmp_path / "sync" / "run-this")
    protocol._owned_buckets = OwnedBuckets(replica_rank=replica_rank, replica_size=replica_size)
    protocol.cell_updaters_of_cell_id = {"cell-0": _RecordingCellUpdater(calls)}
    return protocol


def _use_dist(monkeypatch: pytest.MonkeyPatch, fake_dist: _FakeDist) -> None:
    monkeypatch.setattr(disk, "dist", fake_dist)
    monkeypatch.setattr(rank_errors, "dist", fake_dist)


def _write(protocol: UpdateWeightFromDisk, monkeypatch: pytest.MonkeyPatch, *, rank: int, weight_version: int) -> None:
    _use_dist(monkeypatch, _FakeDist(rank, world_size=2))
    protocol.begin_sync(weight_version, iter_buckets=None)
    for bucket in _BUCKETS:
        protocol.send_bucket(list(bucket))
    protocol.after_base_weights()


@pytest.fixture(autouse=True)
def _no_gloo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disk, "get_gloo_group", lambda: None)
    monkeypatch.setattr(rank_errors, "get_gloo_group", lambda: None)


def test_the_replica_ranks_write_every_tensor_once_and_the_index_finds_it(tmp_path, monkeypatch):
    """Byte-balanced ownership must neither drop nor duplicate a bucket, and the merged index must find its shard."""
    calls: list = []
    ranks = [_protocol(tmp_path, replica_rank=rank, replica_size=2, calls=calls) for rank in range(2)]
    for rank, protocol in enumerate(ranks):
        _write(protocol, monkeypatch, rank=rank, weight_version=3)

    _use_dist(monkeypatch, _FakeDist(0, world_size=2, gathered=[p._weight_map for p in ranks]))
    ranks[0]._write_index({name: shard for p in ranks for name, shard in p._weight_map.items()})

    version_dir = Path(ranks[0]._version_dir)
    weight_map = json.loads((version_dir / "model.safetensors.index.json").read_text())["weight_map"]
    expected = dict(name_tensor for bucket in _BUCKETS for name_tensor in bucket)
    assert sorted(weight_map) == sorted(expected)
    for name, shard in weight_map.items():
        torch.testing.assert_close(safetensors.torch.load_file(version_dir / shard)[name], expected[name])
    assert {shard.split("-")[0] for shard in set(weight_map.values())} == {"rank000", "rank001"}
    assert (version_dir / "config.json").exists()
    assert (version_dir / "modeling_custom.py").exists(), "trust-remote-code models load their code from here"
    assert not (version_dir / "tf_model.h5").exists(), "a weight file in another format would duplicate the model"


def test_the_engines_reload_the_new_version_between_pause_and_resume(tmp_path, monkeypatch):
    calls: list = []
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, calls=calls)
    _write(protocol, monkeypatch, rank=0, weight_version=3)
    _use_dist(monkeypatch, _FakeDist(0, world_size=1))
    protocol.finalize(weight_version=3)

    assert [name for name, _ in calls] == [
        "pause_generation",
        "flush_cache",
        "update_weights_from_disk",
        "continue_generation",
    ]
    assert calls[2][1] == {"model_path": protocol._version_dir, "load_format": "safetensors", "weight_version": "3"}


def test_a_sync_deletes_the_version_before_it(tmp_path, monkeypatch):
    """Each sync writes a full checkpoint, so keeping old ones fills the shared filesystem."""
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, calls=[])
    versions = []
    for weight_version in (1, 2):
        _write(protocol, monkeypatch, rank=0, weight_version=weight_version)
        _use_dist(monkeypatch, _FakeDist(0, world_size=1))
        protocol.finalize(weight_version=weight_version)
        versions.append(Path(protocol._version_dir))

    assert not versions[0].exists()
    assert versions[1].exists()


def test_a_failed_engine_reload_raises(tmp_path, monkeypatch):
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, calls=[])

    def _reject(name, **kwargs):
        done: Future = Future()
        done.set_result({"success": name != "update_weights_from_disk", "error_message": "bad shard"})
        return done

    protocol.cell_updaters_of_cell_id = {"cell-0": SimpleNamespace(submit_client_call=_reject)}
    _write(protocol, monkeypatch, rank=0, weight_version=1)
    _use_dist(monkeypatch, _FakeDist(0, world_size=1))
    with pytest.raises(RuntimeError, match="bad shard"):
        protocol.finalize(weight_version=1)


def test_the_first_sync_deletes_the_runs_that_ended_on_this_host_and_no_live_or_remote_one(tmp_path, monkeypatch):
    """A run ends by a kill and leaves its last version behind; only a run whose trainer is gone may lose it."""
    host, namespace, _ = owner_tag().rsplit("-", 2)
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    sync_dir = tmp_path / "sync"
    for tag in (f"{host}-{namespace}-{ended.pid}", owner_tag(), f"other-host-{namespace}-{ended.pid}"):
        (sync_dir / f"run-{tag}" / "weight_v000004").mkdir(parents=True)
    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, calls=[])
    _write(protocol, monkeypatch, rank=0, weight_version=3)

    assert sorted(path.name for path in sync_dir.iterdir()) == sorted(
        ["run-this", f"run-{owner_tag()}", f"run-other-host-{namespace}-{ended.pid}"]
    )


def test_a_version_that_cannot_be_deleted_is_reported_and_the_sync_goes_on(tmp_path, monkeypatch, caplog):
    """A version dir left behind is a full checkpoint on shared storage, so it must not vanish silently."""

    def _read_only(path):
        raise PermissionError("read-only")

    protocol = _protocol(tmp_path, replica_rank=0, replica_size=1, calls=[])
    for weight_version in (1, 2):
        _write(protocol, monkeypatch, rank=0, weight_version=weight_version)
        _use_dist(monkeypatch, _FakeDist(0, world_size=1))
        if weight_version == 2:
            monkeypatch.setattr(disk.shutil, "rmtree", _read_only)
        protocol.finalize(weight_version=weight_version)

    assert "could not delete" in caplog.text and "weight_v000001" in caplog.text


def test_a_version_directory_rank_0_cannot_create_fails_every_rank(tmp_path, monkeypatch):
    """Only rank 0 creates it; the other ranks must not walk into a collective rank 0 never joins."""
    protocol = _protocol(tmp_path, replica_rank=1, replica_size=2, calls=[])
    _use_dist(monkeypatch, _FakeDist(1, world_size=2, gathered=["OSError: No space left on device", None]))
    with pytest.raises(RuntimeError, match="version directory failed on rank 0: OSError: No space left on device"):
        protocol.begin_sync(1, iter_buckets=None)
