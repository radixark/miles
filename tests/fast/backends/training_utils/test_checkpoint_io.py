"""Checkpoint filesystem errors propagate to the trainer cell."""

import multiprocessing
import os
from pathlib import Path

import pytest

from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir
from miles.utils import distributed_utils
from miles.utils.distributed_utils import RankFailureError


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("directory creation failed")])
def test_directory_errors_propagate(error, tmp_path, monkeypatch):
    def make_dir(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "mkdir", make_dir)
    with pytest.raises(type(error), match=str(error)) as caught:
        write_checkpoint_dir(tmp_path / "checkpoint", lambda _: None)
    assert caught.value is error


def _as_rank0_of_two(monkeypatch, all_gather_object):
    monkeypatch.setattr(distributed_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(distributed_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(distributed_utils.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", object())


def _peer_fails_at(step):
    """Rank 1 reports an error at the given 0-based exchange (prepare, write, finalize)."""
    calls = []

    def all_gather_object(output, local_message, group):
        calls.append(local_message)
        output[:] = [local_message, "OSError: disk full" if len(calls) - 1 == step else None]

    return all_gather_object


@pytest.mark.parametrize("step", [1, 2])
def test_a_peer_failure_removes_the_checkpoint(tmp_path, monkeypatch, step):
    checkpoint = tmp_path / "checkpoint"
    _as_rank0_of_two(monkeypatch, _peer_fails_at(step))

    with pytest.raises(RankFailureError, match="rank 1: OSError: disk full"):
        write_checkpoint_dir(checkpoint, lambda directory: (directory / "shard").write_text("x"), metadata={"step": 1})

    assert not checkpoint.exists()


def test_a_finalize_failure_on_rank0_reaches_every_rank(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    _as_rank0_of_two(monkeypatch, _peer_fails_at(None))

    def fail_touch(self, *args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(Path, "touch", fail_touch)

    with pytest.raises(RankFailureError, match="rank 0: OSError: read-only"):
        write_checkpoint_dir(checkpoint, lambda directory: None, completion_marker=".complete")

    assert not checkpoint.exists()


def test_a_failed_error_exchange_keeps_the_directory_for_live_writers(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    calls = 0

    def all_gather_object(output, local_message, group):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("gloo connection reset")
        output[:] = [None, None]

    _as_rank0_of_two(monkeypatch, all_gather_object)

    with pytest.raises(RuntimeError, match="gloo connection reset"):
        write_checkpoint_dir(checkpoint, lambda directory: (directory / "shard").write_text("x"))

    assert (checkpoint / "shard").exists()


@pytest.mark.parametrize("crash_after_write", [True, False])
def test_crashed_overwrite_leaves_no_metadata(tmp_path, crash_after_write):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_dir(checkpoint, lambda directory: (directory / "old").write_text("old"), metadata={"step": 1})

    def overwrite_and_crash():
        def write_shards(directory):
            if crash_after_write:
                (directory / "value").write_text("partial")
            os._exit(73)

        write_checkpoint_dir(checkpoint, write_shards, metadata={"step": 2})

    child = multiprocessing.get_context("fork").Process(target=overwrite_and_crash)
    child.start()
    child.join(timeout=10)
    assert child.exitcode == 73
    assert not (checkpoint / "old").exists()
    assert not (checkpoint / "META.json").exists()

    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("retry"), metadata={"step": 2})
    assert (checkpoint / "value").read_text() == "retry"
    assert (checkpoint / "META.json").exists()
    assert list(tmp_path.iterdir()) == [checkpoint]
