"""Checkpoint filesystem errors propagate to the trainer cell."""

import multiprocessing
import os
from pathlib import Path

import pytest

from miles.backends.training_utils.checkpoint import io
from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("directory creation failed")])
def test_directory_errors_propagate(error, tmp_path, monkeypatch):
    def make_dir(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "mkdir", make_dir)
    with pytest.raises(type(error), match=str(error)) as caught:
        write_checkpoint_dir(tmp_path / "checkpoint", lambda _: None)
    assert caught.value is error


@pytest.mark.parametrize("crash_after_write", [True, False])
def test_a_crashed_overwrite_leaves_nothing_behind(tmp_path, crash_after_write):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_dir(checkpoint, lambda directory: (directory / "old").write_text("old"))

    def overwrite_and_crash():
        def write_shards(directory):
            if crash_after_write:
                (directory / "value").write_text("partial")
            os._exit(73)

        write_checkpoint_dir(checkpoint, write_shards)

    child = multiprocessing.get_context("fork").Process(target=overwrite_and_crash)
    child.start()
    child.join(timeout=10)
    assert child.exitcode == 73
    assert not (checkpoint / "old").exists()

    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("retry"))
    assert (checkpoint / "value").read_text() == "retry"
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_a_rank_that_does_not_see_rank_0s_directory_creates_its_own(tmp_path, monkeypatch):
    """On node-local disks rank 0's mkdir is invisible to the other nodes; their first write found no directory."""
    monkeypatch.setattr(io.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(io.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(io.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(io.dist, "broadcast_object_list", lambda objects, src, group: None)
    monkeypatch.setattr(io.dist, "all_gather_object", lambda gathered, obj, group: gathered.__setitem__(1, obj))
    monkeypatch.setattr(io, "get_gloo_group", lambda: None)
    checkpoint = tmp_path / "node1" / "iter_0000019"

    write_checkpoint_dir(checkpoint, lambda directory: (directory / "shard_1").write_text("rank 1"))

    assert (checkpoint / "shard_1").read_text() == "rank 1"
