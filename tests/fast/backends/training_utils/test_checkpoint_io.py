"""Checkpoint filesystem errors propagate to the trainer cell."""

import multiprocessing
import os
from pathlib import Path

import pytest

from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("directory creation failed")])
def test_directory_errors_propagate(error, tmp_path, monkeypatch):
    def make_dir(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "mkdir", make_dir)
    with pytest.raises(type(error), match=str(error)) as caught:
        write_checkpoint_dir(tmp_path / "checkpoint", lambda _: None)
    assert caught.value is error


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
