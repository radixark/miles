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


@pytest.mark.parametrize("crash_before_publish", [True, False])
def test_crashed_overwrite_keeps_a_complete_checkpoint(tmp_path, crash_before_publish):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("old"))
    old_version = checkpoint.resolve()

    def overwrite_and_crash():
        replace = os.replace

        def crash_at_publish(source, destination):
            if Path(destination) == checkpoint and crash_before_publish:
                os._exit(73)
            replace(source, destination)
            if Path(source) == checkpoint or Path(destination) == checkpoint:
                os._exit(73)

        os.replace = crash_at_publish
        write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("new"))

    child = multiprocessing.get_context("fork").Process(target=overwrite_and_crash)
    child.start()
    child.join(timeout=10)
    assert child.exitcode == 73
    assert (checkpoint / "value").read_text() == ("old" if crash_before_publish else "new")
    assert (old_version / "value").read_text() == "old"

    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("retry"))
    assert (checkpoint / "value").read_text() == "retry"
