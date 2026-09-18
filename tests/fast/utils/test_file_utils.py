import os
import stat
import tempfile
from pathlib import Path

import pytest

from miles.utils.file_utils import atomic_write_text


class TestAtomicWriteText:
    def test_leaves_no_temporary_file_behind(self, tmp_path):
        """The write goes through a rename, and a leftover temp file would confuse a reader listing the directory."""
        path = tmp_path / "state.json"

        atomic_write_text(path, "{}")

        assert [entry.name for entry in tmp_path.iterdir()] == ["state.json"]

    def test_two_writers_of_the_same_pid_do_not_share_a_temporary(self, tmp_path, monkeypatch):
        """The launcher and the wrapper sit in different pid namespaces over one directory."""
        path = tmp_path / "state.json"
        names = set()
        real_mkstemp = tempfile.mkstemp

        def recording_mkstemp(**kwargs):
            handle, name = real_mkstemp(**kwargs)
            names.add(name)
            return handle, name

        monkeypatch.setattr("miles.utils.file_utils.tempfile.mkstemp", recording_mkstemp)
        atomic_write_text(path, "first")
        atomic_write_text(path, "second")

        assert len(names) == 2
        assert path.read_text() == "second"

    def test_removes_the_temporary_when_the_write_fails(self, tmp_path, monkeypatch):
        """A crashed write that leaves debris behind turns one failure into a directory nobody can read."""
        path = tmp_path / "state.json"
        monkeypatch.setattr(os, "replace", _raising)

        try:
            atomic_write_text(path, "{}")
        except RuntimeError:
            pass

        assert list(tmp_path.iterdir()) == []

    def test_publishes_a_file_readable_by_other_users(self, tmp_path: Path) -> None:
        """The published file is readable across user IDs on shared storage."""
        path = tmp_path / "state.json"

        atomic_write_text(path=path, text="{}")

        assert stat.S_IMODE(path.stat().st_mode) == 0o644

    def test_failed_replacement_preserves_the_previous_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed replacement preserves the published file and removes its temporary."""
        path = tmp_path / "state.json"
        path.write_text("old")
        monkeypatch.setattr(os, "replace", _raising)

        with pytest.raises(RuntimeError, match="no rename today"):
            atomic_write_text(path=path, text="new")

        assert path.read_text() == "old"
        assert list(tmp_path.iterdir()) == [path]


def _raising(*args, **kwargs):
    raise RuntimeError("no rename today")
