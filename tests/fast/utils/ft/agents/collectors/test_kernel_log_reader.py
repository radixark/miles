from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.agents.collectors.kernel_log_reader import DmesgSubprocessReader, KmsgFileReader


class TestKmsgFileReader:
    def test_read_after_close_returns_empty(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"init line\n")

        reader = KmsgFileReader(kmsg_path=kmsg_file)
        reader.close()

        assert reader.read_new_lines() == []

    def test_close_idempotent(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"")

        reader = KmsgFileReader(kmsg_path=kmsg_file)
        reader.close()
        reader.close()
        assert reader._fd is None

    def test_close_handles_os_error(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"")

        reader = KmsgFileReader(kmsg_path=kmsg_file)
        os.close(reader._fd)
        reader.close()
        assert reader._fd is None

    def test_blocking_io_error_breaks_loop(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"")

        reader = KmsgFileReader(kmsg_path=kmsg_file)

        with patch("os.read", side_effect=[b"line1\nline2\n", BlockingIOError]):
            lines = reader.read_new_lines()

        assert lines == ["line1", "line2"]
        reader.close()

    def test_empty_read_returns_empty(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"")

        reader = KmsgFileReader(kmsg_path=kmsg_file)

        with patch("os.read", return_value=b""):
            lines = reader.read_new_lines()

        assert lines == []
        reader.close()


class TestDmesgSubprocessReader:
    def test_returns_lines_on_success(self) -> None:
        reader = DmesgSubprocessReader()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "kernel: message 1\nkernel: message 2\n"

        with patch("subprocess.run", return_value=mock_result):
            lines = reader.read_new_lines()

        assert lines == ["kernel: message 1", "kernel: message 2"]

    def test_returns_empty_on_nonzero_returncode(self) -> None:
        reader = DmesgSubprocessReader()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            lines = reader.read_new_lines()

        assert lines == []

    def test_nonzero_returncode_still_advances_time(self) -> None:
        """H-5: dmesg failure used to leave _last_dmesg_time stale, causing
        the next successful call to re-read the same time window and
        double-count XIDs. Now time always advances."""
        reader = DmesgSubprocessReader()
        time_before = reader._last_dmesg_time

        mock_fail = MagicMock(returncode=1, stdout="", stderr="permission denied")
        with patch("subprocess.run", return_value=mock_fail):
            reader.read_new_lines()

        time_after_fail = reader._last_dmesg_time
        assert time_after_fail > time_before

        mock_ok = MagicMock(returncode=0, stdout="line 1\n")
        with patch("subprocess.run", return_value=mock_ok) as mock_run:
            lines = reader.read_new_lines()

        assert lines == ["line 1"]
        call_args = mock_run.call_args[0][0]
        since_arg = call_args[2]
        assert time_after_fail.strftime("%H:%M:%S") in since_arg

    def test_returns_empty_on_timeout(self) -> None:
        reader = DmesgSubprocessReader()

        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="dmesg", timeout=5)):
            lines = reader.read_new_lines()

        assert lines == []

    def test_returns_empty_on_empty_stdout(self) -> None:
        reader = DmesgSubprocessReader()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            lines = reader.read_new_lines()

        assert lines == []

    def test_nonzero_returncode_logs_warning_with_stderr(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        reader = DmesgSubprocessReader()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "dmesg: read kernel buffer failed: Operation not permitted"

        with caplog.at_level(logging.WARNING), patch("subprocess.run", return_value=mock_result):
            lines = reader.read_new_lines()

        assert lines == []
        assert "non-zero returncode=1" in caplog.text
        assert "Operation not permitted" in caplog.text

    def test_close_is_noop(self) -> None:
        reader = DmesgSubprocessReader()
        reader.close()


class TestKmsgFileReaderRealFile:
    """Test KmsgFileReader with real file I/O, no mocking of os.read."""

    def test_seeks_past_existing_content(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_text("old line 1\nold line 2\n")

        reader = KmsgFileReader(kmsg_path=kmsg_file)
        lines = reader.read_new_lines()
        assert lines == []
        reader.close()

    def test_reads_appended_content(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_text("")

        reader = KmsgFileReader(kmsg_path=kmsg_file)

        with open(kmsg_file, "a") as f:
            f.write("new line 1\nnew line 2\n")

        lines = reader.read_new_lines()
        assert lines == ["new line 1", "new line 2"]
        reader.close()

    def test_multiple_appends_read_incrementally(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_text("")

        reader = KmsgFileReader(kmsg_path=kmsg_file)

        with open(kmsg_file, "a") as f:
            f.write("batch 1\n")
        assert reader.read_new_lines() == ["batch 1"]

        with open(kmsg_file, "a") as f:
            f.write("batch 2\nbatch 3\n")
        assert reader.read_new_lines() == ["batch 2", "batch 3"]

        reader.close()
