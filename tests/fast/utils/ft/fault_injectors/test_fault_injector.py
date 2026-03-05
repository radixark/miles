from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil

from miles.utils.ft.e2e.fault_injector import _TRAINING_CMDLINE_PATTERNS, FaultInjectorActor


def _get_inner_class() -> type:
    """Unwrap Ray's ActorClass wrapper to get the plain Python class."""
    meta = getattr(FaultInjectorActor, "__ray_metadata__", None)
    if meta is not None:
        return meta.modified_class
    return FaultInjectorActor  # type: ignore[return-value]


def _make_actor() -> FaultInjectorActor:
    cls = _get_inner_class()
    return cls()


class TestFindTrainingProcesses:
    def _make_proc_info(
        self,
        pid: int,
        cmdline: list[str],
        name: str = "python",
    ) -> MagicMock:
        proc = MagicMock()
        proc.info = {"pid": pid, "cmdline": cmdline, "name": name}
        return proc

    def test_matches_megatron_process(self) -> None:
        procs = [
            self._make_proc_info(100, ["python", "-m", "megatron.training"]),
            self._make_proc_info(200, ["bash", "-c", "echo hello"]),
        ]
        with patch("psutil.process_iter", return_value=procs):
            results = _make_actor().find_training_processes()

        assert len(results) == 1
        assert results[0]["pid"] == 100

    def test_matches_torchrun(self) -> None:
        procs = [
            self._make_proc_info(300, ["torchrun", "--nproc_per_node=8", "train.py"]),
        ]
        with patch("psutil.process_iter", return_value=procs):
            results = _make_actor().find_training_processes()

        assert len(results) == 1
        assert results[0]["pid"] == 300

    def test_matches_run_deepseek(self) -> None:
        procs = [
            self._make_proc_info(400, ["python", "miles/run_deepseek_v3.py"]),
        ]
        with patch("psutil.process_iter", return_value=procs):
            results = _make_actor().find_training_processes()

        assert len(results) == 1

    def test_no_match_returns_empty(self) -> None:
        procs = [
            self._make_proc_info(500, ["vim", "somefile.txt"]),
        ]
        with patch("psutil.process_iter", return_value=procs):
            results = _make_actor().find_training_processes()

        assert results == []

    def test_handles_access_denied(self) -> None:
        proc = MagicMock()
        proc.info.__getitem__ = MagicMock(side_effect=psutil.AccessDenied(pid=1))
        proc.info.get = MagicMock(side_effect=psutil.AccessDenied(pid=1))

        with patch("psutil.process_iter", return_value=[proc]):
            results = _make_actor().find_training_processes()

        assert results == []


class TestDiskOperations:
    def test_fill_and_cleanup_disk(self, tmp_path: Path) -> None:
        actor = _make_actor()

        fill_path = str(tmp_path / "fill_file")
        size_bytes = 1024 * 1024  # 1 MB
        actor.fill_disk(path=fill_path, size_bytes=size_bytes)

        p = Path(fill_path)
        assert p.exists()
        assert p.stat().st_size == size_bytes
        assert fill_path in actor._filled_paths

        actor.cleanup_disk(path=fill_path)
        assert not p.exists()
        assert fill_path not in actor._filled_paths

    def test_cleanup_disk_nonexistent_file(self, tmp_path: Path) -> None:
        _make_actor().cleanup_disk(path=str(tmp_path / "nonexistent"))

    def test_cleanup_all(self, tmp_path: Path) -> None:
        actor = _make_actor()

        path1 = str(tmp_path / "file1")
        path2 = str(tmp_path / "file2")
        actor.fill_disk(path=path1, size_bytes=1024)
        actor.fill_disk(path=path2, size_bytes=1024)

        assert Path(path1).exists()
        assert Path(path2).exists()

        actor.cleanup_all()

        assert not Path(path1).exists()
        assert not Path(path2).exists()
        assert actor._filled_paths == []
        assert actor._stress_pids == []


class TestSignalOperations:
    def test_kill_process_sends_sigkill(self) -> None:
        actor = _make_actor()
        with patch("os.kill") as mock_kill:
            actor.kill_process(pid=42)
            mock_kill.assert_called_once_with(42, signal.SIGKILL)

    def test_kill_process_custom_signal(self) -> None:
        actor = _make_actor()
        with patch("os.kill") as mock_kill:
            actor.kill_process(pid=42, sig=signal.SIGTERM)
            mock_kill.assert_called_once_with(42, signal.SIGTERM)

    def test_stop_process_sends_sigstop(self) -> None:
        actor = _make_actor()
        with patch("os.kill") as mock_kill:
            actor.stop_process(pid=99)
            mock_kill.assert_called_once_with(99, signal.SIGSTOP)

    def test_continue_process_sends_sigcont(self) -> None:
        actor = _make_actor()
        with patch("os.kill") as mock_kill:
            actor.continue_process(pid=99)
            mock_kill.assert_called_once_with(99, signal.SIGCONT)


class TestGpuStressOperations:
    def test_start_gpu_stress_records_pid(self) -> None:
        actor = _make_actor()
        mock_proc = MagicMock()
        mock_proc.pid = 1234

        with patch("subprocess.Popen", return_value=mock_proc):
            pid = actor.start_gpu_stress()

        assert pid == 1234
        assert 1234 in actor._stress_pids

    def test_stop_gpu_stress_kills_and_removes_pid(self) -> None:
        actor = _make_actor()
        actor._stress_pids.append(555)

        with patch("os.kill") as mock_kill:
            actor.stop_gpu_stress(pid=555)

        mock_kill.assert_called_once_with(555, signal.SIGKILL)
        assert 555 not in actor._stress_pids

    def test_stop_gpu_stress_ignores_process_lookup_error(self) -> None:
        actor = _make_actor()
        actor._stress_pids.append(999)

        with patch("os.kill", side_effect=ProcessLookupError):
            actor.stop_gpu_stress(pid=999)

        assert 999 not in actor._stress_pids


class TestTrainingCmdlinePatterns:
    def test_all_patterns_are_lowercase(self) -> None:
        for pattern in _TRAINING_CMDLINE_PATTERNS:
            assert pattern == pattern.lower()
