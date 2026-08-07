import json

import pytest

from miles.utils.external_utils.command_utils.helm_backend import run_state


class TestRunDir:
    def test_places_a_run_under_the_shared_root(self):
        """Every pod resolves the same run directory from the shared storage mount and the run id."""
        assert str(run_state.run_dir("/cluster-storage/miles_data", "260101-000000-000")).endswith(
            "/cluster-storage/miles_data/miles-runs/260101-000000-000"
        )

    def test_keeps_the_exit_file_in_a_state_subdirectory(self):
        """Grouping the machine-written state keeps it out of the way of a run's own outputs."""
        path = run_state.orchestrator_exit_path("/runs/abc")

        assert path.as_posix() == "/runs/abc/state/orchestrator.exit"


class TestWriteOrchestratorState:
    def test_creates_the_state_directory(self, tmp_path):
        """The wrapper is the first writer, so nothing has created the directory yet."""
        path = run_state.orchestrator_exit_path(tmp_path / "run")

        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)

        assert json.loads(path.read_text())["status"] == "started"

    def test_records_the_exit_code(self, tmp_path):
        """The launcher passes this code through as its own, so a failed run fails the caller."""
        path = tmp_path / "orchestrator.exit"

        run_state.write_orchestrator_state(path, run_state.STATUS_EXITED, exit_code=42)

        assert json.loads(path.read_text())["exit_code"] == 42

    def test_leaves_no_temporary_file_behind(self, tmp_path):
        """The write goes through a rename, and a leftover temp file would confuse a reader listing the directory."""
        path = tmp_path / "orchestrator.exit"

        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)

        assert [entry.name for entry in tmp_path.iterdir()] == ["orchestrator.exit"]

    def test_overwrites_a_previous_run_of_the_same_id(self, tmp_path):
        path = tmp_path / "orchestrator.exit"
        run_state.write_orchestrator_state(path, run_state.STATUS_EXITED, exit_code=1)

        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)

        state = run_state.read_orchestrator_state(path)
        assert (state.status, state.exit_code) == ("started", None)

    def test_two_writers_of_the_same_pid_do_not_share_a_temporary(self, tmp_path):
        """The launcher and the wrapper sit in different pid namespaces over one directory."""
        path = tmp_path / "orchestrator.exit"
        names = set()

        real_mkstemp = run_state.tempfile.mkstemp

        def recording_mkstemp(**kwargs):
            handle, name = real_mkstemp(**kwargs)
            names.add(name)
            return handle, name

        run_state.tempfile.mkstemp = recording_mkstemp
        try:
            run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)
            run_state.write_orchestrator_state(path, run_state.STATUS_EXITED, exit_code=0)
        finally:
            run_state.tempfile.mkstemp = real_mkstemp

        assert len(names) == 2

    def test_rejects_an_unknown_status(self, tmp_path):
        """Only the two statuses the launcher understands may reach the file."""
        with pytest.raises(AssertionError):
            run_state.write_orchestrator_state(tmp_path / "orchestrator.exit", "running")


class TestReadOrchestratorState:
    def test_returns_none_when_the_run_has_not_written_yet(self, tmp_path):
        """The launcher polls from the moment helm returns, before any pod has started."""
        assert run_state.read_orchestrator_state(tmp_path / "missing") is None

    def test_surfaces_a_storage_fault_instead_of_calling_it_absent(self, tmp_path):
        """A shared-filesystem fault read as "no verdict yet" would hang the launcher on a run it lost."""
        directory = tmp_path / "orchestrator.exit"
        directory.mkdir()

        with pytest.raises(OSError):
            run_state.read_orchestrator_state(directory)

    def test_returns_none_for_a_half_written_file(self, tmp_path):
        """A truncated read must look like "not ready" rather than crash the launcher."""
        path = tmp_path / "orchestrator.exit"
        path.write_text('{"status": "exi')

        assert run_state.read_orchestrator_state(path) is None

    def test_reports_only_the_exited_status_as_terminal(self, tmp_path):
        """A started run keeps the launcher polling; an exited one ends it."""
        path = tmp_path / "orchestrator.exit"

        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)
        assert not run_state.read_orchestrator_state(path).is_terminal

        run_state.write_orchestrator_state(path, run_state.STATUS_EXITED, exit_code=0)
        assert run_state.read_orchestrator_state(path).is_terminal
