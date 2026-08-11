import json

import pydantic
import pytest

from miles.utils.external_utils.command_utils.helm_backend.naming import _orchestrator_state_path
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)


def _write(path, status: OrchestratorStatus, *, exit_code: int | None = None) -> None:
    OrchestratorState(status=status, exit_code=exit_code).write(path)


def _state_file(tmp_path):
    path = _orchestrator_state_path(tmp_path, "260101-000000-000001")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class TestWriteOrchestratorState:
    def test_creates_the_state_directory(self, tmp_path):
        """The wrapper is the first writer, so nothing has created the directory yet."""
        path = _orchestrator_state_path(tmp_path / "run", "abc123")

        _write(path, OrchestratorStatus.STARTED)

        assert json.loads(path.read_text())["status"] == "started"

    def test_records_the_exit_code(self, tmp_path):
        """The launcher passes this code through as its own, so a failed run fails the caller."""
        path = _state_file(tmp_path)

        _write(path, OrchestratorStatus.EXITED, exit_code=42)

        assert json.loads(path.read_text())["exit_code"] == 42

    def test_overwrites_a_previous_run_of_the_same_id(self, tmp_path):
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.EXITED, exit_code=1)

        _write(path, OrchestratorStatus.STARTED)

        state = OrchestratorState.read(path)
        assert (state.status, state.exit_code) == (OrchestratorStatus.STARTED, None)

    def test_rejects_an_unknown_status(self):
        """Only the two statuses the launcher understands may reach the file."""
        with pytest.raises(pydantic.ValidationError):
            OrchestratorState(status="running")

    def test_refuses_to_record_a_finished_run_without_its_exit_code(self):
        """The launcher passes this number to its own caller, so a terminal state without one says nothing."""
        with pytest.raises(pydantic.ValidationError, match="exit code"):
            OrchestratorState(status=OrchestratorStatus.EXITED)


class TestReadOrchestratorState:
    def test_returns_none_when_the_run_has_not_written_yet(self, tmp_path):
        """The launcher polls from the moment helm returns, before any pod has started."""
        assert OrchestratorState.read(tmp_path / "missing") is None

    def test_surfaces_a_storage_fault_instead_of_calling_it_absent(self, tmp_path):
        """A shared-filesystem fault read as "no verdict yet" would hang the launcher on a run it lost."""
        directory = _state_file(tmp_path)
        directory.mkdir()

        with pytest.raises(OSError):
            OrchestratorState.read(directory)

    def test_returns_none_for_a_half_written_file(self, tmp_path):
        """A truncated read must look like "not ready" rather than crash the entrypoint."""
        path = _state_file(tmp_path)
        path.write_text('{"status": "exi')

        assert OrchestratorState.read(path) is None

    def test_reports_only_the_exited_status_as_terminal(self, tmp_path):
        """A started run keeps the launcher polling; an exited one ends it."""
        path = _state_file(tmp_path)

        _write(path, OrchestratorStatus.STARTED)
        assert not OrchestratorState.read(path).is_terminal

        _write(path, OrchestratorStatus.EXITED, exit_code=0)
        assert OrchestratorState.read(path).is_terminal
