from miles.utils.external_utils.command_utils.helm_backend.naming import RunFiles, _orchestrator_state_path
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)


def _write(path, status: OrchestratorStatus, *, exit_code: int | None = None) -> None:
    OrchestratorState(status=status, exit_code=exit_code).write(path)


def _state_file(tmp_path):
    return _orchestrator_state_path(tmp_path, "260101-000000-000001")


class TestRunDir:
    def test_places_a_run_under_the_shared_root(self):
        """Every pod resolves the same run directory from the shared storage mount and the run id."""
        assert str(RunFiles.run_dir(shared_root="/cluster-storage/miles_data", run_id="260101-000000-000")).endswith(
            "/cluster-storage/miles_data/miles-runs/260101-000000-000"
        )

    def test_keeps_the_state_file_in_a_state_subdirectory(self):
        """Grouping the machine-written state keeps it out of the way of a run's own outputs."""
        path = _orchestrator_state_path("/runs/abc", "abc123")

        assert path.as_posix() == "/runs/abc/state/orchestrator-abc123.state"


class TestLatestExitFile:
    def test_names_no_file_before_a_launch_has_written_one(self, tmp_path):
        """A run directory a launch has only just created holds no verdict to collect."""
        assert RunFiles.latest_state_file(run_directory=tmp_path) is None

    def test_picks_the_newest_launch_rather_than_the_newest_write(self, tmp_path):
        """An earlier launch torn down after a later one started writes last, and its verdict is not the run's."""
        later = _orchestrator_state_path(tmp_path, "260101-000200-000001")
        earlier = _orchestrator_state_path(tmp_path, "260101-000100-000002")
        _write(later, OrchestratorStatus.EXITED, exit_code=0)
        _write(earlier, OrchestratorStatus.EXITED, exit_code=1)

        assert RunFiles.latest_state_file(run_directory=tmp_path) == later
