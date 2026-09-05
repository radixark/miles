import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.naming import (
    RunFiles,
    _orchestrator_state_path,
    platform_account_name,
)
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)
from miles.utils.workers.types import PlatformAccess


def _write(path, status: OrchestratorStatus, *, exit_code: int | None = None) -> None:
    OrchestratorState(status=status, exit_code=exit_code).write(path)


def _record(run_directory, launch_token: str, state_file) -> None:
    path = Path(run_directory) / "launches" / f"launch-{launch_token}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"state_file": str(state_file)}))


def _state_file(tmp_path):
    return _orchestrator_state_path(tmp_path, "260101-000000-000001")


class TestPlatformAccountName:
    def test_platform_account_name_refuses_no_platform_access(self) -> None:
        """A worker without platform access must not receive a platform service account."""
        with pytest.raises(AssertionError, match="never reaches the platform"):
            platform_account_name(release="miles-run-example-train", access=PlatformAccess.NONE)


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

    def test_gives_every_launch_its_own_record_file(self):
        """Two launches of one run must not overwrite each other's record of what they launched."""
        first = RunFiles.new_record_file(run_directory="/runs/abc")
        second = RunFiles.new_record_file(run_directory="/runs/abc")

        assert first.parent.as_posix() == "/runs/abc/launches"
        assert first != second


class TestLatestExitFile:
    def test_names_no_file_before_a_launch_has_recorded_one(self, tmp_path):
        """A run directory a launch has only just created holds no verdict to collect."""
        assert RunFiles.latest_state_file(run_directory=tmp_path) is None

    def test_names_the_newest_launch_s_state_file_before_it_is_written(self, tmp_path):
        """A generation whose pods never came up must not hand the previous generation's verdict over."""
        _record(tmp_path, "260101-000100-000002", _orchestrator_state_path(tmp_path, "260101-000100-000002"))
        pending = _orchestrator_state_path(tmp_path, "260101-000200-000001")
        _record(tmp_path, "260101-000200-000001", pending)

        assert RunFiles.latest_state_file(run_directory=tmp_path) == pending

    def test_picks_the_newest_launch_rather_than_the_newest_write(self, tmp_path):
        """An earlier launch torn down after a later one started writes last, and its verdict is not the run's."""
        later = _orchestrator_state_path(tmp_path, "260101-000200-000001")
        earlier = _orchestrator_state_path(tmp_path, "260101-000100-000002")
        _write(later, OrchestratorStatus.EXITED, exit_code=0)
        _write(earlier, OrchestratorStatus.EXITED, exit_code=1)
        _record(tmp_path, "260101-000100-000002", earlier)
        _record(tmp_path, "260101-000200-000001", later)

        assert RunFiles.latest_state_file(run_directory=tmp_path) == later


def _mint_token(monkeypatch: pytest.MonkeyPatch, *, when: datetime, tail: int) -> str:
    monkeypatch.setattr(naming, "datetime", SimpleNamespace(now=lambda: when))
    monkeypatch.setattr(naming.random, "Random", lambda: SimpleNamespace(randint=lambda low, high: tail))
    return naming._new_launch_token()


class TestNewLaunchToken:
    def test_two_launches_of_one_second_order_by_the_time_they_were_minted(self, monkeypatch):
        """Everything that picks the newest launch sorts these names, and the random tail ordered them by chance."""
        earlier = _mint_token(monkeypatch, when=datetime(2026, 1, 1, 0, 0, 0, 1), tail=999999)
        later = _mint_token(monkeypatch, when=datetime(2026, 1, 1, 0, 0, 0, 2), tail=0)

        assert earlier < later

    def test_a_token_carries_the_microsecond_it_was_minted_at(self, monkeypatch):
        """Two launches of one run are apart by microseconds, which is the resolution the name has to keep."""
        token = _mint_token(monkeypatch, when=datetime(2026, 1, 1, 0, 0, 0, 123), tail=7)

        assert token == "260101-000000-000123-000007"

    def test_two_tokens_minted_in_one_microsecond_still_differ(self, monkeypatch):
        """The random tail no longer decides the order, but it is still what keeps two names apart."""
        when = datetime(2026, 1, 1, 0, 0, 0, 5)

        assert _mint_token(monkeypatch, when=when, tail=1) != _mint_token(monkeypatch, when=when, tail=2)

    def test_every_part_of_a_real_token_is_fixed_width(self):
        """A part that can be shorter sorts before a longer one whatever time it names."""
        assert re.fullmatch(r"\d{6}-\d{6}-\d{6}-\d{6}", naming._new_launch_token())

    def test_a_real_token_never_names_a_time_before_the_one_minted_earlier(self):
        """The clock part is what the newest-launch lookup reads, and it has to be non-decreasing."""
        stamp_length = len("260101-000000-000000")

        first = naming._new_launch_token()
        second = naming._new_launch_token()

        assert first[:stamp_length] <= second[:stamp_length]


class TestSupersededMarker:
    def test_the_marker_sits_beside_the_state_file_it_supersedes(self):
        """Both the launcher that writes it and the orchestrator that reads it know only that path."""
        marker = RunFiles.superseded_marker(state_file="/runs/abc/state/orchestrator-x.state")

        assert marker.as_posix() == "/runs/abc/state/orchestrator-x.state.superseded"

    def test_every_generation_has_a_marker_of_its_own(self):
        """A run is relaunched many times, and one shared marker would defuse every generation at once."""
        first = RunFiles.superseded_marker(state_file="/runs/abc/state/orchestrator-1.state")
        second = RunFiles.superseded_marker(state_file="/runs/abc/state/orchestrator-2.state")

        assert first != second

    def test_a_path_and_the_string_spelling_of_it_name_one_marker(self):
        """The launcher holds a Path and the wrapper is given a string, and they have to meet."""
        as_text = RunFiles.superseded_marker(state_file="/runs/abc/state/orchestrator-x.state")

        assert RunFiles.superseded_marker(state_file=Path("/runs/abc/state/orchestrator-x.state")) == as_text
