from pathlib import Path

import pytest

from tests.e2e.ft.conftest_ft.app import _DUMPS_ROOT_ENV, resolve_dump_dir


def test_dump_dir_hangs_off_the_configured_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cluster says where dumps go through the environment its infra file sets."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path / "dumps"))
    assert resolve_dump_dir("scenario_x", run_id="run-a") == str(tmp_path / "dumps" / "run-a" / "scenario_x")


def test_an_empty_configured_root_is_not_a_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset variable and one set to nothing both mean the cluster configured no root."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, "")
    monkeypatch.setattr("os.makedirs", lambda path, exist_ok: None)

    assert resolve_dump_dir("scenario_x", run_id="run-a") == "/node_public/dumps/run-a/scenario_x"


def test_two_runs_of_one_test_do_not_share_a_dump_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The run id in the path is what stops one run's rmtree deleting another's dumps."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path))
    first = resolve_dump_dir("scenario_x", run_id="run-a")

    assert resolve_dump_dir("scenario_x", run_id="run-b") != first


def test_the_dump_directory_exists_when_it_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers write into the returned path without creating it themselves."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path / "dumps"))
    assert Path(resolve_dump_dir("scenario_x", run_id="run-a")).is_dir()
