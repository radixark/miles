import json
import os
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.fast.utils.env_report.conftest import SAMPLE_PIP_INSPECT

from miles.utils.env_report.reporter import ENV_REPORT_PREFIX, NodeEnvReport, collect_and_print_node_env_report


class TestCollectAndPrintNodeEnvReport:
    def _mock_pip_inspect(self) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(SAMPLE_PIP_INSPECT),
            stderr="",
        )

    def test_returns_structured_report(self) -> None:
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training",
                rank=0,
                partial_env_report='{"flavor": "test"}',
            )

        assert isinstance(report, NodeEnvReport)
        assert report.role == "training"
        assert report.rank == 0
        assert report.launcher_env_report == {"flavor": "test"}
        assert len(report.editable_packages) == 2
        assert len(report.full_pip_list) == 4

    def test_prints_single_line_json(self, capsys) -> None:
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=self._mock_pip_inspect()):
            collect_and_print_node_env_report(
                role="rollout",
                rank=3,
                partial_env_report="",
            )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.startswith(ENV_REPORT_PREFIX)]
        assert len(lines) == 1
        json_str = lines[0].removeprefix(ENV_REPORT_PREFIX)
        parsed = json.loads(json_str)
        assert parsed["role"] == "rollout"
        assert parsed["rank"] == 3

    def test_printed_json_has_sorted_keys(self, capsys) -> None:
        """Verify JSON output uses sort_keys for deterministic cross-process comparison."""
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=self._mock_pip_inspect()):
            collect_and_print_node_env_report(
                role="training",
                rank=0,
                partial_env_report='{"b": 2, "a": 1}',
            )

        captured = capsys.readouterr()
        line = next(x for x in captured.out.splitlines() if x.startswith(ENV_REPORT_PREFIX))
        json_str = line.removeprefix(ENV_REPORT_PREFIX)
        keys = list(json.loads(json_str).keys())
        assert keys == sorted(keys), f"Top-level keys not sorted: {keys}"

    def test_empty_partial_env_report(self) -> None:
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training",
                rank=0,
                partial_env_report="",
            )
        assert report.launcher_env_report is None

    def test_invalid_json_partial_env_report(self) -> None:
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training",
                rank=0,
                partial_env_report="not json",
            )
        assert report.launcher_env_report is None

    def test_report_serializable(self) -> None:
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training",
                rank=0,
                partial_env_report='{"x": 1}',
            )
        report_dict = asdict(report)
        json_str = json.dumps(report_dict, default=str)
        parsed = json.loads(json_str)
        assert parsed["editable_packages"][0]["name"] == "miles"


# ---------------------------------------------------------------------------
# Integration tests: real editable package + real git repo
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_COMMITTER_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(repo),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )


@pytest.fixture()
def editable_package(tmp_path: Path):
    """Create a real editable Python package with a git repo, pip install -e it, yield info, cleanup."""
    pkg_name = f"envreporttest{uuid.uuid4().hex[:8]}"
    repo = tmp_path / pkg_name
    repo.mkdir()
    src = repo / pkg_name
    src.mkdir()
    (src / "__init__.py").write_text('__version__ = "0.0.1"\n')
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "{pkg_name}"\nversion = "0.0.1"\n'
        f'[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n'
    )

    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    result = subprocess.run(
        ["pip", "install", "-e", str(repo), "--no-build-isolation", "-q"],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"pip install -e failed (read-only env?): {result.stderr[:200]}")

    yield {"pkg_name": pkg_name, "repo": repo, "commit": commit}

    subprocess.run(["pip", "uninstall", "-y", pkg_name], capture_output=True)


class TestRealEditablePackage:
    """Integration tests: create a real editable package, pip install -e, run env report."""

    def test_detects_clean_editable_package(self, editable_package, capsys) -> None:
        """Verify env report finds the package with correct git commit, not dirty."""
        pkg_name = editable_package["pkg_name"]
        repo = editable_package["repo"]
        expected_commit = editable_package["commit"]

        # Step 1: Run the full collection (no mocks)
        report = collect_and_print_node_env_report(
            role="training",
            rank=0,
            partial_env_report='{"test": true}',
        )

        # Step 2: Verify the package appears in editable_packages
        editable_names = {pkg.name for pkg in report.editable_packages}
        assert pkg_name in editable_names, f"{pkg_name} not in editable packages: {editable_names}"

        pkg_info = next(p for p in report.editable_packages if p.name == pkg_name)
        assert pkg_info.location == str(repo)

        # Step 3: Verify git info — clean repo
        git_info = next(
            (r for r in report.git_repos if r.package_name == pkg_name),
            None,
        )
        assert git_info is not None, f"git info not found for {pkg_name}"
        assert git_info.commit == expected_commit
        assert git_info.dirty is False
        assert git_info.diff_stat == ""

        # Step 4: Verify single-line JSON output is parseable and contains this package
        captured = capsys.readouterr()
        report_lines = [line for line in captured.out.splitlines() if line.startswith(ENV_REPORT_PREFIX)]
        assert len(report_lines) == 1
        parsed = json.loads(report_lines[0].removeprefix(ENV_REPORT_PREFIX))
        parsed_editable_names = {p["name"] for p in parsed["editable_packages"]}
        assert pkg_name in parsed_editable_names
        parsed_git = {r["package_name"]: r for r in parsed["git_repos"]}
        assert parsed_git[pkg_name]["commit"] == expected_commit
        assert parsed_git[pkg_name]["dirty"] is False

        # Step 5: Verify package also in full_pip_list
        full_names = {p["name"] for p in report.full_pip_list}
        assert pkg_name in full_names

    def test_detects_dirty_editable_package_staged(self, editable_package) -> None:
        """Make repo dirty with staged changes, verify env report detects it."""
        pkg_name = editable_package["pkg_name"]
        repo = editable_package["repo"]
        expected_commit = editable_package["commit"]

        # Step 1: Stage an uncommitted file
        (repo / "staged_change.txt").write_text("staged\n")
        _git(repo, "add", "staged_change.txt")

        # Step 2: Run collection
        report = collect_and_print_node_env_report(
            role="training",
            rank=0,
            partial_env_report="",
        )

        # Step 3: Verify dirty + diff_stat mentions the file
        git_info = next(
            (r for r in report.git_repos if r.package_name == pkg_name),
            None,
        )
        assert git_info is not None
        assert git_info.commit == expected_commit
        assert git_info.dirty is True
        assert "staged_change.txt" in git_info.diff_stat

    def test_detects_dirty_editable_package_unstaged(self, editable_package) -> None:
        """Make repo dirty with unstaged changes, verify env report detects it."""
        pkg_name = editable_package["pkg_name"]
        repo = editable_package["repo"]

        # Step 1: Modify a tracked file without staging
        init_py = repo / pkg_name / "__init__.py"
        init_py.write_text('__version__ = "0.0.2"\n')

        # Step 2: Run collection
        report = collect_and_print_node_env_report(
            role="training",
            rank=0,
            partial_env_report="",
        )

        # Step 3: Verify dirty
        git_info = next(
            (r for r in report.git_repos if r.package_name == pkg_name),
            None,
        )
        assert git_info is not None
        assert git_info.dirty is True
        assert "__init__.py" in git_info.diff_stat
