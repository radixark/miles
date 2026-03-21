import json
import os
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from miles.utils.env_report import (
    ENV_REPORT_PREFIX,
    EditablePackageInfo,
    NodeEnvReport,
    _collect_git_info,
    _collect_pip_info,
    _is_editable,
    _parse_pip_entry,
    collect_and_print_node_env_report,
    decode_env_report,
)

_SAMPLE_PIP_INSPECT = {
    "version": "1",
    "pip_version": "24.0",
    "installed": [
        {
            "metadata": {"name": "miles", "version": "0.2.1"},
            "direct_url": {
                "url": "file:///workspace/miles",
                "dir_info": {"editable": True},
            },
        },
        {
            "metadata": {"name": "sglang", "version": "0.4.0"},
            "direct_url": {
                "url": "file:///workspace/sglang",
                "dir_info": {"editable": True},
            },
        },
        {
            "metadata": {"name": "torch", "version": "2.5.0"},
        },
        {
            "metadata": {"name": "numpy", "version": "1.26.0"},
            "direct_url": {
                "url": "https://files.pythonhosted.org/numpy-1.26.0.tar.gz",
                "archive_info": {},
            },
        },
    ],
}


class TestParsePipEntry:
    def test_normal_package(self) -> None:
        entry = _parse_pip_entry({"metadata": {"name": "torch", "version": "2.5.0"}})
        assert entry == {"name": "torch", "version": "2.5.0"}

    def test_missing_metadata(self) -> None:
        entry = _parse_pip_entry({})
        assert entry == {"name": "", "version": ""}


class TestIsEditable:
    def test_editable_package(self) -> None:
        pkg = {"direct_url": {"url": "file:///workspace/miles", "dir_info": {"editable": True}}}
        assert _is_editable(pkg) is True

    def test_non_editable_package(self) -> None:
        assert _is_editable({"metadata": {"name": "torch"}}) is False

    def test_archive_url_not_editable(self) -> None:
        pkg = {"direct_url": {"url": "https://example.com/foo.tar.gz", "archive_info": {}}}
        assert _is_editable(pkg) is False


class TestCollectPipInfo:
    def test_parses_pip_inspect_output(self) -> None:
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(_SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch("miles.utils.env_report.subprocess.run", return_value=mock_result):
            editable, full_list = _collect_pip_info()

        assert len(full_list) == 4
        assert full_list[0] == {"name": "miles", "version": "0.2.1"}
        assert full_list[2] == {"name": "torch", "version": "2.5.0"}

        assert len(editable) == 2
        assert editable[0] == EditablePackageInfo(
            name="miles",
            version="0.2.1",
            location="/workspace/miles",
        )
        assert editable[1] == EditablePackageInfo(
            name="sglang",
            version="0.4.0",
            location="/workspace/sglang",
        )

    def test_pip_inspect_failure_returns_empty(self) -> None:
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        with patch("miles.utils.env_report.subprocess.run", return_value=mock_result):
            editable, full_list = _collect_pip_info()
        assert editable == []
        assert full_list == []

    def test_pip_inspect_exception_returns_empty(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", side_effect=OSError("no pip")):
            editable, full_list = _collect_pip_info()
        assert editable == []
        assert full_list == []

    def test_pip_inspect_excludes_pythonpath_from_env(self) -> None:
        """PYTHONPATH must be excluded when running pip inspect, otherwise pip
        misses editable packages whose source is on the PYTHONPATH."""
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(_SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch.dict(os.environ, {"PYTHONPATH": "/workspace/Megatron-LM"}):
            with patch("miles.utils.env_report.subprocess.run", return_value=mock_result) as mock_run:
                _collect_pip_info()

        passed_env = mock_run.call_args.kwargs.get("env")
        assert passed_env is not None, "subprocess.run must be called with explicit env"
        assert "PYTHONPATH" not in passed_env


class TestDecodeEnvReport:
    def test_decodes_base64_json(self) -> None:
        import base64

        data = {"flavor": "test"}
        encoded = base64.b64encode(json.dumps(data).encode()).decode()
        assert decode_env_report(encoded) == data

    def test_decodes_raw_json(self) -> None:
        assert decode_env_report('{"x": 1}') == {"x": 1}

    def test_returns_none_for_empty(self) -> None:
        assert decode_env_report("") is None

    def test_returns_none_for_invalid(self) -> None:
        assert decode_env_report("not json at all!!!") is None


class TestCollectGitInfo:
    def test_collects_commit_and_diff(self, tmp_path) -> None:
        subprocess.run(["git", "init", str(tmp_path)], capture_output=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "init"],
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "test",
                "GIT_COMMITTER_NAME": "test",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_EMAIL": "t@t",
                "HOME": str(tmp_path),
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            },
        )

        info = _collect_git_info(package_name="test_pkg", location=str(tmp_path))
        assert info is not None
        assert len(info.commit) == 40
        assert info.package_name == "test_pkg"

    def test_missing_directory_returns_none(self) -> None:
        assert _collect_git_info(package_name="x", location="/nonexistent") is None

    def test_empty_location_returns_none(self) -> None:
        assert _collect_git_info(package_name="x", location="") is None

    def test_not_a_git_repo_returns_none(self, tmp_path) -> None:
        assert _collect_git_info(package_name="x", location=str(tmp_path)) is None


class TestCollectAndPrintNodeEnvReport:
    def _mock_pip_inspect(self) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(_SAMPLE_PIP_INSPECT),
            stderr="",
        )

    def test_returns_structured_report(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
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
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
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
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
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
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training",
                rank=0,
                partial_env_report="",
            )
        assert report.launcher_env_report is None

    def test_invalid_json_partial_env_report(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training",
                rank=0,
                partial_env_report="not json",
            )
        assert report.launcher_env_report is None

    def test_report_serializable(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
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
