import logging
import os
import subprocess
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.fast.utils.env_report.conftest import make_args

from miles.utils.audit_utils.event_logger.models import EnvReportGitRepoInfo
from miles.utils.env_report import git_state
from miles.utils.env_report.collector import collect_env_report, collect_env_report_snapshot
from miles.utils.env_report.git_state import collect_git_info


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

        info = collect_git_info(package_name="test_pkg", location=str(tmp_path))
        assert info is not None
        assert len(info.commit) == 40
        assert info.package_name == "test_pkg"

    def test_missing_directory_returns_none(self) -> None:
        assert collect_git_info(package_name="x", location="/nonexistent") is None

    def test_empty_location_returns_none(self) -> None:
        assert collect_git_info(package_name="x", location="") is None

    def test_not_a_git_repo_returns_none(self, tmp_path) -> None:
        assert collect_git_info(package_name="x", location=str(tmp_path)) is None


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
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
        check=check,
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

    def test_detects_clean_editable_package(self, editable_package) -> None:
        """Verify env report finds the package with correct git commit, not dirty."""
        pkg_name = editable_package["pkg_name"]
        repo = editable_package["repo"]
        expected_commit = editable_package["commit"]

        # Step 1: Run the full collection (no mocks)
        report = collect_env_report(snapshot=collect_env_report_snapshot(make_args(env_report='{"test": true}')))

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

        # Step 4: Verify package also in full_pip_list
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
        report = collect_env_report(snapshot=collect_env_report_snapshot(make_args()))

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
        report = collect_env_report(snapshot=collect_env_report_snapshot(make_args()))

        # Step 3: Verify dirty
        git_info = next(
            (r for r in report.git_repos if r.package_name == pkg_name),
            None,
        )
        assert git_info is not None
        assert git_info.dirty is True
        assert "__init__.py" in git_info.diff_stat


def _make_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    (path / "tracked.txt").write_text("hello\n")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-m", "init")


def _make_conflicted_repo(path: Path, *, ours: str, theirs: str) -> None:
    _make_repo(path)
    trunk = _git(path, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    _git(path, "checkout", "-b", "theirs")
    (path / "tracked.txt").write_text(theirs)
    _git(path, "commit", "-a", "-m", "theirs")
    _git(path, "checkout", trunk)
    (path / "tracked.txt").write_text(ours)
    _git(path, "commit", "-a", "-m", "ours")
    _git(path, "merge", "theirs", check=False)
    assert _git(path, "ls-files", "--unmerged").stdout != ""


def _completed(stdout: bytes) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=stdout, stderr=b"")


def _git_info(path: Path) -> EnvReportGitRepoInfo:
    info = collect_git_info(package_name="pkg", location=str(path))
    assert info is not None
    return info


class TestUncommittedHash:
    def test_a_failed_patch_collection_has_no_uncommitted_hash(self) -> None:
        """A failed patch command must not manufacture the hash of a clean working tree."""

        def run_git(*, args: list[str], location: str) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=["git", *args],
                returncode=1 if args == ["diff", "--patch-with-stat", "HEAD"] else 0,
                stdout=b"commit" if args == ["rev-parse", "HEAD"] else b"",
                stderr=b"patch failed" if args == ["diff", "--patch-with-stat", "HEAD"] else b"",
            )

        with patch("miles.utils.env_report.git_state._run_git", side_effect=run_git):
            info = collect_git_info(package_name="pkg", location=".")

        assert info is not None
        assert info.uncommitted_hash is None

    def test_an_unreadable_untracked_file_is_named_and_marked_unhashed(self) -> None:
        """An untracked file lost before hashing must remain visible and contribute an unreadable marker."""
        raw_path = b"vanished.py"

        def run_git(*, args: list[str], location: str) -> subprocess.CompletedProcess:
            stdout = b"commit" if args == ["rev-parse", "HEAD"] else b""
            if args == ["ls-files", "--others", "--exclude-standard", "-z"]:
                stdout = raw_path + b"\0"
            return subprocess.CompletedProcess(args=["git", *args], returncode=0, stdout=stdout, stderr=b"")

        with patch("miles.utils.env_report.git_state._run_git", side_effect=run_git):
            info = collect_git_info(package_name="pkg", location=".")
            repeated_info = collect_git_info(package_name="pkg", location=".")

        def run_clean_git(*, args: list[str], location: str) -> subprocess.CompletedProcess:
            stdout = b"commit" if args == ["rev-parse", "HEAD"] else b""
            return subprocess.CompletedProcess(args=["git", *args], returncode=0, stdout=stdout, stderr=b"")

        with patch("miles.utils.env_report.git_state._run_git", side_effect=run_clean_git):
            clean_info = collect_git_info(package_name="pkg", location=".")

        assert info is not None
        assert repeated_info is not None
        assert clean_info is not None
        assert info.untracked_paths == ["vanished.py"]
        assert info.untracked_unhashed_paths == ["vanished.py"]
        assert info.uncommitted_hash == repeated_info.uncommitted_hash
        assert info.uncommitted_hash != clean_info.uncommitted_hash

    def test_a_clean_repo_has_a_stable_hash_and_is_not_dirty(self, tmp_path: Path) -> None:
        """A clean checkout must report a hash that two ranks can compare without any diff to read."""
        _make_repo(tmp_path)

        info = _git_info(tmp_path)

        assert info.dirty is False
        assert info.untracked_paths == []
        assert info.untracked_paths_truncated is False
        assert info.uncommitted_hash is not None
        assert len(info.uncommitted_hash) == 64
        assert _git_info(tmp_path).uncommitted_hash == info.uncommitted_hash

    def test_a_modified_tracked_file_changes_the_hash(self, tmp_path: Path) -> None:
        """The whole point is that an edited working tree cannot look like the committed one."""
        _make_repo(tmp_path)
        clean_hash = _git_info(tmp_path).uncommitted_hash

        (tmp_path / "tracked.txt").write_text("changed\n")
        info = _git_info(tmp_path)

        assert info.dirty is True
        assert info.uncommitted_hash != clean_hash
        assert "tracked.txt" in info.diff_stat

    def test_an_untracked_file_alone_makes_the_repo_dirty(self, tmp_path: Path) -> None:
        """git diff sees nothing when a whole new module is dropped in, so dirty must not come from the diff alone."""
        _make_repo(tmp_path)
        clean_hash = _git_info(tmp_path).uncommitted_hash

        (tmp_path / "new_module.py").write_text("x = 1\n")
        info = _git_info(tmp_path)

        assert info.dirty is True
        assert info.untracked_paths == ["new_module.py"]
        assert info.uncommitted_hash != clean_hash

    def test_the_content_of_an_untracked_file_is_part_of_the_hash(self, tmp_path: Path) -> None:
        """Comparing ranks by file name only would call two different patches the same run."""
        _make_repo(tmp_path)
        (tmp_path / "new_module.py").write_text("x = 1\n")
        first = _git_info(tmp_path).uncommitted_hash

        (tmp_path / "new_module.py").write_text("x = 2\n")

        assert _git_info(tmp_path).uncommitted_hash != first

    def test_tracked_and_untracked_changes_both_reach_the_hash(self, tmp_path: Path) -> None:
        """Either kind of change alone must not collide with having both."""
        _make_repo(tmp_path)
        (tmp_path / "tracked.txt").write_text("changed\n")
        tracked_only = _git_info(tmp_path).uncommitted_hash

        (tmp_path / "new_module.py").write_text("x = 1\n")
        both = _git_info(tmp_path).uncommitted_hash

        assert both != tracked_only
        assert _git_info(tmp_path).untracked_paths == ["new_module.py"]

    def test_the_same_change_in_two_locations_hashes_the_same(self, tmp_path: Path) -> None:
        """Ranks check out the same code under different paths, so the hash must not encode the path or mtime."""
        left = tmp_path / "left"
        right = tmp_path / "somewhere" / "else" / "right"
        for repo in (left, right):
            _make_repo(repo)
            (repo / "tracked.txt").write_text("changed\n")
            (repo / "new_module.py").write_text("x = 1\n")

        assert _git_info(left).commit != "" and _git_info(right).commit != ""
        assert _git_info(left).uncommitted_hash == _git_info(right).uncommitted_hash

    def test_an_ignored_file_is_not_part_of_the_hash(self, tmp_path: Path) -> None:
        """Build outputs and checkpoints are ignored on purpose; hashing them would make every rank differ."""
        _make_repo(tmp_path)
        (tmp_path / ".gitignore").write_text("junk/\n")
        _git(tmp_path, "add", ".gitignore")
        _git(tmp_path, "commit", "-m", "ignore junk")
        baseline = _git_info(tmp_path).uncommitted_hash

        (tmp_path / "junk").mkdir()
        (tmp_path / "junk" / "output.bin").write_bytes(b"noise")
        info = _git_info(tmp_path)

        assert info.untracked_paths == []
        assert info.dirty is False
        assert info.uncommitted_hash == baseline

    def test_a_huge_untracked_file_is_hashed_by_size_and_says_so(self, tmp_path: Path) -> None:
        """Reading a multi-gigabyte checkpoint would hang the report, and skipping it silently would hide it."""
        _make_repo(tmp_path)
        (tmp_path / "huge.bin").write_bytes(b"a" * 32)

        with patch("miles.utils.env_report.git_state._UNTRACKED_MAX_FILE_BYTES", 8):
            info = _git_info(tmp_path)
            (tmp_path / "huge.bin").write_bytes(b"b" * 32)
            same_size = _git_info(tmp_path)
            (tmp_path / "huge.bin").write_bytes(b"b" * 33)
            other_size = _git_info(tmp_path)

        assert info.untracked_paths == ["huge.bin"]
        assert info.untracked_unhashed_paths == ["huge.bin"]
        assert same_size.uncommitted_hash == info.uncommitted_hash
        assert other_size.uncommitted_hash != info.uncommitted_hash

    def test_too_many_untracked_files_truncate_the_list_visibly(self, tmp_path: Path) -> None:
        """A repo full of stray files must not turn the report into a directory listing without saying so."""
        _make_repo(tmp_path)
        for index in range(5):
            (tmp_path / f"stray{index}.txt").write_text(str(index))

        with patch("miles.utils.env_report.git_state._UNTRACKED_MAX_FILES", 2):
            info = _git_info(tmp_path)

        assert info.untracked_paths == ["stray0.txt", "stray1.txt"]
        assert info.untracked_paths_truncated is True
        assert info.uncommitted_hash is not None

    def test_truncated_untracked_file_count_contributes_to_the_hash(self, tmp_path: Path) -> None:
        """Different omitted file counts must not collide when the selected untracked files are identical."""
        _make_repo(tmp_path)
        for name in ("a.txt", "b.txt", "c.txt"):
            (tmp_path / name).write_text(name)

        with patch("miles.utils.env_report.git_state._UNTRACKED_MAX_FILES", 2):
            three_files = _git_info(tmp_path)
            (tmp_path / "d.txt").write_text("d.txt")
            four_files = _git_info(tmp_path)

        assert three_files.untracked_paths == four_files.untracked_paths == ["a.txt", "b.txt"]
        assert three_files.uncommitted_hash != four_files.uncommitted_hash

    def test_a_small_untracked_file_is_hashed_by_content(self, tmp_path: Path) -> None:
        """The size branch must stay reserved for the huge files it exists for."""
        _make_repo(tmp_path)
        (tmp_path / "small.txt").write_text("x")

        assert _git_info(tmp_path).untracked_unhashed_paths == []

    def test_two_unmerged_indexes_holding_different_conflicts_do_not_share_a_hash(self, tmp_path: Path) -> None:
        """A half-merged checkout is exactly the state a rank is wrong in, and two of them must not compare equal."""
        left = tmp_path / "left"
        right = tmp_path / "right"
        _make_conflicted_repo(left, ours="ours\n", theirs="theirs-left\n")
        _make_conflicted_repo(right, ours="ours\n", theirs="theirs-right\n")

        info = _git_info(left)

        assert info.dirty is True
        assert info.uncommitted_hash != _git_info(right).uncommitted_hash

    def test_a_combined_patch_is_split_off_the_stat_rather_than_dropped(self) -> None:
        """git describes an unmerged path with a combined header, and folding it into the stat hashes none of it."""
        stdout = b" f.txt | 4 ++++\n 1 file changed\n\ndiff --cc f.txt\n@@@ -1,1 -1,1 +1,2 @@@\n++conflict\n"

        with patch("miles.utils.env_report.git_state._run_git", return_value=_completed(stdout)):
            diff = git_state._collect_diff(location=".")

        assert diff.stat == "f.txt | 4 ++++\n 1 file changed"
        assert diff.patch == stdout[stdout.index(b"diff --cc ") :]

    def test_an_unmerged_path_line_is_part_of_the_patch_not_of_the_stat(self) -> None:
        """git names a path it cannot diff instead of diffing it, and that name is the only trace of the conflict."""
        stdout = b" f.txt | Unmerged\n 0 files changed\n\n* Unmerged path f.txt\n"

        with patch("miles.utils.env_report.git_state._run_git", return_value=_completed(stdout)):
            diff = git_state._collect_diff(location=".")

        assert diff.stat == "f.txt | Unmerged\n 0 files changed"
        assert diff.patch == b"* Unmerged path f.txt\n"


class TestTheReportedGitState:
    def test_a_repo_whose_head_moves_during_collection_is_reported_once_it_settles(self) -> None:
        """A commit taken before a swap and a diff taken after it describe a tree that never existed."""
        commits = iter(["moving", "settled", "settled", "settled"])

        with patch("miles.utils.env_report.git_state._run_git", side_effect=_head_of_call(commits)):
            info = collect_git_info(package_name="miles", location=".")

        assert info is not None
        assert info.commit == "settled"

    def test_a_repo_whose_head_never_settles_is_not_reported_at_all(self, caplog) -> None:
        """A wrong git state is worse than a missing one, and the next report will try again."""
        commits = iter(["one", "two", "three", "four"])

        with patch("miles.utils.env_report.git_state._run_git", side_effect=_head_of_call(commits)):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.git_state"):
                info = collect_git_info(package_name="miles", location=".")

        assert info is None
        assert "its HEAD keeps moving" in caplog.text


def _head_of_call(commits):
    def _run(*, args: list[str], location: str) -> subprocess.CompletedProcess:
        stdout = next(commits).encode() if args[:2] == ["rev-parse", "HEAD"] else b""
        return subprocess.CompletedProcess(args=["git", *args], returncode=0, stdout=stdout, stderr=b"")

    return _run


def _make_repo_with_subpackage(root: Path) -> Path:
    _make_repo(root)
    package = root / "python"
    package.mkdir()
    (package / "module.py").write_text("x = 1\n")
    _git(root, "add", "python/module.py")
    _git(root, "commit", "-m", "add the package directory")
    return package


class TestTheUntrackedFilesOfAnEditablePackageInstalledFromASubdirectory:
    def test_a_file_outside_the_package_directory_is_still_enumerated(self, tmp_path: Path) -> None:
        """sglang installs from python/, and the commit and the diff describe the whole repository."""
        package = _make_repo_with_subpackage(tmp_path)
        (tmp_path / "root_stray.py").write_text("x = 1\n")

        assert _git_info(package).untracked_paths == ["root_stray.py"]

    def test_a_file_inside_the_package_directory_is_named_from_the_repository_root(self, tmp_path: Path) -> None:
        """Two ranks installed under different paths must agree on the name of the same untracked file."""
        package = _make_repo_with_subpackage(tmp_path)
        (package / "sub_stray.py").write_text("x = 1\n")

        assert _git_info(package).untracked_paths == ["python/sub_stray.py"]

    def test_a_file_outside_the_package_directory_reaches_the_hash(self, tmp_path: Path) -> None:
        """An untracked module the report cannot see is exactly the difference between two ranks it hides."""
        package = _make_repo_with_subpackage(tmp_path)
        clean_hash = _git_info(package).uncommitted_hash

        (tmp_path / "root_stray.py").write_text("x = 1\n")

        assert _git_info(package).uncommitted_hash != clean_hash
        assert _git_info(package).dirty is True

    def test_the_content_of_a_file_outside_the_package_directory_reaches_the_hash(self, tmp_path: Path) -> None:
        """Hashing by name alone would call two different patches of the same repository one run."""
        package = _make_repo_with_subpackage(tmp_path)
        (tmp_path / "root_stray.py").write_text("x = 1\n")
        first = _git_info(package).uncommitted_hash

        (tmp_path / "root_stray.py").write_text("x = 2\n")

        assert _git_info(package).uncommitted_hash != first

    def test_the_subdirectory_and_the_root_report_the_same_untracked_state(self, tmp_path: Path) -> None:
        """The location a package was installed from must not change what the report says about its repository."""
        package = _make_repo_with_subpackage(tmp_path)
        (tmp_path / "root_stray.py").write_text("x = 1\n")
        (package / "sub_stray.py").write_text("x = 1\n")

        assert _git_info(package).untracked_paths == _git_info(tmp_path).untracked_paths
        assert _git_info(package).uncommitted_hash == _git_info(tmp_path).uncommitted_hash


class TestResolvingTheRepositoryRoot:
    def test_a_subdirectory_of_a_repository_resolves_to_the_repository_root(self, tmp_path: Path) -> None:
        """Everything the untracked enumeration reads is relative to this path."""
        package = _make_repo_with_subpackage(tmp_path)

        root = git_state._resolve_repo_root(location=str(package))

        assert (root / "tracked.txt").exists()

    def test_a_location_git_cannot_answer_for_is_taken_as_the_root_itself(self, tmp_path: Path) -> None:
        """A package outside any repository must be read as it always was, not crash the report."""
        assert git_state._resolve_repo_root(location=str(tmp_path)) == tmp_path
