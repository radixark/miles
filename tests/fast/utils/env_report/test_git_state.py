import os
import subprocess

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
