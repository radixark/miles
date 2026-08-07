import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
import types
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import EditablePackageInfo, EnvReportEvent, GitRepoInfo, NodeEnvReport
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.env_report import (
    LAUNCHER_REPORT_ENV_VAR,
    EnvReporter,
    _collect_git_info,
    _collect_pip_info,
    _is_editable,
    _parse_pip_entry,
    collect_key_versions,
    collect_node_env_report,
    collect_process_env_snapshot,
    decode_env_report,
    dump_args,
    log_env_report,
    redact,
    redact_argv,
    redact_env_vars,
    start_env_reporting,
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


def _args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**{"env_report": "", "env_report_interval_seconds": 3600.0, **overrides})


def _mock_pip_inspect() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["pip", "inspect"], returncode=0, stdout=json.dumps(_SAMPLE_PIP_INSPECT), stderr=""
    )


@pytest.fixture()
def mocked_pip_inspect():
    with patch("miles.utils.env_report.subprocess.run", return_value=_mock_pip_inspect()):
        yield


@pytest.fixture()
def event_log_dir(tmp_path: Path) -> Path:
    set_event_logger(EventLogger(log_dir=tmp_path, source=MainProcessIdentity()))
    yield tmp_path
    set_event_logger(None)


@pytest.fixture()
def without_event_logger():
    set_event_logger(None)
    yield
    set_event_logger(None)


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
            editable, full_list = _collect_pip_info({})

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
            editable, full_list = _collect_pip_info({})
        assert editable == []
        assert full_list == []

    def test_pip_inspect_exception_returns_empty(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", side_effect=OSError("no pip")):
            editable, full_list = _collect_pip_info({})
        assert editable == []
        assert full_list == []

    def test_runs_pip_with_the_environment_it_is_given(self) -> None:
        """The caller hands in a snapshot; reading os.environ here would race whoever mutates it."""
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(_SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch("miles.utils.env_report.subprocess.run", return_value=mock_result) as mock_run:
            _collect_pip_info({"HOME": "/root"})

        assert mock_run.call_args.kwargs["env"] == {"HOME": "/root"}


class TestCollectProcessEnvSnapshot:
    def test_excludes_pythonpath_from_the_pip_probe(self) -> None:
        """pip inspect misses editable packages whose source is on the PYTHONPATH."""
        with patch.dict(os.environ, {"PYTHONPATH": "/workspace/Megatron-LM"}):
            snapshot = collect_process_env_snapshot(_args())

        assert "PYTHONPATH" not in snapshot.probe_env

    def test_snapshots_the_environment_the_probe_will_run_with(self) -> None:
        """The probe runs on another thread, while the caller is still setting RANK and friends."""
        with patch.dict(os.environ, {"MILES_TEST_ENV_REPORT_MARK": "before"}):
            snapshot = collect_process_env_snapshot(_args())
            os.environ["MILES_TEST_ENV_REPORT_MARK"] = "after"

            assert snapshot.probe_env["MILES_TEST_ENV_REPORT_MARK"] == "before"


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


class TestRedactArgv:
    def test_hides_the_value_of_a_secret_flag(self) -> None:
        """A hashed wandb_key in args is pointless while the same key sits in argv verbatim."""
        argv = redact_argv(["train.py", "--wandb-key", "s3cret", "--reward-key", "reward"])
        assert "s3cret" not in argv
        assert argv[:2] == ["train.py", "--wandb-key"]
        assert argv[-2:] == ["--reward-key", "reward"]

    def test_hides_the_value_of_an_inline_secret_flag(self) -> None:
        argv = redact_argv(["train.py", "--wandb-key=s3cret"])
        assert "s3cret" not in argv[1]
        assert argv[1].startswith("--wandb-key=redacted-sha256:")

    def test_keeps_an_argv_without_secrets_unchanged(self) -> None:
        argv = ["train.py", "--reward-key", "reward", "--lr=1e-4"]
        assert redact_argv(argv) == argv

    def test_a_trailing_secret_flag_hides_nothing_that_follows(self) -> None:
        assert redact_argv(["train.py", "--wandb-key"]) == ["train.py", "--wandb-key"]


class TestRedact:
    def test_same_secret_hashes_to_same_digest(self) -> None:
        """Skew auditing needs to compare secrets across processes without revealing them."""
        assert redact("hunter2") == redact("hunter2")
        assert redact("hunter2") != redact("hunter3")
        assert "hunter2" not in redact("hunter2")


class TestRedactEnvVars:
    @pytest.mark.parametrize(
        "name",
        [
            "WANDB_API_KEY",
            "HF_TOKEN",
            "MY_SECRET",
            "DB_PASSWORD",
            "PG_PASSWD",
            "GCP_CREDENTIALS",
            "hf_token",
            "NEON_DATABASE_URL",
        ],
    )
    def test_redacts_a_secret_named_variable(self, name: str) -> None:
        """Every secret-ish name suffix is redacted, whatever its case."""
        assert "s3cret" not in redact_env_vars({name: "s3cret"})[name]

    @pytest.mark.parametrize(
        "name", ["TOKENIZERS_PARALLELISM", "KEYRING_PATH", "SSH_KEY_FILE", "CUDA_VISIBLE_DEVICES"]
    )
    def test_keeps_a_variable_that_merely_contains_a_secret_word(self, name: str) -> None:
        """Hashing every name containing 'key' would erase exactly the values an audit reads."""
        assert redact_env_vars({name: "plain"})[name] == "plain"

    def test_sorts_and_keeps_every_variable(self) -> None:
        """The report dumps all env vars, so redaction must never drop one."""
        redacted = redact_env_vars({"ZZZ": "z", "HF_TOKEN": "t", "RANK": "3"})
        assert list(redacted.keys()) == ["HF_TOKEN", "RANK", "ZZZ"]
        assert redacted["RANK"] == "3"


class TestDumpArgs:
    def test_keeps_serializable_values(self) -> None:
        dump = dump_args(_args(lr=1e-4, tags=["a"], nested={"x": 1}, flag=True, missing=None))
        assert dump.skipped_names == []
        assert dump.values["lr"] == 1e-4
        assert dump.values["tags"] == ["a"]
        assert dump.values["nested"] == {"x": 1}
        assert dump.values["missing"] is None

    def test_skips_unserializable_values(self) -> None:
        """A non-JSON arg is skipped by name instead of being coerced to a lossy string."""
        dump = dump_args(_args(model=object(), lr=1.0))
        assert dump.skipped_names == ["model"]
        assert "model" not in dump.values
        assert dump.values["lr"] == 1.0

    def test_redacts_a_declared_secret_arg(self) -> None:
        dump = dump_args(_args(wandb_key="abc"))
        assert dump.values["wandb_key"].startswith("redacted-sha256:")

    def test_keeps_an_unset_secret_arg_as_none(self) -> None:
        """--wandb-key defaults to None, and hashing that would crash every process at startup."""
        assert dump_args(_args(wandb_key=None)).values["wandb_key"] is None

    def test_snapshots_nested_values_instead_of_referencing_them(self) -> None:
        """The dump outlives the caller, which is free to keep mutating its own args."""
        tags = ["a"]
        dump = dump_args(_args(tags=tags))
        tags.append("b")
        assert dump.values["tags"] == ["a"]

    def test_keeps_dataset_column_args_that_merely_end_in_key(self) -> None:
        """--reward-key names a dataset column; hashing it would hide the run's actual configuration."""
        dump = dump_args(_args(reward_key="reward", input_key="prompt"))
        assert dump.values["reward_key"] == "reward"
        assert dump.values["input_key"] == "prompt"

    def test_dump_is_json_serializable(self) -> None:
        dump = dump_args(_args(model=object(), lr=1.0))
        assert json.loads(json.dumps(dump.model_dump()))["values"]["lr"] == 1.0


class TestCollectKeyVersions:
    def test_reports_python_and_known_packages(self) -> None:
        versions = collect_key_versions(
            [{"name": "torch", "version": "2.5.0"}, {"name": "SGLang", "version": "0.4.0"}]
        )
        assert versions["python"] == ".".join(str(part) for part in sys.version_info[:3])
        assert versions["sglang"] == "0.4.0"
        assert "platform" in versions

    def test_ignores_unknown_packages(self) -> None:
        versions = collect_key_versions([{"name": "numpy", "version": "1.26.0"}])
        assert "numpy" not in versions

    def test_reports_torch_cuda_when_torch_is_imported(self) -> None:
        torch = types.SimpleNamespace(__version__="2.5.0", version=types.SimpleNamespace(cuda="12.4"))
        with patch.dict(sys.modules, {"torch": torch}):
            versions = collect_key_versions([])

        assert versions["torch"] == "2.5.0"
        assert versions["torch_cuda"] == "12.4"

    def test_reports_an_empty_cuda_version_for_a_cpu_torch(self) -> None:
        torch = types.SimpleNamespace(__version__="2.5.0", version=types.SimpleNamespace(cuda=None))
        with patch.dict(sys.modules, {"torch": torch}):
            assert collect_key_versions([])["torch_cuda"] == ""

    def test_reports_nothing_about_an_unimported_torch(self) -> None:
        """torch is read from sys.modules, so an unimported torch costs nothing and reports nothing."""
        with patch.dict(sys.modules, {"torch": None}):
            assert "torch_cuda" not in collect_key_versions([])


class TestCollectNodeEnvReport:
    def _collect(self, **overrides) -> NodeEnvReport:
        return collect_node_env_report(snapshot=collect_process_env_snapshot(_args(**overrides)))

    def test_returns_structured_report(self, mocked_pip_inspect) -> None:
        report = self._collect(env_report='{"flavor": "test"}')

        assert isinstance(report, NodeEnvReport)
        assert report.process.launcher_env_report == {"flavor": "test"}
        assert len(report.editable_packages) == 2
        assert len(report.full_pip_list) == 4

    def test_records_process_identity_context(self, mocked_pip_inspect) -> None:
        """The audit needs to know which host and command line produced this report."""
        report = self._collect(lr=1.0)
        assert report.process.hostname
        assert report.process.argv == sys.argv
        assert report.process.args.values["lr"] == 1.0

    def test_records_redacted_environment(self, mocked_pip_inspect) -> None:
        with patch.dict(os.environ, {"MILES_TEST_ENV_REPORT_TOKEN": "hunter2", "MILES_TEST_ENV_REPORT_RANK": "7"}):
            report = self._collect()

        assert report.process.env_vars["MILES_TEST_ENV_REPORT_RANK"] == "7"
        assert "hunter2" not in report.process.env_vars["MILES_TEST_ENV_REPORT_TOKEN"]

    def test_leaves_the_launcher_record_out_of_the_environment_dump(self, mocked_pip_inspect) -> None:
        """The launcher's record is already stored decoded, and it is large enough to matter."""
        with patch.dict(os.environ, {LAUNCHER_REPORT_ENV_VAR: '{"flavor": "test"}'}):
            report = self._collect(env_report='{"flavor": "test"}')

        assert LAUNCHER_REPORT_ENV_VAR not in report.process.env_vars
        assert report.process.launcher_env_report == {"flavor": "test"}

    def test_records_key_versions(self, mocked_pip_inspect) -> None:
        report = self._collect()
        assert report.key_versions["sglang"] == "0.4.0"

    def test_empty_partial_env_report(self, mocked_pip_inspect) -> None:
        assert self._collect(env_report="").process.launcher_env_report is None

    def test_invalid_json_partial_env_report(self, mocked_pip_inspect) -> None:
        assert self._collect(env_report="not json").process.launcher_env_report is None

    def test_report_serializable(self, mocked_pip_inspect) -> None:
        report = self._collect(env_report='{"x": 1}', model=object())
        parsed = json.loads(report.model_dump_json())
        assert parsed["editable_packages"][0]["name"] == "miles"
        assert parsed["process"]["args"]["skipped_names"] == ["model"]


class TestLogEnvReport:
    def _log(self, **overrides) -> None:
        log_env_report(snapshot=collect_process_env_snapshot(_args(**overrides)))

    def test_writes_one_event_the_analyzer_can_read_back(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """The report is stored as a normal event, so replaying a run's jsonl recovers its environment."""
        self._log(lr=1.0)

        events = read_events(event_log_dir)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, EnvReportEvent)
        assert event.source == MainProcessIdentity()
        assert event.report.process.args.values["lr"] == 1.0
        assert event.report.process.hostname

    def test_summarises_the_report_on_stdout_instead_of_dumping_it(
        self, mocked_pip_inspect, event_log_dir: Path, caplog
    ) -> None:
        """A full report is tens of kilobytes; logging it per process would drown the logs."""
        with caplog.at_level(logging.INFO, logger="miles.utils.env_report"):
            self._log(lr=1.0)

        assert "op=env_report" in caplog.text
        assert "num_packages=4" in caplog.text
        assert "PYTHONUNBUFFERED" not in caplog.text

    def test_summarises_the_report_when_no_event_logger_is_configured(
        self, mocked_pip_inspect, without_event_logger, caplog
    ) -> None:
        """A run without an event dir still leaves a trace instead of silently dropping the report."""
        with caplog.at_level(logging.INFO, logger="miles.utils.env_report"):
            self._log()

        assert "op=env_report" in caplog.text
        assert "stored=false" in caplog.text


class TestEnvReporter:
    def _reporter(self, *, interval_seconds: float) -> EnvReporter:
        return EnvReporter(snapshot=collect_process_env_snapshot(_args()), interval_seconds=interval_seconds)

    def test_reports_from_a_background_thread(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """Collection shells out, so it must not sit on the caller's startup path."""
        collecting = threading.Event()
        release = threading.Event()

        def block(*, snapshot):
            collecting.set()
            release.wait(timeout=30.0)

        with patch("miles.utils.env_report.log_env_report", side_effect=block):
            reporter = start_env_reporting(_args())
            assert collecting.wait(timeout=30.0)
            release.set()
            reporter.stop()

    def test_snapshots_the_environment_before_the_caller_changes_it(
        self, mocked_pip_inspect, event_log_dir: Path
    ) -> None:
        """The reporting thread must not read args or os.environ while the caller is still mutating them."""
        tags = ["a"]
        args = _args(lr=1.0, tags=tags)

        reporter = start_env_reporting(args)
        args.lr = 2.0
        tags.append("b")
        reporter.stop()

        recorded = read_events(event_log_dir)[0].report.process.args
        assert recorded.values["lr"] == 1.0
        assert recorded.values["tags"] == ["a"]

    def test_reports_again_after_the_interval(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """Code can be swapped under a long-lived process, so one report at startup is not enough."""
        reported_twice = threading.Event()
        commits = iter(["commit-one", "commit-two"])

        def changing_git_info(*, package_name: str, location: str):
            try:
                commit = next(commits)
            except StopIteration:
                reported_twice.set()
                return None
            return GitRepoInfo(package_name=package_name, location=location, commit=commit, dirty=False, diff_stat="")

        reporter = self._reporter(interval_seconds=0.01)
        with patch("miles.utils.env_report._collect_git_info", side_effect=changing_git_info):
            reporter.start()
            assert reported_twice.wait(timeout=30.0)
            reporter.stop()

        reports = [event.report for event in read_events(event_log_dir)]
        assert [repo.commit for report in reports[:2] for repo in report.git_repos][:2] == [
            "commit-one",
            "commit-two",
        ]

    @pytest.mark.parametrize("interval_seconds", [0.0, -1.0])
    def test_a_non_positive_interval_reports_only_at_startup(
        self, mocked_pip_inspect, event_log_dir: Path, interval_seconds: float
    ) -> None:
        reporter = self._reporter(interval_seconds=interval_seconds)

        reporter.start()
        reporter.stop()
        time.sleep(0.2)

        assert len(read_events(event_log_dir)) == 1

    @pytest.mark.parametrize("interval_seconds", [float("nan"), float("inf")])
    def test_a_non_finite_interval_is_refused(self, interval_seconds: float) -> None:
        """nan never waits and inf overflows the wait, so both silently break the reporter."""
        with pytest.raises(AssertionError):
            self._reporter(interval_seconds=interval_seconds)

    def test_a_failing_report_does_not_stop_the_reporter(
        self, mocked_pip_inspect, event_log_dir: Path, caplog
    ) -> None:
        """A broken environment probe must never take the process, or the next report, down with it."""
        succeeded = threading.Event()
        outcomes = iter([RuntimeError("boom")])

        def fail_once(*, snapshot):
            try:
                raise next(outcomes)
            except StopIteration:
                succeeded.set()

        reporter = self._reporter(interval_seconds=0.01)
        with patch("miles.utils.env_report.log_env_report", side_effect=fail_once):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report"):
                reporter.start()
                assert succeeded.wait(timeout=30.0)
                reporter.stop()

        assert "Failed to log the env report" in caplog.text


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

    def test_detects_clean_editable_package(self, editable_package) -> None:
        """Verify env report finds the package with correct git commit, not dirty."""
        pkg_name = editable_package["pkg_name"]
        repo = editable_package["repo"]
        expected_commit = editable_package["commit"]

        # Step 1: Run the full collection (no mocks)
        report = collect_node_env_report(snapshot=collect_process_env_snapshot(_args(env_report='{"test": true}')))

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
        report = collect_node_env_report(snapshot=collect_process_env_snapshot(_args()))

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
        report = collect_node_env_report(snapshot=collect_process_env_snapshot(_args()))

        # Step 3: Verify dirty
        git_info = next(
            (r for r in report.git_repos if r.package_name == pkg_name),
            None,
        )
        assert git_info is not None
        assert git_info.dirty is True
        assert "__init__.py" in git_info.diff_stat
