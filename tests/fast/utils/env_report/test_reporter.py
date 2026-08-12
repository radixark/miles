import logging
import random
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.fast.utils.env_report.conftest import make_args, mock_pip_inspect

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import EnvReportEvent, EnvReportGitRepoInfo
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.env_report.collector import collect_env_report, collect_env_report_snapshot
from miles.utils.env_report.reporter import EnvReporter, _log_env_report, _summarise_repo, start_env_reporting


class TestSummariseRepo:
    def test_dirty_repo_summaries_are_distinguished_by_truncated_uncommitted_hash(self) -> None:
        """Dirty trees at one commit must be distinguished by exactly twelve hash characters."""
        first = _repo_info(commit="abc", dirty=True, uncommitted_hash="1" * 64)
        second = _repo_info(commit="abc", dirty=True, uncommitted_hash="2" * 64)

        assert _summarise_repo(first) == "abc-dirty-111111111111"
        assert _summarise_repo(second) == "abc-dirty-222222222222"

    def test_repo_summary_distinguishes_clean_state_from_dirty_state_with_unknown_hash(self) -> None:
        """A clean commit stays verbatim while a dirty tree without a hash is explicitly unknown."""
        clean = _repo_info(commit="abc", dirty=False, uncommitted_hash=None)
        dirty = _repo_info(commit="abc", dirty=True, uncommitted_hash=None)

        assert _summarise_repo(clean) == "abc"
        assert _summarise_repo(dirty) == "abc-dirty-unknown"


def _repo_info(*, commit: str, dirty: bool, uncommitted_hash: str | None) -> EnvReportGitRepoInfo:
    return EnvReportGitRepoInfo(
        package_name="miles",
        location="/repo",
        commit=commit,
        dirty=dirty,
        diff_stat="",
        uncommitted_hash=uncommitted_hash,
        untracked_paths=[],
        untracked_paths_truncated=False,
        untracked_unhashed_paths=[],
    )


class TestLogEnvReport:
    def _log(self, **overrides) -> None:
        _log_env_report(report=collect_env_report(snapshot=collect_env_report_snapshot(make_args(**overrides))))

    def test_writes_one_event_the_analyzer_can_read_back(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """The report is stored as a normal event, so replaying a run's jsonl recovers its environment."""
        self._log(lr=1.0)

        events = read_events(event_log_dir)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, EnvReportEvent)
        assert event.source == SimpleProcessIdentity(component="main")
        assert event.report.process.args.values["lr"] == 1.0
        assert event.report.process.hostname

    def test_summarises_the_report_on_stdout_instead_of_dumping_it(
        self, mocked_pip_inspect, event_log_dir: Path, caplog
    ) -> None:
        """A full report is tens of kilobytes; logging it per process would drown the logs."""
        with caplog.at_level(logging.INFO, logger="miles.utils.env_report.reporter"):
            self._log(lr=1.0)

        assert "op=env_report" in caplog.text
        assert "num_packages=4" in caplog.text
        assert "PYTHONUNBUFFERED" not in caplog.text

    def test_summarises_the_report_when_no_event_logger_is_configured(
        self, mocked_pip_inspect, without_event_logger, caplog
    ) -> None:
        """A run without an event dir still leaves a trace instead of silently dropping the report."""
        with caplog.at_level(logging.INFO, logger="miles.utils.env_report.reporter"):
            self._log()

        assert "op=env_report" in caplog.text
        assert "stored=false" in caplog.text


class TestTheStartupReport:
    def test_a_thread_start_failure_only_warns_and_returns_none(self, mocked_pip_inspect, caplog) -> None:
        """A thread start failure must warn with its traceback without disrupting process startup."""
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.reporter"):
                reporter = start_env_reporting(make_args())

        assert reporter is None
        assert "Failed to start the env reporting" in caplog.text
        assert caplog.records[-1].exc_info is not None

    def test_is_stored_before_the_caller_gets_its_reporter_back(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """A process that dies in its first seconds is exactly the one whose startup has to be on record."""
        never_finish = threading.Event()

        with patch("miles.utils.env_report.reporter.collect_env_report", side_effect=lambda **_: never_finish.wait()):
            start_env_reporting(make_args(lr=1.0))

            events = read_events(event_log_dir)
            never_finish.set()

        assert [event.report.packages_probed for event in events] == [False]
        assert events[0].report.process.args.values["lr"] == 1.0

    def test_names_no_package_it_has_not_probed(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """An empty pip list that claims to be probed would read as a machine with nothing installed."""
        never_finish = threading.Event()

        with patch("miles.utils.env_report.reporter.collect_env_report", side_effect=lambda **_: never_finish.wait()):
            start_env_reporting(make_args())
            report = read_events(event_log_dir)[0].report
            never_finish.set()

        assert report.full_pip_list == []
        assert report.git_repos == []
        assert report.packages_probed is False


class TestEnvReporter:
    def _reporter(self, *, interval_seconds: float, **overrides) -> EnvReporter:
        return EnvReporter(args=make_args(**overrides), interval_seconds=interval_seconds)

    def test_reports_from_a_background_thread(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """Collection shells out, so it must not sit on the caller's startup path."""
        collecting = threading.Event()
        release = threading.Event()

        def block(**kwargs):
            collecting.set()
            release.wait(timeout=30.0)

        with patch("miles.utils.env_report.reporter.collect_env_report", side_effect=block):
            reporter = start_env_reporting(make_args())
            assert collecting.wait(timeout=30.0)
            release.set()
            reporter.stop()

    def test_stop_returns_when_an_environment_probe_is_blocked(self, mocked_pip_inspect) -> None:
        """Stopping uses its bounded join even when an active environment probe cannot finish."""
        collecting = threading.Event()
        release = threading.Event()
        stop_completed = threading.Event()

        def block(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            collecting.set()
            release.wait(timeout=30.0)
            return mock_pip_inspect()

        def stop_reporter() -> None:
            reporter.stop()
            stop_completed.set()

        with patch("miles.utils.env_report.collector.subprocess.run", side_effect=block):
            reporter = self._reporter(interval_seconds=60.0)
            stopper: threading.Thread | None = None
            try:
                with patch("miles.utils.env_report.reporter._STOP_TIMEOUT_SECONDS", 0.01):
                    reporter.start()
                    assert collecting.wait(timeout=30.0)
                    stopper = threading.Thread(target=stop_reporter, daemon=True)
                    stopper.start()

                    assert stop_completed.wait(timeout=1.0)
            finally:
                release.set()
                if stopper is not None:
                    stopper.join(timeout=30.0)
                reporter.stop()

    def test_re_reads_the_environment_the_process_settled_on(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """A rank writes MASTER_ADDR after its logger is configured, and that is what it actually ran with."""
        args = make_args(lr=1.0)
        reporter = EnvReporter(args=args, interval_seconds=0.01)
        args.lr = 2.0

        with patch("miles.utils.env_report.reporter.SETTLED_DELAY_SECONDS", 0.0):
            reporter.start()
            _wait_for_events(event_log_dir, count=3)
            reporter.stop()

        recorded = [event.report.process.args.values["lr"] for event in read_events(event_log_dir)]
        assert recorded[0] == 1.0
        assert recorded[-1] == 2.0

    def test_settles_before_it_falls_back_to_the_interval(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """One hour is far too long to wait for the environment a process only finishes building at startup."""
        reporter = self._reporter(interval_seconds=3600.0)
        delays: list[float] = []

        with patch.object(reporter._stopped, "wait", side_effect=lambda delay: delays.append(delay) or True):
            reporter._run()

        assert delays == [300.0]

    def test_an_interval_shorter_than_the_settling_delay_wins(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """Asking for a report every minute must not be answered with a five minute wait."""
        reporter = self._reporter(interval_seconds=60.0)
        delays: list[float] = []

        with patch.object(reporter._stopped, "wait", side_effect=lambda delay: delays.append(delay) or True):
            reporter._run()

        assert delays == [60.0]

    def test_the_jitter_does_not_disturb_the_process_random_number_generator(self, mocked_pip_inspect) -> None:
        """Megatron seeds the global RNG for reproducibility, and this thread runs alongside it."""
        reporter = self._reporter(interval_seconds=60.0)
        random.seed(1234)
        expected = random.random()

        random.seed(1234)
        reporter._next_delay_seconds()

        assert random.random() == expected

    def test_an_interval_longer_than_a_wait_can_express_still_waits(self, mocked_pip_inspect) -> None:
        """A wait longer than TIMEOUT_MAX raises, which would end the process's only reporter."""
        reporter = self._reporter(interval_seconds=threading.TIMEOUT_MAX)

        assert reporter._next_delay_seconds() <= threading.TIMEOUT_MAX

    @pytest.mark.parametrize("interval_seconds", [0.0, -1.0])
    def test_a_non_positive_interval_reports_only_at_startup(
        self, mocked_pip_inspect, event_log_dir: Path, interval_seconds: float
    ) -> None:
        """The reporter must end by itself, rather than look finished because it was stopped."""
        reporter = self._reporter(interval_seconds=interval_seconds)

        reporter.start()
        reporter._thread.join(timeout=30.0)

        assert not reporter._thread.is_alive()
        assert [event.report.packages_probed for event in read_events(event_log_dir)] == [False, True]

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

        def fail_once(**kwargs):
            try:
                raise next(outcomes)
            except StopIteration:
                succeeded.set()

        reporter = self._reporter(interval_seconds=0.01)
        with patch("miles.utils.env_report.reporter.collect_env_report", side_effect=fail_once):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.reporter"):
                reporter.start()
                assert succeeded.wait(timeout=30.0)
                reporter.stop()

        assert "Failed to log the env report" in caplog.text

    def test_a_failing_snapshot_only_warns(self, event_log_dir: Path, caplog) -> None:
        """The snapshot runs on the caller's own thread, so an audit failure must not end its startup."""
        with patch("miles.utils.env_report.reporter.collect_env_report_snapshot", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.reporter"):
                start_env_reporting(make_args())

        assert "Failed to start the env reporting" in caplog.text
        assert read_events(event_log_dir) == []


def _wait_for_events(log_dir: Path, *, count: int) -> None:
    deadline = threading.Event()
    for _ in range(3000):
        if len(read_events(log_dir)) >= count:
            return
        deadline.wait(0.01)
    raise AssertionError(f"only {len(read_events(log_dir))} events were written, expected {count}")
