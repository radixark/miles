import logging
import math
import random
import threading
from typing import Any

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import EnvReport, EnvReportEvent, EnvReportGitRepoInfo
from miles.utils.env_report.collector import (
    collect_env_report,
    collect_env_report_snapshot,
    collect_unprobed_env_report,
)
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)
_REPORTER_THREAD_NAME = "env-report"
_INTERVAL_JITTER_RATIO = 0.5
_STOP_TIMEOUT_SECONDS = 5.0
_SUMMARY_HASH_CHARS = 12
SETTLED_DELAY_SECONDS = 300.0


def start_env_reporting(args: Any) -> "EnvReporter | None":
    try:
        reporter = EnvReporter(args=args, interval_seconds=args.env_report_interval_seconds)
        reporter.start()
        return reporter
    except Exception:
        logger.warning("Failed to start the env reporting of this process", exc_info=True)
        return None


class EnvReporter:
    def __init__(self, *, args: Any, interval_seconds: float) -> None:
        assert math.isfinite(interval_seconds), (
            f"--env-report-interval-seconds is {interval_seconds}, which is neither a delay nor a way to say "
            f"'only at startup'; pass a finite number"
        )

        self._args = args
        self._interval_seconds = interval_seconds
        self._jitter = random.Random()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=_REPORTER_THREAD_NAME, daemon=True)
        _log_env_report(report=collect_unprobed_env_report(snapshot=collect_env_report_snapshot(args)))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=_STOP_TIMEOUT_SECONDS)

    def _run(self) -> None:
        delay = min(SETTLED_DELAY_SECONDS, self._interval_seconds) if self._interval_seconds > 0 else 0.0
        while True:
            self._report_once()
            if self._interval_seconds <= 0 or self._stopped.wait(delay):
                return
            delay = self._next_delay_seconds()

    def _report_once(self) -> None:
        try:
            snapshot = collect_env_report_snapshot(self._args)
            _log_env_report(report=collect_env_report(snapshot=snapshot))
        except Exception:
            logger.warning("Failed to log the env report", exc_info=True)

    def _next_delay_seconds(self) -> float:
        jittered = self._interval_seconds * (1.0 + self._jitter.random() * _INTERVAL_JITTER_RATIO)
        return min(jittered, threading.TIMEOUT_MAX)


def _log_env_report(*, report: EnvReport) -> EnvReport:
    if is_event_logger_initialized():
        get_event_logger().log(EnvReportEvent, {"report": report}, print_log=False)
    _log_report_summary(report)

    return report


def _log_report_summary(report: EnvReport) -> None:
    log_structured(
        logger.info,
        tag="audit",
        op="env_report_summary",
        hostname=report.process.hostname,
        versions=report.key_versions,
        repos={repo.package_name: _summarise_repo(repo) for repo in report.git_repos},
        num_packages=len(report.full_pip_list),
        num_env_vars=len(report.process.env_vars),
        packages_probed=report.packages_probed,
        stored=is_event_logger_initialized(),
    )


def _summarise_repo(repo: EnvReportGitRepoInfo) -> str:
    if not repo.dirty:
        return repo.commit
    return f"{repo.commit}-dirty-{repo.uncommitted_hash[:_SUMMARY_HASH_CHARS] if repo.uncommitted_hash else 'unknown'}"
