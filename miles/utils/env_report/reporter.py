import logging
from typing import Any

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import EnvReport, EnvReportEvent, EnvReportGitRepoInfo
from miles.utils.env_report.collector import collect_env_report, collect_env_report_snapshot
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)


def start_env_reporting(args: Any) -> "EnvReporter | None":
    try:
        reporter = EnvReporter(args=args)
        reporter.start()
        return reporter
    except Exception:
        logger.warning("Failed to start the env reporting of this process", exc_info=True)
        return None


class EnvReporter:
    def __init__(self, *, args: Any) -> None:
        self._args = args

    def start(self) -> None:
        self._report_once()

    def _report_once(self) -> None:
        try:
            snapshot = collect_env_report_snapshot(self._args)
            _log_env_report(report=collect_env_report(snapshot=snapshot))
        except Exception:
            logger.warning("Failed to log the env report", exc_info=True)


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
        stored=is_event_logger_initialized(),
    )


def _summarise_repo(repo: EnvReportGitRepoInfo) -> str:
    return f"{repo.commit}-dirty" if repo.dirty else repo.commit
