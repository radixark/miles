import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from miles.utils.env_report.collector import EditablePackageInfo, collect_pip_info
from miles.utils.env_report.git_state import GitRepoInfo, collect_git_info
from miles.utils.env_report.launcher_report import decode_env_report

logger = logging.getLogger(__name__)


@dataclass
class NodeEnvReport:
    role: str
    rank: int
    launcher_env_report: dict[str, Any] | None
    editable_packages: list[EditablePackageInfo]
    git_repos: list[GitRepoInfo]
    full_pip_list: list[dict[str, str]]


def collect_and_print_node_env_report(
    *,
    role: str,
    rank: int,
    partial_env_report: str,
) -> NodeEnvReport:
    """Collect environment info for this node, print to stdout, return structured report.

    Called during actor init. Only performs collection when partial_env_report is non-empty.

    Args:
        role: Actor role, e.g. "training" or "rollout"
        rank: Actor rank
        partial_env_report: JSON string from launcher (may contain launch config info)
    """
    launcher_report = decode_env_report(partial_env_report)

    editable_packages, full_pip_list = collect_pip_info()

    git_repos = [
        info for pkg in editable_packages if (info := collect_git_info(package_name=pkg.name, location=pkg.location))
    ]

    report = NodeEnvReport(
        role=role,
        rank=rank,
        launcher_env_report=launcher_report,
        editable_packages=editable_packages,
        git_repos=git_repos,
        full_pip_list=full_pip_list,
    )

    _print_report(report)
    return report


ENV_REPORT_PREFIX = "ENV_REPORT_JSON="


def _print_report(report: NodeEnvReport) -> None:
    print(f"{ENV_REPORT_PREFIX}{json.dumps(asdict(report), separators=(',', ':'), sort_keys=True, default=str)}")
