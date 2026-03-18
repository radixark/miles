import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EditablePackageInfo:
    name: str
    version: str
    location: str


@dataclass
class GitRepoInfo:
    package_name: str
    location: str
    commit: str
    dirty: bool
    diff_stat: str


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
    launcher_report = None
    if partial_env_report:
        try:
            launcher_report = json.loads(partial_env_report)
        except json.JSONDecodeError:
            logger.warning("Failed to parse partial_env_report", exc_info=True)

    editable_packages, full_pip_list = _collect_pip_info()

    git_repos = [
        info
        for pkg in editable_packages
        if (info := _collect_git_info(package_name=pkg.name, location=pkg.location))
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


def _collect_pip_info() -> tuple[list[EditablePackageInfo], list[dict[str, str]]]:
    """Collect all pip info in a single `pip inspect` call.

    Returns (editable_packages, full_pip_list).
    """
    try:
        result = subprocess.run(
            ["pip", "inspect"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning("pip inspect failed: %s", result.stderr)
            return [], []

        data = json.loads(result.stdout)
        installed: list[dict[str, Any]] = data.get("installed", [])

        editable_packages: list[EditablePackageInfo] = []
        full_pip_list: list[dict[str, str]] = []

        for pkg in installed:
            metadata = pkg.get("metadata", {})
            name = metadata.get("name", "")
            version = metadata.get("version", "")
            full_pip_list.append({"name": name, "version": version})

            direct_url = pkg.get("direct_url")
            if direct_url and direct_url.get("dir_info", {}).get("editable"):
                url = direct_url.get("url", "")
                location = url.removeprefix("file://")
                editable_packages.append(EditablePackageInfo(
                    name=name, version=version, location=location,
                ))

        return editable_packages, full_pip_list
    except Exception:
        logger.warning("Failed to collect pip info", exc_info=True)
        return [], []


def _collect_git_info(*, package_name: str, location: str) -> GitRepoInfo | None:
    if not location or not os.path.isdir(location):
        return None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=location,
        )
        if commit_result.returncode != 0:
            return None
        commit = commit_result.stdout.strip()

        diff_result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=location,
        )
        diff_stat = diff_result.stdout.strip()
        dirty = bool(diff_stat)

        return GitRepoInfo(
            package_name=package_name,
            location=location,
            commit=commit,
            dirty=dirty,
            diff_stat=diff_stat,
        )
    except Exception:
        logger.warning(
            "Failed to collect git info for %s at %s", package_name, location, exc_info=True
        )
        return None


def _print_report(report: NodeEnvReport) -> None:
    print(f"========== ENV REPORT (role={report.role}, rank={report.rank}) ==========")
    print(json.dumps(asdict(report), indent=2, default=str))
    print("=" * 60)
