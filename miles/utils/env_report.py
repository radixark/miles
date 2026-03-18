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

    editable_packages = _collect_editable_packages()

    git_repos = [
        info
        for pkg in editable_packages
        if (info := _collect_git_info(package_name=pkg.name, location=pkg.location))
    ]

    full_pip_list = _collect_full_pip_list()

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


def _collect_editable_packages() -> list[EditablePackageInfo]:
    """Collect editable packages with locations via batched pip show."""
    try:
        result = subprocess.run(
            ["pip", "list", "--editable", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("pip list --editable failed: %s", result.stderr)
            return []

        packages = json.loads(result.stdout)
        if not packages:
            return []

        locations = _batch_get_package_locations([pkg["name"] for pkg in packages])
        return [
            EditablePackageInfo(
                name=pkg["name"],
                version=pkg["version"],
                location=locations.get(pkg["name"], ""),
            )
            for pkg in packages
        ]
    except Exception:
        logger.warning("Failed to collect editable packages", exc_info=True)
        return []


def _batch_get_package_locations(package_names: list[str]) -> dict[str, str]:
    """Get locations for multiple packages in a single pip show call."""
    try:
        result = subprocess.run(
            ["pip", "show", *package_names],
            capture_output=True,
            text=True,
            timeout=30,
        )

        locations: dict[str, str] = {}
        current_name = ""
        for line in result.stdout.splitlines():
            if line.startswith("Name:"):
                current_name = line.split(":", 1)[1].strip()
            elif line.startswith("Editable project location:"):
                locations[current_name] = line.split(":", 1)[1].strip()
            elif line.startswith("Location:") and current_name not in locations:
                locations[current_name] = line.split(":", 1)[1].strip()

        return locations
    except Exception:
        logger.warning("Failed to batch get package locations", exc_info=True)
        return {}


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


def _collect_full_pip_list() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("pip list failed: %s", result.stderr)
            return []
        return json.loads(result.stdout)
    except Exception:
        logger.warning("Failed to collect full pip list", exc_info=True)
        return []


def _print_report(report: NodeEnvReport) -> None:
    print(f"========== ENV REPORT (role={report.role}, rank={report.rank}) ==========")
    print(json.dumps(asdict(report), indent=2, default=str))
    print("=" * 60)
