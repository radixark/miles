import base64
import hashlib
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

_SECRET_ENV_VAR_PATTERN = re.compile(
    r"(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|DATABASE_URL)$", re.IGNORECASE
)
_SECRET_ARG_NAMES = frozenset({"wandb_key"})
_SECRET_ARG_FLAGS = frozenset(f"--{name.replace('_', '-')}" for name in _SECRET_ARG_NAMES)
_REDACTED_PREFIX = "redacted-sha256:"
_REDACTED_HASH_CHARS = 16
_KEY_PACKAGE_NAMES = (
    "miles",
    "sglang",
    "sglang-router",
    "megatron-core",
    "transformers",
    "ray",
    "flashinfer-python",
    "vllm",
)


class EditablePackageInfo(FrozenStrictBaseModel):
    name: str
    version: str
    location: str


class GitRepoInfo(FrozenStrictBaseModel):
    package_name: str
    location: str
    commit: str
    dirty: bool
    diff_stat: str


class ArgsDump(FrozenStrictBaseModel):
    values: dict[str, Any]
    skipped_names: list[str]


class NodeEnvReport(FrozenStrictBaseModel):
    role: str
    rank: int
    hostname: str
    argv: list[str]
    args: ArgsDump
    env_vars: dict[str, str]
    key_versions: dict[str, str]
    launcher_env_report: dict[str, Any] | None
    editable_packages: list[EditablePackageInfo]
    git_repos: list[GitRepoInfo]
    full_pip_list: list[dict[str, str]]


def decode_env_report(raw: str) -> dict[str, Any] | None:
    """Decode an env report string (base64-encoded JSON or raw JSON)."""
    if not raw:
        return None
    try:
        decoded = base64.b64decode(raw).decode()
        return json.loads(decoded)
    except Exception:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse env report", exc_info=True)
            return None


def collect_and_print_node_env_report(*, role: str, rank: int, args: Any) -> NodeEnvReport:
    report = collect_node_env_report(role=role, rank=rank, args=args)
    _print_report(report)
    return report


def collect_node_env_report(*, role: str, rank: int, args: Any) -> NodeEnvReport:
    editable_packages, full_pip_list = _collect_pip_info()

    git_repos = [
        info for pkg in editable_packages if (info := _collect_git_info(package_name=pkg.name, location=pkg.location))
    ]

    return NodeEnvReport(
        role=role,
        rank=rank,
        hostname=socket.gethostname(),
        argv=redact_argv(sys.argv),
        args=dump_args(args),
        env_vars=redact_env_vars(dict(os.environ)),
        key_versions=collect_key_versions(full_pip_list),
        launcher_env_report=decode_env_report(args.env_report),
        editable_packages=editable_packages,
        git_repos=git_repos,
        full_pip_list=full_pip_list,
    )


def redact_env_vars(env_vars: dict[str, str]) -> dict[str, str]:
    return {
        name: redact(value) if _SECRET_ENV_VAR_PATTERN.search(name) else value
        for name, value in sorted(env_vars.items())
    }


def redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for item in argv:
        if hide_next:
            redacted.append(redact(item))
            hide_next = False
            continue

        flag, separator, value = item.partition("=")
        if flag not in _SECRET_ARG_FLAGS:
            redacted.append(item)
            continue

        redacted.append(f"{flag}={redact(value)}" if separator else item)
        hide_next = not separator

    return redacted


def redact(value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:_REDACTED_HASH_CHARS]
    return f"{_REDACTED_PREFIX}{digest}"


def dump_args(args: Any) -> ArgsDump:
    declared = dict(vars(args))
    serializable = _json_snapshot(declared)
    if serializable is None:
        serializable = {}
        for name, value in declared.items():
            if (snapshot := _json_snapshot({name: value})) is not None:
                serializable.update(snapshot)

    values = {
        name: redact(value) if name in _SECRET_ARG_NAMES and isinstance(value, str) else value
        for name, value in sorted(serializable.items())
    }
    return ArgsDump(values=values, skipped_names=sorted(declared.keys() - serializable.keys()))


def collect_key_versions(full_pip_list: list[dict[str, str]]) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    installed = {entry["name"].lower(): entry["version"] for entry in full_pip_list}
    versions.update({name: installed[name] for name in _KEY_PACKAGE_NAMES if name in installed})

    if (torch := sys.modules.get("torch")) is not None:
        versions["torch"] = torch.__version__
        versions["torch_cuda"] = torch.version.cuda or ""

    return versions


def _json_snapshot(values: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return json.loads(json.dumps(values))
    except (TypeError, ValueError):
        return None


def _collect_pip_info() -> tuple[list[EditablePackageInfo], list[dict[str, str]]]:
    """Collect all pip info in a single `pip inspect` call.

    Returns (editable_packages, full_pip_list).
    """
    try:
        # TODO: remove this workaround and still make Megatron detected
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run(
            ["pip", "inspect"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if result.returncode != 0:
            logger.warning("pip inspect failed: %s", result.stderr)
            return [], []

        data = json.loads(result.stdout)
        installed: list[dict[str, Any]] = data.get("installed", [])

        full_pip_list = [_parse_pip_entry(pkg) for pkg in installed]
        editable_packages = [
            EditablePackageInfo(
                name=entry["name"],
                version=entry["version"],
                location=pkg["direct_url"]["url"].removeprefix("file://"),
            )
            for pkg, entry in zip(installed, full_pip_list, strict=True)
            if _is_editable(pkg)
        ]

        return editable_packages, full_pip_list
    except Exception:
        logger.warning("Failed to collect pip info", exc_info=True)
        return [], []


def _parse_pip_entry(pkg: dict[str, Any]) -> dict[str, str]:
    metadata = pkg.get("metadata", {})
    return {"name": metadata.get("name", ""), "version": metadata.get("version", "")}


def _is_editable(pkg: dict[str, Any]) -> bool:
    direct_url = pkg.get("direct_url")
    return bool(direct_url and direct_url.get("dir_info", {}).get("editable"))


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
            ["git", "diff", "--stat", "HEAD"],
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
        logger.warning("Failed to collect git info for %s at %s", package_name, location, exc_info=True)
        return None


ENV_REPORT_PREFIX = "ENV_REPORT_JSON="


def _print_report(report: NodeEnvReport) -> None:
    print(f"{ENV_REPORT_PREFIX}{report.model_dump_json()}")
