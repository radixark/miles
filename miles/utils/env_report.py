import base64
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import (
    ArgsDump,
    EditablePackageInfo,
    EnvReportEvent,
    GitRepoInfo,
    NodeEnvReport,
    ProcessEnvFacts,
)
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessEnvSnapshot:
    facts: ProcessEnvFacts
    # TODO: remove the PYTHONPATH workaround and still make Megatron detected
    probe_env: dict[str, str]


_SECRET_ENV_VAR_PATTERN = re.compile(
    r"(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|DATABASE_URL)$", re.IGNORECASE
)
_SECRET_ARG_NAMES = frozenset({"wandb_key"})
_SECRET_ARG_FLAGS = frozenset(f"--{name.replace('_', '-')}" for name in _SECRET_ARG_NAMES)
_REPORTER_THREAD_NAME = "env-report"
_INTERVAL_JITTER_RATIO = 0.5
_STOP_TIMEOUT_SECONDS = 5.0
LAUNCHER_REPORT_ENV_VAR = "MILES_SCRIPT_ENV_REPORT"
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


def start_env_reporting(args: Any) -> "EnvReporter":
    reporter = EnvReporter(
        snapshot=collect_process_env_snapshot(args),
        interval_seconds=args.env_report_interval_seconds,
    )
    reporter.start()
    return reporter


class EnvReporter:
    def __init__(self, *, snapshot: ProcessEnvSnapshot, interval_seconds: float) -> None:
        assert math.isfinite(interval_seconds), (
            f"--env-report-interval-seconds is {interval_seconds}, which is neither a delay nor a way to say "
            f"'only at startup'; pass a finite number"
        )

        self._snapshot = snapshot
        self._interval_seconds = interval_seconds
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=_REPORTER_THREAD_NAME, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=_STOP_TIMEOUT_SECONDS)

    def _run(self) -> None:
        while True:
            try:
                log_env_report(snapshot=self._snapshot)
            except Exception:
                logger.warning("Failed to log the env report", exc_info=True)
            if self._interval_seconds <= 0 or self._stopped.wait(self._next_delay_seconds()):
                return

    def _next_delay_seconds(self) -> float:
        return self._interval_seconds * (1.0 + random.random() * _INTERVAL_JITTER_RATIO)


def log_env_report(*, snapshot: ProcessEnvSnapshot) -> NodeEnvReport:
    report = collect_node_env_report(snapshot=snapshot)

    if is_event_logger_initialized():
        get_event_logger().log(EnvReportEvent, {"report": report}, print_log=False)
    _log_report_summary(report)

    return report


def _log_report_summary(report: NodeEnvReport) -> None:
    log_structured(
        logger.info,
        tag="audit",
        op="env_report",
        hostname=report.process.hostname,
        versions=report.key_versions,
        repos={repo.package_name: f"{repo.commit}{'-dirty' if repo.dirty else ''}" for repo in report.git_repos},
        num_packages=len(report.full_pip_list),
        num_env_vars=len(report.process.env_vars),
        stored=is_event_logger_initialized(),
    )


def collect_process_env_snapshot(args: Any) -> ProcessEnvSnapshot:
    environ = dict(os.environ)
    env_vars = {name: value for name, value in environ.items() if name != LAUNCHER_REPORT_ENV_VAR}

    facts = ProcessEnvFacts(
        hostname=socket.gethostname(),
        argv=redact_argv(sys.argv),
        args=dump_args(args),
        env_vars=redact_env_vars(env_vars),
        launcher_env_report=decode_env_report(args.env_report),
    )
    return ProcessEnvSnapshot(facts=facts, probe_env={k: v for k, v in environ.items() if k != "PYTHONPATH"})


def collect_node_env_report(*, snapshot: ProcessEnvSnapshot) -> NodeEnvReport:
    editable_packages, full_pip_list = _collect_pip_info(snapshot.probe_env)

    git_repos = [
        info for pkg in editable_packages if (info := _collect_git_info(package_name=pkg.name, location=pkg.location))
    ]

    return NodeEnvReport(
        process=snapshot.facts,
        key_versions=collect_key_versions(full_pip_list),
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


def _collect_pip_info(env: dict[str, str]) -> tuple[list[EditablePackageInfo], list[dict[str, str]]]:
    """Collect all pip info in a single `pip inspect` call.

    Returns (editable_packages, full_pip_list).
    """
    try:
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
