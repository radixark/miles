import json
import logging
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Any
from miles.utils.audit_utils.event_logger.models import (
    EnvReport,
    EnvReportArgsDump,
    EnvReportEditablePackageInfo,
    EnvReportProcessFacts,
)
from miles.utils.env_report.git_state import collect_git_info
from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR, read_launcher_report
from miles.utils.env_report.redaction import redact_arg, redact_argv, redact_env_vars

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvReportSnapshot:
    facts: EnvReportProcessFacts
    # TODO: remove the PYTHONPATH workaround and still make Megatron detected
    probe_env: dict[str, str]


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


def collect_env_report_snapshot(args: Any) -> EnvReportSnapshot:
    environ = dict(os.environ)
    env_vars = {name: value for name, value in environ.items() if name != LAUNCHER_REPORT_ENV_VAR}

    facts = EnvReportProcessFacts(
        hostname=socket.gethostname(),
        argv=redact_argv(sys.argv),
        args=_dump_args(args),
        env_vars=redact_env_vars(env_vars),
        launcher_env_report=read_launcher_report(args.env_report),
    )
    return EnvReportSnapshot(facts=facts, probe_env={k: v for k, v in environ.items() if k != "PYTHONPATH"})


def collect_env_report(*, snapshot: EnvReportSnapshot) -> EnvReport:
    editable_packages, full_pip_list = _collect_pip_info(snapshot.probe_env)

    git_repos = [
        info for pkg in editable_packages if (info := collect_git_info(package_name=pkg.name, location=pkg.location))
    ]

    return EnvReport(
        process=snapshot.facts,
        key_versions=_collect_key_versions(full_pip_list),
        editable_packages=editable_packages,
        git_repos=git_repos,
        full_pip_list=full_pip_list,
        packages_probed=True,
    )


def collect_unprobed_env_report(*, snapshot: EnvReportSnapshot) -> EnvReport:
    return EnvReport(
        process=snapshot.facts,
        key_versions=_collect_key_versions([]),
        editable_packages=[],
        git_repos=[],
        full_pip_list=[],
        packages_probed=False,
    )


def _collect_key_versions(full_pip_list: list[dict[str, str]]) -> dict[str, str]:
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


def _dump_args(args: Any) -> EnvReportArgsDump:
    declared = dict(vars(args))
    if (serializable := _json_snapshot(declared)) is None:
        serializable = {
            name: snapshot[name]
            for name, value in declared.items()
            if (snapshot := _json_snapshot({name: value})) is not None
        }

    values = {name: redact_arg(name, value) for name, value in sorted(serializable.items())}
    return EnvReportArgsDump(values=values, skipped_names=sorted(declared.keys() - serializable.keys()))


def _json_snapshot(values: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return json.loads(json.dumps(values))
    except (TypeError, ValueError):
        return None


def _collect_pip_info(env: dict[str, str]) -> tuple[list[EnvReportEditablePackageInfo], list[dict[str, str]]]:
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
            EnvReportEditablePackageInfo(
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
