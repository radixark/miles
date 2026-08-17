from __future__ import annotations

from pathlib import Path

from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR
from miles.utils.env_report.redaction import redact_argv, redact_env_vars
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class LaunchRecord(FrozenStrictBaseModel):
    run_id: str
    release: str
    namespace: str
    state_file: str
    values_file: str
    worker_argv: list[str]
    orchestrator_command: list[str]
    env: dict[str, str]
    reachable_at: dict[str, str]

    @classmethod
    def compute(cls, *, plan: LaunchPlan, values_file: Path, reachable_at: dict[str, str]) -> LaunchRecord:
        return cls(
            run_id=plan.run_id,
            release=plan.release,
            namespace=plan.namespace,
            state_file=plan.state_file,
            values_file=str(values_file),
            worker_argv=redact_argv(plan.worker_argv),
            orchestrator_command=redact_argv(plan.orchestrator_command),
            env=redact_env_vars(plan.env),
            reachable_at=reachable_at,
        )

    def write(self, *, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, self.model_dump_json(indent=2))


def installed_launch_record_file(*, manifest: Manifest) -> str | None:
    for described in manifest.objects:
        for found in described.containers:
            for entry in found.env:
                if entry.name == LAUNCHER_REPORT_ENV_VAR:
                    return entry.value
    return None
