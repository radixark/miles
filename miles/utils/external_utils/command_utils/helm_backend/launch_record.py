from __future__ import annotations

from pathlib import Path

from miles.utils.env_report import LAUNCHER_REPORT_ENV_VAR as LAUNCHER_REPORT_ENV_VAR
from miles.utils.external_utils.command_utils.helm_backend import run_state
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class LaunchRecord(FrozenStrictBaseModel):
    run_id: str
    release: str
    namespace: str
    train_argv: list[str]
    worker_argv: list[str]
    orchestrator_command: list[str]
    env: dict[str, str]


def env_with_launch_record(env: dict[str, str], *, record: LaunchRecord) -> dict[str, str]:
    return {**env, LAUNCHER_REPORT_ENV_VAR: record.model_dump_json()}


def write_launch_record(run_directory: Path, *, record: LaunchRecord, generation: int) -> Path:
    path = run_state.launch_record_path(run_directory, generation=generation)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2))
    return path
