from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel

RUNS_DIR_NAME = "miles-runs"
STATE_DIR_NAME = "state"
ORCHESTRATOR_EXIT_FILE_NAME = "orchestrator.exit"
VALUES_FILE_NAME = "values.yaml"

STATUS_STARTED = "started"
STATUS_EXITED = "exited"

DEFAULT_MOUNT_PATH = "/cluster-storage"
DEFAULT_RUNS_SUB_PATH = "miles_data"


def shared_root_of(infra_values: dict[str, Any]) -> str:
    infra = infra_values.get("infra") or {}
    mount_path = (infra.get("sharedStorage") or {}).get("mountPath") or DEFAULT_MOUNT_PATH
    return f"{mount_path.rstrip('/')}/{runs_sub_path_of(infra)}".rstrip("/")


def runs_sub_path_of(infra: dict[str, Any]) -> str:
    if "paths" not in infra:
        return DEFAULT_RUNS_SUB_PATH
    paths = infra["paths"]
    if paths is None:
        return ""
    if "runsSubPath" not in paths:
        return DEFAULT_RUNS_SUB_PATH
    return (paths["runsSubPath"] or "").rstrip("/")


def run_dir(shared_root: str | Path, run_id: str) -> Path:
    return Path(shared_root) / RUNS_DIR_NAME / run_id


def orchestrator_exit_path(run_directory: str | Path) -> Path:
    return Path(run_directory) / STATE_DIR_NAME / ORCHESTRATOR_EXIT_FILE_NAME


def values_path(run_directory: str | Path) -> Path:
    return Path(run_directory) / VALUES_FILE_NAME


class OrchestratorState(FrozenStrictBaseModel):
    status: str
    exit_code: int | None
    timestamp: float
    generation: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status == STATUS_EXITED


def write_orchestrator_state(
    path: str | Path, status: str, exit_code: int | None = None, generation: int | None = None
) -> None:
    assert status in (STATUS_STARTED, STATUS_EXITED), status
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if generation is None:
        generation = max(1, current_generation(path))
    payload = json.dumps(
        {"status": status, "exit_code": exit_code, "timestamp": time.time(), "generation": generation}
    )
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as file:
            file.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def current_generation(path: str | Path) -> int:
    state = read_orchestrator_state(path)
    return 0 if state is None else state.generation


def reset_for_new_generation(path: str | Path, expected_generation: int) -> int:
    observed = current_generation(path)
    if observed > expected_generation:
        return observed
    write_orchestrator_state(path, STATUS_STARTED, generation=expected_generation + 1)
    return expected_generation + 1


def read_orchestrator_state(path: str | Path) -> OrchestratorState | None:
    try:
        payload = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    return OrchestratorState(
        status=payload["status"],
        exit_code=payload.get("exit_code"),
        timestamp=payload.get("timestamp", 0.0),
        generation=payload.get("generation", 0),
    )
