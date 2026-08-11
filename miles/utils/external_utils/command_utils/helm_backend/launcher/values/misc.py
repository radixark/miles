from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import InfraValues
from miles.utils.pydantic_utils import FrozenStrictBaseModel

STATIC_WORKERS_SECTION = "staticWorkers"
INFERENCE_ENGINES_SECTION = "inferenceEngines"
TRAINER_ENGINES_SECTION = "trainerEngines"

SECTION_OF_CATEGORY = {
    None: STATIC_WORKERS_SECTION,
    POOL_CATEGORY_INFERENCE_ENGINE: INFERENCE_ENGINES_SECTION,
    POOL_CATEGORY_TRAINER_ENGINE: TRAINER_ENGINES_SECTION,
}

_INFRA_KEY = "infra"
_VALUES_FILE_NAME = "values.yaml"


class LaunchPlan(FrozenStrictBaseModel):
    run_id: str
    release: str
    namespace: str
    state_file: str
    orchestrator_command: list[str]
    worker_argv: list[str]
    env: dict[str, str] = {}
    prepare_cmd: dict[str, str] = {}


class InfraInfo:
    @staticmethod
    def load(chart: str | Path, helm_values_files: list[str]) -> InfraValues:
        return InfraValues.model_validate(_load_helm_values(chart, helm_values_files).get(_INFRA_KEY))

    @staticmethod
    def shared_root(infra: InfraValues) -> str:
        mount_path = infra.shared_storage.mount_path.rstrip("/")
        runs_sub_path = (infra.paths.runs_sub_path if infra.paths is not None else None) or ""
        return f"{mount_path}/{runs_sub_path.rstrip('/')}".rstrip("/")


def _load_helm_values(chart: str | Path, values_files: list[str] | list[Path]) -> Any:
    def load(values_file: Path) -> Any:
        return yaml.safe_load(values_file.read_text()) or {}

    def merge(base: Any, override: Any) -> Any:
        if not isinstance(override, dict) or not isinstance(base, dict):
            return base if override is None else override

        result = dict(base)
        for key, value in override.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = merge(result.get(key), value)
        return result

    values = load(Path(chart) / _VALUES_FILE_NAME)
    for values_file in values_files:
        values = merge(values, load(Path(values_file)))
    return values
