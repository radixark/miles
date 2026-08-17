from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.external_utils.command_utils.common import (
    MOONCAKE_BACKEND_NAME,
    MOONCAKE_INIT_KWARGS_FLAG,
    ArgvManipulator,
)
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    InfraValues,
    MooncakeSection,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.object_store_config import MOONCAKE_MASTER_ADDRESS_KEY
from miles.utils.pydantic_utils import FrozenStrictBaseModel

STATIC_WORKERS_SECTION = "staticWorkers"
INFERENCE_ENGINES_SECTION = "inferenceEngines"
TRAINER_ENGINES_SECTION = "trainerEngines"

SECTION_OF_CATEGORY = {
    None: STATIC_WORKERS_SECTION,
    POOL_CATEGORY_INFERENCE_ENGINE: INFERENCE_ENGINES_SECTION,
    POOL_CATEGORY_TRAINER_ENGINE: TRAINER_ENGINES_SECTION,
}

_MOONCAKE_COMPONENT = "mooncake-master"
_INFRA_KEY = "infra"
_VALUES_FILE_NAME = "values.yaml"


class MooncakePlan(FrozenStrictBaseModel):
    init_kwargs: dict[str, Any]
    port: int


class LaunchPlan(FrozenStrictBaseModel):
    run_id: str
    release: str
    namespace: str
    state_file: str
    orchestrator_command: list[str]
    worker_argv: list[str]
    env: dict[str, str] = {}
    launch_record: str | None = None
    colocate: bool = False
    mooncake_plan: MooncakePlan | None = None
    prepare_cmd: dict[str, str] = {}
    extra_manifests: list[str] = []


class MooncakeInfo:
    @staticmethod
    def plan_of_args(args: Namespace) -> MooncakePlan | None:
        if args.object_store_backend != MOONCAKE_BACKEND_NAME:
            return None

        init_kwargs = args.mooncake_store_init_kwargs
        assert (
            init_kwargs
        ), f"{MOONCAKE_INIT_KWARGS_FLAG} is missing, so the mooncake master address cannot be rewritten"
        address = init_kwargs.get(MOONCAKE_MASTER_ADDRESS_KEY)
        assert (
            isinstance(address, str) and ":" in address
        ), f"{MOONCAKE_MASTER_ADDRESS_KEY} is {address!r} and carries no port, so the in-cluster address cannot be built"
        return MooncakePlan(init_kwargs=init_kwargs, port=int(address.rsplit(":", 1)[1]))

    @staticmethod
    def with_cluster_master(train_argv: list[str], *, plan: MooncakePlan | None, host: str) -> list[str]:
        if plan is None:
            return train_argv

        kwargs = MooncakeInfo.cluster_init_kwargs(plan, host=host)
        return ArgvManipulator.replacing_value(train_argv, MOONCAKE_INIT_KWARGS_FLAG, json.dumps(kwargs))

    @staticmethod
    def cluster_init_kwargs(plan: MooncakePlan, *, host: str) -> dict[str, Any]:
        return {**plan.init_kwargs, MOONCAKE_MASTER_ADDRESS_KEY: f"{host}:{plan.port}"}

    @staticmethod
    def master_service_host(release: str, namespace: str) -> str:
        return RunNames.service_fqdn(name=MooncakeInfo.master_object_name(release), namespace=namespace)

    @staticmethod
    def master_object_name(release: str) -> str:
        return naming.component_name(release, _MOONCAKE_COMPONENT)

    @staticmethod
    def section(plan: MooncakePlan) -> MooncakeSection:
        return MooncakeSection(enabled=True, rpc_port=plan.port)


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
