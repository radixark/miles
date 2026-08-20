from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR
from miles.utils.external_utils.colocate_pairing.config import PairingConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import RUN_ID_MAX_LENGTH
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.env_vars import (
    BASE_GPU_ID_ENV_VAR,
    CELL_INDEX_ENV_VAR,
    NAMESPACE_ENV_VAR,
    POD_INDEX_ENV_VAR,
    RELEASE_ENV_VAR,
)
from miles.utils.workers.naming import POOL_NAME_MAX_LENGTH

_DNS_LABEL = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
_OPTIONAL_DNS_LABEL = r"^([a-z0-9]([-a-z0-9]*[a-z0-9])?)?$"
_DNS_SUBDOMAIN = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
_OPTIONAL_DNS_SUBDOMAIN = r"^([a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*)?$"

_NO_PARENT_TRAVERSAL = {"not": {"pattern": r"(^|/)\.\.(/|$)"}}
_PLATFORM_OWNED_ENV_VARS = [
    CELL_INDEX_ENV_VAR,
    POD_INDEX_ENV_VAR,
    BASE_GPU_ID_ENV_VAR,
    NAMESPACE_ENV_VAR,
    RELEASE_ENV_VAR,
]
_ENV_KEYS = {
    "propertyNames": {
        "pattern": "^[ -<>-~]+$",
        "not": {"enum": ["PYTHONPATH", LAUNCHER_REPORT_ENV_VAR, *_PLATFORM_OWNED_ENV_VARS]},
    }
}

_OBJECT_NAME_MAX = 63
_PORT_NAME_MAX = 15
_KUBERNETES_NAME_MAX = 253
WORKBENCH_OBJECT_NAME_MAX = 52

_PoolName = Annotated[str, Field(min_length=1, max_length=POOL_NAME_MAX_LENGTH, pattern=_DNS_LABEL)]
_ObjectName = Annotated[str, Field(min_length=1, max_length=_OBJECT_NAME_MAX, pattern=_DNS_LABEL)]
_Port = Annotated[int, Field(ge=1, le=65535)]
_AbsolutePath = Annotated[str, Field(pattern="^/", json_schema_extra=_NO_PARENT_TRAVERSAL)]
_OptionalAbsolutePath = Annotated[str, Field(pattern="^(/.*)?$", json_schema_extra=_NO_PARENT_TRAVERSAL)]
_RelativePath = Annotated[str, Field(pattern="^([^/].*)?$", json_schema_extra=_NO_PARENT_TRAVERSAL)]
_EnvVars = Annotated[dict[str, str], Field(json_schema_extra=_ENV_KEYS)]

_Resources = dict[str, Any]


class ValuesModel(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, alias_generator=to_camel, populate_by_name=True)

    def as_values(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class PortEntry(ValuesModel):
    name: Annotated[str, Field(min_length=1, max_length=_PORT_NAME_MAX)]
    port: _Port


class PoolEntry(ValuesModel):
    name: _PoolName
    object_name: _ObjectName
    pool_id: str | None = None
    command: Annotated[list[str], Field(min_length=1)]
    ports: list[PortEntry] | None = None
    env: _EnvVars | None = None
    meta: dict[str, str] | None = None
    replicas: Annotated[int, Field(ge=1)] | None = None
    size: Annotated[int, Field(ge=1)] | None = None
    resources: _Resources | None = None
    restart_at: Annotated[str, Field(min_length=1)] | None = None
    service_account_name: _ObjectName | None = None


class ObjectNames(ValuesModel):
    orchestrator: _ObjectName
    mooncake_master: _ObjectName
    colocate_pairing: _ObjectName
    uninstall: _ObjectName
    uninstall_manifest: _ObjectName


class AutoUninstallSection(ValuesModel):
    enabled: bool
    service_account: _ObjectName | None = None


class OrchestratorSection(ValuesModel):
    command: list[str] | None = None
    resources: _Resources | None = None
    restart_at: Annotated[str, Field(min_length=1)] | None = None


class MooncakeSection(ValuesModel):
    enabled: bool | None = None
    rpc_port: _Port | None = None
    metrics_port: _Port | None = None
    resources: _Resources | None = None


class RunValues(ValuesModel):
    id: Annotated[str, Field(max_length=RUN_ID_MAX_LENGTH, pattern=_DNS_LABEL)]
    state_file: Annotated[str, Field(min_length=1, pattern="^/")] | None = None
    launch_record: Annotated[str, Field(min_length=1, pattern="^/")] | None = None
    object_names: ObjectNames
    orchestrator: OrchestratorSection | None = None
    static_workers: list[PoolEntry] | None = None
    inference_engines: list[PoolEntry] | None = None
    trainer_engines: list[PoolEntry] | None = None
    env: _EnvVars | None = None
    mooncake: MooncakeSection | None = None
    colocate: PairingConfig | None = None
    auto_uninstall: AutoUninstallSection | None = None


class CommandJobValues(ValuesModel):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"enabled": {"const": True}}, "required": ["enabled"]},
                    "then": {
                        "required": ["name", "objectName", "command"],
                        "properties": {
                            "name": {"minLength": 1},
                            "objectName": {"minLength": 1},
                            "command": {"minItems": 1},
                        },
                    },
                }
            ]
        }
    )

    enabled: bool | None = None
    name: Annotated[str, Field(max_length=POOL_NAME_MAX_LENGTH, pattern=_OPTIONAL_DNS_LABEL)] | None = None
    object_name: Annotated[str, Field(max_length=_OBJECT_NAME_MAX, pattern=_OPTIONAL_DNS_LABEL)] | None = None
    command: list[str] | None = None
    completions: Annotated[int, Field(ge=1)] | None = None
    gpus_per_pod: Annotated[int, Field(ge=0)] | None = None
    active_deadline_seconds: Annotated[int, Field(ge=1)] | None = None
    ttl_seconds_after_finished: Annotated[int, Field(ge=0)] | None = None


class Image(ValuesModel):
    repository: Annotated[str, Field(min_length=1)]
    tag: Annotated[str, Field(min_length=1)]
    pull_policy: Literal["Always", "IfNotPresent", "Never"] | None = None
    pull_secrets: (
        list[Annotated[str, Field(min_length=1, max_length=_KUBERNETES_NAME_MAX, pattern=_DNS_SUBDOMAIN)]] | None
    ) = None


class SharedStorage(ValuesModel):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"type": {"const": "hostPath"}}, "required": ["type"]},
                    "then": {"properties": {"hostPath": {"minLength": 1}}, "required": ["hostPath"]},
                },
                {
                    "if": {"properties": {"type": {"const": "pvc"}}, "required": ["type"]},
                    "then": {"properties": {"pvcClaimName": {"minLength": 1}}, "required": ["pvcClaimName"]},
                },
            ]
        }
    )

    type: Literal["hostPath", "pvc", "none"]
    mount_path: _AbsolutePath
    host_path: _OptionalAbsolutePath | None = None
    pvc_claim_name: Annotated[str, Field(max_length=_KUBERNETES_NAME_MAX, pattern=_OPTIONAL_DNS_SUBDOMAIN)] | None = (
        None
    )


class Repos(ValuesModel):
    miles: _RelativePath | None = None
    megatron: _RelativePath | None = None
    sglang: _RelativePath | None = None


class Paths(ValuesModel):
    runs_sub_path: _RelativePath | None = None
    repos: Repos | None = None


class NodeLocalStorage(ValuesModel):
    host_path: _OptionalAbsolutePath | None = None
    mount_path: _AbsolutePath | None = None


class Scheduling(ValuesModel):
    node_selector: dict[str, str] | None = None
    tolerations: list[dict[str, Any]] | None = None
    affinity: dict[str, Any] | None = None


class InfraValues(ValuesModel):
    image: Image
    shared_storage: SharedStorage
    paths: Paths | None = None
    node_local_storage: NodeLocalStorage | None = None
    scheduling: Scheduling | None = None
    env: _EnvVars | None = None


class RbacSection(ValuesModel):
    create: bool | None = None
    leader_worker_sets: bool | None = None


class ServiceAccountSection(ValuesModel):
    name: Annotated[str, Field(max_length=_KUBERNETES_NAME_MAX, pattern=_OPTIONAL_DNS_SUBDOMAIN)] | None = None


class Uninstaller(ValuesModel):
    service_account: _ObjectName | None = None


class WorkbenchResources(ValuesModel):
    requests: dict[str, str | float] | None = None
    limits: dict[str, str | float] | None = None


class MilesRunChartValues(ValuesModel):
    infra: InfraValues | None = None
    run: RunValues | None = None
    extra_manifests: list[str] | None = None
    command_job: CommandJobValues | None = None


class MilesWorkbenchChartValues(ValuesModel):
    infra: InfraValues
    object_name: Annotated[str, Field(max_length=WORKBENCH_OBJECT_NAME_MAX, pattern=_OPTIONAL_DNS_LABEL)] | None = None
    resources: WorkbenchResources | None = None
    rbac: RbacSection | None = None
    service_account: ServiceAccountSection | None = None
    uninstaller: Uninstaller | None = None
