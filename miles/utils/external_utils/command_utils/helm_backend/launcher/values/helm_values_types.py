from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel, to_snake

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

_VOLUME_SOURCES = ("hostPath", "persistentVolumeClaim", "emptyDir")
_DEV_SHM_SOURCES = ("hostPath", "emptyDir")
_QUANTITY = r"^[0-9]+(\.[0-9]+)?([EPTGMk]i?|m)?$"
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
    platform_reader: _ObjectName
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


def _assert_one_source(volume: ValuesModel, sources: tuple[str, ...]) -> None:
    declared = [source for source in sources if getattr(volume, to_snake(source)) is not None]
    assert len(declared) == 1, (
        f"a volume declares exactly one of {list(sources)}, but this one declares {declared}: a volume with none "
        f"is a mount kubernetes cannot satisfy, and one with several is a values file whose reader has to guess"
    )


class HostPathSource(ValuesModel):
    path: _AbsolutePath
    type: Literal["Directory", "DirectoryOrCreate"] | None = None


class PersistentVolumeClaimSource(ValuesModel):
    claim_name: Annotated[str, Field(min_length=1, max_length=_KUBERNETES_NAME_MAX, pattern=_DNS_SUBDOMAIN)]


class EmptyDirSource(ValuesModel):
    medium: Literal["", "Memory"] | None = None
    size_limit: Annotated[str, Field(pattern=_QUANTITY)] | None = None


class VolumeMount(ValuesModel):
    mount_path: _AbsolutePath
    sub_path: _RelativePath | None = None
    read_only: bool | None = None


class VolumeEntry(ValuesModel):
    model_config = ConfigDict(json_schema_extra={"oneOf": [{"required": [key]} for key in _VOLUME_SOURCES]})

    name: _ObjectName
    host_path: HostPathSource | None = None
    persistent_volume_claim: PersistentVolumeClaimSource | None = None
    empty_dir: EmptyDirSource | None = None
    mounts: Annotated[list[VolumeMount], Field(min_length=1)]

    @model_validator(mode="after")
    def _declares_exactly_one_source(self) -> VolumeEntry:
        _assert_one_source(self, _VOLUME_SOURCES)
        return self


class DevShm(ValuesModel):
    model_config = ConfigDict(json_schema_extra={"oneOf": [{"required": [key]} for key in _DEV_SHM_SOURCES]})

    mount_path: _AbsolutePath
    host_path: HostPathSource | None = None
    empty_dir: EmptyDirSource | None = None

    @model_validator(mode="after")
    def _declares_exactly_one_source(self) -> DevShm:
        _assert_one_source(self, _DEV_SHM_SOURCES)
        return self


class Paths(ValuesModel):
    runs_root: _AbsolutePath | None = None


class Scheduling(ValuesModel):
    node_selector: dict[str, str] | None = None
    tolerations: list[dict[str, Any]] | None = None
    affinity: dict[str, Any] | None = None


class InfraValues(ValuesModel):
    image: Image
    volumes: list[VolumeEntry]
    paths: Paths | None = None
    dev_shm: DevShm | None = None
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
