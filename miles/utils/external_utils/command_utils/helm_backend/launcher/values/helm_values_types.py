from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from miles.utils.pydantic_utils import FrozenStrictBaseModel

_DNS_SUBDOMAIN = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
_DNS_LABEL = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
_OPTIONAL_DNS_LABEL = r"^([a-z0-9]([-a-z0-9]*[a-z0-9])?)?$"
_OPTIONAL_DNS_SUBDOMAIN = r"^([a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*)?$"

_NO_PARENT_TRAVERSAL = {"not": {"pattern": r"(^|/)\.\.(/|$)"}}
_ENV_KEYS = {"propertyNames": {"pattern": "^[ -<>-~]+$", "not": {"const": "PYTHONPATH"}}}

_OBJECT_NAME_MAX = 63
_KUBERNETES_NAME_MAX = 253
WORKBENCH_OBJECT_NAME_MAX = 52

_ObjectName = Annotated[str, Field(min_length=1, max_length=_OBJECT_NAME_MAX, pattern=_DNS_LABEL)]
_AbsolutePath = Annotated[str, Field(pattern="^/", json_schema_extra=_NO_PARENT_TRAVERSAL)]
_OptionalAbsolutePath = Annotated[str, Field(pattern="^(/.*)?$", json_schema_extra=_NO_PARENT_TRAVERSAL)]
_RelativePath = Annotated[str, Field(pattern="^([^/].*)?$", json_schema_extra=_NO_PARENT_TRAVERSAL)]
_EnvVars = Annotated[dict[str, str], Field(json_schema_extra=_ENV_KEYS)]


class ValuesModel(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, alias_generator=to_camel, populate_by_name=True)

    def as_values(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


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


class Scheduling(ValuesModel):
    node_selector: dict[str, str] | None = None
    tolerations: list[dict[str, Any]] | None = None
    affinity: dict[str, Any] | None = None


class InfraValues(ValuesModel):
    image: Image
    shared_storage: SharedStorage
    paths: Paths | None = None
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


class MilesWorkbenchChartValues(ValuesModel):
    infra: InfraValues
    object_name: Annotated[str, Field(max_length=WORKBENCH_OBJECT_NAME_MAX, pattern=_OPTIONAL_DNS_LABEL)] | None = None
    resources: WorkbenchResources | None = None
    rbac: RbacSection | None = None
    service_account: ServiceAccountSection | None = None
    uninstaller: Uninstaller | None = None
