# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import AliasChoices, BeforeValidator, Field

from miles.utils.pydantic_utils import FrozenPartialBaseModel


def _absent_as_empty_map(value: Any) -> Any:
    return {} if value is None else value


def _absent_as_empty_list(value: Any) -> Any:
    return [] if value is None else value


def _absent_as_one(value: Any) -> Any:
    return 1 if value is None else value


StringMap = Annotated[dict[str, str], BeforeValidator(_absent_as_empty_map)]


class PodMetadata(FrozenPartialBaseModel):
    name: str
    uid: str
    resource_version: str | None = Field(
        default=None, validation_alias=AliasChoices("resource_version", "resourceVersion")
    )
    deletion_timestamp: datetime | None = Field(
        default=None, validation_alias=AliasChoices("deletion_timestamp", "deletionTimestamp")
    )
    labels: StringMap = {}
    annotations: StringMap = {}


class PodSchedulingGate(FrozenPartialBaseModel):
    name: str


class PodSpec(FrozenPartialBaseModel):
    node_name: str | None = Field(default=None, validation_alias=AliasChoices("node_name", "nodeName"))
    subdomain: str | None = None
    node_selector: StringMap = Field(default={}, validation_alias=AliasChoices("node_selector", "nodeSelector"))
    scheduling_gates: Annotated[list[PodSchedulingGate], BeforeValidator(_absent_as_empty_list)] = Field(
        default=[], validation_alias=AliasChoices("scheduling_gates", "schedulingGates")
    )


class PodCondition(FrozenPartialBaseModel):
    type: str
    status: str
    reason: str | None = None


class ContainerStateTerminated(FrozenPartialBaseModel):
    container_id: str | None = Field(default=None, validation_alias=AliasChoices("container_id", "containerID"))


class ContainerStateRunning(FrozenPartialBaseModel):
    started_at: datetime | None = Field(default=None, validation_alias=AliasChoices("started_at", "startedAt"))


class ContainerState(FrozenPartialBaseModel):
    running: ContainerStateRunning | None = None
    terminated: ContainerStateTerminated | None = None


class ContainerStatus(FrozenPartialBaseModel):
    name: str = ""
    container_id: str | None = Field(default=None, validation_alias=AliasChoices("container_id", "containerID"))
    restart_count: int = Field(validation_alias=AliasChoices("restart_count", "restartCount"))
    state: Annotated[ContainerState, BeforeValidator(_absent_as_empty_map)] = ContainerState()
    last_state: Annotated[ContainerState, BeforeValidator(_absent_as_empty_map)] = Field(
        default=ContainerState(), validation_alias=AliasChoices("last_state", "lastState")
    )


class PodStatus(FrozenPartialBaseModel):
    phase: str | None = None
    pod_ip: str | None = Field(default=None, validation_alias=AliasChoices("pod_ip", "podIP"))
    conditions: Annotated[list[PodCondition], BeforeValidator(_absent_as_empty_list)] = []
    container_statuses: Annotated[list[ContainerStatus], BeforeValidator(_absent_as_empty_list)] = Field(
        default=[], validation_alias=AliasChoices("container_statuses", "containerStatuses")
    )


class Pod(FrozenPartialBaseModel):
    metadata: PodMetadata
    spec: Annotated[PodSpec, BeforeValidator(_absent_as_empty_map)] = PodSpec()
    status: Annotated[PodStatus, BeforeValidator(_absent_as_empty_map)] = PodStatus()


class PodList(FrozenPartialBaseModel):
    items: Annotated[list[Pod], BeforeValidator(_absent_as_empty_list)] = []


class JobCondition(FrozenPartialBaseModel):
    type: str
    status: str


class JobStatus(FrozenPartialBaseModel):
    conditions: Annotated[list[JobCondition], BeforeValidator(_absent_as_empty_list)] = []


class Job(FrozenPartialBaseModel):
    status: Annotated[JobStatus, BeforeValidator(_absent_as_empty_map)] = JobStatus()


class ObjectReference(FrozenPartialBaseModel):
    name: str | None = None
    kind: str | None = None


class Event(FrozenPartialBaseModel):
    involved_object: Annotated[ObjectReference, BeforeValidator(_absent_as_empty_map)] = Field(
        default=ObjectReference(), validation_alias=AliasChoices("involved_object", "involvedObject")
    )
    reason: str | None = None
    message: str | None = None
    count: Annotated[int, BeforeValidator(_absent_as_one)] = 1
    type: str | None = None


class EventList(FrozenPartialBaseModel):
    items: Annotated[list[Event], BeforeValidator(_absent_as_empty_list)] = []


class WatchFrameMetadata(FrozenPartialBaseModel):
    resource_version: str | None = Field(
        default=None, validation_alias=AliasChoices("resource_version", "resourceVersion")
    )


class WatchFrame(FrozenPartialBaseModel):
    metadata: Annotated[WatchFrameMetadata, BeforeValidator(_absent_as_empty_map)] = WatchFrameMetadata()
    code: int | None = None
    reason: str | None = None
