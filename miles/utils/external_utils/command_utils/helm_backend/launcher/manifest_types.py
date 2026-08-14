from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import yaml
from pydantic import Field

from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import STATE_FILE_FLAG
from miles.utils.pydantic_utils import FrozenOpenBaseModel, FrozenStrictBaseModel


class EnvEntry(FrozenOpenBaseModel):
    name: str
    value: str = ""


class Container(FrozenOpenBaseModel):
    name: str
    command: list[str] = []
    env: list[EnvEntry] = []


class PodSpec(FrozenOpenBaseModel):
    containers: list[Container] = []


class PodTemplate(FrozenOpenBaseModel):
    spec: PodSpec | None = None


class ObjectSpec(FrozenOpenBaseModel):
    replicas: int | None = None
    template: PodTemplate | None = None


class ObjectMetadata(FrozenOpenBaseModel):
    name: str
    namespace: str | None = None


class ObjectIdentity(NamedTuple):
    api_version: str
    kind: str
    namespace: str
    name: str

    def __str__(self) -> str:
        return "/".join(self)


class ManifestObject(FrozenOpenBaseModel):
    api_version: str = Field(default="", alias="apiVersion")
    kind: str
    metadata: ObjectMetadata
    spec: ObjectSpec | None = None

    def identity(self, *, default_namespace: str) -> ObjectIdentity:
        return ObjectIdentity(
            api_version=self.api_version,
            kind=self.kind,
            namespace=self.metadata.namespace or default_namespace,
            name=self.metadata.name,
        )

    @property
    def replicas(self) -> int | None:
        return self.spec.replicas if self.spec is not None else None

    @property
    def body(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True, by_alias=True)

    def containers_named(self, container: str) -> list[Container]:
        if self.kind != "StatefulSet" or self.spec is None or self.spec.template is None:
            return []
        pod = self.spec.template.spec
        return [described for described in (pod.containers if pod is not None else []) if described.name == container]


class Manifest(FrozenStrictBaseModel):
    namespace: str
    objects: list[ManifestObject]

    @classmethod
    def parse(cls, rendered: str, *, namespace: str) -> Manifest:
        return cls(namespace=namespace, objects=[document for document in yaml.safe_load_all(rendered) if document])

    @property
    def by_identity(self) -> dict[ObjectIdentity, ManifestObject]:
        identified: dict[ObjectIdentity, ManifestObject] = {}
        for described in self.objects:
            identity = described.identity(default_namespace=self.namespace)
            assert identity not in identified, (
                f"two objects of this release are both {identity}, so one of them would hide the other "
                f"from the check that decides whether a relaunch may restart pods"
            )
            identified[identity] = described
        return identified

    def state_file(self, *, container: str) -> Path | None:
        for described in self.objects:
            for found in described.containers_named(container):
                if STATE_FILE_FLAG in found.command:
                    return Path(found.command[found.command.index(STATE_FILE_FLAG) + 1])
        return None
