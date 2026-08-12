from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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


class ManifestObject(FrozenOpenBaseModel):
    kind: str
    metadata: ObjectMetadata
    spec: ObjectSpec | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.metadata.name}"

    @property
    def replicas(self) -> int | None:
        return self.spec.replicas if self.spec is not None else None

    @property
    def body(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)

    def containers_named(self, container: str) -> list[Container]:
        if self.kind != "StatefulSet" or self.spec is None or self.spec.template is None:
            return []
        pod = self.spec.template.spec
        return [described for described in (pod.containers if pod is not None else []) if described.name == container]


class Manifest(FrozenStrictBaseModel):
    objects: list[ManifestObject]

    @classmethod
    def parse(cls, rendered: str) -> Manifest:
        return cls(objects=[document for document in yaml.safe_load_all(rendered) if document])

    @property
    def by_key(self) -> dict[str, ManifestObject]:
        return {described.key: described for described in self.objects}

    def state_file(self, *, container: str) -> Path | None:
        for described in self.objects:
            for found in described.containers_named(container):
                if STATE_FILE_FLAG in found.command:
                    return Path(found.command[found.command.index(STATE_FILE_FLAG) + 1])
        return None
