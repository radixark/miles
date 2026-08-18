from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import yaml
from pydantic import Field

from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import STATE_FILE_FLAG
from miles.utils.pydantic_utils import FrozenOpenBaseModel, FrozenStrictBaseModel

RESTART_AT_ANNOTATION = "miles.radixark.io/restart-at"
STATEFUL_SET_KIND = "StatefulSet"


class ManifestObjectKey(FrozenStrictBaseModel):
    kind: str
    name: str


class EnvEntry(FrozenOpenBaseModel):
    name: str
    value: str = ""


class Container(FrozenOpenBaseModel):
    name: str
    command: list[str] = []
    env: list[EnvEntry] = []


class PodSpec(FrozenOpenBaseModel):
    containers: list[Container] = []


class PodTemplateMetadata(FrozenOpenBaseModel):
    annotations: dict[str, Any] = {}


class PodTemplate(FrozenOpenBaseModel):
    metadata: PodTemplateMetadata | None = None
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

    @property
    def key(self) -> ManifestObjectKey:
        return ManifestObjectKey(kind=self.kind, name=self.name)


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
    def restart_at(self) -> str | None:
        if self.spec is None or self.spec.template is None or self.spec.template.metadata is None:
            return None
        return self.spec.template.metadata.annotations.get(RESTART_AT_ANNOTATION)

    @property
    def body(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True, by_alias=True)

    @property
    def containers(self) -> list[Container]:
        if self.spec is None or self.spec.template is None:
            return []
        pod = self.spec.template.spec
        return list(pod.containers) if pod is not None else []

    def container_named(self, container: str) -> Container | None:
        found = [described for described in self.containers if described.name == container]
        assert (
            len(found) <= 1
        ), f"{self.kind}/{self.metadata.name} declares {len(found)} containers named {container!r}"
        return found[0] if found else None


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

    def object_keyed(self, *, key: ManifestObjectKey) -> ManifestObject | None:
        matched = [described for identity, described in self.by_identity.items() if identity.key == key]
        assert len(matched) <= 1, (
            f"this release holds {len(matched)} objects keyed {key.kind}/{key.name}, so looking one up would answer "
            f"for whichever rendered first"
        )
        return matched[0] if matched else None

    def restart_at(self, *, object_name: str) -> str | None:
        described = self.object_keyed(key=ManifestObjectKey(kind=STATEFUL_SET_KIND, name=object_name))
        return described.restart_at if described is not None else None

    def flag_value(self, flag: str, *, stateful_set: str, container: str) -> str | None:
        found = self._container(stateful_set=stateful_set, container=container)
        command = list(found.command) if found is not None else []
        if flag not in command:
            return None

        value_index = command.index(flag) + 1
        assert value_index < len(command), (
            f"container {container!r} of {stateful_set} ends its command with {flag}, which takes a value, so this "
            f"launch cannot tell what the installed release was told"
        )
        return command[value_index]

    def state_file(self, *, stateful_set: str, container: str) -> Path | None:
        named = self.flag_value(STATE_FILE_FLAG, stateful_set=stateful_set, container=container)
        return Path(named) if named is not None else None

    def _container(self, *, stateful_set: str, container: str) -> Container | None:
        named = [
            described
            for described in self.objects
            if described.kind == STATEFUL_SET_KIND and described.metadata.name == stateful_set
        ]
        assert len(named) <= 1, (
            f"this release holds {len(named)} StatefulSets named {stateful_set}, so reading a flag off one of them "
            f"would answer for whichever rendered first"
        )
        return named[0].container_named(container) if named else None
