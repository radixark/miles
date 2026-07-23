from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass
class ValueSpec:
    TODO


class ObjectStoreGetResult:
    TODO


class BaseObjectStore(ABC):
    def put(self, key: str, value: Any, value_spec: ValueSpec):
        raise NotImplementedError

    def get(self, key: str) -> ObjectStoreGetResult:
        raise NotImplementedError

    def remove(self, key: str):
        raise NotImplementedError


class RayObjectStore(BaseObjectStore):
    TODO


class MooncakeObjectStore(BaseObjectStore):
    TODO
