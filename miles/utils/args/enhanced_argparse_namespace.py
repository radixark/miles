import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar


class EnhancedArgparseNamespace(argparse.Namespace):
    __slots__ = ("_mutable_depth",)
    _mutable_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, **values: Any) -> None:
        vars(self).update(values)
        object.__setattr__(self, "_mutable_depth", 0)

    def __setattr__(self, name: str, value: Any) -> None:
        self._check_can_mutate(name)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        self._check_can_mutate(name)
        super().__delattr__(name)

    def _check_can_mutate(self, field_name: str) -> None:
        if not (self._mutable_depth > 0 and field_name in self._mutable_fields):
            raise AttributeError(f"Configuration field {field_name!r} is immutable")

    @contextmanager
    def mutable(self) -> Iterator["EnhancedArgparseNamespace"]:
        object.__setattr__(self, "_mutable_depth", self._mutable_depth + 1)
        try:
            yield self
        finally:
            object.__setattr__(self, "_mutable_depth", self._mutable_depth - 1)
