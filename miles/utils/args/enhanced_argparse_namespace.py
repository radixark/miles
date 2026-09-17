import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from importlib import import_module
from typing import Any, ClassVar

import torch
from pydantic_core import core_schema


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

    def __reduce__(self) -> tuple[Any, tuple[()], dict[str, Any]]:
        return type(self), (), vars(self).copy()

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: {
                    name: _ConfigNamespaceValueCodec.serialize(item) for name, item in sorted(vars(value).items())
                },
                when_used="json",
            ),
        )

    @classmethod
    def _validate(cls, value: Any) -> "EnhancedArgparseNamespace":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**{name: _ConfigNamespaceValueCodec.deserialize(item) for name, item in value.items()})
        raise ValueError("Configuration must be a namespace or mapping")


class _ConfigNamespaceValueCodec:
    @staticmethod
    def serialize(value: Any) -> Any:
        if isinstance(value, torch.dtype):
            return {"__torch_dtype__": str(value).removeprefix("torch.")}
        if isinstance(value, Enum):
            return {"__enum__": f"{type(value).__module__}:{type(value).__name__}:{value.name}"}
        if isinstance(value, tuple):
            return {"__tuple__": [_ConfigNamespaceValueCodec.serialize(item) for item in value]}
        if isinstance(value, list):
            return [_ConfigNamespaceValueCodec.serialize(item) for item in value]
        if isinstance(value, dict):
            return {name: _ConfigNamespaceValueCodec.serialize(item) for name, item in sorted(value.items())}
        return value

    @staticmethod
    def deserialize(value: Any) -> Any:
        if isinstance(value, list):
            return [_ConfigNamespaceValueCodec.deserialize(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"__torch_dtype__"}:
            dtype = vars(torch)[value["__torch_dtype__"]]
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Invalid torch dtype {value!r}")
            return dtype
        if set(value) == {"__enum__"}:
            module_name, class_name, member_name = value["__enum__"].split(":")
            if module_name not in {"megatron.core.transformer.enums", "megatron.core.enums"}:
                raise ValueError(f"Unsupported backend enum module {module_name!r}")
            enum_class = vars(import_module(module_name))[class_name]
            if not isinstance(enum_class, type) or not issubclass(enum_class, Enum):
                raise ValueError(f"Invalid backend enum class {class_name!r}")
            return enum_class[member_name]
        if set(value) == {"__tuple__"}:
            return tuple(_ConfigNamespaceValueCodec.deserialize(item) for item in value["__tuple__"])
        return {name: _ConfigNamespaceValueCodec.deserialize(item) for name, item in value.items()}
