from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable
from typing import Any

from miles.utils.workers.rpc.common.serialization import RpcSerializer


@dataclasses.dataclass(frozen=True)
class RpcMethodSpec:
    name: str
    serializer: RpcSerializer


def collect_rpc_method_specs(worker_cls: type) -> dict[str, RpcMethodSpec]:
    specs: dict[str, RpcMethodSpec] = {}

    for name in sorted(dir(worker_cls)):
        if name.startswith("_"):
            continue
        static_attr = inspect.getattr_static(worker_cls, name)
        if isinstance(static_attr, (classmethod, staticmethod, property)):
            continue
        if not callable(static_attr):
            continue
        specs[name] = _build_method_spec(worker_cls=worker_cls, name=name, fn=inspect.unwrap(static_attr))

    return specs


def _build_method_spec(*, worker_cls: type, name: str, fn: Callable[..., Any]) -> RpcMethodSpec:
    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)

    query_fields: dict[str, Any] = {}
    for param in list(signature.parameters.values())[1:]:
        default = ... if param.default is inspect.Parameter.empty else param.default
        query_fields[param.name] = (hints[param.name], default)

    return RpcMethodSpec(
        name=name,
        serializer=RpcSerializer.create(
            query_model_name=f"{worker_cls.__name__}{name.title().replace('_', '')}Query",
            query_fields=query_fields,
            result_annotation=hints["return"],
        ),
    )
