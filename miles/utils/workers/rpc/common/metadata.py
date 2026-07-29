from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable
from typing import Any, TypeVar

from miles.utils.workers.rpc.common.serialization import RpcSerializer

DEFAULT_CONCURRENCY_GROUP = "default"

_RPC_CONFIG_ATTR = "_miles_rpc_config"

_F = TypeVar("_F", bound=Callable[..., Any])


def rpc(*, concurrency_group: str = DEFAULT_CONCURRENCY_GROUP) -> Callable[[_F], _F]:
    config = _RpcConfig(concurrency_group=concurrency_group)

    def decorator(fn: _F) -> _F:
        setattr(fn, _RPC_CONFIG_ATTR, config)
        return fn

    return decorator


@dataclasses.dataclass(frozen=True)
class RpcMethodSpec:
    name: str
    concurrency_group: str
    is_async: bool
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


@dataclasses.dataclass(frozen=True)
class _RpcConfig:
    concurrency_group: str


def _build_method_spec(*, worker_cls: type, name: str, fn: Callable[..., Any]) -> RpcMethodSpec:
    config: _RpcConfig = getattr(fn, _RPC_CONFIG_ATTR, _RpcConfig(concurrency_group=DEFAULT_CONCURRENCY_GROUP))
    is_async = inspect.iscoroutinefunction(inspect.unwrap(fn))
    if is_async and config.concurrency_group != DEFAULT_CONCURRENCY_GROUP:
        raise TypeError(
            f"{worker_cls.__name__}.{name} is async; concurrency groups only serialize sync methods, "
            f"so concurrency_group={config.concurrency_group!r} would be silently ignored"
        )

    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)

    query_fields: dict[str, Any] = {}
    for param in list(signature.parameters.values())[1:]:
        default = ... if param.default is inspect.Parameter.empty else param.default
        query_fields[param.name] = (hints[param.name], default)

    return RpcMethodSpec(
        name=name,
        concurrency_group=config.concurrency_group,
        is_async=is_async,
        serializer=RpcSerializer.create(
            query_model_name=f"{worker_cls.__name__}{name.title().replace('_', '')}Query",
            query_fields=query_fields,
            result_annotation=hints["return"],
        ),
    )
