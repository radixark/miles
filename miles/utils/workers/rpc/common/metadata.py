from __future__ import annotations

import dataclasses
import functools
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
    positional_parameter_names: tuple[str, ...]


def canonicalize_method_arguments(
    *, spec: RpcMethodSpec, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    if len(args) > (max_positional := len(spec.positional_parameter_names)):
        raise TypeError(f"{spec.name}() takes at most {max_positional} positional arguments, got {len(args)}")
    named = dict(zip(spec.positional_parameter_names, args, strict=False))
    if overlap := sorted(named.keys() & kwargs.keys()):
        raise TypeError(f"{spec.name}() got multiple values for {overlap}")
    return {**named, **kwargs}


def collect_rpc_method_specs(worker_cls: type) -> dict[str, RpcMethodSpec]:
    return dict(_collect_rpc_method_specs(worker_cls))


def declared_concurrency_groups(worker_cls: type) -> dict[str, str]:
    groups = {}
    for name in sorted(dir(worker_cls)):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(worker_cls, name)
        if not callable(attr):
            continue
        if (group := _find_rpc_config(attr).concurrency_group) != DEFAULT_CONCURRENCY_GROUP:
            groups[name] = group
    return groups


@functools.cache
def _collect_rpc_method_specs(worker_cls: type) -> dict[str, RpcMethodSpec]:
    specs: dict[str, RpcMethodSpec] = {}

    for name in sorted(dir(worker_cls)):
        if name.startswith("_"):
            continue
        static_attr = inspect.getattr_static(worker_cls, name)
        if isinstance(static_attr, (classmethod, staticmethod, property)):
            continue
        if not callable(static_attr):
            continue
        specs[name] = _build_method_spec(worker_cls=worker_cls, name=name, attr=static_attr)

    return specs


@dataclasses.dataclass(frozen=True)
class _RpcConfig:
    concurrency_group: str


def _find_rpc_config(attr: Callable[..., Any]) -> _RpcConfig:
    layer: Any = attr
    while layer is not None:
        config = getattr(layer, _RPC_CONFIG_ATTR, None)
        if config is not None:
            return config
        layer = getattr(layer, "__wrapped__", None)
    return _RpcConfig(concurrency_group=DEFAULT_CONCURRENCY_GROUP)


def _build_method_spec(*, worker_cls: type, name: str, attr: Callable[..., Any]) -> RpcMethodSpec:
    fn = inspect.unwrap(attr)
    if not inspect.isroutine(fn):
        raise TypeError(
            f"{worker_cls.__name__}.{name} is a public callable attribute but not a method, "
            f"so it cannot be exposed over rpc; make it private or move it off the worker class"
        )

    config = _find_rpc_config(attr)
    is_async = inspect.iscoroutinefunction(fn)
    if is_async and config.concurrency_group != DEFAULT_CONCURRENCY_GROUP:
        raise TypeError(
            f"{worker_cls.__name__}.{name} is async; concurrency groups only serialize sync methods, "
            f"so concurrency_group={config.concurrency_group!r} would be silently ignored"
        )

    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)

    parameters = list(signature.parameters.values())
    if len(parameters) == 0:
        raise TypeError(f"{worker_cls.__name__}.{name} must take a receiver parameter for rpc exposure")
    if parameters[0].name != "self":
        raise TypeError(
            f"{worker_cls.__name__}.{name} must name its receiver parameter 'self' for rpc exposure, "
            f"got {parameters[0].name!r}; otherwise it would be silently dropped from the wire"
        )
    if parameters[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        raise TypeError(
            f"{worker_cls.__name__}.{name} must take its receiver parameter positionally for rpc exposure, "
            f"got a {parameters[0].kind.description} parameter"
        )

    query_fields: dict[str, Any] = {}
    for param in parameters[1:]:
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            raise TypeError(
                f"{worker_cls.__name__}.{name} must not use *args/**kwargs or positional-only parameters "
                f"for rpc exposure"
            )
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"{worker_cls.__name__}.{name} parameter '{param.name}' must be type-annotated")
        default = ... if param.default is inspect.Parameter.empty else param.default
        query_fields[param.name] = (hints[param.name], default)

    if signature.return_annotation is inspect.Signature.empty:
        raise TypeError(f"{worker_cls.__name__}.{name} must have a return type annotation")

    return RpcMethodSpec(
        name=name,
        concurrency_group=config.concurrency_group,
        is_async=is_async,
        positional_parameter_names=tuple(
            param.name for param in parameters[1:] if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ),
        serializer=RpcSerializer.create(
            query_model_name=f"{worker_cls.__name__}{name.title().replace('_', '')}Query",
            query_fields=query_fields,
            result_annotation=hints["return"],
        ),
    )
