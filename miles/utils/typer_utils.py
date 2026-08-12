import dataclasses
import functools
import inspect
import typing
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any, TypeVar, overload

import typer

_F = TypeVar("_F", bound=Callable[..., object])

SCRIPT_ENV_VAR_PREFIX = "MILES_SCRIPT_"


@overload
def dataclass_cli(func: _F) -> _F: ...


@overload
def dataclass_cli(
    func: None = None,
    *,
    env_var_prefix: str = SCRIPT_ENV_VAR_PREFIX,
) -> Callable[[_F], _F]: ...


def dataclass_cli(
    func: _F | None = None,
    *,
    env_var_prefix: str = SCRIPT_ENV_VAR_PREFIX,
) -> _F | Callable[[_F], _F]:
    """Turn a function whose first param is a dataclass into a typer-compatible CLI.

    Modified from https://github.com/fastapi/typer/issues/154#issuecomment-1544876144

    Supports field ``metadata`` keys:
    - ``"help"``: passed as ``help=`` to ``typer.Option``

    Usage::

        @app.command()
        @dataclass_cli                              # bare — uses MILES_SCRIPT_ env prefix
        def cmd(args: MyArgs): ...

        @app.command()
        @dataclass_cli(env_var_prefix="")            # no env-var binding
        def cmd(args: MyArgs): ...
    """
    if func is None:
        return functools.partial(dataclass_cli, env_var_prefix=env_var_prefix)  # type: ignore[return-value]

    return _wrap(func, env_var_prefix=env_var_prefix)


def _resolve_default(field: dataclasses.Field, param: inspect.Parameter) -> object:
    """Call a default_factory now; click would otherwise type-cast dataclasses' sentinel."""
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return param.default


def dataclass_from_env(dataclass_cls: type, *, env_var_prefix: str = SCRIPT_ENV_VAR_PREFIX) -> Any:
    """Build the dataclass from the environment alone, letting click read it exactly as it would on the cli."""
    built: list[Any] = []

    def build(**kwargs: object) -> None:
        built.append(_build(dataclass_cls, kwargs))

    build.__signature__ = inspect.Signature(_cli_parameters(dataclass_cls, env_var_prefix=env_var_prefix))
    build.__name__ = dataclass_cls.__name__

    app = typer.Typer(add_completion=False)
    app.command()(build)
    typer.main.get_command(app)(args=[], standalone_mode=False)

    return built[0]


def _wrap(func: _F, *, env_var_prefix: str) -> _F:
    hints: dict[str, type] = typing.get_type_hints(func)
    first_param_name: str = next(iter(inspect.signature(func).parameters))
    dataclass_cls: type = hints[first_param_name]

    def wrapped(**kwargs: object) -> object:
        data: object = _build(dataclass_cls, kwargs)
        _print_arguments(data)
        return func(data)

    wrapped.__signature__ = inspect.Signature(_cli_parameters(dataclass_cls, env_var_prefix=env_var_prefix))
    wrapped.__doc__ = func.__doc__
    wrapped.__name__ = func.__name__  # type: ignore[attr-defined]
    wrapped.__qualname__ = func.__qualname__  # type: ignore[attr-defined]

    return wrapped  # type: ignore[return-value]


def _cli_parameters(dataclass_cls: type, *, env_var_prefix: str) -> list[inspect.Parameter]:
    assert dataclasses.is_dataclass(dataclass_cls)

    old_parameters: list[inspect.Parameter] = list(inspect.signature(dataclass_cls.__init__).parameters.values())
    if old_parameters and old_parameters[0].name == "self":
        del old_parameters[0]

    resolved_hints: dict[str, type] = typing.get_type_hints(dataclass_cls)
    fields_by_name: dict[str, dataclasses.Field] = {  # type: ignore[type-arg]
        f.name: f for f in dataclasses.fields(dataclass_cls)
    }

    new_parameters: list[inspect.Parameter] = []
    for param in old_parameters:
        field: dataclasses.Field = fields_by_name[param.name]  # type: ignore[type-arg]

        typer_kwargs: dict[str, object] = {}
        if env_var_prefix:
            typer_kwargs["envvar"] = f"{env_var_prefix}{param.name.upper()}"
        if "help" in field.metadata:
            typer_kwargs["help"] = field.metadata["help"]

        resolved_type: type = resolved_hints.get(param.name, param.annotation)
        new_annotation = Annotated[_repeatable_as_list(resolved_type), typer.Option(**typer_kwargs)]
        default = _resolve_default(field, param)

        new_parameters.append(
            param.replace(annotation=new_annotation, default=list(default) if isinstance(default, tuple) else default)
        )
    return new_parameters


def _repeatable_as_list(annotation: type) -> type:
    """click has no variadic tuple, so a repeatable option reaches it as the list it is spelled with."""
    if typing.get_origin(annotation) is tuple and typing.get_args(annotation)[1:] == (Ellipsis,):
        return list[typing.get_args(annotation)[0]]  # type: ignore[misc, return-value]
    return annotation


def _build(dataclass_cls: type, kwargs: dict[str, object]) -> Any:
    hints: dict[str, type] = typing.get_type_hints(dataclass_cls)
    return dataclass_cls(
        **{
            name: tuple(value) if typing.get_origin(hints[name]) is tuple and isinstance(value, list) else value
            for name, value in kwargs.items()
        }
    )


def _print_arguments(data: object) -> None:
    fields = dataclasses.fields(data)
    max_key_len = max(len(f.name) for f in fields)
    sep = "+" + "-" * (max_key_len + 2) + "+" + "-" * 52 + "+"
    print(sep)
    print(f"| {'Argument':<{max_key_len}} | {'Value':<50} |")
    print(sep)
    for f in fields:
        val_raw = getattr(data, f.name)
        val = str(val_raw.value if isinstance(val_raw, Enum) else val_raw)
        if len(val) > 50:
            val = val[:47] + "..."
        print(f"| {f.name:<{max_key_len}} | {val:<50} |")
    print(sep)
