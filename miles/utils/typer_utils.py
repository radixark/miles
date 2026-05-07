import dataclasses
import functools
import inspect
import typing
from collections.abc import Callable
from typing import Annotated, TypeVar, overload

import typer

_F = TypeVar("_F", bound=Callable[..., object])


@overload
def dataclass_cli(func: _F) -> _F: ...


@overload
def dataclass_cli(
    func: None = None,
    *,
    env_var_prefix: str = "MILES_SCRIPT_",
) -> Callable[[_F], _F]: ...


def dataclass_cli(
    func: _F | None = None,
    *,
    env_var_prefix: str = "MILES_SCRIPT_",
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


def _wrap(func: _F, *, env_var_prefix: str) -> _F:
    hints: dict[str, type] = typing.get_type_hints(func)
    first_param_name: str = next(iter(inspect.signature(func).parameters))
    dataclass_cls: type = hints[first_param_name]
    assert dataclasses.is_dataclass(dataclass_cls)

    init_sig: inspect.Signature = inspect.signature(dataclass_cls.__init__)
    old_parameters: list[inspect.Parameter] = list(init_sig.parameters.values())
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
        new_annotation = Annotated[resolved_type, typer.Option(**typer_kwargs)]

        new_parameters.append(param.replace(annotation=new_annotation))

    def wrapped(**kwargs: object) -> object:
        data: object = dataclass_cls(**kwargs)
        fields = dataclasses.fields(data)
        max_key_len = max(len(f.name) for f in fields)
        sep = "+" + "-" * (max_key_len + 2) + "+" + "-" * 52 + "+"
        print(sep)
        print(f"| {'Argument':<{max_key_len}} | {'Value':<50} |")
        print(sep)
        for f in fields:
            val = str(getattr(data, f.name))
            if len(val) > 50:
                val = val[:47] + "..."
            print(f"| {f.name:<{max_key_len}} | {val:<50} |")
        print(sep)
        return func(data)

    wrapped.__signature__ = init_sig.replace(parameters=new_parameters)  # type: ignore[attr-defined]
    wrapped.__doc__ = func.__doc__
    wrapped.__name__ = func.__name__  # type: ignore[attr-defined]
    wrapped.__qualname__ = func.__qualname__  # type: ignore[attr-defined]

    return wrapped  # type: ignore[return-value]
