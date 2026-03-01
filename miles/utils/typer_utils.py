import dataclasses
import functools
import inspect
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
    - ``"flag"``: custom CLI flag name (e.g. ``"--sp"``)

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
    sig: inspect.Signature = inspect.signature(func)
    first_param: inspect.Parameter = list(sig.parameters.values())[0]
    dataclass_cls: type = first_param.annotation
    assert dataclasses.is_dataclass(dataclass_cls)

    init_sig: inspect.Signature = inspect.signature(dataclass_cls.__init__)
    old_parameters: list[inspect.Parameter] = list(init_sig.parameters.values())
    if old_parameters and old_parameters[0].name == "self":
        del old_parameters[0]

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

        flag: str | None = field.metadata.get("flag")
        if flag is not None:
            new_annotation = Annotated[param.annotation, typer.Option(flag, **typer_kwargs)]
        else:
            new_annotation = Annotated[param.annotation, typer.Option(**typer_kwargs)]

        new_parameters.append(param.replace(annotation=new_annotation))

    def wrapped(**kwargs: object) -> object:
        data: object = dataclass_cls(**kwargs)
        return func(data)

    wrapped.__signature__ = init_sig.replace(parameters=new_parameters)  # type: ignore[attr-defined]
    wrapped.__doc__ = func.__doc__
    wrapped.__name__ = func.__name__  # type: ignore[attr-defined]
    wrapped.__qualname__ = func.__qualname__  # type: ignore[attr-defined]

    return wrapped  # type: ignore[return-value]


# unit test
if __name__ == "__main__":
    from typer.testing import CliRunner

    @dataclasses.dataclass
    class DemoArgs:
        name: str
        count: int = 1

    app = typer.Typer()

    @app.command()
    @dataclass_cli
    def main(args: DemoArgs):
        print(f"{args.name}|{args.count}")

    runner = CliRunner()

    res1 = runner.invoke(app, [], env={"MILES_SCRIPT_NAME": "EnvName", "MILES_SCRIPT_COUNT": "10"})
    print(f"{res1.stdout=}")
    assert res1.exit_code == 0
    assert "EnvName|10" in res1.stdout.strip()

    res2 = runner.invoke(app, ["--count", "999"], env={"MILES_SCRIPT_NAME": "EnvName"})
    print(f"{res2.stdout=}")
    assert res2.exit_code == 0
    assert "EnvName|999" in res2.stdout.strip()

    # Test with metadata (help + flag)
    @dataclasses.dataclass
    class MetaArgs:
        verbose: bool = dataclasses.field(
            default=False,
            metadata={"help": "Enable verbose output", "flag": "--verbose"},
        )

    app2 = typer.Typer()

    @app2.command()
    @dataclass_cli(env_var_prefix="")
    def meta_cmd(args: MetaArgs):
        print(f"verbose={args.verbose}")

    res3 = runner.invoke(app2, ["--help"])
    print(f"{res3.stdout=}")
    assert res3.exit_code == 0
    assert "Enable verbose output" in res3.stdout

    print("All Tests Passed!")
