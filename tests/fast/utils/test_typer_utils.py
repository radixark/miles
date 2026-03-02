import dataclasses

import typer
from typer.testing import CliRunner

from miles.utils.typer_utils import dataclass_cli

runner = CliRunner()


# ---------------------------------------------------------------------------
# Test dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _SimpleArgs:
    name: str
    count: int = 1


@dataclasses.dataclass
class _BoolArgs:
    verbose: bool = False


@dataclasses.dataclass
class _MetaHelpArgs:
    level: int = dataclasses.field(
        default=0,
        metadata={"help": "Verbosity level"},
    )


@dataclasses.dataclass
class _MetaFlagArgs:
    short_param: str = dataclasses.field(
        default="x",
        metadata={"flag": "--sp"},
    )


@dataclasses.dataclass
class _MetaHelpAndFlagArgs:
    verbose: bool = dataclasses.field(
        default=False,
        metadata={"help": "Enable verbose output", "flag": "--verbose"},
    )


@dataclasses.dataclass
class _MultiFieldArgs:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False


@dataclasses.dataclass
class _AllRequiredArgs:
    first: str
    second: int


# ---------------------------------------------------------------------------
# Bare decorator: @dataclass_cli
# ---------------------------------------------------------------------------


class TestBareDecorator:
    def test_env_vars(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            print(f"{args.name}|{args.count}")

        result = runner.invoke(
            app, [], env={"MILES_SCRIPT_NAME": "EnvName", "MILES_SCRIPT_COUNT": "10"}
        )
        assert result.exit_code == 0
        assert "EnvName|10" in result.stdout

    def test_cli_flag_overrides_env_var(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            print(f"{args.name}|{args.count}")

        result = runner.invoke(
            app, ["--count", "999"], env={"MILES_SCRIPT_NAME": "EnvName"}
        )
        assert result.exit_code == 0
        assert "EnvName|999" in result.stdout

    def test_default_value_used_when_omitted(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            print(f"{args.name}|{args.count}")

        result = runner.invoke(app, ["--name", "Alice"])
        assert result.exit_code == 0
        assert "Alice|1" in result.stdout

    def test_all_flags_explicit(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            print(f"{args.name}|{args.count}")

        result = runner.invoke(app, ["--name", "Bob", "--count", "42"])
        assert result.exit_code == 0
        assert "Bob|42" in result.stdout

    def test_missing_required_field_fails(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            print(args.name)

        result = runner.invoke(app, [])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Parameterized decorator: @dataclass_cli(env_var_prefix=...)
# ---------------------------------------------------------------------------


class TestParameterizedDecorator:
    def test_empty_prefix_disables_env_vars(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _SimpleArgs) -> None:
            print(f"{args.name}|{args.count}")

        result = runner.invoke(
            app,
            [],
            env={"MILES_SCRIPT_NAME": "ShouldBeIgnored"},
        )
        assert result.exit_code != 0

    def test_custom_prefix(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="MY_APP_")
        def cmd(args: _SimpleArgs) -> None:
            print(f"{args.name}|{args.count}")

        result = runner.invoke(
            app, [], env={"MY_APP_NAME": "Custom", "MY_APP_COUNT": "77"}
        )
        assert result.exit_code == 0
        assert "Custom|77" in result.stdout

    def test_custom_prefix_ignores_default_prefix(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="MY_APP_")
        def cmd(args: _SimpleArgs) -> None:
            print(f"{args.name}|{args.count}")

        result = runner.invoke(
            app, [], env={"MILES_SCRIPT_NAME": "Wrong"}
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Field metadata: help & flag
# ---------------------------------------------------------------------------


class TestFieldMetadata:
    def test_help_appears_in_help_text(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _MetaHelpArgs) -> None:
            pass

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Verbosity level" in result.stdout

    def test_custom_flag_name(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MetaFlagArgs) -> None:
            print(f"val={args.short_param}")

        result = runner.invoke(app, ["--sp", "hello"])
        assert result.exit_code == 0
        assert "val=hello" in result.stdout

    def test_help_and_flag_together(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MetaHelpAndFlagArgs) -> None:
            print(f"verbose={args.verbose}")

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Enable verbose output" in result.stdout

        result = runner.invoke(app, ["--verbose"])
        assert result.exit_code == 0
        assert "verbose=True" in result.stdout


# ---------------------------------------------------------------------------
# Bool fields
# ---------------------------------------------------------------------------


class TestBoolFields:
    def test_bool_default_false(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _BoolArgs) -> None:
            print(f"verbose={args.verbose}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "verbose=False" in result.stdout

    def test_bool_flag_enables(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _BoolArgs) -> None:
            print(f"verbose={args.verbose}")

        result = runner.invoke(app, ["--verbose"])
        assert result.exit_code == 0
        assert "verbose=True" in result.stdout


# ---------------------------------------------------------------------------
# Multiple fields
# ---------------------------------------------------------------------------


class TestMultipleFields:
    def test_all_defaults(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MultiFieldArgs) -> None:
            print(f"{args.host}:{args.port}:{args.debug}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "localhost:8080:False" in result.stdout

    def test_override_some(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MultiFieldArgs) -> None:
            print(f"{args.host}:{args.port}:{args.debug}")

        result = runner.invoke(app, ["--port", "9090", "--debug"])
        assert result.exit_code == 0
        assert "localhost:9090:True" in result.stdout

    def test_all_required_both_provided(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _AllRequiredArgs) -> None:
            print(f"{args.first}|{args.second}")

        result = runner.invoke(app, ["--first", "a", "--second", "2"])
        assert result.exit_code == 0
        assert "a|2" in result.stdout

    def test_all_required_missing_one_fails(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _AllRequiredArgs) -> None:
            pass

        result = runner.invoke(app, ["--first", "a"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Wrapped function attributes
# ---------------------------------------------------------------------------


class TestWrappedAttributes:
    def test_preserves_name(self) -> None:
        @dataclass_cli
        def my_func(args: _SimpleArgs) -> None:
            """My docstring."""

        assert my_func.__name__ == "my_func"
        assert my_func.__qualname__.endswith("my_func")

    def test_preserves_docstring(self) -> None:
        @dataclass_cli
        def my_func(args: _SimpleArgs) -> None:
            """My docstring."""

        assert my_func.__doc__ == "My docstring."

    def test_preserves_name_parameterized(self) -> None:
        @dataclass_cli(env_var_prefix="")
        def another_func(args: _SimpleArgs) -> None:
            """Another doc."""

        assert another_func.__name__ == "another_func"
        assert another_func.__doc__ == "Another doc."
