import dataclasses
import enum
from typing import Optional

import pytest
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
class _MetaHelpAndFlagArgs:
    verbose: bool = dataclasses.field(
        default=False,
        metadata={"help": "Enable verbose output"},
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


@dataclasses.dataclass
class _SingleFieldArgs:
    value: str = "default"


@dataclasses.dataclass
class _FloatArgs:
    rate: float = 0.01


@dataclasses.dataclass
class _OptionalStrArgs:
    tag: Optional[str] = None


class _Color(str, enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclasses.dataclass
class _EnumArgs:
    color: _Color = _Color.RED


@dataclasses.dataclass
class _FieldWithDefaultNoMeta:
    count: int = dataclasses.field(default=5)


@dataclasses.dataclass
class _DefaultFactoryArgs:
    items: str = dataclasses.field(default_factory=lambda: "a,b,c")


@dataclasses.dataclass
class _MixedMetaArgs:
    plain: str = "hello"
    with_help: int = dataclasses.field(
        default=0,
        metadata={"help": "A number"},
    )
    with_flag: str = dataclasses.field(
        default="x",
    )
    with_both: bool = dataclasses.field(
        default=False,
        metadata={"help": "Toggle it"},
    )


@dataclasses.dataclass
class _EmptyMetaArgs:
    value: str = dataclasses.field(default="ok", metadata={})


@dataclasses.dataclass
class _MultipleRequiredArgs:
    alpha: str
    beta: str
    gamma: str


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

    def test_default_prefix_when_called_with_parens(self) -> None:
        """@dataclass_cli() with no args should behave same as bare @dataclass_cli."""
        app = typer.Typer()

        @app.command()
        @dataclass_cli()
        def cmd(args: _SimpleArgs) -> None:
            print(f"{args.name}|{args.count}")

        result = runner.invoke(
            app, [], env={"MILES_SCRIPT_NAME": "ViaParens", "MILES_SCRIPT_COUNT": "3"}
        )
        assert result.exit_code == 0
        assert "ViaParens|3" in result.stdout


# ---------------------------------------------------------------------------
# Field metadata: help
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

    def test_empty_metadata_treated_as_no_metadata(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EmptyMetaArgs) -> None:
            print(f"value={args.value}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "value=ok" in result.stdout

    def test_mixed_metadata_fields(self) -> None:
        """Dataclass with a mix of plain fields and help-annotated fields."""
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MixedMetaArgs) -> None:
            print(f"{args.plain}|{args.with_help}|{args.with_flag}|{args.with_both}")

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "A number" in result.stdout
        assert "Toggle it" in result.stdout

        result = runner.invoke(app, ["--with-help", "7", "--with-flag", "y", "--with-both"])
        assert result.exit_code == 0
        assert "hello|7|y|True" in result.stdout


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

    def test_bool_no_flag_disables(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _BoolArgs) -> None:
            print(f"verbose={args.verbose}")

        result = runner.invoke(app, ["--no-verbose"])
        assert result.exit_code == 0
        assert "verbose=False" in result.stdout


# ---------------------------------------------------------------------------
# Various field types
# ---------------------------------------------------------------------------


class TestFieldTypes:
    def test_float_field(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _FloatArgs) -> None:
            print(f"rate={args.rate}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "rate=0.01" in result.stdout

        result = runner.invoke(app, ["--rate", "0.5"])
        assert result.exit_code == 0
        assert "rate=0.5" in result.stdout

    def test_optional_str_default_none(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _OptionalStrArgs) -> None:
            print(f"tag={args.tag}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "tag=None" in result.stdout

    def test_optional_str_provided(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _OptionalStrArgs) -> None:
            print(f"tag={args.tag}")

        result = runner.invoke(app, ["--tag", "v1"])
        assert result.exit_code == 0
        assert "tag=v1" in result.stdout

    def test_enum_field_default(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EnumArgs) -> None:
            print(f"color={args.color.value}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "color=red" in result.stdout

    def test_enum_field_override(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EnumArgs) -> None:
            print(f"color={args.color.value}")

        result = runner.invoke(app, ["--color", "blue"])
        assert result.exit_code == 0
        assert "color=blue" in result.stdout

    def test_enum_invalid_value_fails(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EnumArgs) -> None:
            pass

        result = runner.invoke(app, ["--color", "yellow"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# dataclasses.field without metadata / with default_factory
# ---------------------------------------------------------------------------


class TestDataclassField:
    def test_field_with_default_no_metadata(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _FieldWithDefaultNoMeta) -> None:
            print(f"count={args.count}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "count=5" in result.stdout

    def test_field_with_default_factory_override(self) -> None:
        """default_factory defaults show as <factory> in the signature,
        so only the explicit-override path is reliable."""
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _DefaultFactoryArgs) -> None:
            print(f"items={args.items}")

        result = runner.invoke(app, ["--items", "x,y"])
        assert result.exit_code == 0
        assert "items=x,y" in result.stdout


# ---------------------------------------------------------------------------
# Single field / many required fields
# ---------------------------------------------------------------------------


class TestFieldCounts:
    def test_single_field_dataclass(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _SingleFieldArgs) -> None:
            print(f"value={args.value}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "value=default" in result.stdout

    def test_multiple_required_all_provided(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MultipleRequiredArgs) -> None:
            print(f"{args.alpha}|{args.beta}|{args.gamma}")

        result = runner.invoke(
            app, ["--alpha", "a", "--beta", "b", "--gamma", "c"]
        )
        assert result.exit_code == 0
        assert "a|b|c" in result.stdout

    def test_multiple_required_missing_any_fails(self) -> None:
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MultipleRequiredArgs) -> None:
            pass

        result = runner.invoke(app, ["--alpha", "a", "--beta", "b"])
        assert result.exit_code != 0


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
# Return value passthrough
# ---------------------------------------------------------------------------


class TestReturnValue:
    def test_return_value_is_forwarded(self) -> None:
        @dataclass_cli(env_var_prefix="")
        def compute(args: _SimpleArgs) -> str:
            return f"result:{args.name}"

        value = compute(name="test", count=1)
        assert value == "result:test"

    def test_return_none(self) -> None:
        @dataclass_cli(env_var_prefix="")
        def noop(args: _SimpleArgs) -> None:
            pass

        value = noop(name="x", count=0)
        assert value is None


# ---------------------------------------------------------------------------
# Dataclass instance correctness
# ---------------------------------------------------------------------------


class TestDataclassInstance:
    def test_receives_correct_dataclass_type(self) -> None:
        received: list = []

        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _SimpleArgs) -> None:
            received.append(args)

        runner.invoke(app, ["--name", "check", "--count", "7"])
        assert len(received) == 1
        assert isinstance(received[0], _SimpleArgs)
        assert received[0].name == "check"
        assert received[0].count == 7

    def test_bool_field_receives_python_bool(self) -> None:
        received: list = []

        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _BoolArgs) -> None:
            received.append(args)

        runner.invoke(app, ["--verbose"])
        assert received[0].verbose is True

    def test_enum_field_receives_enum_instance(self) -> None:
        received: list = []

        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EnumArgs) -> None:
            received.append(args)

        runner.invoke(app, ["--color", "green"])
        assert received[0].color is _Color.GREEN


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

    def test_no_docstring(self) -> None:
        @dataclass_cli
        def no_doc(args: _SimpleArgs) -> None:
            pass

        assert no_doc.__doc__ is None

    def test_has_signature(self) -> None:
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _SimpleArgs) -> None:
            pass

        sig = inspect.signature(cmd)
        param_names = list(sig.parameters.keys())
        assert "name" in param_names
        assert "count" in param_names
        assert "args" not in param_names


# ---------------------------------------------------------------------------
# Error: non-dataclass annotation
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_non_dataclass_raises(self) -> None:
        class NotADataclass:
            pass

        with pytest.raises(AssertionError):

            @dataclass_cli
            def bad(args: NotADataclass) -> None:
                pass

    def test_non_dataclass_raises_parameterized(self) -> None:
        class NotADataclass:
            pass

        with pytest.raises(AssertionError):

            @dataclass_cli(env_var_prefix="")
            def bad(args: NotADataclass) -> None:
                pass


# ---------------------------------------------------------------------------
# Env var naming: uppercase conversion
# ---------------------------------------------------------------------------


class TestEnvVarNaming:
    def test_underscore_field_name_uppercased(self) -> None:
        @dataclasses.dataclass
        class _SnakeArgs:
            my_long_name: str = "default"

        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SnakeArgs) -> None:
            print(f"val={args.my_long_name}")

        result = runner.invoke(
            app, [], env={"MILES_SCRIPT_MY_LONG_NAME": "from_env"}
        )
        assert result.exit_code == 0
        assert "val=from_env" in result.stdout


# ---------------------------------------------------------------------------
# import needed for signature inspection
# ---------------------------------------------------------------------------
import inspect
