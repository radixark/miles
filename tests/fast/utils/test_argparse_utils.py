"""Tests for DataclassArgparseBridge."""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from miles.utils.argparse_utils import DataclassArgparseBridge


@dataclasses.dataclass(frozen=True)
class _SampleArgs:
    mandatory_str: str
    mandatory_int: int
    mandatory_float: float
    defaulted_str: str = "default"
    defaulted_int: int = 10
    defaulted_float: float = 0.5
    flag: bool = False
    optional_str: str | None = None
    optional_int: int | None = None
    optional_float: float | None = None


_BRIDGE: DataclassArgparseBridge[_SampleArgs] = DataclassArgparseBridge(
    _SampleArgs,
    prefix="sample",
    group_title="sample args",
)


@dataclasses.dataclass(frozen=True)
class _NoPrefixArgs:
    name: str
    debug: bool = False


_NO_PREFIX_BRIDGE: DataclassArgparseBridge[_NoPrefixArgs] = DataclassArgparseBridge(
    _NoPrefixArgs,
    prefix="",
)


class TestRegisterOnParser:
    def test_required_fields_are_required(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)

        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_all_field_types(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(
            [
                "--sample-mandatory-str",
                "hello",
                "--sample-mandatory-int",
                "42",
                "--sample-mandatory-float",
                "3.14",
                "--sample-defaulted-str",
                "custom",
                "--sample-defaulted-int",
                "99",
                "--sample-defaulted-float",
                "1.5",
                "--sample-flag",
                "--sample-optional-str",
                "mytag",
                "--sample-optional-int",
                "100",
                "--sample-optional-float",
                "0.9",
            ]
        )

        assert namespace.sample_mandatory_str == "hello"
        assert namespace.sample_mandatory_int == 42
        assert namespace.sample_mandatory_float == 3.14
        assert namespace.sample_defaulted_str == "custom"
        assert namespace.sample_defaulted_int == 99
        assert namespace.sample_defaulted_float == 1.5
        assert namespace.sample_flag is True
        assert namespace.sample_optional_str == "mytag"
        assert namespace.sample_optional_int == 100
        assert namespace.sample_optional_float == 0.9

    def test_defaults_are_applied(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(
            [
                "--sample-mandatory-str",
                "test",
                "--sample-mandatory-int",
                "1",
                "--sample-mandatory-float",
                "2.0",
            ]
        )

        assert namespace.sample_defaulted_str == "default"
        assert namespace.sample_defaulted_int == 10
        assert namespace.sample_defaulted_float == 0.5
        assert namespace.sample_flag is False
        assert namespace.sample_optional_str is None
        assert namespace.sample_optional_int is None
        assert namespace.sample_optional_float is None

    def test_no_prefix(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _NO_PREFIX_BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(["--name", "foo", "--debug"])
        assert namespace.name == "foo"
        assert namespace.debug is True


class TestFromNamespace:
    def test_constructs_dataclass_from_namespace(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(
            [
                "--sample-mandatory-str",
                "world",
                "--sample-mandatory-int",
                "7",
                "--sample-mandatory-float",
                "1.0",
                "--sample-flag",
                "--sample-optional-str",
                "x",
            ]
        )

        instance: _SampleArgs = _BRIDGE.from_namespace(namespace)

        assert instance.mandatory_str == "world"
        assert instance.mandatory_int == 7
        assert instance.mandatory_float == 1.0
        assert instance.defaulted_str == "default"
        assert instance.flag is True
        assert instance.optional_str == "x"
        assert instance.optional_int is None

    def test_no_prefix_from_namespace(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _NO_PREFIX_BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(["--name", "bar"])
        instance: _NoPrefixArgs = _NO_PREFIX_BRIDGE.from_namespace(namespace)

        assert instance.name == "bar"
        assert instance.debug is False


class TestToCliArgs:
    def test_serializes_all_types(self) -> None:
        instance: _SampleArgs = _SampleArgs(
            mandatory_str="hello",
            mandatory_int=42,
            mandatory_float=3.14,
            flag=True,
            optional_str="mytag",
            optional_int=100,
            optional_float=0.9,
        )

        cli: str = _BRIDGE.to_cli_args(instance)

        assert "--sample-mandatory-str hello" in cli
        assert "--sample-mandatory-int 42" in cli
        assert "--sample-mandatory-float 3.14" in cli
        assert "--sample-flag" in cli
        assert "--sample-optional-str mytag" in cli
        assert "--sample-optional-int 100" in cli
        assert "--sample-optional-float 0.9" in cli

    def test_skips_none_and_false(self) -> None:
        instance: _SampleArgs = _SampleArgs(
            mandatory_str="test",
            mandatory_int=1,
            mandatory_float=2.0,
        )

        cli: str = _BRIDGE.to_cli_args(instance)

        assert "--sample-flag" not in cli
        assert "--sample-optional-str" not in cli
        assert "--sample-optional-int" not in cli
        assert "--sample-optional-float" not in cli

    def test_no_prefix_serialization(self) -> None:
        instance: _NoPrefixArgs = _NoPrefixArgs(name="foo", debug=True)
        cli: str = _NO_PREFIX_BRIDGE.to_cli_args(instance)

        assert "--name foo" in cli
        assert "--debug" in cli


class TestRoundTrip:
    def test_to_cli_args_then_parse_back(self) -> None:
        original: _SampleArgs = _SampleArgs(
            mandatory_str="roundtrip",
            mandatory_int=99,
            mandatory_float=2.5,
            defaulted_str="custom",
            defaulted_int=77,
            defaulted_float=1.1,
            flag=True,
            optional_str="t",
            optional_int=50,
            optional_float=0.1,
        )

        cli: str = _BRIDGE.to_cli_args(original)

        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)
        namespace: argparse.Namespace = parser.parse_args(cli.split())

        restored: _SampleArgs = _BRIDGE.from_namespace(namespace)
        assert restored == original

    def test_round_trip_with_defaults(self) -> None:
        original: _SampleArgs = _SampleArgs(
            mandatory_str="minimal",
            mandatory_int=1,
            mandatory_float=2.0,
        )

        cli: str = _BRIDGE.to_cli_args(original)

        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)
        namespace: argparse.Namespace = parser.parse_args(cli.split())

        restored: _SampleArgs = _BRIDGE.from_namespace(namespace)
        assert restored == original


class TestValidation:
    def test_rejects_non_dataclass(self) -> None:
        with pytest.raises(TypeError, match="not a dataclass"):
            DataclassArgparseBridge(str, prefix="x")

    def test_rejects_unsupported_type(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class _Bad:
            data: list[str] = dataclasses.field(default_factory=list)

        bridge: DataclassArgparseBridge[_Bad] = DataclassArgparseBridge(_Bad, prefix="bad")
        parser: argparse.ArgumentParser = argparse.ArgumentParser()

        with pytest.raises(TypeError, match="Unsupported field type"):
            bridge.register_on_parser(parser)
