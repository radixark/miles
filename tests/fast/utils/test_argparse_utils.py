"""Tests for DataclassArgparseBridge."""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from miles.utils.argparse_utils import DataclassArgparseBridge


@dataclasses.dataclass(frozen=True)
class _SampleArgs:
    name: str
    count: int
    rate: float = 0.5
    label: str = "default"
    verbose: bool = False
    tag: str | None = None
    limit: int | None = None
    threshold: float | None = None


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

        namespace: argparse.Namespace = parser.parse_args([
            "--sample-name", "hello",
            "--sample-count", "42",
            "--sample-rate", "1.5",
            "--sample-verbose",
            "--sample-tag", "mytag",
            "--sample-limit", "100",
            "--sample-threshold", "0.9",
        ])

        assert namespace.sample_name == "hello"
        assert namespace.sample_count == 42
        assert namespace.sample_rate == 1.5
        assert namespace.sample_verbose is True
        assert namespace.sample_tag == "mytag"
        assert namespace.sample_limit == 100
        assert namespace.sample_threshold == 0.9

    def test_defaults_are_applied(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args([
            "--sample-name", "test",
            "--sample-count", "1",
        ])

        assert namespace.sample_rate == 0.5
        assert namespace.sample_label == "default"
        assert namespace.sample_verbose is False
        assert namespace.sample_tag is None
        assert namespace.sample_limit is None
        assert namespace.sample_threshold is None

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

        namespace: argparse.Namespace = parser.parse_args([
            "--sample-name", "world",
            "--sample-count", "7",
            "--sample-verbose",
            "--sample-tag", "x",
        ])

        instance: _SampleArgs = _BRIDGE.from_namespace(namespace)

        assert instance.name == "world"
        assert instance.count == 7
        assert instance.rate == 0.5
        assert instance.verbose is True
        assert instance.tag == "x"
        assert instance.limit is None

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
            name="hello",
            count=42,
            rate=1.5,
            verbose=True,
            tag="mytag",
            limit=100,
            threshold=0.9,
        )

        cli: str = _BRIDGE.to_cli_args(instance)

        assert "--sample-name hello" in cli
        assert "--sample-count 42" in cli
        assert "--sample-rate 1.5" in cli
        assert "--sample-verbose" in cli
        assert "--sample-tag mytag" in cli
        assert "--sample-limit 100" in cli
        assert "--sample-threshold 0.9" in cli

    def test_skips_none_and_false(self) -> None:
        instance: _SampleArgs = _SampleArgs(
            name="test",
            count=1,
        )

        cli: str = _BRIDGE.to_cli_args(instance)

        assert "--sample-verbose" not in cli
        assert "--sample-tag" not in cli
        assert "--sample-limit" not in cli
        assert "--sample-threshold" not in cli

    def test_no_prefix_serialization(self) -> None:
        instance: _NoPrefixArgs = _NoPrefixArgs(name="foo", debug=True)
        cli: str = _NO_PREFIX_BRIDGE.to_cli_args(instance)

        assert "--name foo" in cli
        assert "--debug" in cli


class TestRoundTrip:
    def test_to_cli_args_then_parse_back(self) -> None:
        original: _SampleArgs = _SampleArgs(
            name="roundtrip",
            count=99,
            rate=2.5,
            label="custom",
            verbose=True,
            tag="t",
            limit=50,
            threshold=0.1,
        )

        cli: str = _BRIDGE.to_cli_args(original)

        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)
        namespace: argparse.Namespace = parser.parse_args(cli.split())

        restored: _SampleArgs = _BRIDGE.from_namespace(namespace)
        assert restored == original

    def test_round_trip_with_defaults(self) -> None:
        original: _SampleArgs = _SampleArgs(name="minimal", count=1)

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
