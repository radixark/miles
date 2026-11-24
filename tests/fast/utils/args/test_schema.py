import argparse
from typing import Annotated

import pytest
from pydantic import Field

from miles.utils.args.schema import Arg, BaseConfig, reset_arg, validate_complete_config


class _ChildConfig(BaseConfig):
    count: int = 3


class _ContainerConfig(BaseConfig):
    child: _ChildConfig
    children: list[_ChildConfig]
    by_name: dict[str, _ChildConfig]
    transient: int = Field(default=0, exclude=True)


class TestCompleteConfig:
    def test_explicit_defaults_and_excluded_runtime_fields_are_accepted(self) -> None:
        """Explicit default values remain valid without serializing excluded runtime state."""
        config = validate_complete_config(
            _ContainerConfig,
            {"child": {"count": 3}, "children": [{"count": 0}], "by_name": {"x": {"count": 4}}},
        )

        assert config.child.count == 3
        assert config.children[0].count == 0
        assert config.by_name["x"].count == 4

    @pytest.mark.parametrize(
        ("payload", "path"),
        [
            ({"child": {}, "children": [], "by_name": {}}, "_ContainerConfig.child"),
            ({"child": {"count": 1}, "children": [{}], "by_name": {}}, "_ContainerConfig.children[0]"),
            ({"child": {"count": 1}, "children": [], "by_name": {"x": {}}}, "_ContainerConfig.by_name['x']"),
        ],
    )
    def test_nested_omissions_report_the_exact_payload_location(self, payload: dict, path: str) -> None:
        """Nested defaults cannot conceal incomplete transported configurations."""
        with pytest.raises(ValueError) as error:
            validate_complete_config(_ContainerConfig, payload)

        assert path in str(error.value)
        assert "count" in str(error.value)


class TestArgumentSchema:
    def test_aliases_optional_types_and_boolean_flags_parse_together(self) -> None:
        """CLI aliases retain their declared destination and optional scalar conversion."""

        class Config(BaseConfig):
            count: Annotated[int | None, Arg(aliases=("-n",))] = None
            enabled: Annotated[bool, Arg()] = False
            internal: str = "private"

        parser = argparse.ArgumentParser()
        Config.add_arguments(parser)

        assert vars(parser.parse_args(["-n", "7", "--enabled"])) == {"count": 7, "enabled": True}
        assert vars(parser.parse_args([])) == {"count": None, "enabled": False}

    def test_default_factories_are_independent_between_parsers(self) -> None:
        """Mutating one parser's list default cannot contaminate another parser."""

        class Config(BaseConfig):
            items: Annotated[list[int], Arg(type_parser=int, nargs="+")] = Field(default_factory=lambda: [2])

        first = argparse.ArgumentParser()
        second = argparse.ArgumentParser()
        Config.add_arguments(first)
        Config.add_arguments(second)
        first.parse_args([]).items.append(9)

        assert second.parse_args([]).items == [2]
        assert second.parse_args(["--items", "4", "5"]).items == [4, 5]

    def test_ambiguous_argument_metadata_is_rejected(self) -> None:
        """A field cannot silently select one of two competing CLI declarations."""

        class Config(BaseConfig):
            value: Annotated[int, Arg(), Arg()] = 1

        with pytest.raises(ValueError, match="Multiple argument declarations"):
            Config.add_arguments(argparse.ArgumentParser())

    def test_unsupported_types_require_an_explicit_converter(self) -> None:
        """Complex values cannot accidentally fall through scalar CLI inference."""

        class Config(BaseConfig):
            values: Annotated[list[int], Arg()] = []

        with pytest.raises(TypeError, match="Explicit type_parser"):
            Config.add_arguments(argparse.ArgumentParser())


class TestResetArgument:
    def test_reset_changes_default_without_duplicating_an_existing_option(self) -> None:
        """Reset preserves explicit user values while replacing the fallback."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--count", type=int, default=1, choices=[1, 2])
        reset_arg(parser=parser, name="--count", type=int, default=2, choices=[1, 2])

        assert parser.parse_args([]).count == 2
        assert parser.parse_args(["--count", "1"]).count == 1

    @pytest.mark.parametrize("kwargs", [{"type": str}, {"type": int, "choices": [3]}, {"action": "store_true"}])
    def test_incompatible_resets_leave_the_original_default_intact(self, kwargs: dict) -> None:
        """Rejected declarations must not partially change the existing option."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--count", type=int, default=1, choices=[1, 2])

        with pytest.raises(AssertionError, match="Cannot reset"):
            reset_arg(parser=parser, name="--count", default=9, **kwargs)

        assert parser.parse_args([]).count == 1

    def test_reset_registers_a_missing_option(self) -> None:
        """Backend-specific absent options are registered with the requested contract."""
        parser = argparse.ArgumentParser()
        reset_arg(parser=parser, name="--count", type=int, default=2)

        assert parser.parse_args(["--count", "8"]).count == 8


class TestConfigSlice:
    def test_slicing_validates_the_leaf_and_drops_unowned_fields(self) -> None:
        """A component receives only its own schema from the complete configuration."""

        class Whole(_ChildConfig):
            secret: str

        source = Whole(count=7, secret="private")
        result = _ChildConfig.slice_from(source)

        assert result.model_dump() == {"count": 7}
        assert source.secret == "private"

    def test_slicing_rejects_missing_fields_even_when_the_leaf_has_defaults(self) -> None:
        """Defaults must not hide an ownership mismatch between component schemas."""

        class Empty(BaseConfig):
            pass

        with pytest.raises(AssertionError, match="it lacks.*count"):
            _ChildConfig.slice_from(Empty())


class TestResetMetadata:
    def test_reset_metadata_updates_the_backend_default(self) -> None:
        """Annotated reset declarations preserve the backend's explicit CLI values."""

        class Config(BaseConfig):
            count: Annotated[int, Arg(reset=True)] = 4

        parser = argparse.ArgumentParser()
        parser.add_argument("--count", type=int, default=1)
        Config.add_arguments(parser)

        assert parser.parse_args([]).count == 4
        assert parser.parse_args(["--count", "9"]).count == 9

    def test_reset_metadata_rejects_aliases_before_touching_the_parser(self) -> None:
        """Reset declarations cannot ambiguously target several option names."""

        class Config(BaseConfig):
            count: Annotated[int, Arg(reset=True, aliases=("-n",))] = 4

        parser = argparse.ArgumentParser()
        parser.add_argument("--count", type=int, default=1)
        with pytest.raises(ValueError, match="exactly one"):
            Config.add_arguments(parser)
        assert parser.parse_args([]).count == 1

    def test_explicit_const_and_positional_arguments_keep_their_semantics(self) -> None:
        """Optional-value flags distinguish omission, a bare flag and an explicit scalar."""

        class Config(BaseConfig):
            path: Annotated[str, Arg(cli_name="path")]
            limit: Annotated[int | None, Arg(nargs="?", const=7)] = None

        parser = argparse.ArgumentParser()
        Config.add_arguments(parser)

        assert vars(parser.parse_args(["data"])) == {"path": "data", "limit": None}
        assert vars(parser.parse_args(["data", "--limit"])) == {"path": "data", "limit": 7}
        assert vars(parser.parse_args(["data", "--limit", "9"])) == {"path": "data", "limit": 9}
