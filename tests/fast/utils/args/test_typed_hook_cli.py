import argparse
from types import SimpleNamespace
from typing import Annotated

import pytest

from miles.utils.args.custom_function import (
    CustomFunctionConfig,
    add_user_provided_function_arguments,
    resolve_custom_function_configs,
)
from miles.utils.args.schema import Arg, BaseConfig
from miles.utils.function_registry import function_registry


class _HookConfig(BaseConfig):
    count: Annotated[int, Arg()] = 3


class _RunConfig(BaseConfig):
    first: Annotated[CustomFunctionConfig, Arg()] = CustomFunctionConfig(path="default-hook")
    second: Annotated[CustomFunctionConfig | None, Arg()] = None


class TestTypedHookCli:
    def test_typed_hook_defaults_and_explicit_paths_remain_cli_strings(self) -> None:
        """Typed hook declarations expose path strings to argparse before config resolution."""
        parser = argparse.ArgumentParser()
        _RunConfig.add_arguments(parser)

        assert vars(parser.parse_args([])) == {"first": "default-hook", "second": None}
        assert vars(parser.parse_args(["--first", "replacement", "--second", "optional"])) == {
            "first": "replacement",
            "second": "optional",
        }

    def test_partial_argument_preparation_selects_the_schema_before_final_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prepared default hook registers its CLI options and receives their parsed values."""
        parser = argparse.ArgumentParser()
        _RunConfig.add_arguments(parser)
        monkeypatch.setattr("sys.argv", ["test", "--count", "7"])

        def prepare(args: argparse.Namespace) -> None:
            args.first = "prepared-hook"

        with function_registry.temporary("prepared-hook", SimpleNamespace(config_class=_HookConfig)):
            add_user_provided_function_arguments(parser, modify_args=prepare, config_class=_RunConfig)
            args = parser.parse_args()
            prepare(args)
            resolve_custom_function_configs(args, config_class=_RunConfig)

        assert args.first.path == "prepared-hook"
        assert args.first.config.count == 7
        assert args.second is None
        assert vars(args).keys() == {"first", "second"}
