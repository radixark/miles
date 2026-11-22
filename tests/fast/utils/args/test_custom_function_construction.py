import argparse
from types import SimpleNamespace
from typing import Annotated

import pytest

from miles.utils.args.custom_function import CustomFunctionConfig, resolve_custom_function_configs
from miles.utils.args.schema import Arg, BaseConfig
from miles.utils.function_registry import function_registry


class _HookConfig(BaseConfig):
    threshold: Annotated[float, Arg()] = 0.5


class _RunConfig(BaseConfig):
    hook: CustomFunctionConfig


class TestCustomFunctionConstruction:
    def test_schema_less_hook_preserves_unrelated_flat_arguments(self) -> None:
        """A hook without a schema keeps its path without consuming another component's fields."""
        args = argparse.Namespace(hook="schema-less", threshold=0.8)
        with function_registry.temporary("schema-less", SimpleNamespace()):
            resolve_custom_function_configs(args, config_class=_RunConfig)

        assert args.hook.path == "schema-less"
        assert args.hook.config is None
        assert args.threshold == 0.8

    def test_invalid_schema_value_fails_before_flat_arguments_are_removed(self) -> None:
        """Invalid custom values cannot silently fall back to defaults or erase the original inputs."""
        args = argparse.Namespace(hook="typed-hook", threshold="invalid")
        original = vars(args).copy()
        with function_registry.temporary("typed-hook", SimpleNamespace(config_class=_HookConfig)):
            with pytest.raises(ValueError, match="threshold"):
                resolve_custom_function_configs(args, config_class=_RunConfig)

        assert vars(args) == original

    def test_omitted_schema_value_uses_the_declared_default(self) -> None:
        """Constructing a hook uses its schema defaults when the flat parser omitted that field."""
        args = argparse.Namespace(hook="typed-hook")
        with function_registry.temporary("typed-hook", SimpleNamespace(config_class=_HookConfig)):
            resolve_custom_function_configs(args, config_class=_RunConfig)

        assert args.hook.path == "typed-hook"
        assert type(args.hook.config) is _HookConfig
        assert args.hook.config.threshold == 0.5
