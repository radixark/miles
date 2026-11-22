import argparse
import pickle
from types import SimpleNamespace

import pytest

from miles.utils.args.custom_function import (
    CustomFunctionConfig,
    add_user_provided_function_arguments,
    resolve_custom_function_configs,
)
from miles.utils.args.schema import BaseConfig
from miles.utils.function_registry import function_registry


class _RunConfig(BaseConfig):
    first: CustomFunctionConfig
    second: CustomFunctionConfig | None = None


def _register_legacy_options(parser: argparse.ArgumentParser, /) -> argparse.ArgumentParser:
    parser.add_argument("--required-count", type=int, required=True)
    parser.add_argument("--labels", nargs="+", default=["default"])
    return parser


class TestLegacyHookTransport:
    def test_shared_legacy_hook_registers_once_and_restores_its_dynamic_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy registration and transport preserve custom values after flat ownership is removed."""
        hook = SimpleNamespace(add_arguments=_register_legacy_options)
        parser = argparse.ArgumentParser()
        parser.add_argument("--first", default="legacy-hook")
        parser.add_argument("--second", default="legacy-hook")
        monkeypatch.setattr("sys.argv", ["test", "--required-count", "9", "--labels", "a", "b"])

        with function_registry.temporary("legacy-hook", hook):
            add_user_provided_function_arguments(parser, config_class=_RunConfig)
            args = parser.parse_args()
            resolve_custom_function_configs(args, config_class=_RunConfig)
            restored = CustomFunctionConfig.model_validate_json(args.first.model_dump_json())
            unpickled = pickle.loads(pickle.dumps(args.first))

        assert vars(args).keys() == {"first", "second"}
        assert args.first.config.model_dump() == {"required_count": 9, "labels": ["a", "b"]}
        assert args.second.config.model_dump() == args.first.config.model_dump()
        assert type(restored.config) is hook.config_class
        assert type(unpickled.config) is hook.config_class
        assert restored == unpickled == args.first

    def test_malformed_declared_hook_schema_fails_before_arguments_are_removed(self) -> None:
        """An invalid hook schema cannot silently become a legacy configuration."""
        args = argparse.Namespace(first="bad-hook", second=None, required_count=9)
        hook = SimpleNamespace(config_class=dict, add_arguments=_register_legacy_options)

        with function_registry.temporary("bad-hook", hook):
            with pytest.raises(AssertionError, match="bad-hook.config_class must inherit BaseConfig"):
                resolve_custom_function_configs(args, config_class=_RunConfig)

        assert vars(args) == {"first": "bad-hook", "second": None, "required_count": 9}
