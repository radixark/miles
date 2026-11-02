import argparse
import pickle
from types import SimpleNamespace
from typing import Annotated

import pytest

from miles.utils.args.custom_function import (
    CustomFunctionConfig,
    _adapt_legacy_custom_config,
    resolve_custom_function_configs,
    resolve_custom_function_overrides,
)
from miles.utils.args.schema import Arg, BaseConfig
from miles.utils.function_registry import function_registry


class _HookConfig(BaseConfig):
    threshold: Annotated[float, Arg()] = 0.5


class _RunConfig(BaseConfig):
    first: CustomFunctionConfig
    second: CustomFunctionConfig | None


class TestCustomFunctionResolution:
    def test_shared_hook_fields_are_removed_only_after_both_hooks_are_built(self) -> None:
        """Multiple hooks sharing a field must each receive it before the flat field is removed."""
        args = argparse.Namespace(first="hook", second="hook", threshold=0.8, unrelated=7)
        with function_registry.temporary("hook", SimpleNamespace(config_class=_HookConfig)):
            resolve_custom_function_configs(args, config_class=_RunConfig)

        assert args.first.config.threshold == 0.8
        assert args.second.config.threshold == 0.8
        assert vars(args).keys() == {"first", "second", "unrelated"}
        assert args.unrelated == 7

    def test_an_absent_optional_hook_remains_absent(self) -> None:
        """Optional hooks must not be replaced with empty path configurations."""
        args = argparse.Namespace(first="hook", second=None, threshold=0.8)
        with function_registry.temporary("hook", SimpleNamespace(config_class=_HookConfig)):
            resolve_custom_function_configs(args, config_class=_RunConfig)

        assert args.second is None
        assert args.first.path == "hook"

    def test_typed_hook_configuration_survives_json_and_pickle(self) -> None:
        """Transport resolves the hook's concrete schema instead of losing custom fields."""
        config = CustomFunctionConfig(path="hook", config=_HookConfig(threshold=0.7))
        with function_registry.temporary("hook", SimpleNamespace(config_class=_HookConfig)):
            restored = CustomFunctionConfig.model_validate_json(config.model_dump_json())
            unpickled = pickle.loads(pickle.dumps(config))

        assert type(restored.config) is _HookConfig
        assert restored.config.threshold == 0.7
        assert unpickled == restored

    def test_serialized_hooks_cannot_fill_missing_fields_from_defaults(self) -> None:
        """Incomplete hook payloads are rejected even when local schema defaults exist."""
        with function_registry.temporary("hook", SimpleNamespace(config_class=_HookConfig)):
            with pytest.raises(ValueError, match="Incomplete configuration.*threshold"):
                CustomFunctionConfig.model_validate({"path": "hook", "config": {}})

    def test_same_path_override_reuses_the_resolved_configuration(self) -> None:
        """A trainer override naming the same hook preserves already resolved custom values."""
        original = CustomFunctionConfig(path="hook", config=_HookConfig(threshold=0.7))
        args = argparse.Namespace(custom_model_provider_path=original)
        with function_registry.temporary("hook", SimpleNamespace(config_class=_HookConfig)):
            overrides = resolve_custom_function_overrides({"custom_model_provider_path": "hook"}, args=args)

        assert overrides["custom_model_provider_path"] is original
        assert args.custom_model_provider_path is original


class TestLegacyCustomArguments:
    def test_positional_callbacks_and_required_fields_keep_their_contract(self) -> None:
        """Legacy positional callbacks retain required fields and CLI value conversion."""

        def register(parser: argparse.ArgumentParser, /) -> argparse.ArgumentParser:
            parser.add_argument("--required-value", required=True, type=int)
            parser.add_argument("--optional-value", default="fallback")
            return parser

        config_class = _adapt_legacy_custom_config(register)
        parser = argparse.ArgumentParser()
        config_class.add_arguments(parser=parser)
        parsed = parser.parse_args(["--required-value", "9"])

        assert config_class.model_validate(vars(parsed)).model_dump() == {
            "required_value": 9,
            "optional_value": "fallback",
        }
        with pytest.raises(ValueError):
            config_class.model_validate({})

    def test_suppressed_defaults_cannot_be_silently_synthesized(self) -> None:
        """A legacy suppressed argument remains required in the transport schema."""

        def register(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--value", default=argparse.SUPPRESS)

        config_class = _adapt_legacy_custom_config(register)
        with pytest.raises(ValueError):
            config_class.model_validate({})
        assert config_class.model_validate({"value": "present"}).model_dump() == {"value": "present"}


class TestCustomArgumentRegistration:
    def test_shared_config_classes_are_registered_once_across_distinct_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Distinct hooks sharing a schema must not register duplicate option strings."""
        from miles.utils.args.custom_function import add_user_provided_function_arguments

        parser = argparse.ArgumentParser()
        parser.add_argument("--first", default="first-hook")
        parser.add_argument("--second", default="second-hook")
        monkeypatch.setattr("sys.argv", ["test", "--threshold", "0.9"])
        with function_registry.temporary("first-hook", SimpleNamespace(config_class=_HookConfig)):
            with function_registry.temporary("second-hook", SimpleNamespace(config_class=_HookConfig)):
                result = add_user_provided_function_arguments(parser, config_class=_RunConfig)

        assert result is parser
        assert parser.parse_args(["--threshold", "0.9"]).threshold == 0.9

    def test_partial_discovery_defers_missing_modules_until_final_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial parsing tolerates unavailable hooks while final resolution fails explicitly."""
        from miles.utils.args.custom_function import add_user_provided_function_arguments

        parser = argparse.ArgumentParser()
        parser.add_argument("--first", default="missing_test_hook_package.hook")
        parser.add_argument("--second", default=None)
        monkeypatch.setattr("sys.argv", ["test"])
        assert add_user_provided_function_arguments(parser, config_class=_RunConfig) is parser
        with pytest.raises(ModuleNotFoundError):
            resolve_custom_function_configs(parser.parse_args([]), config_class=_RunConfig)


class TestPerTrainerHookOverrides:
    def test_changed_path_uses_merged_custom_fields_without_mutating_either_input(self) -> None:
        """A trainer's replacement hook receives override values while the shared run stays unchanged."""
        original = CustomFunctionConfig(path="old-hook", config=_HookConfig(threshold=0.2))
        args = argparse.Namespace(custom_model_provider_path=original, threshold=0.3, lr=1e-6)
        overrides = {"custom_model_provider_path": "new-hook", "threshold": 0.9, "lr": 2e-6}
        before = overrides.copy()
        with function_registry.temporary("old-hook", SimpleNamespace(config_class=_HookConfig)):
            with function_registry.temporary("new-hook", SimpleNamespace(config_class=_HookConfig)):
                resolved = resolve_custom_function_overrides(overrides, args=args)

        replacement = resolved["custom_model_provider_path"]
        assert replacement.path == "new-hook"
        assert type(replacement.config) is _HookConfig
        assert replacement.config.threshold == 0.9
        assert resolved["lr"] == 2e-6
        assert overrides == before
        assert vars(args) == {"custom_model_provider_path": original, "threshold": 0.3, "lr": 1e-6}
        assert original.config.threshold == 0.2

    def test_none_explicitly_disables_a_previously_resolved_hook(self) -> None:
        """A trainer can disable an inherited hook without changing the shared resolved configuration."""
        original = CustomFunctionConfig(path="hook", config=_HookConfig(threshold=0.7))
        args = argparse.Namespace(custom_model_provider_path=original)
        overrides = {"custom_model_provider_path": None, "lr": 2e-6}

        resolved = resolve_custom_function_overrides(overrides, args=args)

        assert resolved == {"custom_model_provider_path": None, "lr": 2e-6}
        assert overrides == {"custom_model_provider_path": None, "lr": 2e-6}
        assert args.custom_model_provider_path is original

    @pytest.mark.parametrize("invalid_path", [7, {"path": "hook"}])
    def test_non_string_hook_overrides_are_rejected(self, invalid_path: object) -> None:
        """Malformed trainer hook paths fail explicitly rather than becoming arbitrary configurations."""
        args = argparse.Namespace(custom_model_provider_path=None)
        overrides = {"custom_model_provider_path": invalid_path}

        with pytest.raises(AssertionError, match="custom_model_provider_path.*must be a path string"):
            resolve_custom_function_overrides(overrides, args=args)

        assert overrides == {"custom_model_provider_path": invalid_path}
        assert args.custom_model_provider_path is None
