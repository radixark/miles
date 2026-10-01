import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ConfigDict, SerializeAsAny, create_model, model_validator

from miles.utils.args.schema import BaseConfig, validate_complete_config
from miles.utils.function_registry import load_function
from miles.utils.workers.argv_utils import with_relax_parser_required_args, with_suppressed_parser_help


class CustomFunctionConfig(BaseConfig):
    model_config = ConfigDict(frozen=True)

    path: str
    config: SerializeAsAny[BaseConfig] | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_config(cls, values: Any) -> Any:
        if not isinstance(values, dict) or not isinstance(values.get("config"), dict):
            return values
        fn = load_function(values["path"])
        config_class = _compute_config_class(fn, path=values["path"])
        assert config_class is not None
        return values | {"config": validate_complete_config(config_class, values["config"])}

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return _restore_custom_function_config, (self.model_dump(),)


def add_user_provided_function_arguments(
    parser: argparse.ArgumentParser,
    *,
    modify_args: Callable[[argparse.Namespace], None],
) -> argparse.ArgumentParser:
    try:
        with with_relax_parser_required_args(parser), with_suppressed_parser_help(parser):
            args_partial, _ = parser.parse_known_args()
    except SystemExit:
        return parser

    modify_args(args_partial)
    infos = _compute_custom_function_field_infos(args_partial, partial=True)
    registered_paths: set[str] = set()
    registered_config_classes: set[type[BaseConfig]] = set()
    for info in infos:
        if info.path in registered_paths:
            continue
        registered_paths.add(info.path)
        if (config_class := info.config_class) is not None and config_class not in registered_config_classes:
            config_class.add_arguments(parser=parser)
            registered_config_classes.add(config_class)
    return parser


def resolve_custom_function_configs(args: argparse.Namespace) -> None:
    custom_arg_names: set[str] = set()
    for info in _compute_custom_function_field_infos(args):
        config = None
        if (config_class := info.config_class) is not None:
            config = config_class.model_validate(
                {
                    key: getattr(args, key) for key in config_class.model_fields if hasattr(args, key)
                }  # config-access-exempt: schema-selected fields
            )
            custom_arg_names.update(config_class.model_fields)
        setattr(args, info.name, CustomFunctionConfig(path=info.path, config=config))

    for name in custom_arg_names:
        if hasattr(args, name):  # config-access-exempt: schema-selected field
            delattr(args, name)


@dataclass(frozen=True)
class _CustomFunctionFieldInfo:
    name: str
    path: str
    fn: Any
    config_class: type[BaseConfig] | None


def _compute_custom_function_field_infos(
    args: argparse.Namespace, *, partial: bool = False
) -> list[_CustomFunctionFieldInfo]:
    from miles.utils.args.runtime import AllConfig

    paths = [
        (
            name,
            getattr(args, name, None) if partial else getattr(args, name),
        )  # config-access-exempt: schema-selected field
        for name, field in AllConfig.model_fields.items()
        if field.annotation in {CustomFunctionConfig, CustomFunctionConfig | None}
    ]
    infos = []
    for name, path in paths:
        if path is None:
            continue
        try:
            fn = load_function(path)
        except (ModuleNotFoundError, ValueError):
            if partial:
                continue
            raise
        if partial and fn is None:
            continue
        config_class = _compute_config_class(fn, path=path)
        infos.append(_CustomFunctionFieldInfo(name=name, path=path, fn=fn, config_class=config_class))
    return infos


def _compute_config_class(fn: Any, *, path: str) -> type[BaseConfig] | None:
    config_class = getattr(fn, "config_class", None)  # config-access-exempt: custom hook protocol discovery
    if config_class is not None:
        assert isinstance(config_class, type) and issubclass(
            config_class, BaseConfig
        ), f"{path}.config_class must inherit BaseConfig"
        return config_class

    add_arguments = getattr(fn, "add_arguments", None)  # config-access-exempt: legacy hook protocol discovery
    if callable(add_arguments):
        fn.config_class = _adapt_legacy_custom_config(add_arguments)
        return fn.config_class

    return None


def _adapt_legacy_custom_config(add_arguments: Callable[[argparse.ArgumentParser], Any]) -> type[BaseConfig]:
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)

    fields = {
        action.dest: (
            Any,
            ... if action.required or action.default == argparse.SUPPRESS else action.default,
        )
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
    }
    config_class = create_model("LegacyCustomFunctionConfig", __base__=BaseConfig, **fields)

    config_class.add_arguments = add_arguments
    return config_class


def _restore_custom_function_config(values: dict[str, Any]) -> CustomFunctionConfig:
    return CustomFunctionConfig.model_validate(values)
