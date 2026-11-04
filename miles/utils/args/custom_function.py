import argparse
import copy
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

import yaml
from omegaconf.errors import OmegaConfBaseException
from pydantic import ConfigDict, SerializeAsAny, create_model, model_validator

from miles.utils.args.schema import BaseConfig, validate_complete_config
from miles.utils.function_registry import load_function
from miles.utils.rollout_data_config import load_rollout_data_config
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
    modify_args: Callable[[argparse.Namespace], None] | None = None,
    config_class: type[BaseConfig] | None = None,
) -> argparse.ArgumentParser:
    try:
        with with_relax_parser_required_args(parser), with_suppressed_parser_help(parser):
            args_partial, _ = parser.parse_known_args()
    except SystemExit:
        return parser

    if modify_args is not None:
        modify_args(args_partial)
    infos = _compute_custom_function_field_infos(args_partial, partial=True, config_class=config_class)
    if config_class is None:
        infos += _compute_rollout_dataset_function_infos(args_partial, partial=True)
    registered_paths: set[str] = set()
    registered_config_classes: set[type[BaseConfig]] = set()
    hook_option_strings: set[str] = set()
    for info in infos:
        if info.path in registered_paths:
            continue
        registered_paths.add(info.path)
        if (
            function_config_class := info.config_class
        ) is not None and function_config_class not in registered_config_classes:
            _add_function_arguments(
                parser, function_config_class, path=info.path, hook_option_strings=hook_option_strings
            )
            registered_config_classes.add(function_config_class)
    return parser


def _add_function_arguments(
    parser: argparse.ArgumentParser,
    config_class: type[BaseConfig],
    *,
    path: str,
    hook_option_strings: set[str] | None = None,
) -> None:
    """Register a hook's flags on the real parser, or skip a hook whose flags an earlier hook declared identically.

    Hook flags form one run-wide namespace: two hooks can share flags, e.g. a wrapper reusing agentic_tool_call's
    add_arguments, and then see the same value. The first registration already built their groups and defaults,
    so the second adds nothing. Per-dataset values belong in a flag of their own or the dataset's `agent` block.
    """
    hook_option_strings = set() if hook_option_strings is None else hook_option_strings
    # a scratch run only compares the hook's flags with already-declared ones
    scratch = argparse.ArgumentParser(add_help=False)
    config_class.add_arguments(parser=scratch)
    shared, new = [], []
    for action in scratch._actions:
        # only flags an earlier hook registered can be shared; a clash with a core flag keeps argparse's handling
        existing = {parser._option_string_actions[o] for o in action.option_strings if o in hook_option_strings}
        if not existing:
            new.append(action)
            continue
        (other,) = existing
        if (other.dest, other.default, other.type, other.nargs) != (
            action.dest,
            action.default,
            action.type,
            action.nargs,
        ):
            raise ValueError(
                f"{path} declares {'/'.join(action.option_strings)} differently from an earlier hook "
                f"(dest/default/type/nargs {other.dest, other.default, other.type, other.nargs} vs "
                f"{action.dest, action.default, action.type, action.nargs})"
            )
        shared.append(action)
    if not shared:
        before = set(parser._option_string_actions)
        config_class.add_arguments(parser=parser)
        hook_option_strings.update(set(parser._option_string_actions) - before)
    elif new:
        # registering only part of a hook would split its argument groups
        raise ValueError(
            f"{path} re-declares {', '.join('/'.join(a.option_strings) for a in shared)} from an earlier hook but "
            f"also adds {', '.join('/'.join(a.option_strings) or a.dest for a in new)}; a hook must share all of its "
            "flags with an earlier hook or none"
        )


def resolve_custom_function_configs(args: argparse.Namespace, *, config_class: type[BaseConfig] | None = None) -> None:
    custom_arg_names: set[str] = set()
    for info in _compute_custom_function_field_infos(args, config_class=config_class):
        resolved = _resolve_custom_function_config(info=info, args=args)
        if resolved.config is not None:
            custom_arg_names.update(type(resolved.config).model_fields)
        setattr(args, info.name, resolved)

    if config_class is None:
        # functions a --rollout-data-yaml-config dataset names; no args field holds them
        args.rollout_dataset_functions = {}
        args.rollout_dataset_agent_functions = []
        for info in _compute_rollout_dataset_function_infos(args, partial=False):
            resolved = _resolve_custom_function_config(info=info, args=args)
            if resolved.config is not None:
                custom_arg_names.update(type(resolved.config).model_fields)
            args.rollout_dataset_functions[info.path] = resolved
            if (
                info.dataset_field == "custom_agent_function_path"
                and resolved not in args.rollout_dataset_agent_functions
            ):
                args.rollout_dataset_agent_functions.append(resolved)

    for name in custom_arg_names:
        if hasattr(args, name):  # config-access-exempt: schema-selected field
            delattr(args, name)


def resolve_custom_function_overrides(overrides: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    resolved = overrides.copy()
    values = copy.deepcopy(argparse.Namespace(**(vars(args) | overrides)))
    for info in _compute_custom_function_field_infos(argparse.Namespace(**overrides), field_names=overrides.keys()):
        original = vars(args)[info.name]
        resolved[info.name] = (
            original
            if isinstance(original, CustomFunctionConfig) and original.path == info.path
            else _resolve_custom_function_config(info=info, args=values)
        )
    return resolved


@dataclass(frozen=True)
class _CustomFunctionFieldInfo:
    name: str
    path: str
    fn: Any
    config_class: type[BaseConfig] | None
    # the --rollout-data-yaml-config field that named this hook, e.g. "custom_agent_function_path"; None for flag hooks
    dataset_field: str | None = None


def _resolve_custom_function_config(
    *, info: _CustomFunctionFieldInfo, args: argparse.Namespace
) -> CustomFunctionConfig:
    config = None
    if (config_class := info.config_class) is not None:
        config = config_class.model_validate(
            {
                key: getattr(args, key) for key in config_class.model_fields if hasattr(args, key)
            }  # config-access-exempt: schema-selected fields
        )
    return CustomFunctionConfig(path=info.path, config=config)


def _compute_custom_function_field_infos(
    args: argparse.Namespace,
    *,
    partial: bool = False,
    config_class: type[BaseConfig] | None = None,
    field_names: Collection[str] | None = None,
) -> list[_CustomFunctionFieldInfo]:
    from miles.utils.args.runtime import AllConfig

    paths = [
        (
            name,
            getattr(args, name, None) if partial else getattr(args, name),
        )  # config-access-exempt: schema-selected field
        for name, field in (config_class or AllConfig).model_fields.items()
        if field.annotation in {CustomFunctionConfig, CustomFunctionConfig | None}
        and (field_names is None or name in field_names)
    ]
    infos = []
    for name, path in paths:
        if path is None:
            continue
        assert isinstance(path, str), f"Custom function field {name!r} must be a path string"
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


_ROLLOUT_DATASET_FUNCTION_FIELDS = ("custom_generate_function_path", "custom_agent_function_path")


def _compute_rollout_dataset_function_infos(
    args: argparse.Namespace, *, partial: bool
) -> list[_CustomFunctionFieldInfo]:
    """Hooks a --rollout-data-yaml-config dataset names, so their own flags are registered and resolved."""
    value = getattr(args, "rollout_data_yaml_config", None)  # config-access-exempt: absent from partial parses
    if value is None:
        return []
    try:
        datasets = load_rollout_data_config(value)
    except (OSError, ValueError, yaml.YAMLError, OmegaConfBaseException):
        # a missing file, invalid YAML or failed schema is reported by startup validation, not the pre-parse
        if partial:
            return []
        raise
    infos = []
    for dataset in datasets:
        for field_name in _ROLLOUT_DATASET_FUNCTION_FIELDS:
            if (path := getattr(dataset, field_name)) is None:  # config-access-exempt: fixed field names
                continue
            try:
                fn = load_function(path)
            except (ModuleNotFoundError, ValueError):
                if partial:
                    continue
                raise
            if partial and fn is None:
                continue
            infos.append(
                _CustomFunctionFieldInfo(
                    name=f"rollout_dataset:{dataset.name}:{field_name}",
                    path=path,
                    fn=fn,
                    config_class=_compute_config_class(fn, path=path),
                    dataset_field=field_name,
                )
            )
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

    def _add_arguments(parser: argparse.ArgumentParser) -> Any:
        return add_arguments(parser)

    config_class.add_arguments = staticmethod(_add_arguments)
    return config_class


def _restore_custom_function_config(values: dict[str, Any]) -> CustomFunctionConfig:
    return CustomFunctionConfig.model_validate(values)
