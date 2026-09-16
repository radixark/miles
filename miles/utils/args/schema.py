import argparse
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo

from miles.utils.pydantic_utils import StrictBaseModel

_ConfigT = TypeVar("_ConfigT", bound=BaseModel)
A = Annotated


_UNSET = object()


@dataclass(frozen=True)
class Arg:
    help: str | None = None
    choices: list[Any] | tuple[Any, ...] | None = None
    aliases: tuple[str, ...] = ()
    cli_name: str | None = None
    type_parser: Any = _UNSET
    nargs: str | int | None = None
    required: bool = False
    action: str | type[argparse.Action] | None = None
    const: Any = _UNSET
    metavar: str | tuple[str, ...] | None = None
    reset: bool = False
    existing: bool = False


# Adapted from sglang/srt/arg_groups/arg_utils.py:add_cli_args_from_dataclass.
class BaseConfig(StrictBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        owned_names = cls.__dict__.get("__annotations__", {})
        for name, field in cls.model_fields.items():
            if name not in owned_names:
                continue
            if (argument := _argument_metadata(field)) is not None:
                _add_argument(
                    parser=parser,
                    name=name,
                    annotation=field.annotation,
                    field=field,
                    argument=argument,
                )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        values = vars(args)
        missing = cls.model_fields.keys() - values.keys()
        if missing:
            raise ValueError(f"Missing parsed fields for {cls.__name__}: {sorted(missing)}")
        return {name: values[name] for name in cls.model_fields}


def add_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    config_classes: Iterable[type[BaseConfig]],
) -> None:
    for config_class in _collect_traits(config_classes=config_classes):
        if parser._registry_get("miles_config_traits", config_class, False):
            continue
        if "add_arguments" in config_class.__dict__:
            config_class.add_arguments(parser=parser)
        else:
            BaseConfig.add_arguments.__func__(config_class, parser=parser)
        parser.register("miles_config_traits", config_class, True)


def validate_argument_compatibility(
    *, parser: argparse.ArgumentParser, name: str, action: argparse.Action, kwargs: Mapping[str, Any]
) -> None:
    if "action" in kwargs or "type" in kwargs:
        requested_action = kwargs.get("action")
        expected_action = parser._registry_get("action", requested_action, requested_action)
        if type(action) is not expected_action:
            raise ValueError(f"Argument {name} has incompatible action: {type(action).__name__}")

    properties = {
        "type": action.type,
        "nargs": action.nargs,
        "choices": action.choices,
        "required": action.required,
        "dest": action.dest,
        "const": action.const,
    }
    expected = {key: kwargs[key] for key in properties if key in kwargs}
    if "nargs" not in expected:
        if kwargs.get("action") in {"store_true", "store_false", "store_const", "append_const", "count"}:
            expected["nargs"] = 0
        elif "type" in kwargs and kwargs.get("action") in {None, "store", "append", "extend"}:
            expected["nargs"] = None
    for key, value in expected.items():
        if properties[key] != value:
            raise ValueError(f"Argument {name} has incompatible {key}: expected {value!r}, got {properties[key]!r}")


def validate_complete_config(config_class: type[_ConfigT], payload: Mapping[str, Any]) -> _ConfigT:
    config = config_class.model_validate(payload)
    _validate_complete_value(value=config, path=config_class.__name__)
    return config


def _collect_traits(*, config_classes: Iterable[type[BaseConfig]]) -> list[type[BaseConfig]]:
    traits: list[type[BaseConfig]] = []
    seen: set[type[BaseConfig]] = set()

    def visit(config_class: type[BaseConfig]) -> None:
        if config_class in seen or config_class is BaseConfig:
            return
        seen.add(config_class)
        for base in config_class.__bases__:
            if issubclass(base, BaseConfig):
                visit(base)
        owned_names = config_class.__dict__.get("__annotations__", {})
        owns_arguments = any(
            _argument_metadata(field) is not None
            for name, field in config_class.model_fields.items()
            if name in owned_names
        )
        if owns_arguments or "add_arguments" in config_class.__dict__:
            traits.append(config_class)

    for config_class in config_classes:
        visit(config_class)
    return traits


def _argument_metadata(field: FieldInfo) -> Arg | None:
    arguments = [metadata for metadata in field.metadata if isinstance(metadata, Arg)]
    if len(arguments) > 1:
        raise ValueError("Multiple argument declarations in one annotation")
    return arguments[0] if arguments else None


def _add_argument(
    *,
    parser: argparse.ArgumentParser,
    name: str,
    annotation: Any,
    field: FieldInfo,
    argument: Arg,
) -> None:
    flags = (argument.cli_name or "--" + name.replace("_", "-"), *argument.aliases)
    if argument.reset and len(flags) != 1:
        raise ValueError("Reset arguments require exactly one argument name")
    if argument.existing and argument.reset:
        raise ValueError("Existing arguments cannot reset defaults")
    if argument.existing and not field.is_required():
        raise ValueError("Existing arguments cannot declare defaults")
    kwargs = _argument_kwargs(name=name, annotation=annotation, field=field, argument=argument)
    if argument.existing:
        action = _find_existing_argument(parser=parser, flags=flags)
        validate_argument_compatibility(parser=parser, name=flags[0], action=action, kwargs=kwargs)
        return
    if argument.reset:
        from miles.utils.arguments import reset_arg

        reset_arg(parser=parser, name=flags[0], **kwargs)
    else:
        parser.add_argument(*flags, **kwargs)


def _argument_kwargs(*, name: str, annotation: Any, field: FieldInfo, argument: Arg) -> dict[str, Any]:
    from miles.utils.args.custom_function import CustomFunctionConfig

    kwargs = {
        key: value
        for key, value in vars(argument).items()
        if key not in {"aliases", "cli_name", "type_parser", "reset", "existing"}
        and value is not None
        and value is not _UNSET
    }
    if (argument.cli_name or "--" + name.replace("_", "-")).startswith("-"):
        kwargs["dest"] = name
    else:
        kwargs.pop("required", None)

    if not field.is_required():
        kwargs["default"] = deepcopy(field.get_default(call_default_factory=True))
        if isinstance(kwargs["default"], CustomFunctionConfig):
            kwargs["default"] = kwargs["default"].path

    annotation = _unwrap_optional(annotation)
    if argument.type_parser is not _UNSET:
        kwargs["type"] = argument.type_parser
    elif annotation is CustomFunctionConfig:
        kwargs["type"] = str
    elif argument.action is None and annotation is bool:
        kwargs["action"] = "store_true"
    elif argument.action in {None, "store", "append", "extend"}:
        kwargs["type"] = _infer_type_parser(annotation)
    if argument.const is not _UNSET:
        kwargs["const"] = argument.const
    return kwargs


def _unwrap_optional(annotation: Any) -> Any:
    if get_origin(annotation) in {Union, UnionType}:
        members = [member for member in get_args(annotation) if member is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


def _infer_type_parser(annotation: Any) -> Callable[[str], Any]:
    if annotation not in {str, int, float, bool}:
        raise TypeError(f"Explicit type_parser is required for {annotation!r}")
    return annotation


def _find_existing_argument(*, parser: argparse.ArgumentParser, flags: tuple[str, ...]) -> argparse.Action:
    for action in parser._actions:
        if all(flag in action.option_strings for flag in flags):
            return action
        if not action.option_strings and flags == (action.dest,):
            return action
    raise ValueError(f"Existing argument was not registered: {flags}")


def _validate_complete_value(*, value: Any, path: str) -> None:
    if isinstance(value, BaseModel):
        fields = type(value).model_fields
        missing = {name for name, field in fields.items() if field.exclude is not True} - value.model_fields_set
        if missing:
            raise ValueError(f"Incomplete configuration {path}: missing fields {sorted(missing)}")
        for name, field in fields.items():
            if field.exclude is not True:
                _validate_complete_value(value=value.__getattribute__(name), path=f"{path}.{name}")
    elif isinstance(value, Mapping):
        for name, item in value.items():
            _validate_complete_value(value=item, path=f"{path}[{name!r}]")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_complete_value(value=item, path=f"{path}[{index}]")
