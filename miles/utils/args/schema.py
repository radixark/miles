import argparse
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import ConfigDict
from pydantic.fields import FieldInfo

from miles.utils.pydantic_utils import StrictBaseModel

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


# Adapted from sglang/srt/arg_groups/arg_utils.py:add_cli_args_from_dataclass.
class BaseConfig(StrictBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        for name, field in cls.model_fields.items():
            if (argument := _argument_metadata(field)) is not None:
                _add_argument(
                    parser=parser,
                    name=name,
                    annotation=field.annotation,
                    field=field,
                    argument=argument,
                )


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
    kwargs = _argument_kwargs(name=name, annotation=annotation, field=field, argument=argument)
    parser.add_argument(*flags, **kwargs)


def _argument_kwargs(*, name: str, annotation: Any, field: FieldInfo, argument: Arg) -> dict[str, Any]:
    kwargs = {
        key: value
        for key, value in vars(argument).items()
        if key not in {"aliases", "cli_name", "type_parser"} and value is not None and value is not _UNSET
    }
    if (argument.cli_name or "--" + name.replace("_", "-")).startswith("-"):
        kwargs["dest"] = name
    else:
        kwargs.pop("required", None)

    if not field.is_required():
        kwargs["default"] = deepcopy(field.get_default(call_default_factory=True))

    annotation = _unwrap_optional(annotation)
    if argument.type_parser is not _UNSET:
        kwargs["type"] = argument.type_parser
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
