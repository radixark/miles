import argparse
import contextlib
import dataclasses
import json
import shlex
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple, TypeVar

from miles.utils.pydantic_utils import FrozenStrictBaseModel

CONFIG_JSON_FLAG = "--config-json"

_INTERPRETER_FLAGS_TAKING_A_VALUE = frozenset({"-X", "-W", "-Q", "--check-hash-based-pycs"})
_INTERPRETER_FLAGS_ENDING_THE_PREFIX = frozenset({"-c", "-m", "-", "--"})

_ConfigT = TypeVar("_ConfigT", bound=FrozenStrictBaseModel)
_ArgsT = TypeVar("_ArgsT")


def python_argv_prefix() -> list[str]:
    prefix = [sys.executable]
    tokens = iter(sys.orig_argv[1:])
    for token in tokens:
        if not token.startswith("-") or token in _INTERPRETER_FLAGS_ENDING_THE_PREFIX:
            break
        prefix.append(token)
        if token in _INTERPRETER_FLAGS_TAKING_A_VALUE:
            prefix.append(next(tokens))
    return prefix


# ==================== config argv ====================


def config_to_argv(config: FrozenStrictBaseModel) -> list[str]:
    argv = [CONFIG_JSON_FLAG, config.model_dump_json()]

    parsed = parse_config_argv(type(config), argv)
    assert parsed == config, f"config argv roundtrip mismatch: {parsed!r} != {config!r}"
    return argv


def parse_config_argv(config_cls: type[_ConfigT], argv: list[str] | None) -> _ConfigT:
    parser = argparse.ArgumentParser()
    parser.add_argument(CONFIG_JSON_FLAG, required=True)
    args = parser.parse_args(argv)
    return config_cls.model_validate_json(args.config_json)


def dataclass_to_values(args_obj: object) -> dict[str, object]:
    return {field.name: getattr(args_obj, field.name) for field in dataclasses.fields(args_obj)}


def render_cli_argv(
    input_values: Mapping[str, object],
    *,
    expected_obj: _ArgsT,
    make_parser: Callable[[], argparse.ArgumentParser],
    from_parsed: Callable[[argparse.Namespace], _ArgsT],
    always_render_fields: Sequence[str] = (),
    field_to_dest: Mapping[str, str] | None = None,
    uncompared_fields: frozenset[str] = frozenset(),
) -> list[str]:
    actions_by_dest = _actions_by_dest(make_parser())
    field_to_dest = field_to_dest or {}

    def action_of(field_name: str) -> argparse.Action:
        return _resolve_action(actions_by_dest, field_name=field_name, field_to_dest=field_to_dest or {})

    def render(field_name: str, value: object) -> list[str]:
        return _render_action_argv(action_of(field_name), value)

    argv = [
        token
        for name in always_render_fields
        for token in render(
            name,
            (
                input_values[name]
                if name in input_values and input_values[name] is not None
                else getattr(expected_obj, name)
            ),
        )
    ]
    for name, value in input_values.items():
        if name in always_render_fields or value is None:
            continue
        action = action_of(name)
        if not _is_renderable(action, value):
            continue
        if value == action.default:
            continue
        argv.extend(render(name, value))

    parsed = from_parsed(make_parser().parse_args(argv))
    mismatch = _describe_mismatch(parsed, expected_obj, uncompared_fields=uncompared_fields)
    assert not mismatch, f"cli argv roundtrip mismatch on {mismatch}"
    return argv


def _describe_mismatch(parsed: _ArgsT, wanted: _ArgsT, *, uncompared_fields: frozenset[str]) -> str:
    return ", ".join(
        f"{field.name}: parsed {getattr(parsed, field.name)!r} != wanted {getattr(wanted, field.name)!r}"
        for field in dataclasses.fields(wanted)
        if field.name not in uncompared_fields and getattr(parsed, field.name) != getattr(wanted, field.name)
    )


def _actions_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions_by_dest: dict[str, argparse.Action] = {}
    for action in parser._actions:
        actions_by_dest.setdefault(action.dest, action)
    return actions_by_dest


def _resolve_action(
    actions_by_dest: dict[str, argparse.Action],
    *,
    field_name: str,
    field_to_dest: Mapping[str, str],
) -> argparse.Action:
    dest = field_to_dest.get(field_name, field_name)
    action = actions_by_dest.get(dest)
    if action is not None and action.option_strings:
        return action

    raise AssertionError(
        f"{field_name!r} cannot be rendered: the parser registers no option for dest {dest!r}. "
        f"Add an entry to field_to_dest, or pass the value through the native passthrough path."
    )


def _is_renderable(action: argparse.Action, value: object) -> bool:
    if value is None:
        return False
    if isinstance(action, argparse.BooleanOptionalAction):
        return True
    return action.nargs != 0 or value == action.const


def _render_action_argv(action: argparse.Action, value: object) -> list[str]:
    if isinstance(action, argparse.BooleanOptionalAction):
        return [_boolean_option_string(action, value=bool(value))]

    if action.nargs == 0:
        if value == action.default:
            return []
        flag = _long_option_string(action)
        assert (
            value == action.const
        ), f"{flag} cannot be rendered: the CLI only has a flag for {action.const!r}, not {value!r}"
        return [flag]

    flag = _long_option_string(action)

    if isinstance(action, argparse._AppendAction):
        argv: list[str] = []
        for item in value:
            argv.append(flag)
            argv.extend(_scalar_tokens(item))
        return argv

    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        if isinstance(value, dict):
            return [flag, *(f"{key}={item}" for key, item in value.items())]
        return [flag, *(str(item) for item in value)]

    if isinstance(value, dict | list | tuple):
        return [flag, json.dumps(value)]

    return [flag, str(value)]


def _scalar_tokens(item: object) -> list[str]:
    if isinstance(item, list | tuple):
        return [str(element) for element in item]
    return [str(item)]


def _long_option_string(action: argparse.Action) -> str:
    long_options = [option for option in action.option_strings if option.startswith("--")]
    return long_options[0] if long_options else action.option_strings[0]


def _boolean_option_string(action: argparse.Action, *, value: bool) -> str:
    negative = [option for option in action.option_strings if option.startswith("--no-")]
    positive = [option for option in action.option_strings if not option.startswith("--no-")]
    if value:
        assert positive, f"{action.dest!r} cannot be rendered: no positive option string"
        return positive[0]
    assert negative, f"{action.dest!r} cannot be rendered: no negative option string"
    return negative[0]


# ==================== parser reflection ====================


def parse_declared_args(text: str, *, parser: argparse.ArgumentParser) -> dict[str, object]:
    tokens = shlex.split(text)
    with with_relax_parser_required_args(parser):
        namespace, unknown = parser.parse_known_args(tokens)
    assert not unknown, f"the argument parser does not declare {unknown} of {text!r}"

    action_by_option_string = parser._option_string_actions
    dests = []
    for token in tokens:
        if not token.startswith("--"):
            continue
        assert token in action_by_option_string, f"the argument parser does not declare {token!r}"
        dests.append(action_by_option_string[token].dest)
    return {dest: getattr(namespace, dest) for dest in dests}


def coerce_dict_to_args(
    values: Mapping[str, Any], *, parser: argparse.ArgumentParser, allowed_names: frozenset[str], context: str
) -> dict[str, Any]:
    dest_of_option_name = _compute_dest_of_option_names(parser)
    arg_specs = _compute_arg_specs(parser)
    allowed_dests = frozenset(dest_of_option_name.get(name, name) for name in allowed_names)
    return {
        (dest := dest_of_option_name.get(name, name)): _coerce_value(
            value, dest=dest, spec=arg_specs.get(dest), allowed_dests=allowed_dests, context=context
        )
        for name, value in values.items()
    }


def _coerce_value(
    value: Any, *, dest: str, spec: "_ArgSpec | None", allowed_dests: frozenset[str], context: str
) -> Any:
    assert dest in allowed_dests, (
        f"{context} sets {dest!r}, which it may not override; only these are allowed: {sorted(allowed_dests)}. "
        f"Everything else is read from the base command line, so setting it here would be silently ignored"
    )
    assert value is not None, f"{context} sets {dest!r} with no value"
    assert spec is not None, (
        f"{dest!r} is allowed, but the argument parser declares no such argument, so the value {value!r} "
        f"of {context} cannot be typed"
    )

    if spec.type is bool:
        assert isinstance(value, bool), f"{context} sets {dest!r} to {value!r}, which is not a boolean"
        return value

    assert not isinstance(value, bool) and isinstance(
        value, (int, float, str)
    ), f"{context} sets {dest!r} to {value!r}, which is not a {spec.type.__name__}"
    coerced = _coerce_scalar(value, dest=dest, spec=spec, context=context)
    assert (
        spec.choices is None or coerced in spec.choices
    ), f"{context} sets {dest!r} to {value!r}, but the command line only accepts {list(spec.choices)}"
    return coerced


def _coerce_scalar(value: int | float | str, *, dest: str, spec: "_ArgSpec", context: str) -> Any:
    try:
        return spec.type(str(value))
    except ValueError as exception:
        raise AssertionError(
            f"{context} sets {dest!r} to {value!r}, which the command line would reject: {exception}"
        ) from exception


class _ArgSpec(NamedTuple):
    dest: str
    type: type
    choices: tuple[Any, ...] | None


def _compute_arg_specs(parser: argparse.ArgumentParser) -> dict[str, _ArgSpec]:
    return {action.dest: _compute_arg_spec(action) for action in parser._actions}


def _compute_dest_of_option_names(parser: argparse.ArgumentParser) -> dict[str, str]:
    return {
        option.removeprefix("--").replace("-", "_"): action.dest
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }


def _compute_arg_spec(action: argparse.Action) -> _ArgSpec:
    choices = None if action.choices is None else tuple(action.choices)
    return _ArgSpec(dest=action.dest, type=_compute_arg_type(action), choices=choices)


@contextlib.contextmanager
def with_relax_parser_required_args(parser: argparse.ArgumentParser) -> Iterator[None]:
    required = [action for action in parser._actions if action.required]
    for action in required:
        action.required = False
    try:
        yield
    finally:
        for action in required:
            action.required = True


def _compute_arg_type(action: argparse.Action) -> type:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        return bool
    if action.type is None:
        return str
    return action.type
