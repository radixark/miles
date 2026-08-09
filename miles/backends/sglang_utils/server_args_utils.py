import argparse
import dataclasses

from sglang.srt.server_args import ServerArgs

from miles.utils.workers.argv_utils import _actions_by_dest, _render_action_argv, _resolve_action

_IDENTITY_FIELDS = ("model_path", "host", "port")

_UNCOMPARED_FIELDS = frozenset({"random_seed"})


def server_args_to_argv(server_args_dict: dict) -> list[str]:
    parser = _make_cli_parser()
    actions_by_dest = _actions_by_dest(parser)

    def render(name: str, value: object) -> list[str]:
        action = _resolve_action(actions_by_dest, field_name=name, dest_prefix="", field_to_dest={})
        return _render_action_argv(action, value)

    required_argv = [item for name in _IDENTITY_FIELDS for item in render(name, server_args_dict[name])]
    cli_defaults = parser.parse_args(required_argv)

    argv = list(required_argv)
    for name, value in server_args_dict.items():
        if name in _IDENTITY_FIELDS or value == getattr(cli_defaults, name):
            continue
        argv.extend(render(name, value))

    mismatch = _describe_mismatch(parse_server_args_argv(argv), ServerArgs(**server_args_dict))
    assert not mismatch, f"sglang argv roundtrip mismatch on {mismatch}"
    return argv


def parse_server_args_argv(argv: list[str]) -> ServerArgs:
    return ServerArgs.from_cli_args(_make_cli_parser().parse_args(argv))


def _describe_mismatch(parsed: ServerArgs, wanted: ServerArgs) -> str:
    return ", ".join(
        f"{field.name}: parsed {getattr(parsed, field.name)!r} != wanted {getattr(wanted, field.name)!r}"
        for field in dataclasses.fields(wanted)
        if field.name not in _UNCOMPARED_FIELDS and getattr(parsed, field.name) != getattr(wanted, field.name)
    )


def _make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser
