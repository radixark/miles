import argparse
import dataclasses

from sglang.srt.server_args import ServerArgs

from miles.utils.workers.argv_utils import render_cli_option

_IDENTITY_FIELDS = ("model_path", "host", "port")

# Each engine deliberately draws its own seed, so the launched process is expected
# to differ here from whatever this process happened to draw while checking.
_UNCOMPARED_FIELDS = frozenset({"random_seed"})


def server_args_to_argv(server_args_dict: dict) -> list[str]:
    """Render the launch command line from the arguments as sglang has not yet seen them.

    Rendering a constructed ServerArgs instead would spell out values its __post_init__
    already derived, and some of those derivations are not idempotent -- under DP
    attention sglang divides chunked_prefill_size by dp_size and scales
    schedule_conservativeness every time it runs -- so the launched process would apply
    them a second time on top of the ones baked into the command line.
    """
    required_argv = [item for name in _IDENTITY_FIELDS for item in render_cli_option(name, server_args_dict[name])]
    cli_defaults = _make_cli_parser().parse_args(required_argv)

    argv = list(required_argv)
    for name, value in server_args_dict.items():
        if name in _IDENTITY_FIELDS or value == getattr(cli_defaults, name):
            continue
        argv.extend(render_cli_option(name, value))

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
