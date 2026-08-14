import argparse

from sglang.srt.server_args import ServerArgs

from miles.utils.workers.argv_utils import render_cli_argv


def server_args_to_argv(server_args: ServerArgs) -> list[str]:
    return render_cli_argv(
        server_args,
        make_parser=_make_cli_parser,
        from_parsed=ServerArgs.from_cli_args,
        required_argv=_required_argv(server_args),
    )


def parse_server_args_argv(argv: list[str]) -> ServerArgs:
    return ServerArgs.from_cli_args(_make_cli_parser().parse_args(argv))


def _required_argv(server_args: ServerArgs) -> list[str]:
    argv = [
        "--model-path",
        server_args.model_path,
        "--host",
        server_args.host,
        "--port",
        str(server_args.port),
    ]

    if server_args.disaggregation_mode != "null":
        argv.extend(["--disaggregation-mode", server_args.disaggregation_mode])

    return argv


def _make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser
