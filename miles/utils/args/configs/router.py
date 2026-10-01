import argparse
from typing import Any

from sglang_router.launch_router import RouterArgs

from miles.utils.args.schema import A, Arg, BaseConfig


_ROUTER_DEST_PREFIX = "router_"


class RouterConfig(BaseConfig):
    router_args: dict[str, Any]

    use_miles_router: A[
        bool,
        Arg(help="Whether to use MilesRouter for text-based routing instead of SGLang token-based routing"),
    ] = False
    miles_router_timeout: A[float | None, Arg(help="Timeout for MilesRouter HTTP requests in seconds.")] = None
    miles_router_max_connections: A[int | None, Arg(help="Max connections for MilesRouter HTTP client.")] = None
    miles_router_health_check_failure_threshold: A[
        int,
        Arg(help="Number of consecutive failures before marking a worker as unhealthy."),
    ] = 3

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        parser = _make_prefixed_cli_parser()
        values = vars(args)
        return {
            "router_args": {
                action.dest.removeprefix(_ROUTER_DEST_PREFIX): values[action.dest] for action in parser._actions
            }
        }


def _make_prefixed_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    return parser
