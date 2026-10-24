import argparse

from sglang_router.launch_router import RouterArgs

from miles.utils.args.schema import A, Arg, BaseConfig


class RouterConfig(BaseConfig):
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
