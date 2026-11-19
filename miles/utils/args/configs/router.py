import argparse
from typing import Any, ClassVar

try:
    from sglang_router.launch_router import RouterArgs
except ImportError:
    RouterArgs = None  # type: ignore[assignment,misc]

from miles.router.backend import RouterKind
from miles.router.dynamo.config import DynamoConfig
from miles.utils.args.schema import A, Arg, BaseConfig

_ROUTER_DEST_PREFIX = "router_"


def _add_router_cli_args(parser: argparse.ArgumentParser) -> None:
    if RouterArgs is not None:
        RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
        return
    # Minimal fallback flags when sglang_router is not installed (e.g. TPU runs with --router miles).
    parser.add_argument("--router-policy", dest="router_policy", type=str, default="cache_aware")
    parser.add_argument("--router-prefill-policy", dest="router_prefill_policy", type=str, default=None)
    parser.add_argument("--router-decode-policy", dest="router_decode_policy", type=str, default=None)
    parser.add_argument("--router-dp-aware", dest="router_dp_aware", action="store_true", default=False)
    parser.add_argument("--router-assignment-mode", dest="router_assignment_mode", type=str, default="random")
    parser.add_argument(
        "--router-disable-health-check", dest="router_disable_health_check", action="store_true", default=False
    )


# TODO: use all sglang router arguments with `--sglang-router` prefix
class RouterConfig(BaseConfig):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset(
        {"sglang_model_routers", "sglang_router_ip", "sglang_router_port"}
    )

    sglang_model_routers: dict[str, tuple[str, int]] | None
    sglang_router_ip: A[str | None, Arg(help="IP address of the SGLang router")] = None
    sglang_router_port: A[int | None, Arg(help="Port of the SGLang router")] = None
    sglang_router_policy: A[
        str | None,
        Arg(help="Routing policy for the SGLang router (e.g., 'consistent_hashing', 'round_robin')"),
    ] = None
    sglang_router_request_timeout_secs: A[
        int,
        Arg(help="Timeout for requests to the SGLang router in seconds"),
    ] = 14400

    router_args: dict[str, Any]

    dynamo_config: A[
        DynamoConfig | None,
        Arg(type_parser=str, help="Dynamo router/sidecar JSON configuration."),
    ] = None

    router: A[
        RouterKind,
        Arg(
            choices=list(RouterKind),
            type_parser=RouterKind,
            help="The router in front of each model's inference engines: sgl-router or Miles.",
        ),
    ] = RouterKind.SGLANG
    miles_router_timeout: A[float | None, Arg(help="Timeout for MilesRouter HTTP requests in seconds.")] = None
    miles_router_max_connections: A[
        int | None,
        Arg(help="Max connections for MilesRouter HTTP client. When unset, the HTTP client has no connection cap."),
    ] = None
    miles_router_health_check_failure_threshold: A[
        int,
        Arg(help="Number of consecutive failures before marking a worker as unhealthy."),
    ] = 3

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        parser.add_argument(
            "--use-miles-router",
            dest="router",
            action="store_const",
            const=RouterKind.MILES,
            help="Alias for --router miles.",
        )
        _add_router_cli_args(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        parser = _make_prefixed_cli_parser()
        values = vars(args)
        return {
            "router_args": {
                action.dest.removeprefix(_ROUTER_DEST_PREFIX): values[action.dest] for action in parser._actions
            }
        }

    @classmethod
    def arg_names(cls) -> set[str]:
        return {action.dest for action in _make_prefixed_cli_parser()._actions}


def _make_prefixed_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    _add_router_cli_args(parser)
    return parser
