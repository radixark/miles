import argparse
from collections.abc import Mapping

from sglang_router.launch_router import RouterArgs

from miles.utils.workers.argv_utils import render_cli_argv

_ROUTER_DEST_PREFIX = "router_"


def compute_sglang_router_args(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    prometheus_port: int,
    has_pd_disaggregation: bool,
) -> dict[str, object]:
    router_args = _extract_router_cli_values(args)
    router_args.update(
        host=host,
        port=port,
        prometheus_port=prometheus_port,
        log_level="warn",
        request_timeout_secs=args.sglang_router_request_timeout_secs,
    )

    if args.sglang_router_policy:
        router_args["policy"] = args.sglang_router_policy

    if has_pd_disaggregation:
        router_args["pd_disaggregation"] = True

    return router_args


def router_args_to_argv(router_args: Mapping[str, object]) -> list[str]:
    return render_cli_argv(
        router_args,
        expected_obj=RouterArgs.from_cli_args(argparse.Namespace(**router_args)),
        make_parser=_make_cli_parser,
        from_parsed=RouterArgs.from_cli_args,
    )


def parse_router_args_argv(argv: list[str]) -> RouterArgs:
    return RouterArgs.from_cli_args(_make_cli_parser().parse_args(argv))


def _make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    return parser


def _extract_router_cli_values(args: argparse.Namespace) -> dict[str, object]:
    prefixed_defaults = vars(_make_prefixed_cli_parser().parse_args([]))
    return {name.removeprefix(_ROUTER_DEST_PREFIX): getattr(args, name) for name in prefixed_defaults}


def _make_prefixed_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    return parser
