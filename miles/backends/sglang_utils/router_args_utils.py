import argparse

from sglang_router.launch_router import RouterArgs

from miles.utils.workers.argv_utils import render_cli_argv

_ROUTER_FIELD_TO_DEST = {
    "server_cert_path": "tls_cert_path",
    "server_key_path": "tls_key_path",
    "prefill_urls": "prefill",
    "decode_urls": "decode",
}


def compute_sglang_router_args(
    args, *, host: str, port: int, prometheus_port: int, has_pd_disaggregation: bool
) -> RouterArgs:
    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = host
    router_args.port = port
    router_args.prometheus_port = prometheus_port
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if args.sglang_router_policy:
        router_args.policy = args.sglang_router_policy

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True

    return router_args


def router_args_to_argv(router_args: RouterArgs) -> list[str]:
    return render_cli_argv(
        router_args,
        make_parser=_make_cli_parser,
        from_parsed=RouterArgs.from_cli_args,
        field_to_dest=_ROUTER_FIELD_TO_DEST,
    )


def parse_router_args_argv(argv: list[str]) -> RouterArgs:
    return RouterArgs.from_cli_args(_make_cli_parser().parse_args(argv))


def _make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    return parser
