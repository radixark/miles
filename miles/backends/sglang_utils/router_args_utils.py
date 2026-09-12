import argparse
import ipaddress
import logging
import socket
from collections.abc import Mapping

from sglang_router.launch_router import RouterArgs

from miles.utils.http_utils import MILES_HOST_IP_ENV, _wrap_ipv6
from miles.utils.workers.argv_utils import render_cli_argv

logger = logging.getLogger(__name__)

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


def compute_sglang_router_bind_host(host: str) -> str:
    """The ``--host`` sglang_router can bind for a worker advertised as ``host``.

    The router parses ``host:port`` as a socket address, so it takes an IPv4 literal or a bracketed IPv6
    literal and nothing else. The worker manager may advertise a placed node under a hostname (Ray on Slurm
    reports nodes that way, and so may ``MILES_HOST_IP``); that name stays the connect-side address and is
    resolved here only for the router's own listening socket.
    """
    if _is_ip_literal(host):
        return _wrap_ipv6(host)

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise RuntimeError(_unresolvable_bind_host_message(host, reason=str(e))) from e
    candidates = list(dict.fromkeys(info[4][0] for info in infos))
    if not candidates:
        raise RuntimeError(_unresolvable_bind_host_message(host, reason="the resolver returned no address"))
    if len(candidates) > 1:
        logger.warning(
            f"The sglang router's advertised host {host!r} resolves to {candidates}; binding the first resolver "
            f"candidate {candidates[0]!r}. Configure {MILES_HOST_IP_ENV} or Ray's --node-ip-address as a specific "
            f"IP literal if deterministic interface selection is required."
        )
    return _wrap_ipv6(candidates[0])


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _unresolvable_bind_host_message(host: str, *, reason: str) -> str:
    return (
        f"Cannot bind the sglang router on {host!r}: {reason}. sglang_router needs an IP literal for --host, "
        f"so configure {MILES_HOST_IP_ENV} on that node, or Ray's --node-ip-address, as a specific IP literal "
        f"(another hostname would fail the same way)."
    )


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
