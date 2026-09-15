from __future__ import annotations

import os
import socket
import sys
from typing import Any

import uvicorn

from miles.utils.arguments import parse_args
from miles.utils.function_registry import load_function
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.factory import get_backend_capability
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.serving.utils import (
    compute_serve_worker_spec,
    override_argv,
    parse_own_args,
    split_worker_argv,
)
from miles.utils.workers.serving.worker_identity import read_worker_identity, read_worker_in_pod_index
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_spec import RPC_PORT_NAME, PortInfo, ServeWorkerSpec

IPV4_WILDCARD_HOST = "0.0.0.0"
IPV6_WILDCARD_HOST = "::"


def main() -> None:
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    args = parse_own_args(own_argv)
    _log(f"start own_argv={own_argv} worker_argv={worker_argv}")

    spec = compute_serve_worker_spec(specs_fn=args.specs, pool_id=args.pool_id, worker_argv=worker_argv)
    worker = create_worker(spec, specs_fn=args.specs, worker_argv=worker_argv)
    _log(f"pool_id={args.pool_id} worker_class={spec.worker_class}")

    port = _rpc_port_of(spec).effective_static_port(worker_in_pod_index=read_worker_in_pod_index(os.environ))
    server_socket = _create_server_socket(port=port)
    app = create_rpc_app(worker)
    _log(f"serve address={server_socket.getsockname()}")
    with server_socket:
        uvicorn.Server(uvicorn.Config(app)).run(sockets=[server_socket])


def _create_server_socket(*, port: int) -> socket.socket:
    if socket.has_dualstack_ipv6():
        return socket.create_server(
            (IPV6_WILDCARD_HOST, port),
            family=socket.AF_INET6,
            dualstack_ipv6=True,
        )
    return socket.create_server((IPV4_WILDCARD_HOST, port), family=socket.AF_INET)


def create_worker(spec: ServeWorkerSpec, *, specs_fn: str, worker_argv: list[str]) -> Any:
    identity = read_worker_identity(scheduling=spec.scheduling, environ=os.environ)
    _log(f"identity={identity}")
    capability = DeferredBackendCapability(create=lambda: _backend_capability(specs_fn, worker_argv))
    return load_function(spec.worker_class)(**spec.ctor_kwargs(identity.ctor_context(capability=capability)))


def _backend_capability(specs_fn: str, worker_argv: list[str]) -> BackendCapability:
    with override_argv(worker_argv):
        cluster_backend = ClusterBackend(parse_args().cluster_backend)
    return get_backend_capability(specs=load_function(specs_fn)(worker_argv), cluster_backend=cluster_backend)


def _rpc_port_of(spec: ServeWorkerSpec) -> PortInfo:
    ports = [port_info for port_info in spec.port_infos if port_info.name == RPC_PORT_NAME]
    assert len(ports) == 1, f"spec '{spec.name}' declares {len(ports)} rpc ports, so this process cannot pick one"
    return ports[0]


def _log(message: str) -> None:
    print(f"[serve_inner] {message}", flush=True)


if __name__ == "__main__":
    main()
