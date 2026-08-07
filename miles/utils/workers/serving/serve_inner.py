from __future__ import annotations

import sys
from typing import Any

import uvicorn

from miles.utils.arguments import parse_args_from_argv
from miles.utils.function_registry import load_function
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.factory import get_backend_capability
from miles.utils.workers.pod_context import read_pod_rank, read_rank_in_pod
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.serving.utils import parse_own_args, serve_spec_of, split_worker_argv
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_spec import RPC_PORT_NAME, ServeWorkerSpec

DEFAULT_HOST = "0.0.0.0"


def main() -> None:
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    args = parse_own_args(own_argv)

    spec = serve_spec_of(specs_fn=args.specs, pool_id=args.pool_id, worker_argv=worker_argv)
    worker = create_worker(spec, specs_fn=args.specs, worker_argv=worker_argv)

    app = create_rpc_app(worker)
    uvicorn.run(app, host=DEFAULT_HOST, port=_rpc_port_of(spec) + read_rank_in_pod())


def create_worker(spec: ServeWorkerSpec, *, specs_fn: str, worker_argv: list[str]) -> Any:
    rank = read_pod_rank(scheduling=spec.scheduling)
    capability = DeferredBackendCapability(create=lambda: _backend_capability(specs_fn, worker_argv))
    return load_function(spec.worker_class)(**spec.ctor_kwargs(rank.ctor_context(capability=capability)))


def _backend_capability(specs_fn: str, worker_argv: list[str]) -> BackendCapability:
    return get_backend_capability(
        specs=load_function(specs_fn)(worker_argv),
        cluster_backend=ClusterBackend(parse_args_from_argv(worker_argv).cluster_backend),
    )


def _rpc_port_of(spec: ServeWorkerSpec) -> int:
    ports = [port_info.static_port for port_info in spec.port_infos if port_info.name == RPC_PORT_NAME]
    assert len(ports) == 1, f"spec '{spec.name}' declares {len(ports)} rpc ports, so this process cannot pick one"
    return ports[0]


if __name__ == "__main__":
    main()
