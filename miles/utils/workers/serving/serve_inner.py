from __future__ import annotations

import os
import sys
from typing import Any

import uvicorn

from miles.ray.specs.entrypoint import SERVE_SPEC_CLASSES
from miles.utils.function_registry import load_function
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.factory import get_backend_capability
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.serving.utils import create_server_socket, parse_own_args, parse_serve_worker_config
from miles.utils.workers.serving.worker_identity import (
    read_worker_identity,
    read_worker_in_pod_index,
    read_worker_metadata,
)
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_spec import RPC_PORT_NAME, BaseServeSpec, PortInfo


def main() -> None:
    own_args = parse_own_args(sys.argv[1:])

    worker_config = parse_serve_worker_config(own_args.config)
    spec_class = SERVE_SPEC_CLASSES[worker_config.worker_type]
    spec = spec_class.create(spec_class.config_class.model_validate(worker_config.args))
    worker = create_worker(spec, static_connections=worker_config.static_connections)
    _log(f"pool_id={spec.name} worker_class={spec.worker_class}")

    port = _rpc_port_of(spec).effective_static_port(worker_in_pod_index=read_worker_in_pod_index(os.environ))
    app = create_rpc_app(worker)
    with create_server_socket(port=port) as server_socket:
        _log(f"serve address={server_socket.getsockname()}")
        uvicorn.Server(uvicorn.Config(app)).run(sockets=[server_socket])


def create_worker(spec: BaseServeSpec, *, static_connections: StaticConnConfig) -> Any:
    identity = read_worker_identity(os.environ)
    _log(f"identity={identity}")
    capability = DeferredBackendCapability(create=lambda: _backend_capability(spec, static_connections))
    context = identity.ctor_context(args=spec.args, capability=capability)
    return load_function(spec.worker_class)(**spec.ctor_kwargs(context))


def _backend_capability(spec: BaseServeSpec, static_connections: StaticConnConfig) -> BackendCapability:
    cluster_backend = ClusterBackend(spec.args.cluster_backend)
    return get_backend_capability(cluster_backend=cluster_backend, static_connections=static_connections)


def _rpc_port_of(spec: BaseServeSpec) -> PortInfo:
    ports = [port_info for port_info in read_worker_metadata(os.environ).port_infos if port_info.name == RPC_PORT_NAME]
    assert len(ports) == 1, f"spec '{spec.name}' declares {len(ports)} rpc ports, so this process cannot pick one"
    return ports[0]


def _log(message: str) -> None:
    print(f"[serve_inner] {message}", flush=True)


if __name__ == "__main__":
    main()
