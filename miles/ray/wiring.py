from __future__ import annotations

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime import AllConfig
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.connection_config import build_static_conn_config
from miles.utils.workers.types import ClusterBackend


def launch_worker_manager(args) -> factory.LaunchedWorkerManager | None:
    return factory.launch_worker_manager(args, compute_specs(args))


async def shutdown_worker_manager(worker_manager: factory.LaunchedWorkerManager | None) -> None:
    if worker_manager is not None:
        await worker_manager.shutdown()


def compute_backend_capability(args: AllConfig) -> BackendCapability:
    return factory.get_backend_capability(
        static_connections=build_static_conn_config(specs=compute_specs(args), scaling=ScalingConfig.slice_from(args)),
        cluster_backend=ClusterBackend(args.cluster_backend),
    )
