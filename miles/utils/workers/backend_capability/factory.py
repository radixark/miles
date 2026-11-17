from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.ray_worker_manager import (
    RayWorkerManager,
    launch_ray_worker_manager,
    shutdown_ray_worker_manager,
)
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_spec import BaseSpec


@dataclass(frozen=True)
class LaunchedWorkerManager:
    """The worker manager a driver launched, whichever backend it runs on; the driver only ever shuts it down."""

    shutdown: Callable[[], Awaitable[None]]


def launch_worker_manager(args: Any, specs: list[BaseSpec]) -> LaunchedWorkerManager | None:
    """None where the platform creates the run's workers itself, so the driver has no manager to launch."""
    match ClusterBackend(args.cluster_backend):
        case ClusterBackend.KUBERNETES:
            return None
        case ClusterBackend.RAY:
            handle = launch_ray_worker_manager(args, specs)
            return LaunchedWorkerManager(shutdown=partial(shutdown_ray_worker_manager, handle))
        case ClusterBackend.MONARCH:
            # torchmonarch is an optional dependency, imported only by runs that pick it
            from miles.utils.workers.monarch.manager import (
                launch_monarch_worker_manager,
                shutdown_monarch_worker_manager,
            )

            manager = launch_monarch_worker_manager(args, specs)
            return LaunchedWorkerManager(shutdown=partial(shutdown_monarch_worker_manager, manager))


def get_backend_capability(
    *,
    cluster_backend: ClusterBackend,
    static_connections: StaticConnConfig,
) -> BackendCapability:
    match cluster_backend:
        case ClusterBackend.KUBERNETES:
            return compute_helm_backend_capability(config=static_connections)
        case ClusterBackend.RAY:
            return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())
        case ClusterBackend.MONARCH:
            # torchmonarch is an optional dependency, imported only by runs that pick it
            from miles.utils.workers.monarch.manager import installed_monarch_backend_capability

            return installed_monarch_backend_capability()
