from __future__ import annotations

from miles.ray.placement_group import create_placement_groups
from miles.ray.specs.entrypoint import compute_specs
from miles.utils.arguments import parse_args_from_argv
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.kubernetes import (
    KubernetesBackendCapability,
    compute_kubernetes_backend_capability,
)
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.kubernetes.client import create_kubernetes_client
from miles.utils.workers.worker_provider.kubernetes.helm.env import (
    current_label_keys,
    current_namespace,
    current_release,
)


def launch_worker_manager(args):
    if ClusterBackend(args.cluster_backend) is ClusterBackend.KUBERNETES:
        return None
    return _launch_ray_worker_manager(args)


def get_backend_capability(args) -> BackendCapability:
    if ClusterBackend(args.cluster_backend) is ClusterBackend.KUBERNETES:
        return _kubernetes_backend_capability_from_args(args)
    return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())


def create_worker_backend_capability(*, worker_argv: list[str]) -> BackendCapability:
    return DeferredBackendCapability(create=lambda: get_backend_capability(parse_args_from_argv(worker_argv)))


def _kubernetes_backend_capability_from_args(args) -> KubernetesBackendCapability:
    return compute_kubernetes_backend_capability(
        specs=compute_specs(args),
        namespace=current_namespace(),
        release=current_release(),
        kubernetes_client_factory=create_kubernetes_client,
        num_gpus_per_node=args.num_gpus_per_node,
        label_keys=current_label_keys(),
    )


def _launch_ray_worker_manager(args):
    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(specs, pgs)
