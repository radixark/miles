from enum import Enum

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ClusterBackend(Enum):
    RAY = "ray"
    KUBERNETES = "kubernetes"

    def provides_ray_cluster(self) -> bool:
        """Whether a Ray cluster backs the run, which the Ray-only features (dashboard, prometheus collector,
        distributed post, the ray object store) need."""
        return self is ClusterBackend.RAY

    def launches_workers_from_driver(self) -> bool:
        """Whether the driver launches its workers through a worker manager it owns, rather than the platform
        creating them before the driver starts."""
        return self is ClusterBackend.RAY

    def auto_resumes_cells(self) -> bool:
        """Whether the platform recreates a suspended cell by itself, so nobody has to resume it."""
        return self is ClusterBackend.KUBERNETES


class WorkerCommBackend(Enum):
    RAY = "ray"
    RPC = "rpc"


class DeployComponent(Enum):
    ALL = "all"
    PRIMARY = "primary"
    TRAINER = "trainer"
    INFERENCE = "inference"

    def selects(self, component: "DeployComponent") -> bool:
        assert component is not DeployComponent.ALL, "`all` is a selector over components, never a component itself"
        return self is DeployComponent.ALL or self is component

    def deploys_orchestration_script(self) -> bool:
        return self.selects(DeployComponent.PRIMARY)

    def deploys_own_inference_engines(self) -> bool:
        return self.selects(DeployComponent.INFERENCE)

    def is_split(self) -> bool:
        return self is not DeployComponent.ALL

    def takes_instance_id(self) -> bool:
        return self in (DeployComponent.TRAINER, DeployComponent.INFERENCE)


class PlatformAccess(Enum):
    NONE = "none"
    READ = "read"
    READ_DELETE = "read-delete"


class HotRestartComponent(Enum):
    ORCHESTRATION = "orchestration"
    ROLLOUT_EXECUTOR = "rollout_executor"


HOT_RESTART_SEPARATOR = ","


def parse_hot_restart(value: str) -> list[HotRestartComponent]:
    components = [
        HotRestartComponent(stripped) for name in value.split(HOT_RESTART_SEPARATOR) if (stripped := name.strip())
    ]
    assert (
        components or not value
    ), f"--hot-restart {value!r} names no component; pass one of {[one.value for one in HotRestartComponent]}"
    return components


class DeploymentIdentity(FrozenStrictBaseModel):
    run_uuid: str
    deploy_component: str
    deploy_instance_id: str | None = None
    trainer_id: str | None = None


_SUPPORTED_WORKER_COMM_BACKENDS = {
    ClusterBackend.RAY: (WorkerCommBackend.RAY, WorkerCommBackend.RPC),
    ClusterBackend.KUBERNETES: (WorkerCommBackend.RPC,),
}


def resolve_worker_comm_backend(*, cluster_backend: ClusterBackend, requested: str | None) -> WorkerCommBackend:
    if requested is None:
        return _SUPPORTED_WORKER_COMM_BACKENDS[cluster_backend][0]

    backend = WorkerCommBackend(requested)
    supported = _SUPPORTED_WORKER_COMM_BACKENDS[cluster_backend]
    assert backend in supported, (
        f"--worker-comm-backend {backend.value} is not available under --cluster-backend {cluster_backend.value}, "
        f"which speaks {[one.value for one in supported]}"
    )
    return backend
