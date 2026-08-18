from enum import Enum

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ClusterBackend(Enum):
    RAY = "ray"
    KUBERNETES = "kubernetes"


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


class HotRestartComponent(Enum):
    ORCHESTRATION = "orchestration"
    ROLLOUT_EXECUTOR = "rollout_executor"


HOT_RESTART_SEPARATOR = ","


def parse_hot_restart(value: str) -> list[HotRestartComponent]:
    return [HotRestartComponent(name.strip()) for name in value.split(HOT_RESTART_SEPARATOR) if name.strip()]


class DeploymentIdentity(FrozenStrictBaseModel):
    run_uuid: str
    deploy_component: str
    deploy_instance_id: str | None = None


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
