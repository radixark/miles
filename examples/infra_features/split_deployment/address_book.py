from dataclasses import dataclass

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.ray.specs.inference import (
    INFERENCE_CONTROLLER_ADDR_FLAG,
    INFERENCE_CONTROLLER_POOL_ID,
    INFERENCE_CONTROLLER_WORKER_CLASS,
)
from miles.ray.specs.train import (
    TRAINER_CONTROLLER_ADDRS_FLAG,
    TRAINER_CONTROLLER_WORKER_CLASS,
    compute_trainer_controller_pool_id,
)
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import get_mooncake_object_store_args
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName, RunNames
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_cell_addrs
from miles.utils.workers.worker_spec import RPC_PORT_NAME, SchedulingSpec, ServeWorkerSpec

DEFAULT_TRAINER_ID: str = ACTOR_ROLE
INIT_EXPECTED_NUM_CELLS_FLAG: str = "--init-expected-num-cells"


@dataclass(frozen=True)
class RunAddressBook:
    run_id: str
    run_uuid: str
    namespace: str

    @classmethod
    def of_config(cls, config: command_utils.ExecuteTrainConfig) -> "RunAddressBook":
        assert config.namespace, (
            "the deployments of one run reach each other by in-cluster release names, and a namespace is half "
            "of every such name"
        )
        assert config.run_uuid is not None, (
            "the deployments of one run are joined by nothing but the run uuid, and a split launch is given none "
            "of its own, so every command installing a part of it has to carry the same --run-uuid"
        )
        return cls(run_id=config.run_id, run_uuid=config.run_uuid, namespace=config.namespace)

    def release(self, deploy_component: DeployComponent, deploy_instance_id: str | None = None) -> str:
        return ReleaseName(
            run_id=self.run_id, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id
        ).serialize()

    def trainer_controller_addrs_arg(self, *, deploy_instance_id_of_trainer_id: dict[str, str | None]) -> str:
        entries = [
            f"{trainer_id}="
            + self._rpc_addr(
                release=self.release(DeployComponent.TRAINER, deploy_instance_id),
                pool_id=compute_trainer_controller_pool_id(trainer_id),
                worker_class=TRAINER_CONTROLLER_WORKER_CLASS,
            )
            for trainer_id, deploy_instance_id in deploy_instance_id_of_trainer_id.items()
        ]
        return f"{TRAINER_CONTROLLER_ADDRS_FLAG} {' '.join(entries)} "

    def inference_controller_addr_arg(self) -> str:
        addr = self._rpc_addr(
            release=self.release(DeployComponent.PRIMARY),
            pool_id=INFERENCE_CONTROLLER_POOL_ID,
            worker_class=INFERENCE_CONTROLLER_WORKER_CLASS,
        )
        return f"{INFERENCE_CONTROLLER_ADDR_FLAG} {addr} "

    def shared_object_store_args(self) -> str:
        return get_mooncake_object_store_args(
            master_host=MooncakeInfo.master_service_host(self.release(DeployComponent.PRIMARY), self.namespace)
        )

    def _rpc_addr(self, *, release: str, pool_id: str, worker_class: str) -> str:
        rpc = static_cell_addrs(
            spec=_controller_spec(pool_id=pool_id, worker_class=worker_class), release=release, cell_index=0
        )[RPC_PORT_NAME]
        return f"{RunNames.service_fqdn(name=rpc.host, namespace=self.namespace)}:{rpc.port}"


def init_expected_num_cells_arg(num_cells_per_model: int) -> str:
    return f"{INIT_EXPECTED_NUM_CELLS_FLAG} {num_cells_per_model} "


def _controller_spec(*, pool_id: str, worker_class: str) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=pool_id,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0, num_cpus_per_worker=1),
        worker_class=worker_class,
        ctor_kwargs=lambda _ctx: {},
    )
