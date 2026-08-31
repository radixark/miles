import abc
import logging
import os
import random
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

import ray
import torch
import torch.distributed as dist

import miles.utils.eval_config
from miles.utils import object_store
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.env_report import collect_and_print_node_env_report
from miles.utils.ft_utils.heartbeat_utils import HeartbeatStatus, SimpleHeartbeat
from miles.utils.logging_utils import configure_logger
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.misc import NodeProbeMixin, get_current_node_ip, get_free_port
from miles.utils.test_utils.det_process_group import DET_NCCL_BACKEND_NAME, register_det_nccl_backend
from miles.utils.test_utils.fault_injector import inject_fault as _inject_fault

if TYPE_CHECKING:
    from miles.ray.rollout.inference_controller import UpdatableEngines


logger = logging.getLogger(__name__)

TRAINER_CONCURRENCY_GROUPS = {"heartbeat_status": 1, "default": 1, "fault_injector": 1, "kill_self": 1}
TRAINER_METHOD_CONCURRENCY_GROUPS = {
    "get_heartbeat_status": "heartbeat_status",
    "inject_fault": "fault_injector",
    "kill_self": "kill_self",
}


def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    if not cvd:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))


class TrainRayActor(NodeProbeMixin):
    def __init__(
        self,
        *,
        args,
        world_size: int,
        rank: int,
        indep_dp_store_addr: str,
        role: Literal["actor", "critic"],
        cell_index: int,
    ):
        configure_logger(
            args, source=TrainProcessIdentity(component=role, cell_index=cell_index, rank_within_cell=rank)
        )
        self.args = args

        self._heartbeat = SimpleHeartbeat()
        self._world_size = world_size
        self._rank = rank
        self._indep_dp_store_addr = indep_dp_store_addr

        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

        object_store.init_instance(args)

    def propose_master_addr_and_port(self) -> tuple[str, int]:
        return get_current_node_ip(), get_free_port(start_port=random.randint(20000, 21000))

    def configure_master_addr_and_port(self, *, master_addr: str, master_port: int) -> None:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

    # TODO mv the args into ctor
    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        self.args = args
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher

        if env_report := args.env_report:
            collect_and_print_node_env_report(
                role=role,
                rank=self._rank,
                partial_env_report=env_report,
            )

        torch.serialization.add_safe_globals([miles.utils.eval_config.EvalDatasetConfig])

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

        if args.debug_deterministic_collective:
            register_det_nccl_backend()
            args.distributed_backend = DET_NCCL_BACKEND_NAME
            logger.info("Deterministic collectives: training world uses the det_nccl backend")

        # Use hybrid backend when FSDP CPU offload is enabled with a CPU backend
        backend = args.distributed_backend
        if getattr(args, "fsdp_cpu_offload", False) and getattr(args, "fsdp_cpu_backend", None):
            cpu_backend = args.fsdp_cpu_backend
            backend = f"cpu:{cpu_backend},cuda:{args.distributed_backend}"
            logger.info(f"FSDP CPU offload enabled, using hybrid backend: {backend}")

        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        try:
            if torch.version.hip is not None:
                logger.info("Detected ROCm/HIP environment, skipping NUMA affinity setup")
                # will find the coresponding API to implement ROCm version as below
            else:
                import pynvml

                pynvml.nvmlInit()

                local_rank = int(os.environ["RANK"]) % args.num_gpus_per_node

                handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
                pynvml.nvmlDeviceSetCpuAffinity(handle)

                logger.info(f"Set NUMA affinity for GPU {local_rank}")
                pynvml.nvmlShutdown()

        except ImportError:
            logger.info("Warning: pynvml not available, skipping NUMA affinity setup")
        except Exception as e:
            logger.info(f"Warning: Failed to set NUMA affinity: {e}")

        self._heartbeat.bump()

    def get_heartbeat_status(self) -> HeartbeatStatus:
        return self._heartbeat.status()

    def inject_fault(self, mode: str) -> None:
        _inject_fault(mode=mode)

    def kill_self(self) -> None:
        os._exit(1)

    def clear_memory(self):
        print_memory("before TrainRayActor.clear_memory")
        clear_memory()
        print_memory("after TrainRayActor.clear_memory")

    @abc.abstractmethod
    def sleep(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def wake_up(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, rollout_id, rollout_data_ref, external_data=None):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, rollout_id, force_sync=False):
        raise NotImplementedError

    def export_hf(self, rollout_id: int, path: str) -> None:
        """Export current weights as an HF checkpoint to ``path`` (eval snapshots)."""
        raise NotImplementedError(f"{type(self).__name__} does not support HF export")

    @abc.abstractmethod
    def update_weights(self, info: "UpdatableEngines") -> int | None:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_parallel_config(self):
        raise NotImplementedError

    def set_rollout_executor(self, rollout_executor):
        self.rollout_executor = rollout_executor
        if self.args.rank == 0:
            ray.get(self.rollout_executor.set_train_parallel_config.remote(self.train_parallel_config))
