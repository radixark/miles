import abc
import logging
import os
import random
from datetime import timedelta
from typing import Literal

import ray
import torch
import torch.distributed as dist

import miles.utils.eval_config
from miles.ray.ray_actor import RayActor
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.env_report import collect_and_print_node_env_report
from miles.utils.heartbeat_utils import HeartbeatStatus, SimpleHeartbeat
from miles.utils.logging_utils import configure_logger
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.process_identity import TrainProcessIdentity
from miles.utils.test_utils.fault_injector import inject_fault as _inject_fault

logger = logging.getLogger(__name__)


def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))


class TrainRayActor(RayActor):
    def __init__(
        self,
        args,
        world_size: int,
        rank: int,
        master_addr,
        master_port,
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
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    # TODO mv the args into ctor
    def init(self, args, role, with_ref=False):
        self.args = args
        self.role = role
        self.with_ref = with_ref

        if env_report := args.env_report:
            collect_and_print_node_env_report(
                role=role,
                rank=self._rank,
                partial_env_report=env_report,
            )

        torch.serialization.add_safe_globals([miles.utils.eval_config.EvalDatasetConfig])

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

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

    @ray.method(concurrency_group="heartbeat_status")
    def get_heartbeat_status(self) -> HeartbeatStatus:
        return self._heartbeat.status()

    @ray.method(concurrency_group="fault_injector")
    def inject_fault(self, mode: str) -> None:
        _inject_fault(mode=mode)

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
    def train(self, rollout_id, rollout_data_ref):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, rollout_id, force_sync=False):
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self):
        raise NotImplementedError

    @abc.abstractmethod
    def connect_actor_critic(self, critic_group):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_parallel_config(self):
        raise NotImplementedError

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        if self.args.rank == 0:
            ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
