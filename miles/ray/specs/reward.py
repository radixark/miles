from typing import Any, ClassVar

from typing_extensions import Self

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.args.configs.reward_pool import RewardPoolConfig
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime import RolloutConfig
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_spec import BaseServeSpec, SchedulingSpec, WorkerCtorContext, WorkerLaunchContext

REWARD_POOL_ID_PREFIX = "reward"
REWARD_WORKER_CLASS = "miles.ray.rollout.reward_worker.RewardWorker"
# A GPU reward worker owns its bundle, as a dedicated engine does; the share is Ray's admission count, not memory.
_NUM_GPUS_PER_REWARD_WORKER = 0.2
_NUM_CPUS_PER_GPU_REWARD_WORKER = 0.2
_NUM_CPUS_PER_CPU_REWARD_WORKER = 1


class RewardPoolSpec(BaseServeSpec):
    worker_type: ClassVar[str] = "reward"
    config_class = RolloutConfig
    args: RolloutConfig
    pool: RewardPoolConfig
    worker_class: str = REWARD_WORKER_CLASS

    @classmethod
    def create(cls, config: RolloutConfig) -> list[Self]:
        return [cls(args=config, pool=pool, name=reward_pool_id(pool.name)) for pool in config.reward_pools]

    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        if not self.pool.uses_gpu:
            return SchedulingSpec(
                num_cells=self.pool.num_workers,
                num_workers_per_cell=1,
                num_gpus_per_worker=0,
                num_cpus_per_worker=_NUM_CPUS_PER_CPU_REWARD_WORKER,
            )
        return SchedulingSpec(
            num_cells=self.pool.num_workers,
            num_workers_per_cell=1,
            num_gpus_per_worker=_NUM_GPUS_PER_REWARD_WORKER,
            num_cpus_per_worker=_NUM_CPUS_PER_GPU_REWARD_WORKER,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=self.pool.num_gpus_per_node,
            pg_name="rollout",
            pg_slot_offset=scaling.reward_scaling.pools[self.pool.name].gpu_offset,
        )

    def env_var(self, ctx: WorkerLaunchContext) -> dict[str, str]:
        # The worker picks its GPU from gpu_ids, so Ray must not narrow the visible devices to a fractional share.
        return {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} if self.pool.uses_gpu else {}

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return dict(args=ctx.args, pool=self.pool, gpu_ids=ctx.gpu_ids)


def reward_pool_id(name: str) -> str:
    return f"{REWARD_POOL_ID_PREFIX}-{name}"


def reward_worker_name(name: str, *, cell_index: int) -> str:
    return compute_worker_name(pool_id=reward_pool_id(name), cell_index=cell_index)


def reward_cell_id(name: str, *, cell_index: int) -> str:
    return compute_cell_id(pool_id=reward_pool_id(name), cell_index=cell_index)
