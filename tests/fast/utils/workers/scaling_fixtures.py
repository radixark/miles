from tests.fast.utils.workers.fake_specs import FakeCommandSpec

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.worker_spec import SchedulingSpec


class _ScalingCommandSpec(FakeCommandSpec):
    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        return SchedulingSpec(
            num_cells=scaling.actor_num_nodes,
            num_workers_per_cell=scaling.actor_num_gpus_per_node,
            num_gpus_per_worker=0,
            num_gpu_slots_per_worker=0,
            num_gpus_per_node=1,
        )
