from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.ray.specs.train import TrainerSpec
from miles.utils.args.runtime import TrainerConfig
from miles.utils.workers.worker_spec import WorkerCtorContext


class TestTrainerWorkerContext:
    def test_world_size_uses_the_runtime_cell_without_deployment_scaling(self) -> None:
        """A trainer joins its runtime cell without recomputing a deployment-wide size."""
        args = TrainerConfig.model_construct(trainer_role="critic")
        context = WorkerCtorContext(
            args=args,
            cell_index=3,
            worker_in_cell_index=1,
            num_workers_per_cell=2,
            gpu_ids=[0],
            capability=FakeBackendCapability(),
        )
        spec = TrainerSpec.model_construct()

        kwargs = spec.ctor_kwargs(context)

        assert kwargs["world_size"] == 2
        assert kwargs["rank"] == 1
        assert kwargs["cell_index"] == 3
