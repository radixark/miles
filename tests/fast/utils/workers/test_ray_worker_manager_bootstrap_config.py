from tests.fast.utils.workers.test_ray_worker_manager_serve import _WORKER_CLASS_PATH, _CtorKwargsProbe

from miles.utils.workers.ray_worker_manager import bootstrapped_worker_class
from miles.utils.workers.worker_spec import WorkerLaunchContext


class TestBootstrapConfig:
    def test_the_worker_recipe_receives_its_pool_config_and_complete_cell_width(self) -> None:
        """A worker recipe must retain its sliced config and full cell width across Ray bootstrap."""
        config = object()
        recipe = _CtorKwargsProbe()
        launch_context = WorkerLaunchContext(
            args=config,
            cell_index=2,
            worker_in_cell_index=3,
            num_workers_per_cell=8,
            gpu_ids=[4, 5],
        )
        worker_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        worker = worker_class(ctor_kwargs=recipe, context=launch_context)

        [context] = recipe.contexts
        assert context.args is config
        assert context.num_workers_per_cell == 8
        assert (context.cell_index, context.worker_in_cell_index, context.gpu_ids) == (2, 3, [4, 5])
        assert worker.kwargs == {"rank": 3, "role": "actor"}
