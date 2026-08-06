from __future__ import annotations

from types import SimpleNamespace

from miles.ray.specs.rollout import (
    ROLLOUT_EXECUTOR_SPEC_NAME,
    ROLLOUT_EXECUTOR_WORKER_CLASS,
    rollout_executor_cell_id,
    rollout_executor_worker_name,
    spec_rollout_executor,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(pin_rollout_manager_to_head=False)


class TestRolloutExecutorSpec:
    def test_a_run_asks_for_exactly_one_gpuless_worker(self):
        """One executor per run, and it must claim no gpu or the scheduler would reserve a whole slot."""
        spec = spec_rollout_executor(_args())

        assert spec.name == ROLLOUT_EXECUTOR_SPEC_NAME
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpus_per_worker == 0

    def test_the_worker_class_is_the_executor_itself(self):
        """The spec names the class a pod or actor constructs, so it must be the real implementation."""
        assert spec_rollout_executor(_args()).worker_class == ROLLOUT_EXECUTOR_WORKER_CLASS

    def test_the_ctor_kwargs_carry_only_args(self):
        """The executor resolves its own addresses in init(), so nothing else has to be injected."""
        spec = spec_rollout_executor(_args())

        assert list(spec.ctor_kwargs(None)) == ["args"]

    def test_the_worker_and_cell_names_are_stable(self):
        """The driver looks the executor up by name, so these names are part of the release's contract."""
        assert rollout_executor_worker_name() == "rollout-executor-0-0"
        assert rollout_executor_cell_id() == "rollout-executor-0"
