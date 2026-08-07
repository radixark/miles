from __future__ import annotations

from types import SimpleNamespace

from tests.fast.fixtures.capability_fixtures import FakeBackendCapability, SingleWorkerProvider

from miles.ray.specs.rollout import (
    ROLLOUT_EXECUTOR_POOL_ID,
    ROLLOUT_EXECUTOR_WORKER_CLASS,
    create_rollout_executor_handle,
    spec_rollout_executor,
)
from miles.utils.workers.worker_spec import WorkerCtorContext


def _args(*, debug_train_only: bool = False, use_session_server: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        pin_rollout_manager_to_head=False,
        debug_train_only=debug_train_only,
        use_session_server=use_session_server,
    )


class TestRolloutExecutorSpec:
    def test_a_run_asks_for_exactly_one_gpuless_worker(self):
        """One executor per run, and it must claim no gpu or the scheduler would reserve a whole slot."""
        spec = spec_rollout_executor(_args())

        assert spec.name == ROLLOUT_EXECUTOR_POOL_ID
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpus_per_worker == 0

    def test_the_worker_class_is_the_executor_itself(self):
        """The spec names the class a pod or actor constructs, so it must be the real implementation."""
        assert spec_rollout_executor(_args()).worker_class == ROLLOUT_EXECUTOR_WORKER_CLASS

    def test_the_ctor_kwargs_hand_the_worker_the_providers_it_resolves_with(self):
        """The executor resolves its own addresses in init(), so its spec names exactly what that takes."""
        capability = FakeBackendCapability(static_provider=object())
        context = WorkerCtorContext(cell_id="cell-0", worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs = spec_rollout_executor(_args(use_session_server=True)).ctor_kwargs(context)

        assert sorted(kwargs) == ["args", "router_provider", "session_server_provider"]
        assert kwargs["router_provider"] is capability.static_provider
        assert kwargs["session_server_provider"] is capability.static_provider
        assert capability.requested_static_pool_ids == ["inference-router-0", "session-server"]

    def test_a_run_without_session_servers_is_given_no_session_provider(self):
        """Nothing is deployed to wait for, and a provider would make the executor wait for it anyway."""
        capability = FakeBackendCapability(static_provider=object())
        context = WorkerCtorContext(cell_id="cell-0", worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs = spec_rollout_executor(_args(use_session_server=False)).ctor_kwargs(context)

        assert kwargs["session_server_provider"] is None

    def test_the_handle_is_whichever_worker_the_pool_deploys(self):
        """Nothing may guess the executor's worker name, so the handle comes from the pool it was asked for."""
        handle = object()
        provider = SingleWorkerProvider(handle)
        capability = FakeBackendCapability(static_provider=provider)

        assert create_rollout_executor_handle(capability=capability) is handle
        assert provider.single_handle_pool_ids == [ROLLOUT_EXECUTOR_POOL_ID]
