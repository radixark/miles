from __future__ import annotations

from typing import Any

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.ray.specs import entrypoint
from miles.ray.specs.bootstrap import compute_ctor_kwargs, serve_spec_of
from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils import arguments
from miles.utils.workers.worker_spec import SchedulingSpec, ServeWorkerSpec, WorkerCtorContext

TRAINER_POOL_ID = compute_trainer_pool_id("actor")
WORKER_ARGV = ["--cluster-backend", "kubernetes", "--rollout-num-gpus", "8"]

ADDRESSING_POOL_ID = "inference-controller"
WATCHED_POOLS = ["inference-engine-0-0", "inference-engine-0-1"]
ROUTER_WORKER_NAME = "inference-router-0-0-0"


@pytest.fixture
def run_argv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> list[str]:
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 8}])
    )
    args = make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=8,
        num_gpus_per_node=8,
        cluster_backend="kubernetes",
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
    )

    def parse(argv: list[str], add_custom_arguments: Any = None) -> Any:
        assert argv == WORKER_ARGV
        return args

    monkeypatch.setattr(arguments, "parse_args_from_argv", parse)
    return WORKER_ARGV


def context(*, cell_index: int = 0, worker_in_cell_index: int = 0) -> WorkerCtorContext:
    return WorkerCtorContext(
        cell_id=f"trainer-actor-{cell_index}",
        cell_ordinal=cell_index,
        worker_in_cell_index=worker_in_cell_index,
        gpu_ids=[worker_in_cell_index],
        capability=FakeBackendCapability(),
    )


def ctor_context(capability: FakeBackendCapability) -> WorkerCtorContext:
    return WorkerCtorContext(
        cell_id="trainer-actor-0", cell_ordinal=0, worker_in_cell_index=0, gpu_ids=[0], capability=capability
    )


def addressing_spec() -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=ADDRESSING_POOL_ID,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class="miles.ray.rollout.inference_controller.InferenceController",
        ctor_kwargs=lambda ctx: dict(
            engine_provider=ctx.capability.cells(pool_ids=WATCHED_POOLS),
            router_provider=ctx.capability.static(worker_name=ROUTER_WORKER_NAME),
        ),
    )


class TestServeSpecOf:
    def test_finds_the_trainer_spec_the_pod_was_launched_for(self, run_argv):
        """The pod knows its spec name and the run's argv, and has to rebuild everything else from them."""
        assert serve_spec_of(pool_id=TRAINER_POOL_ID, worker_argv=run_argv).name == TRAINER_POOL_ID

    def test_finds_the_rollout_executor_spec_too(self, run_argv):
        """The executor is a served static worker, so it takes the same bootstrap path as a trainer rank."""
        assert serve_spec_of(pool_id=ROLLOUT_EXECUTOR_POOL_ID, worker_argv=run_argv).name == (ROLLOUT_EXECUTOR_POOL_ID)

    def test_refuses_a_spec_this_run_does_not_have(self, run_argv):
        """A stale command line would otherwise build a worker for a run shape that no longer exists."""
        with pytest.raises(AssertionError, match="trainer-critic"):
            serve_spec_of(pool_id="trainer-critic", worker_argv=run_argv)

    def test_refuses_a_spec_that_is_launched_as_a_command(self, run_argv):
        """A router has no worker class to construct, so serving it would be a category error."""
        with pytest.raises(AssertionError, match="inference-router-0"):
            serve_spec_of(pool_id="inference-router-0", worker_argv=run_argv)


class TestComputeCtorKwargs:
    def test_gives_a_trainer_rank_the_keywords_its_constructor_declares(self, run_argv):
        """MegatronTrainRayActor is keyword-only, so these names are the pod's whole construction contract."""
        kwargs = compute_ctor_kwargs(pool_id=TRAINER_POOL_ID, worker_argv=run_argv, context=context(worker_in_cell_index=3))

        assert set(kwargs) == {"args", "world_size", "rank", "indep_dp_store_addr", "role", "cell_id"}
        assert (kwargs["rank"], kwargs["role"], kwargs["world_size"]) == (3, "actor", 8)

    def test_gives_the_rollout_executor_the_run_arguments(self, run_argv):
        """The executor is constructed from the parsed args alone, which only the pod's argv can rebuild."""
        kwargs = compute_ctor_kwargs(pool_id=ROLLOUT_EXECUTOR_POOL_ID, worker_argv=run_argv, context=context())

        assert set(kwargs) == {"args"}
        assert kwargs["args"].cluster_backend == "kubernetes"

    def test_the_cell_a_rank_belongs_to_reaches_its_constructor(self, run_argv):
        """Trainer cells differ only by this index, and it decides where a rank writes its offload files."""
        kwargs = compute_ctor_kwargs(pool_id=TRAINER_POOL_ID, worker_argv=run_argv, context=context())

        assert kwargs["cell_id"] == "trainer-actor-0"


class TestProviderRequests:
    @pytest.fixture
    def addressing_run(self, run_argv, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        monkeypatch.setattr(entrypoint, "compute_specs", lambda _args: [addressing_spec()])
        return run_argv

    def test_a_spec_names_the_pools_its_worker_will_observe(self, addressing_run):
        """The worker never learns which backend reports those cells, only which pool_ids it wants reported."""
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        compute_ctor_kwargs(pool_id=ADDRESSING_POOL_ID, worker_argv=addressing_run, context=ctor_context(capability))

        assert capability.requested_pool_ids == [WATCHED_POOLS]

    def test_a_spec_names_the_statically_addressed_worker_it_calls(self, addressing_run):
        """A router is addressed rather than observed, and only the backend knows how to redeem that name."""
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        compute_ctor_kwargs(pool_id=ADDRESSING_POOL_ID, worker_argv=addressing_run, context=ctor_context(capability))

        assert capability.requested_worker_names == [ROUTER_WORKER_NAME]

    def test_the_providers_it_asked_for_become_its_constructor_arguments(self, addressing_run):
        """The whole point of the injection: the worker is handed the capability, it does not go looking."""
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        kwargs = compute_ctor_kwargs(
            pool_id=ADDRESSING_POOL_ID, worker_argv=addressing_run, context=ctor_context(capability)
        )

        assert kwargs["engine_provider"] is capability.cells_provider
        assert kwargs["router_provider"] is capability.static_provider

    def test_a_spec_that_addresses_nobody_asks_the_factory_for_nothing(self, run_argv):
        """Building a provider costs a watch of the namespace, which a trainer rank has no use for."""
        capability = FakeBackendCapability()

        compute_ctor_kwargs(pool_id=TRAINER_POOL_ID, worker_argv=run_argv, context=ctor_context(capability))

        assert (capability.requested_pool_ids, capability.requested_worker_names) == ([], [])
