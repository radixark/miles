from __future__ import annotations

from argparse import Namespace

from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.specs.rollout import (
    ROLLOUT_EXECUTOR_POOL_ID,
    ROLLOUT_EXECUTOR_WORKER_CLASS,
    rollout_executor_cell_id,
    rollout_executor_worker_name,
    spec_rollout_executor,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import SECTION_OF_CATEGORY, LaunchPlan
from miles.utils.function_registry import load_function
from miles.utils.misc import NodeProbeMixin
from miles.utils.workers.ray_worker_manager import bootstrapped_worker_class
from miles.utils.workers.worker_spec import WorkerCtorContext


def _args(*, debug_train_only: bool = False, use_session_server: bool = False, **overrides) -> Namespace:
    return make_args(
        pin_rollout_manager_to_head=False,
        debug_train_only=debug_train_only,
        use_session_server=use_session_server,
        **overrides,
    )


def _layout() -> LaunchPlan:
    return LaunchPlan(
        run_id="260101-000000-000",
        state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
        release="miles-run-260101",
        namespace="rl",
        orchestrator_command=["python", "/repo/train.py"],
        worker_argv=["--rollout-num-gpus", "8"],
    )


class TestRolloutExecutorSpec:
    def test_a_run_asks_for_exactly_one_gpuless_worker(self):
        """One executor per run, and it must claim no gpu or the scheduler would reserve a whole slot."""
        spec = spec_rollout_executor(_args())

        assert spec.name == ROLLOUT_EXECUTOR_POOL_ID
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpus_per_worker == 0

    def test_the_worker_class_is_the_executor_itself(self):
        """The spec names the class a pod or actor constructs, so it must resolve to the real implementation."""
        assert load_function(spec_rollout_executor(_args()).worker_class) is RolloutExecutor

    def test_the_worker_class_answers_the_managers_node_probe(self):
        """alloc_ports() probes the node before it reads port_infos, so a worker without the probe dies at launch."""
        bootstrapped = bootstrapped_worker_class(ROLLOUT_EXECUTOR_WORKER_CLASS)

        assert issubclass(bootstrapped, RolloutExecutor)
        assert issubclass(bootstrapped, NodeProbeMixin)

    def test_the_ctor_kwargs_hand_the_worker_the_providers_it_resolves_with(self):
        """The executor resolves its own addresses in init(), so its spec names exactly what that takes."""
        capability = FakeBackendCapability(static_provider=object())
        context = WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs = spec_rollout_executor(_args(use_session_server=True)).ctor_kwargs(context)

        assert sorted(kwargs) == ["args", "router_providers", "session_server_provider"]
        assert kwargs["router_providers"] == [capability.static_provider]
        assert kwargs["session_server_provider"] is capability.static_provider
        assert capability.requested_static_pool_ids == ["inference-router-0", "session-server"]

    def test_a_run_without_session_servers_is_given_no_session_provider(self):
        """Nothing is deployed to wait for, and a provider would make the executor wait for it anyway."""
        capability = FakeBackendCapability(static_provider=object())
        context = WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs = spec_rollout_executor(_args(use_session_server=False)).ctor_kwargs(context)

        assert kwargs["session_server_provider"] is None

    def test_the_worker_and_cell_names_are_stable(self):
        """The driver looks the executor up by name, so these names are part of the release's contract."""
        assert rollout_executor_worker_name() == "rollout-executor-0-0"
        assert rollout_executor_cell_id() == "rollout-executor-0"

    def test_it_renders_into_static_workers_with_its_rpc_port(self):
        """The release has to contain the executor pod, or the address book would point at nothing."""
        spec = spec_rollout_executor(_args())

        values = build_values([spec], _layout()).as_values()

        (entry,) = values["run"]["staticWorkers"]
        assert SECTION_OF_CATEGORY[spec.category] == "staticWorkers"
        assert entry["name"] == ROLLOUT_EXECUTOR_POOL_ID
        assert entry["ports"] == [{"name": "rpc", "port": 8000}]
        assert entry["command"][entry["command"].index("--pool-id") + 1] == ROLLOUT_EXECUTOR_POOL_ID
        assert spec.worker_class == ROLLOUT_EXECUTOR_WORKER_CLASS
        assert "resources" not in entry
