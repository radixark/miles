from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_args_with_sglang_config, make_sglang_config_yaml

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_spec import BaseWorkerSpec, WorkerCtorContext


def _unusable_capability():
    class _Unusable:
        def static_worker_provider(self, *, pool_id):
            raise AssertionError(f"a train-only run asked for a worker provider for {pool_id!r}")

    return _Unusable()


class TestComputeSpecs:
    def test_launches_the_controller_then_routers_then_the_session_server_then_every_engine(self, tmp_path):
        """The manager's whole inventory comes from here, so every component must be listed exactly once."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2},
                    {"worker_type": "placeholder", "num_gpus": 4, "num_gpus_per_engine": 4},
                    {"worker_type": "decode", "num_gpus": 8, "num_gpus_per_engine": 4},
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16, use_session_server=False)

        specs = compute_specs(args)

        assert [spec.name for spec in specs] == [
            "rollout-executor",
            "multi-lora-controller",
            "inference-controller",
            "inference-router-0",
            "session-server",
            "inference-engine-0-0",
            "inference-engine-0-2",
            "trainer-controller-actor",
            "trainer-engine-actor",
        ]

    def test_a_train_only_debug_run_specs_no_inference_at_all(self):
        """--debug-train-only leaves --rollout-num-gpus unset, so anything that sizes an engine
        fleet from it cannot even be built, let alone launched."""
        args = make_args(rollout_num_gpus=None, debug_train_only=True, use_session_server=True)

        specs = {spec.name: spec for spec in compute_specs(args)}

        assert not [name for name in specs if name.startswith(("inference-router", "inference-engine"))]
        assert specs["session-server"].scheduling.num_cells == 0
        # the controller resolves the same config again inside its own actor, where a failure
        # is a dead actor rather than a readable error at launch
        assert compute_router_providers(args, capability=_unusable_capability()) == []

    def test_a_disabled_session_server_is_listed_with_no_cells(self, tmp_path):
        """Disabling the session server must not remove it from the inventory, only empty it."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4, use_session_server=False)

        specs = {spec.name: spec for spec in compute_specs(args)}

        assert specs["session-server"].scheduling.num_cells == 0
        assert specs["inference-engine-0-0"].scheduling.num_cells == 2

    def test_debug_train_only_lists_no_inference_engine(self, tmp_path):
        """--debug-train-only must instantiate no sglang engine, since its bundles are the trainer's own gpus."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            use_session_server=False,
            colocate=True,
            debug_train_only=True,
        )

        specs = compute_specs(args)

        assert [spec.name for spec in specs if spec.name.startswith("inference-engine")] == []

    def test_debug_train_only_lists_no_router(self, tmp_path):
        """A router has no engine to route to here, so paying a worker for it is pure waste."""
        specs = _debug_train_only_specs(tmp_path)

        assert [spec.name for spec in specs if spec.name.startswith("inference-router")] == []

    def test_debug_train_only_still_lists_the_trainer_and_an_empty_session_server(self, tmp_path):
        """Dropping the router must not drop the training side, and the session server survives with no cells."""
        specs = {spec.name: spec for spec in _debug_train_only_specs(tmp_path)}

        assert list(specs) == [
            "rollout-executor",
            "multi-lora-controller",
            "inference-controller",
            "session-server",
            "trainer-controller-actor",
            "trainer-engine-actor",
        ]
        assert specs["session-server"].scheduling.num_cells == 0


def _debug_train_only_specs(tmp_path) -> list[BaseWorkerSpec]:
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}])
    )
    args = make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=8,
        use_session_server="v1",
        session_server_workers=2,
        colocate=True,
        debug_train_only=True,
    )
    return compute_specs(args)


class TestDeployComponentFiltering:
    @staticmethod
    def _args(tmp_path, **overrides):
        return make_args_with_sglang_config(
            tmp_path,
            server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}],
            **{
                "rollout_num_gpus": 4,
                "use_session_server": True,
                "use_critic": True,
                "critic_num_nodes": 1,
                "critic_num_gpus_per_node": 2,
                **overrides,
            },
        )

    def test_a_trainer_deployment_holds_the_trainer_controllers_and_their_ranks_only(self, tmp_path):
        """A trainer release that also installed engines would double the run's gpu bill."""
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer"))

        assert [spec.name for spec in specs] == [
            "trainer-controller-actor",
            "trainer-controller-critic",
            "trainer-engine-actor",
            "trainer-engine-critic",
        ]

    def test_the_primary_deployment_holds_everything_the_other_two_do_not(self, tmp_path):
        """primary is defined by subtraction, so anything unclaimed has to land here rather than nowhere."""
        names = [spec.name for spec in compute_specs(self._args(tmp_path, deploy_component="primary"))]

        assert not [name for name in names if name.startswith("trainer-")]
        assert not [name for name in names if name.startswith("inference-engine")]
        assert "inference-controller" in names
        assert "inference-router-0" in names

    def test_an_inference_deployment_holds_the_engines_only(self, tmp_path):
        """The controller and the routers drive the run, and a second copy of them would serve nobody."""
        names = [spec.name for spec in compute_specs(self._args(tmp_path, deploy_component="inference"))]

        assert names
        assert all(name.startswith("inference-engine") for name in names)

    def test_every_worker_of_a_trainer_deployment_can_build_its_constructor_arguments(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """A spec that asks this release for a pool it does not install aborts the pod before it ever serves."""
        monkeypatch.setenv(RELEASE_ENV_VAR, "miles-run-260813-trainer")
        monkeypatch.setenv(NAMESPACE_ENV_VAR, "rl")
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer"))
        capability = compute_helm_backend_capability(specs=specs)
        context = WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs_by_name = {
            spec.name: spec.ctor_kwargs(context) for spec in specs if spec.name.startswith("trainer-controller-")
        }

        assert sorted(kwargs_by_name) == ["trainer-controller-actor", "trainer-controller-critic"]
        assert all("inference_controller" not in kwargs for kwargs in kwargs_by_name.values())

    def test_the_three_subsets_partition_the_whole_run(self, tmp_path):
        """A worker in no subset would never be deployed, and one in two would be deployed twice."""
        whole = [spec.name for spec in compute_specs(self._args(tmp_path, deploy_component="all"))]
        parts = [
            spec.name
            for component in ("primary", "trainer", "inference")
            for spec in compute_specs(self._args(tmp_path, deploy_component=component))
        ]

        assert sorted(whole) == sorted(parts)
