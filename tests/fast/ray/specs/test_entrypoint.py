from __future__ import annotations

from collections import Counter

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config
from tests.fast.ray.rollout.conftest import make_sglang_config_yaml

from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.inference import INFERENCE_REGISTRATION_REPORTER_POOL_ID, compute_router_providers
from miles.utils.args.runtime import AllConfig
from miles.utils.workers.connection_config import build_static_conn_config
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_spec import BaseSpec, WorkerCtorContext

_SPLIT_DEPLOYMENT_ARGV = (
    "--run-uuid",
    "0123456789abcdef",
    "--worker-comm-backend",
    "rpc",
    "--object-store-backend",
    "mooncake",
)


def _split_deployment_argv(deploy_component: str, *, trainer_ids: list[str]) -> list[str]:
    match DeployComponent(deploy_component):
        case DeployComponent.ALL:
            return []
        case DeployComponent.PRIMARY:
            addrs = [f"{trainer_id}=10.0.0.1:8000" for trainer_id in trainer_ids]
            return [*_SPLIT_DEPLOYMENT_ARGV, "--trainer-controller-addrs", *addrs]
        case DeployComponent.TRAINER:
            return [*_SPLIT_DEPLOYMENT_ARGV, *_REMOTE_OBJECT_STORE_ARGV]
        case DeployComponent.INFERENCE:
            return [
                *_SPLIT_DEPLOYMENT_ARGV,
                *_REMOTE_OBJECT_STORE_ARGV,
                "--deploy-instance-id",
                "inference",
                "--inference-controller-addr",
                "controller:8000",
            ]


_REMOTE_OBJECT_STORE_ARGV = ("--mooncake-store-init-kwargs", '{"master_server_address": "10.0.0.1:50051"}')


def _write_sglang_config(tmp_path, server_groups: list[dict]) -> str:
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(make_sglang_config_yaml(server_groups=server_groups))
    return str(config_path)


def _unusable_capability():
    class _Unusable:
        def static_worker_provider(self, *, pool_id):
            raise AssertionError(f"a train-only run asked for a worker provider for {pool_id!r}")

    return _Unusable()


class TestComputeSpecs:
    def test_launches_the_controller_then_routers_then_the_session_server_then_every_engine(self, tmp_path):
        """The manager's whole inventory comes from here, so every component must be listed exactly once."""
        sglang_config = _write_sglang_config(
            tmp_path,
            [
                {"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2},
                {"worker_type": "placeholder", "num_gpus": 4, "num_gpus_per_engine": 4},
                {"worker_type": "decode", "num_gpus": 8, "num_gpus_per_engine": 4},
            ],
        )
        args = parse_megatron_test_config("--sglang-config", sglang_config, "--rollout-num-gpus", "16")

        specs = compute_specs(args)

        assert [spec.name for spec in specs] == [
            "rollout-executor",
            "inference-controller",
            "inference-router-0",
            "session-server",
            "inference-engine-all-0-0",
            "inference-engine-all-0-2",
            "trainer-controller-actor",
            "trainer-engine-actor",
        ]

    def test_a_train_only_debug_run_specs_no_inference_at_all(self):
        """--debug-train-only leaves --rollout-num-gpus unset, so anything that sizes an engine
        fleet from it cannot even be built, let alone launched."""
        args = parse_megatron_test_config("--debug-train-only", "--use-session-server")

        specs = {spec.name: spec for spec in compute_specs(args)}

        assert not [name for name in specs if name.startswith(("inference-router", "inference-engine"))]
        assert specs["session-server"].scheduling(args).num_cells == 0
        # the controller resolves the same config again inside its own actor, where a failure
        # is a dead actor rather than a readable error at launch
        assert compute_router_providers(args, capability=_unusable_capability()) == []

    def test_a_disabled_session_server_is_listed_with_no_cells(self, tmp_path):
        """Disabling the session server must not remove it from the inventory, only empty it."""
        sglang_config = _write_sglang_config(
            tmp_path, [{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
        )
        args = parse_megatron_test_config("--sglang-config", sglang_config, "--rollout-num-gpus", "4")

        specs = {spec.name: spec for spec in compute_specs(args)}

        assert specs["session-server"].scheduling(args).num_cells == 0
        assert specs["inference-engine-all-0-0"].scheduling(args).num_cells == 2

    def test_debug_train_only_lists_no_inference_engine(self, tmp_path):
        """--debug-train-only must instantiate no sglang engine, since its bundles are the trainer's own gpus."""
        sglang_config = _write_sglang_config(
            tmp_path, [{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
        )
        args = parse_megatron_test_config(
            "--sglang-config", sglang_config, "--rollout-num-gpus", "8", "--colocate", "--debug-train-only"
        )

        specs = compute_specs(args)

        assert [spec.name for spec in specs if spec.name.startswith("inference-engine")] == []

    def test_debug_train_only_lists_no_router(self, tmp_path):
        """A router has no engine to route to here, so paying a worker for it is pure waste."""
        specs = _debug_train_only_specs(tmp_path)

        assert [spec.name for spec in specs if spec.name.startswith("inference-router")] == []

    def test_debug_train_only_still_lists_the_trainer_and_an_empty_session_server(self, tmp_path):
        """Dropping the router must not drop the training side, and the session server survives with no cells."""
        args = _debug_train_only_args(tmp_path)
        specs = {spec.name: spec for spec in compute_specs(args)}

        assert list(specs) == [
            "rollout-executor",
            "inference-controller",
            "session-server",
            "trainer-controller-actor",
            "trainer-engine-actor",
        ]
        assert specs["session-server"].scheduling(args).num_cells == 0


def _debug_train_only_args(tmp_path) -> AllConfig:
    sglang_config = _write_sglang_config(
        tmp_path, [{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
    )
    return parse_megatron_test_config(
        "--sglang-config",
        sglang_config,
        "--rollout-num-gpus",
        "8",
        "--use-session-server",
        "v1",
        "--session-server-workers",
        "2",
        "--colocate",
        "--debug-train-only",
    )


def _debug_train_only_specs(tmp_path) -> list[BaseSpec]:
    return compute_specs(_debug_train_only_args(tmp_path))


class TestDeployComponentFiltering:
    @staticmethod
    def _args(tmp_path, *argv: str, deploy_component: str = "all", use_critic: bool = True) -> AllConfig:
        sglang_config = _write_sglang_config(
            tmp_path, [{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
        )
        critic_argv = ["--advantage-estimator", "ppo", "--critic-num-nodes", "1", "--critic-num-gpus-per-node", "2"]
        split_argv = _split_deployment_argv(
            deploy_component, trainer_ids=["actor", "critic"] if use_critic else ["actor"]
        )
        return parse_megatron_test_config(
            "--sglang-config",
            sglang_config,
            "--rollout-num-gpus",
            "4",
            "--use-session-server",
            "--deploy-component",
            deploy_component,
            *(critic_argv if use_critic else []),
            *split_argv,
            *argv,
        )

    def test_a_trainer_deployment_holds_the_trainer_controllers_and_their_ranks_only(self, tmp_path):
        """A trainer release that also installed engines would double the run's gpu bill."""
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer", use_critic=False))

        assert [spec.name for spec in specs] == ["trainer-controller-actor", "trainer-engine-actor"]

    def test_the_primary_deployment_holds_everything_the_other_two_do_not(self, tmp_path):
        """primary is defined by subtraction, so anything unclaimed has to land here rather than nowhere."""
        names = [spec.name for spec in compute_specs(self._args(tmp_path, deploy_component="primary"))]

        assert not [name for name in names if name.startswith("trainer-")]
        assert not [name for name in names if name.startswith("inference-engine")]
        assert "inference-controller" in names
        assert "inference-router-0" in names

    def test_every_worker_of_a_trainer_deployment_can_build_its_constructor_arguments(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """A spec that asks this release for a pool it does not install aborts the pod before it ever serves."""
        monkeypatch.setenv(RELEASE_ENV_VAR, "miles-run-260813-trainer")
        monkeypatch.setenv(NAMESPACE_ENV_VAR, "rl")
        args = self._args(tmp_path, deploy_component="trainer", use_critic=False)
        specs = compute_specs(args)
        capability = compute_helm_backend_capability(config=build_static_conn_config(specs=specs, scaling=args))
        kwargs_by_name = {
            spec.name: spec.ctor_kwargs(
                WorkerCtorContext(
                    args=spec.args,
                    cell_index=0,
                    worker_in_cell_index=0,
                    num_workers_per_cell=1,
                    gpu_ids=[],
                    capability=capability,
                )
            )
            for spec in specs
            if spec.name.startswith("trainer-controller-")
        }

        assert sorted(kwargs_by_name) == ["trainer-controller-actor"]
        assert all("inference_controller" not in kwargs for kwargs in kwargs_by_name.values())

    def test_a_subset_carries_exactly_the_workers_of_its_own_component(self, tmp_path):
        """A worker two subsets carry is installed twice, and one no subset carries is never installed at all."""
        whole = Counter(
            spec.deploy_component
            for spec in compute_specs(self._args(tmp_path, deploy_component="all", use_critic=False))
        )

        for component in (DeployComponent.PRIMARY, DeployComponent.TRAINER, DeployComponent.INFERENCE):
            carried = [
                spec
                for spec in compute_specs(self._args(tmp_path, deploy_component=component.value, use_critic=False))
                if spec.name != INFERENCE_REGISTRATION_REPORTER_POOL_ID
            ]

            assert {spec.deploy_component for spec in carried} == {component}
            assert len(carried) == whole[component]

    def test_a_named_trainer_deployment_holds_the_one_trainer_its_arguments_describe(self, tmp_path):
        """Its arguments give one model's configuration, so the instance names the release and selects nothing."""
        args = self._args(
            tmp_path,
            "--deploy-instance-id",
            "a-actor",
            "--megatron-config",
            encode_megatron_config("a"),
            deploy_component="trainer",
            use_critic=False,
        )

        specs = compute_specs(args)

        assert [spec.name for spec in specs] == ["trainer-controller-a-actor", "trainer-engine-a-actor"]

    def test_an_inference_deployment_holds_the_engines_and_the_one_reporter(self, tmp_path):
        """An engine release carries no controller and no router; it only announces the engines it launches."""
        specs = compute_specs(self._args(tmp_path, deploy_component="inference"))

        assert [spec.name for spec in specs] == ["inference-registration-reporter", "inference-engine-inference-0-0"]

    @pytest.mark.parametrize("component", ["all", "primary", "trainer"])
    def test_only_an_inference_deployment_carries_a_reporter(self, tmp_path, component):
        """A reporter beside the controller it reports into would register a deployment into itself."""
        specs = compute_specs(self._args(tmp_path, deploy_component=component, use_critic=component != "trainer"))

        assert "inference-registration-reporter" not in [spec.name for spec in specs]
