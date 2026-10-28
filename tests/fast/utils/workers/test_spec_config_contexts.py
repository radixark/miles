import shlex

from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import (
    LAYOUT,
    build_values_as_launched,
)
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import FakeRayCluster
from tests.fast.utils.workers.fake_specs import FakeCommandSpec, FakeServeSpec
from tests.fast.utils.workers.test_ray_worker_manager_serve import DemoServeWorker

from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.workers.ray_worker_manager import RayWorkerManager, _build_serve_worker
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_spec import SchedulingSpec


class _SpecConfig(BaseLeafConfig):
    token: str


class TestSpecConfigContexts:
    async def test_each_ray_worker_retains_its_spec_config_through_constructor_bootstrap(
        self, fake_ray_cluster: FakeRayCluster
    ) -> None:
        """Workers in different pools receive their own sliced config through both launch and ctor contexts."""
        worker_path = f"{DemoServeWorker.__module__}.{DemoServeWorker.__qualname__}"
        specs = [
            FakeServeSpec(
                name=name,
                args=_SpecConfig(token=token),
                port_infos=[],
                fixed_scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
                worker_class=worker_path,
                env_vars=lambda ctx: {"POOL_TOKEN": ctx.args.token},
                make_ctor_kwargs=lambda ctx: {"token": ctx.args.token, "config": ctx.args},
            )
            for name, token in [("first", "first-token"), ("second", "second-token")]
        ]
        manager = RayWorkerManager()

        await manager.init(worker_manager_args(), specs, {}, comm_backend=WorkerCommBackend.RAY)
        workers = [
            _build_serve_worker(worker_class_path=worker_path, **kwargs) for kwargs in fake_ray_cluster.ctor_kwargs
        ]

        assert [worker.kwargs["token"] for worker in workers] == ["first-token", "second-token"]
        assert [handle.options["runtime_env"]["env_vars"]["POOL_TOKEN"] for handle in fake_ray_cluster.handles] == [
            "first-token",
            "second-token",
        ]
        assert workers[0].kwargs["config"] is specs[0].args
        assert workers[1].kwargs["config"] is specs[1].args

    def test_helm_command_and_environment_use_the_same_per_spec_config(self) -> None:
        """Helm command and environment rendering preserve the config carried by their own pool spec."""
        spec = FakeCommandSpec(
            name="router",
            args=_SpecConfig(token="per spec"),
            port_infos=[],
            fixed_scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
            command=lambda ctx: shlex.join(["python", "-m", "router", "--token", ctx.args.token]),
            env_vars=lambda ctx: {"POOL_TOKEN": ctx.args.token},
        )

        values = build_values_as_launched([spec], LAYOUT).as_values()
        [entry] = values["run"]["staticWorkers"]

        assert entry["command"] == ["python", "-m", "router", "--token", "per spec"]
        assert entry["env"]["POOL_TOKEN"] == "per spec"
        assert spec.args.token == "per spec"


class TestTrainerControllerDiscovery:
    def test_every_sliced_trainer_gets_its_own_controller_dns_address(self) -> None:
        """The Helm launcher discovers every trainer controller created from its typed trainer slice."""
        from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
        from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

        from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import (
            _compute_trainer_controller_addrs,
        )

        args = parse_megatron_test_config("--megatron-config", encode_megatron_config("alpha", "beta"))

        addresses = _compute_trainer_controller_addrs(args, release="r", namespace="train-ns")

        assert addresses == {
            "alpha-actor": "r-miles-run-trainer-controller-alpha-actor-0.r-miles-run-trainer-controller-alpha-actor.train-ns.svc:8000",
            "beta-actor": "r-miles-run-trainer-controller-beta-actor-0.r-miles-run-trainer-controller-beta-actor.train-ns.svc:8000",
        }
