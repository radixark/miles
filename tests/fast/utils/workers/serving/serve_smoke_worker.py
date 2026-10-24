import os
from typing import Any, ClassVar, Self

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseServeSpec,
    PortInfo,
    SchedulingSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
)

SMOKE_EXTRA_ENV_VAR = "MILES_SERVE_SMOKE_EXTRA_ENV_NAME"
POOL_ID = "e2e-pool"


class SmokeWorker:
    def __init__(self, argv: list[str]):
        self._argv = argv

    def demo_sync(self, a: int, b: int) -> int:
        return a + b

    def report_argv(self) -> list[str]:
        return self._argv

    def report_env(self, name: str) -> str | None:
        return os.environ.get(name)


class SmokeWorkerConfig(BaseLeafConfig):
    rpc_port: int
    worker_argv: list[str]


class SmokeServeSpec(BaseServeSpec):
    worker_type: ClassVar[str] = "serve-smoke"
    config_class: ClassVar[type[BaseLeafConfig]] = SmokeWorkerConfig
    args: SmokeWorkerConfig
    name: str = POOL_ID
    worker_class: str = f"{__name__}.SmokeWorker"

    @classmethod
    def create(cls, config: SmokeWorkerConfig) -> Self:
        return cls(args=config, port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=config.rpc_port)])

    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        return SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0)

    def env_var(self, ctx: WorkerLaunchContext) -> dict[str, str]:
        return {
            "MILES_SERVE_SMOKE_ENV": ",".join(self.args.worker_argv),
            "MILES_SERVE_SMOKE_POOL_ID": POOL_ID,
            **({name: "0"} if (name := os.environ.get(SMOKE_EXTRA_ENV_VAR)) else {}),
        }

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return dict(argv=self.args.worker_argv)
