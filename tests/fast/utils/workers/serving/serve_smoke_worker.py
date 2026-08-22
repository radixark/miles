import os

from tests.fast.utils.workers.import_probe import report_imported_top_level_modules

from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

IMPORTED_MODULES_ENV_VAR = "MILES_SERVE_SMOKE_IMPORTED_MODULES"
SMOKE_EXTRA_ENV_VAR = "MILES_SERVE_SMOKE_EXTRA_ENV_NAME"
POOL_ID = "e2e-pool"
RPC_PORT_FLAG = "--rpc-port"


class SmokeWorker:
    def __init__(self, argv: list[str]):
        self._argv = argv

    def demo_sync(self, a: int, b: int) -> int:
        return a + b

    def report_argv(self) -> list[str]:
        return self._argv

    def report_env(self, name: str) -> str | None:
        return os.environ.get(name)


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    return [
        ServeWorkerSpec(
            name=POOL_ID,
            port_infos=[PortInfo(name="rpc", static_port=rpc_port_of(worker_argv))],
            env_var=lambda context: {
                "MILES_SERVE_SMOKE_ENV": ",".join(worker_argv),
                "MILES_SERVE_SMOKE_POOL_ID": POOL_ID,
                IMPORTED_MODULES_ENV_VAR: report_imported_top_level_modules(),
                **({name: "0"} if (name := os.environ.get(SMOKE_EXTRA_ENV_VAR)) else {}),
            },
            scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
            worker_class=f"{__name__}.SmokeWorker",
            ctor_kwargs=lambda context: dict(argv=worker_argv),
        )
    ]


def rpc_port_of(worker_argv: list[str]) -> int:
    assert RPC_PORT_FLAG in worker_argv, f"the smoke run's argv must carry {RPC_PORT_FLAG}, got {worker_argv}"
    return int(worker_argv[worker_argv.index(RPC_PORT_FLAG) + 1])
