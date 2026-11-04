from miles.backends.sglang_utils.sglang_config import SglangScalingConfig
from miles.ray.specs.entrypoint import SERVE_SPEC_CLASSES
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.workers.connection_config import (
    WORKER_METADATA_ANNOTATION,
    StaticConnConfig,
    build_worker_annotations,
)
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR, WORKER_METADATA_ENV_VAR
from miles.utils.workers.serving import serve
from miles.utils.workers.serving.worker_config import ServeWorkerConfig
from miles.utils.workers.worker_spec import BaseServeSpec

REGISTERED_SERVE_MODULE = "tests.fast.utils.workers.serving.registered_serve"
REGISTERED_SERVE_INNER_MODULE = "tests.fast.utils.workers.serving.registered_serve_inner"


def register_test_serve_specs() -> None:
    from tests.fast.utils.workers.conformance import ConformanceServeSpec
    from tests.fast.utils.workers.e2e.e2e_worker import E2eServeSpec, FailingEnvE2eServeSpec
    from tests.fast.utils.workers.serving.serve_smoke_worker import SmokeServeSpec

    for spec_class in (ConformanceServeSpec, E2eServeSpec, FailingEnvE2eServeSpec, SmokeServeSpec):
        SERVE_SPEC_CLASSES[spec_class.worker_type] = spec_class


def serve_config_argv(*, spec_class: type[BaseServeSpec], config: BaseLeafConfig) -> list[str]:
    worker_config = ServeWorkerConfig(
        worker_type=spec_class.worker_type,
        args=config.model_dump(mode="json"),
        static_connections=StaticConnConfig(static_conn_infos={}),
    )
    return ["--config", worker_config.model_dump_json()]


def pod_env(spec: BaseServeSpec, *, cell_index: int = 0) -> dict[str, str]:
    scaling = ScalingConfig(sglang_scaling=SglangScalingConfig(groups={}))
    annotations = build_worker_annotations(spec=spec, scaling=scaling)
    return {WORKER_METADATA_ENV_VAR: annotations[WORKER_METADATA_ANNOTATION], CELL_INDEX_ENV_VAR: str(cell_index)}


def main() -> None:
    register_test_serve_specs()
    serve.SERVE_INNER_MODULE = REGISTERED_SERVE_INNER_MODULE
    serve.main()


if __name__ == "__main__":
    main()
