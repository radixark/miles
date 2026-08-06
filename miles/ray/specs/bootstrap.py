from typing import Any

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.arguments import parse_args_from_argv
from miles.utils.workers.worker_spec import ServeWorkerSpec, WorkerCtorContext

CTOR_KWARGS_FN = "miles.ray.specs.bootstrap.compute_ctor_kwargs"


def compute_ctor_kwargs(*, pool_id: str, worker_argv: list[str], context: WorkerCtorContext) -> dict[str, Any]:
    return serve_spec_of(pool_id=pool_id, worker_argv=worker_argv).ctor_kwargs(context)


def serve_spec_of(*, pool_id: str, worker_argv: list[str]) -> ServeWorkerSpec:
    specs = compute_specs(parse_args_from_argv(worker_argv))
    matched = [spec for spec in specs if spec.name == pool_id]
    assert len(matched) == 1, (
        f"the run described by this pod's argv has {[spec.name for spec in specs]}, not one spec named "
        f"'{pool_id}'; the pod and the launcher disagree about what this run is"
    )

    spec = matched[0]
    assert isinstance(spec, ServeWorkerSpec), f"spec '{pool_id}' is a {type(spec).__name__}, which is not served"
    return spec
