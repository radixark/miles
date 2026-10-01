from __future__ import annotations

import os
import sys

from miles.ray.specs.entrypoint import SERVE_SPEC_CLASSES
from miles.utils.workers.argv_utils import python_argv_prefix
from miles.utils.workers.env_vars import PLATFORM_IDENTITY_ENV_VARS
from miles.utils.workers.serving.utils import parse_own_args, parse_serve_worker_config
from miles.utils.workers.serving.worker_identity import read_worker_identity
from miles.utils.workers.worker_spec import WorkerLaunchContext

SERVE_INNER_MODULE = "miles.utils.workers.serving.serve_inner"


def main() -> None:
    own_args = parse_own_args(sys.argv[1:])

    worker_config = parse_serve_worker_config(own_args.config)
    spec_class = SERVE_SPEC_CLASSES[worker_config.worker_type]
    spec = spec_class.create(spec_class.config_class.model_validate(worker_config.args))
    identity = read_worker_identity(os.environ)
    env_vars = spec.env_var(
        WorkerLaunchContext(
            args=spec.args,
            cell_index=identity.cell_index,
            worker_in_cell_index=identity.worker_in_cell_index,
            num_workers_per_cell=identity.num_workers_per_cell,
            gpu_ids=identity.gpu_ids,
        )
    )
    overridden = sorted(name for name in PLATFORM_IDENTITY_ENV_VARS if name in env_vars)
    assert not overridden, (
        f"spec {spec.name} sets {overridden}, which the platform owns; a worker that read the spec's value "
        f"would report the identity of another worker and bind that worker's ports"
    )
    _log(f"pool_id={spec.name} env_vars={env_vars}")

    inner_argv = [*python_argv_prefix(), "-m", SERVE_INNER_MODULE, *sys.argv[1:]]
    _log(f"exec {SERVE_INNER_MODULE} pool_id={spec.name}")
    os.execve(sys.executable, inner_argv, dict(os.environ) | env_vars)


def _log(message: str) -> None:
    print(f"[serve] {message}", flush=True)


if __name__ == "__main__":
    main()
