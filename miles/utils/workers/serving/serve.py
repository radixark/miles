from __future__ import annotations

import os
import sys

from miles.utils.workers.pod_context import read_pod_rank
from miles.utils.workers.serving.utils import parse_own_args, serve_spec_of, split_worker_argv
from miles.utils.workers.worker_spec import WorkerLaunchContext

SERVE_INNER_MODULE = "miles.utils.workers.serving.serve_inner"


def main() -> None:
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    args = parse_own_args(own_argv)
    _log(f"start own_argv={own_argv} worker_argv={worker_argv}")

    spec = serve_spec_of(specs_fn=args.specs, pool_id=args.pool_id, worker_argv=worker_argv)
    rank = read_pod_rank(scheduling=spec.scheduling)
    env_vars = spec.env_var(
        WorkerLaunchContext(
            cell_index=rank.cell_index, worker_in_cell_index=rank.worker_in_cell_index, gpu_ids=rank.gpu_ids
        )
    )
    _log(f"pool_id={args.pool_id} env_vars={env_vars}")

    inner_argv = [sys.executable, "-m", SERVE_INNER_MODULE, *own_argv, "--", *worker_argv]
    _log(f"exec {inner_argv}")
    os.execve(sys.executable, inner_argv, dict(os.environ) | env_vars)


def _log(message: str) -> None:
    print(f"[serve] {message}", flush=True)


if __name__ == "__main__":
    main()
