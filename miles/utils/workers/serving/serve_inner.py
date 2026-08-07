from __future__ import annotations

import argparse
import sys
from typing import Any

import uvicorn

from miles.ray.wiring import create_worker_backend_capability
from miles.utils.function_registry import load_function
from miles.utils.workers.pod_context import PodRank, read_pod_rank
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.serving.serve_common import build_base_parser, split_worker_argv


def main() -> None:
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    args = build_base_parser("Serve a worker over rpc").parse_args(own_argv)

    rank = read_pod_rank(ranks_per_pod=args.ranks_per_pod, gpu_slots_per_rank=args.gpu_slots_per_rank)
    worker = create_worker(args, worker_argv=worker_argv, rank=rank)

    app = create_rpc_app(worker)
    uvicorn.run(app, host=args.host, port=args.port + rank.rank_in_pod)


def create_worker(args: argparse.Namespace, *, worker_argv: list[str], rank: PodRank) -> Any:
    factory = load_function(args.worker)
    if args.ctor_kwargs_fn is None:
        return factory(worker_argv)

    assert args.pool_id, "--ctor-kwargs-fn computes the keywords of one named spec, so --pool-id is required"

    context = rank.ctor_context(
        pool_id=args.pool_id, capability=create_worker_backend_capability(worker_argv=worker_argv)
    )
    ctor_kwargs = load_function(args.ctor_kwargs_fn)(pool_id=args.pool_id, worker_argv=worker_argv, context=context)
    return factory(**ctor_kwargs)


if __name__ == "__main__":
    main()
