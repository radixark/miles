from __future__ import annotations

import sys

import uvicorn

from miles.utils.function_registry import load_function
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.serving.serve_common import build_base_parser, split_worker_argv


def main() -> None:
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    args = build_base_parser("Serve a worker over rpc").parse_args(own_argv)

    factory = load_function(args.worker)
    worker = factory(worker_argv)

    app = create_rpc_app(worker)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
