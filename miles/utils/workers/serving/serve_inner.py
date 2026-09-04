from __future__ import annotations

import argparse
import sys

import uvicorn

from miles.utils.function_registry import load_function
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.serving.utils import split_worker_argv

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def main() -> None:
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    args = parse_own_args(own_argv)

    factory = load_function(args.worker)
    worker = factory(worker_argv)

    app = create_rpc_app(worker)
    uvicorn.run(app, host=args.host, port=args.port)


def parse_own_args(own_argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a worker over rpc")
    parser.add_argument("--worker", required=True, help="Worker factory as 'package.module.callable'")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args(own_argv)


if __name__ == "__main__":
    main()
