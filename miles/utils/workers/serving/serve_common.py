from __future__ import annotations

import argparse

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--worker", required=True, help="Worker factory as 'package.module.callable'")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--spec-name", default=None, help="Which worker spec this process serves")
    parser.add_argument(
        "--ctor-kwargs-fn",
        default=None,
        help="Constructor keyword computation function as 'package.module.callable'",
    )
    parser.add_argument("--ranks-per-pod", type=int, default=1, help="How many ranks share this pod")
    parser.add_argument("--gpu-slots-per-rank", type=int, default=0, help="How many gpus each of those ranks holds")
    return parser


def split_worker_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []

    separator_index = argv.index("--")
    return argv[:separator_index], argv[separator_index + 1 :]
