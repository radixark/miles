from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from miles.utils.args.schema import validate_complete_config
from miles.utils.workers.serving.runtime_config import RuntimeConfig

IPV4_WILDCARD_HOST = "0.0.0.0"
IPV6_WILDCARD_HOST = "::"


def create_server_socket(*, port: int) -> socket.socket:
    if socket.has_dualstack_ipv6():
        return socket.create_server(
            (IPV6_WILDCARD_HOST, port),
            family=socket.AF_INET6,
            dualstack_ipv6=True,
        )
    return socket.create_server((IPV4_WILDCARD_HOST, port), family=socket.AF_INET)


@contextmanager
def override_argv(argv: list[str]) -> Iterator[None]:
    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        yield
    finally:
        sys.argv = original_argv


@contextmanager
def override_env(env: dict[str, str]) -> Iterator[None]:
    original = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def split_worker_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []

    separator_index = argv.index("--")
    return argv[:separator_index], argv[separator_index + 1 :]


def parse_own_args(own_argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one pool of a miles run")
    parser.add_argument("--config", required=True)
    return parser.parse_args(own_argv)


def parse_runtime_config(value: str) -> RuntimeConfig:
    return validate_complete_config(config_class=RuntimeConfig, payload=json.loads(value))
