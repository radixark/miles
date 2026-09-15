from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from miles.utils.function_registry import load_function
from miles.utils.workers.worker_spec import ServeWorkerSpec


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
    parser.add_argument("--specs", required=True, help="Spec table of the run as 'package.module.callable'")
    parser.add_argument("--pool-id", required=True, help="Which pool of that run this process serves")
    return parser.parse_args(own_argv)


def compute_serve_worker_spec(*, specs_fn: str, pool_id: str, worker_argv: list[str]) -> ServeWorkerSpec:
    specs = load_function(specs_fn)(worker_argv)
    matched = [spec for spec in specs if spec.name == pool_id]
    assert len(matched) == 1, (
        f"the run described by this pod's argv has {[spec.name for spec in specs]}, not one spec named "
        f"'{pool_id}'; the pod and the launcher disagree about what this run is"
    )

    spec = matched[0]
    assert isinstance(spec, ServeWorkerSpec), f"spec '{pool_id}' is a {type(spec).__name__}, which is not served"
    return spec
