from __future__ import annotations

import argparse
import os
import sys

from miles.utils.function_registry import load_function
from miles.utils.workers.serving.utils import split_worker_argv


def main() -> None:
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])

    parser = argparse.ArgumentParser(description="Compute worker env vars, then exec into the rpc server")
    parser.add_argument("--env-var-fn", default=None, help="Env var computation function as 'package.module.callable'")
    args, inner_own_argv = parser.parse_known_args(own_argv)
    _log(f"start own_argv={own_argv} worker_argv={worker_argv}")

    env = dict(os.environ)
    if args.env_var_fn is not None:
        computed_env_vars: dict[str, str] = load_function(args.env_var_fn)(worker_argv)
        _log(f"env_var_fn={args.env_var_fn} computed={computed_env_vars}")
        env.update(computed_env_vars)

    inner_argv = [
        sys.executable,
        "-m",
        "miles.utils.workers.serving.serve_inner",
        *inner_own_argv,
        "--",
        *worker_argv,
    ]
    _log(f"exec {inner_argv}")
    os.execve(sys.executable, inner_argv, env)


def _log(message: str) -> None:
    print(f"[serve] {message}", flush=True)


if __name__ == "__main__":
    main()
