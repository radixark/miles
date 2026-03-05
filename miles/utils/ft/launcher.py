"""Training job launcher wrapper.

Sets environment variables from ``--runtime-env-json`` and then execs the
trailing command.  Designed to sit between ``ray job submit`` and the real
training script so that future concerns (FT controller, health checks, …)
can be added here without touching command_utils.py.

Usage::

    python -m miles.utils.ft.launcher \
        --runtime-env-json '{"env_vars": {"K": "V"}}' \
        -- python3 train.py --lr 0.001
"""

from __future__ import annotations

import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _apply_env_vars(runtime_env_json: str) -> None:
    parsed = json.loads(runtime_env_json)
    env_vars: dict[str, str] = parsed.get("env_vars", {})
    for key, value in env_vars.items():
        os.environ[key] = value
    if env_vars:
        logger.info("launcher set %d env vars: %s", len(env_vars), list(env_vars.keys()))


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    runtime_env_json: str | None = None
    command: list[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--runtime-env-json" and i + 1 < len(args):
            runtime_env_json = args[i + 1]
            i += 2
        elif args[i].startswith("--runtime-env-json="):
            runtime_env_json = args[i].split("=", 1)[1]
            i += 1
        elif args[i] == "--":
            command = args[i + 1 :]
            break
        else:
            raise SystemExit(f"launcher: unexpected argument: {args[i]}")

    if runtime_env_json is not None:
        _apply_env_vars(runtime_env_json)

    if not command:
        raise SystemExit("launcher: no command given after '--'")

    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
