# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SWE-bench OpenEnv environment.

Exposes SweDockerEnvironment over HTTP/WebSocket (compatible with EnvClient):
    - POST /reset, POST /step, GET /state, GET /schema, WS /ws

Usage:
    python -m swe_env.server.app --port 8004

Environment Variables:
    SWE_TASKS_DIR: Directory of harbor-format SWE-bench task dirs (required)
    SWE_DEFAULT_TASK_ID: Task id used when reset() gets none
    MAX_CONCURRENT_ENVS: Max concurrent WebSocket sessions (default: 8)
"""

import os


try:
    from openenv.core.env_server.http_server import create_app

    from swe_env.models import SweAction, SweObservation

    from .swe_env_environment import SweDockerEnvironment
except Exception:
    from models import SweAction, SweObservation

    from openenv.core.env_server.http_server import create_app
    from server.swe_env_environment import SweDockerEnvironment


max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    SweDockerEnvironment,
    SweAction,
    SweObservation,
    env_name="swe_env (Docker mode)",
    max_concurrent_envs=max_concurrent,
)


def main(host: str = "0.0.0.0", port: int = 8004):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
